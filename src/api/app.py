"""
FastAPI Application Factory
- lifespan으로 startup/shutdown 관리
- CORS, Correlation ID 미들웨어 적용
- /api/v1 하위 라우터 등록
"""

from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.middleware import CorrelationIDMiddleware
from api.routes.health import router as health_router
from api.routes.search import router as search_router
from common.exceptions import LangGraphBaseError
from common.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """앱 시작/종료 시 리소스 초기화·정리."""
    settings: Settings = app.state.settings

    # ── Startup ───────────────────────────────────────────────────────────────
    from common.logging import configure_logging
    configure_logging(settings)

    logger.info(
        "application_startup",
        extra={"environment": settings.environment, "use_mock": settings.use_mock},
    )

    from app_factory import create_engine, create_app as _create_app
    from core.compiler.schema_registry import SchemaRegistry

    # use_mock=False 시 드라이버를 먼저 생성해 SchemaRegistry와 공유
    neo4j_driver = None
    milvus_client = None
    if not settings.use_mock:
        from app_factory import _open_neo4j, _open_milvus
        try:
            neo4j_driver  = _open_neo4j(settings)
            milvus_client = _open_milvus(settings)
        except Exception as e:
            logger.error("DB 연결 실패: %s — Mock 모드로 폴백합니다", e)

    engine = create_engine(
        neo4j_driver=neo4j_driver,
        milvus_client=milvus_client,
        settings=settings,
    )
    schema_registry = SchemaRegistry(driver=neo4j_driver)  # None → mock 스키마

    app.state.schema_registry = schema_registry
    app.state.engine = engine
    app.state.neo4j_driver = neo4j_driver

    # Agent app은 LLM 연결이 필요하므로 초기화 실패 시 degraded 모드로 기동
    try:
        from app_factory import make_config_service
        cfg = make_config_service()
        app.state.config_service = cfg

        app.state.agent_app = _create_app(
            engine=engine,
            config_service=cfg,
            settings=settings,
        )
        logger.info("agent_app_initialized", extra={"llm_base_url": settings.llm_base_url})
    except Exception as e:
        app.state.agent_app = None
        logger.warning("agent_app_init_failed: %s — running in engine-only mode", e)

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    if hasattr(engine, "_cache") and hasattr(engine._cache, "clear"):
        engine._cache.clear()
    if neo4j_driver is not None:
        neo4j_driver.close()
    logger.info("application_shutdown")


def create_fastapi_app(settings: Optional[Settings] = None) -> FastAPI:
    """
    FastAPI 앱 인스턴스 생성.

    Args:
        settings: 미지정 시 get_settings() 싱글턴 사용.
    """
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="ResearchNexus API",
        description="연구 데이터베이스 검색 에이전트 API (Paper, Patent, Report, Project, Researcher)",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings

    # ── Middleware ─────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(CorrelationIDMiddleware)

    # ── Static / Root ─────────────────────────────────────────────────────────
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(str(static_dir / "index.html"))

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(search_router, prefix="/api/v1")

    # ── Global Exception Handler ───────────────────────────────────────────────
    @app.exception_handler(LangGraphBaseError)
    async def domain_error_handler(request: Request, exc: LangGraphBaseError):
        return JSONResponse(
            status_code=exc.http_status,
            content={"detail": exc.detail},
        )

    return app


# uvicorn api.app:app 호환용 모듈 레벨 인스턴스
app = create_fastapi_app()


if __name__ == "__main__":
    import uvicorn
    s = get_settings()
    uvicorn.run(
        "api.app:app",
        host=s.api_host,
        port=s.api_port,
        reload=(s.environment == "development"),
        log_level=s.log_level.lower(),
    )
