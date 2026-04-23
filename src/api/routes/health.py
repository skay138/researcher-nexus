"""
Health & Schema 엔드포인트
GET /api/v1/health  — 컴포넌트별 상태 확인
GET /api/v1/schema  — 현재 DB 스키마 조회
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.schemas import HealthComponentStatus, HealthResponse, SchemaResponse
from core.compiler.schema_registry import SchemaRegistry

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """
    전체 시스템 헬스체크.
    components: engine, schema_registry 상태 반환.
    하나라도 error이면 HTTP 503.
    """
    components: dict[str, HealthComponentStatus] = {}

    # Engine 초기화 여부
    engine = getattr(request.app.state, "engine", None)
    if engine is not None:
        components["engine"] = HealthComponentStatus(status="ok")
    else:
        components["engine"] = HealthComponentStatus(
            status="error", detail="ExecutionEngine not initialized"
        )

    # SchemaRegistry 응답 확인
    schema_registry: SchemaRegistry = getattr(request.app.state, "schema_registry", None)
    if schema_registry is not None:
        try:
            schema_registry.get_schema_for_llm()
            components["schema_registry"] = HealthComponentStatus(status="ok")
        except Exception as e:
            components["schema_registry"] = HealthComponentStatus(
                status="degraded", detail=str(e)
            )
    else:
        components["schema_registry"] = HealthComponentStatus(
            status="error", detail="SchemaRegistry not initialized"
        )

    overall = (
        "ok"
        if all(c.status == "ok" for c in components.values())
        else "degraded"
    )
    settings = getattr(request.app.state, "settings", None)
    env = settings.environment if settings else "unknown"

    response = HealthResponse(
        status=overall,
        environment=env,
        components=components,
    )

    status_code = 200 if overall == "ok" else 503
    return JSONResponse(content=response.model_dump(), status_code=status_code)


@router.get("/schema", response_model=SchemaResponse)
async def get_schema(request: Request) -> SchemaResponse:
    """현재 DB 스키마를 JSON으로 반환."""
    schema_registry: SchemaRegistry = request.app.state.schema_registry
    schema_text = schema_registry.get_schema_for_llm()

    node_types = list(schema_registry.CONCEPT_MAPPING.keys())
    mock_schema = schema_registry._mock_schema()

    # 노드 타입과 관계 목록 파싱
    node_type_list = [
        "Project", "Researcher", "Organization",
        "Paper", "Patent", "Report",
    ]
    relation_list = list(set(schema_registry.CONCEPT_MAPPING.values()))

    return SchemaResponse(
        schema_text=schema_text,
        node_types=node_type_list,
        relations=sorted(relation_list),
        concept_mapping=schema_registry.CONCEPT_MAPPING,
    )
