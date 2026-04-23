"""
구조화 로깅 설정
- production: JSON 포맷 (ELK/CloudWatch 등 로그 집계 시스템 호환)
- development: 가독성 포맷 (색상 없음, 타임스탬프 포함)
"""

from __future__ import annotations
import json
import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.settings import Settings


class HealthCheckFilter(logging.Filter):
    """
    uvicorn.access 로거에서 /api/v1/health 요청을 필터링한다.
    Docker Compose healthcheck 핑 요청이 로그에 준적되는 것을 방지.
    """
    _PATHS = ("/api/v1/health",)

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(path in msg for path in self._PATHS)


class JsonFormatter(logging.Formatter):
    """JSON 구조화 로그 포맷터."""

    _CONTEXT_KEYS = frozenset({
        "session_id", "correlation_id", "query",
        "method", "path", "status_code", "elapsed_ms",
        "provider", "environment", "error", "detail",
    })

    def format(self, record: logging.LogRecord) -> str:
        log: dict = {
            "ts":     self.formatTime(record, self.datefmt),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }

        # extra= 로 전달된 컨텍스트 키만 포함
        for key in self._CONTEXT_KEYS:
            val = record.__dict__.get(key)
            if val is not None:
                log[key] = val

        if record.exc_info:
            log["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    """개발 환경용 가독성 포맷터."""
    FMT = "%(asctime)s [%(levelname)s] %(name)s | %(message)s"

    def __init__(self):
        super().__init__(fmt=self.FMT, datefmt="%Y-%m-%d %H:%M:%S")


def configure_logging(settings: "Settings") -> None:
    """
    설정에 따라 루트 로거와 핸들러 구성.
    이미 핸들러가 있으면 중복 등록 방지.
    """
    root_logger = logging.getLogger()

    if root_logger.handlers:
        return  # 이미 설정됨

    handler = logging.StreamHandler(sys.stdout)

    if settings.environment == "production":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(HumanFormatter())

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    # 외부 라이브러리 노이즈 억제 (root가 DEBUG여도 WARNING 이상만 출력)
    _suppress = [
        # HTTP 클라이언트
        "httpx", "httpcore", "urllib3",
        # LangChain / LangGraph
        "langchain", "langgraph", "langchain_core", "langchain_community",
        # Neo4j 드라이버
        "neo4j", "neo4j.io", "neo4j.pool", "neo4j.work", "neo4j.bolt",
        # Milvus / gRPC
        "pymilvus", "grpc", "grpc._channel",
        # Redis
        "redis", "redis.connection",
        # 기타
        "asyncio", "sentence_transformers", "filelock", "huggingface_hub",
    ]
    for name in _suppress:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Health check 핑 요청을 uvicorn access 로그에서 제외
    logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())
