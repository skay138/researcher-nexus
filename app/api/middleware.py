"""
FastAPI Middleware
- CorrelationIDMiddleware: 요청마다 UUID 주입, 응답 헤더에 반환
- 요청/응답 구조화 로깅
"""

from __future__ import annotations
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    X-Correlation-ID 헤더를 처리.
    - 요청에 헤더가 있으면 재사용, 없으면 UUID 생성
    - 응답 헤더에 동일 ID 포함
    - request.state.correlation_id 로 하위 레이어에서 접근 가능
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = (
            request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        )
        request.state.correlation_id = correlation_id

        t0 = time.monotonic()
        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            logger.error(
                "request_error",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "correlation_id": correlation_id,
                    "elapsed_ms": elapsed_ms,
                    "error": str(exc),
                },
            )
            raise

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        if request.url.path != "/api/v1/health":
            logger.info(
                "request_complete",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "correlation_id": correlation_id,
                    "elapsed_ms": elapsed_ms,
                },
            )

        response.headers["X-Correlation-ID"] = correlation_id
        return response
