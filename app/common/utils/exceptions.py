"""
Application Exception Hierarchy
- 모든 도메인 예외의 공통 기반 제공
- 계층 구조로 catch 범위 조절 가능
- HTTP 상태 코드 힌트 포함 (API 레이어에서 활용)
"""

from __future__ import annotations
from typing import Optional


class LangGraphBaseError(Exception):
    """애플리케이션 최상위 예외. HTTP 500."""
    http_status: int = 500

    def __init__(self, message: str, *, detail: Optional[str] = None):
        super().__init__(message)
        self.detail = detail or message


# ── Configuration ─────────────────────────────────────────────────────────────

class ConfigurationError(LangGraphBaseError):
    """설정 값이 유효하지 않거나 필수 값이 누락된 경우. HTTP 500."""
    http_status = 500


# ── Schema ────────────────────────────────────────────────────────────────────

class SchemaError(LangGraphBaseError):
    """스키마 레지스트리 오류 (알 수 없는 관계 개념 등). HTTP 422."""
    http_status = 422


class UnknownRelationConcept(SchemaError):
    """SchemaRegistry.resolve_relation() 실패 시."""
    pass


# ── Query Validation ──────────────────────────────────────────────────────────

class QueryValidationError(LangGraphBaseError):
    """QueryPlan 또는 입력값 검증 실패. HTTP 400."""
    http_status = 400


class InvalidNodeType(QueryValidationError):
    """스키마에 없는 노드 타입. HTTP 400."""
    pass


class InvalidFilterKey(QueryValidationError):
    """허용되지 않는 필터 키. HTTP 400."""
    pass


class CypherInjectionDetected(QueryValidationError):
    """Cypher 인젝션 가능성이 있는 입력값. HTTP 400."""
    pass


# ── Execution ─────────────────────────────────────────────────────────────────

class ExecutionError(LangGraphBaseError):
    """쿼리 실행 중 발생한 오류. HTTP 500."""
    http_status = 500


class DBConnectionError(ExecutionError):
    """데이터베이스 연결 실패. HTTP 503."""
    http_status = 503


class VectorSearchError(ExecutionError):
    """벡터 DB 검색 실패. HTTP 502."""
    http_status = 502


class GraphQueryError(ExecutionError):
    """그래프 DB 쿼리 실패. HTTP 502."""
    http_status = 502


class PruningError(ExecutionError):
    """BeamPruner 실행 실패. HTTP 500."""
    http_status = 500


# ── LLM ───────────────────────────────────────────────────────────────────────

class LLMError(LangGraphBaseError):
    """LLM 호출 실패. HTTP 502."""
    http_status = 502


class LLMTimeoutError(LLMError):
    """LLM 응답 타임아웃. HTTP 504."""
    http_status = 504


# ── Tool ──────────────────────────────────────────────────────────────────────

class ToolError(LangGraphBaseError):
    """시맨틱 도구 실행 실패. HTTP 500."""
    http_status = 500


class EngineNotInitializedError(ToolError):
    """ExecutionEngine이 초기화되지 않은 상태에서 도구 호출. HTTP 503."""
    http_status = 503


# ── Cache ─────────────────────────────────────────────────────────────────────

class CacheError(LangGraphBaseError):
    """캐시 작업 실패. HTTP 500."""
    http_status = 500
