"""
API Request / Response Pydantic 모델
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────────────────────

class QueryConfigSchema(BaseModel):
    """
    요청별 설정 오버라이드. 미전달 필드는 DB 기본값 사용.

    Core:
        beam_width:     hop당 유지할 최대 노드 수
        max_results:    최종 반환 결과 수
    LLM:
        model:          LLM 모델명 (예: "qwen2.5:14b", "llama3", "mistral")
        temperature:    LLM temperature (0.0 ~ 1.0)
        max_tool_calls: 에이전트 도구 호출 최대 횟수
    """
    beam_width:     Optional[int]   = Field(None, gt=0, le=500)
    max_results:    Optional[int]   = Field(None, gt=0, le=200)
    model:          Optional[str]   = Field(None, max_length=100)
    temperature:    Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tool_calls: Optional[int]   = Field(None, gt=0, le=10)


class AgentQueryRequest(BaseModel):
    """LangGraph 에이전트 자연어 질의"""
    query:      str               = Field(..., min_length=1, max_length=1000, description="자연어 질문")
    session_id: str               = Field("default", description="체크포인팅용 세션 ID")
    config:     Optional[QueryConfigSchema] = Field(None, description="요청별 설정 오버라이드")


class EngineSearchRequest(BaseModel):
    """QueryPlan 직접 실행"""
    plan:           Dict[str, Any] = Field(..., description="QueryPlan JSON 스키마")
    original_query: str            = Field("", description="BeamPruner 문맥용 원본 쿼리")
    config:         Optional[QueryConfigSchema] = Field(None, description="요청별 설정 오버라이드")


# ── Response ──────────────────────────────────────────────────────────────────

class NodeResultSchema(BaseModel):
    id: str
    type: str
    name: Optional[str] = None
    text: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class ExecutionStatsSchema(BaseModel):
    elapsed_s: float
    db_calls: int
    cache_hits: int
    hop_counts: List[int]
    pruned_total: int


class EngineSearchResponse(BaseModel):
    results: List[NodeResultSchema]
    stats: ExecutionStatsSchema


class HealthComponentStatus(BaseModel):
    status: str       # "ok" | "degraded" | "error"
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str       # "ok" | "degraded"
    environment: str
    components: Dict[str, HealthComponentStatus]


class SchemaResponse(BaseModel):
    schema_text: str
    node_types: List[str]
    relations: List[str]
    concept_mapping: Dict[str, str]


# ── SSE Event ─────────────────────────────────────────────────────────────────

class SSEToolCallEvent(BaseModel):
    type: str = "tool_call"
    tool: str
    args: List[str]   # 인자 키 목록


class SSETokenEvent(BaseModel):
    type: str = "token"
    content: str


class SSEDoneEvent(BaseModel):
    type: str = "done"
    answer: str
    session_id: str


class SSEErrorEvent(BaseModel):
    type: str = "error"
    message: str
