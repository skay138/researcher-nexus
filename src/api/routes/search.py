"""
Search 엔드포인트
POST /api/v1/agent/query  — LangGraph 에이전트 (SSE 스트리밍)
POST /api/v1/engine/search — QueryPlan 직접 실행 (동기)
"""

from __future__ import annotations
import json
import logging
import time
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from api.schemas import (
    AgentQueryRequest,
    EngineSearchRequest,
    EngineSearchResponse,
    ExecutionStatsSchema,
    NodeResultSchema,
)
from common.exceptions import LangGraphBaseError
from common.query_plan import QueryPlan
from common.query_config import QueryConfig, RequestConfig

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Search"])


# ────────────────────────────────────────────────────────────────────────────
# Agent Query (SSE Streaming)
# ────────────────────────────────────────────────────────────────────────────

@router.post("/agent/query")
async def agent_query(body: AgentQueryRequest, request: Request):
    """
    LangGraph 에이전트로 자연어 질의 처리.
    Server-Sent Events로 토큰 및 도구 호출 스트리밍.

    SSE 이벤트 타입:
    - tool_call: {"type": "tool_call", "tool": "...", "args": [...]}
    - token:     {"type": "token", "content": "..."}
    - done:      {"type": "done", "answer": "...", "session_id": "..."}
    - error:     {"type": "error", "message": "..."}
    """
    agent_app = getattr(request.app.state, "agent_app", None)
    settings = request.app.state.settings
    repo = getattr(request.app.state, "repo", None)

    if agent_app is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    correlation_id = getattr(request.state, "correlation_id", "-")

    # API 파라미터 → repo 기본값 → 내장 기본값 순으로 설정값 결정
    api_cfg = None
    if body.config:
        api_cfg = QueryConfig(
            beam_width=body.config.beam_width,
            max_results=body.config.max_results,
            sparse_weight=body.config.sparse_weight,
            dense_weight=body.config.dense_weight,
            entry_min_score=body.config.entry_min_score,
            entry_score_ratio=body.config.entry_score_ratio,
            model=body.config.model,
            temperature=body.config.temperature,
            max_tool_calls=body.config.max_tool_calls,
        )
    resolved = RequestConfig._resolve(repo, api_cfg)

    logger.info(
        "agent_query_start",
        extra={
            "query": body.query[:100],
            "session_id": body.session_id,
            "correlation_id": correlation_id,
        },
    )

    return EventSourceResponse(
        _stream_agent(agent_app, body, settings, query_config=resolved),
        media_type="text/event-stream",
    )


async def _stream_agent(
    agent_app, body: AgentQueryRequest, settings,
    query_config: Optional[QueryConfig] = None,
) -> AsyncGenerator[str, None]:
    """app.stream()을 SSE 이벤트로 변환하는 비동기 제너레이터."""
    from langchain_core.messages import HumanMessage

    # 요청별 resolved config + 원본 쿼리 등록 → 도구 실행 시 RequestConfig.current()로 접근
    RequestConfig.set_current(query_config, original_query=body.query)

    max_tool_calls = (query_config.max_tool_calls if query_config and query_config.max_tool_calls else 3)

    initial_state = {
        "messages":        [HumanMessage(content=body.query)],
        "tool_call_count": 0,
        "total_db_calls":  0,
        "session_id":      body.session_id,
        "start_time":      time.time(),
        "max_tool_calls":  max_tool_calls,
    }
    run_config = {
        "configurable": {"thread_id": body.session_id},
        "recursion_limit": settings.recursion_limit,
    }

    answer_parts: list[str] = []

    try:
        async for msg_chunk, metadata in agent_app.astream(
            initial_state, config=run_config, stream_mode="messages"
        ):
            node = metadata.get("langgraph_node")
            if node not in ["Planner", "Agent"]:
                continue

            # Planner: 도구 호출 추출
            if node == "Planner":
                for tc in (getattr(msg_chunk, "tool_calls", None) or []):
                    if tc.get("name"):
                        yield json.dumps({
                            "type": "tool_call",
                            "tool": tc["name"],
                            "args": list((tc.get("args") or {}).keys()),
                        }, ensure_ascii=False)
                continue  # Planner의 텍스트("DONE")은 버림

            # Agent: 텍스트 토큰 스트리밍
            if node == "Agent":
                content = getattr(msg_chunk, "content", "")
                if isinstance(content, str) and content:
                    answer_parts.append(content)
                    yield json.dumps({"type": "token", "content": content}, ensure_ascii=False)

        answer = "".join(answer_parts) or "(응답 없음)"
        
        # 도구 실행 결과를 바탕으로 출처(sources) 추출
        current_state = agent_app.get_state(run_config)
        all_messages = current_state.values.get("messages", [])
        
        from langchain_core.messages import ToolMessage
        from services.semantic_tools import extract_sources_from_tool_results
        
        all_tool_contents = [
            m.content for m in all_messages 
            if isinstance(m, ToolMessage) and isinstance(m.content, str)
        ]
        
        sources = extract_sources_from_tool_results(all_tool_contents)

        yield json.dumps({
            "type": "done",
            "answer": answer,
            "session_id": body.session_id,
            "sources": sources
        }, ensure_ascii=False)

    except LangGraphBaseError as e:
        logger.error("agent_query_failed: %s", e, extra={"session_id": body.session_id})
        yield json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
    except Exception as e:
        logger.exception("agent_query_unexpected_error", extra={"session_id": body.session_id})
        yield json.dumps({"type": "error", "message": f"Internal error: {type(e).__name__}"}, ensure_ascii=False)


# ────────────────────────────────────────────────────────────────────────────
# Engine Search (Sync QueryPlan)
# ────────────────────────────────────────────────────────────────────────────

@router.post("/engine/search", response_model=EngineSearchResponse)
async def engine_search(body: EngineSearchRequest, request: Request):
    """
    QueryPlan JSON을 직접 실행하여 결과를 동기 반환.
    LLM 없이 Core Engine만 사용.
    """
    engine = getattr(request.app.state, "engine", None)
    repo = getattr(request.app.state, "repo", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    api_cfg = None
    if body.config:
        api_cfg = QueryConfig(
            beam_width=body.config.beam_width,
            max_results=body.config.max_results,
        )
    resolved = RequestConfig._resolve(repo, api_cfg)

    try:
        plan = QueryPlan.model_validate(body.plan)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid QueryPlan: {e}")

    try:
        results, stats = engine.run(plan, original_query=body.original_query, config=resolved)
    except LangGraphBaseError as e:
        raise HTTPException(status_code=e.http_status, detail=e.detail)
    except Exception as e:
        logger.exception("engine_search_error")
        raise HTTPException(status_code=500, detail=f"Execution failed: {type(e).__name__}")

    return EngineSearchResponse(
        results=[
            NodeResultSchema(
                id=r.id, type=r.type, name=r.name, text=r.text, meta=r.meta or {}
            )
            for r in results
        ],
        stats=ExecutionStatsSchema(
            elapsed_s=round(stats.total_elapsed_s, 3),
            db_calls=stats.db_calls,
            cache_hits=stats.cache_hits,
            hop_counts=stats.hop_counts,
            pruned_total=stats.pruned_total,
        ),
    )
