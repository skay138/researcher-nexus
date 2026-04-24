"""
Semantic Tools (High-level)
- LLM에게 노출되는 도구: execute_dynamic_search, get_node_by_ids
- 내부적으로 Vector DB 진입과 Neo4j 그래프 탐색을 명시적으로 구분하여 QueryPlan AST로 조합
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging
import json
import threading

from langchain_core.tools import tool

from common.query_config import RequestConfig
from common.exceptions import InvalidNodeType, ToolError
from common.query_plan import (
    EntrySearch, FinalFilter, HopDirection, HopSpec, QueryPlan,
)
from core.executor.execution_engine import ExecutionEngine, NodeResult

logger = logging.getLogger(__name__)

VALID_NODE_TYPES: frozenset = frozenset({
    "Project", "Researcher", "Organization", "Paper", "Patent", "Report",
})

_engine: Optional[ExecutionEngine] = None
_engine_lock = threading.Lock()


def set_engine(engine: ExecutionEngine) -> None:
    global _engine
    with _engine_lock:
        _engine = engine


def _get_engine() -> ExecutionEngine:
    from common.exceptions import EngineNotInitializedError
    with _engine_lock:
        if _engine is None:
            raise EngineNotInitializedError("ExecutionEngine not initialized.")
        return _engine


# ────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────────

def _parse_hops(neo4j_hops: List[Dict[str, str]]) -> List[HopSpec]:
    """hop dict 리스트 → HopSpec 리스트 변환. from_type/to_type 유효성 검증 포함."""
    parsed: List[HopSpec] = []
    for i, hop_dict in enumerate(neo4j_hops):
        from_type = hop_dict.get("from_type", "")
        to_type   = hop_dict.get("to_type", "")

        if from_type not in VALID_NODE_TYPES:
            raise InvalidNodeType(
                f"hop[{i}].from_type은 {sorted(VALID_NODE_TYPES)} 중 하나여야 합니다. "
                f"입력값: {from_type!r}"
            )
        if to_type not in VALID_NODE_TYPES:
            raise InvalidNodeType(
                f"hop[{i}].to_type은 {sorted(VALID_NODE_TYPES)} 중 하나여야 합니다. "
                f"입력값: {to_type!r}"
            )

        dir_val = hop_dict.get("direction", "out").lower()
        if dir_val == "in":
            direction = HopDirection.INBOUND
        elif dir_val == "both":
            direction = HopDirection.BOTH
        else:
            direction = HopDirection.OUTBOUND

        parsed.append(HopSpec(
            from_type=from_type,
            relation_concept=hop_dict.get("relation_concept", ""),
            to_type=to_type,
            direction=direction,
            filters=hop_dict.get("filters", {}),
        ))
    return parsed


_PROTECTED_KEYS = frozenset({"id", "type", "name", "text", "path", "score"})


def _format_results(results: List[NodeResult], stats: Any) -> str:
    """순수 JSON 반환. None 필드 제거로 토큰 낭비 방지."""
    if not results:
        return json.dumps({"total": 0, "path": "", "results": []}, ensure_ascii=False)

    items = []
    for r in results:
        # core 필드 보호: meta 키가 겹쳐도 덮어쓰지 않음
        safe_meta = {
            k: v for k, v in r.meta.items()
            if k not in _PROTECTED_KEYS and v is not None
        }
        item: Dict[str, Any] = {"id": r.id, "type": r.type}
        if r.name is not None:
            item["name"] = r.name
        if r.text is not None:
            item["text"] = r.text
        if r.path:
            item["path"] = r.path
        item["score"] = round(r.meta.get("score", 0.0), 3)
        item.update(safe_meta)
        items.append(item)

    return json.dumps(
        {"total": len(items), "path": stats.path_summary, "results": items},
        ensure_ascii=False,
    )


# ────────────────────────────────────────────────────────────────────────────
# Tools
# ────────────────────────────────────────────────────────────────────────────

@tool
def execute_dynamic_search(
    vector_search_concept: str,
    vector_search_node_type: str,
    neo4j_hops: List[Dict[str, str]],
    vector_search_filters: Optional[Dict[str, Any]] = None,
    final_vector_filter_concept: Optional[str] = None,
    max_results: int = 20,
) -> str:
    """
    Vector DB에서의 진입점 검색과 Neo4j의 그래프 탐색(hops)을 동적으로 결합하여 데이터를 찾습니다.

    Args:
        vector_search_concept: 시작 노드를 찾기 위한 Vector DB 의미 검색 키워드
                               (예: "해양 사업", "한국해양과학기술원")
        vector_search_node_type: 시작 노드의 타입
                                 ("Project", "Organization", "Researcher", "Paper", "Patent", "Report")
        neo4j_hops: 시작 노드에서부터 따라갈 그래프 관계(edge) 리스트.
                    단순 벡터 검색은 빈 배열 []로 설정.
                    각 요소: {"from_type": "...", "relation_concept": "...",
                              "to_type": "...", "direction": "in"|"out"|"both"}
        vector_search_filters: 시작 노드 검색 시 필터 (예: {"year": {"gte": 2020, "lt": 2025}})
        final_vector_filter_concept: (옵션) 탐색 완료 후 최종 노드에 적용할 의미 필터 키워드
                                     (예: "자율운항"). 노드 타입명은 넣지 마세요.
        max_results: 반환할 최대 결과 수 (기본 20). 실제 상한은 서버 설정을 따릅니다.
    """
    # ── 입력 검증 ─────────────────────────────────────────────────────────────
    if vector_search_node_type not in VALID_NODE_TYPES:
        raise InvalidNodeType(
            f"vector_search_node_type은 {sorted(VALID_NODE_TYPES)} 중 하나여야 합니다. "
            f"입력값: {vector_search_node_type!r}"
        )
    max_results = max(1, max_results)

    engine = _get_engine()
    cfg    = RequestConfig.current()

    # ── Hop 파싱 및 검증 ──────────────────────────────────────────────────────
    try:
        parsed_hops = _parse_hops(neo4j_hops)
    except InvalidNodeType:
        raise
    except Exception as e:
        raise ToolError(f"neo4j_hops 파싱 실패: {e}") from e

    # ── QueryPlan 조립 ────────────────────────────────────────────────────────
    # cfg.max_results: API 파라미터 > repo > 기본값 순으로 이미 결정된 값
    plan = QueryPlan(
        entry_search=EntrySearch(
            concept=vector_search_concept,
            node_type=vector_search_node_type,
            filters=vector_search_filters or {},
        ),
        traversal_hops=parsed_hops,
        final_filter=FinalFilter(
            concept=final_vector_filter_concept,
            node_type=parsed_hops[-1].to_type if parsed_hops else vector_search_node_type,
        ) if final_vector_filter_concept else None,
        max_results=cfg.max_results,
        reasoning=f"execute_dynamic_search({vector_search_concept!r})",
    )

    logger.info("[Tool] QueryPlan:\n%s", plan.describe())

    # BeamPruner 쿼리 컨텍스트: 사용자 원본 질의 > final_filter > entry concept 순 우선
    query_context = cfg.original_query or final_vector_filter_concept or vector_search_concept

    try:
        results, stats = engine.run(plan, original_query=query_context, config=cfg.to_query_config())
    except Exception as e:
        logger.error("[Tool] execute_dynamic_search 실패: %s (%s)", e, type(e).__name__)
        raise ToolError(f"검색 실행 중 오류가 발생했습니다: {type(e).__name__}") from e

    logger.info(
        "[Tool] 완료: results=%d | elapsed=%.2fs | db_calls=%d | cache_hits=%d",
        len(results), stats.total_elapsed_s, stats.db_calls, stats.cache_hits,
    )
    return _format_results(results, stats)


@tool
def get_node_by_ids(node_ids: List[str]) -> str:
    """
    이전 검색 결과에서 얻은 ID로 특정 노드들의 상세 정보를 직접 조회합니다.
    벡터/그래프 탐색 없이 Neo4j에서 즉시 반환합니다.
    후속 질문("이 논문의 저자를 모두 보여줘", "앞 결과 중 ID xxx의 상세 정보")에 사용하세요.

    Args:
        node_ids: 조회할 노드 ID 목록 (이전 검색 결과의 "id" 필드값)
    """
    if not node_ids:
        return json.dumps({"total": 0, "path": "", "results": []}, ensure_ascii=False)

    engine = _get_engine()
    cfg    = RequestConfig.current()

    try:
        results = engine.fetch_details_fn(node_ids[:cfg.max_results])
    except Exception as e:
        logger.error("[Tool] get_node_by_ids 실패: %s", e)
        raise ToolError(f"노드 조회 중 오류가 발생했습니다: {type(e).__name__}") from e

    class _DirectStats:
        path_summary = "직접 ID 조회"

    return _format_results(results, _DirectStats())


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def extract_sources_from_tool_results(tool_results: List[str]) -> List[dict]:
    """ToolMessage 목록에서 출처 정보를 추출. 순수 JSON 포맷 기준."""
    seen_ids: set = set()
    sources: List[dict] = []

    for raw in tool_results:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue

        global_path = data.get("path", "")

        for item in data.get("results", []):
            item_id = item.get("id")
            if not item_id or item_id in seen_ids:
                continue
            seen_ids.add(item_id)

            source: Dict[str, Any] = {
                "no":   len(sources) + 1,
                "id":   item_id,
                "type": item.get("type", "Unknown"),
                "name": item.get("name", ""),
                "path": item.get("path") or global_path,
                "score": item.get("score"),
            }
            if "authors" in item:
                source["authors"] = item["authors"]
            if "year" in item:
                source["year"] = item["year"]
            sources.append(source)

    return sources


SEMANTIC_TOOLS = [execute_dynamic_search, get_node_by_ids]
