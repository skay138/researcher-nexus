"""
Semantic Tools (High-level)
- LLM에게 노출되는 유일한 범용 도구: execute_dynamic_search
- 내부적으로 Vector DB 진입과 Neo4j 그래프 탐색을 명시적으로 구분하여 QueryPlan AST로 조합
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging
import json
import threading

from langchain_core.tools import tool

from common.query_plan import (
    EntrySearch, FinalFilter, HopDirection, HopSpec, QueryPlan,
)
from core.executor.execution_engine import ExecutionEngine, NodeResult

logger = logging.getLogger(__name__)

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
        vector_search_concept: 시작 노드를 찾기 위한 Vector DB 의미 검색 키워드 (예: "해양 사업", "한국해양과학기술원")
        vector_search_node_type: 시작 노드의 타입 ("Project", "Organization", "Researcher", "Paper", "Patent", "Report")
        neo4j_hops: 시작 노드에서부터 따라갈 그래프 관계(edge) 리스트.
                    각 요소는 {"from_type": "...", "relation_concept": "...", "to_type": "...", "direction": "in" 또는 "out"} 형식의 딕셔너리.
                    예: [{"from_type": "Project", "relation_concept": "participation", "to_type": "Researcher", "direction": "in"}]
        vector_search_filters: 시작 노드 검색 시 필터 (예: {"year": {"lt": 2025}})
        final_vector_filter_concept: (옵션) 그래프 탐색이 다 끝난 최종 노드들에 대해 다시 적용할 Vector DB 필터 키워드 (예: "보트")
        max_results: 최대 결과 개수 (기본 20)
    """
    engine = _get_engine()

    # 1. Neo4j Hops 생성
    parsed_hops = []
    for hop_dict in neo4j_hops:
        dir_val = hop_dict.get("direction", "out").lower()
        if dir_val == "in":
            direction = HopDirection.INBOUND
        elif dir_val == "both":
            direction = HopDirection.BOTH
        else:
            direction = HopDirection.OUTBOUND

        parsed_hops.append(
            HopSpec(
                from_type=hop_dict.get("from_type", ""),
                relation_concept=hop_dict.get("relation_concept", ""),
                to_type=hop_dict.get("to_type", ""),
                direction=direction,
                filters=hop_dict.get("filters", {})
            )
        )

    # 2. QueryPlan 조립
    plan = QueryPlan(
        entry_search=EntrySearch(
            concept=vector_search_concept,
            node_type=vector_search_node_type,
            filters=vector_search_filters or {},
        ),
        traversal_hops=parsed_hops,
        final_filter=FinalFilter(
            concept=final_vector_filter_concept,
            node_type=parsed_hops[-1].to_type if parsed_hops else vector_search_node_type
        ) if final_vector_filter_concept else None,
        max_results=max_results,
        reasoning=f"execute_dynamic_search({vector_search_concept!r})"
    )

    logger.info("[Tool] QueryPlan:\n%s", plan.describe())

    results, stats = engine.run(plan, original_query=final_vector_filter_concept or vector_search_concept)

    logger.info("[Tool] 최종 반환 노드 수=%d | elapsed=%.2fs | db_calls=%d | cache_hits=%d",
                len(results), stats.total_elapsed_s, stats.db_calls, stats.cache_hits)

    return _format_results(results, stats)


def _format_results(results: List[NodeResult], stats: Any) -> str:
    if not results:
        return "조건에 맞는 결과를 찾지 못했습니다."

    out = [
        f"총 {len(results)}건의 데이터를 찾았습니다.",
        f"[탐색 경로 요약] {stats.path_summary}",
        f"(상세 통계: DB 호출 {stats.db_calls}회, 캐시 적중 {stats.cache_hits}회)",
        ""
    ]
    
    out.append(json.dumps({"results": [
        {
            "id": r.id, "type": r.type, "name": r.name,
            "text": r.text, "path": r.path, **r.meta
        }
        for r in results
    ]}, ensure_ascii=False))
    
    return "\n".join(out)


def extract_sources_from_tool_results(tool_results: List[str]) -> List[dict]:
    seen_ids: set = set()
    sources: List[dict] = []

    for raw in tool_results:
        # 폴백용 글로벌 경로 요약
        global_path = ""
        if "[탐색 경로 요약] " in raw:
            path_start = raw.find("[탐색 경로 요약] ") + len("[탐색 경로 요약] ")
            path_end = raw.find("\n", path_start)
            global_path = raw[path_start:path_end].strip()

        try:
            json_str = raw[raw.find("{"):] if "{" in raw else ""
            if not json_str:
                continue
            data = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            continue
            
        for item in data.get("results", []):
            item_id = item.get("id")
            if not item_id or item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            
            # 개별 경로(Provenance) 우선, 없으면 글로벌 경로
            res_path = item.get("path") or global_path
            
            source = {
                "no":   len(sources) + 1,
                "id":   item_id,
                "type": item.get("type", "Unknown"),
                "name": item.get("name", ""),
                "path": res_path
            }
            if "authors" in item:
                source["authors"] = item["authors"]
            if "year" in item:
                source["year"] = item["year"]
            sources.append(source)

    return sources

# LLM에게 노출되는 유일한 도구
SEMANTIC_TOOLS = [execute_dynamic_search]
