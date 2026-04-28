"""
Semantic Tools (High-level)
- LLM에게 노출되는 도구: execute_dynamic_search, get_node_by_ids
- 내부적으로 Vector DB 진입과 Neo4j 그래프 탐색을 명시적으로 구분하여 QueryPlan AST로 조합
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging
import json

from langchain_core.tools import tool

from common.config.query_config import RequestConfig
from common.utils.exceptions import InvalidNodeType, ToolError
from common.types.query_plan import (
    EntrySearch, FinalFilter, HopDirection, HopSpec, QueryPlan,
)
from core.executor.execution_engine import ExecutionEngine
from common.types.results import NodeResult

logger = logging.getLogger(__name__)

VALID_NODE_TYPES: frozenset = frozenset({
    "Project", "Researcher", "Organization", "Paper", "Patent", "Report",
})


# ────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────────

def _parse_hops(neo4j_hops: List[Dict[str, str]], entry_node_type: str = "") -> List[HopSpec]:
    """hop dict 리스트 → HopSpec 리스트 변환. from_type/to_type 유효성 및 연속성 검증 포함."""
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

        # 연속성 검증: hop[0].from_type은 entry와 일치해야 하고
        # hop[i].from_type은 hop[i-1].to_type과 일치해야 함
        expected = entry_node_type if i == 0 else parsed[i - 1].to_type
        if expected and from_type != expected:
            raise InvalidNodeType(
                f"hop[{i}].from_type 불일치: "
                f"{'entry' if i == 0 else f'hop[{i-1}].to_type'}은 {expected!r}인데 "
                f"hop[{i}].from_type이 {from_type!r}입니다. "
                f"플랜의 탐색 경로가 연결되지 않습니다."
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
        payload: Dict[str, Any] = {"total": 0, "path": "", "results": []}
        failed_hop = getattr(stats, "failed_hop", None)
        entry_count = getattr(stats, "entry_count", 0)
        if failed_hop:
            payload["diagnostics"] = failed_hop
        elif entry_count == 0:
            payload["diagnostics"] = "Entry 벡터 검색 결과 없음 — vector_search_concept를 더 구체적으로 변경하세요."
        return json.dumps(payload, ensure_ascii=False)

    items = []
    for r in results:
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
# Tool Factory
# ────────────────────────────────────────────────────────────────────────────

def make_semantic_tools(engine: ExecutionEngine, fetch_details_fn) -> list:
    """engine과 fetch_details_fn을 클로저로 캡처한 LangChain 도구 목록 반환."""

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
        if vector_search_node_type not in VALID_NODE_TYPES:
            return json.dumps({
                "total": 0, "path": "", "results": [],
                "diagnostics": f"vector_search_node_type 오류: {vector_search_node_type!r}은 유효하지 않습니다. "
                               f"허용값: {sorted(VALID_NODE_TYPES)}",
            }, ensure_ascii=False)
        max_results = max(1, max_results)

        cfg = RequestConfig.current()

        try:
            parsed_hops = _parse_hops(neo4j_hops, entry_node_type=vector_search_node_type)
        except InvalidNodeType as e:
            return json.dumps({
                "total": 0, "path": "", "results": [],
                "diagnostics": f"플랜 오류 — {e}",
            }, ensure_ascii=False)
        except Exception as e:
            raise ToolError(f"neo4j_hops 파싱 실패: {e}") from e

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
    def get_details_by_ids(node_ids: List[str]) -> str:
        """
        노드 ID 목록으로 MariaDB에서 상세 정보(초록, 저자, 키워드, 전체 설명)를 조회합니다.
        execute_dynamic_search의 기본 정보만으로는 부족할 때 사용합니다.

        사용 시점:
        - 사용자가 논문·특허·보고서의 내용·초록을 요청한 경우
        - 저자·발명자 목록이 필요한 경우
        - 특정 ID의 상세 정보를 명시적으로 요청한 경우

        Args:
            node_ids: 조회할 노드 ID 목록 (검색 결과의 "id" 필드값)
        """
        if not node_ids:
            return json.dumps({"total": 0, "path": "", "results": []}, ensure_ascii=False)

        cfg = RequestConfig.current()

        try:
            results = fetch_details_fn(node_ids[:cfg.max_results])
        except Exception as e:
            logger.error("[Tool] get_node_by_ids 실패: %s", e)
            raise ToolError(f"노드 조회 중 오류가 발생했습니다: {type(e).__name__}") from e

        class _DirectStats:
            path_summary = "직접 ID 조회"

        return _format_results(results, _DirectStats())

    return [execute_dynamic_search, get_details_by_ids]

