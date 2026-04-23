"""
Execution Engine
- QueryPlan을 hop-by-hop으로 실행
- 각 hop 후 BeamPruner로 결과 압축
- Redis 캐싱으로 중복 쿼리 방지
- 실행 통계 수집 (Observability)
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional
import logging
import time

from common.query_plan import EntrySearch, FinalFilter, HopSpec, QueryPlan
from common.config_service import QueryConfig
from core.compiler.cypher_compiler import CypherCompiler
from core.executor.beam_pruner import BeamPruner
from common.cache import CacheBackend, make_cache_key


class _NullCache:
    """cache=None 시 사용하는 no-op 콜렉. 캐싱 기능 없이 항상 DB를 품다."""
    def get(self, key: str): return None
    def set(self, key: str, value) -> None: pass
    def clear(self) -> None: pass

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Result types
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class NodeResult:
    id:   str
    type: str
    name: Optional[str]    = None
    text: Optional[str]    = None
    path: str              = ""              # 이 노드에 도달하기까지의 탐색 경로 (Provenance)
    meta: Dict[str, Any]   = field(default_factory=dict)


@dataclass
class LayerTiming:
    label:      str
    elapsed_ms: float
    extra:      Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionStats:
    total_elapsed_s:    float             = 0.0
    hop_counts:         List[int]         = field(default_factory=list)  # 각 hop 후 노드 수
    path_summary:       str               = ""  # 에이전트용 가벼운 탐색 경로 요약 (문자열)
    cache_hits:         int               = 0
    db_calls:           int               = 0
    pruned_total:       int               = 0
    layer_timings:      List[LayerTiming] = field(default_factory=list)

    def timing_summary(self) -> str:
        if not self.layer_timings:
            return "(no timing data)"
        lines = ["─── Layer Timing ──────────────────────────────────────────────"]
        for t in self.layer_timings:
            extra_str = ("  " + "  ".join(f"{k}={v}" for k, v in t.extra.items())) if t.extra else ""
            lines.append(f"  {t.label:<50} {t.elapsed_ms:>8.1f} ms{extra_str}")
        lines.append(f"  {'TOTAL':<50} {self.total_elapsed_s * 1000:>8.1f} ms")
        lines.append("────────────────────────────────────────────────────────────")
        return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# Timing helper
# ────────────────────────────────────────────────────────────────────────────

@contextmanager
def _timed(stats: ExecutionStats, label: str, **extra) -> Generator[None, None, None]:
    """계층별 소요시간을 stats.layer_timings에 누적한다."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        ms = (time.perf_counter() - t0) * 1000
        stats.layer_timings.append(LayerTiming(label=label, elapsed_ms=ms, extra=dict(extra)))
        logger.debug("[⏱] %-50s %8.1f ms", label, ms)


# ────────────────────────────────────────────────────────────────────────────
# Execution Engine
# ────────────────────────────────────────────────────────────────────────────

class ExecutionEngine:
    """
    QueryPlan 실행기.

    외부 DB는 콜백(callable)으로 주입받아 테스트 용이성 확보:
      - vector_search_fn:   (query, node_type, filters, top_k) → List[str]  (IDs)
      - graph_query_fn:     (cypher_query) → List[dict]
      - fetch_details_fn:   (ids) → List[NodeResult]
      - fetch_texts_fn:     (ids) → List[str]  (for BeamPruner)
    """

    def __init__(
        self,
        compiler:         CypherCompiler,
        pruner:           BeamPruner,
        vector_search_fn: Callable,
        graph_query_fn:   Callable,
        fetch_details_fn: Callable,
        fetch_texts_fn:   Callable,
        cache:            Optional[CacheBackend] = None,
    ):
        self.compiler         = compiler
        self.pruner           = pruner
        self.vector_search_fn = vector_search_fn
        self.graph_query_fn   = graph_query_fn
        self.fetch_details_fn = fetch_details_fn
        self.fetch_texts_fn   = fetch_texts_fn
        self._cache: CacheBackend = cache if cache is not None else _NullCache()  # type: ignore[assignment]

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def run(
        self,
        plan: QueryPlan,
        original_query: str = "",
        config: Optional[QueryConfig] = None,
    ) -> tuple[List[NodeResult], ExecutionStats]:
        """
        QueryPlan 전체 실행.

        Args:
            plan:           LLM이 생성한 플랜
            original_query: 사용자 원본 쿼리 (BeamPruner에 전달)
            config:         요청별 설정 오버라이드 (beam_width, max_results 등)

        Returns:
            (결과 노드 목록, 실행 통계)
        """
        beam_width             = config.beam_width             if config and config.beam_width             is not None else None
        max_results            = config.max_results            if config and config.max_results            is not None else plan.max_results
        vector_score_threshold = config.vector_score_threshold if config and config.vector_score_threshold is not None else None

        stats = ExecutionStats()
        t_start = time.time()

        logger.info("ExecutionEngine.run()\n%s", plan.describe())

        # Step 1 & 2: Vector DB 진입 및 그래프 탐색
        # 개별 노드별 도달 경로(Provenance) 추적용 맵
        provenance: Dict[str, str] = {}
        entry_ids = self._run_entry_search(plan.entry_search, stats, provenance,
                                           score_threshold=vector_score_threshold)

        if not entry_ids:
            logger.warning("Entry search returned no results.")
            stats.total_elapsed_s = time.time() - t_start
            return [], stats

        current_ids = entry_ids
        # 이미 거쳐온 노드들을 다시 방문하지 않도록 히스토리 유지
        traversal_history_ids = set(current_ids)
        query_context = original_query or plan.entry_search.concept

        for hop_idx, hop in enumerate(plan.traversal_hops):
            current_ids = self._run_single_hop(
                hop, current_ids, hop_idx, query_context, max_results, stats,
                exclude_ids=list(traversal_history_ids),
                provenance=provenance,
                beam_width=beam_width,
                score_threshold=vector_score_threshold,
            )
            if not current_ids:
                logger.warning("Hop %d returned no results, stopping.", hop_idx)
                break
            traversal_history_ids.update(current_ids)

        # Step 3: 최종 필터 (있을 경우 Vector 재검색으로 의미 필터)
        if plan.final_filter and current_ids:
            current_ids = self._run_final_filter(
                plan.final_filter, current_ids, stats,
                score_threshold=vector_score_threshold,
            )

        # Step 4: 상세 정보 로드
        results = self._fetch_details(current_ids[:max_results], stats)
        
        # 각 결과물에 추적된 개별 경로 주입
        for r in results:
            r.path = provenance.get(r.id, stats.path_summary)

        stats.total_elapsed_s = time.time() - t_start
        logger.info(
            "Execution done: %d results in %.2fs | db_calls=%d cache_hits=%d pruned=%d",
            len(results), stats.total_elapsed_s,
            stats.db_calls, stats.cache_hits, stats.pruned_total,
        )
        logger.info("\n%s", stats.timing_summary())
        return results, stats

    # ------------------------------------------------------------------ #
    # Steps
    # ------------------------------------------------------------------ #

    def _run_entry_search(
        self, entry: EntrySearch, stats: ExecutionStats, provenance: Dict[str, str],
        score_threshold: Optional[float] = None,
    ) -> List[str]:
        cache_key = make_cache_key("entry", entry.model_dump())

        def _clean(s): return " ".join(str(s).split()) if s else ""

        if cached := self._cache.get(cache_key):
            stats.cache_hits += 1
            stats.hop_counts.append(len(cached))
            path_seg = f"시작({entry.node_type}: '{entry.concept}')"

            sample_nodes = self.fetch_details_fn(cached)
            if sample_nodes:
                path_seg += f"['{_clean(sample_nodes[0].name or sample_nodes[0].id)}']"
            for sn in sample_nodes:
                provenance[sn.id] = f"시작({sn.type}: '{entry.concept}')['{_clean(sn.name or sn.id)}']"

            stats.path_summary = path_seg
            logger.info("[Engine] Entry '%s' (%s): %d건 (cache hit)", entry.concept, entry.node_type, len(cached))
            return cached

        with _timed(stats, "L4 | Milvus  vector_search",
                    concept=entry.concept, node_type=entry.node_type):
            ids = self.vector_search_fn(
                entry.concept, entry.node_type, entry.filters, entry.top_k,
                score_threshold=score_threshold,
            )
        stats.db_calls += 1
        stats.hop_counts.append(len(ids))

        logger.info("[Engine] Entry '%s' (%s): %d건", entry.concept, entry.node_type, len(ids))

        path_seg = f"시작({entry.node_type}: '{entry.concept}')"
        if ids:
            sample_nodes = self.fetch_details_fn(ids)
            if sample_nodes:
                path_seg += f"['{_clean(sample_nodes[0].name or sample_nodes[0].id)}']"

            for sn in sample_nodes:
                clean_name = _clean(sn.name or sn.id)
                provenance[sn.id] = f"시작({sn.type}: '{entry.concept}')['{clean_name}']"

            logger.debug("[Engine] Entry 상위 노드: %s",
                         ", ".join(f"[{sn.id}] {_clean(sn.name or sn.id)}" for sn in sample_nodes[:3]))

        stats.path_summary = path_seg
        self._cache.set(cache_key, ids)
        return ids

    def _run_single_hop(
        self,
        hop:           HopSpec,
        start_ids:     List[str],
        hop_idx:       int,
        query_context: str,
        max_results:   int,
        stats:         ExecutionStats,
        exclude_ids:   Optional[List[str]] = None,
        provenance:    Optional[Dict[str, str]] = None,
        beam_width:    Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[str]:
        cache_key = make_cache_key(
            "hop", {"hop": hop.model_dump(), "ids": sorted(start_ids), "ctx": query_context, "ex": sorted(exclude_ids or [])}
        )
        if cached := self._cache.get(cache_key):
            stats.cache_hits += 1
            stats.hop_counts.append(len(cached))
            stats.path_summary += f" -[{hop.relation_concept}]-> {hop.to_type}"
            logger.info("[Engine] Hop %d (%s): %d건 (cache hit)", hop_idx + 1, hop.relation_concept, len(cached))
            return cached

        # L2: Cypher 컴파일
        with _timed(stats, f"L2 | compile        hop {hop_idx + 1}"):
            cypher = self.compiler.compile_single_hop(
                hop, start_ids, limit=self.pruner.beam_width * 3, exclude_ids=exclude_ids
            )
        # L4: Neo4j 그래프 쿼리
        with _timed(stats, f"L4 | Neo4j   graph_query hop {hop_idx + 1}"):
            raw = self.graph_query_fn(cypher)
        stats.db_calls += 1

        result_ids = [r["id"] for r in raw]
        before = len(result_ids)

        # L3: BeamPruner 의미적 압축 (내부에서 fetch_texts 호출 포함)
        with _timed(stats, f"L3 | BeamPrune      hop {hop_idx + 1}", before=before):
            result_ids = self.pruner.prune(result_ids, query_context, self.fetch_texts_fn,
                                           beam_width=beam_width, score_threshold=score_threshold)
        stats.pruned_total += max(0, before - len(result_ids))
        stats.hop_counts.append(len(result_ids))

        logger.info("[Engine] Hop %d (%s -> %s): %d건 (pruned %d)",
                    hop_idx + 1, hop.relation_concept, hop.to_type,
                    len(result_ids), before - len(result_ids))

        def _clean(s): return " ".join(str(s).split()) if s else ""

        path_seg = f" -[{hop.relation_concept}]-> {hop.to_type}"
        result_id_set = set(result_ids)
        first_found = False

        for r_node in raw:
            rid = r_node["id"]
            if rid not in result_id_set:
                continue

            p_id = r_node.get("start_id")
            name = _clean(r_node.get("name") or rid)

            if provenance is not None and p_id and p_id in provenance:
                provenance[rid] = provenance[p_id] + f" -[{hop.relation_concept}]-> {hop.to_type}('{name}')"

            if not first_found and rid in result_ids[:3]:
                path_seg += f"('{name}')"
                first_found = True

        logger.debug("[Engine] Hop %d 상위 노드: %s", hop_idx + 1,
                     ", ".join(f"[{r['id']}] {_clean(r.get('name') or r['id'])}"
                               for r in raw[:3] if r["id"] in result_id_set))

        stats.path_summary += path_seg
        self._cache.set(cache_key, result_ids)
        return result_ids

    def _run_final_filter(
        self,
        ffilter:     FinalFilter,
        current_ids: List[str],
        stats:       ExecutionStats,
        score_threshold: Optional[float] = None,
    ) -> List[str]:
        if not ffilter.concept:
            return current_ids

        # 현재 ID 집합 안에서 의미 필터: Vector 재검색 + intersection
        with _timed(stats, "L4 | Milvus  final_filter", concept=ffilter.concept):
            filtered = self.vector_search_fn(
                ffilter.concept,
                ffilter.node_type or "",
                {**ffilter.filters, "id_in": current_ids},
                top_k=500,
                score_threshold=score_threshold,
            )
        stats.db_calls += 1
        result = [i for i in filtered if i in set(current_ids)]
        logger.info(
            "Final filter '%s': %d → %d nodes",
            ffilter.concept, len(current_ids), len(result),
        )
        return result

    def _fetch_details(
        self, ids: List[str], stats: ExecutionStats
    ) -> List[NodeResult]:
        if not ids:
            return []
        with _timed(stats, "L4 | Neo4j   fetch_details", count=len(ids)):
            results = self.fetch_details_fn(ids)
        stats.db_calls += 1
        return results

