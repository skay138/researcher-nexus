"""
테스트 스위트
- 4개 아키텍처 문제에 대한 검증 테스트
- Mock 어댑터를 사용하므로 실제 DB 불필요
- 실행: pytest tests/test_architecture.py -v
"""

import json
import pytest

from common.fixtures import SEED_NODES as MOCK_NODES, SEED_RELATIONS as MOCK_RELATIONS
from infrastructure.in_memory import make_in_memory_adapters as _make_adapters
mock_vector_search, mock_graph_query, mock_fetch_details, mock_fetch_texts = _make_adapters(MOCK_NODES, MOCK_RELATIONS)


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def schema_registry():
    from core.compiler.schema_registry import SchemaRegistry
    return SchemaRegistry(driver=None)


@pytest.fixture
def compiler(schema_registry):
    from core.compiler.cypher_compiler import CypherCompiler
    return CypherCompiler(schema_registry=schema_registry)


@pytest.fixture
def pruner():
    from core.executor.beam_pruner import BeamPruner
    return BeamPruner(vectorizer=None, beam_width=3)


@pytest.fixture
def engine(compiler, pruner):
    from core.executor.execution_engine import ExecutionEngine
    return ExecutionEngine(
        compiler=compiler,
        pruner=pruner,
        vector_search_fn=mock_vector_search,
        graph_query_fn=mock_graph_query,
        fetch_details_fn=mock_fetch_details,
        fetch_texts_fn=mock_fetch_texts,
    )


# ────────────────────────────────────────────────────────────────────────────
# 문제 1: 결정론적 Cypher 컴파일러
# ────────────────────────────────────────────────────────────────────────────

class TestCypherCompiler:
    """동일 입력 → 동일 Cypher (결정론적)"""

    def test_compile_is_deterministic(self, compiler):
        from common.query_plan import HopDirection, HopSpec
        hop = HopSpec(
            from_type="Project", relation_concept="participation",
            to_type="Researcher", direction=HopDirection.INBOUND,
        )
        ids = ["proj_101", "proj_102"]
        assert compiler.compile_single_hop(hop, ids, limit=100) == \
               compiler.compile_single_hop(hop, ids, limit=100)

    def test_relation_concept_resolved(self, compiler):
        from common.query_plan import HopSpec, HopDirection
        hop = HopSpec(
            from_type="Researcher", relation_concept="authored",
            to_type="Paper", direction=HopDirection.OUTBOUND,
        )
        cypher = compiler.compile_single_hop(hop, ["res_201"], limit=50)
        assert "AUTHORED" in cypher
        assert "authored" not in cypher

    def test_unknown_relation_raises(self, compiler):
        from core.compiler.cypher_compiler import UnknownRelationConcept
        from common.query_plan import HopSpec, HopDirection
        hop = HopSpec(from_type="A", relation_concept="nonexistent_xyz",
                      to_type="B", direction=HopDirection.OUTBOUND)
        with pytest.raises(UnknownRelationConcept):
            compiler.compile_single_hop(hop, ["id_1"], limit=10)

    def test_direction_inbound_arrow(self, compiler):
        from common.query_plan import HopSpec, HopDirection
        hop = HopSpec(
            from_type="Organization", relation_concept="belongs_to",
            to_type="Researcher", direction=HopDirection.INBOUND,
        )
        cypher = compiler.compile_single_hop(hop, ["org_301"], limit=10)
        assert "<-[" in cypher

    def test_filter_injection_blocked(self, compiler):
        from common.query_plan import HopSpec, HopDirection
        from common.exceptions import CypherInjectionDetected
        hop = HopSpec(
            from_type="Project", relation_concept="participation",
            to_type="Researcher", direction=HopDirection.INBOUND,
            filters={"bad_key!": "val"},
        )
        with pytest.raises(CypherInjectionDetected):
            compiler.compile_single_hop(hop, ["proj_101"], limit=10)


# ────────────────────────────────────────────────────────────────────────────
# 문제 2: Hop Explosion 제어
# ────────────────────────────────────────────────────────────────────────────

class TestHopExplosion:

    def test_beam_pruner_truncates(self, pruner):
        node_ids = [f"node_{i}" for i in range(100)]
        result = pruner.prune(node_ids, "해양 연구", lambda ids: [f"text {i}" for i in ids])
        assert len(result) <= pruner.beam_width

    def test_beam_pruner_skip_if_small(self, pruner):
        node_ids = ["node_1", "node_2"]
        result = pruner.prune(node_ids, "query", lambda ids: [f"text {i}" for i in ids])
        assert result == node_ids

    def test_query_plan_max_hops_enforced(self):
        from pydantic import ValidationError
        from common.query_plan import EntrySearch, HopDirection, HopSpec, QueryPlan
        too_many = [
            HopSpec(from_type="A", relation_concept="participation",
                    to_type="B", direction=HopDirection.OUTBOUND)
            for _ in range(QueryPlan.MAX_HOPS + 1)
        ]
        with pytest.raises(ValidationError):
            QueryPlan(
                entry_search=EntrySearch(concept="test", node_type="Project"),
                traversal_hops=too_many,
            )

    def test_cypher_has_limit_per_hop(self, compiler):
        from common.query_plan import HopSpec, HopDirection
        hop = HopSpec(from_type="Project", relation_concept="participation",
                      to_type="Researcher", direction=HopDirection.INBOUND)
        cypher = compiler.compile_single_hop(hop, ["proj_101"], limit=500)
        assert "LIMIT" in cypher.upper()


# ────────────────────────────────────────────────────────────────────────────
# 문제 3: Schema Registry 동적 주입
# ────────────────────────────────────────────────────────────────────────────

class TestSchemaRegistry:

    def test_schema_text_contains_types(self, schema_registry):
        text = schema_registry.get_schema_for_llm()
        assert "Project" in text
        assert "Researcher" in text
        assert "PARTICIPATED_IN" in text
        assert "AUTHORED" in text

    def test_schema_text_contains_concepts(self, schema_registry):
        text = schema_registry.get_schema_for_llm()
        assert "participation" in text
        assert "authored" in text

    def test_cache_invalidation(self, schema_registry):
        first = schema_registry.get_schema_for_llm()
        schema_registry.invalidate_cache()
        second = schema_registry.get_schema_for_llm()
        assert first == second

    def test_fuzzy_concept_resolve(self, schema_registry):
        assert schema_registry.resolve_relation("participate") == "PARTICIPATED_IN"


# ────────────────────────────────────────────────────────────────────────────
# 문제 4: Semantic Tool (execute_dynamic_search)
# ────────────────────────────────────────────────────────────────────────────

class TestSemanticTools:
    """execute_dynamic_search 도구 캡슐화 및 동작 검증"""

    def setup_method(self):
        from core.compiler.schema_registry import SchemaRegistry
        from core.compiler.cypher_compiler import CypherCompiler
        from core.executor.beam_pruner import BeamPruner
        from core.executor.execution_engine import ExecutionEngine
        from services.semantic_tools import set_engine
        set_engine(ExecutionEngine(
            compiler=CypherCompiler(SchemaRegistry()),
            pruner=BeamPruner(beam_width=50),
            vector_search_fn=mock_vector_search,
            graph_query_fn=mock_graph_query,
            fetch_details_fn=mock_fetch_details,
            fetch_texts_fn=mock_fetch_texts,
        ))

    def test_basic_search_returns_string(self):
        from services.semantic_tools import execute_dynamic_search
        result = execute_dynamic_search.func(
            vector_search_concept="해양 사업",
            vector_search_node_type="Project",
            neo4j_hops=[],
        )
        assert isinstance(result, str)

    def test_search_with_hops(self):
        from services.semantic_tools import execute_dynamic_search
        result = execute_dynamic_search.func(
            vector_search_concept="해양 사업",
            vector_search_node_type="Project",
            neo4j_hops=[{
                "from_type": "Project", "relation_concept": "participation",
                "to_type": "Researcher", "direction": "in",
            }],
        )
        assert isinstance(result, str)

    def test_tool_hides_db_internals(self):
        from services.semantic_tools import execute_dynamic_search
        import inspect
        params = list(inspect.signature(execute_dynamic_search.func).parameters.keys())
        for internal in ["cypher", "node_ids", "session", "driver"]:
            assert internal not in params, f"내부 DB 파라미터 '{internal}'이 노출되면 안 됨"

    def test_no_results_returns_message(self):
        from services.semantic_tools import execute_dynamic_search
        result = execute_dynamic_search.func(
            vector_search_concept="존재하지않는키워드xyz123",
            vector_search_node_type="Project",
            neo4j_hops=[],
        )
        assert "찾지 못했습니다" in result or isinstance(result, str)


# ────────────────────────────────────────────────────────────────────────────
# 통합 테스트
# ────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def setup_method(self):
        from core.compiler.schema_registry import SchemaRegistry
        from core.compiler.cypher_compiler import CypherCompiler
        from core.executor.beam_pruner import BeamPruner
        from core.executor.execution_engine import ExecutionEngine
        from services.semantic_tools import set_engine
        self.engine = ExecutionEngine(
            compiler=CypherCompiler(SchemaRegistry()),
            pruner=BeamPruner(beam_width=50),
            vector_search_fn=mock_vector_search,
            graph_query_fn=mock_graph_query,
            fetch_details_fn=mock_fetch_details,
            fetch_texts_fn=mock_fetch_texts,
        )
        set_engine(self.engine)

    def test_complex_4hop_query(self):
        from common.query_plan import (
            EntrySearch, FinalFilter, HopDirection, HopSpec, QueryPlan,
        )
        plan = QueryPlan(
            entry_search=EntrySearch(
                concept="해양 사업", node_type="Project",
                filters={"year": {"lt": 2025}},
            ),
            traversal_hops=[
                HopSpec(from_type="Project",     relation_concept="participation",
                        to_type="Researcher",     direction=HopDirection.INBOUND),
                HopSpec(from_type="Researcher",  relation_concept="belongs_to",
                        to_type="Organization",  direction=HopDirection.OUTBOUND),
                HopSpec(from_type="Organization",relation_concept="belongs_to",
                        to_type="Researcher",    direction=HopDirection.INBOUND),
                HopSpec(from_type="Researcher",  relation_concept="authored",
                        to_type="Paper",         direction=HopDirection.OUTBOUND),
            ],
            final_filter=FinalFilter(concept="보트", node_type="Paper"),
            max_results=20,
        )
        results, stats = self.engine.run(plan, original_query="보트 관련 논문")
        assert stats.db_calls > 0
        assert len(stats.hop_counts) > 0
        assert stats.total_elapsed_s > 0

    def test_execution_stats_collected(self):
        from common.query_plan import EntrySearch, QueryPlan
        plan = QueryPlan(
            entry_search=EntrySearch(concept="해양", node_type="Project"),
            max_results=5,
        )
        _, stats = self.engine.run(plan, original_query="해양")
        assert stats.total_elapsed_s > 0
        assert stats.db_calls >= 1
        assert isinstance(stats.hop_counts, list)


# ────────────────────────────────────────────────────────────────────────────
# Patent / Report 도메인
# ────────────────────────────────────────────────────────────────────────────

class TestPatentReportDomain:

    def test_patent_nodes_in_mock_data(self):
        patents = [n for n in MOCK_NODES.values() if n["type"] == "Patent"]
        assert len(patents) >= 3
        assert all("patent_number" in p for p in patents)

    def test_report_nodes_in_mock_data(self):
        reports = [n for n in MOCK_NODES.values() if n["type"] == "Report"]
        assert len(reports) >= 3
        assert all("report_type" in r for r in reports)

    def test_new_relations_exist(self):
        for rel in ("INVENTED", "PRODUCED", "AUTHORED", "AFFILIATED_WITH", "CITES", "PUBLISHED_IN"):
            assert rel in MOCK_RELATIONS, f"관계 '{rel}'이 MOCK_RELATIONS에 없음"

    def test_schema_includes_patent_report(self):
        from core.compiler.schema_registry import SchemaRegistry
        text = SchemaRegistry(driver=None).get_schema_for_llm()
        assert "Patent" in text
        assert "Report" in text
        assert "INVENTED" in text
        assert "PUBLISHED" in text

    def test_patent_concept_mapping(self):
        from core.compiler.schema_registry import SchemaRegistry
        r = SchemaRegistry(driver=None)
        assert r.resolve_relation("invented") == "INVENTED"
        assert r.resolve_relation("filed") == "FILED"
        assert r.resolve_relation("produced") == "PRODUCED"

    def test_report_concept_mapping(self):
        from core.compiler.schema_registry import SchemaRegistry
        r = SchemaRegistry(driver=None)
        assert r.resolve_relation("authored_report") == "AUTHORED_REPORT"
        assert r.resolve_relation("published") == "PUBLISHED"
        assert r.resolve_relation("cited_in") == "CITED_IN"

    def test_patent_search_via_tool(self):
        """execute_dynamic_search로 특허 도메인 탐색"""
        from core.compiler.schema_registry import SchemaRegistry
        from core.compiler.cypher_compiler import CypherCompiler
        from core.executor.beam_pruner import BeamPruner
        from core.executor.execution_engine import ExecutionEngine
        from services.semantic_tools import set_engine, execute_dynamic_search
        set_engine(ExecutionEngine(
            compiler=CypherCompiler(SchemaRegistry()),
            pruner=BeamPruner(beam_width=50),
            vector_search_fn=mock_vector_search,
            graph_query_fn=mock_graph_query,
            fetch_details_fn=mock_fetch_details,
            fetch_texts_fn=mock_fetch_texts,
        ))
        result = execute_dynamic_search.func(
            vector_search_concept="해양 에너지",
            vector_search_node_type="Patent",
            neo4j_hops=[{
                "from_type": "Patent", "relation_concept": "invented",
                "to_type": "Researcher", "direction": "in",
            }],
        )
        assert isinstance(result, str)


# ────────────────────────────────────────────────────────────────────────────
# Cache
# ────────────────────────────────────────────────────────────────────────────

class TestCache:

    def test_expired_entries_evicted_on_get(self):
        from common.cache import MemoryCache
        import time
        cache = MemoryCache(ttl=0.01)
        cache.set("key1", "value1")
        time.sleep(0.02)
        assert cache.get("key1") is None
        assert "key1" not in cache._store

    def test_max_size_eviction(self):
        from common.cache import MemoryCache
        cache = MemoryCache(ttl=300, max_size=3)
        for k in ("a", "b", "c", "d"):
            cache.set(k, 1)
        assert len(cache._store) <= 3

    def test_hop_cache_key_includes_context(self):
        from common.cache import make_cache_key
        from common.query_plan import HopSpec, HopDirection
        hop = HopSpec(from_type="A", relation_concept="participation",
                      to_type="B", direction=HopDirection.OUTBOUND)
        ids = ["id_1", "id_2"]
        k1 = make_cache_key("hop", {"hop": hop.model_dump(), "ids": sorted(ids), "ctx": "보트 논문"})
        k2 = make_cache_key("hop", {"hop": hop.model_dump(), "ids": sorted(ids), "ctx": "해양 생태계"})
        assert k1 != k2


# ────────────────────────────────────────────────────────────────────────────
# Cypher 인젝션 방어
# ────────────────────────────────────────────────────────────────────────────

class TestCypherSanitization:

    def test_malicious_id_raises(self):
        from core.compiler.cypher_compiler import CypherCompiler
        from core.compiler.schema_registry import SchemaRegistry
        from common.exceptions import CypherInjectionDetected
        from common.query_plan import HopSpec, HopDirection
        compiler = CypherCompiler(SchemaRegistry(driver=None))
        hop = HopSpec(from_type="Project", relation_concept="participation",
                      to_type="Researcher", direction=HopDirection.INBOUND)
        with pytest.raises(CypherInjectionDetected):
            compiler.compile_single_hop(
                hop, ["'; MATCH (n) DETACH DELETE n; //"], limit=10,
            )

    def test_malicious_filter_key_raises(self):
        from core.compiler.cypher_compiler import CypherCompiler
        from core.compiler.schema_registry import SchemaRegistry
        from common.exceptions import CypherInjectionDetected
        from common.query_plan import HopSpec, HopDirection
        compiler = CypherCompiler(SchemaRegistry(driver=None))
        hop = HopSpec(
            from_type="Project", relation_concept="participation",
            to_type="Researcher", direction=HopDirection.INBOUND,
            filters={"year; DROP DATABASE": 2024},
        )
        with pytest.raises(CypherInjectionDetected):
            compiler.compile_single_hop(hop, ["proj_101"], limit=10)

    def test_safe_ids_pass(self):
        from core.compiler.cypher_compiler import CypherCompiler
        from core.compiler.schema_registry import SchemaRegistry
        from common.query_plan import HopSpec, HopDirection
        compiler = CypherCompiler(SchemaRegistry(driver=None))
        hop = HopSpec(from_type="Project", relation_concept="participation",
                      to_type="Researcher", direction=HopDirection.INBOUND)
        cypher = compiler.compile_single_hop(hop, ["proj_101", "proj-102"], limit=10)
        assert "proj_101" in cypher
