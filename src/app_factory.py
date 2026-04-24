"""
Application Factory
- 모든 컴포넌트를 조립하여 실행 가능한 에이전트 반환
- LangGraph 관련 임포트는 지연 로딩 → Core 레이어 단독 테스트 가능
"""

from __future__ import annotations
from typing import Optional
import logging

from core.compiler.schema_registry import SchemaRegistry
from core.compiler.cypher_compiler import CypherCompiler
from core.executor.beam_pruner import BeamPruner
from core.executor.execution_engine import ExecutionEngine
from common.cache import make_cache
from common.query_config import RequestConfig
from infrastructure.config_repository import MemoryConfigRepository
from services.semantic_tools import set_engine

logger = logging.getLogger(__name__)


def make_llm(provider: str, model: str, temperature: float, base_url: str):
    """
    LLM 클라이언트 팔토리.

    provider:
        "ollama"            → ChatOllama  (Ollama 로컈 서버)
        "vllm" | "triton"   → ChatOpenAI  (OpenAI 호환 엔드포인트, api_key="EMPTY")
    """
    provider = provider.lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
        )

    if provider in ("vllm", "triton", "openai_compatible"):
        from langchain_openai import ChatOpenAI
        # vLLM / Triton은 OpenAI 호환 엔드포인트 제공 (base_url/v1)
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=base_url.rstrip("/") + "/v1",
            api_key="EMPTY",
        )

    raise ValueError(
        f"Unknown llm_provider: {provider!r}. "
        "Supported: ollama | vllm | triton"
    )


def _open_neo4j(settings) -> object:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    driver.verify_connectivity()
    logger.info("Neo4j 연결 성공: %s", settings.neo4j_uri)
    return driver


def _open_milvus(settings) -> object:
    from pymilvus import MilvusClient
    client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
    logger.info("Milvus 연결 성공: %s:%s", settings.milvus_host, settings.milvus_port)
    return client


def make_config_repo(overrides=None) -> MemoryConfigRepository:
    """ConfigRepository 생성. 실제 RDB 연동 전까지 MemoryConfigRepository 사용."""
    return MemoryConfigRepository(overrides)


def create_engine(
    neo4j_driver=None,
    milvus_client=None,
    config_repo=None,
    settings=None,
) -> ExecutionEngine:
    """
    실행 엔진 생성 (LangGraph 없이도 사용 가능).
    neo4j_driver / milvus_client 미전달 시 settings에서 자동 연결.
    core/LLM 기본 설정은 config_repo(DB)에서 읽는다.
    """
    if settings is None:
        from common.settings import get_settings
        settings = get_settings()

    repo = config_repo or make_config_repo()
    beam_width = RequestConfig._resolve(repo).beam_width

    if neo4j_driver is None:
        neo4j_driver = _open_neo4j(settings)
    if milvus_client is None:
        milvus_client = _open_milvus(settings)

    schema_registry = SchemaRegistry(driver=neo4j_driver)
    compiler = CypherCompiler(schema_registry=schema_registry)

    st_model = settings.sentence_transformer_model
    from sentence_transformers import SentenceTransformer
    vectorizer = SentenceTransformer(st_model)
    logger.info("SentenceTransformer loaded: %s", st_model)

    pruner = BeamPruner(beam_width=beam_width)

    cache = make_cache(
        redis_url=settings.redis_url,
        ttl=settings.cache_ttl_seconds,
    )

    from infrastructure.neo4j import make_graph_query_fn, make_fetch_details_fn
    from infrastructure.milvus import make_vector_search_fn

    graph_fn   = make_graph_query_fn(neo4j_driver)
    details_fn = make_fetch_details_fn(neo4j_driver)
    vector_fn  = make_vector_search_fn(
        milvus_client,
        embedding_fn=vectorizer.encode,
    )
    logger.info("DB 연결 완료 (Neo4j + Milvus)")

    engine = ExecutionEngine(
        compiler=compiler,
        pruner=pruner,
        vector_search_fn=vector_fn,
        graph_query_fn=graph_fn,
        fetch_details_fn=details_fn,
        cache=cache,
    )
    set_engine(engine)
    return engine


def create_app(
    engine,
    config_repo,
    settings,
):
    """
    LangGraph 에이전트 앱 생성.
    Returns:
        app (CompiledGraph)
    """
    from services.agent_graph import build_graph

    schema_registry = engine.compiler.schema_registry
    defaults = RequestConfig._resolve(config_repo)

    llm = make_llm(
        provider    = settings.llm_provider if settings else "ollama",
        model       = defaults.model,
        temperature = defaults.temperature,
        base_url    = settings.llm_base_url,
    )

    app = build_graph(schema_registry=schema_registry, llm=llm, max_tool_calls=defaults.max_tool_calls)
    
    logger.info(
        "LangGraph Agent 준비 완료 (provider=%s, base_url=%s, model=%s, max_tool_calls=%d)",
        settings.llm_provider if settings else "ollama",
        settings.llm_base_url,
        defaults.model, defaults.max_tool_calls,
    )
    return app
