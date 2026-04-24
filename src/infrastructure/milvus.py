"""
Milvus Vector Database Adapter — Hybrid Search (Dense COSINE + BM25)
사용법:
    from infrastructure.milvus import make_vector_search_fn
    from pymilvus import MilvusClient
    from sentence_transformers import SentenceTransformer

    client   = MilvusClient(uri="http://localhost:19530")
    embedder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    vector_fn = make_vector_search_fn(client, embedding_fn=embedder.encode)

컬렉션 스키마:
    id:        VARCHAR(64)           — 노드 ID (primary key)
    node_type: VARCHAR(32)           — Project / Researcher / Organization / Paper / Patent / Report
    year:      INT64                 — 연도 (필터용)
    text:      VARCHAR(65535)        — BM25 전문검색 원본 텍스트 (analyzer 활성화)
    sparse:    SPARSE_FLOAT_VECTOR   — BM25 Function이 text에서 자동 생성
    dense:     FLOAT_VECTOR(768)     — KR-SBERT dense 임베딩
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
import logging
import time

logger = logging.getLogger(__name__)

COLLECTION_NAME = "research_nodes"
VECTOR_DIM = 768  # KR-SBERT-V40K-klueNLI-augSTS 출력 차원


def make_vector_search_fn(
    client,
    embedding_fn: Callable[[List[str]], Any],
    collection_name: str = COLLECTION_NAME,
) -> Callable:
    """
    Milvus 클라이언트로 hybrid vector_search_fn 콜백 생성.
    인터페이스: (concept, node_type, filters, top_k) → List[tuple[str, float]]

    Dense(COSINE) + Sparse(BM25)를 WeightedRanker로 합산하여 반환.
    가중치는 요청별 RequestConfig(sparse_weight, dense_weight)에서 읽는다.
    """
    def vector_search(
        concept: str,
        node_type: str,
        filters: Dict[str, Any],
        top_k: int = 20,
    ) -> List[tuple[str, float]]:
        from pymilvus import AnnSearchRequest
        from common.query_config import RequestConfig
        cfg = RequestConfig.current()
        sparse_weight = cfg.sparse_weight
        dense_weight  = cfg.dense_weight

        t_embed = time.perf_counter()
        vector = embedding_fn([concept])[0].tolist()
        embed_ms = (time.perf_counter() - t_embed) * 1000

        # ── 필터 표현식 조합 ───────────────────────────────────────────────────
        filter_parts: List[str] = []

        if node_type:
            filter_parts.append(f'node_type == "{node_type}"')

        id_in: Optional[List[str]] = filters.get("id_in")
        if id_in:
            ids_expr = ", ".join(f'"{i}"' for i in id_in)
            filter_parts.append(f"id in [{ids_expr}]")

        year_filter = filters.get("year")
        if isinstance(year_filter, dict):
            if "lt"  in year_filter: filter_parts.append(f"year < {year_filter['lt']}")
            if "lte" in year_filter: filter_parts.append(f"year <= {year_filter['lte']}")
            if "gt"  in year_filter: filter_parts.append(f"year > {year_filter['gt']}")
            if "gte" in year_filter: filter_parts.append(f"year >= {year_filter['gte']}")

        expr = " && ".join(filter_parts)  # 빈 문자열 = 필터 없음
        search_limit = top_k * 2  # 각 채널에서 여유분 확보 후 RRF

        # ── Dense 요청 (KR-SBERT COSINE) ──────────────────────────────────────
        dense_req = AnnSearchRequest(
            data=[vector],
            anns_field="dense",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=search_limit,
            expr=expr or None,
        )
        # ── Sparse 요청 (BM25 전문검색) ───────────────────────────────────────
        sparse_req = AnnSearchRequest(
            data=[concept],
            anns_field="sparse",
            param={"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}},
            limit=search_limit,
            expr=expr or None,
        )

        t_search = time.perf_counter()
        try:
            from pymilvus import WeightedRanker
            results = client.hybrid_search(
                collection_name=collection_name,
                reqs=[dense_req, sparse_req],
                ranker=WeightedRanker(sparse_weight, dense_weight),
                limit=top_k,
                output_fields=["id"],
            )
        except Exception as e:
            # hybrid_search 미지원 환경(로컬 테스트 등) — dense fallback
            logger.warning("[Milvus] hybrid_search 실패, dense 단독 fallback: %s", e)
            results = client.search(
                collection_name=collection_name,
                data=[vector],
                limit=top_k,
                filter=expr or None,
                output_fields=["id"],
                search_params={"metric_type": "COSINE", "params": {}},
            )
        search_ms = (time.perf_counter() - t_search) * 1000

        if not results or not results[0]:
            logger.debug(
                "[Milvus] hybrid_search '%s' (%s): embed=%.1f ms  search=%.1f ms  hits=0",
                concept, node_type or "*", embed_ms, search_ms,
            )
            return []

        hits = results[0]
        logger.debug(
            "[Milvus] hybrid_search '%s' (%s): embed=%.1f ms  search=%.1f ms  hits=%d",
            concept, node_type or "*", embed_ms, search_ms, len(hits),
        )
        return [(hit["entity"]["id"], float(hit["distance"])) for hit in hits]

    return vector_search


def ensure_collection(client, collection_name: str = COLLECTION_NAME) -> None:
    """
    Milvus 하이브리드 컬렉션이 없으면 생성.
    Dense(HNSW/COSINE) + Sparse(BM25) 이중 인덱스.
    """
    from pymilvus import DataType, Function, FunctionType

    if client.has_collection(collection_name):
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id",        DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field("node_type", DataType.VARCHAR, max_length=32)
    schema.add_field("year",      DataType.INT64)
    schema.add_field(
        "text", DataType.VARCHAR, max_length=65535,
        enable_analyzer=True,
        analyzer_params={"tokenizer": "standard"},
    )
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

    # BM25: text 필드 → sparse 벡터 자동 변환
    schema.add_function(Function(
        name="text_to_sparse",
        input_field_names=["text"],
        output_field_names=["sparse"],
        function_type=FunctionType.BM25,
    ))

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense",
        metric_type="COSINE",
        index_type="HNSW",
        params={"M": 16, "efConstruction": 200},
    )
    index_params.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"bm25_k1": 1.2, "bm25_b": 0.75},
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    logger.info("[Milvus] 컬렉션 '%s' 생성 완료 (Dense COSINE + BM25 hybrid)", collection_name)
