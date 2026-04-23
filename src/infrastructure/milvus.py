"""
Milvus Vector Database Adapter
사용법:
    from adapters.milvus import make_vector_search_fn
    from pymilvus import MilvusClient
    from sentence_transformers import SentenceTransformer

    client   = MilvusClient(uri="http://localhost:19530")
    embedder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    vector_fn = make_vector_search_fn(client, embedding_fn=embedder.encode)

컬렉션 스키마 (create_collection.py로 생성):
    id:        VARCHAR(64)  - 노드 ID (primary key)
    node_type: VARCHAR(32)  - Project / Researcher / Organization / Paper / Patent / Report
    year:      INT64        - 연도 (필터용)
    vector:    FLOAT_VECTOR(768) - KR-SBERT 임베딩
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
    default_score_threshold: float = 0.0,
) -> Callable:
    """
    Milvus 클라이언트로 vector_search_fn 콜백 생성.
    인터페이스: (concept, node_type, filters, top_k, score_threshold=None) → List[str]

    Args:
        client:                  pymilvus MilvusClient 인스턴스
        embedding_fn:            텍스트 리스트 → numpy 배열 변환 함수
        collection_name:         Milvus 컬렉션명
        default_score_threshold: COSINE 유사도 기본 임계값 (요청별 오버라이드 가능).
                                 0.3 권장 — 낮은 관련성 결과 차단.
    """
    _score_threshold = default_score_threshold
    def vector_search(
        concept: str,
        node_type: str,
        filters: Dict[str, Any],
        top_k: int = 20,
        score_threshold: Optional[float] = None,
    ) -> List[str]:
        t_embed = time.perf_counter()
        vector = embedding_fn([concept])[0].tolist()
        embed_ms = (time.perf_counter() - t_embed) * 1000

        filter_parts: List[str] = []

        if node_type:
            filter_parts.append(f'node_type == "{node_type}"')

        # id_in 필터 (final_filter에서 사용)
        id_in: Optional[List[str]] = filters.get("id_in")
        if id_in:
            ids_expr = ", ".join(f'"{i}"' for i in id_in)
            filter_parts.append(f"id in [{ids_expr}]")

        # 연도 필터
        year_filter = filters.get("year")
        if isinstance(year_filter, dict):
            if "lt" in year_filter:
                filter_parts.append(f"year < {year_filter['lt']}")
            if "lte" in year_filter:
                filter_parts.append(f"year <= {year_filter['lte']}")
            if "gt" in year_filter:
                filter_parts.append(f"year > {year_filter['gt']}")
            if "gte" in year_filter:
                filter_parts.append(f"year >= {year_filter['gte']}")

        expr = " && ".join(filter_parts) if filter_parts else ""

        t_search = time.perf_counter()
        results = client.search(
            collection_name=collection_name,
            data=[vector],
            limit=top_k,
            filter=expr or None,
            output_fields=["id"],
            search_params={"metric_type": "COSINE", "params": {}},
        )
        search_ms = (time.perf_counter() - t_search) * 1000

        if not results:
            logger.debug(
                "[Milvus] vector_search '%s' (%s): embed=%.1f ms  search=%.1f ms  hits=0",
                concept, node_type or "*", embed_ms, search_ms,
            )
            return []

        hits = results[0]
        # per-request 오버라이드 우선, 없으면 생성 시 설정된 기본값 사용
        effective_threshold = score_threshold if score_threshold is not None else _score_threshold
        if effective_threshold > 0.0:
            # COSINE: distance 값이 높을수록 유사 (범위: -1 ~ 1)
            hits = [h for h in hits if h.get("distance", 0.0) >= effective_threshold]

        logger.debug(
            "[Milvus] vector_search '%s' (%s): embed=%.1f ms  search=%.1f ms  hits=%d",
            concept, node_type or "*", embed_ms, search_ms, len(hits),
        )
        return [hit["entity"]["id"] for hit in hits]

    return vector_search


def ensure_collection(client, collection_name: str = COLLECTION_NAME) -> None:
    """
    Milvus 컬렉션이 없으면 생성.
    시드 스크립트(scripts/seed_data.py)에서 호출.
    """
    from pymilvus import DataType

    if client.has_collection(collection_name):
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id",        DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field("node_type", DataType.VARCHAR, max_length=32)
    schema.add_field("year",      DataType.INT64)
    schema.add_field("vector",    DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

    index_params = client.prepare_index_params()
    index_params.add_index("vector", metric_type="COSINE", index_type="HNSW",
                            params={"M": 16, "efConstruction": 200})

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
