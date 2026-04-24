"""
데이터 시드 스크립트 — Neo4j + Milvus에 샘플 데이터 적재
사용법:
    python scripts/seed_data.py
    python scripts/seed_data.py --neo4j-only
    python scripts/seed_data.py --milvus-only
    python scripts/seed_data.py --clear   # 기존 데이터 삭제 후 재적재
"""

from __future__ import annotations
import argparse
import logging
import sys
import os

# 프로젝트 루트 내 src 디렉토리를 PYTHONPATH에 추가
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_dir, "src"))

from common.settings import get_settings
from common.fixtures import SEED_NODES, SEED_RELATIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Neo4j 시드
# ────────────────────────────────────────────────────────────────────────────

def seed_neo4j(driver, clear: bool = False) -> None:
    logger.info("Neo4j 시드 시작 (clear=%s)", clear)
    with driver.session() as session:
        if clear:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("기존 데이터 삭제 완료")

        # 유니크 제약 생성 (id 필드 기준)
        for label in ("Project", "Researcher", "Organization", "Paper", "Patent", "Report"):
            session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
            )

        # 노드 생성
        for node_id, props in SEED_NODES.items():
            label = props["type"]
            node_props = {k: v for k, v in props.items() if k != "type"}
            session.run(
                f"MERGE (n:{label} {{id: $id}}) SET n += $props",
                id=node_id, props=node_props,
            )
        logger.info("%d개 노드 생성 완료", len(SEED_NODES))

        # 관계 생성
        rel_count = 0
        for rel_type, rels in SEED_RELATIONS.items():
            for rel in rels:
                session.run(
                    f"MATCH (a {{id: $from_id}}), (b {{id: $to_id}}) "
                    f"MERGE (a)-[:{rel_type}]->(b)",
                    from_id=rel["from"], to_id=rel["to"],
                )
                rel_count += 1
        logger.info("%d개 관계 생성 완료", rel_count)

    logger.info("Neo4j 시드 완료")


# ────────────────────────────────────────────────────────────────────────────
# Milvus 시드
# ────────────────────────────────────────────────────────────────────────────

def seed_milvus(client, embedding_fn, clear: bool = False) -> None:
    from infrastructure.milvus import ensure_collection, COLLECTION_NAME

    logger.info("Milvus 시드 시작 (clear=%s)", clear)

    if clear and client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        logger.info("기존 컬렉션 삭제 완료")

    ensure_collection(client)

    # 임베딩 생성
    # name을 앞에 배치: 짧은 쿼리와의 유사도 향상 (bi-encoder 특성상 title이 검색 신호에 더 강함)
    ids, texts, node_types, years = [], [], [], []
    for node_id, props in SEED_NODES.items():
        name = props.get("name", "")
        body = (props.get("text") or props.get("abstract")
                or props.get("summary") or "")
        extra = " ".join(str(props.get(k, "")) for k in
                         ("topic", "expertise", "keywords") if props.get(k))
        full_text = " ".join(filter(None, [name, body, extra]))

        ids.append(node_id)
        texts.append(full_text)
        node_types.append(props["type"])
        years.append(props.get("year") or 0)

    logger.info("임베딩 생성 중 (%d개)...", len(texts))
    vectors = embedding_fn(texts)

    # text 필드: BM25 전문검색용 원본 텍스트 (sparse 벡터는 Milvus가 자동 생성)
    data = [
        {"id": nid, "node_type": nt, "year": yr, "text": txt, "dense": vec.tolist()}
        for nid, nt, yr, txt, vec in zip(ids, node_types, years, texts, vectors)
    ]
    client.insert(collection_name=COLLECTION_NAME, data=data)
    client.flush(collection_name=COLLECTION_NAME)

    logger.info("Milvus 시드 완료 (%d개 벡터)", len(data))


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="샘플 데이터를 Neo4j/Milvus에 적재")
    parser.add_argument("--neo4j-only", action="store_true")
    parser.add_argument("--milvus-only", action="store_true")
    parser.add_argument("--clear", action="store_true", help="기존 데이터 삭제 후 재적재")
    args = parser.parse_args()

    settings = get_settings()
    do_neo4j  = not args.milvus_only
    do_milvus = not args.neo4j_only

    if do_neo4j:
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password),
            )
            driver.verify_connectivity()
            seed_neo4j(driver, clear=args.clear)
            driver.close()
        except Exception as e:
            logger.error("Neo4j 시드 실패: %s", e)
            if not do_milvus:
                sys.exit(1)

    if do_milvus:
        try:
            from pymilvus import MilvusClient
            from sentence_transformers import SentenceTransformer

            client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
            embedder = SentenceTransformer(settings.sentence_transformer_model)
            seed_milvus(client, embedder.encode, clear=args.clear)
        except Exception as e:
            logger.error("Milvus 시드 실패: %s", e)
            sys.exit(1)


if __name__ == "__main__":
    main()
