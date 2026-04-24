"""
Neo4j Graph Database Adapter
사용법:
    from infrastructure.neo4j import make_graph_query_fn, make_fetch_details_fn
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))
    graph_fn   = make_graph_query_fn(driver)
    details_fn = make_fetch_details_fn(driver)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging
import time

from core.executor.execution_engine import NodeResult

logger = logging.getLogger(__name__)


def make_graph_query_fn(driver):
    """
    Neo4j 드라이버로 graph_query_fn 콜백 생성.
    인터페이스: (cypher: str) → List[dict]
    """
    def graph_query(cypher: str) -> List[Dict[str, Any]]:
        t0 = time.perf_counter()
        with driver.session() as session:
            result = session.run(cypher)
            rows = [dict(r.data()) for r in result]
        logger.debug("[Neo4j] graph_query: %d rows  %.1f ms", len(rows), (time.perf_counter() - t0) * 1000)
        return rows
    return graph_query


def make_fetch_details_fn(driver):
    """
    Neo4j 드라이버로 fetch_details_fn 콜백 생성.
    인터페이스: (ids: List[str]) → List[NodeResult]

    노드 속성:
    - id, name: 필수
    - text / abstract / summary: 요약 텍스트 (우선순위 순)
    - authors: Paper 노드의 경우 AUTHORED 역방향 조회
    """
    def fetch_details(ids: List[str]) -> List[NodeResult]:
        if not ids:
            return []
        t0 = time.perf_counter()
        with driver.session() as session:
            # 노드 기본 정보
            node_result = session.run(
                "MATCH (n) WHERE n.id IN $ids RETURN n",
                ids=ids,
            )
            node_map: Dict[str, NodeResult] = {}
            for record in node_result:
                node = record["n"]
                props = dict(node.items())
                nid = props.get("id", "")
                node_type = list(node.labels)[0] if node.labels else "Unknown"
                text = props.get("text") or props.get("abstract") or props.get("summary")
                exclude = {"id", "type", "name", "text", "abstract", "summary"}
                meta = {k: v for k, v in props.items() if k not in exclude}
                node_map[nid] = NodeResult(
                    id=nid, type=node_type,
                    name=props.get("name"),
                    text=text, meta=meta,
                )

            # Paper → 저자 이름 일괄 조회
            paper_ids = [nid for nid, nr in node_map.items() if nr.type == "Paper"]
            if paper_ids:
                author_result = session.run(
                    "MATCH (r:Researcher)-[:AUTHORED]->(p:Paper) "
                    "WHERE p.id IN $ids RETURN p.id AS paper_id, r.name AS author_name",
                    ids=paper_ids,
                )
                for ar in author_result:
                    pid = ar["paper_id"]
                    if pid in node_map:
                        node_map[pid].meta.setdefault("authors", []).append(ar["author_name"])

        logger.debug("[Neo4j] fetch_details: %d nodes  %.1f ms", len(node_map), (time.perf_counter() - t0) * 1000)
        return [node_map[nid] for nid in ids if nid in node_map]

    return fetch_details


