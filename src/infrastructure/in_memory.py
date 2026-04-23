"""
In-Memory DB Adapter (replaces adapters/mock.py)
- 실제 Neo4j / Milvus 없이 전체 파이프라인 동작 가능
- 데이터는 외부(shared/fixtures.py 등)에서 주입 — 하드코딩 없음
- make_in_memory_adapters(nodes, relations) → (vector_fn, graph_fn, details_fn, texts_fn)
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple

from core.executor.execution_engine import NodeResult


def make_in_memory_adapters(
    nodes: Dict[str, Dict],
    relations: Dict[str, List[Dict]],
    keyword_threshold: float = 1.0,
) -> Tuple:
    """
    인메모리 어댑터 4종을 생성하여 튜플로 반환.

    Returns:
        (vector_search_fn, graph_query_fn, fetch_details_fn, fetch_texts_fn)
    """
    vector_fn   = _make_vector_search(nodes, keyword_threshold)
    graph_fn    = _make_graph_query(nodes, relations)
    details_fn  = _make_fetch_details(nodes, relations)
    texts_fn    = _make_fetch_texts(nodes)
    return vector_fn, graph_fn, details_fn, texts_fn


# ────────────────────────────────────────────────────────────────────────────
# Internal factories
# ────────────────────────────────────────────────────────────────────────────

def _make_vector_search(nodes: Dict[str, Dict], keyword_threshold: float):
    def vector_search(
        concept: str,
        node_type: str,
        filters: Dict[str, Any],
        top_k: int = 20,
        score_threshold: float = None,  # 키워드 매칭에는 미적용
    ) -> List[str]:
        concept_lower = concept.lower()
        keywords = [kw for kw in concept_lower.split() if kw]

        scored: List[tuple] = []
        for nid, node in nodes.items():
            if node_type and node["type"] != node_type:
                continue

            id_in = filters.get("id_in")
            if id_in and nid not in id_in:
                continue

            year = node.get("year")
            if year:
                year_filter = filters.get("year", {})
                if isinstance(year_filter, dict):
                    if "lt" in year_filter and year >= year_filter["lt"]:
                        continue
                    if "gt" in year_filter and year <= year_filter["gt"]:
                        continue

            searchable = " ".join(str(v) for v in node.values()).lower()

            if not keywords:
                scored.append((1.0, nid))
                continue

            match_count = sum(1 for kw in keywords if kw in searchable)
            score = match_count / len(keywords)
            if score >= keyword_threshold:
                scored.append((score, nid))

        scored.sort(key=lambda x: -x[0])
        return [nid for _, nid in scored[:top_k]]

    return vector_search


def _make_graph_query(nodes: Dict[str, Dict], relations: Dict[str, List[Dict]]):
    def graph_query(cypher: str) -> List[Dict]:
        start_ids_match = re.search(r"WHERE n0\.id IN \[([^\]]+)\]", cypher)
        if not start_ids_match:
            return []
        start_ids = [s.strip().strip("'").strip('"') for s in start_ids_match.group(1).split(',')]

        exclude_ids = []
        exclude_match = re.search(r"NOT n1\.id IN \[([^\]]+)\]", cypher)
        if exclude_match:
            exclude_ids = [s.strip().strip("'").strip('"') for s in exclude_match.group(1).split(',')]
        exclude_set = set(exclude_ids)

        limit_match = re.search(r"LIMIT (\d+)", cypher)
        limit = int(limit_match.group(1)) if limit_match else 100

        rel_match = re.search(r'\[:(\w+)\]', cypher)
        if not rel_match:
            return []
        rel_type = rel_match.group(1)
        inbound  = "<-[" in cypher
        outbound = "]->" in cypher
        both     = not inbound and not outbound

        rels = relations.get(rel_type, [])
        results = []

        for rel in rels:
            target_node = None
            s_id = None

            if inbound or both:
                if rel["to"] in start_ids:
                    target_node = nodes.get(rel["from"])
                    s_id = rel["to"]
            if (outbound or both) and not target_node:
                if rel["from"] in start_ids:
                    target_node = nodes.get(rel["to"])
                    s_id = rel["from"]

            if target_node:
                tid = target_node["id"]
                if tid in exclude_set:
                    continue
                results.append({
                    "id":       tid,
                    "type":     target_node["type"],
                    "name":     target_node.get("name"),
                    "start_id": s_id,
                })

        seen: set = set()
        deduped = []
        for r in results:
            if r["id"] not in seen:
                seen.add(r["id"])
                deduped.append(r)

        return deduped[:limit]

    return graph_query


def _make_fetch_details(nodes: Dict[str, Dict], relations: Dict[str, List[Dict]]):
    # 논문 ID → 저자 이름 역방향 인덱스
    paper_authors: Dict[str, List[str]] = {}
    for rel in relations.get("AUTHORED", []):
        researcher = nodes.get(rel["from"])
        paper_id = rel["to"]
        if researcher and paper_id:
            paper_authors.setdefault(paper_id, []).append(researcher["name"])

    def fetch_details(ids: List[str]) -> List[NodeResult]:
        results = []
        for nid in ids:
            node = nodes.get(nid)
            if node:
                text = node.get("text") or node.get("abstract") or node.get("summary")
                exclude = {"id", "type", "name", "text", "abstract", "summary"}
                meta = {k: v for k, v in node.items() if k not in exclude}
                if node["type"] == "Paper" and nid in paper_authors:
                    meta["authors"] = paper_authors[nid]
                results.append(NodeResult(
                    id=node["id"],
                    type=node["type"],
                    name=node.get("name"),
                    text=text,
                    meta=meta,
                ))
        return results

    return fetch_details


def _make_fetch_texts(nodes: Dict[str, Dict]):
    _EXTRA = {"topic", "expertise", "keywords", "patent_number",
              "report_type", "filing_date", "inventors"}

    def fetch_texts(ids: List[str]) -> List[str]:
        texts = []
        for nid in ids:
            node = nodes.get(nid, {})
            text = (node.get("text") or node.get("abstract")
                    or node.get("summary") or node.get("name", "") or nid)
            extra = " ".join(str(v) for k, v in node.items() if k in _EXTRA and v)
            texts.append(f"{text} {extra}".strip())
        return texts

    return fetch_texts
