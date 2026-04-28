"""
Tool Result Utilities
execute_dynamic_search / get_details_by_ids가 반환하는
{"total", "path", "results"} JSON 포맷을 파싱·머지·포맷하는 유틸리티.
"""

from __future__ import annotations
from typing import Any, Dict, List
import json


def merge_tool_results(
    tool_results: List[str],
) -> tuple[List[str], Dict[str, Dict[str, Any]]]:
    """
    여러 도구 결과(Milvus + MariaDB)를 ID 기준으로 머지.

    - path/score: 최초 등장(Milvus) 기준 유지
    - authors/year/text/expertise/topic: 나중 값(MariaDB)으로 보강

    Returns:
        order:  삽입 순서 ID 목록
        merged: ID → 머지된 필드 dict
    """
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for raw in tool_results:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue

        global_path = data.get("path", "")

        for item in data.get("results", []):
            item_id = item.get("id")
            if not item_id:
                continue

            if item_id not in merged:
                merged[item_id] = {
                    "id":        item_id,
                    "type":      item.get("type", "Unknown"),
                    "name":      item.get("name", ""),
                    "path":      item.get("path") or global_path,
                    "score":     item.get("score"),
                    "year":      item.get("year", ""),
                    "text":      item.get("text", ""),
                    "authors":   item.get("authors") or [],
                    "expertise": item.get("expertise", ""),
                    "topic":     item.get("topic", ""),
                }
                order.append(item_id)
            else:
                for key in ("authors", "year", "text", "expertise", "topic"):
                    val = item.get(key)
                    if val is not None and val != "" and val != []:
                        merged[item_id][key] = val

    return order, merged


def extract_sources_from_tool_results(tool_results: List[str]) -> List[dict]:
    """ToolMessage 목록에서 출처 정보(API 응답용 list[dict])를 추출."""
    order, merged = merge_tool_results(tool_results)
    return [{"no": i, **merged[item_id]} for i, item_id in enumerate(order, 1)]


def numbered_search_context(tool_results: List[str]) -> str:
    """ToolMessage JSON을 번호 붙은 텍스트 블록으로 변환 (LLM context용)."""
    order, merged = merge_tool_results(tool_results)

    lines: List[str] = []
    for counter, item_id in enumerate(order, 1):
        item    = merged[item_id]
        itype   = item["type"]
        authors = item["authors"]

        lines.append(f"[{counter}] ({itype}) {item['name']}")
        lines.append(f"  id: {item_id}")
        if authors:
            lines.append(f"  저자: {', '.join(authors)}")
        elif itype in ("Paper", "Patent", "Report"):
            lines.append("  저자: 저자 정보 없음")
        if item["year"]:
            lines.append(f"  연도: {item['year']}")
        if itype == "Researcher":
            lines.append(f"  전문분야: {item['expertise'] or '정보 없음'}")
            if item["topic"]:
                lines.append(f"  주제: {item['topic']}")
        if item["text"]:
            lines.append(f"  요약: {item['text']}")
        lines.append("")

    return "\n".join(lines) if lines else "(검색 결과 없음)"
