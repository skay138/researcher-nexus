"""
Query Plan 모델
- LLM이 출력하는 추상 플랜 (구체 쿼리 없음)
- Pydantic으로 스키마 강제 → hallucination 감지
"""

from __future__ import annotations
from typing import Any, ClassVar, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator


class HopDirection(str, Enum):
    INBOUND  = "in"    # <-[REL]-
    OUTBOUND = "out"   # -[REL]->
    BOTH     = "both"  # -[REL]-


class HopSpec(BaseModel):
    """단일 hop 명세. LLM은 추상 개념만 작성, 컴파일러가 Cypher로 변환."""

    from_type:        str           # 출발 노드 타입  (e.g. "Project")
    relation_concept: str           # 추상 관계 개념  (e.g. "participation")
    to_type:          str           # 도착 노드 타입  (e.g. "Researcher")
    direction:        HopDirection  = HopDirection.OUTBOUND
    filters:          Dict[str, Any] = Field(default_factory=dict)

    @field_validator("relation_concept")
    @classmethod
    def lowercase_concept(cls, v: str) -> str:
        return v.lower().replace(" ", "_")


class EntrySearch(BaseModel):
    """Vector DB 진입 검색 명세"""

    concept:  str                    # 의미 검색 쿼리
    node_type: str                   # 검색 대상 노드 타입
    filters:  Dict[str, Any] = Field(default_factory=dict)
    top_k:    int = Field(default=20, ge=1, le=200)


class FinalFilter(BaseModel):
    """최종 결과에 적용할 추가 필터"""

    concept:   Optional[str]  = None  # 의미 필터 (Vector 재검색)
    node_type: Optional[str]  = None
    filters:   Dict[str, Any] = Field(default_factory=dict)


class QueryPlan(BaseModel):
    """
    LLM의 전체 출력 스키마.
    - 구체적인 Cypher 쿼리를 포함하지 않음
    - 컴파일러가 이 플랜을 받아 실행 가능한 쿼리로 변환
    """

    entry_search:     EntrySearch
    traversal_hops:   List[HopSpec]   = Field(default_factory=list)
    final_filter:     Optional[FinalFilter] = None
    max_results:      int = Field(default=20, ge=1, le=500)
    reasoning:        Optional[str]   = None  # LLM의 추론 과정 (디버깅용)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    MAX_HOPS: ClassVar[int] = 5

    @field_validator("traversal_hops")
    @classmethod
    def limit_hops(cls, hops: List[HopSpec]) -> List[HopSpec]:
        if len(hops) > cls.MAX_HOPS:
            raise ValueError(
                f"traversal_hops는 최대 {cls.MAX_HOPS}개입니다. "
                f"요청된 hop 수: {len(hops)}"
            )
        return hops

    @model_validator(mode="after")
    def validate_hop_types(self) -> "QueryPlan":
        """hop chain의 타입 연속성 검증"""
        if not self.traversal_hops:
            return self

        prev_to = self.traversal_hops[0].from_type
        for i, hop in enumerate(self.traversal_hops):
            if hop.from_type != prev_to and i > 0:
                raise ValueError(
                    f"Hop {i}: from_type='{hop.from_type}'이 "
                    f"이전 hop의 to_type='{prev_to}'와 일치하지 않습니다."
                )
            prev_to = hop.to_type
        return self

    def describe(self) -> str:
        """디버깅용 플랜 요약"""
        if not self.traversal_hops:
            hops_str = "none"
        else:
            hops_str = f"({self.traversal_hops[0].from_type})"
            for h in self.traversal_hops:
                if h.direction == HopDirection.INBOUND:
                    hops_str += f" <-[{h.relation_concept}]- ({h.to_type})"
                elif h.direction == HopDirection.OUTBOUND:
                    hops_str += f" -[{h.relation_concept}]-> ({h.to_type})"
                else:
                    hops_str += f" -[{h.relation_concept}]- ({h.to_type})"

        return (
            f"Entry: {self.entry_search.concept!r} ({self.entry_search.node_type}) filters={self.entry_search.filters}\n"
            f"Hops ({len(self.traversal_hops)}): {hops_str}\n"
            f"Final filter: {self.final_filter}\n"
            f"Max results: {self.max_results}"
        )
