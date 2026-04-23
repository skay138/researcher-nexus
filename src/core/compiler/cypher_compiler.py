"""
Cypher Compiler
- QueryPlan → 실행 가능한 Cypher 쿼리 (결정론적, LLM 개입 없음)
- hop별 LIMIT 강제 (hop explosion 1차 방어선)
- 스키마 검증 포함
"""

from __future__ import annotations
from typing import Optional, List
import logging
import re

from common.query_plan import HopDirection, HopSpec, QueryPlan
from core.compiler.schema_registry import SchemaRegistry
from common.exceptions import CypherInjectionDetected, UnknownRelationConcept

logger = logging.getLogger(__name__)


class HopLimitExceeded(Exception):
    pass


# ID에 허용되는 문자: 영문자, 숫자, 하이픈, 언더스코어
_SAFE_ID_PATTERN = re.compile(r'^[\w\-]+$')
# 필터 키에 허용되는 문자: 영문자, 숫자, 언더스코어 (속성명)
_SAFE_KEY_PATTERN = re.compile(r'^\w+$')
# Cypher 비교 연산자 화이트리스트
_CYPHER_OPS = {"lt": "<", "lte": "<=", "gt": ">", "gte": ">="}


class CypherCompiler:
    """
    QueryPlan을 Neo4j Cypher 쿼리로 컴파일.

    설계 원칙:
    - 완전히 결정론적: 동일 입력 → 동일 쿼리
    - LLM 출력(relation_concept)을 SchemaRegistry를 통해 검증 후 변환
    - hop별 LIMIT으로 중간 폭발 방지
    """

    # hop별 중간 결과 최대 노드 수
    HOP_FANOUT_LIMIT: int = 500

    def __init__(self, schema_registry: SchemaRegistry):
        self.schema_registry = schema_registry

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def compile_traversal(
        self,
        plan: QueryPlan,
        entry_ids: List[str],
    ) -> str:
        """
        traversal_hops 전체를 단일 Cypher로 컴파일.

        Args:
            plan:       LLM이 생성한 QueryPlan
            entry_ids:  Vector DB 검색으로 얻은 진입 노드 ID 목록

        Returns:
            실행 가능한 Cypher 쿼리 문자열
        """
        if not plan.traversal_hops:
            # 탐색 없이 진입 노드만 반환
            return (
                f"MATCH (n) WHERE n.id IN {self._ids_literal(entry_ids)}\n"
                f"RETURN n.id AS id, labels(n)[0] AS type LIMIT {plan.max_results}"
            )

        lines: List[str] = []

        # 진입 노드 바인딩
        lines.append(
            f"MATCH (n0)\n"
            f"WHERE n0.id IN {self._ids_literal(entry_ids)}"
        )

        for i, hop in enumerate(plan.traversal_hops):
            rel_type = self._resolve_relation(hop)
            arrow_left, arrow_right = self._direction_arrows(hop.direction)

            # WHERE 절 생성 (필터가 있을 때)
            where_clause = self._build_where(f"n{i+1}", hop.filters)

            lines.append(
                f"WITH n{i}\n"
                f"MATCH (n{i}){arrow_left}[:{rel_type}]{arrow_right}(n{i+1}:{hop.to_type})\n"
                f"{where_clause}"
                f"WITH n{i+1} LIMIT {self.HOP_FANOUT_LIMIT}"
            )

        last = len(plan.traversal_hops)
        lines.append(
            f"RETURN DISTINCT n{last}.id AS id, "
            f"labels(n{last})[0] AS type, "
            f"n{last}.name AS name\n"
            f"LIMIT {plan.max_results}"
        )

        query = "\n".join(lines)
        logger.debug("Compiled Cypher:\n%s", query)
        return query

    def compile_single_hop(
        self,
        hop: HopSpec,
        start_ids: List[str],
        limit: int,
        exclude_ids: Optional[List[str]] = None,
    ) -> str:
        """
        단일 hop Cypher 컴파일 (hop-by-hop 실행 모드용).
        BeamPruner와 함께 사용할 때 더 세밀한 제어 가능.
        """
        rel_type = self._resolve_relation(hop)
        arrow_left, arrow_right = self._direction_arrows(hop.direction)
        
        # 필터 구성
        where_clauses = []
        # 1. 기본 필터
        if hop.filters:
            where_clauses.append(self._build_where("n1", hop.filters).replace("WHERE ", ""))
        # 2. 경로 히스토리 제외 (사이클 방지)
        if exclude_ids:
            where_clauses.append(f"NOT n1.id IN {self._ids_literal(exclude_ids)}")
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        return (
            f"MATCH (n0)\n"
            f"WHERE n0.id IN {self._ids_literal(start_ids)}\n"
            f"MATCH (n0){arrow_left}[:{rel_type}]{arrow_right}(n1:{hop.to_type})\n"
            f"{where_clause}\n"
            f"RETURN DISTINCT n1.id AS id, labels(n1)[0] AS type, n1.name AS name, n0.id AS start_id\n"
            f"LIMIT {limit}"
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _resolve_relation(self, hop: HopSpec) -> str:
        try:
            return self.schema_registry.resolve_relation(hop.relation_concept)
        except ValueError as e:
            raise UnknownRelationConcept(str(e)) from e

    @staticmethod
    def _direction_arrows(direction: HopDirection):
        """(left_arrow, right_arrow) 쌍 반환"""
        return {
            HopDirection.OUTBOUND: ("-",  "->"),
            HopDirection.INBOUND:  ("<-", "-"),
            HopDirection.BOTH:     ("-",  "-"),
        }[direction]

    @staticmethod
    def _ids_literal(ids: List[str]) -> str:
        """Python 리스트 → Cypher 리스트 리터럴 (인젝션 방어 포함).

        NOTE: 실제 Neo4j 사용 시에는 파라미터 바인딩 방식을 권장합니다.
              session.run("MATCH (n) WHERE n.id IN $ids", ids=list)
        """
        for node_id in ids:
            if not _SAFE_ID_PATTERN.match(node_id):
                raise CypherInjectionDetected(
                    f"Invalid node ID detected: '{node_id}'. "
                    "Only alphanumeric characters, hyphens, and underscores are allowed."
                )
        quoted = ", ".join(f"'{i}'" for i in ids)
        return f"[{quoted}]"

    @staticmethod
    def _build_where(var: str, filters: dict) -> str:
        if not filters:
            return ""

        conditions = []
        for key, val in filters.items():
            if not _SAFE_KEY_PATTERN.match(key):
                raise CypherInjectionDetected(
                    f"Invalid filter key: '{key}'. Only alphanumeric and underscore allowed."
                )
            if isinstance(val, bool):
                conditions.append(f"{var}.{key} = {str(val).lower()}")
            elif isinstance(val, str):
                escaped = val.replace("\\", "\\\\").replace("'", "\\'")
                conditions.append(f"{var}.{key} = '{escaped}'")
            elif isinstance(val, (int, float)):
                conditions.append(f"{var}.{key} = {val}")
            elif isinstance(val, dict):
                for op, v in val.items():
                    cypher_op = _CYPHER_OPS.get(op)
                    if not cypher_op:
                        raise CypherInjectionDetected(
                            f"Invalid filter operator: '{op}'. Allowed: {list(_CYPHER_OPS)}"
                        )
                    if isinstance(v, (int, float)):
                        conditions.append(f"{var}.{key} {cypher_op} {v}")
                    elif isinstance(v, str):
                        escaped = v.replace("\\", "\\\\").replace("'", "\\'")
                        conditions.append(f"{var}.{key} {cypher_op} '{escaped}'")
                    else:
                        raise CypherInjectionDetected(
                            f"Unsupported filter value type for key '{key}': {type(v)}"
                        )
            else:
                raise CypherInjectionDetected(
                    f"Unsupported filter value type for key '{key}': {type(val)}"
                )

        return f"WHERE {' AND '.join(conditions)}\n" if conditions else ""
