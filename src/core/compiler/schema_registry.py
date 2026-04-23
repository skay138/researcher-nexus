"""
Schema Registry
- Neo4j에서 실시간 스키마를 읽어 LLM에 동적 주입
- 스키마 드리프트(silent failure) 방지
- 추상 개념(relation_concept) → 실제 relation type 매핑
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class NodeTypeInfo:
    label: str
    properties: List[str]
    sample_count: int = 0


@dataclass
class RelationInfo:
    type: str                 # 실제 DB relation type (e.g. "PARTICIPATED_IN")
    from_label: str
    to_label: str
    concepts: List[str]       # LLM이 사용할 추상 개념 (e.g. ["participation", "member"])


class SchemaRegistry:
    """
    Neo4j 스키마를 읽어 LLM 프롬프트용 텍스트로 변환.
    개념 → 실제 relation 매핑 테이블을 관리.
    """

    # 추상 개념 → 실제 relation type 매핑
    # 실제 환경에서는 DB나 설정 파일에서 로드
    CONCEPT_MAPPING: Dict[str, str] = {
        # Project ↔ Researcher
        "participation":    "PARTICIPATED_IN",
        "member":           "PARTICIPATED_IN",
        # Researcher ↔ Organization
        "belongs_to":       "AFFILIATED_WITH",
        "affiliation":      "AFFILIATED_WITH",
        "affiliated_with":  "AFFILIATED_WITH",
        # Researcher ↔ Paper
        "authored":         "AUTHORED",
        "wrote":            "AUTHORED",
        # Paper ↔ Paper
        "cites":            "CITES",
        "referenced":       "CITES",
        # Researcher → Project
        "manages":          "MANAGES",
        # Researcher ↔ Researcher
        "supervised":       "SUPERVISED_BY",
        # ── Patent (신규) ───────────────────────────────────────────────
        "invented":         "INVENTED",
        "invention":        "INVENTED",
        "filed":            "FILED",
        "patent_filing":    "FILED",
        # Project → Patent / Report
        "produced":         "PRODUCED",
        "produced_output":  "PRODUCED",
        # Researcher → Report
        "authored_report":  "AUTHORED_REPORT",
        "report_author":    "AUTHORED_REPORT",
        # Organization → Report
        "published":        "PUBLISHED",
        "publication":      "PUBLISHED",
        # Project → Report
        "published_in":     "PUBLISHED_IN",
        # Paper → Patent
        "cited_in":         "CITED_IN",
        "cited_in_patent":  "CITED_IN",
    }

    def __init__(self, driver=None):
        """
        Args:
            driver: Neo4j driver 인스턴스. None이면 mock 스키마 사용.
        """
        self.driver = driver
        self._cache: Optional[str] = None
        self._cache_ts: float = 0
        self._cache_ttl: float = 300.0  # 5분 캐시

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def resolve_relation(self, concept: str) -> str:
        """
        추상 개념 → 실제 Neo4j relation type 변환.

        Args:
            concept: LLM이 출력한 추상 개념 (e.g. "participation")

        Returns:
            실제 relation type (e.g. "PARTICIPATED_IN")

        Raises:
            ValueError: 매핑 테이블에 없는 개념
        """
        normalized = concept.lower().replace(" ", "_").replace("-", "_")
        if normalized in self.CONCEPT_MAPPING:
            return self.CONCEPT_MAPPING[normalized]

        # 부분 일치 시도 (어근 포함, SequenceMatcher 기반)
        import difflib
        best_key, best_val, best_ratio = None, None, 0.0
        for key, val in self.CONCEPT_MAPPING.items():
            # 정확한 포함 관계
            if key in normalized or normalized in key or key.startswith(normalized) or normalized.startswith(key):
                logger.warning(
                    "Fuzzy match for concept '%s' → '%s' (via key '%s')",
                    concept, val, key,
                )
                return val
            # 유사도 기반 (어근이 비슷한 경우)
            ratio = difflib.SequenceMatcher(None, normalized, key).ratio()
            if ratio > best_ratio:
                best_ratio, best_key, best_val = ratio, key, val

        if best_ratio >= 0.75:
            logger.warning(
                "Fuzzy match (ratio=%.2f) for concept '%s' → '%s' (via key '%s')",
                best_ratio, concept, best_val, best_key,
            )
            return best_val

        raise ValueError(
            f"Unknown relation concept: '{concept}'. "
            f"Valid concepts: {list(self.CONCEPT_MAPPING.keys())}"
        )

    def get_schema_for_llm(self) -> str:
        """
        LLM 시스템 프롬프트에 주입할 스키마 텍스트 반환.
        캐시 TTL이 지나면 Neo4j에서 재조회.
        """
        now = time.time()
        if self._cache and (now - self._cache_ts) < self._cache_ttl:
            return self._cache

        schema = self._load_schema()
        self._cache = schema
        self._cache_ts = now
        return schema

    def invalidate_cache(self) -> None:
        self._cache = None
        self._cache_ts = 0

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _load_schema(self) -> str:
        if self.driver is None:
            return self._mock_schema()

        try:
            return self._query_neo4j_schema()
        except Exception as e:
            logger.warning("Neo4j schema query failed (%s), using mock schema", e)
            return self._mock_schema()

    def _query_neo4j_schema(self) -> str:
        """실제 Neo4j에서 스키마 조회"""
        with self.driver.session() as session:
            # 노드 타입과 속성 조회
            node_result = session.run("""
                CALL db.schema.nodeTypeProperties()
                YIELD nodeType, propertyName
                RETURN nodeType, collect(propertyName) AS properties
            """)
            nodes = {r["nodeType"]: r["properties"] for r in node_result}

            # relation 타입 조회
            rel_result = session.run("""
                CALL db.schema.relTypeProperties()
                YIELD relType
                RETURN DISTINCT relType
            """)
            relations = [r["relType"] for r in rel_result]

        return self._format_schema(nodes, relations)

    def _format_schema(
        self,
        nodes: Dict[str, List[str]],
        relations: List[str],
    ) -> str:
        node_lines = "\n".join(
            f"  - {label}: {', '.join(props[:5])}"
            for label, props in nodes.items()
        )
        rel_lines = "\n".join(f"  - {r}" for r in relations)
        concept_lines = "\n".join(
            f"  '{concept}' → {rtype}"
            for concept, rtype in self.CONCEPT_MAPPING.items()
        )

        return f"""## 데이터베이스 스키마 (실시간 조회)

### 노드 타입
{node_lines}

### Relation 타입
{rel_lines}

### 추상 개념 → Relation 매핑 (이 개념만 사용하세요)
{concept_lines}
"""

    def _mock_schema(self) -> str:
        """Neo4j 없을 때 사용하는 Mock 스키마"""
        return self._format_schema(
            nodes={
                "Project":      ["id", "name", "year", "topic", "budget"],
                "Researcher":   ["id", "name", "email", "expertise"],
                "Organization": ["id", "name", "type", "country"],
                "Paper":        ["id", "name", "abstract", "year", "keywords"],
                "Patent":       ["id", "name", "patent_number", "year", "filing_date", "abstract"],
                "Report":       ["id", "name", "report_type", "year", "summary"],
            },
            relations=[
                "PARTICIPATED_IN", "AFFILIATED_WITH", "AUTHORED",
                "CITES", "MANAGES", "SUPERVISED_BY",
                "INVENTED", "FILED", "PRODUCED",
                "AUTHORED_REPORT", "PUBLISHED", "PUBLISHED_IN", "CITED_IN",
            ],
        )
