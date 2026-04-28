"""
MariaDB Adapter — 노드 원본 데이터 저장소 (source of truth)
- 도메인별 테이블 분리: organizations, researchers, projects, papers, patents, reports
- paper_authors: 논문-저자 역정규화 (JOIN 없이 authors 즉시 반환)
- id → type 조회는 UNION으로 처리 (별도 registry 테이블 불필요)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging
import urllib.parse

from common.types.results import NodeResult

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# DDL
# ────────────────────────────────────────────────────────────────────────────

_DDL_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS organizations (
        id VARCHAR(255) NOT NULL,
        name VARCHAR(512) NOT NULL,
        description TEXT,
        PRIMARY KEY (id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS researchers (
        id VARCHAR(255) NOT NULL,
        name VARCHAR(512) NOT NULL,
        expertise TEXT,
        topic VARCHAR(255),
        PRIMARY KEY (id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS projects (
        id VARCHAR(255) NOT NULL,
        name VARCHAR(512) NOT NULL,
        year INT,
        description TEXT,
        topic VARCHAR(255),
        PRIMARY KEY (id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS papers (
        id VARCHAR(255) NOT NULL,
        name VARCHAR(512) NOT NULL,
        year INT,
        abstract TEXT,
        keywords TEXT,
        PRIMARY KEY (id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS patents (
        id VARCHAR(255) NOT NULL,
        name VARCHAR(512) NOT NULL,
        year INT,
        abstract TEXT,
        keywords TEXT,
        patent_number VARCHAR(64),
        PRIMARY KEY (id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS reports (
        id VARCHAR(255) NOT NULL,
        name VARCHAR(512) NOT NULL,
        year INT,
        summary TEXT,
        report_type VARCHAR(64),
        PRIMARY KEY (id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS paper_authors (
        paper_id VARCHAR(255) NOT NULL,
        author_name VARCHAR(512) NOT NULL,
        display_order INT NOT NULL DEFAULT 0,
        PRIMARY KEY (paper_id, author_name),
        INDEX idx_paper_id (paper_id),
        FOREIGN KEY (paper_id) REFERENCES papers(id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS system_config (
        `key` VARCHAR(100) NOT NULL,
        `value` TEXT NOT NULL,
        PRIMARY KEY (`key`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
]

# node_type → (table_name, text_column)
_TYPE_TABLE: Dict[str, tuple] = {
    "Organization": ("organizations", "description"),
    "Researcher":   ("researchers",   None),
    "Project":      ("projects",      "description"),
    "Paper":        ("papers",        "abstract"),
    "Patent":       ("patents",       "abstract"),
    "Report":       ("reports",       "summary"),
}

# id → type를 한 번에 조회하는 UNION 쿼리
_ID_TYPE_UNION = " UNION ALL ".join(
    f"SELECT id, '{node_type}' AS node_type FROM {table}"
    for node_type, (table, _) in _TYPE_TABLE.items()
)


# ────────────────────────────────────────────────────────────────────────────
# URL 파싱
# ────────────────────────────────────────────────────────────────────────────

def _parse_url(mariadb_url: str) -> Dict[str, Any]:
    """mysql[+pymysql]://user:pass@host:port/db → pymysql.connect() kwargs"""
    url = mariadb_url.strip()
    url = url.replace("mysql+pymysql://", "mysql://").replace("mariadb+pymysql://", "mysql://")
    parsed = urllib.parse.urlparse(url)
    return {
        "host":     parsed.hostname or "localhost",
        "port":     parsed.port or 3306,
        "user":     urllib.parse.unquote(parsed.username or "root"),
        "password": urllib.parse.unquote(parsed.password or ""),
        "database": parsed.path.lstrip("/") or "nexus",
        "charset":  "utf8mb4",
    }


# ────────────────────────────────────────────────────────────────────────────
# Schema 초기화
# ────────────────────────────────────────────────────────────────────────────

def ensure_schema(mariadb_url: str) -> None:
    """DDL 실행 — 테이블이 없으면 생성 (idempotent)."""
    import pymysql
    params = _parse_url(mariadb_url)
    with pymysql.connect(**params) as conn:
        with conn.cursor() as cur:
            for stmt in _DDL_STATEMENTS:
                cur.execute(stmt)
        conn.commit()
    logger.info("MariaDB schema ensured")


# ────────────────────────────────────────────────────────────────────────────
# Fetch Details
# ────────────────────────────────────────────────────────────────────────────

def make_fetch_details_fn(mariadb_url: str):
    """
    MariaDB 기반 fetch_details_fn 생성.
    인터페이스: (ids: List[str]) → List[NodeResult]

    - 6개 도메인 테이블 UNION으로 id → type 조회 (registry 테이블 불필요)
    - 타입별 도메인 테이블에서 상세 정보 조회
    - Paper: paper_authors에서 저자 목록 추가
    """
    import pymysql
    import pymysql.cursors

    conn_params = _parse_url(mariadb_url)
    _conn: Optional[Any] = None

    def _get_conn():
        nonlocal _conn
        try:
            if _conn is not None and _conn.open:
                _conn.ping(reconnect=True)
                return _conn
        except Exception:
            pass
        _conn = pymysql.connect(**conn_params, cursorclass=pymysql.cursors.DictCursor)
        return _conn

    def fetch_details(ids: List[str]) -> List[NodeResult]:
        if not ids:
            return []

        conn = _get_conn()
        placeholders = ", ".join(["%s"] * len(ids))
        logger.debug("[fetch_details] queried=%s", ids)

        with conn.cursor() as cur:
            # 1) id → type: 모든 도메인 테이블 UNION (각 테이블 PK 인덱스 lookup)
            cur.execute(
                f"SELECT id, node_type FROM ({_ID_TYPE_UNION}) t WHERE id IN ({placeholders})",
                ids,
            )
            id_type_map: Dict[str, str] = {row["id"]: row["node_type"] for row in cur.fetchall()}

        # 2) 타입별 그룹핑
        type_ids: Dict[str, List[str]] = {}
        for nid in ids:
            node_type = id_type_map.get(nid)
            if node_type:
                type_ids.setdefault(node_type, []).append(nid)

        node_map: Dict[str, NodeResult] = {}

        with conn.cursor() as cur:
            for node_type, nids in type_ids.items():
                table, text_col = _TYPE_TABLE[node_type]
                ph = ", ".join(["%s"] * len(nids))
                cur.execute(f"SELECT * FROM {table} WHERE id IN ({ph})", nids)

                for row in cur.fetchall():
                    nid = row["id"]
                    text = row.get(text_col) if text_col else None
                    exclude = {"id", "name", text_col} if text_col else {"id", "name"}
                    meta = {k: v for k, v in row.items() if k not in exclude and v is not None}
                    node_map[nid] = NodeResult(
                        id=nid,
                        type=node_type,
                        name=row.get("name"),
                        text=text,
                        meta=meta,
                    )

            # 3) Paper 저자
            paper_ids = type_ids.get("Paper", [])
            if paper_ids:
                ph = ", ".join(["%s"] * len(paper_ids))
                cur.execute(
                    f"SELECT paper_id, author_name FROM paper_authors "
                    f"WHERE paper_id IN ({ph}) ORDER BY display_order",
                    paper_ids,
                )
                for row in cur.fetchall():
                    pid = row["paper_id"]
                    if pid in node_map:
                        node_map[pid].meta.setdefault("authors", []).append(row["author_name"])

        results = [node_map[nid] for nid in ids if nid in node_map]
        missing = [nid for nid in ids if nid not in node_map]
        logger.debug(
            "[fetch_details] found=%d | missing=%s | paper_authors=%d",
            len(results),
            missing or "none",
            sum(len(r.meta.get("authors", [])) for r in results),
        )
        return results

    return fetch_details


# ────────────────────────────────────────────────────────────────────────────
# Seed helpers (scripts/seed_data.py 에서 호출)
# ────────────────────────────────────────────────────────────────────────────

def seed_nodes(mariadb_url: str, nodes: Dict[str, Dict], relations: Dict[str, List[Dict]], clear: bool = False) -> None:
    """SEED_NODES / SEED_RELATIONS → MariaDB 삽입. clear=True이면 기존 데이터 삭제 후 재삽입."""
    import pymysql
    import pymysql.cursors

    params = _parse_url(mariadb_url)
    conn = pymysql.connect(**params, cursorclass=pymysql.cursors.DictCursor)

    try:
        with conn.cursor() as cur:
            if clear:
                for t in ["system_config", "paper_authors", "organizations", "researchers",
                           "projects", "papers", "patents", "reports"]:
                    cur.execute(f"DELETE FROM {t}")
                logger.info("MariaDB 기존 데이터 삭제 완료")

            for node_id, props in nodes.items():
                node_type = props["type"]
                if node_type == "Organization":
                    cur.execute(
                        "INSERT INTO organizations (id, name, description) VALUES (%s, %s, %s) "
                        "ON DUPLICATE KEY UPDATE name=VALUES(name), description=VALUES(description)",
                        (node_id, props.get("name", ""), props.get("text") or props.get("description")),
                    )
                elif node_type == "Researcher":
                    cur.execute(
                        "INSERT INTO researchers (id, name, expertise, topic) VALUES (%s, %s, %s, %s) "
                        "ON DUPLICATE KEY UPDATE name=VALUES(name), expertise=VALUES(expertise), topic=VALUES(topic)",
                        (node_id, props.get("name", ""), props.get("expertise"), props.get("topic")),
                    )
                elif node_type == "Project":
                    cur.execute(
                        "INSERT INTO projects (id, name, year, description, topic) VALUES (%s, %s, %s, %s, %s) "
                        "ON DUPLICATE KEY UPDATE name=VALUES(name), year=VALUES(year), "
                        "description=VALUES(description), topic=VALUES(topic)",
                        (node_id, props.get("name", ""), props.get("year"),
                         props.get("text") or props.get("description"), props.get("topic")),
                    )
                elif node_type == "Paper":
                    cur.execute(
                        "INSERT INTO papers (id, name, year, abstract, keywords) VALUES (%s, %s, %s, %s, %s) "
                        "ON DUPLICATE KEY UPDATE name=VALUES(name), year=VALUES(year), "
                        "abstract=VALUES(abstract), keywords=VALUES(keywords)",
                        (node_id, props.get("name", ""), props.get("year"),
                         props.get("abstract") or props.get("text"), props.get("keywords")),
                    )
                elif node_type == "Patent":
                    cur.execute(
                        "INSERT INTO patents (id, name, year, abstract, keywords, patent_number) "
                        "VALUES (%s, %s, %s, %s, %s, %s) "
                        "ON DUPLICATE KEY UPDATE name=VALUES(name), year=VALUES(year), "
                        "abstract=VALUES(abstract), keywords=VALUES(keywords), patent_number=VALUES(patent_number)",
                        (node_id, props.get("name", ""), props.get("year"),
                         props.get("abstract") or props.get("text"), props.get("keywords"),
                         props.get("patent_number")),
                    )
                elif node_type == "Report":
                    cur.execute(
                        "INSERT INTO reports (id, name, year, summary, report_type) VALUES (%s, %s, %s, %s, %s) "
                        "ON DUPLICATE KEY UPDATE name=VALUES(name), year=VALUES(year), "
                        "summary=VALUES(summary), report_type=VALUES(report_type)",
                        (node_id, props.get("name", ""), props.get("year"),
                         props.get("summary") or props.get("text"), props.get("report_type")),
                    )

            paper_author_order: Dict[str, int] = {}
            for rel in relations.get("AUTHORED", []):
                researcher = nodes.get(rel["from"])
                paper_id = rel["to"]
                if researcher and paper_id in nodes:
                    order = paper_author_order.get(paper_id, 0)
                    paper_author_order[paper_id] = order + 1
                    cur.execute(
                        "INSERT INTO paper_authors (paper_id, author_name, display_order) "
                        "VALUES (%s, %s, %s) "
                        "ON DUPLICATE KEY UPDATE display_order=VALUES(display_order)",
                        (paper_id, researcher.get("name", ""), order),
                    )

        conn.commit()
        logger.info("MariaDB 시드 완료 (%d개 노드)", len(nodes))
    finally:
        conn.close()
