# ResearchNexus

논문·특허·보고서·연구자를 그래프로 연결해 탐색하는 연구 데이터베이스 검색 에이전트입니다.

---

## 해결한 문제

| # | 문제 | 해결 방법 |
|---|------|-----------|
| 1 | LLM이 Cypher 직접 생성 (확률적) | `CypherCompiler` 분리 — 완전 결정론적 |
| 2 | Hop Explosion 제어 없음 | Cypher `LIMIT` + `BeamPruner` 2중 방어 |
| 3 | 스키마를 LLM에 하드코딩 | `SchemaRegistry` 런타임 동적 주입 |
| 4 | Low-level 도구 3개 직접 조합 | `execute_dynamic_search` 단일 시맨틱 도구 |

---

## 아키텍처

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│  L1: Intent Layer               │  ← LLM 담당 (Planner + Agent)
│  · 사용자 의도 → QueryPlan      │
│  · SchemaRegistry 런타임 주입    │
└────────────────┬────────────────┘
                 │  QueryPlan (추상, Cypher 없음)
                 ▼
┌─────────────────────────────────┐
│  L2: Compile Layer              │  ← 결정론적
│  CypherCompiler                 │
│  · relation_concept → CYPHER    │
│  · hop별 LIMIT 강제              │
└────────────────┬────────────────┘
                 │  Cypher Query
                 ▼
┌─────────────────────────────────┐
│  L3: Execution Layer            │  ← hop-by-hop
│  ExecutionEngine                │
│  · Vector DB → Graph (hop loop) │
│  · BeamPruner: 중간 결과 압축   │
│  · Redis 캐시                    │
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│  L4: Data Layer                 │
│  Neo4j │ Milvus │ Redis         │
└─────────────────────────────────┘
```

---

## 폴더 구조

```
researcher-nexus/
├── src/                         # 애플리케이션 소스 코드
│   ├── main.py                  # 서버 진입점
│   ├── app_factory.py           # create_engine / create_app
│   │
│   ├── api/                     # FastAPI REST API
│   │   ├── app.py
│   │   ├── middleware.py        # Correlation ID
│   │   ├── schemas.py
│   │   └── routes/
│   │       ├── health.py        # GET /api/v1/health
│   │       └── search.py        # POST /api/v1/agent/query (SSE)
│   │                            # POST /api/v1/engine/search
│   │
│   ├── common/                  # 공통 모듈
│   │   ├── settings.py          # pydantic-settings 중앙 설정 (인프라)
│   │   ├── config_service.py    # QueryConfig + ConfigService (실행 파라미터)
│   │   ├── query_plan.py        # LLM 출력 스키마 (Pydantic)
│   │   ├── cache.py             # Redis 캐시 추상화
│   │   ├── exceptions.py        # 도메인 예외 계층
│   │   └── logging.py           # 구조화 로깅
│   │
│   ├── core/                    # 비즈니스 로직 (DB 의존 없음)
│   │   ├── compiler/
│   │   │   ├── schema_registry.py   # 동적 스키마 주입 + concept 매핑
│   │   │   └── cypher_compiler.py   # 결정론적 Cypher 생성
│   │   └── executor/
│   │       ├── beam_pruner.py       # Hop Explosion 2차 방어
│   │       └── execution_engine.py  # hop-by-hop 실행 + 캐싱 + 통계
│   │
│   ├── infrastructure/          # DB 어댑터 (콜백 주입 패턴)
│   │   ├── config_repository.py # Mock ConfigRepository (RDB 전환 예정)
│   │   ├── neo4j.py             # Neo4j 어댑터
│   │   └── milvus.py            # Milvus 어댑터
│   │
│   └── services/                # 애플리케이션 서비스
│       ├── agent_graph.py       # LangGraph 3-node 에이전트
│       └── semantic_tools.py    # execute_dynamic_search (단일 도구)
│
├── tests/                       # pytest 테스트 스위트
├── scripts/
│   └── seed_data.py             # Neo4j + Milvus 초기 데이터 적재
├── docker-compose.yml           # Neo4j + Milvus + Redis 로컬 스택
├── pytest.ini                   # pythonpath = src
└── .env.example
```

### 설정 계층

| 계층 | 위치 | 대상 |
|------|------|------|
| **인프라 설정** | `.env` → `settings.py` | DB URI, 포트, Redis URL, LLM 접속 URL 등 |
| **실행 파라미터** | `config_repository.py` → `ConfigService` | beam_width, max_results, model, temperature 등 |
| **요청별 오버라이드** | API 파라미터 (`QueryConfigSchema`) | 위 실행 파라미터를 요청 단위로 덮어씀 |

---

## 빠른 시작

### 1. 의존성 설치

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 환경 설정

```bash
cp .env.example .env
# .env에서 NEO4J_PASSWORD, REDIS_URL 등 실제 값으로 수정
```

### 3. 인프라 기동 (Docker)

```bash
docker compose up -d
docker compose ps
```

### 4. 초기 데이터 적재

```bash
# Neo4j + Milvus 모두 시드
python scripts/seed_data.py

# 개별 시드
python scripts/seed_data.py --neo4j-only
python scripts/seed_data.py --milvus-only

# 데이터 초기화 후 재적재
python scripts/seed_data.py --clear
```

### 5. 서버 실행

```bash
# 개발 (Windows)
$env:PYTHONPATH="src"; uvicorn src.main:app --reload

# 개발 (macOS/Linux)
PYTHONPATH=src uvicorn src.main:app --reload

# 또는 직접 실행
python src/main.py
```

### 6. 테스트

```bash
pytest tests/ -v
```

---

## Docker 환경

### 서비스 목록

| 서비스 | 포트 | 용도 |
|--------|------|------|
| **api** | **8000** | **FastAPI 서버** |
| Neo4j | 7687 (Bolt) / 7474 (UI) | 그래프 DB |
| Milvus | 19530 | 벡터 DB |
| Redis | 6379 | 쿼리 캐시 |
| MinIO | 9001 (UI) | Milvus 오브젝트 스토리지 |
| etcd | — (내부) | Milvus 메타데이터 |

```bash
# 전체 스택 기동
docker compose up -d

# Milvus GUI (선택)
docker compose --profile tools up -d
# → http://localhost:8080

# 중지 (데이터 유지)
docker compose down

# 중지 + 볼륨 삭제 (데이터 초기화)
docker compose down -v
```

> **Hot-reload**: `ENVIRONMENT=development`(기본값) 설정 시 소스 수정 시 api 컨테이너가 자동 재시작됩니다.

---

## API 엔드포인트

```
GET  /api/v1/health           — 헬스체크 + 컴포넌트 상태
GET  /api/v1/schema           — 현재 DB 스키마 조회
POST /api/v1/agent/query      — LLM 에이전트 쿼리 (SSE 스트리밍)
POST /api/v1/engine/search    — QueryPlan 직접 실행 (동기)
```

**에이전트 쿼리 예시**

```bash
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "보트 관련 논문을 쓴 연구자의 소속 기관은?", "session_id": "test-1"}' \
  --no-buffer
```

**요청별 파라미터 오버라이드 예시**

```bash
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "해양 에너지 관련 특허를 발명한 연구자",
    "session_id": "test-2",
    "config": {
      "model": "llama3:8b",
      "temperature": 0.2,
      "beam_width": 30,
      "max_results": 10
    }
  }' \
  --no-buffer
```

---

## Hop Explosion 방어 메커니즘

```
Project [2개]
    │  PARTICIPATED_IN (inbound)   ← Cypher LIMIT 500
    ▼
Researcher [→ BeamPruner → 50개]
    │  BELONGS_TO (outbound)       ← Cypher LIMIT 500
    ▼
Organization [→ BeamPruner → 50개]
    │  BELONGS_TO (inbound)        ← Cypher LIMIT 500
    ▼
Researcher [→ BeamPruner → 50개]
    │  AUTHORED (outbound)         ← Cypher LIMIT 500
    ▼
Paper [→ FinalFilter (Vector 재검색)]
    ▼
최종 결과 [max_results개]
```
