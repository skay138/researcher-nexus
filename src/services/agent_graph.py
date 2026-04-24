"""
LangGraph Agent Graph
- Agent Node: LLM은 의도 분류 + 답변 생성만 담당
- Tools Node: Semantic Tools 실행 (DB 조합은 내부에서)
- 런타임 스키마 주입으로 hallucination 방지
- Grounding Pass: 최종 답변은 반드시 도구 결과만 근거로 생성
- Checkpointing으로 장애 복구
- InstrumentedToolNode로 실행 통계 수집
"""

from __future__ import annotations
from typing import Annotated, Any, List, Optional, TypedDict
import logging
import time

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from core.compiler.schema_registry import SchemaRegistry
from common.exceptions import LLMError
from services.semantic_tools import SEMANTIC_TOOLS

logger = logging.getLogger(__name__)


def _safe_utf8(text: str) -> str:
    """surrogate 문자를 replacement char(U+FFFD)로 교체하여 안전한 UTF-8 반환."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def _extract_tool_results(messages: List[BaseMessage]) -> List[str]:
    """현재 턴(가장 최근 HumanMessage 이후)에 생성된 ToolMessage 내용만 추출."""
    from langchain_core.messages import ToolMessage, HumanMessage
    
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break
            
    current_turn_messages = messages[last_human_idx + 1:] if last_human_idx != -1 else messages
    
    return [
        m.content for m in current_turn_messages
        if isinstance(m, ToolMessage) and isinstance(m.content, str) and m.content.strip()
    ]


def _get_original_query(messages: List[BaseMessage]) -> str:
    """가장 최근의 HumanMessage 내용 반환 (멀티턴 대응)."""
    from langchain_core.messages import HumanMessage
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


# Grounding pass 전용 시스템 프롬프트 — 도구 결과 외 정보 사용 금지
_GROUNDING_SYSTEM = """당신은 연구 데이터베이스 검색 결과를 정확하게 요약·전달하는 전문가입니다.

## 절대 원칙
- 아래 [검색 결과]에 명시된 정보만 사용합니다. 사전 학습 지식으로 내용을 보완하거나 추측하지 마세요.
- [검색 결과]에 없는 인물·기관·논문·특허·프로젝트는 절대 언급하지 마세요.

## 응답 규칙
1. **결과 없음**: 검색 결과가 비어있거나 `(검색 결과 없음)`이면 "조건에 맞는 결과를 찾지 못했습니다."라고 답하고 종료하세요.
2. **확인 불가 정보**: 검색 결과에서 확인할 수 없는 사실은 "검색 결과에서 확인되지 않습니다"라고 명시하세요.
3. **각주 표기**: 본문에서 검색 결과를 인용할 때 해당 항목의 번호를 [1], [2] 형식으로 표기하세요. 답변 하단에 별도 참고문헌 목록은 작성하지 마세요.
4. **인과관계 금지**: 검색 결과에 명시되지 않은 항목 간 인과관계("A의 저자가 B를 집필했으므로…")를 임의로 서술하지 마세요. "요청하신 조건에 부합하는 결과는 다음과 같습니다."처럼 담백하게 서술하세요.
5. **저자·발명자**: 항목의 `authors` 필드 값을 그대로 사용하세요. 필드가 없거나 비어있으면 "저자 정보 없음"으로 표기하고 이름을 추측하지 마세요.

## 필드별 표기 규칙
- **연구자(Researcher)**: `expertise`(전문분야), `topic`(주제) 필드 값을 그대로 사용하세요. 해당 필드가 없거나 비어있으면 "전문분야 정보 없음"으로 표기하고 절대 추측·보완하지 마세요.
- **논문·특허·보고서**: `authors` 필드가 없으면 "저자 정보 없음". `text`(요약)가 없으면 "요약 없음".
- 검색 결과에 제공된 필드 값 이외의 정보(소속, 연구 경력, 수상 이력 등)를 절대 추가하지 마세요.

## 응답 형식
- 질문의 언어로 답변합니다 (한국어 질문 → 한국어).
- 각 결과 항목의 핵심 정보(명칭, 저자/발명자, 연도, 전문분야, 요약)를 간결하게 전달하세요.
- 불필요한 서론·맺음말 없이 결과 전달에 집중하세요."""

# 도구 결과 없이 직접 답변할 때 사용하는 시스템 프롬프트
_DIRECT_ANSWER_SYSTEM = """당신은 연구 데이터베이스 검색 어시스턴트입니다.

## 질문 유형별 응답 방식

**일반 대화 및 시스템 안내** (인사, 기능 문의, 사용법 질문 등)
→ 간결하고 친절하게 직접 답변하세요.
→ 이 시스템은 논문·특허·보고서·연구자·기관·프로젝트 데이터를 그래프 기반으로 탐색합니다.

**사실 조회 질문** (특정 논문·연구자·기관·특허·프로젝트 정보 요청)
→ "해당 정보는 데이터베이스 검색이 필요합니다. 검색 결과 없이는 정확한 답변이 어렵습니다."라고 안내하세요.
→ 사전 지식으로 연구 사실(논문 목록, 연구자 이력 등)을 임의로 생성하지 마세요.

**시스템 범위 외 질문** (연구 데이터베이스와 무관한 일반 지식, 코딩 등)
→ "이 시스템은 연구 데이터베이스 검색에 특화되어 있어 해당 질문에는 답변하기 어렵습니다."라고 안내하세요.

## 원칙
- 질문의 언어로 답변합니다 (한국어 질문 → 한국어).
- 불확실한 정보를 확정적으로 서술하지 마세요."""

def _numbered_search_context(tool_results: List[str]) -> str:
    """
    ToolMessage JSON 배열을 번호 붙은 텍스트 블록으로 변환.
    LLM이 [1], [2] 형식의 각주를 달아 답변할 수 있도록 사전 포맷팅.
    """
    import json
    lines: List[str] = []
    counter = 1
    seen: set = set()

    for raw in tool_results:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue  # 파싱 불가 항목은 LLM 컨텍스트에서 제외

        for item in data.get("results", []):
            item_id = item.get("id", "")
            if item_id in seen:
                continue
            seen.add(item_id)

            name      = item.get("name", "")
            authors   = item.get("authors") or []
            year      = item.get("year", "")
            text      = item.get("text", "")
            itype     = item.get("type", "")
            expertise = item.get("expertise", "")
            topic     = item.get("topic", "")

            parts = [f"[{counter}] ({itype}) {name}"]
            if authors:
                parts.append(f"  저자: {', '.join(authors)}")
            elif itype in ("Paper", "Patent", "Report"):
                parts.append("  저자: 저자 정보 없음")

            if year:
                parts.append(f"  연도: {year}")
            if itype == "Researcher":
                parts.append(f"  전문분야: {expertise or '정보 없음'}")
                if topic:
                    parts.append(f"  주제: {topic}")
            if text:
                parts.append(f"  요약: {text}")
            lines.append("\n".join(parts))
            counter += 1

    return "\n\n".join(lines) if lines else "(검색 결과 없음)"


def _grounding_pass(llm: Any, tool_results: List[str], original_query: str, messages: List[BaseMessage]) -> Any:
    """
    도구 결과만을 근거로 최종 답변 재생성.
    LLM의 parametric knowledge 사용을 차단하기 위해
    메시지 히스토리 없이 도구 결과만 컨텍스트로 주입.
    검색 결과는 [1][2] 번호가 붙은 구조화된 텍스트로 변환해 주입.
    """
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    numbered_context = _numbered_search_context(tool_results)
    
    # 최근 2턴 정도의 대화 내용만 요약해서 컨텍스트로 추가 (지칭어 해상용)
    history_context = []
    for m in messages[-4:]: # 최근 2턴 (Human + AI * 2)
        if isinstance(m, HumanMessage):
            history_context.append(f"User: {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            history_context.append(f"Assistant: {m.content[:200]}...")
            
    history_str = "\n".join(history_context)
    full_prompt_body = f"## 대화 맥락 (최근)\n{history_str}\n\n질문: {original_query}\n\n[검색 결과]\n{numbered_context}"

    logger.info (
        "[GroundingPass] query=%r | history_turns=%d | context_lines=%d",
        original_query, len(history_context), len(numbered_context.splitlines()),
    )

    return llm.invoke([
        SystemMessage(content=_GROUNDING_SYSTEM),
        HumanMessage(content=full_prompt_body),
    ])


# ────────────────────────────────────────────────────────────────────────────
# State
# ────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:        Annotated[List[BaseMessage], add_messages]
    # Observability 필드
    tool_call_count: int
    total_db_calls:  int
    session_id:      Optional[str]
    start_time:      float
    max_tool_calls:  int   # 무한 루프 방지용 호출 한도 (기본 3)


# ────────────────────────────────────────────────────────────────────────────
# System Prompt Builder (런타임 스키마 주입)
# ────────────────────────────────────────────────────────────────────────────

_DEFAULT_MAX_TOOL_CALLS = 3


def build_planner_prompt(schema_registry: SchemaRegistry) -> str:
    """Planner 에이전트 프롬프트. 도구 선택 및 다단계 탐색에 특화."""
    schema_text = schema_registry.get_schema_for_llm()

    return f"""당신은 연구 데이터베이스 탐색 전문 에이전트(Planner)입니다.
`execute_dynamic_search` 도구를 사용하여 Neo4j 그래프 DB와 Vector DB에서 논문·특허·보고서·연구자·기관·프로젝트 데이터를 탐색합니다.

## 도구 호출 전략

### 1단계: 검색 필요 여부 판단
- 인사·기능 문의·시스템 안내 질문이면 도구를 호출하지 말고 직접 "DONE"을 출력하세요.
- 특정 데이터를 조회하는 질문이면 아래 절차대로 도구를 호출하세요.

### 2단계: 최적 진입점 선택
- `vector_search_concept`: 질문에서 가장 구체적이고 고유한 엔티티 또는 핵심 개념을 사용합니다.
  - 좋은 예: "자율운항선박 충돌 회피", "김철수 (연구자명)", "그린십 프로젝트"
  - 나쁜 예: "논문", "연구자", "최신" (너무 광범위)
- `vector_search_node_type`: 탐색 시작 노드 타입. 질문에서 첫 번째로 언급되는 구체적 엔티티 타입.

### 3단계: 홉(hop) 설계
- 홉이 필요 없는 경우 (단순 벡터 검색): `neo4j_hops`를 빈 배열 `[]`로 설정하세요.
  - 예: "해양 관련 논문 목록", "신약 개발 특허 검색"
- 홉이 필요한 경우: 질문의 탐색 경로를 관계 단위로 분해하세요. 최대 4홉을 권장합니다.
- `final_vector_filter_concept`: 최종 노드를 **주제·토픽**으로 좁힐 때만 사용합니다. 노드 타입명("논문", "연구자")은 넣지 마세요.

### 4단계: 호출 횟수 관리
- 기본은 **1회 호출**입니다.
- 결과가 부족하거나 빈 배열이면 파라미터를 조정해 **최대 {_DEFAULT_MAX_TOOL_CALLS}회까지** 재호출하세요.
- 아래 조건 중 하나라도 해당되면 **추가 도구 호출 없이 "DONE"을 출력**하세요:
  - 결과가 5개 이상 수집된 경우
  - 이전 결과에 이미 필요한 정보가 충분히 포함된 경우
  - 도구 호출 횟수가 {_DEFAULT_MAX_TOOL_CALLS}회에 도달한 경우
  - 재호출해도 결과가 개선되지 않을 것으로 판단되는 경우

## 관계 방향 (direction)

| direction | Cypher 패턴 | 사용 시점 |
|-----------|-------------|-----------|
| `"out"` | `(A)-[REL]->(B)` | A에서 B 방향으로 저장된 관계를 순방향 탐색 |
| `"in"` | `(B)-[REL]->(A)` | B→A 방향으로 저장된 관계를 역방향으로 탐색 |

- "프로젝트에 참여한 연구자" → Researcher-[PARTICIPATED_IN]->Project 구조이므로 Project에서 Researcher 역방향: `direction: "in"`
- "연구자가 소속된 기관" → Researcher-[BELONGS_TO]->Organization 구조이므로 순방향: `direction: "out"`

## 도구 선택 기준

| 상황 | 사용 도구 |
|------|-----------|
| 개념·키워드로 탐색 (일반 검색, 그래프 탐색) | `execute_dynamic_search` |
| 이전 결과의 특정 ID 상세 조회 ("이 논문 자세히", "앞 결과 중 ID xxx") | `get_node_by_ids` |

## 멀티턴 대화
"그 연구자", "해당 프로젝트" 같은 지칭어는 이전 턴 결과에서 실제 엔티티명을 찾아 `vector_search_concept`에 사용하세요.
이전 결과의 ID를 직접 참조하는 후속 질문은 `get_node_by_ids`를 사용하세요.

## 호출 예시

### 예시 1: 단순 벡터 검색 (홉 없음) — "수소 연료전지 관련 최신 논문"
```
vector_search_concept: "수소 연료전지"
vector_search_node_type: "Paper"
neo4j_hops: []
```

### 예시 2: 1홉 — "해양 기술 특허를 출원한 연구자 목록"
```
vector_search_concept: "해양 기술"
vector_search_node_type: "Patent"
neo4j_hops: [
  {{"from_type": "Patent", "relation_concept": "invented", "to_type": "Researcher", "direction": "in"}}
]
```

### 예시 3: 2홉 — "김철수 연구자가 소속된 기관이 수행한 프로젝트"
```
vector_search_concept: "김철수"
vector_search_node_type: "Researcher"
neo4j_hops: [
  {{"from_type": "Researcher", "relation_concept": "belongs_to", "to_type": "Organization", "direction": "out"}},
  {{"from_type": "Organization", "relation_concept": "participation", "to_type": "Project", "direction": "in"}}
]
```

### 예시 4: 4홉 + 필터 + final_filter — "2023년 이전 해양 사업에 참여한 연구자가 소속된 기관의 다른 연구자가 작성한 자율운항 관련 논문"
```
vector_search_concept: "해양 사업"
vector_search_node_type: "Project"
vector_search_filters: {{"year": {{"lt": 2023}}}}
neo4j_hops: [
  {{"from_type": "Project", "relation_concept": "participation", "to_type": "Researcher", "direction": "in"}},
  {{"from_type": "Researcher", "relation_concept": "belongs_to", "to_type": "Organization", "direction": "out"}},
  {{"from_type": "Organization", "relation_concept": "belongs_to", "to_type": "Researcher", "direction": "in"}},
  {{"from_type": "Researcher", "relation_concept": "authored", "to_type": "Paper", "direction": "out"}}
]
final_vector_filter_concept: "자율운항"
```

### 예시 5: K-해양 프로젝트에서 생산된 보고서
```
vector_search_concept: "K-해양 프로젝트"
vector_search_node_type: "Project"
neo4j_hops: [
  {{"from_type": "Project", "relation_concept": "produced", "to_type": "Report", "direction": "out"}}
]
```

## 사용 가능한 스키마
{schema_text}
"""


# ────────────────────────────────────────────────────────────────────────────
# Instrumented Tool Node
# ────────────────────────────────────────────────────────────────────────────

class InstrumentedToolNode(ToolNode):
    """
    ToolNode를 래핑하여 실행 시간 및 호출 통계 수집.
    """

    def invoke(
        self,
        state: AgentState,
        config: Optional[Any] = None,
    ) -> dict:
        t0 = time.time()
        result = super().invoke(state, config)
        elapsed = time.time() - t0

        new_count = state.get("tool_call_count", 0) + 1
        logger.info(
            "[ToolNode] call #%d | elapsed=%.2fs",
            new_count, elapsed,
        )

        return {
            **result,
            "tool_call_count": new_count,
        }


# ────────────────────────────────────────────────────────────────────────────
# Planner & Agent Nodes
# ────────────────────────────────────────────────────────────────────────────

def make_planner_node(llm: Any, schema_registry: SchemaRegistry):
    """
    도구 선택 및 다단계 탐색을 수행하는 Planner 노드
    """
    llm_with_tools = llm.bind_tools(SEMANTIC_TOOLS)

    def planner_node(state: AgentState) -> dict:
        system_msg = SystemMessage(content=build_planner_prompt(schema_registry))
        raw_messages = [system_msg] + list(state["messages"])

        # surrogate 제어
        safe_messages = [
            msg.model_copy(update={"content": _safe_utf8(msg.content)})
            if isinstance(getattr(msg, "content", None), str) else msg
            for msg in raw_messages
        ]

        try:
            response = llm_with_tools.invoke(safe_messages)
        except Exception as e:
            logger.error(
                "[PlannerNode] LLM invocation failed: %s (%s) | session=%s",
                e, type(e).__name__, state.get("session_id", "?"),
            )
            raise LLMError(f"LLM invocation failed: {e}") from e

        # 도구 호출 방어를 위해 content만 surrogate 처리 (슬라이싱 제거)
        tool_calls = getattr(response, "tool_calls", []) or []
        content = getattr(response, "content", "")
        safe_content = _safe_utf8(content) if isinstance(content, str) else content

        if isinstance(content, str):
            response = response.model_copy(
                update={"content": safe_content}
            )

        logger.info(
            "[PlannerNode] tool_calls=%d | session=%s",
            len(tool_calls),
            state.get("session_id", "?"),
        )

        return {"messages": [response]}

    return planner_node


def make_agent_node(llm: Any):
    """
    도구 결과(또는 바로)를 바탕으로 최종 답변을 생성하는 노드
    """
    def agent_node(state: AgentState) -> dict:
        tool_results = _extract_tool_results(state["messages"])
        original_query = _get_original_query(state["messages"])

        if tool_results:
            try:
                response = _grounding_pass(llm, tool_results, original_query, state["messages"])
                logger.info(
                    "[AgentNode] grounding_pass_applied | session=%s",
                    state.get("session_id", "?"),
                )
            except Exception as e:
                logger.error(
                    "[AgentNode] grounding_pass_failed: %s (%s) | session=%s",
                    e, type(e).__name__, state.get("session_id", "?"),
                )
                raise LLMError(f"Grounding pass failed: {e}") from e
        else:
            from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
            
            history_context = []
            # Planner의 불필요한 메세지를 제외하기 위해 마지막 메세지(-1) 전까지의 이력 수집
            for m in state["messages"][-6:-1]:
                if isinstance(m, HumanMessage):
                    history_context.append(f"User: {m.content}")
                elif isinstance(m, AIMessage) and m.content and m.content != "DONE":
                    history_context.append(f"Assistant: {m.content[:200]}...")
            
            history_str = "\n".join(history_context)
            full_prompt_body = f"## 대화 맥락 (최근)\n{history_str}\n\n질문: {original_query}"

            response = llm.invoke([
                SystemMessage(content=_DIRECT_ANSWER_SYSTEM),
                HumanMessage(content=full_prompt_body),
            ])
            logger.info(
                "[AgentNode] direct_pass_applied | session=%s",
                state.get("session_id", "?"),
            )

        if isinstance(getattr(response, "content", None), str):
            response = response.model_copy(
                update={"content": _safe_utf8(response.content)}
            )

        return {"messages": [response]}

    return agent_node


# ────────────────────────────────────────────────────────────────────────────
# Routing
# ────────────────────────────────────────────────────────────────────────────

def _make_routing(max_calls: int = _DEFAULT_MAX_TOOL_CALLS):
    """Planner 다음 엣지 라우팅: 호출 횟수 초과 시 Agent로 강제 이동.
    state["max_tool_calls"]가 있으면 요청별 오버라이드로 우선 적용."""
    def _route(state: AgentState) -> str:
        limit = state.get("max_tool_calls") or max_calls
        if state.get("tool_call_count", 0) >= limit:
            logger.warning(
                "[Planner] max_tool_calls=%d reached, forcing Agent | session=%s",
                limit, state.get("session_id", "?"),
            )
            return "__end__"
        return tools_condition(state)
    return _route


# ────────────────────────────────────────────────────────────────────────────
# Graph Builder
# ────────────────────────────────────────────────────────────────────────────

def build_graph(
    schema_registry: SchemaRegistry,
    llm: Any,
    checkpointer=None,
    max_tool_calls: int = _DEFAULT_MAX_TOOL_CALLS,
) -> Any:
    """
    LangGraph 에이전트 그래프 빌드.

    Args:
        schema_registry: 런타임 스키마 주입용
        llm:              도구가 바인딩되기 전의 LLM 인스턴스
        checkpointer:    장애 복구용 (기본: MemorySaver)

    Returns:
        컴파일된 LangGraph 앱
    """
    planner_fn = make_planner_node(llm, schema_registry)
    agent_fn   = make_agent_node(llm)
    tools_node = InstrumentedToolNode(SEMANTIC_TOOLS)

    workflow = StateGraph(AgentState)
    workflow.add_node("Planner", planner_fn)
    workflow.add_node("Tools", tools_node)
    workflow.add_node("Agent", agent_fn)

    workflow.set_entry_point("Planner")

    # Planner가 도구를 호출하면 Tools로, 호출이 없거나 한도 초과 시 Agent로 이동
    workflow.add_conditional_edges(
        "Planner",
        _make_routing(max_tool_calls),
        {"tools": "Tools", "__end__": "Agent"}
    )

    workflow.add_edge("Tools", "Planner")
    workflow.add_edge("Agent", END)

    cp = checkpointer or MemorySaver()
    return workflow.compile(checkpointer=cp)


# ────────────────────────────────────────────────────────────────────────────
# Run helper
# ────────────────────────────────────────────────────────────────────────────

def run_query(app, query: str, session_id: str = "default") -> str:
    """
    에이전트 실행 헬퍼.

    Args:
        app:        build_graph()로 생성한 앱
        query:      사용자 질문
        session_id: 체크포인팅용 세션 ID

    Returns:
        에이전트 최종 답변 텍스트
    """
    from langchain_core.messages import HumanMessage
    from common.settings import get_settings

    initial_state: AgentState = {
        "messages":        [HumanMessage(content=query)],
        "tool_call_count": 0,
        "total_db_calls":  0,
        "session_id":      session_id,
        "start_time":      time.time(),
        "max_tool_calls":  _DEFAULT_MAX_TOOL_CALLS,
    }

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": get_settings().recursion_limit,
    }
    final_state = app.invoke(initial_state, config=config)

    # 마지막 AI 메시지 추출
    for msg in reversed(final_state["messages"]):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            return msg.content

    return "(응답 없음)"
