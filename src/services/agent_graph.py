"""
LangGraph Agent Graph
- Planner:        execute_dynamic_search만 사용 — 검색·탐색 전담
- SearchTools:    Planner 도구 실행 노드
- DetailEnricher: 검색 완료 후 상세 정보 필요 여부 판단 (LLM)
- DetailTools:    get_details_by_ids 실행 노드 (MariaDB)
- Agent:          도구 결과 기반 최종 답변 생성 (Grounding Pass)

흐름:
  Planner ─→ SearchTools ─→ Planner (loop, max_tool_calls)
          └→ DetailEnricher ─→ DetailTools ─→ Agent
                             └→ Agent
"""

from __future__ import annotations
from typing import Annotated, Any, List, Optional, TypedDict, Dict
import logging
import time

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from core.compiler.schema_registry import SchemaRegistry
from common.utils.exceptions import LLMError
from services.tool_result import numbered_search_context

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


def _extract_latest_tool_results(messages: List[BaseMessage]) -> List[str]:
    """
    현재 턴의 도구 결과가 없으면 직전 턴의 도구 결과로 fallback.
    멀티턴에서 "앞 결과의 초록 보여줘" 같은 후속 상세 요청 처리용.
    """
    from langchain_core.messages import ToolMessage, HumanMessage

    human_positions = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]

    # 최신 턴부터 역순으로 도구 결과 탐색
    for turn_idx in range(len(human_positions) - 1, -1, -1):
        start = human_positions[turn_idx]
        end = human_positions[turn_idx + 1] if turn_idx + 1 < len(human_positions) else len(messages)
        turn_results = [
            m.content for m in messages[start:end]
            if isinstance(m, ToolMessage) and isinstance(m.content, str) and m.content.strip()
        ]
        if turn_results:
            return turn_results

    return []


def _get_original_query(messages: List[BaseMessage]) -> str:
    """가장 최근의 HumanMessage 내용 반환 (멀티턴 대응)."""
    from langchain_core.messages import HumanMessage
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


# ────────────────────────────────────────────────────────────────────────────
# System Prompts
# ────────────────────────────────────────────────────────────────────────────

_GROUNDING_SYSTEM = """당신은 연구 데이터베이스 검색 결과를 전달하는 전문가이다.

## 절대 원칙
- [검색 결과]에 명시된 정보만 사용한다. 사전 학습 지식으로 내용을 추측·보완하지 않는다.
- [검색 결과]에 없는 인물·기관·논문·특허·프로젝트는 절대 언급하지 않는다.

## 응답 규칙
1. **결과 없음**: 검색 결과가 비어있거나 `(검색 결과 없음)`이면 "조건에 맞는 결과를 찾지 못했습니다."라고만 답한다.
2. **있는 것만 전달**: 검색 결과에 있는 필드 값만 전달한다. 없는 필드는 언급하지 않는다.
3. **중복 금지**: 같은 정보를 두 번 반복하지 않는다.
4. **마무리 문구 금지**: "추가 정보는 확인되지 않습니다" 류의 면책 문구로 답변을 끝내지 않는다.
5. **각주**: 결과 항목을 본문에서 언급할 때 `[1]`, `[2]` 형식으로 표기한다.
6. **ID 숨김**: 검색 결과의 `id:` 줄은 내부 참조용이므로 답변에 포함하지 않는다.

## 응답 형식
- 질문의 언어로 답변한다.
- 각 결과 항목의 핵심 정보를 간결하게 전달한다.
- 서론·맺음말 없이 결과 전달에 집중한다."""

_DIRECT_ANSWER_SYSTEM = """당신은 연구 데이터베이스 검색 어시스턴트이다.

## 질문 유형별 응답 방식

**일반 대화 및 시스템 안내** (인사, 기능 문의, 사용법 질문 등)
→ 간결하고 친절하게 직접 답변한다.
→ 이 시스템은 논문·특허·보고서·연구자·기관·프로젝트 데이터를 그래프 기반으로 탐색한다.

**사실 조회 질문** (특정 논문·연구자·기관·특허·프로젝트 정보 요청)
→ "검색을 실행했지만 결과를 가져오지 못했습니다. 질문을 구체적으로 바꿔 다시 시도해 주세요."라고 안내한다.
→ 사전 지식으로 연구 사실을 임의로 생성하지 않는다.

**시스템 범위 외 질문**
→ "이 시스템은 연구 데이터베이스 검색에 특화되어 있어 해당 질문에는 답변하기 어렵습니다."라고 안내한다.

## 원칙
- 질문의 언어로 답변한다 (한국어 질문 → 한국어).
- 불확실한 정보를 확정적으로 서술하지 않는다."""

_ENRICHER_SYSTEM = """당신은 검색 결과 보강 판단 전문가이다.
사용자 질문과 현재 검색 결과를 검토하여, 상세 정보가 더 필요한지 판단한다.

## get_details_by_ids 호출 기준 (다음 중 하나라도 해당되면 호출)
- 사용자가 논문·특허·보고서의 '내용', '초록', '전체 설명'을 요청
- 사용자가 저자·발명자 목록을 요청했으나 검색 결과에 authors 필드 없음
- 사용자가 특정 ID의 상세 정보를 명시적으로 요청
- 검색 결과의 text 필드가 없거나 매우 짧아 질문에 답하기 부족

## 호출하지 않아야 할 경우
- 검색 결과에 이미 충분한 정보가 있는 경우 (이름, 연도, 요약 포함)
- 사용자가 목록·개수·존재 여부만 확인하는 경우
- 참조 가능한 검색 결과가 전혀 없는 경우
- 인사·기능 안내 등 검색과 무관한 질문
- 제공된 검색 결과가 현재 질문의 주제와 **다른 경우** (이 경우 새 검색이 필요하므로 호출하지 않는다)

## 멀티턴 참고
검색 결과는 현재 턴 또는 이전 턴의 결과이다.
"앞 결과", "그 논문", "방금 찾은" 같은 지칭어는 제공된 결과를 참조한다.
제공된 결과가 현재 질문과 무관한 주제라면 호출하지 않는다.

호출이 필요한 경우, 관련 ID를 모두 추출하여 한 번에 호출한다."""



def _grounding_pass(llm: Any, tool_results: List[str], original_query: str, messages: List[BaseMessage]) -> Any:
    """도구 결과만을 근거로 최종 답변 재생성."""
    from langchain_core.messages import HumanMessage, AIMessage

    numbered_context = numbered_search_context(tool_results)

    history_context = []
    for m in messages[-4:]:
        if isinstance(m, HumanMessage):
            history_context.append(f"User: {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            history_context.append(f"Assistant: {m.content[:200]}...")

    history_str = "\n".join(history_context)
    full_prompt_body = (
        f"## 대화 맥락 (최근)\n{history_str}\n\n"
        f"질문: {original_query}\n\n[검색 결과]\n{numbered_context}"
    )

    logger.info(
        "[GroundingPass] query=%r | context_lines=%d",
        original_query, len(numbered_context.splitlines()),
    )

    return llm.invoke([
        SystemMessage(content=_GROUNDING_SYSTEM),
        HumanMessage(content=full_prompt_body),
    ])


# ────────────────────────────────────────────────────────────────────────────
# State
# ────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:          Annotated[List[BaseMessage], add_messages]
    tool_call_count:   int
    empty_result_count: int   # 연속 빈 결과 횟수 — 2회 도달 시 강제 탈출
    total_db_calls:    int
    session_id:        Optional[str]
    start_time:        float
    max_tool_calls:    int


# ────────────────────────────────────────────────────────────────────────────
# Planner System Prompt
# ────────────────────────────────────────────────────────────────────────────

_DEFAULT_MAX_TOOL_CALLS = 3


def build_planner_prompt(schema_registry: SchemaRegistry) -> str:
    schema_text = schema_registry.get_schema_for_llm()

    return f"""당신은 연구 데이터베이스 탐색 에이전트(Planner)이다.
`execute_dynamic_search` 도구로 논문, 특허, 보고서, 연구자, 기관, 프로젝트를 탐색한다.
상세 정보(초록, 저자 목록 등)는 검색 후 자동 보강된다. 이 단계에서는 검색에만 집중한다.

## 출력 규칙

출력은 오직 두 가지이다. **자연어 답변 금지.**
- **도구 호출**: 기관, 연구자, 논문, 특허, 보고서, 프로젝트에 관한 모든 질문
- **"DONE"**: 아래 경우에만 허용
  - 순수 인사·시스템 기능 문의 (예: "안녕", "어떤 기능이야")
  - 이전 결과가 현재 질문과 **동일한** 주제·대상이고 상세 요청인 경우
  - 빈 결과 2회 연속 → 데이터 없음 판단
  - 도구 호출 {_DEFAULT_MAX_TOOL_CALLS}회 도달

**이전 결과가 현재 질문과 다른 주제·대상이면 반드시 새로 검색한다.**

## 도구 호출 전략

**진입점**: `vector_search_concept`에 가장 구체적인 엔티티·핵심 개념을 사용한다.
- 좋은 예: "자율운항선박 충돌 회피", "김철수", "그린십 프로젝트"
- 나쁜 예: "논문", "연구자", "최신" (너무 광범위)

**홉 설계**: 탐색 경로를 관계 단위로 분해한다. 홉 없으면 `neo4j_hops: []`.
`final_vector_filter_concept`: 최종 노드를 주제로 좁힐 때만 사용한다 (노드 타입명 제외).

**재호출**: 빈 결과 시 **1회만** 재시도한다.
- entry 0건 → `vector_search_concept` 변경
- hop 0 rows → 스키마 방향 재확인 후 `direction` 수정

## 관계 방향 (direction)

스키마의 `(From)-[REL]->(To)` 기준으로 결정한다.
- `"out"`: 현재 노드가 From → To 방향으로 탐색 (순방향)
- `"in"`: 현재 노드가 To, 역방향으로 From 탐색 (역방향)

예 — `(Researcher)-[AFFILIATED_WITH]->(Organization)`:
- "기관 소속 연구자": Organization 진입 → Researcher 역탐색 → `direction: "in"`
- "연구자 소속 기관": Researcher 진입 → Organization 순방향 → `direction: "out"`

## 멀티턴

"그 연구자", "해당 프로젝트" 같은 지칭어는 이전 턴 결과에서 실제 엔티티명을 추출하여 `vector_search_concept`에 사용한다.

## 호출 예시

### 예시 1: 벡터 검색 — "수소 연료전지 관련 논문"
```
vector_search_concept: "수소 연료전지"
vector_search_node_type: "Paper"
neo4j_hops: []
```

### 예시 2: 1홉 — "KRISO 소속 연구자 목록"
```
vector_search_concept: "KRISO"
vector_search_node_type: "Organization"
neo4j_hops: [
  {{"from_type": "Organization", "relation_concept": "affiliated_with", "to_type": "Researcher", "direction": "in"}}
]
```
※ AFFILIATED_WITH = (Researcher)→(Organization). Organization 진입 시 Researcher는 역방향 → direction: "in"

### 예시 3: 2홉 — "김철수가 소속된 기관이 수행한 프로젝트"
```
vector_search_concept: "김철수"
vector_search_node_type: "Researcher"
neo4j_hops: [
  {{"from_type": "Researcher", "relation_concept": "belongs_to", "to_type": "Organization", "direction": "out"}},
  {{"from_type": "Organization", "relation_concept": "produced", "to_type": "Project", "direction": "out"}}
]
```

### 예시 4: 2홉 — "특정 프로젝트에 참여한 연구자들이 소속된 기관"
```
vector_search_concept: "그린십 프로젝트"
vector_search_node_type: "Project"
neo4j_hops: [
  {{"from_type": "Project", "relation_concept": "participation", "to_type": "Researcher", "direction": "in"}},
  {{"from_type": "Researcher", "relation_concept": "belongs_to", "to_type": "Organization", "direction": "out"}}
]
```

## 사용 가능한 스키마
{schema_text}
"""


# ────────────────────────────────────────────────────────────────────────────
# Tool Result Logging
# ────────────────────────────────────────────────────────────────────────────

def _log_tool_results(label: str, messages: list, elapsed: float, call_num: Optional[int] = None) -> None:
    """ToolMessage 결과를 파싱하여 요약 로그 출력."""
    import json
    from langchain_core.messages import ToolMessage

    for msg in messages:
        if not (isinstance(msg, ToolMessage) and isinstance(msg.content, str)):
            continue
        try:
            data = json.loads(msg.content)
            total   = data.get("total", "?")
            path    = data.get("path", "") or "-"
            results = data.get("results", [])
            names   = [r.get("name") or r.get("id", "?") for r in results[:3]]
            preview = ", ".join(names)
            if len(results) > 3:
                preview += f" ... +{len(results) - 3}"
            num_str = f" #{call_num}" if call_num is not None else ""
            logger.info(
                "[%s]%s total=%s | path=%r | %.2fs | [%s]",
                label, num_str, total, path, elapsed, preview,
            )
        except (json.JSONDecodeError, TypeError):
            logger.info("[%s] elapsed=%.2fs | (파싱 불가)", label, elapsed)


# ────────────────────────────────────────────────────────────────────────────
# Instrumented Tool Nodes
# ────────────────────────────────────────────────────────────────────────────

class InstrumentedToolNode(ToolNode):
    """SearchTools — tool_call_count / empty_result_count 갱신 + 결과 로깅."""

    def invoke(self, state: AgentState, config: Optional[Any] = None) -> dict:
        import json
        from langchain_core.messages import ToolMessage

        t0 = time.time()
        result = super().invoke(state, config)
        elapsed = time.time() - t0
        new_count = state.get("tool_call_count", 0) + 1
        _log_tool_results("SearchTools", result.get("messages", []), elapsed, new_count)

        # 빈 결과(total=0) 연속 횟수 추적
        is_empty = all(
            json.loads(m.content).get("total", 1) == 0
            for m in result.get("messages", [])
            if isinstance(m, ToolMessage)
            and isinstance(m.content, str)
            and m.content.strip()
            and not m.content.startswith("{\"type\": \"error\"}")
        )
        prev_empty = state.get("empty_result_count", 0)
        new_empty = (prev_empty + 1) if is_empty else 0

        return {**result, "tool_call_count": new_count, "empty_result_count": new_empty}


class InstrumentedDetailToolNode(ToolNode):
    """DetailTools — 결과 로깅 (tool_call_count 증가 없음)."""

    def invoke(self, state: AgentState, config: Optional[Any] = None) -> dict:
        t0 = time.time()
        result = super().invoke(state, config)
        elapsed = time.time() - t0
        _log_tool_results("DetailTools", result.get("messages", []), elapsed)

        # 검증(test_questions.py) 및 루프 제어를 위해 카운트 증가
        prev_count = state.get("tool_call_count", 0)
        return {**result, "tool_call_count": prev_count + 1}


# ────────────────────────────────────────────────────────────────────────────
# Nodes
# ────────────────────────────────────────────────────────────────────────────

def make_planner_node(llm: Any, schema_registry: SchemaRegistry, search_tools: list):
    llm_with_tools = llm.bind_tools(search_tools)

    def planner_node(state: AgentState) -> dict:
        system_msg = SystemMessage(content=build_planner_prompt(schema_registry))
        raw_messages = [system_msg] + list(state["messages"])

        safe_messages = [
            msg.model_copy(update={"content": _safe_utf8(msg.content)})
            if isinstance(getattr(msg, "content", None), str) else msg
            for msg in raw_messages
        ]

        try:
            response = llm_with_tools.invoke(safe_messages)
        except Exception as e:
            logger.error("[PlannerNode] LLM failed: %s | session=%s", e, state.get("session_id", "?"))
            raise LLMError(f"LLM invocation failed: {e}") from e

        tool_calls = getattr(response, "tool_calls", []) or []
        content = getattr(response, "content", "")
        if isinstance(content, str):
            response = response.model_copy(update={"content": _safe_utf8(content)})

        logger.info(
            "[PlannerNode] new_calls=%d | total_calls=%d | session=%s",
            len(tool_calls), state.get("tool_call_count", 0), state.get("session_id", "?"),
        )
        return {"messages": [response]}

    return planner_node


def make_detail_enricher_node(llm: Any, detail_tools: list):
    """
    검색 완료 후 실행. 상세 정보(초록·저자 등)가 필요한지 LLM이 판단하여
    필요하면 get_details_by_ids를 호출하고, 충분하면 그대로 통과.
    """
    llm_with_detail = llm.bind_tools(detail_tools)

    def detail_enricher_node(state: AgentState) -> dict:
        from langchain_core.messages import AIMessage, HumanMessage
        from services.tool_result import merge_tool_results

        # 현재 턴 결과 없으면 직전 턴 결과로 fallback (멀티턴 후속 상세 질문 대응)
        tool_results = _extract_latest_tool_results(state["messages"])
        original_query = _get_original_query(state["messages"])
        numbered = numbered_search_context(tool_results) if tool_results else "(검색 결과 없음)"

        prompt = (
            f"사용자 질문: {original_query}\n\n"
            f"현재 검색 결과:\n{numbered}\n\n"
            f"상세 정보(초록, 저자, 전체 설명)가 더 필요하면 get_details_by_ids를 호출하고, "
            f"충분하다면 아무 도구도 호출하지 마세요."
        )

        try:
            response = llm_with_detail.invoke([
                SystemMessage(content=_ENRICHER_SYSTEM),
                HumanMessage(content=prompt),
            ])
        except Exception as e:
            logger.error("[DetailEnricher] LLM failed: %s — skipping enrichment", e)
            return {"messages": [AIMessage(content="")]}

        tool_calls = getattr(response, "tool_calls", []) or []
        logger.info(
            "[DetailEnricher] detail_calls_count=%d | session=%s",
            len(tool_calls), state.get("session_id", "?"),
        )
        return {"messages": [response]}

    return detail_enricher_node


def make_agent_node(llm: Any):
    """도구 결과 기반 최종 답변 생성 (Grounding Pass)."""
    def agent_node(state: AgentState) -> dict:
        # 현재 턴 결과가 없으면 최신 결과(보강된 정보 포함) 사용
        tool_results = _extract_latest_tool_results(state["messages"])
        original_query = _get_original_query(state["messages"])
        context = numbered_search_context(tool_results)

        system_msg = SystemMessage(content=_GROUNDING_SYSTEM)
        
        # 멀티턴 맥락 유지를 위해 히스토리 포함 (최근 10개 메시지)
        # 마지막 메시지는 아래 prompt에서 새로 구성하므로 제외
        history = state["messages"][-10:-1] if len(state["messages"]) > 1 else []
        prompt = f"사용자 질문: {original_query}\n\n[검색 결과]\n{context}"

        logger.info("[GroundingPass] query=%r | context_items=%d", original_query, len(tool_results))
        try:
            response = llm.invoke([system_msg] + list(history) + [HumanMessage(content=prompt)])
            if isinstance(getattr(response, "content", None), str):
                response = response.model_copy(update={"content": _safe_utf8(response.content)})
            return {"messages": [response]}
        except Exception as e:
            logger.error("[AgentNode] Grounding failed: %s", e)
            raise LLMError(f"Grounding pass failed: {e}") from e

    return agent_node



# ────────────────────────────────────────────────────────────────────────────
# Routing
# ────────────────────────────────────────────────────────────────────────────

def _make_planner_routing(max_calls: int = _DEFAULT_MAX_TOOL_CALLS):
    """Planner → SearchTools or DetailEnricher."""
    def _route(state: AgentState) -> str:
        limit = state.get("max_tool_calls") or max_calls
        sid = state.get("session_id", "?")

        if state.get("tool_call_count", 0) >= limit:
            logger.warning("[Planner] max_tool_calls=%d reached → enricher | session=%s", limit, sid)
            return "enricher"

        if state.get("empty_result_count", 0) >= 2:
            logger.warning("[Planner] empty_result_count>=2 → enricher (데이터 없음) | session=%s", sid)
            return "enricher"

        return "search_tools" if tools_condition(state) == "tools" else "enricher"
    return _route


def _route_enricher(state: AgentState) -> str:
    """DetailEnricher → DetailTools or Agent."""
    return "detail_tools" if tools_condition(state) == "tools" else "agent"


# ────────────────────────────────────────────────────────────────────────────
# Graph Builder
# ────────────────────────────────────────────────────────────────────────────

def build_graph(
    schema_registry: SchemaRegistry,
    llm: Any,
    tools: list,
    checkpointer=None,
    max_tool_calls: int = _DEFAULT_MAX_TOOL_CALLS,
) -> Any:
    """
    LangGraph 에이전트 그래프 빌드.
    tools는 [execute_dynamic_search, get_details_by_ids] 순서로 전달.
    """
    search_tools = [t for t in tools if t.name == "execute_dynamic_search"]
    detail_tools  = [t for t in tools if t.name == "get_details_by_ids"]

    planner_fn  = make_planner_node(llm, schema_registry, search_tools)
    enricher_fn = make_detail_enricher_node(llm, detail_tools)
    agent_fn    = make_agent_node(llm)

    search_tools_node = InstrumentedToolNode(search_tools)
    detail_tools_node = InstrumentedDetailToolNode(detail_tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("Planner",       planner_fn)
    workflow.add_node("SearchTools",   search_tools_node)
    workflow.add_node("DetailEnricher", enricher_fn)
    workflow.add_node("DetailTools",   detail_tools_node)
    workflow.add_node("Agent",         agent_fn)

    workflow.set_entry_point("Planner")

    workflow.add_conditional_edges(
        "Planner",
        _make_planner_routing(max_tool_calls),
        {"search_tools": "SearchTools", "enricher": "DetailEnricher"},
    )
    workflow.add_edge("SearchTools", "Planner")

    workflow.add_conditional_edges(
        "DetailEnricher",
        _route_enricher,
        {"detail_tools": "DetailTools", "agent": "Agent"},
    )
    workflow.add_edge("DetailTools", "Agent")
    workflow.add_edge("Agent", END)

    cp = checkpointer or MemorySaver()
    return workflow.compile(checkpointer=cp)


# ────────────────────────────────────────────────────────────────────────────
# Run helper
# ────────────────────────────────────────────────────────────────────────────

def run_query(app, query: str, session_id: str = "default") -> str:
    from langchain_core.messages import HumanMessage
    from common.config.settings import get_settings

    initial_state: AgentState = {
        "messages":          [HumanMessage(content=query)],
        "tool_call_count":   0,
        "empty_result_count": 0,
        "total_db_calls":    0,
        "session_id":        session_id,
        "start_time":        time.time(),
        "max_tool_calls":    _DEFAULT_MAX_TOOL_CALLS,
    }

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": get_settings().recursion_limit,
    }
    final_state = app.invoke(initial_state, config=config)

    for msg in reversed(final_state["messages"]):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            return msg.content

    return "(응답 없음)"
