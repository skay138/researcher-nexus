"""
E2E 검증 스크립트 — 6개 테스트 질문 순차 실행
각 질문 응답 완료 후 다음 질문 진행. 질문 6은 질문 5 세션 이어받아 multi-turn 검증.

사용법:
    python scripts/test_questions.py
    python scripts/test_questions.py --url http://localhost:5000
    python scripts/test_questions.py --question 4          # 특정 번호만
    python scripts/test_questions.py --from 3              # 3번부터
"""

from __future__ import annotations
import argparse
import json
import sys
import time
import uuid

try:
    import requests
except ImportError:
    print("[ERROR] requests 라이브러리가 없습니다: pip install requests")
    sys.exit(1)

QUESTIONS = [
    {
        "no": 1,
        "label": "단순 벡터 검색 (Milvus only)",
        "query": "자율운항선박 관련 연구를 하는 기관이 어디야?",
        "new_session": True,
    },
    {
        "no": 2,
        "label": "단일 홉 (Neo4j 관계 탐색)",
        "query": "KRISO 소속 연구자들이 참여한 프로젝트는?",
        "new_session": True,
    },
    {
        "no": 3,
        "label": "멀티홉 (2~3홉)",
        "query": "해양 관련 프로젝트에 참여한 연구자들이 쓴 논문 제목 알려줘",
        "new_session": True,
    },
    {
        "no": 4,
        "label": "상세 조회 유발 (MariaDB / get_details_by_ids)",
        "query": "자율운항 관련 논문의 초록과 저자 목록을 알려줘",
        "new_session": True,
    },
    {
        "no": 5,
        "label": "연도 필터 (vector_search_filters)",
        "query": "2024년 이후 등록된 수소 관련 특허를 찾아줘",
        "new_session": True,
    },
    {
        "no": 6,
        "label": "후속 질문 (multi-turn — 질문 5 세션 재사용)",
        "query": "그 중 첫 번째 특허의 특허번호와 상세 내용 알려줘",
        "new_session": False,  # 질문 5의 session_id 재사용
    },
]

# ── ANSI 색상 ────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
GREEN  = "\033[32m"
RED    = "\033[31m"
GRAY   = "\033[90m"
BLUE   = "\033[34m"


def _sep(char: str = "─", width: int = 70) -> str:
    return char * width


# ── SSE 스트리밍 ─────────────────────────────────────────────────────────────

def stream_query(endpoint: str, query: str, session_id: str, timeout: int = 120):
    """
    SSE 스트림을 읽어 이벤트를 순서대로 yield.
    반환: (event_type, data_dict)
    """
    payload = {"query": query, "session_id": session_id}
    try:
        with requests.post(endpoint, json=payload, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if raw_line.startswith("data:"):
                    data_str = raw_line[5:].lstrip(" ")
                    if not data_str:
                        continue
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    yield data.get("type", "unknown"), data
    except requests.ConnectionError:
        yield "error", {"message": f"서버에 연결할 수 없습니다: {endpoint}"}
    except requests.Timeout:
        yield "error", {"message": f"타임아웃 ({timeout}s)"}
    except requests.HTTPError as e:
        yield "error", {"message": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}


# ── 단일 질문 실행 ────────────────────────────────────────────────────────────

def run_question(endpoint: str, q: dict, session_id: str) -> tuple[bool, str]:
    """
    질문 하나를 실행하고 결과를 출력.
    반환: (성공여부, 최종답변)
    """
    print(f"\n{BOLD}{CYAN}{'━' * 70}{RESET}")
    print(f"{BOLD}{CYAN}Q{q['no']}. {q['label']}{RESET}")
    print(f"{BOLD}질문: {q['query']}{RESET}")
    print(f"{GRAY}session_id: {session_id}{RESET}")
    print(_sep())

    tokens: list[str] = []
    tool_calls: list[str] = []
    sources: list[dict] = []
    start = time.time()
    success = False
    answer = ""

    for ev_type, data in stream_query(endpoint, q["query"], session_id):
        if ev_type == "tool_call":
            tool_name = data.get("tool", "?")
            args = data.get("args", [])
            tool_calls.append(tool_name)
            print(f"  {YELLOW}[Tool] {tool_name}({', '.join(args)}){RESET}")

        elif ev_type == "token":
            content = data.get("content", "")
            tokens.append(content)
            print(content, end="", flush=True)

        elif ev_type == "done":
            answer = data.get("answer", "")
            sources = data.get("sources", [])
            success = True
            if not tokens:
                # 토큰 스트리밍 없이 done만 온 경우
                print(answer, end="")

        elif ev_type == "error":
            print(f"\n{RED}[ERROR] {data.get('message', '알 수 없는 오류')}{RESET}")
            break

    elapsed = time.time() - start

    # ── 결과 요약 ─────────────────────────────────────────────────────────────
    print(f"\n{_sep()}")
    status = f"{GREEN}✓ 성공{RESET}" if success else f"{RED}✗ 실패{RESET}"
    print(f"  상태: {status}  |  소요: {elapsed:.1f}s  |  도구 호출: {', '.join(tool_calls) or '없음'}")

    if sources:
        print(f"  출처 ({len(sources)}개):")
        for s in sources[:5]:
            score = f"  score={s.get('score', 0):.3f}" if s.get('score') is not None else ""
            print(f"    [{s['no']}] {s.get('type','?')} | {s.get('name','')[:50]}{score}")
        if len(sources) > 5:
            print(f"    ... 외 {len(sources)-5}개")

    # ── 검증 포인트 자동 판정 ─────────────────────────────────────────────────
    _check(q["no"], tool_calls, sources, answer)

    return success, answer


def _check(no: int, tool_calls: list[str], sources: list[dict], answer: str) -> None:
    """각 질문별 예상 동작 자동 판정."""
    checks: list[tuple[bool, str]] = []

    if no == 1:
        checks = [
            ("execute_dynamic_search" in tool_calls, "execute_dynamic_search 호출됨"),
            ("get_details_by_ids" not in tool_calls,  "get_details_by_ids 미호출 (Milvus only)"),
            (bool(sources),                            "결과 있음"),
        ]
    elif no == 2:
        checks = [
            ("execute_dynamic_search" in tool_calls, "execute_dynamic_search 호출됨"),
            (bool(sources),                           "결과 있음"),
        ]
    elif no == 3:
        checks = [
            ("execute_dynamic_search" in tool_calls, "execute_dynamic_search 호출됨"),
            (any(s.get("type") == "Paper" for s in sources), "Paper 타입 결과 포함"),
        ]
    elif no == 4:
        checks = [
            ("execute_dynamic_search" in tool_calls,  "execute_dynamic_search 호출됨"),
            ("get_details_by_ids" in tool_calls,       "get_details_by_ids 호출됨 (MariaDB)"),
            ("초록" in answer or "abstract" in answer.lower() or "저자" in answer,
             "초록 또는 저자 정보 포함"),
        ]
    elif no == 5:
        checks = [
            ("execute_dynamic_search" in tool_calls, "execute_dynamic_search 호출됨"),
            (any(s.get("type") == "Patent" for s in sources), "Patent 타입 결과 포함"),
        ]
    elif no == 6:
        checks = [
            ("get_details_by_ids" in tool_calls, "get_details_by_ids 호출됨 (multi-turn MariaDB)"),
            (bool(answer),                        "응답 있음"),
        ]

    if checks:
        print(f"  {BLUE}[검증]{RESET}")
        for passed, label in checks:
            mark = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
            print(f"    {mark} {label}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ResearchNexus E2E 테스트")
    parser.add_argument("--url", default="http://localhost:5000", help="API base URL")
    parser.add_argument("--question", type=int, help="특정 질문 번호만 실행")
    parser.add_argument("--from", dest="from_no", type=int, default=1, help="N번부터 실행")
    parser.add_argument("--timeout", type=int, default=120, help="질문당 타임아웃(초)")
    args = parser.parse_args()

    endpoint = f"{args.url}/api/v1/agent/query"

    # 헬스체크
    try:
        r = requests.get(f"{args.url}/api/v1/health", timeout=5)
        r.raise_for_status()
        print(f"{GREEN}서버 정상: {args.url}{RESET}")
    except Exception as e:
        print(f"{RED}서버 응답 없음 ({e}). 계속 진행합니다.{RESET}")

    # 실행할 질문 필터링
    questions = QUESTIONS
    if args.question:
        questions = [q for q in QUESTIONS if q["no"] == args.question]
    else:
        questions = [q for q in QUESTIONS if q["no"] >= args.from_no]

    if not questions:
        print(f"{RED}해당하는 질문이 없습니다.{RESET}")
        sys.exit(1)

    # 세션 관리 — 질문 5와 6은 같은 세션 공유
    session_map: dict[int, str] = {}
    prev_session_id: str | None = None

    total_start = time.time()
    results: list[tuple[int, bool]] = []

    for q in questions:
        if q["new_session"]:
            sid = f"test-q{q['no']}-{uuid.uuid4().hex[:8]}"
        else:
            # new_session=False → 직전 질문의 session_id 재사용 (multi-turn)
            sid = prev_session_id or f"test-q{q['no']}-{uuid.uuid4().hex[:8]}"

        session_map[q["no"]] = sid
        success, _ = run_question(endpoint, q, sid)
        results.append((q["no"], success))
        prev_session_id = sid

    # 전체 요약
    total_elapsed = time.time() - total_start
    print(f"\n{BOLD}{_sep('═')}{RESET}")
    print(f"{BOLD}전체 결과 ({total_elapsed:.1f}s){RESET}")
    passed = sum(1 for _, ok in results if ok)
    print(f"  {GREEN}{passed}개 성공{RESET} / {RED}{len(results)-passed}개 실패{RESET} / 총 {len(results)}개")
    for no, ok in results:
        mark = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        q_label = next(q["label"] for q in QUESTIONS if q["no"] == no)
        print(f"  {mark} Q{no}. {q_label}")
    print(_sep('═'))


if __name__ == "__main__":
    main()
