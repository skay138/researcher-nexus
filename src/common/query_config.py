"""
Config Service
- QueryConfig:      요청별 설정값 (API 파라미터 또는 DB 기본값)
- ConfigRepository: DB 설정 조회 프로토콜 (Mock → RDB 교체 가능)
- RequestConfig:    현재 요청의 설정값 접근자 (ContextVar 기반)
                    _resolve(repo, override) → API 파라미터 > repo > 내장 기본값 순
                    asyncio ContextVar로 전파되므로 동기 도구(ToolNode)까지 접근 가능
"""

from __future__ import annotations
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Protocol, runtime_checkable


@dataclass
class QueryConfig:
    """
    요청별 설정값. None인 필드는 _resolve()가 repo/내장 기본값으로 채운다.

    Core settings:
        beam_width:  hop당 유지할 최대 노드 수 (BeamPruner)
        max_results: 최종 반환 결과 수

    LLM settings:
        model:          LLM 모델명 ("qwen2.5:14b", "llama3", "mistral" 등)
        temperature:    LLM temperature
        max_tool_calls: 에이전트 도구 호출 최대 횟수
    """
    beam_width:          Optional[int]   = None
    max_results:         Optional[int]   = None
    sparse_weight:       Optional[float] = None  # hybrid search BM25 가중치
    dense_weight:        Optional[float] = None  # hybrid search dense 가중치
    entry_min_score:     Optional[float] = None  # entry search 절대 하한
    entry_score_ratio:   Optional[float] = None  # entry search top 대비 비율 하한
    model:               Optional[str]   = None
    temperature:            Optional[float] = None
    max_tool_calls:         Optional[int]   = None


@runtime_checkable
class ConfigRepository(Protocol):
    def get(self, key: str) -> Any: ...


# ────────────────────────────────────────────────────────────────────────────
# RequestConfig — 요청별 설정 접근자
# ────────────────────────────────────────────────────────────────────────────

class RequestConfig:
    """
    현재 요청의 설정값 접근자.
    API 파라미터 > ConfigRepository(repo) > 내장 기본값 순으로 반환.

    앱 시작 시 (app_factory):
        repo = make_config_repo()
        beam_width = RequestConfig._resolve(repo).beam_width

    요청 시작 시 (search.py):
        resolved = RequestConfig._resolve(repo, api_override)
        RequestConfig.set_current(resolved, original_query=body.query)

    도구 / 엔진에서:
        cfg = RequestConfig.current()
        cfg.max_results      # int, 항상 non-None
        cfg.original_query   # str
        cfg.to_query_config() # ExecutionEngine 호환용 QueryConfig
    """

    # config_repository 기본값과 동기화 유지
    _DEFAULTS: ClassVar[Dict[str, Any]] = {
        "beam_width":        50,
        "max_results":       20,
        "sparse_weight":     0.3,
        "dense_weight":      1.0,
        "entry_min_score":   0.2,
        "entry_score_ratio": 0.5,
        "model":             "qwen2.5:14b",
        "temperature":       0.0,
        "max_tool_calls":    3,
    }

    _ctx: ClassVar[ContextVar[Optional["RequestConfig"]]] = ContextVar(
        "_request_config_ctx", default=None
    )

    def __init__(
        self,
        resolved: Optional[QueryConfig] = None,
        original_query: str = "",
    ) -> None:
        self._resolved       = resolved
        self._original_query = original_query

    # ── 해석 ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve(
        repo:     Optional[ConfigRepository],
        override: Optional[QueryConfig] = None,
    ) -> QueryConfig:
        """API 파라미터 > repo > 내장 기본값 순으로 완전한 QueryConfig 반환."""
        o = override or QueryConfig()

        def _pick(key: str, api_val: Any) -> Any:
            if api_val is not None:
                return api_val
            if repo is not None:
                val = repo.get(key)
                if val is not None:
                    return val
            return RequestConfig._DEFAULTS.get(key)

        return QueryConfig(
            beam_width        = _pick("beam_width",        o.beam_width),
            max_results       = _pick("max_results",       o.max_results),
            sparse_weight     = _pick("sparse_weight",     o.sparse_weight),
            dense_weight      = _pick("dense_weight",      o.dense_weight),
            entry_min_score   = _pick("entry_min_score",   o.entry_min_score),
            entry_score_ratio = _pick("entry_score_ratio", o.entry_score_ratio),
            model             = _pick("model",             o.model),
            temperature       = _pick("temperature",       o.temperature),
            max_tool_calls    = _pick("max_tool_calls",    o.max_tool_calls),
        )

    # ── 클래스 메서드 ─────────────────────────────────────────────────────────

    @classmethod
    def set_current(
        cls,
        resolved: Optional[QueryConfig],
        original_query: str = "",
    ) -> None:
        """요청 시작 시 호출. asyncio ContextVar로 동기 ToolNode까지 자동 전파."""
        cls._ctx.set(cls(resolved, original_query))

    @classmethod
    def current(cls) -> "RequestConfig":
        """현재 요청의 RequestConfig 반환. 미설정 시 기본값만 있는 인스턴스 반환."""
        inst = cls._ctx.get()
        return inst if inst is not None else cls()

    # ── 설정값 접근 ───────────────────────────────────────────────────────────

    def get(self, key: str, fallback: Any = None) -> Any:
        """resolved > 내장 기본값 순으로 반환."""
        if self._resolved is not None:
            val = getattr(self._resolved, key, None)
            if val is not None:
                return val
        default = self._DEFAULTS.get(key)
        return default if default is not None else fallback

    @property
    def max_results(self) -> int:
        return self.get("max_results")

    @property
    def beam_width(self) -> int:
        return self.get("beam_width")

    @property
    def sparse_weight(self) -> float:
        return self.get("sparse_weight")

    @property
    def dense_weight(self) -> float:
        return self.get("dense_weight")

    @property
    def entry_min_score(self) -> float:
        return self.get("entry_min_score")

    @property
    def entry_score_ratio(self) -> float:
        return self.get("entry_score_ratio")

    @property
    def max_tool_calls(self) -> int:
        return self.get("max_tool_calls")

    @property
    def model(self) -> str:
        return self.get("model")

    @property
    def temperature(self) -> float:
        return self.get("temperature")

    @property
    def original_query(self) -> str:
        return self._original_query

    def to_query_config(self) -> Optional[QueryConfig]:
        """ExecutionEngine.run(config=...) 호환용 QueryConfig 반환."""
        return self._resolved
