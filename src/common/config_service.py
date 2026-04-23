"""
Config Service
- QueryConfig: 요청별 설정값 (API 파라미터 또는 DB 기본값)
- ConfigRepository: DB 설정 조회 프로토콜 (Mock → RDB 교체 가능)
- ConfigService: API 파라미터 > DB 기본값 순으로 최종값 결정
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class QueryConfig:
    """
    요청별 설정값. None인 필드는 ConfigService가 DB 기본값으로 채운다.

    Core settings:
        beam_width:           hop당 유지할 최대 노드 수 (BeamPruner)
        max_results:          최종 반환 결과 수
        vector_score_threshold: 벡터 유사도 최소 임계값 (0.0 ~ 1.0)

    LLM settings:
        model:                LLM 모델명 (서비스에 따라 다름: "qwen2.5:14b", "llama3", "mistral" 등)
        temperature:          LLM temperature
        max_tool_calls:       에이전트 도구 호출 최대 횟수
    """
    beam_width:             Optional[int]   = None
    max_results:            Optional[int]   = None
    vector_score_threshold: Optional[float] = None
    model:                  Optional[str]   = None
    temperature:            Optional[float] = None
    max_tool_calls:         Optional[int]   = None


@runtime_checkable
class ConfigRepository(Protocol):
    def get(self, key: str) -> Any: ...


class ConfigService:
    """
    API 파라미터 우선, 없으면 DB(ConfigRepository)에서 읽어 최종 QueryConfig 반환.
    """

    def __init__(self, repo: ConfigRepository):
        self._repo = repo

    def resolve(self, override: Optional[QueryConfig] = None) -> QueryConfig:
        """override의 None 필드를 repo 기본값으로 채워 반환."""
        o = override or QueryConfig()

        def _pick(key: str, api_val):
            return api_val if api_val is not None else self._repo.get(key)

        return QueryConfig(
            beam_width             = _pick("beam_width",             o.beam_width),
            max_results            = _pick("max_results",            o.max_results),
            vector_score_threshold = _pick("vector_score_threshold", o.vector_score_threshold),
            model                  = _pick("model",                  o.model),
            temperature            = _pick("temperature",            o.temperature),
            max_tool_calls         = _pick("max_tool_calls",         o.max_tool_calls),
        )

    def get_default(self, key: str) -> Any:
        return self._repo.get(key)
