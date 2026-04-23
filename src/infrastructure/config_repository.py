"""
Config Repository
- MockConfigRepository: RDB 없이 동작하는 인메모리 구현 (개발/테스트)
- RDB 연동 준비 시 RDBConfigRepository로 교체 예정

기본값은 이전에 .env로 관리하던 core/LLM 설정값.
인프라 설정(DB URI 등)은 common/settings.py(환경변수)에서 유지.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

_DEFAULTS: Dict[str, Any] = {
    # Core settings
    "beam_width":             50,
    "max_results":            20,
    "vector_score_threshold": 0.6,
    # LLM settings
    "model":                  "qwen2.5:14b",
    "temperature":            0.0,
    "max_tool_calls":         3,
}


class MockConfigRepository:
    """
    인메모리 설정 저장소.
    실제 RDB 연동 전까지 기본값을 제공하며, 런타임 변경도 지원.
    """

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        self._store: Dict[str, Any] = {**_DEFAULTS, **(overrides or {})}

    def get(self, key: str) -> Any:
        return self._store.get(key)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def all(self) -> Dict[str, Any]:
        return dict(self._store)
