"""
Config Repository
- MemoryConfigRepository: RDB 없이 동작하는 인메모리 구현 (개발/프로덕션 기본값)
- RDB 연동 준비 시 RDBConfigRepository로 교체 예정

기본값은 이전에 .env로 관리하던 core/LLM 설정값.
인프라 설정(DB URI 등)은 common/settings.py(환경변수)에서 유지.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

_DEFAULTS: Dict[str, Any] = {
    # Core settings
    "beam_width":        50,
    "max_results":       20,
    # Hybrid search weights
    "sparse_weight":     0.3,
    "dense_weight":      1.0,
    # Entry search score filtering
    "entry_min_score":   0.2,
    "entry_score_ratio": 0.5,
    # LLM settings
    "model":             "qwen2.5:14b",
    "temperature":       0.0,
    "max_tool_calls":    3,
}


class MemoryConfigRepository:
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
