"""
Cache Backend Abstraction
- Redis 전용 백엔드
- 연결 실패 시 즉시 예외 발생 (silent fallback 없음)
"""

from __future__ import annotations
import hashlib
import json
import logging
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class CacheBackend(Protocol):
    def get(self, key: str) -> Optional[Any]: ...
    def set(self, key: str, value: Any) -> None: ...
    def clear(self) -> None: ...


# ────────────────────────────────────────────────────────────────────────────
# Redis Cache
# ────────────────────────────────────────────────────────────────────────────

class RedisCache:
    """Redis 백엔드 캐시. pickle 직렬화 사용."""

    def __init__(self, redis_url: str, ttl: float = 300.0):
        import pickle
        self._pickle = pickle
        self._ttl = int(ttl)
        self._client = self._connect(redis_url)

    def _connect(self, url: str):
        try:
            import redis
            client = redis.from_url(url, decode_responses=False)
            client.ping()
            logger.info("Redis cache connected: %s", url)
            return client
        except Exception as e:
            raise ConnectionError(f"Redis connection failed: {e}") from e

    def get(self, key: str) -> Optional[Any]:
        try:
            raw = self._client.get(key)
            return self._pickle.loads(raw) if raw is not None else None
        except Exception as e:
            logger.warning("Redis get failed for key '%s': %s", key, e)
            return None

    def set(self, key: str, value: Any) -> None:
        try:
            self._client.setex(key, self._ttl, self._pickle.dumps(value))
        except Exception as e:
            logger.warning("Redis set failed for key '%s': %s", key, e)

    def clear(self) -> None:
        try:
            self._client.flushdb()
        except Exception as e:
            logger.warning("Redis clear failed: %s", e)


# ────────────────────────────────────────────────────────────────────────────
# Factory
# ────────────────────────────────────────────────────────────────────────────

def make_cache(
    redis_url: str = "redis://localhost:6379",
    ttl: float = 300.0,
) -> CacheBackend:
    """Redis 캐시 백엔드 생성."""
    return RedisCache(redis_url=redis_url, ttl=ttl)


# ────────────────────────────────────────────────────────────────────────────
# Cache Key Helper (execution_engine에서 재사용)
# ────────────────────────────────────────────────────────────────────────────

def make_cache_key(prefix: str, data: Any) -> str:
    serialized = json.dumps(data, sort_keys=True, default=str)
    h = hashlib.sha256(serialized.encode()).hexdigest()[:16]
    return f"{prefix}:{h}"
