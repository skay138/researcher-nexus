"""
Application Settings
- pydantic-settings 기반 중앙 설정 관리
- 우선순위: OS 환경 변수 > .env 파일 > 코드 기본값
- LLM 설정(model, temperature 등)은 API 파라미터 또는 ConfigRepository(Mock/RDB)로 관리
"""

from __future__ import annotations
from typing import Literal, Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseSettings  # type: ignore[no-redef]
    SettingsConfigDict = None  # type: ignore[assignment]


class Settings(BaseSettings):
    # ── LLM (연결 정보만 — 모델·파라미터는 ConfigRepository에서 관리) ──
    # 지원 provider: ollama | vllm | triton
    llm_provider: str = "ollama"
    llm_base_url: str = "http://localhost:11434"

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = ""
    neo4j_password: str = ""

    # ── Milvus ────────────────────────────────────────────────────────────────
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    # ── Cache ──────────────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: float = 300.0

    # ── Execution ─────────────────────────────────────────────────────────────
    recursion_limit: int = 10
    sentence_transformer_model: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

    # ── Application ───────────────────────────────────────────────────────────
    use_mock: bool = True
    log_level: str = "INFO"
    environment: Literal["development", "production"] = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 5000
    api_cors_origins: list[str] = ["*"]

    if SettingsConfigDict is not None:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
        )
    else:
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False


# 싱글턴 인스턴스 (앱 전역 공유)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """싱글턴 Settings 인스턴스 반환. 테스트에서는 직접 Settings()로 생성."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """테스트용 설정 초기화."""
    global _settings
    _settings = None
