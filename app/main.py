"""
Application entry point.

개발 실행:
    python -m app.main
    uv run uvicorn app.main:app --reload

프로덕션:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from app.api.factory import create_fastapi_app
from app.common.config.settings import get_settings

settings = get_settings()
app = create_fastapi_app(settings)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=(settings.environment == "development"),
        log_level=settings.log_level.lower(),
    )
