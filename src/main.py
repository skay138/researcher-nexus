"""
Application entry point.

개발 실행:
    python src/main.py
    $env:PYTHONPATH="src"; uvicorn src.main:app --reload  (Windows)
    PYTHONPATH=src uvicorn src.main:app --reload          (Mac/Linux)

프로덕션:
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from api.app import create_fastapi_app
from common.settings import get_settings

settings = get_settings()
app = create_fastapi_app(settings)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=(settings.environment == "development"),
        log_level=settings.log_level.lower(),
    )
