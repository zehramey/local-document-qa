from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.error_handlers import register_exception_handlers
from app.api.health import router as health_router
from app.api.models import router as models_router
from app.api.questions import router as questions_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name)
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(questions_router)
app.include_router(models_router)
register_exception_handlers(app)
