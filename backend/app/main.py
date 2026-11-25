from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.admin import router as admin_router
from app.api.auth import router as auth_router
from app.api.models import router as models_router
from app.core.config import settings
from app.services.model_adapters import register_builtin_adapters

# Register only application-owned adapters. User input cannot register code.
register_builtin_adapters()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Apply schema changes through reviewed Alembic migrations, not create_all.
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.3.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.app_env != "production" else None,
    redoc_url=None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-CSRF-Token"],
)
app.include_router(auth_router, prefix=settings.api_v1_prefix)
app.include_router(admin_router, prefix=settings.api_v1_prefix)
app.include_router(models_router, prefix=settings.api_v1_prefix)


@app.get("/health", tags=["operations"])
def health():
    return {"status": "ok", "environment": settings.app_env}
