"""
main.py – FastAPI application entry-point for APM-Nowcast (Module 10 update).

Changes from previous version:
  • Added FastAPI lifespan context manager that initialises the singleton
    NowcastInferenceService on startup (honours CHECKPOINT_DIR env-var).
  • Registered all new v1 routers (predict, nowcast, explain, benchmarks).
  • CORS already includes http://localhost:5173 via settings.cors_origins.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.schemas.health import HealthResponse
from app.services.inference_service import inference_service


# ---------------------------------------------------------------------------
# Lifespan: initialise heavy singletons once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the NowcastInferenceService (loads or builds the model)."""
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", None)
    inference_service.initialize(checkpoint_dir=checkpoint_dir)
    yield
    # Teardown: nothing required for in-process PyTorch model


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "An Uncertainty-Aware Physics-Guided Adaptive Multimodal Deep Learning "
        "Framework for Thunderstorm Initiation and Lightning Nowcasting over India."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Root-level /health (kept for backwards compatibility)
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Root Health Check",
    description="Direct root-level health endpoint verifying service liveness and config.",
)
async def root_health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        timestamp=datetime.now(timezone.utc),
        environment=settings.APP_ENV,
        device_target=settings.DEVICE,
    )


# ---------------------------------------------------------------------------
# Mount versioned API
# ---------------------------------------------------------------------------
app.include_router(api_router, prefix=settings.API_V1_STR)


# ---------------------------------------------------------------------------
# Root info
# ---------------------------------------------------------------------------
@app.get("/", tags=["Root"])
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
