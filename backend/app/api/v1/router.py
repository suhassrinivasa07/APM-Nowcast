from fastapi import APIRouter

from app.api.v1.endpoints import benchmarks, explain, health, nowcast, predict

api_router = APIRouter()

# Health (keep existing tag)
api_router.include_router(health.router, tags=["Health"])

# Nowcast inference & snapshot
api_router.include_router(predict.router, tags=["Nowcast"])
api_router.include_router(nowcast.router, tags=["Nowcast"])

# Explainability & scenario simulation
api_router.include_router(explain.router, tags=["Explainability"])

# Model verification benchmarks
api_router.include_router(benchmarks.router, tags=["Benchmarks"])
