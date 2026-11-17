"""
benchmarks.py – GET /api/v1/benchmarks/metrics
               GET /api/v1/verification/benchmarks   (frontend alias)

Both routes return List[BenchmarkModelMetrics] – the full comparison table
shown in the dashboard's Model Benchmark panel.
"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter

from app.schemas.nowcast import BenchmarkModelMetrics
from app.services.inference_service import inference_service

router = APIRouter()


@router.get(
    "/benchmarks/metrics",
    response_model=List[BenchmarkModelMetrics],
    summary="Model benchmark metrics",
    description=(
        "Returns the multi-model verification metrics table (CSI, POD, FAR, FSS, "
        "Brier score, ECE, flash RMSE) across the four standard lead-time horizons."
    ),
)
async def benchmarks_metrics() -> List[BenchmarkModelMetrics]:
    return [BenchmarkModelMetrics(**row) for row in inference_service.get_benchmark_data()]


@router.get(
    "/verification/benchmarks",
    response_model=List[BenchmarkModelMetrics],
    summary="Verification benchmark metrics (frontend alias)",
    description=(
        "Alias of /benchmarks/metrics consumed by the React dashboard's "
        "Model Benchmark panel."
    ),
)
async def verification_benchmarks() -> List[BenchmarkModelMetrics]:
    return [BenchmarkModelMetrics(**row) for row in inference_service.get_benchmark_data()]
