"""
predict.py – POST /api/v1/predict  and  GET /api/v1/forecast/latest

POST /predict
    Body : PredictRequest  → PredictResponse
    Runs a full model forward pass and returns per-lead-time grid statistics,
    modality attention weights, physics gate stats, and latency.

GET /forecast/latest
    Convenience read-only alias: horizon=0, all sensors enabled.
    Returns NowcastSnapshot (same schema consumed by the React dashboard).
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from app.schemas.nowcast import (
    HorizonInfo,
    ModalityAttentionWeights,
    NowcastSnapshot,
    PhysicsGateStats,
    PredictRequest,
    PredictResponse,
)
from app.services.inference_service import HORIZON_DEFS, inference_service

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Run APM-Nowcast inference",
    description=(
        "Accepts sensor availability flags and an optional horizon index, "
        "runs a full APMNowcastModel forward pass, and returns per-lead-time "
        "grid statistics together with modality attention weights and "
        "inference latency."
    ),
)
async def predict(req: PredictRequest) -> PredictResponse:
    if not inference_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service not initialised. Call startup first.",
        )
    try:
        active_sensors = {
            "satellite": req.satellite,
            "radar": req.radar,
            "lightning": req.lightning,
            "nwp": req.nwp,
        }
        result = inference_service.run_inference(
            horizon_step=req.horizon,
            active_sensors=active_sensors,
            estimate_epistemic=req.estimate_epistemic,
            spatial_size=req.spatial_size,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        ) from exc

    horizon_def = HORIZON_DEFS[req.horizon]
    attn = result["attention_weights"]
    phys = result["physics_gate_stats"]

    return PredictResponse(
        horizon=HorizonInfo(**horizon_def),
        timestamp=datetime.now(timezone.utc).isoformat(),
        lightning_probs_mean=result["lightning_probs_mean"],
        lightning_probs_max=result["lightning_probs_max"],
        flash_rate_mean=result["flash_rate_mean"],
        flash_rate_max=result["flash_rate_max"],
        aleatoric_uncertainty_mean=result["aleatoric_uncertainty_mean"],
        epistemic_uncertainty_mean=result["epistemic_uncertainty_mean"],
        total_uncertainty_mean=result["total_uncertainty_mean"],
        attention_weights=ModalityAttentionWeights(**attn),
        physics_gate_stats=PhysicsGateStats(**phys),
        uncertainty_inflation_factor=result["uncertainty_inflation_factor"],
        inference_device=result["inference_device"],
        inference_ms=result["inference_ms"],
    )


@router.get(
    "/forecast/latest",
    response_model=NowcastSnapshot,
    summary="Latest nowcast snapshot (horizon 0, all sensors)",
    description=(
        "Returns the most recent APM-Nowcast snapshot at lead-time step 0 "
        "with all sensor modalities enabled.  Designed for dashboard cold-start."
    ),
)
async def forecast_latest() -> NowcastSnapshot:
    if not inference_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service not initialised.",
        )
    try:
        snapshot = inference_service.build_snapshot(
            horizon_step=0,
            active_sensors={"satellite": True, "radar": True, "lightning": True, "nwp": True},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Snapshot build failed: {exc}",
        ) from exc

    return NowcastSnapshot(**snapshot)
