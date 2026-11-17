"""
explain.py – POST /api/v1/explain/modalities
            POST /api/v1/scenarios/dropout

POST /explain/modalities
    Returns per-modality attention weights + thermodynamic profile for a given
    sensor availability configuration.

POST /scenarios/dropout
    Simulates sensor-dropout scenarios: when a modality flag is False its
    attention weight is forced to 0.0 and the remaining weights are
    renormalised so they sum to exactly 1.0.  Returns the adjusted snapshot.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, HTTPException, status

from app.schemas.nowcast import (
    ConvectiveStormCell,
    DropoutScenarioRequest,
    DropoutScenarioResponse,
    ExplainModalitiesRequest,
    ExplainModalitiesResponse,
    HorizonInfo,
    ModalityAttentionWeights,
    NowcastSnapshot,
    PhysicsGateStats,
    SevereAlert,
)
from app.services.inference_service import (
    HORIZON_DEFS,
    _compute_attention_and_inflation,
    inference_service,
)

router = APIRouter()

_MODALITY_DESCRIPTIONS = {
    "satellite": "INSAT-3D/3DS geostationary imagery (TIR-1, WV, VIS) — primary convective initiation signal",
    "radar":     "IMD DWR S/C-band network — dominant reflectivity and storm-cell tracking signal",
    "lightning": "NRSC ground LF/VHF sensor network — direct proxy for electrification and updraft intensity",
    "nwp":       "ERA5 reanalysis fields (CAPE, CIN, shear, moisture convergence) — thermodynamic instability context",
}


@router.post(
    "/explain/modalities",
    response_model=ExplainModalitiesResponse,
    summary="Explain per-modality attention weights",
    description=(
        "Runs the APMNowcastModel with the given sensor flags and returns "
        "per-modality attention weights, physics gate statistics, and the "
        "thermodynamic state at the requested lead-time horizon."
    ),
)
async def explain_modalities(req: ExplainModalitiesRequest) -> ExplainModalitiesResponse:
    if not inference_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service not initialised.",
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
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explain failed: {exc}",
        ) from exc

    horizon_def = HORIZON_DEFS[req.horizon]
    attn = result["attention_weights"]
    phys = result["physics_gate_stats"]

    return ExplainModalitiesResponse(
        horizon=HorizonInfo(**horizon_def),
        timestamp=datetime.now(timezone.utc).isoformat(),
        attention_weights=ModalityAttentionWeights(**attn),
        physics_gate_stats=PhysicsGateStats(**phys),
        uncertainty_inflation_factor=result["uncertainty_inflation_factor"],
        modality_descriptions=_MODALITY_DESCRIPTIONS,
    )


@router.post(
    "/scenarios/dropout",
    response_model=DropoutScenarioResponse,
    summary="Simulate sensor-dropout scenario",
    description=(
        "Forces dropped modalities to attention_weight=0.0 and renormalises "
        "the remaining weights so they sum to exactly 1.0. "
        "Returns the adjusted NowcastSnapshot plus a delta_csi scalar "
        "estimating skill degradation from the dropout."
    ),
)
async def scenarios_dropout(req: DropoutScenarioRequest) -> DropoutScenarioResponse:
    if not inference_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service not initialised.",
        )
    try:
        sensor_flags = {
            "satellite": req.satellite,
            "radar": req.radar,
            "lightning": req.lightning,
            "nwp": req.nwp,
        }
        # Compute dropout-masked attention weights
        raw_weights, inflation = _compute_attention_and_inflation(sensor_flags)

        # Build the full snapshot using the dropout-adjusted sensor set
        snapshot_dict = inference_service.build_snapshot(
            horizon_step=req.horizon,
            active_sensors=sensor_flags,
        )

        dropped = [k for k, v in sensor_flags.items() if not v]
        horizon_def = HORIZON_DEFS[req.horizon]
        dropped_str = ", ".join(dropped) if dropped else "none"

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dropout scenario failed: {exc}",
        ) from exc

    cells = [ConvectiveStormCell(**c) for c in snapshot_dict["cells"]]
    alerts = [SevereAlert(**a) for a in snapshot_dict["alerts"]]

    return DropoutScenarioResponse(
        scenario=f"Sensor dropout: {dropped_str}",
        horizon=HorizonInfo(**horizon_def),
        timestamp=datetime.now(timezone.utc).isoformat(),
        attention_weights=ModalityAttentionWeights(**raw_weights),
        uncertainty_inflation_factor=round(inflation, 2),
        cells=cells,
        alerts=alerts,
    )
