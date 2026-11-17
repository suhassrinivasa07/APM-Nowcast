"""
nowcast.py – GET /api/v1/nowcast/snapshot

Returns a full NowcastSnapshot for a given lead-time horizon and sensor
availability configuration.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.nowcast import NowcastSnapshot
from app.services.inference_service import inference_service

router = APIRouter()


@router.get(
    "/nowcast/snapshot",
    response_model=NowcastSnapshot,
    summary="Nowcast snapshot at a given lead-time horizon",
    description=(
        "Returns the APM-Nowcast prediction snapshot for the requested lead-time "
        "horizon with per-modality sensor availability flags."
    ),
)
async def nowcast_snapshot(
    horizon: int = Query(0, ge=0, le=3, description="Lead-time step index (0=+15 min … 3=+60 min)"),
    satellite: bool = Query(True, description="Satellite IR/WV modality available"),
    radar: bool = Query(True, description="Radar DWR modality available"),
    lightning: bool = Query(True, description="Lightning history modality available"),
    nwp: bool = Query(True, description="NWP/ERA5 thermodynamics modality available"),
) -> NowcastSnapshot:
    if not inference_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service not initialised.",
        )
    try:
        snapshot_dict = inference_service.build_snapshot(
            horizon_step=horizon,
            active_sensors={
                "satellite": satellite,
                "radar": radar,
                "lightning": lightning,
                "nwp": nwp,
            },
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Snapshot build failed: {exc}",
        ) from exc

    return NowcastSnapshot(**snapshot_dict)
