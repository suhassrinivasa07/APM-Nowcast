"""
nowcast.py – GET /api/v1/nowcast/snapshot

Returns a full NowcastSnapshot for a given lead-time horizon and sensor
availability configuration.
"""
from __future__ import annotations

from typing import Any, Dict
from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.nowcast import NowcastSnapshot
from app.services.inference_service import inference_service

router = APIRouter()

# In-memory snapshot cache
_snapshots: Dict[str, Dict[str, Any]] = {}


@router.get(
    "/nowcast/snapshot",
    response_model=NowcastSnapshot,
    summary="Nowcast snapshot at a given lead-time horizon",
    description=(
        "Returns the APM-Nowcast prediction snapshot for the requested lead-time "
        "horizon with per-modality sensor availability flags. When no snapshot exists "
        "yet, automatically runs a baseline inference pass via inference_service "
        "(using the default pan-India atmospheric cube) and returns 200 OK."
    ),
)
async def nowcast_snapshot(
    horizon: int = Query(0, ge=0, le=3, description="Lead-time step index (0=+15 min … 3=+60 min)"),
    satellite: bool = Query(True, description="Satellite IR/WV modality available"),
    radar: bool = Query(True, description="Radar DWR modality available"),
    lightning: bool = Query(True, description="Lightning history modality available"),
    nwp: bool = Query(True, description="NWP/ERA5 thermodynamics modality available"),
) -> NowcastSnapshot:
    # Ensure inference service is initialised
    if not inference_service.is_ready:
        inference_service.initialize()

    key = f"h{horizon}_sat{satellite}_rad{radar}_lgt{lightning}_nwp{nwp}"
    snapshot_dict = _snapshots.get(key)

    # When no snapshot exists yet, automatically run a baseline inference pass
    # via inference_service using the default pan-India atmospheric cube
    if snapshot_dict is None:
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
            _snapshots[key] = snapshot_dict
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Baseline inference pass failed: {exc}",
            ) from exc

    return NowcastSnapshot(**snapshot_dict)
