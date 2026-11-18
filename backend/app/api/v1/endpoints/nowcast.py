"""
nowcast.py – GET /api/v1/nowcast/snapshot

Returns a full NowcastSnapshot for a given lead-time horizon and sensor
availability configuration.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Tuple
from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.nowcast import NowcastSnapshot
from app.services.inference_service import inference_service
from app.services.live_weather import live_weather_service

router = APIRouter()

# In-memory snapshot cache with 10-minute TTL (matches Open-Meteo NWP cache TTL)
_snapshots: Dict[str, Tuple[float, Dict[str, Any]]] = {}
SNAPSHOT_TTL = 600.0


@router.get(
    "/nowcast/snapshot",
    response_model=NowcastSnapshot,
    summary="Nowcast snapshot at a given lead-time horizon",
    description=(
        "Returns the APM-Nowcast prediction snapshot for the requested lead-time "
        "horizon with per-modality sensor availability flags. Combines deep learning "
        "model inference with real-time Open-Meteo India NWP atmospheric fields "
        "(or graceful benchmark atmospheric cube fallback if offline)."
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
    cached_entry = _snapshots.get(key)
    now = time.time()

    # Re-use cached snapshot if within 10-minute TTL for instant playback response
    if cached_entry is not None and (now - cached_entry[0] < SNAPSHOT_TTL):
        snapshot_dict = cached_entry[1]
    else:
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
            _snapshots[key] = (now, snapshot_dict)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference snapshot generation failed: {exc}",
            ) from exc

    return NowcastSnapshot(**snapshot_dict)


@router.get(
    "/nowcast/live-weather",
    summary="Live Open-Meteo Pan-India NWP grid",
    description="Returns live parameters across the 14 Pan-India meteorological grid points.",
)
async def get_live_weather():
    grid = await live_weather_service.fetch_live_grid_async()
    points_list = [p.to_dict() for p in grid.values()]
    return {
        "isLive": live_weather_service.is_live,
        "stationCount": len(grid),
        "dataSource": "live_open_meteo" if live_weather_service.is_live else "benchmark_fallback",
        "points": points_list,
        "stations": points_list,
        "lastError": live_weather_service.last_error,
    }
