"""
Readers for NRSC lightning CSV and NetCDF datasets.

Expected CSV columns (flexible detection):
  datetime_utc, latitude, longitude, peak_current_kA, type
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import netCDF4 as nc4
import numpy as np
import pandas as pd

from common.logging_config import get_logger

log = get_logger(__name__)

_DATETIME_CANDIDATES = ["datetime_utc", "datetime", "time", "timestamp", "date_time"]
_LAT_CANDIDATES      = ["latitude", "lat", "Latitude"]
_LON_CANDIDATES      = ["longitude", "lon", "Long", "Longitude"]


def read_lightning_csv(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Read an NRSC lightning CSV file with flexible column detection.

    Returns
    -------
    df   : pd.DataFrame with standardised columns (time_utc, lat, lon, ...)
    meta : dict with time range, bounding box, and row count.
    """
    df = pd.read_csv(str(path), low_memory=False)

    dt_col  = _find_col(df, _DATETIME_CANDIDATES)
    lat_col = _find_col(df, _LAT_CANDIDATES)
    lon_col = _find_col(df, _LON_CANDIDATES)

    if dt_col:
        df["time_utc"] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
    if lat_col and lat_col != "lat":
        df["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    if lon_col and lon_col != "lon":
        df["lon"] = pd.to_numeric(df[lon_col], errors="coerce")

    df = df.dropna(subset=["lat", "lon"] if "lat" in df.columns else [])

    meta: dict[str, Any] = {
        "columns":    list(df.columns),
        "n_records":  len(df),
        "time_start": (df["time_utc"].min().to_pydatetime() if "time_utc" in df.columns else None),
        "time_end":   (df["time_utc"].max().to_pydatetime() if "time_utc" in df.columns else None),
        "bbox": {
            "lat_min": float(df["lat"].min()) if "lat" in df.columns else None,
            "lat_max": float(df["lat"].max()) if "lat" in df.columns else None,
            "lon_min": float(df["lon"].min()) if "lon" in df.columns else None,
            "lon_max": float(df["lon"].max()) if "lon" in df.columns else None,
        },
    }
    return df, meta


def read_lightning_netcdf(path: Path) -> dict[str, Any]:
    """Read NRSC lightning NetCDF and return metadata dict."""
    meta: dict[str, Any] = {
        "variables":   [],
        "valid_time":  None,
        "source_file": str(path),
    }
    try:
        with nc4.Dataset(str(path), "r") as ds:
            meta["variables"] = list(ds.variables.keys())
            if "time" in ds.variables:
                try:
                    t = nc4.num2date(ds.variables["time"][0], ds.variables["time"].units)
                    meta["valid_time"] = datetime(
                        t.year, t.month, t.day,
                        t.hour, t.minute, t.second,
                        tzinfo=timezone.utc,
                    )
                except Exception:
                    pass
    except Exception as exc:
        log.warning("Lightning NetCDF read failed", extra={"exc": str(exc)})
    return meta


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None
