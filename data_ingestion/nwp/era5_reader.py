"""
Read and extract metadata from ERA5 NetCDF files using xarray.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from common.logging_config import get_logger

log = get_logger(__name__)


def read_era5_metadata(path: Path) -> dict[str, Any]:
    """
    Open an ERA5 NetCDF file with xarray and return standardised metadata.
    """
    meta: dict[str, Any] = {
        "variables":   [],
        "time_start":  None,
        "time_end":    None,
        "lat_min":     None,
        "lat_max":     None,
        "lon_min":     None,
        "lon_max":     None,
        "source_file": str(path),
    }
    try:
        ds = xr.open_dataset(str(path), engine="netcdf4")
        meta["variables"] = list(ds.data_vars)

        if "valid_time" in ds.coords:
            times = ds.coords["valid_time"].values
        elif "time" in ds.coords:
            times = ds.coords["time"].values
        else:
            times = []

        if len(times) > 0:
            meta["time_start"] = _ts_to_utc(times[0])
            meta["time_end"]   = _ts_to_utc(times[-1])

        for lat_key in ("latitude", "lat"):
            if lat_key in ds.coords:
                meta["lat_min"] = float(ds.coords[lat_key].values.min())
                meta["lat_max"] = float(ds.coords[lat_key].values.max())
                break

        for lon_key in ("longitude", "lon"):
            if lon_key in ds.coords:
                meta["lon_min"] = float(ds.coords[lon_key].values.min())
                meta["lon_max"] = float(ds.coords[lon_key].values.max())
                break

        ds.close()
    except Exception as exc:
        log.warning("ERA5 metadata read failed", extra={"path": str(path), "exc": str(exc)})
    return meta


def read_era5_array(path: Path, variable: str) -> xr.DataArray:
    """Return a lazy xarray DataArray for a specific ERA5 variable."""
    ds = xr.open_dataset(str(path), engine="netcdf4")
    if variable not in ds:
        raise KeyError(
            f"Variable '{variable}' not found in {path}. "
            f"Available: {list(ds.data_vars)}"
        )
    return ds[variable]


def _ts_to_utc(ts: Any) -> datetime:
    """Convert numpy.datetime64 / pandas.Timestamp to Python UTC datetime."""
    return pd.Timestamp(ts).to_pydatetime().replace(tzinfo=timezone.utc)
