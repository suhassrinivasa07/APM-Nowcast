"""
Extract metadata from IMD DWR NetCDF / HDF5 files.
Handles standard CF-convention NetCDF exports from MOSDAC.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import netCDF4 as nc4
import numpy as np

from common.logging_config import get_logger

log = get_logger(__name__)


def read_radar_metadata(path: Path) -> dict[str, Any]:
    """Return metadata dict for a DWR file. Tries NetCDF then HDF5."""
    suffix = path.suffix.lower()
    if suffix in {".nc", ".nc4"}:
        return _read_netcdf(path)
    return _read_hdf5(path)


def _read_netcdf(path: Path) -> dict[str, Any]:
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
                    times = nc4.num2date(ds.variables["time"][:], ds.variables["time"].units)
                    dt = times[0]
                    meta["valid_time"] = datetime(
                        dt.year, dt.month, dt.day,
                        dt.hour, dt.minute, dt.second,
                        tzinfo=timezone.utc,
                    )
                except Exception:
                    pass
            if "latitude" in ds.variables:
                lats = ds.variables["latitude"][:]
                meta["lat_min"] = float(np.min(lats))
                meta["lat_max"] = float(np.max(lats))
            if "longitude" in ds.variables:
                lons = ds.variables["longitude"][:]
                meta["lon_min"] = float(np.min(lons))
                meta["lon_max"] = float(np.max(lons))
    except Exception as exc:
        log.warning("NetCDF radar read failed", extra={"path": str(path), "exc": str(exc)})
    return meta


def _read_hdf5(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "variables":   [],
        "valid_time":  None,
        "source_file": str(path),
    }
    try:
        with h5py.File(str(path), "r") as fh:
            meta["variables"] = list(fh.keys())
            attrs = dict(fh.attrs)
            for key in ["Datetime", "Date", "Time", "StartDate"]:
                if key in attrs:
                    val = attrs[key]
                    if isinstance(val, bytes):
                        val = val.decode()
                    meta["datetime_raw"] = str(val)
                    break
    except Exception as exc:
        log.warning("HDF5 radar read failed", extra={"path": str(path), "exc": str(exc)})
    return meta
