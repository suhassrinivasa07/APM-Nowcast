"""
ERA5 Atmospheric Reanalysis Preprocessor for APM-Nowcast.

Responsibilities:
- Reads ERA5 single-level and pressure-level NetCDF files using xarray/netCDF4.
- Selects the nearest UTC time slice matching the target nowcast analysis window.
- Computes derived meteorological, thermodynamic, and kinematic proxies:
  - Dewpoint depression (T - Td) and 2m Relative Humidity (%).
  - 10m horizontal wind speed (m/s).
  - Bulk vertical wind shear (m/s) if multi-level winds are available.
- Regrids fields to the CommonGrid while preserving valid observation masks.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from common.logging_config import get_logger
from common.schemas import UTCTimestamp
from preprocessing.common.grid_aligner import CommonGrid, GridAligner
from preprocessing.common.time_sync import TimeSynchronizer, TimeSyncResult

log = get_logger(__name__)


class ERA5Preprocessor:
    """
    Extracts, derives thermodynamic features, and regrids ERA5 reanalysis fields.
    """

    def __init__(self, target_grid: Optional[CommonGrid] = None) -> None:
        self.grid = target_grid or CommonGrid()
        self.aligner = GridAligner(self.grid)
        self.time_sync = TimeSynchronizer()

    def process_file(
        self,
        filepath: Union[str, Path],
        target_time: Optional[Union[datetime, str]] = None,
        max_offset_minutes: float = 90.0,
    ) -> xr.Dataset:
        """
        Open ERA5 file, select time slice, compute derived convective parameters,
        and regrid to the CommonGrid.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"ERA5 file not found: {path}")

        ds = xr.open_dataset(str(path), engine="netcdf4")

        # Standardize coordinate names
        lat_coord = self._find_coord(ds, ["latitude", "lat"])
        lon_coord = self._find_coord(ds, ["longitude", "lon"])
        time_coord = self._find_coord(ds, ["valid_time", "time"])

        if not lat_coord or not lon_coord:
            raise ValueError(f"Missing latitude/longitude coordinates in {path}. Found: {list(ds.coords)}")

        # Select target time slice if dataset has time dimension
        actual_time = None
        sync_result = None
        if time_coord and time_coord in ds.dims and ds.sizes[time_coord] > 0:
            times = [pd.Timestamp(t).to_pydatetime().replace(tzinfo=timezone.utc) for t in ds[time_coord].values]
            if target_time is not None:
                utc_target = UTCTimestamp(dt=target_time).dt
                idx, actual_time, offset_min = self.time_sync.find_nearest_slice_index(
                    times, utc_target, max_offset_minutes=max_offset_minutes
                )
                ds_slice = ds.isel({time_coord: idx})
                sync_result = self.time_sync.evaluate_alignment(
                    actual_time, utc_target, tolerance_minutes=max_offset_minutes
                )
            else:
                ds_slice = ds.isel({time_coord: 0})
                actual_time = times[0]
        else:
            ds_slice = ds

        # Compute derived thermodynamic and dynamic features
        ds_enhanced = self._derive_thermodynamic_features(ds_slice)

        # Regrid to target CommonGrid
        regridded_ds = self.aligner.align_xarray_dataset(
            ds_enhanced, lat_coord=lat_coord, lon_coord=lon_coord, method="linear"
        )

        regridded_ds.attrs.update({
            "source_file": str(path.resolve()),
            "source": "ERA5-Reanalysis",
            "observation_time_utc": actual_time.isoformat() if actual_time else "unknown",
        })
        if sync_result:
            regridded_ds.attrs["temporal_offset_minutes"] = sync_result.offset_minutes
            regridded_ds.attrs["sync_status"] = sync_result.status

        ds.close()
        return regridded_ds

    def _derive_thermodynamic_features(self, ds: xr.Dataset) -> xr.Dataset:
        """Calculate dewpoint depression, relative humidity proxy, and wind magnitude."""
        res = ds.copy()

        # 1. 2m Relative Humidity & Dewpoint Depression
        if "t2m" in res and "d2m" in res:
            t2m = res["t2m"]
            d2m = res["d2m"]
            # Dewpoint depression (K)
            dp_depression = t2m - d2m
            res["dewpoint_depression_2m"] = dp_depression
            res["dewpoint_depression_2m"].attrs = {
                "units": "K",
                "long_name": "2m Dewpoint Depression (T - Td)",
            }

            # Magnus formula approximation for RH (%)
            t_c = t2m - 273.15
            td_c = d2m - 273.15
            # Saturation vapor pressures (hPa)
            es_t = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
            es_td = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
            rh = (es_td / es_t) * 100.0
            rh = xr.where(rh > 100.0, 100.0, rh)
            rh = xr.where(rh < 0.0, 0.0, rh)
            res["relative_humidity_2m"] = rh
            res["relative_humidity_2m"].attrs = {
                "units": "%",
                "long_name": "2m Relative Humidity (Magnus approximation)",
            }

        # 2. 10m Wind Speed
        if "u10" in res and "v10" in res:
            wspd_10m = np.sqrt(res["u10"] ** 2 + res["v10"] ** 2)
            res["wind_speed_10m"] = wspd_10m
            res["wind_speed_10m"].attrs = {
                "units": "m s**-1",
                "long_name": "10m Horizontal Wind Speed",
            }

        # 3. Bulk Wind Shear (if 500 hPa and 10m/850hPa winds exist)
        u_high, v_high = None, None
        if "u" in res and "v" in res and "level" in res.dims:
            # Multi-level dataset: look for 500 hPa
            levels = list(res["level"].values)
            if 500 in levels:
                u_high = res["u"].sel(level=500)
                v_high = res["v"].sel(level=500)
            elif 500.0 in levels:
                u_high = res["u"].sel(level=500.0)
                v_high = res["v"].sel(level=500.0)

        if u_high is not None and v_high is not None and "u10" in res and "v10" in res:
            shear_0_6km = np.sqrt((u_high - res["u10"]) ** 2 + (v_high - res["v10"]) ** 2)
            res["bulk_wind_shear_0_6km"] = shear_0_6km
            res["bulk_wind_shear_0_6km"].attrs = {
                "units": "m s**-1",
                "long_name": "0-6km Bulk Wind Shear Vector Magnitude",
            }

        return res

    def _find_coord(self, ds: xr.Dataset, candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            if cand in ds.coords:
                return cand
            if cand in ds.dims:
                return cand
        return None
