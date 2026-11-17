"""
Radar Data Preprocessor for Indian Doppler Weather Radar (DWR) Networks.

Responsibilities:
- Reads DWR NetCDF / HDF5 volume/PPI scan products.
- Applies reflectivity and Doppler velocity QC (fill value removal, physical bounds).
- Performs ground clutter filtering (detecting high dBZ paired with near-zero radial velocity).
- Generates gridded 2D products: Column-Maximum Reflectivity (Composite Z) and CAPPI.
- Regrids to the CommonGrid while preserving explicit missing/out-of-range masks (NaN != 0).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import h5py
import netCDF4 as nc4
import numpy as np
import xarray as xr

from common.logging_config import get_logger
from preprocessing.common.grid_aligner import CommonGrid, GridAligner
from preprocessing.common.time_sync import TimeSynchronizer, TimeSyncResult

log = get_logger(__name__)

# Valid meteorological reflectivity and velocity thresholds
DBZ_MIN_VALID = -10.0
DBZ_MAX_VALID = 75.0
VEL_MAX_MAGNITUDE = 65.0  # m/s

# Ground clutter thresholds: high reflectivity with near-zero radial motion
CLUTTER_DBZ_THRESHOLD = 40.0
CLUTTER_VEL_THRESHOLD = 1.0  # m/s


class RadarPreprocessor:
    """
    Quality controls and grids DWR radar scans onto the common Indian domain.
    """

    def __init__(
        self,
        target_grid: Optional[CommonGrid] = None,
        station_id: str = "UNKNOWN",
        clutter_filter: bool = True,
    ) -> None:
        self.grid = target_grid or CommonGrid()
        self.aligner = GridAligner(self.grid)
        self.station_id = station_id
        self.clutter_filter = clutter_filter
        self.time_sync = TimeSynchronizer()

    def process_file(
        self,
        filepath: Union[str, Path],
        target_time: Optional[Union[datetime, str]] = None,
        tolerance_minutes: float = 15.0,
    ) -> xr.Dataset:
        """
        Process a DWR NetCDF/HDF5 file into a gridded xarray Dataset.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Radar file not found: {path}")

        if path.suffix.lower() in (".nc", ".nc4"):
            raw_vars, attrs, obs_time = self._read_netcdf(path)
        else:
            raw_vars, attrs, obs_time = self._read_hdf5(path)

        sync_result = None
        if target_time is not None and obs_time is not None:
            sync_result = self.time_sync.evaluate_alignment(
                obs_time, target_time, tolerance_minutes=tolerance_minutes
            )

        src_lats, src_lons = self._extract_coordinates(raw_vars, attrs)

        # Extract and QC reflectivity (DBZ)
        dbz_raw = self._find_variable(raw_vars, ["DBZ", "Z", "Reflectivity", "reflectivity", "dBZ"])
        vel_raw = self._find_variable(raw_vars, ["VEL", "VR", "Velocity", "velocity", "radial_velocity"])

        if dbz_raw is None:
            raise KeyError(f"No reflectivity variable found in {path}. Available: {list(raw_vars.keys())}")

        # QC and Clutter removal
        dbz_qc, vel_qc = self._apply_qc(dbz_raw, vel_raw)

        # Generate 2D products (Composite/Column-max Reflectivity and CAPPI)
        if dbz_qc.ndim == 3:
            # Volume scan: column-max composite
            comp_dbz = np.nanmax(dbz_qc, axis=0)
            cappi_dbz = dbz_qc[0]  # lowest tilt / surface CAPPI
            vel_2d = vel_qc[0] if vel_qc is not None else None
        else:
            comp_dbz = dbz_qc
            cappi_dbz = dbz_qc
            vel_2d = vel_qc

        # Resample onto target CommonGrid
        comp_resampled, comp_mask = self._resample(src_lats, src_lons, comp_dbz)
        cappi_resampled, cappi_mask = self._resample(src_lats, src_lons, cappi_dbz)

        data_vars = {
            "composite_reflectivity": (
                ["latitude", "longitude"],
                comp_resampled,
                {"units": "dBZ", "long_name": "Column-Maximum / Composite Reflectivity"},
            ),
            "composite_reflectivity_valid_mask": (
                ["latitude", "longitude"],
                comp_mask,
                {"description": "True within radar observation range; False where missing/masked/out-of-range"},
            ),
            "reflectivity_cappi": (
                ["latitude", "longitude"],
                cappi_resampled,
                {"units": "dBZ", "long_name": "Lowest Elevation / CAPPI Reflectivity"},
            ),
            "reflectivity_cappi_valid_mask": (
                ["latitude", "longitude"],
                cappi_mask,
                {"description": "True within radar observation range; False where missing/masked/out-of-range"},
            ),
        }

        if vel_2d is not None:
            vel_resampled, vel_mask = self._resample(src_lats, src_lons, vel_2d)
            data_vars["radial_velocity"] = (
                ["latitude", "longitude"],
                vel_resampled,
                {"units": "m/s", "long_name": "Doppler Radial Velocity"},
            )
            data_vars["radial_velocity_valid_mask"] = (
                ["latitude", "longitude"],
                vel_mask,
                {"description": "True for valid Doppler velocity"},
            )

        coords = {
            "latitude": (["latitude"], self.grid.lats, {"units": "degrees_north"}),
            "longitude": (["longitude"], self.grid.lons, {"units": "degrees_east"}),
        }
        if obs_time is not None:
            coords["time"] = (["time"], [np.datetime64(obs_time.astimezone(timezone.utc).replace(tzinfo=None))], {"standard_name": "time"})

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        ds.attrs.update({
            "source_file": str(path.resolve()),
            "station_id": self.station_id,
            "observation_time_utc": obs_time.isoformat() if obs_time else "unknown",
            "spatial_resolution_deg": self.grid.resolution_deg,
        })
        if sync_result:
            ds.attrs["temporal_offset_minutes"] = sync_result.offset_minutes
            ds.attrs["sync_status"] = sync_result.status

        return ds

    def _apply_qc(
        self,
        dbz: np.ndarray,
        vel: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Quality control: mask fill values, remove non-meteorological limits,
        and perform ground clutter removal.
        """
        dbz_clean = dbz.astype(np.float32).copy()

        # Mask fill values (standard radar fill values: -9999, -32768, < -50)
        invalid_fill = (dbz_clean < -50.0) | (dbz_clean > 100.0) | np.isnan(dbz_clean)
        dbz_clean[invalid_fill] = np.nan

        # Mask sub-meteorological limits (below -10 dBZ or above 75 dBZ)
        dbz_clean[(dbz_clean < DBZ_MIN_VALID) | (dbz_clean > DBZ_MAX_VALID)] = np.nan

        vel_clean = None
        if vel is not None:
            vel_clean = vel.astype(np.float32).copy()
            invalid_vel = (np.abs(vel_clean) > VEL_MAX_MAGNITUDE) | (vel_clean < -900.0) | np.isnan(vel_clean)
            vel_clean[invalid_vel] = np.nan

            # Ground clutter filter: high reflectivity with near-zero motion
            if self.clutter_filter:
                clutter_mask = (
                    (dbz_clean >= CLUTTER_DBZ_THRESHOLD)
                    & ~np.isnan(vel_clean)
                    & (np.abs(vel_clean) <= CLUTTER_VEL_THRESHOLD)
                )
                dbz_clean[clutter_mask] = np.nan
                vel_clean[clutter_mask] = np.nan
                log.debug("Clutter filter applied", extra={"flagged_points": int(np.sum(clutter_mask))})

        return dbz_clean, vel_clean

    def _resample(
        self,
        src_lats: np.ndarray,
        src_lons: np.ndarray,
        data_2d: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if src_lats.ndim == 1 and src_lons.ndim == 1:
            return self.aligner.resample_2d_regular(src_lats, src_lons, data_2d, method="nearest")
        return self.aligner.resample_curvilinear(src_lats, src_lons, data_2d, method="nearest")

    def _find_variable(self, variables: Dict[str, np.ndarray], candidates: list[str]) -> Optional[np.ndarray]:
        for cand in candidates:
            if cand in variables:
                return variables[cand]
        # Case-insensitive search
        lower_map = {k.lower(): k for k in variables.keys()}
        for cand in candidates:
            if cand.lower() in lower_map:
                return variables[lower_map[cand.lower()]]
        return None

    def _extract_coordinates(
        self,
        raw_vars: Dict[str, np.ndarray],
        attrs: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        for lat_k in ("latitude", "lat", "Latitude"):
            if lat_k in raw_vars:
                for lon_k in ("longitude", "lon", "Longitude"):
                    if lon_k in raw_vars:
                        return raw_vars[lat_k], raw_vars[lon_k]

        # Fallback: estimate coordinate grid around station or regional default
        first_arr = next(iter(raw_vars.values()))
        shape = first_arr.shape[-2:]
        n_lat, n_lon = shape

        # Approximate DWR domain ~250km radius coverage (~5 degrees span)
        stn_lat = float(attrs.get("latitude", 19.0))
        stn_lon = float(attrs.get("longitude", 73.0))

        lats = np.linspace(stn_lat - 2.5, stn_lat + 2.5, n_lat, dtype=np.float32)
        lons = np.linspace(stn_lon - 2.5, stn_lon + 2.5, n_lon, dtype=np.float32)
        return lats, lons

    def _read_netcdf(self, path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Optional[datetime]]:
        vars_dict = {}
        attrs = {}
        obs_time = None
        with nc4.Dataset(str(path), "r") as ds:
            for k in ds.ncattrs():
                attrs[k] = ds.getncattr(k)
            for k, v in ds.variables.items():
                arr = np.array(v[:])
                # Squeeze singleton time dimension if present
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]
                elif arr.ndim == 4 and arr.shape[0] == 1:
                    arr = arr[0]
                vars_dict[k] = arr

            if "time" in ds.variables:
                try:
                    t_val = ds.variables["time"][0]
                    t_unit = ds.variables["time"].units
                    dt = nc4.num2date(t_val, t_unit)
                    obs_time = datetime(
                        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, tzinfo=timezone.utc
                    )
                except Exception:
                    pass

        return vars_dict, attrs, obs_time

    def _read_hdf5(self, path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Optional[datetime]]:
        vars_dict = {}
        attrs = {}
        obs_time = None
        with h5py.File(str(path), "r") as fh:
            for k, v in fh.attrs.items():
                attrs[k] = v.decode() if isinstance(v, bytes) else str(v)
            for k in fh.keys():
                item = fh[k]
                if isinstance(item, h5py.Dataset):
                    vars_dict[k] = np.array(item)

        return vars_dict, attrs, obs_time
