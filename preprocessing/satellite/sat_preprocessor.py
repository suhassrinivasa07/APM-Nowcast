"""
Satellite Data Preprocessor for INSAT-3D/3DR/3DS Level-1B Products.

Responsibilities:
- Reads HDF5/NetCDF files using h5py / netCDF4.
- Calibrates raw counts/radiances to Brightness Temperature (Kelvin) or Reflectance.
- Performs geolocation alignment and crops to the Indian CommonGrid.
- Preserves explicit validity masks (NaN != 0) and eliminates space/unphysical pixels.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import netCDF4 as nc4
import numpy as np
import xarray as xr

from common.logging_config import get_logger
from common.schemas import UTCTimestamp
from preprocessing.common.grid_aligner import CommonGrid, GridAligner
from preprocessing.common.time_sync import TimeSynchronizer, TimeSyncResult

log = get_logger(__name__)

# Physical constants for Planck inverse calculation (mW/(m2 sr cm-4) and cm K)
C1 = 1.191042e-5
C2 = 1.4387752

# INSAT-3D central wavenumbers in cm^-1
INSAT_CENTRAL_WAVENUMBERS = {
    "IMG_TIR1": 926.0,   # ~10.8 um
    "IMG_TIR2": 833.0,   # ~12.0 um
    "IMG_MIR": 2564.0,   # ~3.9 um
    "IMG_WV": 1470.0,    # ~6.8 um
}

# Physically valid brightness temperature range (Kelvin) for tropical convection
PHYSICAL_TB_MIN = 160.0
PHYSICAL_TB_MAX = 340.0


class SatellitePreprocessor:
    """
    Calibrates and regrids INSAT-3D/3DR/3DS imagery to the target CommonGrid.
    """

    def __init__(
        self,
        target_grid: Optional[CommonGrid] = None,
        channels: Optional[List[str]] = None,
    ) -> None:
        self.grid = target_grid or CommonGrid()
        self.aligner = GridAligner(self.grid)
        self.channels = channels or ["IMG_TIR1", "IMG_MIR", "IMG_WV", "IMG_TIR2"]
        self.time_sync = TimeSynchronizer()

    def process_file(
        self,
        filepath: Union[str, Path],
        target_time: Optional[Union[datetime, str]] = None,
        tolerance_minutes: float = 30.0,
    ) -> xr.Dataset:
        """
        Process an INSAT HDF5 or NetCDF file into a calibrated xarray Dataset.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Satellite file not found: {path}")

        # Extract acquisition metadata & raw datasets
        if path.suffix.lower() in (".nc", ".nc4"):
            raw_data, attrs, obs_time = self._read_netcdf(path)
        else:
            raw_data, attrs, obs_time = self._read_hdf5(path)

        # Temporal sync check
        sync_result = None
        if target_time is not None and obs_time is not None:
            sync_result = self.time_sync.evaluate_alignment(
                obs_time, target_time, tolerance_minutes=tolerance_minutes
            )

        # Geolocation grids
        src_lats, src_lons = self._derive_source_coordinates(raw_data, attrs)

        # Calibrate, mask, and align each channel
        data_vars = {}
        for chan_name in self.channels:
            if chan_name not in raw_data:
                continue

            raw_arr = raw_data[chan_name]
            calibrated_arr = self._calibrate_channel(chan_name, raw_arr, attrs.get(f"{chan_name}_attrs", {}))

            # Resample to target CommonGrid
            if src_lats.ndim == 1 and src_lons.ndim == 1:
                aligned, vmask = self.aligner.resample_2d_regular(
                    src_lats, src_lons, calibrated_arr, method="linear"
                )
            else:
                aligned, vmask = self.aligner.resample_curvilinear(
                    src_lats, src_lons, calibrated_arr, method="nearest"
                )

            # Standardized variable key (e.g. IMG_TIR1 -> tir1)
            std_name = chan_name.lower().replace("img_", "")
            data_vars[std_name] = (
                ["latitude", "longitude"],
                aligned,
                {"units": "K", "long_name": f"Brightness Temperature {chan_name}"},
            )
            data_vars[f"{std_name}_valid_mask"] = (
                ["latitude", "longitude"],
                vmask,
                {"description": "True where observation is valid; False where space/cloud-free/NaN"},
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
            "satellite_name": attrs.get("Satellite_Name", "INSAT-3D"),
            "observation_time_utc": obs_time.isoformat() if obs_time else "unknown",
            "spatial_resolution_deg": self.grid.resolution_deg,
        })
        if sync_result:
            ds.attrs["temporal_offset_minutes"] = sync_result.offset_minutes
            ds.attrs["sync_status"] = sync_result.status

        return ds

    def _calibrate_channel(
        self,
        chan_name: str,
        arr: np.ndarray,
        chan_attrs: Dict[str, Any],
    ) -> np.ndarray:
        """
        Calibrate raw values to physical Brightness Temperature (Kelvin).
        Sets invalid / space pixels to NaN (never silent 0).
        """
        data = arr.astype(np.float32).copy()

        # Check if values are already in realistic Brightness Temperature range (e.g. from synthetic samples)
        valid_finite = data[np.isfinite(data) & (data > 0)]
        if valid_finite.size > 0 and 150.0 <= np.median(valid_finite) <= 350.0:
            # Already in Kelvin Tb
            calibrated = data
        else:
            # Apply Gain/Slope and Offset/Intercept if present
            slope = float(chan_attrs.get("Slope", chan_attrs.get("Gain", 1.0)))
            intercept = float(chan_attrs.get("Intercept", chan_attrs.get("Offset", 0.0)))
            radiance = data * slope + intercept

            # Invert Planck function if wavenumber is known
            wn = INSAT_CENTRAL_WAVENUMBERS.get(chan_name)
            if wn is not None:
                calibrated = np.full_like(radiance, np.nan)
                valid_rad = radiance > 0.0
                if np.any(valid_rad):
                    # Tb = (c2 * wn) / ln(1 + (c1 * wn^3) / L)
                    term = (C1 * (wn ** 3)) / radiance[valid_rad]
                    calibrated[valid_rad] = (C2 * wn) / np.log(1.0 + term)
            else:
                calibrated = radiance

        # Apply strict physical QC masking
        invalid_mask = (
            np.isnan(calibrated)
            | (calibrated < PHYSICAL_TB_MIN)
            | (calibrated > PHYSICAL_TB_MAX)
            | (calibrated <= 0.0)
        )
        calibrated[invalid_mask] = np.nan

        return calibrated

    def _derive_source_coordinates(
        self,
        raw_data: Dict[str, np.ndarray],
        attrs: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Derive 1D or 2D lat/lon coordinates from file arrays or metadata."""
        if "latitude" in raw_data and "longitude" in raw_data:
            return raw_data["latitude"], raw_data["longitude"]

        # If image dimensions known, generate regular grid across Indian coverage
        first_arr = next(iter(raw_data.values()))
        n_lat, n_lon = first_arr.shape[:2]

        # Use nominal center coordinates if present
        center_lat = 22.5
        center_lon = 82.0
        coords_str = attrs.get("Nominal_Central_Point_Coordinates_(lat,lon)")
        if coords_str:
            try:
                parts = coords_str.strip("()").split(",")
                center_lat = float(parts[0])
                center_lon = float(parts[1])
            except Exception:
                pass

        # Estimate extent centered at center_lat, center_lon
        half_span_lat = 15.0
        half_span_lon = 18.0
        lats = np.linspace(center_lat - half_span_lat, center_lat + half_span_lat, n_lat, dtype=np.float32)
        lons = np.linspace(center_lon - half_span_lon, center_lon + half_span_lon, n_lon, dtype=np.float32)

        return lats, lons

    def _read_hdf5(self, path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Optional[datetime]]:
        data = {}
        attrs = {}
        obs_time = None
        with h5py.File(str(path), "r") as fh:
            for k, v in fh.attrs.items():
                attrs[k] = v.decode() if isinstance(v, bytes) else str(v)

            for key in fh.keys():
                item = fh[key]
                if isinstance(item, h5py.Dataset):
                    data[key] = np.array(item)
                    attrs[f"{key}_attrs"] = {
                        ak: (av.decode() if isinstance(av, bytes) else av)
                        for ak, av in item.attrs.items()
                    }

            # Parse observation time
            date_str = attrs.get("Image_Date", "")
            time_str = attrs.get("Image_Time_in_UTC", "")
            if date_str and time_str:
                for fmt in ("%d-%b-%Y %H:%M:%S", "%Y%m%d %H%M%S", "%Y-%m-%d %H:%M:%S"):
                    try:
                        obs_time = datetime.strptime(f"{date_str} {time_str}".strip(), fmt).replace(
                            tzinfo=timezone.utc
                        )
                        break
                    except ValueError:
                        continue

        return data, attrs, obs_time

    def _read_netcdf(self, path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Optional[datetime]]:
        data = {}
        attrs = {}
        obs_time = None
        with nc4.Dataset(str(path), "r") as ds:
            for k in ds.ncattrs():
                attrs[k] = ds.getncattr(k)

            for key, var in ds.variables.items():
                data[key] = np.array(var[:])
                attrs[f"{key}_attrs"] = {ak: var.getncattr(ak) for ak in var.ncattrs()}

            if "time" in ds.variables:
                try:
                    t_val = ds.variables["time"][0]
                    t_unit = ds.variables["time"].units
                    obs_time = nc4.num2date(t_val, t_unit)
                    if hasattr(obs_time, "tzinfo") and obs_time.tzinfo is None:
                        obs_time = obs_time.replace(tzinfo=timezone.utc)
                except Exception:
                    pass

        return data, attrs, obs_time
