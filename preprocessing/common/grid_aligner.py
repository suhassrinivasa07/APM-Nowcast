"""
Common Grid Definition and Spatial Alignment Engine for APM-Nowcast.

Provides uniform Indian domain gridding and reprojection/resampling while
strictly preserving invalid/missing observation masks (NaN != 0).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import scipy.interpolate
import xarray as xr

from common.logging_config import get_logger
from common.schemas import GeoBoundingBox

log = get_logger(__name__)


@dataclass(frozen=True)
class CommonGrid:
    """
    Standard geographical regular lat/lon grid for APM-Nowcast.
    Default covers the Indian sub-continental domain at configurable resolution.
    """

    lat_min: float = 5.0
    lat_max: float = 40.0
    lon_min: float = 65.0
    lon_max: float = 100.0
    resolution_deg: float = 0.04  # ~4km resolution (matching INSAT-3D TIR-1)

    @classmethod
    def from_bbox(cls, bbox: GeoBoundingBox, resolution_deg: float = 0.04) -> "CommonGrid":
        return cls(
            lat_min=bbox.lat_min,
            lat_max=bbox.lat_max,
            lon_min=bbox.lon_min,
            lon_max=bbox.lon_max,
            resolution_deg=resolution_deg,
        )

    @property
    def lats(self) -> np.ndarray:
        """1D array of latitude coordinates from south to north."""
        # Using linspace with exact count to avoid floating-point drift
        num = int(round((self.lat_max - self.lat_min) / self.resolution_deg)) + 1
        return np.linspace(self.lat_min, self.lat_max, num, dtype=np.float32)

    @property
    def lons(self) -> np.ndarray:
        """1D array of longitude coordinates from west to east."""
        num = int(round((self.lon_max - self.lon_min) / self.resolution_deg)) + 1
        return np.linspace(self.lon_min, self.lon_max, num, dtype=np.float32)

    @property
    def shape(self) -> Tuple[int, int]:
        """(n_lat, n_lon) tuple."""
        return len(self.lats), len(self.lons)

    @property
    def lat_edges(self) -> np.ndarray:
        """1D array of latitude cell edges for discrete binning (n_lat + 1)."""
        half_res = self.resolution_deg / 2.0
        edges = np.empty(len(self.lats) + 1, dtype=np.float64)
        edges[:-1] = self.lats - half_res
        edges[-1] = self.lats[-1] + half_res
        return edges

    @property
    def lon_edges(self) -> np.ndarray:
        """1D array of longitude cell edges for discrete binning (n_lon + 1)."""
        half_res = self.resolution_deg / 2.0
        edges = np.empty(len(self.lons) + 1, dtype=np.float64)
        edges[:-1] = self.lons - half_res
        edges[-1] = self.lons[-1] + half_res
        return edges

    def meshgrid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return 2D (lons_2d, lats_2d) coordinate grids."""
        return np.meshgrid(self.lons, self.lats)


class GridAligner:
    """
    Resamples meteorological and satellite arrays to the target CommonGrid.

    Maintains strict quality flags and missing-value masks:
    - Never replaces NaN with 0.
    - Points outside source coverage or invalid in source remain NaN.
    - Creates explicit boolean `valid_mask` array alongside every variable.
    """

    def __init__(self, target_grid: Optional[CommonGrid] = None) -> None:
        self.target_grid = target_grid or CommonGrid()

    def resample_2d_regular(
        self,
        src_lats: np.ndarray,
        src_lons: np.ndarray,
        src_data: np.ndarray,
        method: str = "linear",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample 2D regular-grid data onto target_grid.

        Parameters
        ----------
        src_lats : 1D array of latitudes (monotonically increasing or decreasing)
        src_lons : 1D array of longitudes (monotonically increasing)
        src_data : 2D array of shape (len(src_lats), len(src_lons))
        method   : "linear" or "nearest"

        Returns
        -------
        resampled_data : 2D array of shape target_grid.shape (NaNs preserved)
        valid_mask     : 2D boolean array (True where data is valid and non-NaN)
        """
        # Ensure lat order is strictly increasing for RegularGridInterpolator
        if src_lats[1] < src_lats[0]:
            src_lats = np.flip(src_lats)
            src_data = np.flip(src_data, axis=0)

        # Build interpolator with NaN fill for out-of-bounds
        interpolator = scipy.interpolate.RegularGridInterpolator(
            (src_lats, src_lons),
            src_data,
            method=method,
            bounds_error=False,
            fill_value=np.nan,
        )

        tgt_lons_2d, tgt_lats_2d = self.target_grid.meshgrid()
        # Query points shape: (N, 2) where order is (lat, lon)
        pts = np.column_stack([tgt_lats_2d.ravel(), tgt_lons_2d.ravel()])
        resampled_flat = interpolator(pts)
        resampled = resampled_flat.reshape(self.target_grid.shape).astype(np.float32)

        # Explicit validity mask (never silent 0)
        valid_mask = ~np.isnan(resampled)

        return resampled, valid_mask

    def resample_curvilinear(
        self,
        src_lats_2d: np.ndarray,
        src_lons_2d: np.ndarray,
        src_data_2d: np.ndarray,
        method: str = "nearest",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample satellite swath / curvilinear coordinates to target_grid
        using scipy.interpolate.griddata.

        Parameters
        ----------
        src_lats_2d : 2D array of latitudes
        src_lons_2d : 2D array of longitudes
        src_data_2d : 2D array of values
        method      : "nearest" or "linear"

        Returns
        -------
        resampled_data : 2D array (shape matching target_grid, NaNs preserved)
        valid_mask     : 2D boolean array
        """
        # Filter only valid coordinates and non-NaN values
        valid_in = (
            ~np.isnan(src_lats_2d)
            & ~np.isnan(src_lons_2d)
            & ~np.isnan(src_data_2d)
            & (src_lats_2d >= self.target_grid.lat_min - 1.0)
            & (src_lats_2d <= self.target_grid.lat_max + 1.0)
            & (src_lons_2d >= self.target_grid.lon_min - 1.0)
            & (src_lons_2d <= self.target_grid.lon_max + 1.0)
        )

        if not np.any(valid_in):
            resampled = np.full(self.target_grid.shape, np.nan, dtype=np.float32)
            valid_mask = np.zeros(self.target_grid.shape, dtype=bool)
            return resampled, valid_mask

        points = np.column_stack([src_lats_2d[valid_in].ravel(), src_lons_2d[valid_in].ravel()])
        values = src_data_2d[valid_in].ravel()

        tgt_lons_2d, tgt_lats_2d = self.target_grid.meshgrid()
        resampled = scipy.interpolate.griddata(
            points,
            values,
            (tgt_lats_2d, tgt_lons_2d),
            method=method,
            fill_value=np.nan,
        ).astype(np.float32)

        valid_mask = ~np.isnan(resampled)
        return resampled, valid_mask

    def align_xarray_dataset(
        self,
        ds: xr.Dataset,
        lat_coord: str = "latitude",
        lon_coord: str = "longitude",
        method: str = "linear",
    ) -> xr.Dataset:
        """
        Align an xarray Dataset containing multiple 2D variables to target_grid.
        Preserves coordinates and adds a `valid_mask` for each variable.
        """
        src_lats = ds[lat_coord].values
        src_lons = ds[lon_coord].values

        data_vars = {}
        for var_name, da in ds.data_vars.items():
            arr = da.values
            if arr.ndim == 2:
                aligned, vmask = self.resample_2d_regular(src_lats, src_lons, arr, method=method)
                data_vars[var_name] = (["latitude", "longitude"], aligned, da.attrs)
                data_vars[f"{var_name}_valid_mask"] = (["latitude", "longitude"], vmask)
            elif arr.ndim == 3:
                # Handle leading dimension (e.g. time or level)
                n_lead = arr.shape[0]
                lead_dim = da.dims[0]
                aligned_slices = []
                mask_slices = []
                for i in range(n_lead):
                    al, vm = self.resample_2d_regular(src_lats, src_lons, arr[i], method=method)
                    aligned_slices.append(al)
                    mask_slices.append(vm)
                data_vars[var_name] = (
                    [lead_dim, "latitude", "longitude"],
                    np.stack(aligned_slices, axis=0),
                    da.attrs,
                )
                data_vars[f"{var_name}_valid_mask"] = (
                    [lead_dim, "latitude", "longitude"],
                    np.stack(mask_slices, axis=0),
                )

        coords = {
            "latitude": (["latitude"], self.target_grid.lats, {"units": "degrees_north", "standard_name": "latitude"}),
            "longitude": (["longitude"], self.target_grid.lons, {"units": "degrees_east", "standard_name": "longitude"}),
        }
        for c in ds.coords:
            if c not in (lat_coord, lon_coord) and c in ds.dims:
                coords[c] = ds[c]

        aligned_ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds.attrs)
        aligned_ds.attrs["spatial_resolution_deg"] = self.target_grid.resolution_deg
        aligned_ds.attrs["spatial_coverage"] = f"[{self.target_grid.lat_min}, {self.target_grid.lat_max}]N, [{self.target_grid.lon_min}, {self.target_grid.lon_max}]E"
        return aligned_ds
