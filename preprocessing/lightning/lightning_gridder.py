"""
Discrete Point-to-Grid Lightning Binning Engine for NRSC Lightning Observations.

Scientific Principle:
Point lightning events are discrete Poisson/Bernoulli processes.
THEY MUST NEVER BE SPATIALLY INTERPOLATED AS CONTINUOUS FIELDS.

Instead, events are spatially binned into discrete 2D grid cells:
- Flash Counts (number of strokes per cell).
- Occurrence Mask (binary indicator: 1 if count > 0, else 0).
- Peak Current Statistics (max/mean absolute current in kA per cell, NaN for 0-count cells).
- Type partitioning: Cloud-to-Ground (CG) vs Intra-Cloud (IC).
- Network coverage tracking (distinguishing true 0-count cells from unmonitored cells).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats
import xarray as xr

from common.logging_config import get_logger
from common.schemas import GeoBoundingBox, UTCTimestamp
from data_ingestion.lightning.lightning_reader import read_lightning_csv
from preprocessing.common.grid_aligner import CommonGrid
from preprocessing.common.time_sync import TimeSynchronizer, TimeWindow

log = get_logger(__name__)


class LightningGridder:
    """
    Bins lightning point strokes into gridded count and occurrence maps.
    """

    def __init__(
        self,
        target_grid: Optional[CommonGrid] = None,
        coverage_bbox: Optional[GeoBoundingBox] = None,
    ) -> None:
        self.grid = target_grid or CommonGrid()
        self.coverage_bbox = coverage_bbox or GeoBoundingBox(
            lat_min=self.grid.lat_min,
            lat_max=self.grid.lat_max,
            lon_min=self.grid.lon_min,
            lon_max=self.grid.lon_max,
        )
        self.time_sync = TimeSynchronizer()

    def process_file(
        self,
        filepath: Union[str, Path],
        target_time: Union[datetime, str],
        accumulation_minutes: float = 30.0,
    ) -> xr.Dataset:
        """
        Ingest a lightning CSV file and bin events in the retrospective window
        [target_time - accumulation_minutes, target_time].
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Lightning file not found: {path}")

        df, meta = read_lightning_csv(path)
        return self.grid_dataframe(
            df=df,
            target_time=target_time,
            accumulation_minutes=accumulation_minutes,
            source_file=str(path.resolve()),
        )

    def grid_dataframe(
        self,
        df: pd.DataFrame,
        target_time: Union[datetime, str],
        accumulation_minutes: float = 30.0,
        source_file: str = "memory",
    ) -> xr.Dataset:
        """
        Bin a DataFrame of lightning events onto the CommonGrid.
        """
        utc_target = UTCTimestamp(dt=target_time).dt

        # Find time and coordinate columns
        time_col = self._detect_col(df, ["time_utc", "datetime_utc", "datetime", "time", "timestamp"])
        lat_col = self._detect_col(df, ["lat", "latitude", "Latitude"])
        lon_col = self._detect_col(df, ["lon", "longitude", "Longitude"])
        cur_col = self._detect_col(df, ["peak_current_kA", "current", "peak_current", "current_kA"])
        type_col = self._detect_col(df, ["type", "stroke_type", "lightning_type"])

        # Filter strictly to retrospective window: [T - accumulation_minutes, T]
        if time_col and not df.empty:
            filtered_df, window = self.time_sync.filter_dataframe_window(
                df,
                time_column=time_col,
                target_time=utc_target,
                retrospective_minutes=accumulation_minutes,
            )
        else:
            filtered_df = df.copy()
            window = TimeWindow.create(target_time=utc_target, retrospective_minutes=accumulation_minutes)

        n_lat, n_lon = self.grid.shape
        lat_edges = self.grid.lat_edges
        lon_edges = self.grid.lon_edges

        # Allocate grid arrays
        flash_count = np.zeros((n_lat, n_lon), dtype=np.int32)
        cg_count = np.zeros((n_lat, n_lon), dtype=np.int32)
        ic_count = np.zeros((n_lat, n_lon), dtype=np.int32)
        peak_current_max = np.full((n_lat, n_lon), np.nan, dtype=np.float32)

        if not filtered_df.empty and lat_col and lon_col:
            lats = filtered_df[lat_col].to_numpy(dtype=np.float64)
            lons = filtered_df[lon_col].to_numpy(dtype=np.float64)

            # Filter coordinates within grid domain bounds
            valid_pts = (
                ~np.isnan(lats)
                & ~np.isnan(lons)
                & (lats >= lat_edges[0])
                & (lats <= lat_edges[-1])
                & (lons >= lon_edges[0])
                & (lons <= lon_edges[-1])
            )

            if np.any(valid_pts):
                v_lats = lats[valid_pts]
                v_lons = lons[valid_pts]

                # 2D discrete binning: np.histogram2d
                # Note: histogram2d returns (len(lat_edges)-1, len(lon_edges)-1)
                counts_2d, _, _ = np.histogram2d(v_lats, v_lons, bins=[lat_edges, lon_edges])
                flash_count = counts_2d.astype(np.int32)

                # Peak current maximum per cell
                if cur_col and cur_col in filtered_df.columns:
                    v_curs = np.abs(filtered_df.loc[valid_pts, cur_col].to_numpy(dtype=np.float64))
                    stat_max, _, _, _ = scipy.stats.binned_statistic_2d(
                        v_lats,
                        v_lons,
                        v_curs,
                        statistic="max",
                        bins=[lat_edges, lon_edges],
                    )
                    peak_current_max = stat_max.astype(np.float32)

                # Partition by stroke type if available
                if type_col and type_col in filtered_df.columns:
                    types = filtered_df.loc[valid_pts, type_col].astype(str).str.upper().to_numpy()
                    cg_mask = types == "CG"
                    if np.any(cg_mask):
                        cg_c, _, _ = np.histogram2d(v_lats[cg_mask], v_lons[cg_mask], bins=[lat_edges, lon_edges])
                        cg_count = cg_c.astype(np.int32)

                    ic_mask = np.isin(types, ["IC", "CC"])
                    if np.any(ic_mask):
                        ic_c, _, _ = np.histogram2d(v_lats[ic_mask], v_lons[ic_mask], bins=[lat_edges, lon_edges])
                        ic_count = ic_c.astype(np.int32)

        # Occurrence mask: binary 1 if count > 0, else 0
        occurrence_mask = (flash_count > 0).astype(np.uint8)

        # Detection coverage mask: True where network monitors
        lons_2d, lats_2d = self.grid.meshgrid()
        coverage_mask = (
            (lats_2d >= self.coverage_bbox.lat_min)
            & (lats_2d <= self.coverage_bbox.lat_max)
            & (lons_2d >= self.coverage_bbox.lon_min)
            & (lons_2d <= self.coverage_bbox.lon_max)
        )

        data_vars = {
            "flash_count": (
                ["latitude", "longitude"],
                flash_count,
                {"units": "count", "long_name": f"Accumulated Lightning Flash Count ({accumulation_minutes} min)"},
            ),
            "occurrence_mask": (
                ["latitude", "longitude"],
                occurrence_mask,
                {"units": "binary", "long_name": "Binary Lightning Occurrence (1=True, 0=False)"},
            ),
            "peak_current_max": (
                ["latitude", "longitude"],
                peak_current_max,
                {"units": "kA", "long_name": "Maximum Absolute Peak Current (NaN where 0 flashes)"},
            ),
            "cg_flash_count": (
                ["latitude", "longitude"],
                cg_count,
                {"units": "count", "long_name": "Cloud-to-Ground Flash Count"},
            ),
            "ic_flash_count": (
                ["latitude", "longitude"],
                ic_count,
                {"units": "count", "long_name": "Intra-Cloud Flash Count"},
            ),
            "detection_coverage_mask": (
                ["latitude", "longitude"],
                coverage_mask,
                {"description": "True where lightning sensor network coverage is active"},
            ),
        }

        coords = {
            "latitude": (["latitude"], self.grid.lats, {"units": "degrees_north"}),
            "longitude": (["longitude"], self.grid.lons, {"units": "degrees_east"}),
            "time": (["time"], [np.datetime64(utc_target.astimezone(timezone.utc).replace(tzinfo=None))], {"standard_name": "time"}),
        }

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        ds.attrs.update({
            "source_file": source_file,
            "target_time_utc": utc_target.isoformat(),
            "window_start_utc": window.start_time.isoformat(),
            "window_end_utc": window.end_time.isoformat(),
            "accumulation_minutes": accumulation_minutes,
            "total_flashes_in_window": int(np.sum(flash_count)),
            "spatial_resolution_deg": self.grid.resolution_deg,
        })

        return ds

    def _detect_col(self, df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        lower_cols = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in lower_cols:
                return lower_cols[cand.lower()]
        return None
