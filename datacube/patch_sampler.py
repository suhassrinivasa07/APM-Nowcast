"""
Spatiotemporal Sliding-Window and Convective-Event-Weighted Patch Sampler.

Mitigates severe class imbalance in meteorological nowcasting:
- Deep convective cells occupy < 5% of the Indian subcontinental landmass.
- Standard uniform grid sampling creates 98%+ negative/empty-sky patches.
- This sampler extracts spatial crops (default 64x64) weighted by convective activity
  (lightning strike counts and radar reflectivity thresholds) while preserving
  a controlled proportion of negative background patches to prevent false-alarm bias.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr

from common.logging_config import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class PatchCoordinate:
    """Spatiotemporal bounding box for a single sample."""
    t_start_idx: int               # Index in cube time dimension
    lat_start_idx: int             # Spatial latitude start index
    lon_start_idx: int             # Spatial longitude start index
    patch_height: int              # Spatial patch height H (e.g. 64)
    patch_width: int               # Spatial patch width W (e.g. 64)
    is_convective: bool            # True if active convection detected
    target_flash_count: int = 0    # Total flashes in future horizon
    max_reflectivity_dbz: float = 0.0


class ConvectivePatchSampler:
    """
    Extracts balanced spatiotemporal patch coordinates from a unified data cube.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int] = (64, 64),
        T_in: int = 4,                   # History frames (e.g. 1 hr @ 15m)
        T_out: int = 4,                  # Forecast horizon (e.g. 1 hr @ 15m)
        spatial_stride: int = 32,        # Spatial crop stride (pixels)
        min_reflectivity_dbz: float = 35.0,  # Radar convective threshold
        negative_sample_ratio: float = 0.35, # Fraction of quiet patches to keep
        seed: int = 42,
    ) -> None:
        self.patch_size = patch_size
        self.T_in = T_in
        self.T_out = T_out
        self.T_total = T_in + T_out
        self.spatial_stride = spatial_stride
        self.min_reflectivity_dbz = min_reflectivity_dbz
        self.negative_sample_ratio = negative_sample_ratio
        self.rng = np.random.default_rng(seed)

    def extract_patch_coordinates(
        self,
        cube: xr.Dataset,
        time_indices: Optional[List[int]] = None,
    ) -> List[PatchCoordinate]:
        """
        Scan space-time cube and return balanced patch coordinate list.
        """
        n_times = len(cube["time"])
        n_lat = len(cube["latitude"])
        n_lon = len(cube["longitude"])

        H, W = self.patch_size
        if n_lat < H or n_lon < W:
            # Fallback for small test grids: crop to available size
            H = min(H, n_lat)
            W = min(W, n_lon)

        # Allowable starting time indices
        if time_indices is not None:
            sorted_t = sorted(time_indices)
            valid_t_starts = [
                t for t in sorted_t
                if (t + self.T_total <= n_times) and all((t + k) in sorted_t for k in range(self.T_total))
            ]
        else:
            valid_t_starts = list(range(n_times - self.T_total + 1))

        if not valid_t_starts:
            log.warning("No continuous temporal sequence of length T_in + T_out found.")
            return []

        # Sliding window spatial positions
        lat_starts = list(range(0, n_lat - H + 1, self.spatial_stride))
        if not lat_starts or lat_starts[-1] + H < n_lat:
            lat_starts.append(max(0, n_lat - H))
        lat_starts = sorted(list(set(lat_starts)))

        lon_starts = list(range(0, n_lon - W + 1, self.spatial_stride))
        if not lon_starts or lon_starts[-1] + W < n_lon:
            lon_starts.append(max(0, n_lon - W))
        lon_starts = sorted(list(set(lon_starts)))

        convective_patches: List[PatchCoordinate] = []
        negative_patches: List[PatchCoordinate] = []

        # Extract presence indicators if available
        has_lightning = "flash_count" in cube
        has_radar = "composite_reflectivity" in cube

        for t in valid_t_starts:
            # Future target window: [t + T_in, t + T_in + T_out)
            future_slice = slice(t + self.T_in, t + self.T_total)

            if has_lightning:
                fut_flashes = cube["flash_count"].isel(time=future_slice).values
            else:
                fut_flashes = None

            if has_radar:
                fut_radar = cube["composite_reflectivity"].isel(time=future_slice).values
            else:
                fut_radar = None

            for r in lat_starts:
                for c in lon_starts:
                    # Evaluate convective activity in this patch
                    is_conv = False
                    flash_cnt = 0
                    max_dbz = 0.0

                    if fut_flashes is not None:
                        patch_fl = fut_flashes[:, r : r + H, c : c + W]
                        flash_cnt = int(np.nansum(patch_fl))
                        if flash_cnt > 0:
                            is_conv = True

                    if fut_radar is not None:
                        patch_rd = fut_radar[:, r : r + H, c : c + W]
                        valid_rd = patch_rd[~np.isnan(patch_rd)]
                        if valid_rd.size > 0:
                            max_dbz = float(np.max(valid_rd))
                            if max_dbz >= self.min_reflectivity_dbz:
                                is_conv = True

                    coord = PatchCoordinate(
                        t_start_idx=t,
                        lat_start_idx=r,
                        lon_start_idx=c,
                        patch_height=H,
                        patch_width=W,
                        is_convective=is_conv,
                        target_flash_count=flash_cnt,
                        max_reflectivity_dbz=max_dbz,
                    )

                    if is_conv:
                        convective_patches.append(coord)
                    else:
                        negative_patches.append(coord)

        # Class balancing: subsample negative patches
        n_conv = len(convective_patches)
        if n_conv > 0:
            target_n_neg = max(1, int(round(n_conv * self.negative_sample_ratio)))
            if len(negative_patches) > target_n_neg:
                selected_neg_indices = self.rng.choice(len(negative_patches), size=target_n_neg, replace=False)
                sampled_negatives = [negative_patches[i] for i in selected_neg_indices]
            else:
                sampled_negatives = negative_patches
        else:
            # No convective activity in this block (e.g. quiet period)
            sampled_negatives = negative_patches

        all_patches = convective_patches + sampled_negatives
        self.rng.shuffle(all_patches)

        log.info(
            "Patch sampling completed",
            extra={
                "total_patches": len(all_patches),
                "convective_patches": n_conv,
                "negative_patches": len(sampled_negatives),
            },
        )
        return all_patches
