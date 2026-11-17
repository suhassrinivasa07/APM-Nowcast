"""
Lightweight Headless Quality Control (QC) Plotter for Meteorological Fields.

Generates diagnostic multi-panel PNG plots with:
1. Spatial geophysical variable distribution.
2. Explicit observation validity mask (1 = Valid, 0 = Missing/NaN).
3. Histogram of valid meteorological values.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

# Headless backend must be set before pyplot import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.logging_config import get_logger

log = get_logger(__name__)


class QCPlotter:
    """Headless QC plotter producing standardized PNG diagnostics."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or Path("data/processed/qc_plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_field_qc(
        self,
        data: np.ndarray,
        valid_mask: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        title: str,
        variable_name: str,
        units: str,
        output_filename: str,
        cmap: str = "turbo",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> Path:
        """
        Generate a 3-panel QC summary plot:
        [Spatial Map | Validity Mask | Value Histogram]
        """
        output_path = self.output_dir / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), dpi=120)
        extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]

        # 1. Spatial Geophysical Map
        im1 = ax1.imshow(
            data,
            extent=extent,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        ax1.set_title(f"{variable_name} ({units})", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Longitude (°E)")
        ax1.set_ylabel("Latitude (°N)")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # 2. Validity / Quality Mask
        im2 = ax2.imshow(
            valid_mask.astype(np.float32),
            extent=extent,
            origin="lower",
            cmap="Greens",
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        valid_pct = float(np.mean(valid_mask) * 100.0) if valid_mask.size > 0 else 0.0
        ax2.set_title(f"Validity Mask (Valid: {valid_pct:.1f}%)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Longitude (°E)")
        ax2.set_ylabel("Latitude (°N)")
        cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, ticks=[0, 1])
        cb2.ax.set_yticklabels(["Missing", "Valid"])

        # 3. Histogram of Valid Data
        valid_vals = data[valid_mask & ~np.isnan(data)]
        if len(valid_vals) > 0:
            ax3.hist(valid_vals, bins=35, color="steelblue", edgecolor="black", alpha=0.8)
            ax3.set_title(
                f"Distribution\nMin: {valid_vals.min():.1f} | Mean: {valid_vals.mean():.1f} | Max: {valid_vals.max():.1f}",
                fontsize=11,
            )
            ax3.set_xlabel(f"{variable_name} ({units})")
            ax3.set_ylabel("Pixel Count")
            ax3.grid(True, linestyle="--", alpha=0.5)
        else:
            ax3.text(0.5, 0.5, "No Valid Pixels", ha="center", va="center", transform=ax3.transAxes)
            ax3.set_title("Distribution (Empty)")

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()

        fig.savefig(str(output_path), bbox_inches="tight")
        plt.close(fig)

        log.info("QC plot generated", extra={"path": str(output_path)})
        return output_path
