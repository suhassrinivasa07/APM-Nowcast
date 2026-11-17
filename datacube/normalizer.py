"""
Modality-Specific Feature Normalization and Sentinel Imputation Engine.

Scientific Principles:
1. Fitted EXCLUSIVELY on training splits to prevent data leakage.
2. Statistics computed EXCLUSIVELY over valid pixels (valid_mask == True).
3. Explicit Sentinel Imputation: Missing/masked pixels are imputed with a deterministic
   sentinel (0.0) alongside the binary valid_mask channel so NaNs NEVER propagate
   into PyTorch loss gradients.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from common.logging_config import get_logger
from datacube.schema import CHANNEL_REGISTRY, ChannelSpec, ScalingType

log = get_logger(__name__)


@dataclass
class ChannelNormStats:
    """Statistical scaling parameters for a single variable."""
    mean: float = 0.0
    std: float = 1.0
    min_val: float = 0.0
    max_val: float = 1.0
    log1p_mean: float = 0.0
    log1p_std: float = 1.0
    valid_count: int = 0
    scaling_type: str = ScalingType.ZSCORE.value
    sentinel_value: float = 0.0


class ModalityNormalizer:
    """
    Fits and applies modality-specific feature scalers over atmospheric variables.
    """

    def __init__(self, stats: Optional[Dict[str, ChannelNormStats]] = None) -> None:
        self.stats: Dict[str, ChannelNormStats] = stats or {}

    def is_fitted(self) -> bool:
        return len(self.stats) > 0

    def fit(
        self,
        cube: xr.Dataset,
        train_indices: Optional[List[int]] = None,
    ) -> "ModalityNormalizer":
        """
        Fit normalization statistics exclusively on the training time steps
        and exclusively over valid, unmasked pixels.
        """
        log.info("Fitting ModalityNormalizer on training data...")

        if train_indices is not None:
            train_cube = cube.isel(time=train_indices)
        else:
            train_cube = cube

        fitted_stats: Dict[str, ChannelNormStats] = {}

        for var_name, spec in CHANNEL_REGISTRY.items():
            if var_name not in train_cube:
                continue

            arr = train_cube[var_name].values
            vmask_key = f"{var_name}_valid_mask"
            if vmask_key in train_cube:
                vmask = train_cube[vmask_key].values
            else:
                vmask = ~np.isnan(arr)

            # Filter valid pixels only
            valid_pixels = arr[vmask & ~np.isnan(arr)]
            if spec.clip_min is not None:
                valid_pixels = valid_pixels[valid_pixels >= spec.clip_min]
            if spec.clip_max is not None:
                valid_pixels = valid_pixels[valid_pixels <= spec.clip_max]

            n_valid = len(valid_pixels)

            if n_valid > 0:
                mean_val = float(np.mean(valid_pixels))
                std_val = float(np.std(valid_pixels))
                if std_val < 1e-6:
                    std_val = 1.0

                min_v = float(np.min(valid_pixels))
                max_v = float(np.max(valid_pixels))
                if max_v - min_v < 1e-6:
                    max_v = min_v + 1.0

                # Compute log1p statistics for heavy-tailed counts / reflectivity
                valid_log = np.log1p(np.maximum(0.0, valid_pixels))
                l_mean = float(np.mean(valid_log))
                l_std = float(np.std(valid_log))
                if l_std < 1e-6:
                    l_std = 1.0
            else:
                # Modality has 0 valid pixels in training set (safe defaults)
                mean_val, std_val = 0.0, 1.0
                min_v, max_v = 0.0, 1.0
                l_mean, l_std = 0.0, 1.0

            fitted_stats[var_name] = ChannelNormStats(
                mean=mean_val,
                std=std_val,
                min_val=min_v,
                max_val=max_v,
                log1p_mean=l_mean,
                log1p_std=l_std,
                valid_count=n_valid,
                scaling_type=spec.scaling.value,
                sentinel_value=spec.sentinel_value,
            )

        self.stats = fitted_stats
        log.info("ModalityNormalizer fitting complete", extra={"channels_fitted": len(fitted_stats)})
        return self

    def transform(
        self,
        var_name: str,
        data: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Normalize array based on fitted statistics and impute missing regions with sentinel (0.0).

        Returns
        -------
        normalized : np.ndarray with ZERO NaNs.
        """
        spec = CHANNEL_REGISTRY.get(var_name)
        scaling = spec.scaling if spec else ScalingType.ZSCORE
        sentinel = spec.sentinel_value if spec else 0.0

        out = data.astype(np.float32).copy()
        if valid_mask is None:
            valid_mask = ~np.isnan(out)
        else:
            valid_mask = valid_mask & ~np.isnan(out)

        stats = self.stats.get(var_name, ChannelNormStats())

        # If variable has no valid pixels, fill entirely with sentinel
        if not np.any(valid_mask):
            return np.full_like(out, sentinel)

        v_data = out[valid_mask]

        if scaling == ScalingType.IDENTITY:
            norm_v = v_data

        elif scaling == ScalingType.ZSCORE:
            norm_v = (v_data - stats.mean) / stats.std

        elif scaling == ScalingType.MINMAX:
            norm_v = (v_data - stats.min_val) / (stats.max_val - stats.min_val)
            norm_v = np.clip(norm_v, 0.0, 1.0)

        elif scaling == ScalingType.LOG1P:
            # log(1 + x) then standardize
            log_v = np.log1p(np.maximum(0.0, v_data))
            norm_v = (log_v - stats.log1p_mean) / stats.log1p_std

        else:
            norm_v = (v_data - stats.mean) / stats.std

        # Put normalized values back and impute unobserved/masked pixels with sentinel
        out[valid_mask] = norm_v
        out[~valid_mask] = sentinel

        # Extra safety guarantee: verify zero NaNs remain
        if np.any(np.isnan(out)):
            out[np.isnan(out)] = sentinel

        return out

    def inverse_transform(
        self,
        var_name: str,
        norm_data: np.ndarray,
    ) -> np.ndarray:
        """Denormalize values back to physical units."""
        spec = CHANNEL_REGISTRY.get(var_name)
        scaling = spec.scaling if spec else ScalingType.ZSCORE
        stats = self.stats.get(var_name, ChannelNormStats())

        out = norm_data.astype(np.float32).copy()

        if scaling == ScalingType.IDENTITY:
            return out
        elif scaling == ScalingType.ZSCORE:
            return out * stats.std + stats.mean
        elif scaling == ScalingType.MINMAX:
            return out * (stats.max_val - stats.min_val) + stats.min_val
        elif scaling == ScalingType.LOG1P:
            log_v = out * stats.log1p_std + stats.log1p_mean
            return np.expm1(np.maximum(0.0, log_v))
        return out * stats.std + stats.mean

    def save_to_json(self, path: Union[str, Path]) -> None:
        """Serialize normalization statistics to JSON."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = {k: asdict(v) for k, v in self.stats.items()}
        dest.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load_from_json(cls, path: Union[str, Path]) -> "ModalityNormalizer":
        """Deserialize normalization statistics from JSON."""
        src = Path(path)
        data = json.loads(src.read_text(encoding="utf-8"))
        stats = {k: ChannelNormStats(**v) for k, v in data.items()}
        return cls(stats=stats)
