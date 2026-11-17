"""
Meteorological Skill & Calibration Verification Metrics for APM-Nowcast.

Implements standard NWP/nowcasting evaluation metrics per WMO and WWRP/WGNE
guidelines, all vectorized in PyTorch/NumPy for GPU-accelerated batch evaluation.

Metrics:
  Categorical (at configurable thresholds):
    - POD  — Probability of Detection (Hit Rate)
    - FAR  — False Alarm Ratio
    - CSI  — Critical Success Index (Threat Score)
    - HSS  — Heidke Skill Score

  Spatial neighborhood:
    - FSS  — Fractions Skill Score (Roberts & Lean 2008)

  Probabilistic:
    - Brier Score
    - ECE  — Expected Calibration Error
    - Flash RMSE / Flash MAE (intensity verification)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-7


# ---------------------------------------------------------------------------
# Categorical Contingency Metrics
# ---------------------------------------------------------------------------

def _contingency_counts(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute hits, misses, false alarms, correct negatives from thresholded probs.

    Parameters
    ----------
    probs   : [..., H, W] predicted probabilities in [0, 1]
    targets : [..., H, W] binary observed occurrence (0 or 1)
    threshold : float in (0, 1)

    Returns
    -------
    H_count, M_count, FA_count, CN_count — scalar tensors
    """
    pred_binary = (probs >= threshold).float()
    obs_binary  = (targets > 0.5).float()

    H  = (pred_binary * obs_binary).sum()          # hits
    M  = ((1 - pred_binary) * obs_binary).sum()    # misses
    FA = (pred_binary * (1 - obs_binary)).sum()     # false alarms
    CN = ((1 - pred_binary) * (1 - obs_binary)).sum()  # correct negatives
    return H, M, FA, CN


def pod(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Probability of Detection (Hit Rate) = H / (H + M)."""
    H, M, FA, CN = _contingency_counts(probs, targets, threshold)
    return (H / (H + M + EPS)).item()


def far(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """False Alarm Ratio = FA / (H + FA)."""
    H, M, FA, CN = _contingency_counts(probs, targets, threshold)
    return (FA / (H + FA + EPS)).item()


def csi(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Critical Success Index (Threat Score) = H / (H + M + FA)."""
    H, M, FA, CN = _contingency_counts(probs, targets, threshold)
    return (H / (H + M + FA + EPS)).item()


def hss(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Heidke Skill Score.
    HSS = 2*(H*CN - M*FA) / ((H+M)*(M+CN) + (H+FA)*(FA+CN))
    """
    H, M, FA, CN = _contingency_counts(probs, targets, threshold)
    num = 2.0 * (H * CN - M * FA)
    den = (H + M) * (M + CN) + (H + FA) * (FA + CN)
    return (num / (den + EPS)).item()


# ---------------------------------------------------------------------------
# Spatial Neighborhood: Fractions Skill Score (Roberts & Lean 2008)
# ---------------------------------------------------------------------------

def fss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    scale: int = 5,
) -> float:
    """
    Fractions Skill Score at a given spatial neighborhood scale.

    Uses 2D average pooling to compute observed and forecast fractions
    within (scale × scale) neighborhoods across the entire spatial domain.

    Parameters
    ----------
    probs   : [B, T_out, 1, H, W] or [H, W] predicted probabilities
    targets : same shape, binary occurrence
    threshold : forecast threshold applied to probs
    scale   : integer neighborhood width/height (e.g. 3, 5, 9)

    Returns
    -------
    FSS score in [0, 1]. Perfect forecast = 1.0.
    """
    # Flatten to [N, H, W]
    p = probs.reshape(-1, probs.shape[-2], probs.shape[-1])
    t = (targets > 0.5).float().reshape(-1, targets.shape[-2], targets.shape[-1])
    pred_binary = (p >= threshold).float()

    # Add channel dim for pooling: [N, 1, H, W]
    p_b = pred_binary.unsqueeze(1)
    t_b = t.unsqueeze(1)

    pad = scale // 2
    # Average-pool to get fractions within each neighborhood
    frac_pred = F.avg_pool2d(p_b, kernel_size=scale, stride=1, padding=pad)
    frac_obs  = F.avg_pool2d(t_b, kernel_size=scale, stride=1, padding=pad)

    # FSS numerator: mean squared difference of fractions
    num = ((frac_pred - frac_obs) ** 2).mean()
    # FSS denominator: sum of squared fractions independently
    den = (frac_pred ** 2).mean() + (frac_obs ** 2).mean()

    if den.item() < EPS:
        # No observed lightning and no predicted → perfect score
        return 1.0 if num.item() < EPS else 0.0

    return (1.0 - num / (den + EPS)).item()


def fss_multiscale(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    scales: Tuple[int, ...] = (3, 5, 9),
) -> Dict[str, float]:
    """Return FSS at multiple scales."""
    return {f"fss_{s}x{s}": fss(probs, targets, threshold, s) for s in scales}


# ---------------------------------------------------------------------------
# Probabilistic Metrics
# ---------------------------------------------------------------------------

def brier_score(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Brier Score = mean((p - o)^2).  Lower is better. Random = ~0.25 for 50:50.
    """
    obs = (targets > 0.5).float()
    return ((probs - obs) ** 2).mean().item()


def ece(
    probs: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error.

    Partitions [0,1] into `n_bins` equal-width bins, computes |mean_confidence -
    fraction_positive| per bin, and returns the sample-weighted average.

    Parameters
    ----------
    probs   : [...] predicted probabilities in [0, 1]
    targets : [...] binary occurrence
    n_bins  : number of calibration bins (default 10)

    Returns
    -------
    ECE scalar in [0, 1]. Perfect calibration = 0.0.
    """
    p_flat = probs.reshape(-1).cpu().float()
    o_flat = (targets.reshape(-1) > 0.5).float().cpu()
    n = p_flat.numel()

    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    total_ece = 0.0

    for b in range(n_bins):
        lo, hi = bin_edges[b].item(), bin_edges[b + 1].item()
        # Include right edge in last bin
        if b < n_bins - 1:
            mask = (p_flat >= lo) & (p_flat < hi)
        else:
            mask = (p_flat >= lo) & (p_flat <= hi)

        count = mask.sum().item()
        if count == 0:
            continue
        mean_conf  = p_flat[mask].mean().item()
        frac_pos   = o_flat[mask].mean().item()
        total_ece += (count / n) * abs(mean_conf - frac_pos)

    return total_ece


def flash_rmse(pred: torch.Tensor, targets: torch.Tensor) -> float:
    """Root Mean Squared Error on flash rate predictions."""
    return torch.sqrt(((pred - targets) ** 2).mean()).item()


def flash_mae(pred: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean Absolute Error on flash rate predictions."""
    return torch.abs(pred - targets).mean().item()


# ---------------------------------------------------------------------------
# Per-Lead-Time Metric Aggregation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    probs: torch.Tensor,
    flash_pred: torch.Tensor,
    targets_occ: torch.Tensor,
    targets_flash: torch.Tensor,
    thresholds: Tuple[float, ...] = (0.3, 0.5),
    fss_scales: Tuple[int, ...] = (3, 5, 9),
) -> Dict[str, float]:
    """
    Compute all metrics aggregated across the full batch.

    Parameters
    ----------
    probs         : [B, T_out, 1, H, W]
    flash_pred    : [B, T_out, 1, H, W]
    targets_occ   : [B, T_out, 1, H, W]
    targets_flash : [B, T_out, 1, H, W]

    Returns
    -------
    flat dict of metric names -> scalar floats
    """
    results: Dict[str, float] = {}

    # Categorical metrics per threshold
    for thr in thresholds:
        t_str = str(int(thr * 100))
        results[f"pod_{t_str}"]  = pod(probs,  targets_occ, thr)
        results[f"far_{t_str}"]  = far(probs,  targets_occ, thr)
        results[f"csi_{t_str}"]  = csi(probs,  targets_occ, thr)
        results[f"hss_{t_str}"]  = hss(probs,  targets_occ, thr)

    # Spatial FSS per scale (at primary threshold 0.5)
    for sc in fss_scales:
        results[f"fss_{sc}x{sc}"] = fss(probs, targets_occ, threshold=0.5, scale=sc)
    # Average FSS across all scales
    results["fss_mean"] = float(np.mean([results[f"fss_{sc}x{sc}"] for sc in fss_scales]))

    # Probabilistic
    results["brier"] = brier_score(probs, targets_occ)
    results["ece"]   = ece(probs, targets_occ)

    # Intensity
    results["flash_rmse"] = flash_rmse(flash_pred, targets_flash)
    results["flash_mae"]  = flash_mae(flash_pred, targets_flash)

    return results


def compute_per_leadtime_metrics(
    probs: torch.Tensor,
    flash_pred: torch.Tensor,
    targets_occ: torch.Tensor,
    targets_flash: torch.Tensor,
    thresholds: Tuple[float, ...] = (0.3, 0.5),
    fss_scales: Tuple[int, ...] = (3, 5, 9),
) -> List[Dict[str, float]]:
    """
    Compute metrics per forecast lead-time step.

    Returns list of metric dicts, one per T_out step.
    """
    T_out = probs.shape[1]
    per_step = []
    for t in range(T_out):
        m = compute_all_metrics(
            probs[:, t:t+1],       # keep dim
            flash_pred[:, t:t+1],
            targets_occ[:, t:t+1],
            targets_flash[:, t:t+1],
            thresholds=thresholds,
            fss_scales=fss_scales,
        )
        per_step.append(m)
    return per_step
