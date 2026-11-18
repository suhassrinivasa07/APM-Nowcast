"""
experiments/dropout_stress.py

Sensor-Dropout Robustness Benchmark for APM-Nowcast Module 11.

Evaluates trained models under five simulated operational sensor outages:
  1. Full sensors (baseline)
  2. Radar-Off       (radar modality mask zeroed)
  3. Satellite-Off   (satellite modality mask zeroed)
  4. Lightning-Off   (lightning modality mask zeroed)
  5. Dual-Outage     (radar + lightning both zeroed)

For each scenario × model the benchmark records:
  - CSI@0.3 at all four horizons
  - % CSI retained vs. full-sensor baseline
  - ECE (uncertainty calibration quality)
  - Uncertainty inflation factor (from APMEvaluator)

Covers APM-Nowcast plus the strongest baselines (ConvLSTM, U-Net).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from common.logging_config import get_logger
from datacube.schema import MODALITY_TO_INDEX
from models.base import BaseNowcastModel
from models.apm_nowcast import APMNowcastModel
from models.baselines import ConvLSTMNowcastModel, SpatiotemporalUNetBaseline
from training.evaluator import APMEvaluator
from experiments.runner import (
    ExperimentResult,
    _make_loader,
    seed_everything,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dropout scenario definitions
# ---------------------------------------------------------------------------

# Maps scenario name → list of modality indices to zero in masks
DROPOUT_SCENARIOS: Dict[str, List[int]] = {
    "Full Sensors":      [],
    "Radar-Off":         [MODALITY_TO_INDEX.get("radar",     1)],
    "Satellite-Off":     [MODALITY_TO_INDEX.get("satellite", 0)],
    "Lightning-Off":     [MODALITY_TO_INDEX.get("lightning", 2)],
    "Dual-Outage (R+L)": [MODALITY_TO_INDEX.get("radar",    1),
                           MODALITY_TO_INDEX.get("lightning", 2)],
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DropoutResult:
    """
    Per-model, per-scenario evaluation under sensor dropout.

    Attributes
    ----------
    model_name : str
    scenario_name : str
    per_horizon : list of dicts (T+15 … T+60)
    csi_retained_pct : list of float, % CSI retained vs. full-sensor at each horizon
    aggregate_csi : float, mean CSI across horizons
    aggregate_ece : float, mean ECE across horizons
    """
    model_name: str
    scenario_name: str
    per_horizon: List[Dict[str, float]] = field(default_factory=list)
    csi_retained_pct: List[float] = field(default_factory=list)
    aggregate_csi: float = float("nan")
    aggregate_ece: float = float("nan")
    raw: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SensorDropoutStress
# ---------------------------------------------------------------------------

class SensorDropoutStress:
    """
    Sensor-dropout robustness benchmark.

    Parameters
    ----------
    seed : int
    quick_mode : bool
        Synthetic 32×32 data, all models at minimal size.
    val_loader : Iterable or None
        Required when quick_mode=False.
    device : torch.device or None
    models : dict[str, BaseNowcastModel] or None
        Defaults to [APM-Nowcast, ConvLSTM, U-Net] in quick_mode.
    scenarios : list of str or None
        Subset of DROPOUT_SCENARIOS keys; None → all five.
    T_in, T_out : int
    """

    def __init__(
        self,
        seed: int = 42,
        quick_mode: bool = False,
        val_loader: Optional[Iterable] = None,
        device: Optional[torch.device] = None,
        models: Optional[Dict[str, BaseNowcastModel]] = None,
        scenarios: Optional[List[str]] = None,
        T_in: int = 4,
        T_out: int = 4,
    ) -> None:
        seed_everything(seed)
        self.T_in, self.T_out = T_in, T_out
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scenarios = scenarios or list(DROPOUT_SCENARIOS.keys())

        H = W = 32 if quick_mode else 64
        n_batches = 2 if quick_mode else 4

        if quick_mode:
            self._val_loader = _make_loader(n_batches, H=H, W=W, T_in=T_in, T_out=T_out)
        else:
            assert val_loader is not None, "val_loader required when quick_mode=False"
            self._val_loader = val_loader

        # Default models for the stress test
        if models is not None:
            self._models = models
        elif quick_mode:
            self._models = {
                "APM-Nowcast (Proposed)": APMNowcastModel(
                    T_in=T_in, T_out=T_out, latent_dim=16, hidden_dim=24
                ),
                "ConvLSTM": ConvLSTMNowcastModel(
                    T_in=T_in, T_out=T_out, total_in_channels=11,
                    hidden_dims=(8, 16)
                ),
                "U-Net Spatiotemporal": SpatiotemporalUNetBaseline(
                    T_in=T_in, T_out=T_out, total_in_channels=11, base_channels=8
                ),
            }
        else:
            self._models = {
                "APM-Nowcast (Proposed)": APMNowcastModel(
                    T_in=T_in, T_out=T_out, latent_dim=24, hidden_dim=48
                ),
                "ConvLSTM": ConvLSTMNowcastModel(
                    T_in=T_in, T_out=T_out, total_in_channels=11,
                    hidden_dims=(32, 64)
                ),
                "U-Net Spatiotemporal": SpatiotemporalUNetBaseline(
                    T_in=T_in, T_out=T_out, total_in_channels=11, base_channels=32
                ),
            }

        for m in self._models.values():
            m.to(self.device)
            m.eval()

        self._results: List[DropoutResult] = []

    # ------------------------------------------------------------------

    def _eval_scenario(
        self,
        model_name: str,
        model: BaseNowcastModel,
        scenario_name: str,
        dropout_idxs: List[int],
    ) -> DropoutResult:
        """Evaluate one model under one dropout scenario."""
        # Apply dropout: multi-index zeroing
        # APMEvaluator supports single-index dropout; we apply multi-index
        # by patching the loader's masks inline.
        if len(dropout_idxs) <= 1:
            idx = dropout_idxs[0] if dropout_idxs else None
            evaluator = APMEvaluator(model=model, device=self.device)
            raw = evaluator.evaluate(
                self._val_loader,
                scenario_name=scenario_name,
                dropout_modality_idx=idx,
            )
        else:
            # Multi-index: wrap val_loader to zero out all required columns
            patched = _apply_multi_dropout(self._val_loader, dropout_idxs)
            evaluator = APMEvaluator(model=model, device=self.device)
            raw = evaluator.evaluate(patched, scenario_name=scenario_name)

        per_lt = raw.get("per_leadtime", [])
        agg    = raw.get("aggregate", {})

        per_horizon = [
            {
                "csi_30":   m.get("csi_30", float("nan")),
                "brier":    m.get("brier", float("nan")),
                "ece":      m.get("ece", float("nan")),
                "fss_mean": m.get("fss_mean", float("nan")),
            }
            for m in per_lt
        ]

        return DropoutResult(
            model_name=model_name,
            scenario_name=scenario_name,
            per_horizon=per_horizon,
            aggregate_csi=agg.get("csi_30", float("nan")),
            aggregate_ece=agg.get("ece", float("nan")),
            raw=raw,
        )

    # ------------------------------------------------------------------

    def run(self) -> List[DropoutResult]:
        """
        Evaluate all models × all scenarios.

        Returns flat list of DropoutResult. CSI retention % is computed
        relative to each model's own full-sensor baseline.
        """
        self._results = []
        # {model_name: full-sensor csi per horizon}
        full_csi: Dict[str, List[float]] = {}

        for model_name, model in self._models.items():
            model.eval()
            for scenario_name in self.scenarios:
                dropout_idxs = DROPOUT_SCENARIOS[scenario_name]
                log.info(f"[Dropout] {model_name} | scenario={scenario_name}")
                result = self._eval_scenario(
                    model_name, model, scenario_name, dropout_idxs
                )
                self._results.append(result)

                if scenario_name == "Full Sensors":
                    full_csi[model_name] = [
                        h.get("csi_30", float("nan"))
                        for h in result.per_horizon
                    ]

        # Compute retention % relative to full-sensor baseline
        for result in self._results:
            base = full_csi.get(result.model_name, [])
            retained = []
            for i, h in enumerate(result.per_horizon):
                csi = h.get("csi_30", float("nan"))
                ref = base[i] if i < len(base) else float("nan")
                if ref > 1e-6:
                    retained.append(round(100.0 * csi / ref, 1))
                else:
                    retained.append(float("nan"))
            result.csi_retained_pct = retained

        return self._results

    def results(self) -> List[DropoutResult]:
        return self._results

    def to_dict(self) -> Dict[str, Any]:
        """Serialize results to a JSON-compatible dict keyed by model × scenario."""
        out: Dict[str, Any] = {}
        for r in self._results:
            key = f"{r.model_name} | {r.scenario_name}"
            out[key] = {
                "per_horizon": r.per_horizon,
                "csi_retained_pct": r.csi_retained_pct,
                "aggregate_csi": r.aggregate_csi,
                "aggregate_ece": r.aggregate_ece,
            }
        return out


# ---------------------------------------------------------------------------
# Multi-index dropout helper
# ---------------------------------------------------------------------------

def _apply_multi_dropout(
    loader: Iterable,
    dropout_idxs: List[int],
) -> List[Dict[str, Any]]:
    """
    Return a new list of batches with multiple modality mask columns zeroed.
    Works on both list loaders (quick_mode) and DataLoaders.
    """
    import copy
    patched = []
    for batch in loader:
        b = copy.deepcopy(batch)
        masks = b["modality_masks"]  # [B, T_in, M]
        for idx in dropout_idxs:
            if idx < masks.shape[-1]:
                masks[..., idx] = 0.0
        b["modality_masks"] = masks
        patched.append(b)
    return patched
