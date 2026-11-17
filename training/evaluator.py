"""
APMEvaluator — Multi-horizon & Sensor-Dropout Evaluation Runner.

Evaluates any BaseNowcastModel over a dataset, producing:
  - Per-lead-time meteorological skill scores (POD, FAR, CSI, HSS, FSS, Brier, ECE)
  - Multi-scenario sensor-dropout evaluation (full / radar-off / satellite-off)
  - Structured JSON export and Markdown table summary

Sensor-dropout scenarios replicate realistic operational degraded-data
conditions by zeroing the appropriate column of modality_masks before inference.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from common.logging_config import get_logger
from datacube.schema import NUM_MODALITIES, MODALITY_TO_INDEX
from models.base import BaseNowcastModel
from training.metrics import compute_per_leadtime_metrics

log = get_logger(__name__)

# Modality indices for dropout scenarios
_RADAR_IDX     = MODALITY_TO_INDEX.get("radar", 1)
_SATELLITE_IDX = MODALITY_TO_INDEX.get("satellite", 0)


def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Recursively move all tensors in a batch dict to `device`."""
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {
                kk: vv.to(device, non_blocking=True) if isinstance(vv, torch.Tensor) else vv
                for kk, vv in v.items()
            }
        else:
            out[k] = v
    return out


class APMEvaluator:
    """
    Multi-horizon, multi-scenario evaluator for APM-Nowcast models.

    Parameters
    ----------
    model : BaseNowcastModel
        Any trained model inheriting from BaseNowcastModel.
    device : torch.device or None
        Defaults to CUDA if available, else CPU.
    thresholds : tuple of float
        Occurrence-probability thresholds for categorical metrics.
    fss_scales : tuple of int
        Spatial neighborhood scales for Fractions Skill Score.
    minutes_per_step : int
        Minutes between consecutive forecast lead times (default 15).
    """

    SCENARIOS = {
        "full":            None,          # no dropout
        "radar_off":       _RADAR_IDX,
        "satellite_off":   _SATELLITE_IDX,
    }

    def __init__(
        self,
        model: BaseNowcastModel,
        device: Optional[torch.device] = None,
        thresholds: Tuple[float, ...] = (0.3, 0.5),
        fss_scales: Tuple[int, ...] = (3, 5, 9),
        minutes_per_step: int = 15,
    ) -> None:
        self.model = model
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = self.model.to(self.device)
        self.thresholds = thresholds
        self.fss_scales = fss_scales
        self.minutes_per_step = minutes_per_step

    # ------------------------------------------------------------------
    # Core evaluation pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: Iterable,
        scenario_name: str = "full",
        dropout_modality_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run inference on `data_loader` under a single sensor scenario.

        Parameters
        ----------
        data_loader : iterable of batch dicts (list or DataLoader)
        scenario_name : str label for this scenario
        dropout_modality_idx : modality column index to zero in masks, or None

        Returns
        -------
        Dict with keys:
          - "scenario" : str
          - "num_batches" : int
          - "per_leadtime" : list[dict] — one metric dict per T_out step
          - "aggregate" : dict — metrics averaged across all lead times
          - "lead_times_min" : list[int] — +15, +30, … minutes
        """
        self.model.eval()

        all_probs:      List[torch.Tensor] = []
        all_flash_pred: List[torch.Tensor] = []
        all_occ:        List[torch.Tensor] = []
        all_flash:      List[torch.Tensor] = []
        num_batches = 0

        for batch in data_loader:
            batch = _batch_to_device(batch, self.device)

            inputs = batch.get("inputs_by_modality", {})
            masks  = batch.get(
                "modality_masks",
                torch.ones(1, 1, NUM_MODALITIES, device=self.device),
            ).clone()   # clone so we don't mutate the original

            # Apply sensor dropout
            if dropout_modality_idx is not None:
                masks[..., dropout_modality_idx] = 0.0

            tgt_occ   = batch.get("target_occurrence_frames", None)
            tgt_flash = batch.get("target_lightning_frames",  None)
            if tgt_occ is None or tgt_flash is None:
                continue

            # Ensure shape [B, T_out, 1, H, W]
            if tgt_occ.ndim == 4:
                tgt_occ   = tgt_occ.unsqueeze(2)
            if tgt_flash.ndim == 4:
                tgt_flash = tgt_flash.unsqueeze(2)

            # Forward pass (AMP-compatible if device is CUDA)
            if self.device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = self.model(inputs, masks)
            else:
                outputs = self.model(inputs, masks)

            # Collect predictions as FP32 on CPU
            probs      = outputs.get("lightning_probs",  torch.zeros_like(tgt_occ))
            flash_pred = outputs.get("flash_rate_pred",  torch.zeros_like(tgt_flash))

            all_probs.append(probs.float().cpu())
            all_flash_pred.append(flash_pred.float().cpu())
            all_occ.append(tgt_occ.float().cpu())
            all_flash.append(tgt_flash.float().cpu())
            num_batches += 1

        if not all_probs:
            log.warning(f"[{scenario_name}] No valid batches found during evaluation.")
            return {
                "scenario": scenario_name,
                "num_batches": 0,
                "per_leadtime": [],
                "aggregate": {},
                "lead_times_min": [],
            }

        probs      = torch.cat(all_probs,      dim=0)
        flash_pred = torch.cat(all_flash_pred, dim=0)
        occ        = torch.cat(all_occ,        dim=0)
        flash      = torch.cat(all_flash,      dim=0)

        # Per-lead-time metrics
        per_lt = compute_per_leadtime_metrics(
            probs, flash_pred, occ, flash,
            thresholds=self.thresholds,
            fss_scales=self.fss_scales,
        )

        T_out = probs.shape[1]
        lead_times = [(t + 1) * self.minutes_per_step for t in range(T_out)]

        # Aggregate: mean across lead times for each metric
        agg: Dict[str, float] = {}
        if per_lt:
            all_keys = per_lt[0].keys()
            for key in all_keys:
                agg[key] = float(
                    sum(step[key] for step in per_lt) / len(per_lt)
                )

        log.info(
            f"[{scenario_name}] batches={num_batches} | "
            f"T_out={T_out} | "
            f"agg_csi_50={agg.get('csi_50', 0.0):.4f} | "
            f"agg_brier={agg.get('brier', 1.0):.4f}"
        )

        return {
            "scenario":      scenario_name,
            "num_batches":   num_batches,
            "per_leadtime":  per_lt,
            "aggregate":     agg,
            "lead_times_min": lead_times,
        }

    # ------------------------------------------------------------------
    # Multi-scenario evaluation
    # ------------------------------------------------------------------

    def evaluate_all_scenarios(
        self,
        data_loader: Iterable,
        scenarios: Optional[Sequence[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run evaluation under all (or a subset of) sensor-dropout scenarios.

        Parameters
        ----------
        data_loader : iterable of batches
        scenarios   : subset of SCENARIOS keys; None = all three scenarios

        Returns
        -------
        Dict[scenario_name -> evaluate() result dict]
        """
        if scenarios is None:
            scenarios = list(self.SCENARIOS.keys())

        results: Dict[str, Dict[str, Any]] = {}
        for name in scenarios:
            if name not in self.SCENARIOS:
                log.warning(f"Unknown scenario '{name}', skipping.")
                continue
            dropout_idx = self.SCENARIOS[name]
            log.info(f"Evaluating scenario: {name}")
            results[name] = self.evaluate(
                data_loader,
                scenario_name=name,
                dropout_modality_idx=dropout_idx,
            )

        return results

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    @staticmethod
    def to_json(
        results: Dict[str, Any],
        path: str | Path,
        indent: int = 2,
    ) -> None:
        """
        Serialize evaluation results to a JSON file.

        Parameters
        ----------
        results : output of evaluate() or evaluate_all_scenarios()
        path    : output file path (parent directories created automatically)
        indent  : JSON indentation
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _make_serializable(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_make_serializable(v) for v in obj]
            if isinstance(obj, (torch.Tensor,)):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            if isinstance(obj, float):
                return round(obj, 6)
            return obj

        out_path.write_text(
            json.dumps(_make_serializable(results), indent=indent),
            encoding="utf-8",
        )
        log.info(f"Evaluation results saved to {out_path}")

    @staticmethod
    def to_markdown(
        results: Dict[str, Any],
        primary_threshold_pct: int = 50,
    ) -> str:
        """
        Format evaluation results as a GitHub-style Markdown table.

        Accepts either a single-scenario dict (from evaluate()) or
        a multi-scenario dict (from evaluate_all_scenarios()).

        Parameters
        ----------
        results : dict from evaluate() or evaluate_all_scenarios()
        primary_threshold_pct : threshold percent label (30 or 50)

        Returns
        -------
        Markdown string with one table per scenario
        """
        # Detect format: multi-scenario if values are dicts with "scenario" key
        is_multi = all(
            isinstance(v, dict) and "per_leadtime" in v
            for v in results.values()
        )

        if is_multi:
            scenarios_data = results
        else:
            # Single scenario result
            name = results.get("scenario", "result")
            scenarios_data = {name: results}

        lines: List[str] = []
        thr = primary_threshold_pct

        for scenario_name, data in scenarios_data.items():
            lines.append(f"## Scenario: `{scenario_name}`\n")

            per_lt = data.get("per_leadtime", [])
            lead_times = data.get("lead_times_min", [])
            if not per_lt:
                lines.append("_No data available._\n")
                continue

            # Header
            lines.append(
                f"| Lead Time | POD@{thr} | FAR@{thr} | CSI@{thr} | "
                f"HSS@{thr} | FSS (mean) | Brier | ECE | Flash RMSE |"
            )
            lines.append(
                "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
            )

            for i, step_metrics in enumerate(per_lt):
                lt = lead_times[i] if i < len(lead_times) else (i + 1) * 15

                def g(key: str) -> str:
                    val = step_metrics.get(key, float("nan"))
                    return f"{val:.4f}"

                lines.append(
                    f"| +{lt} min | {g(f'pod_{thr}')} | {g(f'far_{thr}')} | "
                    f"{g(f'csi_{thr}')} | {g(f'hss_{thr}')} | "
                    f"{g('fss_mean')} | {g('brier')} | {g('ece')} | "
                    f"{g('flash_rmse')} |"
                )

            # Aggregate row
            agg = data.get("aggregate", {})
            if agg:
                def ga(key: str) -> str:
                    val = agg.get(key, float("nan"))
                    return f"{val:.4f}"

                lines.append(
                    f"| **Mean** | {ga(f'pod_{thr}')} | {ga(f'far_{thr}')} | "
                    f"{ga(f'csi_{thr}')} | {ga(f'hss_{thr}')} | "
                    f"{ga('fss_mean')} | {ga('brier')} | {ga('ece')} | "
                    f"{ga('flash_rmse')} |"
                )

            lines.append("")

        return "\n".join(lines)
