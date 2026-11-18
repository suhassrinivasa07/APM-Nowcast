"""
experiments/artifacts.py

Automated Paper Artifact Generator for APM-Nowcast Module 11.

Exports
-------
  - JSON  : structured results for downstream analysis (results.json)
  - Markdown : GitHub-style tables (benchmark.md, ablation.md, dropout.md)
  - LaTeX : booktabs tables (benchmark.tex, ablation.tex, dropout.tex)
  - PNG   : non-interactive matplotlib Agg figures (300 DPI):
              skill_by_horizon.png
              sensor_dropout_robustness.png
              reliability_calibration.png

All matplotlib work uses the 'Agg' non-interactive backend so plots render
correctly in headless CI/pytest environments with no display server.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

# Force non-interactive backend before any matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.logging_config import get_logger
from experiments.runner import ExperimentResult, LEAD_TIME_LABELS
from experiments.dropout_stress import DropoutResult

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Colour palette (accessible, colorblind-safe)
# ---------------------------------------------------------------------------
_COLORS = [
    "#2196F3",  # blue         – APM-Nowcast (Proposed)
    "#F44336",  # red          – U-Net
    "#4CAF50",  # green        – ConvLSTM
    "#FF9800",  # orange       – Advection Persistence
    "#9C27B0",  # purple       – Eulerian Persistence
    "#00BCD4",  # cyan         – ablation variants
    "#795548",  # brown
]


def _safe(val: Any) -> str:
    """Format a float nicely; return '-' for NaN."""
    if isinstance(val, float) and math.isnan(val):
        return "—"
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return str(val)


def _int_or_dash(val: Any) -> str:
    if isinstance(val, float) and math.isnan(val):
        return "—"
    try:
        return f"{int(val):,}"
    except (TypeError, ValueError):
        return str(val)


# ---------------------------------------------------------------------------
# ArtifactGenerator
# ---------------------------------------------------------------------------

class ArtifactGenerator:
    """
    Generates all paper artifacts from ExperimentResult / DropoutResult lists.

    Parameters
    ----------
    output_dir : Path or str
        Directory where all artifacts are written.
    benchmark_results : list of ExperimentResult
        From ExperimentRunner.run().
    ablation_results : list of ExperimentResult
        From AblationFactory.run().
    dropout_results : list of DropoutResult
        From SensorDropoutStress.run().
    dpi : int
        Figure DPI (default 300 for publication quality).
    """

    def __init__(
        self,
        output_dir: Path | str,
        benchmark_results: Optional[List[ExperimentResult]] = None,
        ablation_results: Optional[List[ExperimentResult]] = None,
        dropout_results: Optional[List[DropoutResult]] = None,
        dpi: int = 300,
    ) -> None:
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.benchmark = benchmark_results or []
        self.ablation  = ablation_results or []
        self.dropout   = dropout_results or []
        self.dpi = dpi

    # ==================================================================
    # JSON
    # ==================================================================

    def save_json(self, filename: str = "results.json") -> Path:
        """Write combined results to JSON."""
        def _result_dict(r: ExperimentResult) -> Dict[str, Any]:
            return {
                "n_params": r.n_params,
                "latency_ms": r.latency_ms,
                "peak_vram_mb": r.peak_vram_mb,
                "inference_device": r.inference_device,
                "per_horizon": r.per_horizon,
                "aggregate": r.aggregate,
            }

        def _dropout_dict(r: DropoutResult) -> Dict[str, Any]:
            return {
                "per_horizon": r.per_horizon,
                "csi_retained_pct": r.csi_retained_pct,
                "aggregate_csi": r.aggregate_csi,
                "aggregate_ece": r.aggregate_ece,
            }

        payload = {
            "benchmark": {r.model_name: _result_dict(r) for r in self.benchmark},
            "ablation":  {r.model_name: _result_dict(r) for r in self.ablation},
            "dropout": {
                f"{r.model_name}|{r.scenario_name}": _dropout_dict(r)
                for r in self.dropout
            },
        }
        path = self.out / filename
        path.write_text(json.dumps(payload, indent=2, default=str))
        log.info(f"[Artifacts] Saved JSON → {path}")
        return path

    # ==================================================================
    # Markdown helpers
    # ==================================================================

    @staticmethod
    def _md_benchmark_table(results: List[ExperimentResult]) -> str:
        """GitHub-style Markdown table for benchmark / ablation results."""
        lines = [
            "| Model | #Params | Lat (ms) | VRAM (MB) | "
            "CSI@T+15 | CSI@T+30 | CSI@T+45 | CSI@T+60 | "
            "Mean CSI | Mean Brier | Mean ECE |",
            "|:------|--------:|---------:|----------:|"
            "--------:|---------:|---------:|---------:|"
            "---------:|-----------:|--------:|",
        ]
        for r in results:
            csi_per = [
                h.get("csi_30", float("nan")) for h in r.per_horizon
            ]
            # Pad to 4 horizons
            while len(csi_per) < 4:
                csi_per.append(float("nan"))
            csi_str = " | ".join(_safe(c) for c in csi_per[:4])
            mean_csi   = _safe(r.aggregate.get("csi_30", float("nan")))
            mean_brier = _safe(r.aggregate.get("brier", float("nan")))
            mean_ece   = _safe(r.aggregate.get("ece", float("nan")))
            lines.append(
                f"| {r.model_name} | {_int_or_dash(r.n_params)} | "
                f"{r.latency_ms:.1f} | {r.peak_vram_mb:.0f} | "
                f"{csi_str} | {mean_csi} | {mean_brier} | {mean_ece} |"
            )
        return "\n".join(lines)

    @staticmethod
    def _md_dropout_table(results: List[DropoutResult]) -> str:
        """GitHub-style Markdown table for sensor-dropout results."""
        # Group by scenario
        scenarios = []
        seen = set()
        for r in results:
            if r.scenario_name not in seen:
                scenarios.append(r.scenario_name)
                seen.add(r.scenario_name)

        models = []
        seen_m = set()
        for r in results:
            if r.model_name not in seen_m:
                models.append(r.model_name)
                seen_m.add(r.model_name)

        # Build lookup
        idx: Dict[str, Dict[str, DropoutResult]] = {}
        for r in results:
            idx.setdefault(r.model_name, {})[r.scenario_name] = r

        lines = [
            "| Model | Scenario | CSI@T+15 | CSI@T+30 | CSI@T+45 | CSI@T+60 | "
            "Mean CSI | Mean ECE | Ret% (mean) |",
            "|:------|:---------|--------:|---------:|---------:|---------:|"
            "---------:|--------:|------------:|",
        ]
        for m in models:
            for s in scenarios:
                r = idx.get(m, {}).get(s)
                if r is None:
                    continue
                csi_vals = [h.get("csi_30", float("nan")) for h in r.per_horizon]
                while len(csi_vals) < 4:
                    csi_vals.append(float("nan"))
                csi_str = " | ".join(_safe(c) for c in csi_vals[:4])
                retained = r.csi_retained_pct
                while len(retained) < 4:
                    retained.append(float("nan"))
                mean_ret = sum(x for x in retained if not math.isnan(x))
                cnt = sum(1 for x in retained if not math.isnan(x))
                mean_ret_str = f"{mean_ret/cnt:.1f}%" if cnt > 0 else "—"
                lines.append(
                    f"| {m} | {s} | {csi_str} | "
                    f"{_safe(r.aggregate_csi)} | {_safe(r.aggregate_ece)} | "
                    f"{mean_ret_str} |"
                )
        return "\n".join(lines)

    def save_markdown(self) -> Dict[str, Path]:
        """Write all Markdown tables."""
        paths: Dict[str, Path] = {}

        if self.benchmark:
            p = self.out / "benchmark.md"
            p.write_text(
                "# APM-Nowcast Benchmark Results\n\n"
                + self._md_benchmark_table(self.benchmark)
                + "\n"
            )
            log.info(f"[Artifacts] Saved Markdown → {p}")
            paths["benchmark"] = p

        if self.ablation:
            p = self.out / "ablation.md"
            p.write_text(
                "# APM-Nowcast Ablation Study\n\n"
                + self._md_benchmark_table(self.ablation)
                + "\n"
            )
            log.info(f"[Artifacts] Saved Markdown → {p}")
            paths["ablation"] = p

        if self.dropout:
            p = self.out / "dropout.md"
            p.write_text(
                "# APM-Nowcast Sensor-Dropout Robustness\n\n"
                + self._md_dropout_table(self.dropout)
                + "\n"
            )
            log.info(f"[Artifacts] Saved Markdown → {p}")
            paths["dropout"] = p

        return paths

    # ==================================================================
    # LaTeX booktabs helpers
    # ==================================================================

    @staticmethod
    def _tex_escape(s: str) -> str:
        return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

    def _latex_benchmark_table(self, results: List[ExperimentResult]) -> str:
        """Generate a LaTeX booktabs table for benchmark / ablation."""
        rows = []
        for r in results:
            csi_vals = [h.get("csi_30", float("nan")) for h in r.per_horizon]
            while len(csi_vals) < 4:
                csi_vals.append(float("nan"))
            csi_str = " & ".join(_safe(c) for c in csi_vals[:4])
            rows.append(
                f"    {self._tex_escape(r.model_name)} & "
                f"{_int_or_dash(r.n_params)} & "
                f"{r.latency_ms:.1f} & {r.peak_vram_mb:.0f} & "
                f"{csi_str} & "
                f"{_safe(r.aggregate.get('csi_30', float('nan')))} & "
                f"{_safe(r.aggregate.get('brier', float('nan')))} & "
                f"{_safe(r.aggregate.get('ece', float('nan')))} \\\\"
            )

        body = "\n    \\midrule\n".join(rows)
        return (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{APM-Nowcast Benchmark: Multi-Horizon Skill Scores}\n"
            "\\label{tab:benchmark}\n"
            "\\begin{tabular}{lrrrcccccc}\n"
            "\\toprule\n"
            "    Model & \\#Params & Lat (ms) & VRAM & "
            "CSI\\textsubscript{+15} & CSI\\textsubscript{+30} & "
            "CSI\\textsubscript{+45} & CSI\\textsubscript{+60} & "
            "\\={CSI} & \\={Brier} & \\={ECE} \\\\\n"
            "\\midrule\n"
            + body
            + "\n\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )

    def _latex_ablation_table(self, results: List[ExperimentResult]) -> str:
        rows = []
        for r in results:
            csi_vals = [h.get("csi_30", float("nan")) for h in r.per_horizon]
            while len(csi_vals) < 4:
                csi_vals.append(float("nan"))
            csi_str = " & ".join(_safe(c) for c in csi_vals[:4])
            rows.append(
                f"    {self._tex_escape(r.model_name)} & "
                f"{csi_str} & "
                f"{_safe(r.aggregate.get('csi_30', float('nan')))} & "
                f"{_safe(r.aggregate.get('brier', float('nan')))} & "
                f"{_safe(r.aggregate.get('ece', float('nan')))} \\\\"
            )

        body = "\n    \\midrule\n".join(rows)
        return (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{APM-Nowcast Ablation Study}\n"
            "\\label{tab:ablation}\n"
            "\\begin{tabular}{lcccccc}\n"
            "\\toprule\n"
            "    Configuration & "
            "CSI\\textsubscript{+15} & CSI\\textsubscript{+30} & "
            "CSI\\textsubscript{+45} & CSI\\textsubscript{+60} & "
            "\\={CSI} & \\={Brier} & \\={ECE} \\\\\n"
            "\\midrule\n"
            + body
            + "\n\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )

    def _latex_dropout_table(self, results: List[DropoutResult]) -> str:
        rows = []
        for r in results:
            csi_vals = [h.get("csi_30", float("nan")) for h in r.per_horizon]
            while len(csi_vals) < 4:
                csi_vals.append(float("nan"))
            csi_str = " & ".join(_safe(c) for c in csi_vals[:4])
            retained = r.csi_retained_pct
            while len(retained) < 4:
                retained.append(float("nan"))
            mean_ret = [x for x in retained if not math.isnan(x)]
            mean_ret_str = (f"{sum(mean_ret)/len(mean_ret):.1f}"
                            if mean_ret else "—")
            rows.append(
                f"    {self._tex_escape(r.model_name)} & "
                f"{self._tex_escape(r.scenario_name)} & "
                f"{csi_str} & "
                f"{_safe(r.aggregate_csi)} & "
                f"{_safe(r.aggregate_ece)} & "
                f"{mean_ret_str}\\% \\\\"
            )

        body = "\n    \\midrule\n".join(rows)
        return (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{Sensor-Dropout Robustness Benchmark}\n"
            "\\label{tab:dropout}\n"
            "\\begin{tabular}{llcccccc}\n"
            "\\toprule\n"
            "    Model & Scenario & "
            "CSI\\textsubscript{+15} & CSI\\textsubscript{+30} & "
            "CSI\\textsubscript{+45} & CSI\\textsubscript{+60} & "
            "\\={CSI} & \\={ECE} & Ret\\% \\\\\n"
            "\\midrule\n"
            + body
            + "\n\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )

    def save_latex(self) -> Dict[str, Path]:
        """Write all LaTeX tables."""
        paths: Dict[str, Path] = {}

        if self.benchmark:
            p = self.out / "benchmark.tex"
            p.write_text(self._latex_benchmark_table(self.benchmark))
            log.info(f"[Artifacts] Saved LaTeX → {p}")
            paths["benchmark"] = p

        if self.ablation:
            p = self.out / "ablation.tex"
            p.write_text(self._latex_ablation_table(self.ablation))
            log.info(f"[Artifacts] Saved LaTeX → {p}")
            paths["ablation"] = p

        if self.dropout:
            p = self.out / "dropout.tex"
            p.write_text(self._latex_dropout_table(self.dropout))
            log.info(f"[Artifacts] Saved LaTeX → {p}")
            paths["dropout"] = p

        return paths

    # ==================================================================
    # Matplotlib figures (Agg backend – headless safe)
    # ==================================================================

    def save_skill_by_horizon(
        self, filename: str = "skill_by_horizon.png"
    ) -> Path:
        """
        Line plot: CSI@0.3 vs. lead time (+15 … +60 min) for each model.
        """
        horizons = list(range(len(LEAD_TIME_LABELS)))
        fig, ax = plt.subplots(figsize=(8, 5))

        all_results = self.benchmark or self.ablation
        for i, r in enumerate(all_results):
            csi = [h.get("csi_30", float("nan")) for h in r.per_horizon]
            while len(csi) < 4:
                csi.append(float("nan"))
            csi = csi[:4]
            # Replace NaN with None for matplotlib gap handling
            y = [None if math.isnan(c) else c for c in csi]
            color = _COLORS[i % len(_COLORS)]
            linestyle = "-" if "APM" in r.model_name or "Proposed" in r.model_name else "--"
            ax.plot(
                horizons, y,
                label=r.model_name,
                color=color,
                linestyle=linestyle,
                linewidth=2,
                marker="o",
                markersize=5,
            )

        ax.set_xticks(horizons)
        ax.set_xticklabels(LEAD_TIME_LABELS, fontsize=10)
        ax.set_xlabel("Forecast Lead Time", fontsize=12)
        ax.set_ylabel("CSI @ 0.3 threshold", fontsize=12)
        ax.set_title("Nowcast Skill vs. Lead Time", fontsize=13, fontweight="bold")
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        path = self.out / filename
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        log.info(f"[Artifacts] Saved figure → {path}")
        return path

    def save_dropout_robustness(
        self, filename: str = "sensor_dropout_robustness.png"
    ) -> Path:
        """
        Grouped bar chart: mean CSI retained (%) per scenario for each model.
        """
        if not self.dropout:
            # Create a placeholder figure
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No dropout results available",
                    ha="center", va="center", fontsize=14)
            path = self.out / filename
            fig.savefig(path, dpi=self.dpi)
            plt.close(fig)
            return path

        # Gather unique scenarios (excluding Full Sensors) and models
        scenarios = [
            s for s in [
                r.scenario_name for r in self.dropout
                if r.scenario_name != "Full Sensors"
            ]
            if s not in [
                x for x in
                [rs for rs in []]  # dedup via seen set below
            ]
        ]
        seen = set()
        scenarios_ordered = []
        for r in self.dropout:
            if r.scenario_name != "Full Sensors" and r.scenario_name not in seen:
                scenarios_ordered.append(r.scenario_name)
                seen.add(r.scenario_name)

        models_ordered = []
        seen_m: set = set()
        for r in self.dropout:
            if r.model_name not in seen_m:
                models_ordered.append(r.model_name)
                seen_m.add(r.model_name)

        idx_map: Dict[str, Dict[str, DropoutResult]] = {}
        for r in self.dropout:
            idx_map.setdefault(r.model_name, {})[r.scenario_name] = r

        x = np.arange(len(scenarios_ordered))
        width = 0.8 / max(len(models_ordered), 1)
        fig, ax = plt.subplots(figsize=(10, 5))

        for mi, m in enumerate(models_ordered):
            retained_means = []
            for s in scenarios_ordered:
                r = idx_map.get(m, {}).get(s)
                if r is not None and r.csi_retained_pct:
                    vals = [v for v in r.csi_retained_pct if not math.isnan(v)]
                    retained_means.append(sum(vals) / len(vals) if vals else 0.0)
                else:
                    retained_means.append(0.0)
            offset = (mi - len(models_ordered) / 2 + 0.5) * width
            ax.bar(x + offset, retained_means, width * 0.9,
                   label=m, color=_COLORS[mi % len(_COLORS)], alpha=0.85)

        ax.axhline(100.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6,
                   label="Full-sensor baseline (100%)")
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios_ordered, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Mean CSI Retained (%)", fontsize=12)
        ax.set_title("Sensor-Dropout Robustness", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 115)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        path = self.out / filename
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        log.info(f"[Artifacts] Saved figure → {path}")
        return path

    def save_reliability_diagram(
        self, n_bins: int = 10, filename: str = "reliability_calibration.png"
    ) -> Path:
        """
        Reliability / calibration diagram.

        Since we don't have stored probability bins from the evaluator in
        quick_mode (zero ground-truth positives → perfect Brier on zeros),
        we generate a synthetic reliability curve for illustration, clearly
        labelled as 'Illustrative'.
        """
        fig, ax = plt.subplots(figsize=(6, 6))

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_mids  = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")

        all_results = self.benchmark or self.ablation
        for i, r in enumerate(all_results):
            # ECE-based synthetic reliability curve (illustrative)
            ece_val = r.aggregate.get("ece", 0.05)
            if math.isnan(ece_val):
                ece_val = 0.05
            # Add a slight over-confidence bias proportional to ECE
            bias = ece_val * np.sin(np.pi * bin_mids)
            frac_pos = np.clip(bin_mids - bias, 0, 1)
            color = _COLORS[i % len(_COLORS)]
            ax.plot(
                bin_mids, frac_pos,
                label=f"{r.model_name} (ECE={ece_val:.4f})",
                color=color,
                linewidth=1.8,
                marker=".",
                markersize=4,
            )

        ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray", label="Calibration gap")
        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        ax.set_title("Reliability Diagram (Illustrative)", fontsize=13, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        path = self.out / filename
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        log.info(f"[Artifacts] Saved figure → {path}")
        return path

    # ==================================================================
    # One-shot generate all
    # ==================================================================

    def generate_all(self) -> Dict[str, Path]:
        """Generate every artifact and return a dict of name → Path."""
        paths: Dict[str, Path] = {}

        p = self.save_json()
        paths["json"] = p

        md_paths = self.save_markdown()
        paths.update(md_paths)

        tex_paths = self.save_latex()
        paths.update({f"{k}_tex": v for k, v in tex_paths.items()})

        if self.benchmark or self.ablation:
            paths["skill_png"] = self.save_skill_by_horizon()

        if self.dropout:
            paths["dropout_png"] = self.save_dropout_robustness()

        if self.benchmark or self.ablation:
            paths["reliability_png"] = self.save_reliability_diagram()

        log.info(f"[Artifacts] Generated {len(paths)} artifacts in {self.out}")
        return paths
