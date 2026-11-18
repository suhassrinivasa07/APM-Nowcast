"""
experiments/__init__.py

APM-Nowcast Module 11 – Experiments, Ablation Suite & Paper Artifacts Generator.

Public API
----------
ExperimentRunner     - orchestrates benchmark + ablation runs
AblationFactory      - builds ablation model/config variants
SensorDropoutStress  - sensor-outage robustness benchmark
ArtifactGenerator   - exports LaTeX tables, Markdown, JSON, PNG figures
"""
from experiments.runner import ExperimentRunner, ExperimentResult
from experiments.ablations import AblationFactory, AblationConfig
from experiments.dropout_stress import SensorDropoutStress, DropoutResult
from experiments.artifacts import ArtifactGenerator

__all__ = [
    "ExperimentRunner",
    "ExperimentResult",
    "AblationFactory",
    "AblationConfig",
    "SensorDropoutStress",
    "DropoutResult",
    "ArtifactGenerator",
]
