"""
Best-Model & Latest Checkpoint Saver/Loader for APM-Nowcast.

Saves fully reproducible `.pt` checkpoints containing:
  - model state_dict
  - optimizer state_dict
  - GradScaler state_dict (AMP)
  - TrainingConfig
  - ModalityNormalizer statistics (JSON-serializable)
  - Epoch number and validation metrics

Enables Module 10 (FastAPI inference) to load a trained artifact cleanly
without requiring the original training code path.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from common.logging_config import get_logger

log = get_logger(__name__)


class CheckpointManager:
    """
    Manages saving and loading of model checkpoints with optional
    best-model tracking and rolling retention of the last N checkpoints.
    """

    BEST_FILENAME  = "best_model.pt"
    LATEST_FILENAME = "latest_checkpoint.pt"

    def __init__(
        self,
        checkpoint_dir: str | Path,
        keep_last_n: int = 3,
        mode: str = "max",             # 'max' or 'min' for the tracked metric
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.mode = mode
        self._best_metric: Optional[float] = None
        self._epoch_files: list[Path] = []

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler],
        epoch: int,
        val_metrics: Dict[str, float],
        config_dict: Dict[str, Any],
        normalizer_stats: Optional[Dict[str, Any]] = None,
        model_class_name: str = "",
        model_kwargs: Optional[Dict[str, Any]] = None,
        tracked_metric: Optional[float] = None,
    ) -> Path:
        """
        Save a full training checkpoint and update best-model if improved.

        Parameters
        ----------
        model           : trained nn.Module
        optimizer       : current optimizer
        scaler          : AMP GradScaler (may be None for CPU)
        epoch           : completed epoch index (0-based)
        val_metrics     : dict of validation metric scalars
        config_dict     : TrainingConfig.to_dict() snapshot
        normalizer_stats: ModalityNormalizer raw stats dict for portability
        model_class_name: string name for re-instantiation (e.g. 'APMNowcastModel')
        model_kwargs    : constructor kwargs for re-instantiation
        tracked_metric  : scalar to compare for best-model tracking

        Returns
        -------
        Path of the saved checkpoint file
        """
        payload = {
            "epoch": epoch,
            "model_class_name": model_class_name,
            "model_kwargs": model_kwargs or {},
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "val_metrics": val_metrics,
            "training_config": config_dict,
            "normalizer_stats": normalizer_stats or {},
        }

        # --- Save epoch-specific checkpoint ---
        epoch_file = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(payload, epoch_file)
        self._epoch_files.append(epoch_file)
        log.info(f"Checkpoint saved: {epoch_file}")

        # Rolling retention: delete oldest beyond keep_last_n
        while len(self._epoch_files) > self.keep_last_n:
            old = self._epoch_files.pop(0)
            if old.exists() and old != self.checkpoint_dir / self.BEST_FILENAME:
                old.unlink()
                log.debug(f"Removed old checkpoint: {old}")

        # --- Always overwrite latest ---
        shutil.copy2(epoch_file, self.checkpoint_dir / self.LATEST_FILENAME)

        # --- Update best-model if metric improved ---
        if tracked_metric is not None:
            improved = (
                self._best_metric is None
                or (self.mode == "max" and tracked_metric > self._best_metric)
                or (self.mode == "min" and tracked_metric < self._best_metric)
            )
            if improved:
                self._best_metric = tracked_metric
                shutil.copy2(epoch_file, self.checkpoint_dir / self.BEST_FILENAME)
                log.info(f"New best model (metric={tracked_metric:.4f}) saved.")

        return epoch_file

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @staticmethod
    def load(
        path: str | Path,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        map_location: str | torch.device = "cpu",
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        If `model` is provided, its weights are restored in-place.
        If `optimizer` / `scaler` are provided, their states are restored.

        Returns the full raw checkpoint dict for downstream use.
        """
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)

        if model is not None:
            model.load_state_dict(payload["model_state_dict"])
            log.info(f"Model weights loaded from {path}")

        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])

        if scaler is not None and payload.get("scaler_state_dict") is not None:
            scaler.load_state_dict(payload["scaler_state_dict"])

        return payload

    @staticmethod
    def load_best(
        checkpoint_dir: str | Path,
        model: Optional[nn.Module] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Load the best-model checkpoint."""
        best_path = Path(checkpoint_dir) / CheckpointManager.BEST_FILENAME
        return CheckpointManager.load(best_path, model=model, **kwargs)

    @staticmethod
    def load_latest(
        checkpoint_dir: str | Path,
        model: Optional[nn.Module] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Load the latest checkpoint."""
        latest_path = Path(checkpoint_dir) / CheckpointManager.LATEST_FILENAME
        return CheckpointManager.load(latest_path, model=model, **kwargs)
