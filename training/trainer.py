"""
APMTrainer — RTX 3050-Optimized Training Loop for APM-Nowcast.

Supports any BaseNowcastModel (APMNowcastModel, ConvLSTM, UNet) with:
  - torch.amp.autocast + GradScaler (FP16, graceful FP32 CPU fallback)
  - Configurable gradient accumulation (effective batch scaling on 4 GB VRAM)
  - Cosine LR schedule with linear warmup
  - Gradient norm clipping (max_norm=1.0)
  - Early stopping (maximize val_csi by default)
  - Per-epoch JSON history logging
  - Explicit peak VRAM tracking via torch.cuda.max_memory_allocated()
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.logging_config import get_logger
from datacube.schema import NUM_MODALITIES
from losses.composite_loss import APMCompositeLoss
from models.base import BaseNowcastModel
from training.checkpoint import CheckpointManager
from training.config import TrainingConfig
from training.metrics import compute_all_metrics

log = get_logger(__name__)


def _cosine_lr_lambda(
    epoch: int,
    warmup_epochs: int,
    total_epochs: int,
    eta_min_ratio: float = 0.0,
) -> float:
    """Linear warmup then cosine decay LR multiplier."""
    if epoch < warmup_epochs:
        return max(1e-6, epoch / max(1, warmup_epochs))
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1.0 + math.cos(math.pi * progress))


def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move all tensors in a DataLoader batch dict to target device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) if isinstance(vv, torch.Tensor) else vv
                      for kk, vv in v.items()}
        else:
            out[k] = v
    return out


class APMTrainer:
    """
    Training harness for APM-Nowcast models.

    Accepts any BaseNowcastModel and synthetic or real DataLoader batches
    with the structure returned by AtmosphericDataCubeDataset.
    """

    def __init__(
        self,
        model: BaseNowcastModel,
        config: TrainingConfig,
        train_loader: Iterable,
        val_loader: Iterable,
        normalizer_stats: Optional[Dict[str, Any]] = None,
        model_class_name: str = "",
        model_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.normalizer_stats = normalizer_stats or {}
        self.model_class_name = model_class_name
        self.model_kwargs = model_kwargs or {}

        # Device selection
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = self.model.to(self.device)

        # AMP setup
        self.use_amp = config.amp_enabled and self.device.type == "cuda"
        self.scaler: Optional[torch.amp.GradScaler] = (
            torch.amp.GradScaler("cuda") if self.use_amp else None
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )

        # LR scheduler (cosine with warmup)
        eta_min_ratio = config.cosine_eta_min / max(config.lr, 1e-10)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda ep: _cosine_lr_lambda(
                ep, config.warmup_epochs, config.epochs, eta_min_ratio
            ),
        )

        # Loss function
        lw = config.loss_weights
        self.criterion = APMCompositeLoss(
            w_occ=lw.w_occ,
            w_int=lw.w_int,
            w_phys=lw.w_phys,
            w_cal=lw.w_cal,
            focal_gamma=lw.focal_gamma,
            focal_alpha=lw.focal_alpha,
        )

        # Checkpoint manager
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            keep_last_n=config.checkpoint_keep_last_n,
            mode=config.early_stopping_mode,
        )

        # Training state
        self.history: List[Dict[str, Any]] = []
        self._best_val_metric: Optional[float] = None
        self._epochs_no_improve = 0
        self._stopped_early = False

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch with gradient accumulation."""
        self.model.train()
        accum = {}
        step_count = 0
        accum_loss_sum = 0.0
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.train_loader):
            batch = _batch_to_device(batch, self.device)

            inputs   = batch.get("inputs_by_modality", {})
            masks    = batch.get("modality_masks", torch.ones(1, 1, NUM_MODALITIES, device=self.device))
            phys     = batch.get("physics_fields", None)

            # Build targets dict expected by APMCompositeLoss
            tgt_occ   = batch.get("target_occurrence_frames", None)
            tgt_flash = batch.get("target_lightning_frames",  None)

            if tgt_occ is None or tgt_flash is None:
                continue

            # Reshape targets: [B, T_out, H, W] → [B, T_out, 1, H, W]
            if tgt_occ.ndim == 4:
                tgt_occ   = tgt_occ.unsqueeze(2)
            if tgt_flash.ndim == 4:
                tgt_flash = tgt_flash.unsqueeze(2)

            targets = {
                "target_occurrence":  tgt_occ,
                "target_flash_rate":  tgt_flash,
            }

            # Forward pass under AMP
            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = self.model(inputs, masks)
                    loss, loss_metrics = self.criterion(outputs, targets, phys)
                    loss = loss / self.config.grad_accum_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(inputs, masks)
                loss, loss_metrics = self.criterion(outputs, targets, phys)
                loss = loss / self.config.grad_accum_steps
                loss.backward()

            accum_loss_sum += loss.item() * self.config.grad_accum_steps

            # Accumulate sub-loss stats
            for k, v in loss_metrics.items():
                accum[k] = accum.get(k, 0.0) + v

            # Gradient step every grad_accum_steps
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                step_count += 1

        # Flush any remaining accumulated gradients
        loader_len = len(self.train_loader) if hasattr(self.train_loader, "__len__") else (batch_idx + 1)
        if step_count == 0 or loader_len % self.config.grad_accum_steps != 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        n = max(1, loader_len)
        return {k: v / n for k, v in accum.items()}

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _val_epoch(self) -> Dict[str, float]:
        """Evaluate model on validation loader, return aggregated metrics."""
        self.model.eval()
        all_probs, all_flash_pred, all_occ, all_flash = [], [], [], []

        for batch in self.val_loader:
            batch = _batch_to_device(batch, self.device)
            inputs = batch.get("inputs_by_modality", {})
            masks  = batch.get("modality_masks", torch.ones(1, 1, NUM_MODALITIES, device=self.device))

            tgt_occ   = batch.get("target_occurrence_frames", None)
            tgt_flash = batch.get("target_lightning_frames",  None)
            if tgt_occ is None or tgt_flash is None:
                continue
            if tgt_occ.ndim == 4:
                tgt_occ   = tgt_occ.unsqueeze(2)
            if tgt_flash.ndim == 4:
                tgt_flash = tgt_flash.unsqueeze(2)

            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = self.model(inputs, masks)
            else:
                outputs = self.model(inputs, masks)

            all_probs.append(outputs["lightning_probs"].float().cpu())
            all_flash_pred.append(outputs["flash_rate_pred"].float().cpu())
            all_occ.append(tgt_occ.float().cpu())
            all_flash.append(tgt_flash.float().cpu())

        if not all_probs:
            return {"val_csi": 0.0, "val_fss": 0.0, "val_brier": 1.0, "val_ece": 1.0}

        probs      = torch.cat(all_probs,      dim=0)
        flash_pred = torch.cat(all_flash_pred, dim=0)
        occ        = torch.cat(all_occ,        dim=0)
        flash      = torch.cat(all_flash,      dim=0)

        cfg = self.config
        m = compute_all_metrics(
            probs, flash_pred, occ, flash,
            thresholds=cfg.prob_thresholds,
            fss_scales=cfg.fss_scales,
        )
        # Prefix val_
        return {
            "val_csi":        m.get("csi_50", 0.0),
            "val_pod":        m.get("pod_50", 0.0),
            "val_far":        m.get("far_50", 0.0),
            "val_fss":        m.get("fss_mean", 0.0),
            "val_brier":      m.get("brier", 1.0),
            "val_ece":        m.get("ece", 1.0),
            "val_flash_rmse": m.get("flash_rmse", 0.0),
        }

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> List[Dict[str, Any]]:
        """
        Run the full training loop for config.epochs epochs.

        Returns the per-epoch history list (also saved to checkpoint_dir/history.json).
        """
        log.info(f"Training on device: {self.device}")
        log.info(f"AMP enabled: {self.use_amp}")
        log.info(f"Model parameters: {self.model.count_parameters():,}")

        for epoch in range(self.config.epochs):
            t0 = time.time()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            # --- Train ---
            train_metrics = self._train_epoch(epoch)

            # --- Validate ---
            val_metrics = self._val_epoch()

            # --- LR step ---
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # --- Peak VRAM ---
            peak_vram_mb = 0.0
            if self.device.type == "cuda":
                peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

            # --- Epoch record ---
            epoch_record: Dict[str, Any] = {
                "epoch": epoch,
                "lr": current_lr,
                "peak_vram_mb": round(peak_vram_mb, 1),
                "elapsed_s": round(time.time() - t0, 2),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **val_metrics,
            }
            self.history.append(epoch_record)

            if epoch % self.config.log_every_n_epochs == 0:
                log.info(
                    f"Epoch {epoch:03d} | "
                    f"lr={current_lr:.2e} | "
                    f"loss={train_metrics.get('loss_total', 0):.4f} | "
                    f"val_csi={val_metrics.get('val_csi', 0):.4f} | "
                    f"val_brier={val_metrics.get('val_brier', 1):.4f} | "
                    f"vram={peak_vram_mb:.0f} MB"
                )

            # --- Checkpoint ---
            metric_key = self.config.early_stopping_metric
            tracked = val_metrics.get(metric_key, 0.0)
            self.ckpt_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                epoch=epoch,
                val_metrics=val_metrics,
                config_dict=self.config.to_dict(),
                normalizer_stats=self.normalizer_stats,
                model_class_name=self.model_class_name,
                model_kwargs=self.model_kwargs,
                tracked_metric=tracked,
            )

            # --- Early stopping ---
            improved = (
                self._best_val_metric is None
                or (self.config.early_stopping_mode == "max" and tracked > self._best_val_metric)
                or (self.config.early_stopping_mode == "min" and tracked < self._best_val_metric)
            )
            if improved:
                self._best_val_metric = tracked
                self._epochs_no_improve = 0
            else:
                self._epochs_no_improve += 1
                if self._epochs_no_improve >= self.config.early_stopping_patience:
                    log.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {self.config.early_stopping_patience} epochs)"
                    )
                    self._stopped_early = True
                    break

        # Save full history
        history_path = Path(self.config.checkpoint_dir) / "history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        log.info(f"Training history saved to {history_path}")

        return self.history
