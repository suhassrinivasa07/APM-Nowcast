"""
Unit tests for models/base.py (BaseNowcastModel contract and helpers).
"""
import pytest
import torch
import torch.nn as nn

from models.base import BaseNowcastModel


class DummyNowcastModel(BaseNowcastModel):
    """Minimal concrete implementation for testing base model capabilities."""

    def __init__(self, T_in: int = 4, T_out: int = 4) -> None:
        super().__init__(T_in=T_in, T_out=T_out)
        self.conv = nn.Conv2d(11, 8, kernel_size=3, padding=1)
        self.out_occ = nn.Conv2d(8, T_out, kernel_size=1)
        self.out_rate = nn.Conv2d(8, T_out, kernel_size=1)

    def forward(self, inputs_by_modality, modality_masks, **kwargs):
        fused, _ = self.fuse_multimodal_inputs(inputs_by_modality, modality_masks)
        # Take last time step
        x = fused[:, -1]
        feat = torch.relu(self.conv(x))
        occ = self.out_occ(feat).unsqueeze(2)  # [B, T_out, 1, H, W]
        rate = torch.relu(self.out_rate(feat)).unsqueeze(2)

        outputs = {
            "lightning_logits": occ,
            "lightning_probs": torch.sigmoid(occ),
            "flash_rate_pred": rate,
        }
        is_valid = self.validate_output_contract(
            outputs,
            expected_B=fused.shape[0],
            expected_T_out=self.T_out,
            expected_H=fused.shape[-2],
            expected_W=fused.shape[-1],
        )
        assert is_valid, "Output contract validation failed"
        return outputs


def test_base_parameter_counter():
    model = DummyNowcastModel(T_in=4, T_out=4)
    trainable_params = model.count_parameters(trainable_only=True)
    total_params = model.count_parameters(trainable_only=False)

    assert trainable_params > 0
    assert trainable_params == total_params

    # Freeze conv
    for p in model.conv.parameters():
        p.requires_grad = False

    assert model.count_parameters(trainable_only=True) < total_params


def test_base_vram_estimation():
    model = DummyNowcastModel(T_in=4, T_out=4)
    vram_mb_fp32 = model.estimate_vram_mb(batch_size=4, patch_size=(64, 64), dtype=torch.float32)
    vram_mb_fp16 = model.estimate_vram_mb(batch_size=4, patch_size=(64, 64), dtype=torch.float16)

    assert vram_mb_fp32 > 0
    assert vram_mb_fp16 < vram_mb_fp32
    assert vram_mb_fp16 == pytest.approx(vram_mb_fp32 / 2.0, rel=1e-2)


def test_fuse_multimodal_inputs(sample_batch, sample_batch_missing_radar):
    model = DummyNowcastModel()

    # Case 1: Complete batch
    fused, total_c = model.fuse_multimodal_inputs(
        sample_batch["inputs_by_modality"],
        sample_batch["modality_masks"],
    )
    assert fused.shape == (2, 4, 11, 64, 64)
    assert total_c == 11
    assert not torch.isnan(fused).any()

    # Case 2: Missing radar batch
    fused_missing, total_c = model.fuse_multimodal_inputs(
        sample_batch_missing_radar["inputs_by_modality"],
        sample_batch_missing_radar["modality_masks"],
    )
    assert fused_missing.shape == (2, 4, 11, 64, 64)
    assert not torch.isnan(fused_missing).any()
    # Radar channels (indices 3 and 4) should be zeroed out
    radar_slice = fused_missing[:, :, 3:5, :, :]
    assert torch.all(radar_slice == 0.0)


def test_output_contract_validation(sample_batch):
    model = DummyNowcastModel(T_in=4, T_out=4)
    outputs = model(
        sample_batch["inputs_by_modality"],
        sample_batch["modality_masks"],
    )

    assert "lightning_logits" in outputs
    assert "lightning_probs" in outputs
    assert "flash_rate_pred" in outputs

    for k in ("lightning_logits", "lightning_probs", "flash_rate_pred"):
        assert outputs[k].shape == (2, 4, 1, 64, 64)

    # Invalidate output shape to check assertion
    bad_outputs = {
        "lightning_logits": outputs["lightning_logits"][:, :2],  # wrong T_out
        "lightning_probs": outputs["lightning_probs"],
        "flash_rate_pred": outputs["flash_rate_pred"],
    }
    assert not BaseNowcastModel.validate_output_contract(
        bad_outputs, expected_B=2, expected_T_out=4, expected_H=64, expected_W=64
    )
