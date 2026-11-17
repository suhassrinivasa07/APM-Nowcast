"""
Unit tests for models/baselines/convlstm.py (ConvLSTM Baseline).
"""
import pytest
import torch

from models.baselines.convlstm import ConvLSTMCell, ConvLSTMNowcastModel


def test_convlstm_cell_forward():
    cell = ConvLSTMCell(in_channels=16, hidden_channels=32, kernel_size=3)
    x = torch.randn(2, 16, 32, 32)

    # Initial state (None)
    h1, c1 = cell(x)
    assert h1.shape == (2, 32, 32, 32)
    assert c1.shape == (2, 32, 32, 32)

    # Step 2 with previous state
    h2, c2 = cell(x, (h1, c1))
    assert h2.shape == (2, 32, 32, 32)
    assert c2.shape == (2, 32, 32, 32)
    assert not torch.allclose(h1, h2)


def test_convlstm_parameter_budget():
    model = ConvLSTMNowcastModel(T_in=4, T_out=4, hidden_dims=(32, 64))
    param_count = model.count_parameters(trainable_only=True)

    # Must stay strictly below 2.5M parameters
    assert param_count < 2_500_000
    assert param_count > 10_000


def test_convlstm_forward(sample_batch):
    model = ConvLSTMNowcastModel(T_in=4, T_out=4, hidden_dims=(32, 64))
    outputs = model(
        sample_batch["inputs_by_modality"],
        sample_batch["modality_masks"],
    )

    B = sample_batch["batch_size"]
    T_out = sample_batch["T_out"]
    H = sample_batch["height"]
    W = sample_batch["width"]

    for key in ("lightning_logits", "lightning_probs", "flash_rate_pred"):
        assert key in outputs
        assert outputs[key].shape == (B, T_out, 1, H, W)

    # Value ranges
    assert (outputs["lightning_probs"] >= 0.0).all()
    assert (outputs["lightning_probs"] <= 1.0).all()
    assert (outputs["flash_rate_pred"] >= 0.0).all()
    assert not torch.isnan(outputs["lightning_logits"]).any()


def test_convlstm_backward_gradient_flow(sample_batch):
    model = ConvLSTMNowcastModel(T_in=4, T_out=4, hidden_dims=(32, 64))
    model.train()

    outputs = model(
        sample_batch["inputs_by_modality"],
        sample_batch["modality_masks"],
    )

    loss = outputs["lightning_probs"].sum() + outputs["flash_rate_pred"].sum()
    loss.backward()

    # Verify gradients exist and are not NaN
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"


def test_convlstm_missing_modality_robustness(sample_batch_missing_radar):
    model = ConvLSTMNowcastModel(T_in=4, T_out=4, hidden_dims=(32, 64))
    outputs = model(
        sample_batch_missing_radar["inputs_by_modality"],
        sample_batch_missing_radar["modality_masks"],
    )

    assert outputs["lightning_probs"].shape == (2, 4, 1, 64, 64)
    assert not torch.isnan(outputs["lightning_probs"]).any()
    assert not torch.isnan(outputs["flash_rate_pred"]).any()


def test_convlstm_cuda_and_fp16(sample_batch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    model = ConvLSTMNowcastModel(T_in=4, T_out=4, hidden_dims=(32, 64)).to(device)

    cuda_inputs = {
        k: v.to(device) if v is not None else None
        for k, v in sample_batch["inputs_by_modality"].items()
    }
    cuda_masks = sample_batch["modality_masks"].to(device)

    # PyTorch AMP autocast test (crucial for RTX 3050 4GB VRAM)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = model(cuda_inputs, cuda_masks)
        loss = outputs["lightning_logits"].sum() + outputs["flash_rate_pred"].sum()

    loss.backward()
    assert outputs["lightning_probs"].shape == (2, 4, 1, 64, 64)
    assert outputs["lightning_probs"].device.type == "cuda"
