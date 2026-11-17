"""
Unit tests for models/baselines/persistence.py (Eulerian & Advection Baselines).
"""
import pytest
import torch

from models.baselines.persistence import AdvectionPersistenceModel, EulerianPersistenceModel


def test_eulerian_persistence_zero_params():
    model = EulerianPersistenceModel(T_in=4, T_out=4)
    assert model.count_parameters() == 0


def test_eulerian_persistence_forward(sample_batch):
    model = EulerianPersistenceModel(T_in=4, T_out=4)
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

    # Probabilities in [0, 1]
    assert (outputs["lightning_probs"] >= 0.0).all()
    assert (outputs["lightning_probs"] <= 1.0).all()

    # Flash rate non-negative
    assert (outputs["flash_rate_pred"] >= 0.0).all()

    # Persistence verification: forecast horizons should be identical across time
    for t in range(1, T_out):
        assert torch.allclose(outputs["lightning_probs"][:, 0], outputs["lightning_probs"][:, t])


def test_eulerian_persistence_missing_lightning(sample_batch):
    model = EulerianPersistenceModel(T_in=4, T_out=4)
    inputs = dict(sample_batch["inputs_by_modality"])
    inputs["lightning"] = None

    masks = sample_batch["modality_masks"].clone()
    masks[:, :, 2] = 0.0  # lightning index

    outputs = model(inputs, masks)
    assert (outputs["lightning_probs"] == 0.0).all()
    assert (outputs["flash_rate_pred"] == 0.0).all()


def test_advection_persistence_zero_params():
    model = AdvectionPersistenceModel(T_in=4, T_out=4)
    assert model.count_parameters() == 0


def test_advection_persistence_forward(sample_batch):
    model = AdvectionPersistenceModel(T_in=4, T_out=4)
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

    assert (outputs["lightning_probs"] >= 0.0).all()
    assert (outputs["lightning_probs"] <= 1.0).all()
    assert (outputs["flash_rate_pred"] >= 0.0).all()
    assert not torch.isnan(outputs["lightning_probs"]).any()
    assert not torch.isnan(outputs["flash_rate_pred"]).any()


def test_advection_persistence_fallback_to_eulerian_when_motion_unavailable(sample_batch):
    model = AdvectionPersistenceModel(T_in=4, T_out=4)
    # Mask both radar and satellite
    inputs = dict(sample_batch["inputs_by_modality"])
    inputs["radar"] = None
    inputs["satellite"] = None

    masks = sample_batch["modality_masks"].clone()
    masks[:, :, 0] = 0.0  # satellite
    masks[:, :, 1] = 0.0  # radar

    outputs = model(inputs, masks)
    # When radar/satellite motion is unavailable, fallback to Eulerian persistence
    assert outputs["lightning_probs"].shape == (2, 4, 1, 64, 64)
    assert not torch.isnan(outputs["lightning_probs"]).any()
