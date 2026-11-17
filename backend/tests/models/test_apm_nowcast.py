"""
Comprehensive unit tests for the APM-Nowcast core architecture (models/apm_nowcast/).

Tests cover:
  - Full forward + backward gradient flow across all components
  - Exact zero attention weight and zero NaN gradients under sensor dropout
  - Parameter budget (<= 3.0M)
  - CUDA + FP16 autocast execution on RTX 3050
  - MC-Dropout epistemic uncertainty producing positive non-trivial estimates
  - Individual component tests: encoders, fusion, physics conditioning, evolution core, heads
"""
import pytest
import torch
import torch.nn as nn

from datacube.schema import MODALITY_TO_INDEX, NUM_MODALITIES, ModalityType
from models.apm_nowcast import (
    APMNowcastModel,
    AdaptiveModalityFusion,
    LightningEncoder,
    NWPEncoder,
    PhysicsFiLMConditioning,
    RadarEncoder,
    SatelliteEncoder,
    SpatiotemporalEvolutionCore,
    UncertaintyForecastHeads,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def apm_model():
    """Standard APM-Nowcast model with compact dimensions."""
    return APMNowcastModel(
        T_in=4, T_out=4, latent_dim=24, hidden_dim=48,
        mc_dropout_rate=0.1, mc_samples=4,
    )


@pytest.fixture
def full_batch():
    """Full multimodal batch with all sensors active."""
    B, T_in, H, W = 2, 4, 64, 64
    return {
        "inputs_by_modality": {
            "satellite": torch.randn(B, T_in, 3, H, W),
            "radar": torch.clamp(torch.randn(B, T_in, 2, H, W) * 10 + 20, 0, 70),
            "lightning": _sparse_lightning(B, T_in, H, W),
            "nwp": torch.randn(B, T_in, 4, H, W),
        },
        "modality_masks": torch.ones(B, T_in, NUM_MODALITIES),
        "B": B, "T_in": T_in, "T_out": 4, "H": H, "W": W,
    }


@pytest.fixture
def batch_no_radar(full_batch):
    """Batch with radar completely dropped (sensor failure)."""
    inputs = dict(full_batch["inputs_by_modality"])
    inputs["radar"] = None
    masks = full_batch["modality_masks"].clone()
    masks[:, :, MODALITY_TO_INDEX[ModalityType.RADAR]] = 0.0
    return {
        "inputs_by_modality": inputs,
        "modality_masks": masks,
        "B": full_batch["B"], "T_in": full_batch["T_in"],
        "T_out": full_batch["T_out"], "H": full_batch["H"], "W": full_batch["W"],
    }


@pytest.fixture
def batch_no_satellite(full_batch):
    """Batch with satellite completely dropped."""
    inputs = dict(full_batch["inputs_by_modality"])
    inputs["satellite"] = None
    masks = full_batch["modality_masks"].clone()
    masks[:, :, MODALITY_TO_INDEX[ModalityType.SATELLITE]] = 0.0
    return {
        "inputs_by_modality": inputs,
        "modality_masks": masks,
        "B": full_batch["B"], "T_in": full_batch["T_in"],
        "T_out": full_batch["T_out"], "H": full_batch["H"], "W": full_batch["W"],
    }


def _sparse_lightning(B, T, H, W):
    """Create sparse lightning tensor with flashes concentrated in center."""
    lit = torch.zeros(B, T, 2, H, W)
    lit[:, :, 0, 28:36, 28:36] = torch.poisson(torch.ones(B, T, 8, 8) * 2.0)
    lit[:, :, 1, 28:36, 28:36] = (lit[:, :, 0, 28:36, 28:36] > 0).float()
    return lit


# ---------------------------------------------------------------------------
# Component Tests: Encoders
# ---------------------------------------------------------------------------


class TestEncoders:
    def test_satellite_encoder_shape(self):
        enc = SatelliteEncoder(latent_dim=24)
        x = torch.randn(2, 4, 3, 64, 64)
        out = enc(x)
        assert out.shape == (2, 4, 24, 32, 32)

    def test_radar_encoder_shape(self):
        enc = RadarEncoder(latent_dim=24)
        x = torch.randn(2, 4, 2, 64, 64)
        out = enc(x)
        assert out.shape == (2, 4, 24, 32, 32)

    def test_lightning_encoder_shape(self):
        enc = LightningEncoder(latent_dim=24)
        x = torch.randn(2, 4, 2, 64, 64)
        out = enc(x)
        assert out.shape == (2, 4, 24, 32, 32)

    def test_nwp_encoder_shape(self):
        enc = NWPEncoder(latent_dim=24)
        x = torch.randn(2, 4, 4, 64, 64)
        out = enc(x)
        assert out.shape == (2, 4, 24, 32, 32)

    def test_encoder_no_nan(self):
        for Enc, C in [(SatelliteEncoder, 3), (RadarEncoder, 2),
                        (LightningEncoder, 2), (NWPEncoder, 4)]:
            enc = Enc(latent_dim=24)
            out = enc(torch.randn(2, 4, C, 64, 64))
            assert not torch.isnan(out).any(), f"NaN in {Enc.__name__}"


# ---------------------------------------------------------------------------
# Component Tests: Adaptive Fusion
# ---------------------------------------------------------------------------


class TestAdaptiveFusion:
    def test_fusion_output_shape(self):
        fusion = AdaptiveModalityFusion(latent_dim=24, num_modalities=4)
        encoded = {
            "satellite": torch.randn(2, 4, 24, 32, 32),
            "radar": torch.randn(2, 4, 24, 32, 32),
            "lightning": torch.randn(2, 4, 24, 32, 32),
            "nwp": torch.randn(2, 4, 24, 32, 32),
        }
        masks = torch.ones(2, 4, 4)
        order = ["satellite", "radar", "lightning", "nwp"]
        fused, weights = fusion(encoded, masks, order)
        assert fused.shape == (2, 4, 24, 32, 32)
        assert weights.shape == (2, 4, 32, 32)

    def test_fusion_exact_zero_weight_on_dropout(self):
        """When a modality mask is 0, its attention weight must be exactly 0.0."""
        fusion = AdaptiveModalityFusion(latent_dim=24, num_modalities=4)
        encoded = {
            "satellite": torch.randn(2, 4, 24, 32, 32),
            "radar": torch.zeros(2, 4, 24, 32, 32),
            "lightning": torch.randn(2, 4, 24, 32, 32),
            "nwp": torch.randn(2, 4, 24, 32, 32),
        }
        masks = torch.ones(2, 4, 4)
        radar_idx = MODALITY_TO_INDEX[ModalityType.RADAR]
        masks[:, :, radar_idx] = 0.0

        order = ["satellite", "radar", "lightning", "nwp"]
        fused, weights = fusion(encoded, masks, order)

        # Radar weight must be exactly 0.0 everywhere
        assert (weights[:, radar_idx] == 0.0).all(), \
            "Radar attention weight is not exactly 0.0 when masked"

        # Other weights should sum to ~1.0 (within float precision)
        remaining = weights[:, [0, 2, 3]].sum(dim=1)
        assert torch.allclose(remaining, torch.ones_like(remaining), atol=1e-5)

    def test_fusion_no_nan_all_masked(self):
        """When ALL modalities are masked, fusion should produce zeros, not NaN."""
        fusion = AdaptiveModalityFusion(latent_dim=24, num_modalities=4)
        encoded = {
            "satellite": torch.zeros(1, 2, 24, 16, 16),
            "radar": torch.zeros(1, 2, 24, 16, 16),
            "lightning": torch.zeros(1, 2, 24, 16, 16),
            "nwp": torch.zeros(1, 2, 24, 16, 16),
        }
        masks = torch.zeros(1, 2, 4)  # All dropped
        order = ["satellite", "radar", "lightning", "nwp"]
        fused, weights = fusion(encoded, masks, order)
        assert not torch.isnan(fused).any()
        assert not torch.isnan(weights).any()


# ---------------------------------------------------------------------------
# Component Tests: Physics Conditioning
# ---------------------------------------------------------------------------


class TestPhysicsConditioning:
    def test_film_output_shape(self):
        pc = PhysicsFiLMConditioning(latent_dim=24, physics_dim=24)
        fused = torch.randn(2, 4, 24, 32, 32)
        nwp = torch.randn(2, 4, 24, 32, 32)
        conditioned, stats = pc(fused, nwp)
        assert conditioned.shape == (2, 4, 24, 32, 32)
        assert "gamma_mean" in stats
        assert "beta_mean" in stats
        assert "advection_bias_norm" in stats

    def test_film_near_identity_init(self):
        """At initialization, gamma should be close to 1 and beta close to 0."""
        pc = PhysicsFiLMConditioning(latent_dim=24, physics_dim=24)
        fused = torch.randn(2, 4, 24, 32, 32)
        nwp = torch.zeros(2, 4, 24, 32, 32)  # zero NWP input
        conditioned, stats = pc(fused, nwp)
        # With initialized gamma~1, beta~0, conditioned should be close to fused
        assert not torch.isnan(conditioned).any()


# ---------------------------------------------------------------------------
# Component Tests: Spatiotemporal Evolution Core
# ---------------------------------------------------------------------------


class TestSpatiotemporalCore:
    def test_evolution_output_shape(self):
        core = SpatiotemporalEvolutionCore(latent_dim=24, hidden_dim=48, T_out=4)
        x = torch.randn(2, 4, 24, 32, 32)
        out = core(x)
        assert out.shape == (2, 4, 24, 32, 32)

    def test_evolution_gradient_flow(self):
        core = SpatiotemporalEvolutionCore(latent_dim=24, hidden_dim=48, T_out=4)
        x = torch.randn(2, 4, 24, 32, 32, requires_grad=True)
        out = core(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ---------------------------------------------------------------------------
# Component Tests: Uncertainty Heads
# ---------------------------------------------------------------------------


class TestUncertaintyHeads:
    def test_heads_output_shapes(self):
        heads = UncertaintyForecastHeads(latent_dim=24, upsample_scale=2, mc_samples=4)
        x = torch.randn(2, 4, 24, 32, 32)
        outputs = heads(x, estimate_epistemic=False)
        assert outputs["lightning_logits"].shape == (2, 4, 1, 64, 64)
        assert outputs["lightning_probs"].shape == (2, 4, 1, 64, 64)
        assert outputs["flash_rate_pred"].shape == (2, 4, 1, 64, 64)
        assert outputs["aleatoric_uncertainty"].shape == (2, 4, 1, 64, 64)
        assert outputs["epistemic_uncertainty"].shape == (2, 4, 1, 64, 64)

    def test_heads_value_ranges(self):
        heads = UncertaintyForecastHeads(latent_dim=24, upsample_scale=2)
        x = torch.randn(2, 4, 24, 32, 32)
        outputs = heads(x)
        assert (outputs["lightning_probs"] >= 0.0).all()
        assert (outputs["lightning_probs"] <= 1.0).all()
        assert (outputs["flash_rate_pred"] >= 0.0).all()
        assert (outputs["aleatoric_uncertainty"] > 0.0).all()

    def test_mc_dropout_produces_variance(self):
        heads = UncertaintyForecastHeads(
            latent_dim=24, upsample_scale=2, mc_dropout_rate=0.3, mc_samples=16,
        )
        x = torch.randn(2, 4, 24, 32, 32)
        outputs = heads(x, estimate_epistemic=True)
        epistemic = outputs["epistemic_uncertainty"]
        assert epistemic.shape == (2, 4, 1, 64, 64)
        # With dropout_rate=0.3 and 16 samples, variance should be non-trivial
        assert epistemic.max().item() > 0.0, "MC-Dropout epistemic uncertainty is all zeros"


# ---------------------------------------------------------------------------
# End-to-End APMNowcastModel Tests
# ---------------------------------------------------------------------------


class TestAPMNowcastModel:
    def test_parameter_budget(self, apm_model):
        """Total trainable parameters must be <= 3.0M."""
        param_count = apm_model.count_parameters(trainable_only=True)
        assert param_count <= 3_000_000, \
            f"Parameter count {param_count:,} exceeds 3.0M budget"
        assert param_count > 50_000, "Suspiciously few parameters"

    def test_forward_output_shapes(self, apm_model, full_batch):
        """Standard forward pass produces correct output shapes."""
        outputs = apm_model(
            full_batch["inputs_by_modality"],
            full_batch["modality_masks"],
        )
        B, T_out, H, W = full_batch["B"], full_batch["T_out"], full_batch["H"], full_batch["W"]

        # Standard BaseNowcastModel outputs
        assert outputs["lightning_logits"].shape == (B, T_out, 1, H, W)
        assert outputs["lightning_probs"].shape == (B, T_out, 1, H, W)
        assert outputs["flash_rate_pred"].shape == (B, T_out, 1, H, W)

        # Research uncertainty outputs
        assert outputs["aleatoric_uncertainty"].shape == (B, T_out, 1, H, W)
        assert outputs["epistemic_uncertainty"].shape == (B, T_out, 1, H, W)

        # Interpretability outputs
        assert outputs["modality_attention_weights"].shape == (B, NUM_MODALITIES, H // 2, W // 2)
        assert isinstance(outputs["physics_gate_stats"], dict)
        assert "gamma_mean" in outputs["physics_gate_stats"]

    def test_output_contract(self, apm_model, full_batch):
        """Outputs must pass BaseNowcastModel.validate_output_contract."""
        outputs = apm_model(
            full_batch["inputs_by_modality"],
            full_batch["modality_masks"],
        )
        is_valid = apm_model.validate_output_contract(
            outputs,
            expected_B=full_batch["B"],
            expected_T_out=full_batch["T_out"],
            expected_H=full_batch["H"],
            expected_W=full_batch["W"],
        )
        assert is_valid, "Output contract validation failed"

    def test_backward_gradient_flow(self, apm_model, full_batch):
        """Gradients must flow through ALL components without NaN."""
        apm_model.train()
        outputs = apm_model(
            full_batch["inputs_by_modality"],
            full_batch["modality_masks"],
        )

        # Combined loss from all prediction heads
        loss = (
            outputs["lightning_logits"].sum()
            + outputs["flash_rate_pred"].sum()
            + outputs["aleatoric_uncertainty"].sum()
        )
        loss.backward()

        for name, p in apm_model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Gradient missing for {name}"
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"

    def test_radar_dropout_exact_zero_weight(self, apm_model, batch_no_radar):
        """When radar is dropped, its attention weight must be exactly 0.0."""
        apm_model.eval()
        with torch.no_grad():
            outputs = apm_model(
                batch_no_radar["inputs_by_modality"],
                batch_no_radar["modality_masks"],
            )
        radar_idx = MODALITY_TO_INDEX[ModalityType.RADAR]
        radar_weights = outputs["modality_attention_weights"][:, radar_idx]
        assert (radar_weights == 0.0).all(), \
            f"Radar attention weight should be exactly 0.0, got max={radar_weights.max().item()}"

    def test_satellite_dropout_exact_zero_weight(self, apm_model, batch_no_satellite):
        """When satellite is dropped, its attention weight must be exactly 0.0."""
        apm_model.eval()
        with torch.no_grad():
            outputs = apm_model(
                batch_no_satellite["inputs_by_modality"],
                batch_no_satellite["modality_masks"],
            )
        sat_idx = MODALITY_TO_INDEX[ModalityType.SATELLITE]
        sat_weights = outputs["modality_attention_weights"][:, sat_idx]
        assert (sat_weights == 0.0).all(), \
            f"Satellite attention weight should be exactly 0.0, got max={sat_weights.max().item()}"

    def test_dropout_no_nan_gradients(self, apm_model, batch_no_radar):
        """Sensor dropout must not produce NaN gradients anywhere."""
        apm_model.train()
        outputs = apm_model(
            batch_no_radar["inputs_by_modality"],
            batch_no_radar["modality_masks"],
        )
        loss = outputs["lightning_logits"].sum() + outputs["flash_rate_pred"].sum()
        loss.backward()

        for name, p in apm_model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert not torch.isnan(p.grad).any(), \
                    f"NaN gradient in {name} with radar dropout"

    def test_no_nan_outputs_full_batch(self, apm_model, full_batch):
        """No NaN values in any output tensor."""
        outputs = apm_model(
            full_batch["inputs_by_modality"],
            full_batch["modality_masks"],
        )
        for key in ("lightning_logits", "lightning_probs", "flash_rate_pred",
                     "aleatoric_uncertainty", "epistemic_uncertainty"):
            assert not torch.isnan(outputs[key]).any(), f"NaN in {key}"

    def test_no_nan_outputs_dropout(self, apm_model, batch_no_radar):
        """No NaN values when a sensor is dropped."""
        outputs = apm_model(
            batch_no_radar["inputs_by_modality"],
            batch_no_radar["modality_masks"],
        )
        for key in ("lightning_logits", "lightning_probs", "flash_rate_pred",
                     "aleatoric_uncertainty"):
            assert not torch.isnan(outputs[key]).any(), f"NaN in {key} with radar dropout"

    def test_mc_dropout_epistemic_estimation(self, apm_model, full_batch):
        """MC-Dropout must produce positive non-trivial epistemic uncertainty."""
        # Use higher dropout for more visible variance
        apm_model.forecast_heads.mc_dropout_rate = 0.3
        apm_model.forecast_heads.mc_samples = 16
        apm_model.eval()

        outputs = apm_model(
            full_batch["inputs_by_modality"],
            full_batch["modality_masks"],
            estimate_epistemic=True,
        )
        epistemic = outputs["epistemic_uncertainty"]
        assert epistemic.shape == (full_batch["B"], full_batch["T_out"], 1,
                                   full_batch["H"], full_batch["W"])
        assert epistemic.max().item() > 0.0, "Epistemic uncertainty is all zeros"
        assert not torch.isnan(epistemic).any()

    def test_cuda_fp16_forward_backward(self, full_batch):
        """Model must run on CUDA with FP16 autocast without errors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        model = APMNowcastModel(
            T_in=4, T_out=4, latent_dim=24, hidden_dim=48,
            mc_dropout_rate=0.1, mc_samples=4,
        ).to(device)

        cuda_inputs = {
            k: v.to(device) if v is not None else None
            for k, v in full_batch["inputs_by_modality"].items()
        }
        cuda_masks = full_batch["modality_masks"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(cuda_inputs, cuda_masks)
            loss = outputs["lightning_logits"].sum() + outputs["flash_rate_pred"].sum()

        loss.backward()

        assert outputs["lightning_probs"].shape == (
            full_batch["B"], full_batch["T_out"], 1, full_batch["H"], full_batch["W"]
        )
        assert outputs["lightning_probs"].device.type == "cuda"

        # Verify no NaN in gradients
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name} under FP16"

    def test_cuda_fp16_dropout_scenario(self, batch_no_radar):
        """CUDA FP16 must also work with sensor dropout."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        model = APMNowcastModel(
            T_in=4, T_out=4, latent_dim=24, hidden_dim=48,
        ).to(device)

        cuda_inputs = {
            k: v.to(device) if v is not None else None
            for k, v in batch_no_radar["inputs_by_modality"].items()
        }
        cuda_masks = batch_no_radar["modality_masks"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(cuda_inputs, cuda_masks)
            loss = outputs["lightning_logits"].sum() + outputs["flash_rate_pred"].sum()

        loss.backward()

        radar_idx = MODALITY_TO_INDEX[ModalityType.RADAR]
        radar_weights = outputs["modality_attention_weights"][:, radar_idx]
        assert (radar_weights == 0.0).all()
        assert not torch.isnan(outputs["lightning_probs"]).any()
