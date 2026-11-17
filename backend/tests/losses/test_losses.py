"""
Comprehensive unit tests for the APM-Nowcast loss functions (losses/).

Tests cover:
  - Finite scalar outputs and clean backward gradients from every sub-loss
  - APMCompositeLoss with APMNowcastModel outputs AND baseline outputs
  - Physics penalty monotonic increase in stable-air regimes
  - FP16 (torch.amp.autocast) numerical stability on RTX 3050 CUDA
  - Uncertainty calibration with/without aleatoric/epistemic terms
"""
import pytest
import torch
import torch.nn.functional as F

from losses import (
    APMCompositeLoss,
    FocalLoss,
    HeteroscedasticIntensityLoss,
    OccurrenceLoss,
    PhysicsConsistencyLoss,
    SoftDiceLoss,
    UncertaintyCalibrationLoss,
)


# ---------------------------------------------------------------------------
# Shared Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tensor_shapes():
    """Standard tensor dimensions for all tests."""
    return {"B": 2, "T_out": 4, "H": 64, "W": 64}


@pytest.fixture
def apm_outputs(tensor_shapes):
    """Synthetic APMNowcastModel-style outputs with uncertainty."""
    B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
    logits = torch.randn(B, T, 1, H, W, requires_grad=True)
    return {
        "lightning_logits": logits,
        "lightning_probs": torch.sigmoid(logits.detach().clone().requires_grad_(True)),
        "flash_rate_pred": F.softplus(torch.randn(B, T, 1, H, W, requires_grad=True)),
        "aleatoric_uncertainty": F.softplus(torch.randn(B, T, 1, H, W)) + 1e-4,
        "epistemic_uncertainty": F.softplus(torch.randn(B, T, 1, H, W)) * 0.01,
    }


@pytest.fixture
def baseline_outputs(tensor_shapes):
    """Baseline model outputs (no uncertainty keys)."""
    B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
    logits = torch.randn(B, T, 1, H, W, requires_grad=True)
    return {
        "lightning_logits": logits,
        "lightning_probs": torch.sigmoid(logits.detach().clone().requires_grad_(True)),
        "flash_rate_pred": F.softplus(torch.randn(B, T, 1, H, W, requires_grad=True)),
    }


@pytest.fixture
def targets(tensor_shapes):
    """Sparse lightning targets with ~5% positive pixels."""
    B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
    occ = torch.zeros(B, T, 1, H, W)
    # Sparse convective cells in center
    occ[:, :, :, 28:36, 28:36] = (torch.rand(B, T, 1, 8, 8) > 0.5).float()
    flash = torch.zeros(B, T, 1, H, W)
    flash[:, :, :, 28:36, 28:36] = torch.poisson(torch.ones(B, T, 1, 8, 8) * 3.0)
    return {
        "target_occurrence": occ,
        "target_flash_rate": flash,
    }


@pytest.fixture
def physics_fields(tensor_shapes):
    """Synthetic ERA5 physics guidance fields [B, T_in, H, W]."""
    B, H, W = tensor_shapes["B"], tensor_shapes["H"], tensor_shapes["W"]
    T_in = 4
    return {
        "cape": torch.randn(B, T_in, H, W),
        "cin": torch.randn(B, T_in, H, W),
        "relative_humidity_2m": torch.randn(B, T_in, H, W),
        "wind_speed_10m": torch.randn(B, T_in, H, W),
    }


# ---------------------------------------------------------------------------
# Occurrence Loss Tests
# ---------------------------------------------------------------------------


class TestFocalLoss:
    def test_finite_scalar_output(self, tensor_shapes):
        B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
        focal = FocalLoss(gamma=2.0, alpha=0.75)
        logits = torch.randn(B, T, 1, H, W, requires_grad=True)
        targets_occ = (torch.rand(B, T, 1, H, W) > 0.9).float()
        loss = focal(logits, targets_occ)

        assert loss.ndim == 0, "Loss should be scalar"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_backward_gradient_flow(self, tensor_shapes):
        B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
        focal = FocalLoss(gamma=2.0, alpha=0.75)
        logits = torch.randn(B, T, 1, H, W, requires_grad=True)
        targets_occ = (torch.rand(B, T, 1, H, W) > 0.9).float()
        loss = focal(logits, targets_occ)
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_all_zeros_target(self, tensor_shapes):
        """Pure negative (empty sky) batch should produce finite loss."""
        B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
        focal = FocalLoss(gamma=2.0, alpha=0.75)
        logits = torch.randn(B, T, 1, H, W)
        targets_occ = torch.zeros(B, T, 1, H, W)
        loss = focal(logits, targets_occ)
        assert torch.isfinite(loss)

    def test_all_ones_target(self, tensor_shapes):
        """Pure positive batch should produce finite loss."""
        B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
        focal = FocalLoss(gamma=2.0, alpha=0.75)
        logits = torch.randn(B, T, 1, H, W)
        targets_occ = torch.ones(B, T, 1, H, W)
        loss = focal(logits, targets_occ)
        assert torch.isfinite(loss)


class TestSoftDiceLoss:
    def test_perfect_overlap(self, tensor_shapes):
        B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
        dice = SoftDiceLoss()
        mask = torch.zeros(B, T, 1, H, W)
        mask[:, :, :, 28:36, 28:36] = 1.0
        loss = dice(mask, mask)
        assert loss.item() < 0.01, "Perfect overlap should yield near-zero Dice loss"

    def test_no_overlap(self, tensor_shapes):
        B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
        dice = SoftDiceLoss()
        pred = torch.zeros(B, T, 1, H, W)
        pred[:, :, :, 0:8, 0:8] = 1.0
        target = torch.zeros(B, T, 1, H, W)
        target[:, :, :, 56:64, 56:64] = 1.0
        loss = dice(pred, target)
        assert loss.item() > 0.9, "No overlap should yield near-one Dice loss"


class TestOccurrenceLoss:
    def test_combined_loss_finite(self, tensor_shapes):
        B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
        occ_loss = OccurrenceLoss()
        logits = torch.randn(B, T, 1, H, W, requires_grad=True)
        probs = torch.sigmoid(logits)
        targets_occ = (torch.rand(B, T, 1, H, W) > 0.9).float()
        loss = occ_loss(logits, probs, targets_occ)
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


# ---------------------------------------------------------------------------
# Intensity Loss Tests
# ---------------------------------------------------------------------------


class TestIntensityLoss:
    def test_heteroscedastic_loss_finite(self, tensor_shapes):
        B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
        loss_fn = HeteroscedasticIntensityLoss()
        pred = F.softplus(torch.randn(B, T, 1, H, W, requires_grad=True))
        target = torch.poisson(torch.ones(B, T, 1, H, W) * 2.0)
        aleatoric = F.softplus(torch.randn(B, T, 1, H, W)) + 1e-4
        loss = loss_fn(pred, target, aleatoric)
        assert torch.isfinite(loss)
        loss.backward()

    def test_fallback_l1_without_uncertainty(self, tensor_shapes):
        B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
        loss_fn = HeteroscedasticIntensityLoss()
        pred = F.softplus(torch.randn(B, T, 1, H, W, requires_grad=True))
        target = torch.poisson(torch.ones(B, T, 1, H, W) * 2.0)
        loss = loss_fn(pred, target, aleatoric_uncertainty=None)
        assert torch.isfinite(loss)
        loss.backward()

    def test_zero_prediction_zero_target(self, tensor_shapes):
        B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
        loss_fn = HeteroscedasticIntensityLoss()
        pred = torch.zeros(B, T, 1, H, W, requires_grad=True)
        target = torch.zeros(B, T, 1, H, W)
        aleatoric = torch.ones(B, T, 1, H, W) * 0.1
        loss = loss_fn(pred, target, aleatoric)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Physics Consistency Loss Tests
# ---------------------------------------------------------------------------


class TestPhysicsLoss:
    def test_graceful_none_fallback(self, tensor_shapes):
        """physics_fields=None should return exactly 0.0."""
        B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
        phys_loss = PhysicsConsistencyLoss()
        probs = torch.sigmoid(torch.randn(B, T, 1, H, W))
        loss = phys_loss(probs, physics_fields=None)
        assert loss.item() == 0.0

    def test_graceful_empty_dict_fallback(self, tensor_shapes):
        B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
        phys_loss = PhysicsConsistencyLoss()
        probs = torch.sigmoid(torch.randn(B, T, 1, H, W))
        loss = phys_loss(probs, physics_fields={})
        assert loss.item() == 0.0

    def test_finite_with_physics(self, tensor_shapes, physics_fields):
        B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
        phys_loss = PhysicsConsistencyLoss()
        probs = torch.sigmoid(torch.randn(B, T, 1, H, W, requires_grad=True))
        loss = phys_loss(probs, physics_fields)
        assert torch.isfinite(loss)
        loss.backward()

    def test_cape_suppression_monotonic(self, tensor_shapes):
        """
        Physics penalty should INCREASE when high lightning probability
        is predicted in low-CAPE (stable air) regimes.
        """
        B, T, H, W = 1, 4, 32, 32
        phys_loss = PhysicsConsistencyLoss(
            cape_suppression_weight=1.0,
            cin_suppression_weight=0.0,
            moisture_weight=0.0,
            temporal_smoothness_weight=0.0,
        )

        # High lightning probability everywhere
        high_probs = torch.ones(B, T, 1, H, W) * 0.9

        # Scenario A: High CAPE (unstable, convection-favorable)
        # Positive normalized CAPE -> sigmoid(-3*CAPE) ~ 0 -> low suppress
        cape_high = {"cape": torch.ones(B, 4, H, W) * 2.0}
        loss_favorable = phys_loss(high_probs, cape_high)

        # Scenario B: Low CAPE (stable, convection-suppressed)
        # Negative normalized CAPE -> sigmoid(-3*CAPE) ~ 1 -> high suppress
        cape_low = {"cape": torch.ones(B, 4, H, W) * (-2.0)}
        loss_stable = phys_loss(high_probs, cape_low)

        assert loss_stable.item() > loss_favorable.item(), \
            f"Physics penalty should be higher in stable air: stable={loss_stable.item():.4f} vs favorable={loss_favorable.item():.4f}"

    def test_cin_suppression_monotonic(self, tensor_shapes):
        """
        Physics penalty should INCREASE when high lightning probability
        is predicted in high-CIN regimes.
        """
        B, T, H, W = 1, 4, 32, 32
        phys_loss = PhysicsConsistencyLoss(
            cape_suppression_weight=0.0,
            cin_suppression_weight=1.0,
            moisture_weight=0.0,
            temporal_smoothness_weight=0.0,
        )
        high_probs = torch.ones(B, T, 1, H, W) * 0.9

        # Scenario A: Low CIN -> low suppression
        cin_low = {"cin": torch.ones(B, 4, H, W) * (-2.0)}
        loss_low_cin = phys_loss(high_probs, cin_low)

        # Scenario B: High CIN -> high suppression
        cin_high = {"cin": torch.ones(B, 4, H, W) * 2.0}
        loss_high_cin = phys_loss(high_probs, cin_high)

        assert loss_high_cin.item() > loss_low_cin.item(), \
            f"Physics penalty should be higher with high CIN: high={loss_high_cin.item():.4f} vs low={loss_low_cin.item():.4f}"

    def test_temporal_smoothness_penalty(self, tensor_shapes):
        """Flickering predictions should produce higher temporal penalty than smooth ones."""
        B, T, H, W = 1, 4, 32, 32
        phys_loss = PhysicsConsistencyLoss(
            cape_suppression_weight=0.0,
            cin_suppression_weight=0.0,
            moisture_weight=0.0,
            temporal_smoothness_weight=1.0,
        )

        # Pass physics fields to activate physics loss
        dummy_physics = {"cape": torch.zeros(B, 4, H, W)}

        # Smooth: constant probability
        smooth_probs = torch.ones(B, T, 1, H, W) * 0.5
        loss_smooth = phys_loss(smooth_probs, dummy_physics)

        # Flickering: alternating high/low
        flicker_probs = torch.zeros(B, T, 1, H, W)
        flicker_probs[:, 0] = 0.9
        flicker_probs[:, 1] = 0.1
        flicker_probs[:, 2] = 0.9
        flicker_probs[:, 3] = 0.1
        loss_flicker = phys_loss(flicker_probs, dummy_physics)

        assert loss_flicker.item() > loss_smooth.item()


# ---------------------------------------------------------------------------
# Calibration Loss Tests
# ---------------------------------------------------------------------------


class TestCalibrationLoss:
    def test_zero_when_no_uncertainty(self, tensor_shapes):
        B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
        cal_loss = UncertaintyCalibrationLoss()
        probs = torch.sigmoid(torch.randn(B, T, 1, H, W))
        targets_occ = (torch.rand(B, T, 1, H, W) > 0.9).float()
        loss = cal_loss(probs, targets_occ, None, None)
        assert loss.item() == 0.0

    def test_finite_with_both_uncertainties(self, tensor_shapes):
        B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
        cal_loss = UncertaintyCalibrationLoss()
        probs = torch.sigmoid(torch.randn(B, T, 1, H, W, requires_grad=True))
        targets_occ = (torch.rand(B, T, 1, H, W) > 0.9).float()
        aleatoric = F.softplus(torch.randn(B, T, 1, H, W)) + 1e-4
        epistemic = F.softplus(torch.randn(B, T, 1, H, W)) * 0.01
        loss = cal_loss(probs, targets_occ, aleatoric, epistemic)
        assert torch.isfinite(loss)
        loss.backward()

    def test_aleatoric_only(self, tensor_shapes):
        B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
        cal_loss = UncertaintyCalibrationLoss()
        probs = torch.sigmoid(torch.randn(B, T, 1, H, W))
        targets_occ = (torch.rand(B, T, 1, H, W) > 0.9).float()
        aleatoric = F.softplus(torch.randn(B, T, 1, H, W)) + 1e-4
        loss = cal_loss(probs, targets_occ, aleatoric, None)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Composite Loss Tests
# ---------------------------------------------------------------------------


class TestAPMCompositeLoss:
    def test_apm_model_outputs(self, apm_outputs, targets, physics_fields):
        """Full APMNowcastModel output with uncertainty and physics."""
        composite = APMCompositeLoss()
        total_loss, metrics = composite(apm_outputs, targets, physics_fields)

        assert total_loss.ndim == 0, "Total loss should be scalar"
        assert torch.isfinite(total_loss), f"Total loss not finite: {total_loss.item()}"
        total_loss.backward()

        # Metrics dict completeness
        for key in ("loss_total", "loss_occ", "loss_int", "loss_phys", "loss_cal"):
            assert key in metrics, f"Missing metric key: {key}"
            assert isinstance(metrics[key], float), f"{key} should be a float"
            assert not (metrics[key] != metrics[key]), f"{key} is NaN"  # NaN check

    def test_baseline_outputs_no_physics(self, baseline_outputs, targets):
        """Baseline model outputs (no uncertainty, no physics_fields)."""
        composite = APMCompositeLoss()
        total_loss, metrics = composite(baseline_outputs, targets, physics_fields=None)

        assert torch.isfinite(total_loss)
        total_loss.backward()

        # Physics and calibration should be 0.0 for baselines
        assert metrics["loss_phys"] == 0.0
        assert metrics["loss_cal"] == 0.0
        assert metrics["loss_occ"] > 0.0
        assert metrics["loss_int"] >= 0.0

    def test_backward_gradient_flow_all_params(self, targets, physics_fields):
        """Gradients flow through composite loss to a simple differentiable model."""
        # Create differentiable outputs
        logits = torch.randn(2, 4, 1, 64, 64, requires_grad=True)
        flash_raw = torch.randn(2, 4, 1, 64, 64, requires_grad=True)
        outputs = {
            "lightning_logits": logits,
            "lightning_probs": torch.sigmoid(logits),
            "flash_rate_pred": F.softplus(flash_raw),
            "aleatoric_uncertainty": F.softplus(torch.randn(2, 4, 1, 64, 64, requires_grad=True)) + 1e-4,
            "epistemic_uncertainty": F.softplus(torch.randn(2, 4, 1, 64, 64)) * 0.01,
        }

        composite = APMCompositeLoss()
        total_loss, _ = composite(outputs, targets, physics_fields)
        total_loss.backward()

        assert logits.grad is not None, "No gradient for logits"
        assert not torch.isnan(logits.grad).any(), "NaN gradient for logits"
        assert flash_raw.grad is not None, "No gradient for flash_raw"
        assert not torch.isnan(flash_raw.grad).any(), "NaN gradient for flash_raw"

    def test_custom_weights(self, apm_outputs, targets, physics_fields):
        """Custom weight configuration should change the total loss."""
        composite_default = APMCompositeLoss(w_occ=1.0, w_int=0.5, w_phys=0.1, w_cal=0.05)
        composite_phys_heavy = APMCompositeLoss(w_occ=1.0, w_int=0.5, w_phys=5.0, w_cal=0.05)

        loss_default, m_default = composite_default(apm_outputs, targets, physics_fields)
        loss_phys_heavy, m_phys_heavy = composite_phys_heavy(apm_outputs, targets, physics_fields)

        # Same sub-losses but different totals if physics > 0
        if m_default["loss_phys"] > 0:
            assert m_phys_heavy["loss_total"] > m_default["loss_total"]

    def test_all_zero_targets(self, tensor_shapes):
        """Pure empty-sky batch should still produce finite loss."""
        B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
        composite = APMCompositeLoss()
        outputs = {
            "lightning_logits": torch.randn(B, T, 1, H, W, requires_grad=True),
            "lightning_probs": torch.sigmoid(torch.randn(B, T, 1, H, W)),
            "flash_rate_pred": F.softplus(torch.randn(B, T, 1, H, W)),
        }
        targets_zero = {
            "target_occurrence": torch.zeros(B, T, 1, H, W),
            "target_flash_rate": torch.zeros(B, T, 1, H, W),
        }
        total_loss, metrics = composite(outputs, targets_zero, None)
        assert torch.isfinite(total_loss)


# ---------------------------------------------------------------------------
# CUDA + FP16 Tests
# ---------------------------------------------------------------------------


class TestFP16Stability:
    def test_focal_loss_fp16(self, tensor_shapes):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
        focal = FocalLoss(gamma=2.0, alpha=0.75).cuda()
        logits = torch.randn(B, T, 1, H, W, device="cuda", requires_grad=True)
        targets_occ = (torch.rand(B, T, 1, H, W, device="cuda") > 0.9).float()

        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = focal(logits, targets_occ)
        assert torch.isfinite(loss), f"FP16 Focal loss not finite: {loss.item()}"
        loss.backward()
        assert not torch.isnan(logits.grad).any()

    def test_intensity_loss_fp16(self, tensor_shapes):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
        loss_fn = HeteroscedasticIntensityLoss().cuda()
        pred = F.softplus(torch.randn(B, T, 1, H, W, device="cuda", requires_grad=True))
        target = torch.poisson(torch.ones(B, T, 1, H, W, device="cuda") * 2.0)
        aleatoric = F.softplus(torch.randn(B, T, 1, H, W, device="cuda")) + 1e-4

        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = loss_fn(pred, target, aleatoric)
        assert torch.isfinite(loss)
        loss.backward()

    def test_composite_loss_fp16(self, tensor_shapes):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
        device = torch.device("cuda")

        composite = APMCompositeLoss().to(device)
        logits = torch.randn(B, T, 1, H, W, device=device, requires_grad=True)
        outputs = {
            "lightning_logits": logits,
            "lightning_probs": torch.sigmoid(logits),
            "flash_rate_pred": F.softplus(torch.randn(B, T, 1, H, W, device=device, requires_grad=True)),
            "aleatoric_uncertainty": F.softplus(torch.randn(B, T, 1, H, W, device=device)) + 1e-4,
            "epistemic_uncertainty": F.softplus(torch.randn(B, T, 1, H, W, device=device)) * 0.01,
        }
        targets_cuda = {
            "target_occurrence": (torch.rand(B, T, 1, H, W, device=device) > 0.9).float(),
            "target_flash_rate": torch.poisson(torch.ones(B, T, 1, H, W, device=device) * 2.0),
        }
        physics_cuda = {
            "cape": torch.randn(B, 4, H, W, device=device),
            "cin": torch.randn(B, 4, H, W, device=device),
            "relative_humidity_2m": torch.randn(B, 4, H, W, device=device),
        }

        with torch.amp.autocast("cuda", dtype=torch.float16):
            total_loss, metrics = composite(outputs, targets_cuda, physics_cuda)

        assert torch.isfinite(total_loss), f"FP16 composite loss not finite: {total_loss.item()}"
        total_loss.backward()

        for key in ("loss_total", "loss_occ", "loss_int", "loss_phys", "loss_cal"):
            assert key in metrics
            assert not (metrics[key] != metrics[key]), f"{key} is NaN under FP16"

    def test_physics_loss_fp16(self, tensor_shapes):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        B, T, H, W = tensor_shapes["B"], tensor_shapes["T_out"], tensor_shapes["H"], tensor_shapes["W"]
        device = torch.device("cuda")
        phys_loss = PhysicsConsistencyLoss().to(device)
        probs = torch.sigmoid(torch.randn(B, T, 1, H, W, device=device, requires_grad=True))
        physics = {
            "cape": torch.randn(B, 4, H, W, device=device),
            "cin": torch.randn(B, 4, H, W, device=device),
        }
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = phys_loss(probs, physics)
        assert torch.isfinite(loss)
        loss.backward()
