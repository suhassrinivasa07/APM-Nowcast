"""
APM-Nowcast Physics-Guided & Uncertainty-Calibrated Loss Functions.
"""
from losses.calibration_loss import UncertaintyCalibrationLoss
from losses.composite_loss import APMCompositeLoss
from losses.intensity_loss import HeteroscedasticIntensityLoss
from losses.occurrence_loss import FocalLoss, OccurrenceLoss, SoftDiceLoss
from losses.physics_loss import PhysicsConsistencyLoss

__all__ = [
    "FocalLoss",
    "SoftDiceLoss",
    "OccurrenceLoss",
    "HeteroscedasticIntensityLoss",
    "PhysicsConsistencyLoss",
    "UncertaintyCalibrationLoss",
    "APMCompositeLoss",
]
