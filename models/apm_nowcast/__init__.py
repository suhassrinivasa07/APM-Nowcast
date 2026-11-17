"""
APM-Nowcast Core Architecture Package.

The proposed Uncertainty-Aware Physics-Guided Adaptive Multimodal Deep Learning
framework for thunderstorm initiation and lightning nowcasting over India.
"""
from .adaptive_fusion import AdaptiveModalityFusion
from .encoders import (
    LightningEncoder,
    ModalityEncoder,
    NWPEncoder,
    RadarEncoder,
    SatelliteEncoder,
)
from .network import APMNowcastModel
from .physics_conditioning import PhysicsFiLMConditioning
from .spatiotemporal_core import ConvGRUCell, SpatiotemporalEvolutionCore
from .uncertainty_heads import UncertaintyForecastHeads

__all__ = [
    "APMNowcastModel",
    "ModalityEncoder",
    "SatelliteEncoder",
    "RadarEncoder",
    "LightningEncoder",
    "NWPEncoder",
    "AdaptiveModalityFusion",
    "PhysicsFiLMConditioning",
    "ConvGRUCell",
    "SpatiotemporalEvolutionCore",
    "UncertaintyForecastHeads",
]
