"""
APM-Nowcast Baseline Models Package.

Includes:
- EulerianPersistenceModel: Zero-parameter stationary persistence benchmark
- AdvectionPersistenceModel: Lagrangian optical-flow advection benchmark
- ConvLSTMNowcastModel: 2-layer ConvLSTM Encoder-Forecaster architecture
- SpatiotemporalUNetBaseline: Multiscale residual U-Net with multi-horizon heads
"""
from models.baselines.convlstm import ConvLSTMCell, ConvLSTMNowcastModel
from models.baselines.persistence import AdvectionPersistenceModel, EulerianPersistenceModel
from models.baselines.unet_nowcast import DoubleConvBlock, SpatiotemporalUNetBaseline

__all__ = [
    "EulerianPersistenceModel",
    "AdvectionPersistenceModel",
    "ConvLSTMCell",
    "ConvLSTMNowcastModel",
    "DoubleConvBlock",
    "SpatiotemporalUNetBaseline",
]
