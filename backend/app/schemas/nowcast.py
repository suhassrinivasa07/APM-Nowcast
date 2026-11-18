"""
Pydantic v2 request/response schemas for APM-Nowcast Module 10 API.

Mirrors frontend/src/types/nowcast.ts exactly:
  - Field names are identical (camelCase via alias where needed)
  - Every TypeScript interface maps to a Pydantic model
  - JSON responses produced by FastAPI are consumed by NowcastApiService
    without any 422 validation mismatches.
"""
from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared primitive types
# ---------------------------------------------------------------------------

class HorizonInfo(BaseModel):
    """Maps to TypeScript HorizonInfo interface."""
    step: int = Field(..., ge=0, le=3, description="0=+15m, 1=+30m, 2=+45m, 3=+60m")
    minutes: int = Field(..., description="Lead time in minutes")
    label: str = Field(..., description='Human label e.g. "+15 min"')
    leadTimeIso: str = Field(..., description='ISO lead tag e.g. "T+15m"')


class SensorStation(BaseModel):
    """Maps to TypeScript SensorStation interface."""
    id: str
    name: str
    type: Literal["radar", "satellite", "lightning"]
    lat: float
    lon: float
    status: Literal["online", "degraded", "offline"]
    rangeKm: float
    description: str
    band: Optional[str] = None


class ConvectiveStormCell(BaseModel):
    """Maps to TypeScript ConvectiveStormCell interface."""
    id: str
    name: str
    lat: float
    lon: float
    radiusKm: float
    occurrenceProb: float = Field(..., ge=0.0, le=1.0)
    flashDensity: float = Field(..., ge=0.0)
    aleatoricUncertainty: float = Field(..., ge=0.0, le=1.0)
    epistemicUncertainty: float = Field(..., ge=0.0, le=1.0)
    totalUncertainty: float = Field(..., ge=0.0, le=1.0)
    cape: float = Field(..., description="J/kg")
    cin: float = Field(..., description="J/kg")
    growthRate: Literal["intensifying", "steady", "decaying"]
    severity: Literal["low", "moderate", "severe", "extreme"]
    motionSpeedKmh: Optional[float] = None
    bearingDeg: Optional[float] = None
    windSpeed10m: Optional[float] = None


class ModalityAttentionWeights(BaseModel):
    """Maps to TypeScript ModalityAttentionWeights interface."""
    satellite: float = Field(..., ge=0.0, le=1.0)
    radar: float = Field(..., ge=0.0, le=1.0)
    lightning: float = Field(..., ge=0.0, le=1.0)
    nwp: float = Field(..., ge=0.0, le=1.0)


class ThermodynamicState(BaseModel):
    """Maps to TypeScript ThermodynamicState interface."""
    cape: float = Field(..., description="J/kg")
    cin: float = Field(..., description="J/kg")
    moistureConvergence: float
    filmGateActivation: float = Field(..., ge=0.0, le=1.0)
    liftedIndex: float
    shear06km: float
    convectiveRegime: str


class SevereAlert(BaseModel):
    """Maps to TypeScript SevereAlert interface."""
    id: str
    region: str
    severity: Literal["moderate", "severe", "extreme"]
    headline: str
    leadTime: str
    peakFlashRate: float
    affectedDwr: str
    timestamp: str


class NowcastSnapshot(BaseModel):
    """
    Maps to TypeScript NowcastSnapshot interface.
    Returned by GET /api/v1/nowcast/snapshot
    """
    horizon: HorizonInfo
    timestamp: str
    cells: List[ConvectiveStormCell]
    stations: List[SensorStation]
    attentionWeights: ModalityAttentionWeights
    thermodynamics: ThermodynamicState
    uncertaintyInflationFactor: float
    alerts: List[SevereAlert]
    dataSource: Optional[str] = Field("live_nwp", description="Data source: 'live_nwp' or 'benchmark_cube'")
    isLiveNwp: Optional[bool] = Field(True, description="True if real-time Open-Meteo NWP is active")


# ---------------------------------------------------------------------------
# Benchmark verification schemas
# ---------------------------------------------------------------------------

class BenchmarkModelMetrics(BaseModel):
    """
    Maps to TypeScript BenchmarkModelMetrics interface.
    Returned by GET /api/v1/verification/benchmarks and
                GET /api/v1/benchmarks/metrics
    """
    modelName: str
    isProposed: Optional[bool] = None
    csi: List[float] = Field(..., min_length=4, max_length=4)
    pod: List[float] = Field(..., min_length=4, max_length=4)
    far: List[float] = Field(..., min_length=4, max_length=4)
    fss: List[float] = Field(..., min_length=4, max_length=4)
    brier: List[float] = Field(..., min_length=4, max_length=4)
    ece: List[float] = Field(..., min_length=4, max_length=4)
    flashRmse: List[float] = Field(..., min_length=4, max_length=4)


# ---------------------------------------------------------------------------
# Request schemas for POST endpoints
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """
    POST /api/v1/predict — Full multi-horizon nowcast prediction request.
    """
    horizon: int = Field(0, ge=0, le=3, description="Lead-time step index (0–3)")
    satellite: bool = Field(True, description="INSAT-3D satellite sensor availability")
    radar: bool = Field(True, description="DWR radar network availability")
    lightning: bool = Field(True, description="NRSC lightning sensor availability")
    nwp: bool = Field(True, description="ERA5/NWP reanalysis availability")
    estimate_epistemic: bool = Field(False, description="Enable MC-Dropout epistemic uncertainty")
    spatial_size: int = Field(64, ge=32, le=128, description="Spatial patch resolution H=W")
    t_in: int = Field(4, ge=1, le=12, description="Number of input time steps")


class DropoutScenarioRequest(BaseModel):
    """
    POST /api/v1/scenarios/dropout — Sensor dropout simulation request.
    Maps to the scenario simulation in NowcastApiService.evaluate().
    """
    horizon: int = Field(0, ge=0, le=3)
    satellite: bool = True
    radar: bool = True
    lightning: bool = True
    nwp: bool = True


class ExplainModalitiesRequest(BaseModel):
    """
    POST /api/v1/explain/modalities — Modality contribution explanation request.
    """
    horizon: int = Field(0, ge=0, le=3)
    satellite: bool = True
    radar: bool = True
    lightning: bool = True
    nwp: bool = True


# ---------------------------------------------------------------------------
# Response schemas for POST endpoints
# ---------------------------------------------------------------------------

class PhysicsGateStats(BaseModel):
    """FiLM conditioning gate statistics from PhysicsFiLMConditioning."""
    film_scale_mean: float
    film_bias_mean: float
    cape_activation: float


class PredictResponse(BaseModel):
    """
    POST /api/v1/predict response.
    Returns full spatial grids + multi-horizon metadata.
    """
    horizon: HorizonInfo
    timestamp: str
    # Spatial grid statistics (summarized to avoid huge payloads)
    # For a 64×64 spatial domain across T_out=4 horizons
    lightning_probs_mean: List[float]     # [T_out] mean occurrence probability
    lightning_probs_max: List[float]      # [T_out] peak probability (hotspot)
    flash_rate_mean: List[float]          # [T_out] mean flash rate
    flash_rate_max: List[float]           # [T_out] peak flash rate
    aleatoric_uncertainty_mean: List[float]
    epistemic_uncertainty_mean: List[float]
    total_uncertainty_mean: List[float]
    attention_weights: ModalityAttentionWeights
    physics_gate_stats: PhysicsGateStats
    uncertainty_inflation_factor: float
    inference_device: str
    inference_ms: float = Field(..., description="Wall-clock inference time in milliseconds")


class DropoutScenarioResponse(BaseModel):
    """
    POST /api/v1/scenarios/dropout response.
    Sensor dropout returns re-normalized attention weights + snapshot cells.
    """
    scenario: str
    horizon: HorizonInfo
    timestamp: str
    attention_weights: ModalityAttentionWeights
    uncertainty_inflation_factor: float
    cells: List[ConvectiveStormCell]
    alerts: List[SevereAlert]


class ExplainModalitiesResponse(BaseModel):
    """POST /api/v1/explain/modalities response."""
    horizon: HorizonInfo
    timestamp: str
    attention_weights: ModalityAttentionWeights
    physics_gate_stats: PhysicsGateStats
    uncertainty_inflation_factor: float
    modality_descriptions: dict
