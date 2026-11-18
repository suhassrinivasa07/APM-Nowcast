/**
 * APM-Nowcast Meteorological Dashboard Type Definitions
 * Spatiotemporal multimodal nowcasting schema for India domain.
 */

export type ModalityId = 'satellite' | 'radar' | 'lightning' | 'nwp';

export type SensorStatus = 'online' | 'degraded' | 'offline';

export type OverlayMode = 'probability' | 'flash_density' | 'uncertainty';

export type HorizonStep = 0 | 1 | 2 | 3; // +15m, +30m, +45m, +60m

export interface HorizonInfo {
  step: HorizonStep;
  minutes: number;
  label: string; // "+15 min", etc.
  leadTimeIso: string;
}

export interface SensorStation {
  id: string;
  name: string;
  type: 'radar' | 'satellite' | 'lightning';
  lat: number;
  lon: number;
  status: SensorStatus;
  rangeKm: number; // e.g. 250km for DWR
  description: string;
  band?: string; // S-band, C-band, etc.
}

export interface ConvectiveStormCell {
  id: string;
  name: string;
  lat: number;
  lon: number;
  radiusKm: number;
  occurrenceProb: number; // 0.0 - 1.0 (0 - 100%)
  flashDensity: number; // flashes / km² / 15min
  aleatoricUncertainty: number; // 0.0 - 1.0
  epistemicUncertainty: number; // 0.0 - 1.0
  totalUncertainty: number; // 0.0 - 1.0
  cape: number; // J/kg
  cin: number; // J/kg
  growthRate: 'intensifying' | 'steady' | 'decaying';
  severity: 'low' | 'moderate' | 'severe' | 'extreme';
  motionSpeedKmh?: number;
  bearingDeg?: number;
  windSpeed10m?: number;
}

export interface ModalityAttentionWeights {
  satellite: number; // 0.0 - 1.0
  radar: number;
  lightning: number;
  nwp: number;
}

export interface ThermodynamicState {
  cape: number; // Convective Available Potential Energy (J/kg)
  cin: number; // Convective Inhibition (J/kg)
  moistureConvergence: number; // g / (kg * s) * 10^-3
  filmGateActivation: number; // 0.0 - 1.0 (physics gate multiplier)
  liftedIndex: number; // °C
  shear06km: number; // m/s (deep layer wind shear)
  convectiveRegime: string;
}

export interface BenchmarkModelMetrics {
  modelName: string;
  isProposed?: boolean;
  csi: number[]; // [+15m, +30m, +45m, +60m]
  pod: number[];
  far: number[];
  fss: number[];
  brier: number[];
  ece: number[];
  flashRmse: number[];
}

export interface SevereAlert {
  id: string;
  region: string;
  severity: 'moderate' | 'severe' | 'extreme';
  headline: string;
  leadTime: string;
  peakFlashRate: number;
  affectedDwr: string;
  timestamp: string;
}

export interface NowcastSnapshot {
  horizon: HorizonInfo;
  timestamp: string;
  cells: ConvectiveStormCell[];
  stations: SensorStation[];
  attentionWeights: ModalityAttentionWeights;
  thermodynamics: ThermodynamicState;
  uncertaintyInflationFactor: number; // 1.0x (all online) to 3.2x (blackout)
  alerts: SevereAlert[];
  dataSource?: string;
  isLiveNwp?: boolean;
}

export interface SubRegionPreset {
  id: string;
  name: string;
  center: [number, number];
  zoom: number;
  description: string;
}

export interface CellTrajectoryPoint {
  step: HorizonStep;
  label: string;
  lat: number;
  lon: number;
}

export interface CellMotionVector {
  cellId: string;
  cellName: string;
  points: CellTrajectoryPoint[];
  speedKmH: number;
  bearingDeg: number;
  directionLabel: string;
}

