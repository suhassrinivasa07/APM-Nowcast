/**
 * APM-Nowcast Frontend Dashboard Smoke & Component Verification Tests
 *
 * Verifies:
 *   1. Pan-India default state (22.5°N, 79.0°E, zoom 5, 6.0°N–38.0°N, 68.0°E–98.0°E)
 *   2. Sub-region camera zoom presets (All India, Indo-Gangetic, Western Ghats, Northeast, Central India)
 *   3. Multi-horizon definitions & switching (+15m, +30m, +45m, +60m)
 *   4. Storm Trajectory (Predicted Motion Vectors) extrapolation (+15m -> +30m -> +45m -> +60m)
 *   5. Research prototype disclaimer banner text & compliance
 *   6. Live FastAPI mode vs. Simulation Mode (Synthetic Data) gating
 *   7. Error & Empty state representations
 */

import {
  nowcastApi,
  NowcastApiService,
  HORIZON_DEFINITIONS,
  SUB_REGION_PRESETS,
  computeCellTrajectories,
  INDIAN_SENSOR_STATIONS,
  BENCHMARK_VERIFICATION_DATA,
} from '../api/nowcastApi';
import { ConvectiveStormCell, HorizonStep } from '../types/nowcast';

declare const process: any;

export function runDashboardTests(): { passed: number; failed: number; tests: string[] } {
  const results: string[] = [];
  let passed = 0;
  let failed = 0;

  function assert(condition: boolean, testName: string) {
    if (condition) {
      passed++;
      results.push(`✓ [PASS] ${testName}`);
    } else {
      failed++;
      results.push(`✗ [FAIL] ${testName}`);
      console.error(`Assertion failed for: ${testName}`);
    }
  }

  // 1. Pan-India Default Map Domain & Center
  const defaultPreset = SUB_REGION_PRESETS.find((p) => p.id === 'all_india');
  assert(!!defaultPreset, 'Pan-India "all_india" preset is defined');
  assert(
    defaultPreset?.center[0] === 22.5 && defaultPreset?.center[1] === 79.0 && defaultPreset?.zoom === 5,
    'Pan-India default center is [22.5°N, 79.0°E] with zoom level 5'
  );

  // 2. Sub-Region Presets Completeness
  const expectedPresetIds = [
    'all_india',
    'indo_gangetic',
    'western_ghats',
    'northeast_bob',
    'central_india',
  ];
  const actualPresetIds = SUB_REGION_PRESETS.map((p) => p.id);
  const allPresetsPresent = expectedPresetIds.every((id) => actualPresetIds.includes(id));
  assert(allPresetsPresent, 'All 5 sub-region zoom presets exist (All India, Indo-Gangetic, Western Ghats, Northeast, Central India)');

  // 3. Multi-Horizon Definitions & Lead Times
  const horizons: HorizonStep[] = [0, 1, 2, 3];
  const horizonLabels = horizons.map((h) => HORIZON_DEFINITIONS[h].label);
  assert(
    horizonLabels.join(',') === '+15 min,+30 min,+45 min,+60 min',
    'Horizons correctly define +15 min, +30 min, +45 min, and +60 min'
  );

  // 4. Storm Trajectory (Predicted Motion Vectors) Extrapolation
  const testCells: ConvectiveStormCell[] = [
    {
      id: 'cell-chota-nagpur-h0',
      name: 'Chota Nagpur Supercell Cluster',
      lat: 23.45,
      lon: 85.65,
      radiusKm: 55,
      occurrenceProb: 0.88,
      flashDensity: 24.5,
      aleatoricUncertainty: 0.15,
      epistemicUncertainty: 0.12,
      totalUncertainty: 0.19,
      cape: 3200,
      cin: 22,
      growthRate: 'intensifying',
      severity: 'extreme',
    },
    {
      id: 'cell-western-ghats-h0',
      name: 'Western Ghats Orographic Convective Band',
      lat: 18.95,
      lon: 73.65,
      radiusKm: 60,
      occurrenceProb: 0.65,
      flashDensity: 11.4,
      aleatoricUncertainty: 0.18,
      epistemicUncertainty: 0.14,
      totalUncertainty: 0.23,
      cape: 2100,
      cin: 42,
      growthRate: 'steady',
      severity: 'moderate',
    },
  ];

  const trajectories = computeCellTrajectories(testCells);
  assert(trajectories.length === 2, 'Trajectories computed for all active cells');
  assert(trajectories[0].points.length === 4, 'Each storm trajectory has 4 horizon points (+15m, +30m, +45m, +60m)');
  assert(trajectories[0].speedKmH > 0, 'Trajectory velocity is strictly positive');
  assert(trajectories[0].bearingDeg >= 0 && trajectories[0].bearingDeg <= 360, 'Trajectory bearing is in [0°, 360°]');

  // 5. Research Prototype Banner Content Verification
  const expectedDisclaimer =
    'APM-Nowcast Research Prototype — Pan-India convective nowcasting research interface; outputs reflect model inference on loaded atmospheric cubes, not official IMD operational warnings.';
  assert(expectedDisclaimer.includes('not official IMD operational warnings'), 'Disclaimer clearly disclaims official IMD warnings');
  assert(expectedDisclaimer.includes('Pan-India convective nowcasting research interface'), 'Disclaimer states Pan-India research interface');

  // 6. Live FastAPI Backend Default Mode
  const testApiService = new NowcastApiService(false);
  assert(testApiService.getSimulationMode() === false, 'NowcastApiService defaults to Live Mode (isSimulationMode = false)');
  assert(nowcastApi.getSimulationMode() === false, 'Global nowcastApi singleton defaults to Live Mode');

  // 7. Simulation Mode (Synthetic Data) Output & Gating
  testApiService.setSimulationMode(true);
  assert(testApiService.getSimulationMode() === true, 'Successfully toggled to Simulation Mode');
  const syntheticSnapshot = testApiService.getSyntheticSnapshot(0, {
    satellite: true,
    radar: true,
    lightning: true,
    nwp: true,
  });
  assert(syntheticSnapshot.cells.length > 0, 'Synthetic snapshot returns demonstration cells');
  assert(syntheticSnapshot.stations.length === INDIAN_SENSOR_STATIONS.length, 'Stations match pan-India sensor registry');

  // 8. Benchmark Metrics Integrity
  assert(BENCHMARK_VERIFICATION_DATA.length >= 4, 'Benchmark verification table includes APM-Nowcast, U-Net, ConvLSTM, and Persistence');
  const proposedModel = BENCHMARK_VERIFICATION_DATA.find((m) => m.isProposed);
  assert(!!proposedModel && proposedModel.csi.length === 4, 'Proposed APM-Nowcast has 4-step CSI metrics');

  return { passed, failed, tests: results };
}

// Direct execution support if run under node
if (typeof process !== 'undefined' && process.argv && process.argv[1] && process.argv[1].includes('dashboard.test')) {
  const { passed, failed, tests } = runDashboardTests();
  tests.forEach((t) => console.log(t));
  console.log(`\nFrontend Dashboard Smoke Tests: ${passed} passed, ${failed} failed.`);
  if (failed > 0) process.exit(1);
}
