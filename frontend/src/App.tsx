/**
 * APM-Nowcast — Interactive Research & Operational Dashboard
 * Main Application Shell
 */

import React, { useEffect, useState, useCallback } from 'react';
import {
  BenchmarkModelMetrics,
  ConvectiveStormCell,
  HorizonStep,
  ModalityId,
  NowcastSnapshot,
  OverlayMode,
} from './types/nowcast';
import {
  nowcastApi,
  API_BASE_URL,
  HORIZON_DEFINITIONS,
  BENCHMARK_VERIFICATION_DATA,
} from './api/nowcastApi';
import { Navbar } from './components/Navbar';
import { ForecastMap } from './components/ForecastMap';
import { HorizonController } from './components/HorizonController';
import { SensorDropoutPanel } from './components/SensorDropoutPanel';
import { PhysicsMetricsPanel } from './components/PhysicsMetricsPanel';
import {
  Zap,
  RefreshCw,
  AlertTriangle,
} from 'lucide-react';

export const App: React.FC = () => {
  // Operational state
  const [horizonStep, setHorizonStep] = useState<HorizonStep>(0);
  const [overlayMode, setOverlayMode] = useState<OverlayMode>('probability');
  const [selectedCell, setSelectedCell] = useState<ConvectiveStormCell | null>(null);

  // Sensor dropout state
  const [activeSensors, setActiveSensors] = useState<Record<ModalityId, boolean>>({
    radar: true,
    satellite: true,
    lightning: true,
    nwp: true,
  });

  // Data snapshots & connection state
  const [snapshot, setSnapshot] = useState<NowcastSnapshot | null>(null);
  const [benchmarks, setBenchmarks] = useState<BenchmarkModelMetrics[]>(BENCHMARK_VERIFICATION_DATA);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [apiError, setApiError] = useState<string | null>(null);
  const [isBackendOnline, setIsBackendOnline] = useState<boolean>(false);
  const [isSimulationMode, setIsSimulationMode] = useState<boolean>(false); // Live FastAPI mode by default

  // Check backend health periodically
  useEffect(() => {
    const checkHealth = async () => {
      const res = await nowcastApi.checkBackendHealth();
      setIsBackendOnline(res.online);
    };
    checkHealth();
    const interval = setInterval(checkHealth, 8000);
    return () => clearInterval(interval);
  }, []);

  // Fetch or regenerate snapshot whenever horizon or sensors change
  const loadSnapshot = useCallback(async () => {
    setIsLoading(true);
    setApiError(null);
    try {
      const data = await nowcastApi.getNowcastSnapshot(horizonStep, activeSensors);
      setSnapshot(data);
      // Keep selected cell reference synced if still present
      if (selectedCell) {
        const updated = data.cells.find((c) => c.name === selectedCell.name);
        if (updated) setSelectedCell(updated);
      }
    } catch (err: any) {
      console.error('Failed to load snapshot:', err);
      setApiError(
        err?.message ||
          `Unable to connect to live APM-Nowcast FastAPI server (${API_BASE_URL}). Ensure backend is running or switch to Simulation Mode.`
      );
    } finally {
      setIsLoading(false);
    }
  }, [horizonStep, activeSensors, selectedCell]);

  useEffect(() => {
    loadSnapshot();
  }, [loadSnapshot]);

  // Load benchmark metrics once
  useEffect(() => {
    const loadBenchmarks = async () => {
      const b = await nowcastApi.getBenchmarkMetrics();
      setBenchmarks(b);
    };
    loadBenchmarks();
  }, []);

  // Sensor toggle handler
  const handleToggleSensor = (modality: ModalityId) => {
    setActiveSensors((prev) => ({
      ...prev,
      [modality]: !prev[modality],
    }));
  };

  // Sensor dropout presets
  const handleSetPreset = (preset: 'all' | 'radar_blackout' | 'sat_blind' | 'severe') => {
    if (preset === 'all') {
      setActiveSensors({ radar: true, satellite: true, lightning: true, nwp: true });
    } else if (preset === 'radar_blackout') {
      setActiveSensors({ radar: false, satellite: true, lightning: true, nwp: true });
    } else if (preset === 'sat_blind') {
      setActiveSensors({ radar: true, satellite: false, lightning: true, nwp: true });
    } else if (preset === 'severe') {
      setActiveSensors({ radar: false, satellite: false, lightning: true, nwp: true });
    }
  };

  // Toggle simulation mode
  const handleToggleSimulationMode = () => {
    const nextMode = !isSimulationMode;
    setIsSimulationMode(nextMode);
    nowcastApi.setSimulationMode(nextMode);
    loadSnapshot();
  };

  const currentHorizon = HORIZON_DEFINITIONS[horizonStep];
  const peakFlashRate = snapshot
    ? Math.max(...snapshot.cells.map((c) => c.flashDensity), 0)
    : 0;
  const isConvectiveActive = peakFlashRate > 12;

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#090d16',
        color: '#f8fafc',
      }}
    >
      {/* Top Disclaimer / Research Prototype Banner */}
      <div
        role="alert"
        data-testid="prototype-disclaimer"
        style={{
          background: 'linear-gradient(90deg, #1e293b, #0f172a)',
          borderBottom: '1px solid #334155',
          padding: '6px 20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '8px',
          fontSize: '11px',
          color: '#cbd5e1',
          textAlign: 'center',
        }}
      >
        <AlertTriangle size={13} color="#facc15" />
        <span>
          <strong style={{ color: '#facc15' }}>APM-Nowcast Research Prototype</strong> — Pan-India convective nowcasting research interface; outputs reflect model inference on loaded atmospheric cubes, not official IMD operational warnings.
        </span>
      </div>

      {/* Simulation Mode Notice Banner */}
      {isSimulationMode && (
        <div
          data-testid="simulation-mode-notice"
          style={{
            background: 'rgba(234, 179, 8, 0.12)',
            borderBottom: '1px solid rgba(234, 179, 8, 0.3)',
            padding: '4px 20px',
            fontSize: '11px',
            color: '#facc15',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
          }}
        >
          <span>
            ⚠️ Operating in <strong>Simulation Mode (Synthetic Data)</strong> — outputs are demonstration scenarios and not live atmospheric forecasts.
          </span>
          <button
            onClick={() => {
              setIsSimulationMode(false);
              nowcastApi.setSimulationMode(false);
              loadSnapshot();
            }}
            style={{
              background: '#0284c7',
              border: 'none',
              borderRadius: '4px',
              padding: '2px 8px',
              color: '#ffffff',
              fontSize: '10px',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            Switch to Live FastAPI Mode
          </button>
        </div>
      )}

      {/* Top Command Center Header */}
      <Navbar
        isSimulationMode={isSimulationMode}
        onToggleSimulationMode={handleToggleSimulationMode}
        isBackendOnline={isBackendOnline}
        isLiveNwp={Boolean(snapshot?.isLiveNwp ?? (snapshot?.dataSource === 'live_nwp'))}
      />

      {/* Main Command Dashboard Layout */}
      <main
        style={{
          flex: 1,
          padding: '16px 20px',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px',
          maxWidth: '1680px',
          margin: '0 auto',
          width: '100%',
        }}
      >
        {/* Error State: Shown when Live API is unreachable */}
        {apiError && !isSimulationMode ? (
          <div
            data-testid="api-error-card"
            style={{
              background: '#0f172a',
              border: '1px solid #ef4444',
              borderRadius: '12px',
              padding: '36px 24px',
              textAlign: 'center',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '12px',
              boxShadow: '0 8px 30px rgba(239, 68, 68, 0.15)',
              margin: '20px 0',
            }}
          >
            <div
              style={{
                width: '48px',
                height: '48px',
                borderRadius: '50%',
                background: 'rgba(239, 68, 68, 0.15)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <AlertTriangle size={24} color="#ef4444" />
            </div>
            <h2 style={{ margin: 0, fontSize: '18px', color: '#f8fafc', fontWeight: 700 }}>
              Live FastAPI Backend Offline or Unreachable
            </h2>
            <p style={{ margin: 0, color: '#94a3b8', fontSize: '13px', maxWidth: '620px', lineHeight: 1.5 }}>
              Could not fetch live nowcast inferences from{' '}
              <code style={{ background: '#1e293b', padding: '2px 6px', borderRadius: '4px', color: '#38bdf8' }}>
                {API_BASE_URL}/nowcast/snapshot
              </code>
              .<br />
              <span style={{ fontSize: '11px', color: '#f87171' }}>{apiError}</span>
            </p>
            <div style={{ display: 'flex', gap: '12px', marginTop: '8px', flexWrap: 'wrap', justifyContent: 'center' }}>
              <button
                data-testid="retry-button"
                onClick={loadSnapshot}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  background: '#0284c7',
                  border: '1px solid #38bdf8',
                  borderRadius: '6px',
                  padding: '8px 16px',
                  color: '#ffffff',
                  fontWeight: 600,
                  fontSize: '12px',
                  cursor: 'pointer',
                }}
              >
                <RefreshCw size={14} className={isLoading ? 'spin' : ''} />
                Retry Live Backend
              </button>
              <button
                data-testid="switch-simulation-button"
                onClick={() => {
                  setIsSimulationMode(true);
                  nowcastApi.setSimulationMode(true);
                  setApiError(null);
                  loadSnapshot();
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  background: '#1e293b',
                  border: '1px solid #facc15',
                  borderRadius: '6px',
                  padding: '8px 16px',
                  color: '#facc15',
                  fontWeight: 600,
                  fontSize: '12px',
                  cursor: 'pointer',
                }}
              >
                <Zap size={14} />
                Switch to Simulation Mode (Synthetic Data)
              </button>
            </div>
          </div>
        ) : (
          /* Top Split View: Left Controls / Right Geospatial Map */
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'minmax(320px, 380px) 1fr',
              gap: '16px',
              alignItems: 'start',
            }}
          >
            {/* Left Column: Sensor Dropout Simulator + Convective Alerts */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <SensorDropoutPanel
                activeSensors={activeSensors}
                onToggleSensor={handleToggleSensor}
                attentionWeights={
                  snapshot?.attentionWeights || {
                    radar: 0.42,
                    satellite: 0.28,
                    lightning: 0.18,
                    nwp: 0.12,
                  }
                }
                uncertaintyInflation={snapshot?.uncertaintyInflationFactor || 1.0}
                onSetPreset={handleSetPreset}
              />

              {/* Convective Hotspot Summary Card */}
              <div
                style={{
                  background: '#0f172a',
                  border: '1px solid #334155',
                  borderRadius: '12px',
                  padding: '14px',
                  fontSize: '11px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontWeight: 700, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Zap size={15} color="#facc15" /> Active Convective Clusters ({snapshot?.cells.length || 0})
                    {isLoading && <RefreshCw size={12} color="#38bdf8" className="spin" />}
                  </span>
                  <span style={{ color: '#38bdf8', fontWeight: 600 }}>{currentHorizon.label}</span>
                </div>

                {/* Empty State */}
                {(!snapshot?.cells || snapshot.cells.length === 0) && !isLoading && (
                  <div data-testid="empty-state-notice" style={{ padding: '16px 8px', textAlign: 'center', color: '#94a3b8', fontSize: '11px' }}>
                    No active convective storm cells detected in domain at lead time {currentHorizon.label}.
                  </div>
                )}

                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '180px', overflowY: 'auto' }}>
                  {snapshot?.cells.map((cell) => {
                    const isSelected = selectedCell?.id === cell.id;
                    return (
                      <div
                        key={cell.id}
                        onClick={() => setSelectedCell(cell)}
                        style={{
                          background: isSelected ? 'rgba(2, 132, 199, 0.25)' : 'rgba(30, 41, 59, 0.6)',
                          border: isSelected ? '1px solid #38bdf8' : '1px solid #334155',
                          borderRadius: '6px',
                          padding: '6px 8px',
                          cursor: 'pointer',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          transition: 'all 0.15s',
                        }}
                      >
                        <div>
                          <div style={{ fontWeight: 600, color: '#f1f5f9' }}>{cell.name}</div>
                          <div style={{ color: '#94a3b8', fontSize: '10px' }}>
                            {cell.lat}°N, {cell.lon}°E • CAPE: {cell.cape} J/kg
                          </div>
                        </div>
                        <div style={{ textAlign: 'right' }}>
                          <div style={{ fontWeight: 700, color: cell.occurrenceProb > 0.7 ? '#f43f5e' : '#facc15' }}>
                            {Math.round(cell.occurrenceProb * 100)}% Prob
                          </div>
                          <div style={{ fontSize: '10px', color: '#38bdf8' }}>
                            {cell.flashDensity} fl/km²
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Right Column: Map & Lead-Time Scrubber */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              {/* Interactive Leaflet Map */}
              <ForecastMap
                cells={snapshot?.cells || []}
                stations={snapshot?.stations || []}
                horizon={currentHorizon}
                overlayMode={overlayMode}
                onOverlayModeChange={setOverlayMode}
                selectedCell={selectedCell}
                onSelectCell={setSelectedCell}
                uncertaintyInflation={snapshot?.uncertaintyInflationFactor || 1.0}
              />

              {/* Lead-Time Scrubber */}
              <HorizonController
                currentStep={horizonStep}
                onStepChange={setHorizonStep}
                alerts={snapshot?.alerts || []}
                peakFlashRate={peakFlashRate}
                isConvectiveActive={isConvectiveActive}
              />
            </div>
          </div>
        )}

        {/* Bottom Section: Physics & Verification Analytics Drawer */}
        <PhysicsMetricsPanel
          thermodynamics={
            snapshot?.thermodynamics || {
              cape: 2850,
              cin: 25,
              moistureConvergence: 4.5,
              filmGateActivation: 0.82,
              liftedIndex: -6.2,
              shear06km: 21.0,
              convectiveRegime: 'High-CAPE / Weak-CIN Tropical Maritime Convection',
            }
          }
          benchmarks={benchmarks}
        />
      </main>

      {/* Footer */}
      <footer
        style={{
          borderTop: '1px solid #1e293b',
          padding: '12px 24px',
          fontSize: '11px',
          color: '#64748b',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '8px',
          background: '#090d16',
        }}
      >
        <div>
          <strong>APM-Nowcast Operational Prototype</strong> &copy; 2026. Optimized for NVIDIA RTX 3050 Laptop GPU (4GB VRAM) via PyTorch FP16 AMP.
        </div>
        <div style={{ display: 'flex', gap: '16px' }}>
          <span>WMO/WWRP Standard Metrics</span>
          <span>INSAT-3D/3DR TIR-1/WV + IMD DWR + NRSC Lightning</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
