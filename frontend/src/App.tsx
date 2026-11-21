/**
 * APM-Nowcast — Interactive Research & Operational Dashboard
 * Main Application Shell — Single-Viewport Command Layout
 *
 * Layout:
 *   - 100vh locked (no vertical scroll on desktop)
 *   - Left Sidebar (tabs: Live Forecast | Model & Sensor Lab)
 *   - Right: Leaflet Map fills remaining space with floating HorizonController dock at bottom
 *   - Collapsible PhysicsMetricsPanel drawer accessible from sidebar / cluster selection
 */

import React, { useEffect, useState, useCallback, useRef } from 'react';
import L from 'leaflet';
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
import { ForecastMap } from './components/ForecastMap';
import { HorizonController } from './components/HorizonController';
import { SensorDropoutPanel } from './components/SensorDropoutPanel';
import { PhysicsMetricsPanel } from './components/PhysicsMetricsPanel';
import {
  Zap,
  RefreshCw,
  AlertTriangle,
  CloudLightning,
  Server,
  ToggleLeft,
  ToggleRight,
  Gauge,
  Cpu,
  Globe2,
  ChevronDown,
  ChevronUp,
  FlaskConical,
  Radio,
} from 'lucide-react';

// ─── Live UTC Clock hook ───────────────────────────────────────────────────
function useUtcClock() {
  const [utcTime, setUtcTime] = useState('');
  useEffect(() => {
    const update = () => setUtcTime(new Date().toUTCString().replace('GMT', 'UTC'));
    update();
    const id = setInterval(update, 1000);
    return () => clearInterval(id);
  }, []);
  return utcTime;
}

// ─── Domain Risk Banner level ──────────────────────────────────────────────
function domainRiskLevel(cells: ConvectiveStormCell[]): {
  level: 'QUIET' | 'WATCH' | 'WARNING';
  color: string;
  bg: string;
  border: string;
  desc: string;
} {
  const maxProb = cells.length ? Math.max(...cells.map((c) => c.occurrenceProb)) : 0;
  if (maxProb >= 0.6) {
    return {
      level: 'WARNING',
      color: '#f43f5e',
      bg: 'rgba(244, 63, 94, 0.12)',
      border: 'rgba(244, 63, 94, 0.4)',
      desc: 'High-probability convective cells active — elevated lightning risk.',
    };
  }
  if (maxProb >= 0.3) {
    return {
      level: 'WATCH',
      color: '#facc15',
      bg: 'rgba(234, 179, 8, 0.12)',
      border: 'rgba(234, 179, 8, 0.4)',
      desc: 'Moderate convective activity detected — monitor closely.',
    };
  }
  return {
    level: 'QUIET',
    color: '#34d399',
    bg: 'rgba(52, 211, 153, 0.08)',
    border: 'rgba(52, 211, 153, 0.25)',
    desc: 'No significant convective threats detected in domain.',
  };
}

// ─── Severity badge helper ─────────────────────────────────────────────────
function probColor(prob: number) {
  if (prob >= 0.6) return '#f43f5e';
  if (prob >= 0.3) return '#facc15';
  return '#34d399';
}

export const App: React.FC = () => {
  // ─── Operational state ──────────────────────────────────────────────────
  const [horizonStep, setHorizonStep] = useState<HorizonStep>(0);
  const [overlayMode, setOverlayMode] = useState<OverlayMode>('probability');
  const [selectedCell, setSelectedCell] = useState<ConvectiveStormCell | null>(null);
  const [sidebarTab, setSidebarTab] = useState<'forecast' | 'lab'>('forecast');
  const [physicsOpen, setPhysicsOpen] = useState<boolean>(false);

  // ─── Sensor dropout state ───────────────────────────────────────────────
  const [activeSensors, setActiveSensors] = useState<Record<ModalityId, boolean>>({
    radar: true,
    satellite: true,
    lightning: true,
    nwp: true,
  });

  // ─── Data & connection state ────────────────────────────────────────────
  const [snapshot, setSnapshot] = useState<NowcastSnapshot | null>(null);
  const [benchmarks, setBenchmarks] = useState<BenchmarkModelMetrics[]>(BENCHMARK_VERIFICATION_DATA);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [apiError, setApiError] = useState<string | null>(null);
  const [isBackendOnline, setIsBackendOnline] = useState<boolean>(false);
  const [isSimulationMode, setIsSimulationMode] = useState<boolean>(false);

  const utcTime = useUtcClock();

  // ─── Map flyTo ref ──────────────────────────────────────────────────────
  // ForecastMap exposes an imperative handle via a callback ref. We store it.
  const flyToRef = useRef<((lat: number, lon: number) => void) | null>(null);

  // ─── Backend health poll ────────────────────────────────────────────────
  useEffect(() => {
    const check = async () => {
      const res = await nowcastApi.checkBackendHealth();
      setIsBackendOnline(res.online);
    };
    check();
    const id = setInterval(check, 8000);
    return () => clearInterval(id);
  }, []);

  // Selected cell ref to decouple snapshot fetching from user selection
  const selectedCellRef = useRef<ConvectiveStormCell | null>(selectedCell);
  selectedCellRef.current = selectedCell;

  const detailCardRef = useRef<HTMLDivElement>(null);

  // Attach Leaflet DomEvent click and scroll propagation stoppers to detail card
  useEffect(() => {
    if (detailCardRef.current) {
      L.DomEvent.disableClickPropagation(detailCardRef.current);
      L.DomEvent.disableScrollPropagation(detailCardRef.current);
    }
  }, [selectedCell]);

  // ─── Snapshot loader ────────────────────────────────────────────────────
  const loadSnapshot = useCallback(async () => {
    setIsLoading(true);
    setApiError(null);
    try {
      const data = await nowcastApi.getNowcastSnapshot(horizonStep, activeSensors);
      setSnapshot(data);
      // Keep selected cell reference synced if still present and not closed by user
      // Never auto-select data.cells[0] when selectedCell is null
      if (selectedCellRef.current) {
        const updated = data.cells.find((c) => c.name === selectedCellRef.current?.name);
        if (updated) {
          setSelectedCell(updated);
        } else {
          setSelectedCell(null);
        }
      }
    } catch (err: any) {
      console.warn('Live API unreachable or timed out, auto-triggering Simulation Mode:', err);
      // Auto-trigger simulation mode gracefully without blocking screen
      setIsSimulationMode(true);
      nowcastApi.setSimulationMode(true);
      const fallbackData = nowcastApi.getSyntheticSnapshot(horizonStep, activeSensors);
      setSnapshot(fallbackData);
      if (selectedCellRef.current) {
        const updated = fallbackData.cells.find((c) => c.name === selectedCellRef.current?.name);
        setSelectedCell(updated || null);
      }
    } finally {
      setIsLoading(false);
    }
  }, [horizonStep, activeSensors]);

  useEffect(() => {
    loadSnapshot();
  }, [loadSnapshot]);

  // ─── Load benchmarks once ───────────────────────────────────────────────
  useEffect(() => {
    const load = async () => {
      const b = await nowcastApi.getBenchmarkMetrics();
      setBenchmarks(b);
    };
    load();
  }, []);

  // ─── Sensor handlers ────────────────────────────────────────────────────
  const handleToggleSensor = (modality: ModalityId) => {
    setActiveSensors((prev) => ({ ...prev, [modality]: !prev[modality] }));
  };

  const handleSetPreset = (preset: 'all' | 'radar_blackout' | 'sat_blind' | 'severe') => {
    const map = {
      all: { radar: true, satellite: true, lightning: true, nwp: true },
      radar_blackout: { radar: false, satellite: true, lightning: true, nwp: true },
      sat_blind: { radar: true, satellite: false, lightning: true, nwp: true },
      severe: { radar: false, satellite: false, lightning: true, nwp: true },
    };
    setActiveSensors(map[preset]);
  };

  // ─── Simulation mode ────────────────────────────────────────────────────
  const handleToggleSimulationMode = () => {
    const next = !isSimulationMode;
    setIsSimulationMode(next);
    nowcastApi.setSimulationMode(next);
    loadSnapshot();
  };

  // ─── Cell selection → flyTo ─────────────────────────────────────────────
  const handleSelectCell = (cell: ConvectiveStormCell | null) => {
    if (cell === null) {
      setSelectedCell(null);
      return;
    }
    if (selectedCell && selectedCell.id === cell.id) {
      // Toggle / close when clicking already-selected cluster
      setSelectedCell(null);
      return;
    }
    setSelectedCell(cell);
    if (flyToRef.current) {
      flyToRef.current(cell.lat, cell.lon);
    }
  };

  // Escape key closes active selected cluster detail panel
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setSelectedCell(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const currentHorizon = HORIZON_DEFINITIONS[horizonStep];
  const cells = snapshot?.cells || [];
  const peakFlashRate = cells.length ? Math.max(...cells.map((c) => c.flashDensity), 0) : 0;
  const isConvectiveActive = peakFlashRate > 12;
  const risk = domainRiskLevel(cells);
  const sortedCells = [...cells].sort((a, b) => b.occurrenceProb - a.occurrenceProb);

  // ─── Thermodynamic data for selected cell or domain ─────────────────────
  const thermodynamics = snapshot?.thermodynamics || {
    cape: 2850,
    cin: 25,
    moistureConvergence: 4.5,
    filmGateActivation: 0.82,
    liftedIndex: -6.2,
    shear06km: 21.0,
    convectiveRegime: 'High-CAPE / Weak-CIN Tropical Maritime Convection',
  };

  return (
    <div className="app-shell">
      {/* ── Slim Research Prototype Disclaimer Banner ────────────────────── */}
      <div
        role="alert"
        data-testid="prototype-disclaimer"
        style={{
          background: '#0a0f1e',
          borderBottom: '1px solid #1e293b',
          padding: '4px 16px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '8px',
          fontSize: '10px',
          color: '#64748b',
          flexShrink: 0,
          flexWrap: 'wrap',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <AlertTriangle size={11} color="#facc15" />
          <span>
            <strong style={{ color: '#facc15' }}>APM-Nowcast Research Prototype</strong> — Pan-India convective nowcasting research interface; outputs reflect model inference on loaded atmospheric cubes, not official IMD operational warnings.
          </span>
        </div>
        {/* Hardware telemetry relocated from Navbar to here */}
        <div className="hidden-sm-down" style={{ display: 'flex', alignItems: 'center', gap: '14px', fontSize: '10px', color: '#475569' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Cpu size={11} color="#a855f7" /> RTX 3050 (4GB FP16)
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Globe2 size={11} color="#34d399" /> 6.0°N–38.0°N, 68.0°E–98.0°E
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Server size={11} color={isBackendOnline ? '#34d399' : '#eab308'} />
            {isBackendOnline ? 'API :8000 ✓' : 'API Offline'}
          </span>
        </div>
      </div>

      {/* ── Simulation Mode Notice ───────────────────────────────────────── */}
      {isSimulationMode && (
        <div
          data-testid="simulation-mode-notice"
          style={{
            background: 'rgba(234, 179, 8, 0.1)',
            borderBottom: '1px solid rgba(234, 179, 8, 0.25)',
            padding: '3px 20px',
            fontSize: '11px',
            color: '#facc15',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
            flexShrink: 0,
          }}
        >
          ⚠️ <strong>Simulation Mode (Synthetic Data)</strong> — not live atmospheric forecasts.
          <button
            onClick={() => { setIsSimulationMode(false); nowcastApi.setSimulationMode(false); loadSnapshot(); }}
            style={{ background: '#0284c7', border: 'none', borderRadius: '4px', padding: '1px 8px', color: '#fff', fontSize: '10px', fontWeight: 600, cursor: 'pointer' }}
          >
            Switch to Live
          </button>
        </div>
      )}

      {/* ── Compact Top Navbar ───────────────────────────────────────────── */}
      <header
        style={{
          background: '#090d16',
          borderBottom: '1px solid #1e293b',
          padding: '8px 16px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '12px',
          flexShrink: 0,
          position: 'sticky',
          top: 0,
          zIndex: 2000,
        }}
      >
        {/* Logo + Title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', minWidth: 0 }}>
          <div style={{ width: 34, height: 34, minWidth: 34, borderRadius: 9, background: 'linear-gradient(135deg,#0284c7,#7c3aed)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 14px rgba(56,189,248,0.4)' }}>
            <CloudLightning size={20} color="#fff" />
          </div>
          <div style={{ minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h1 style={{ margin: 0, fontSize: '16px', fontWeight: 800, letterSpacing: '-0.02em', color: '#f8fafc', whiteSpace: 'nowrap' }}>APM-Nowcast</h1>
              <span className="hidden-sm-down" style={{ fontSize: '9px', fontWeight: 700, color: '#facc15', background: 'rgba(234,179,8,0.12)', border: '1px solid rgba(234,179,8,0.3)', padding: '1px 5px', borderRadius: 4, whiteSpace: 'nowrap' }}>
                RESEARCH PROTOTYPE
              </span>
            </div>
            <p className="hidden-sm-down" style={{ margin: 0, fontSize: '10px', color: '#64748b', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              Adaptive Multimodal Thunderstorm & Lightning Nowcasting • India Domain
            </p>
          </div>
        </div>

        {/* Live Status Indicator */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexShrink: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: '#94a3b8' }}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: isBackendOnline ? '#34d399' : '#eab308', boxShadow: isBackendOnline ? '0 0 6px #34d399' : 'none', animation: isBackendOnline ? 'pulse-dot 2s infinite' : 'none' }} />
            <span style={{ fontFamily: 'monospace', fontWeight: 600, color: '#f1f5f9', fontSize: '11px' }}>
              {utcTime ? `Live • ${utcTime}` : 'Syncing...'}
            </span>
          </div>

          {/* Mode toggle */}
          <button
            onClick={handleToggleSimulationMode}
            style={{
              background: isSimulationMode ? '#1e293b' : isBackendOnline ? '#0c3252' : '#1e293b',
              border: isSimulationMode ? '1px solid #facc15' : isBackendOnline ? '1px solid #38bdf8' : '1px solid #334155',
              borderRadius: 6,
              padding: '4px 10px',
              color: '#fff',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              fontSize: '11px',
              fontWeight: 600,
            }}
          >
            {isSimulationMode ? (
              <><ToggleRight size={14} color="#facc15" /><span style={{ color: '#facc15' }}>Simulation</span></>
            ) : (
              <><ToggleLeft size={14} color="#38bdf8" /><span style={{ color: '#38bdf8' }}>Live FastAPI</span></>
            )}
          </button>
        </div>
      </header>

      {/* ── Main Command Layout ─────────────────────────────────────────── */}
      <main className="main-command-layout">
        {/* ── LEFT SIDEBAR (Placed below map on mobile via order: 2) ───────── */}
        <aside className="sidebar-panel-section">
          {/* Segmented tab switcher */}
          <div style={{ padding: '10px 12px 0', flexShrink: 0 }}>
            <div style={{ display: 'flex', background: '#090d16', borderRadius: 8, padding: 3, border: '1px solid #1e293b', gap: 2 }}>
              <button
                onClick={() => setSidebarTab('forecast')}
                style={{
                  flex: 1, background: sidebarTab === 'forecast' ? '#0284c7' : 'transparent',
                  border: 'none', borderRadius: 6, padding: '6px 0', fontSize: '11px',
                  fontWeight: 700, color: sidebarTab === 'forecast' ? '#fff' : '#64748b',
                  cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5, transition: 'all 0.2s',
                }}
              >
                <Zap size={12} /> Live Forecast
              </button>
              <button
                onClick={() => setSidebarTab('lab')}
                style={{
                  flex: 1, background: sidebarTab === 'lab' ? '#7c3aed' : 'transparent',
                  border: 'none', borderRadius: 6, padding: '6px 0', fontSize: '11px',
                  fontWeight: 700, color: sidebarTab === 'lab' ? '#fff' : '#64748b',
                  cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5, transition: 'all 0.2s',
                }}
              >
                <FlaskConical size={12} /> Model & Sensor Lab
              </button>
            </div>
          </div>

          {/* Tab content — scrollable */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '10px 12px 12px' }}>

            {/* ══ TAB 1: LIVE FORECAST ═══════════════════════════════════ */}
            {sidebarTab === 'forecast' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>

                {/* Domain Risk Banner */}
                <div style={{
                  background: risk.bg, border: `1px solid ${risk.border}`, borderRadius: 8,
                  padding: '8px 10px', display: 'flex', alignItems: 'center', gap: 8,
                }}>
                  <div style={{
                    fontWeight: 800, fontSize: '15px', letterSpacing: '0.08em',
                    color: risk.color, fontFamily: 'monospace',
                    padding: '2px 7px', borderRadius: 4,
                    background: `${risk.color}18`, border: `1px solid ${risk.color}44`,
                  }}>
                    {risk.level}
                  </div>
                  <span style={{ fontSize: '11px', color: '#cbd5e1', lineHeight: 1.4 }}>{risk.desc}</span>
                </div>

                {/* Active Convective Clusters List */}
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                    <span style={{ fontWeight: 700, fontSize: '12px', color: '#f8fafc', display: 'flex', alignItems: 'center', gap: 5 }}>
                      <Radio size={13} color="#38bdf8" />
                      Active Clusters ({cells.length})
                      {isLoading && <RefreshCw size={11} color="#38bdf8" className="spin" />}
                    </span>
                    <span style={{ color: '#38bdf8', fontWeight: 600, fontSize: '11px' }}>{currentHorizon.label}</span>
                  </div>

                  {/* API Error State */}
                  {apiError && !isSimulationMode && (
                    <div
                      data-testid="api-error-card"
                      style={{
                        background: '#0f172a', border: '1px solid #ef4444', borderRadius: 10, padding: '14px 12px',
                        textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8,
                      }}
                    >
                      <AlertTriangle size={20} color="#ef4444" />
                      <div style={{ fontSize: '12px', fontWeight: 700, color: '#f8fafc' }}>FastAPI Backend Offline</div>
                      <div style={{ fontSize: '11px', color: '#94a3b8', lineHeight: 1.4 }}>
                        Could not reach <code style={{ background: '#1e293b', padding: '1px 4px', borderRadius: 3, color: '#38bdf8', fontSize: '10px' }}>{API_BASE_URL}</code>
                        <br /><span style={{ fontSize: '10px', color: '#f87171' }}>{apiError}</span>
                      </div>
                      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
                        <button
                          data-testid="retry-button"
                          onClick={loadSnapshot}
                          style={{ background: '#0284c7', border: '1px solid #38bdf8', borderRadius: 6, padding: '5px 12px', color: '#fff', fontWeight: 600, fontSize: '11px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}
                        >
                          <RefreshCw size={12} className={isLoading ? 'spin' : ''} /> Retry
                        </button>
                        <button
                          data-testid="switch-simulation-button"
                          onClick={() => { setIsSimulationMode(true); nowcastApi.setSimulationMode(true); setApiError(null); loadSnapshot(); }}
                          style={{ background: '#1e293b', border: '1px solid #facc15', borderRadius: 6, padding: '5px 12px', color: '#facc15', fontWeight: 600, fontSize: '11px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}
                        >
                          <Zap size={12} /> Use Simulation
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Empty State */}
                  {!apiError && cells.length === 0 && !isLoading && (
                    <div data-testid="empty-state-notice" style={{ padding: '14px 8px', textAlign: 'center', color: '#475569', fontSize: '11px', border: '1px dashed #1e293b', borderRadius: 8 }}>
                      No active convective cells detected at {currentHorizon.label}.
                    </div>
                  )}

                  {/* Cluster Cards — sorted by probability desc */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                    {sortedCells.map((cell) => {
                      const isSelected = selectedCell?.id === cell.id;
                      const pColor = probColor(cell.occurrenceProb);
                      return (
                        <div
                          key={cell.id}
                          onClick={() => {
                            if (selectedCell?.id === cell.id) {
                              handleSelectCell(null);
                            } else {
                              handleSelectCell(cell);
                              setPhysicsOpen(true);
                            }
                          }}
                          style={{
                            background: isSelected ? 'rgba(2,132,199,0.18)' : 'rgba(15,23,42,0.7)',
                            border: isSelected ? `1px solid #38bdf8` : '1px solid #1e293b',
                            borderRadius: 8, padding: '8px 10px', cursor: 'pointer',
                            transition: 'all 0.15s',
                          }}
                        >
                          {/* Row 1: name + prob */}
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 4 }}>
                            <div style={{ fontWeight: 700, fontSize: '12px', color: '#f1f5f9', maxWidth: 180 }}>{cell.name}</div>
                            <div style={{
                              fontWeight: 800, fontSize: '12px', color: pColor,
                              background: `${pColor}18`, border: `1px solid ${pColor}44`,
                              padding: '1px 6px', borderRadius: 4, whiteSpace: 'nowrap',
                            }}>
                              {Math.round(cell.occurrenceProb * 100)}%
                            </div>
                          </div>
                          {/* Row 2: metadata */}
                          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2px 8px', fontSize: '10px', color: '#64748b' }}>
                            <span>{cell.lat.toFixed(2)}°N, {cell.lon.toFixed(2)}°E</span>
                            <span style={{ color: '#38bdf8' }}>{cell.flashDensity.toFixed(1)} fl/km²</span>
                            <span>CAPE: <strong style={{ color: '#f87171' }}>{cell.cape} J/kg</strong></span>
                            <span style={{ textTransform: 'capitalize', color: cell.severity === 'extreme' ? '#f43f5e' : cell.severity === 'severe' ? '#fb923c' : '#94a3b8' }}>
                              {cell.severity}
                            </span>
                          </div>
                          {/* Row 3: growth rate */}
                          <div style={{ marginTop: 4, fontSize: '10px', color: '#475569', display: 'flex', alignItems: 'center', gap: 4 }}>
                            <span style={{ color: cell.growthRate === 'intensifying' ? '#f43f5e' : cell.growthRate === 'steady' ? '#facc15' : '#34d399' }}>
                              {cell.growthRate === 'intensifying' ? '▲' : cell.growthRate === 'decaying' ? '▼' : '—'}
                            </span>
                            {cell.growthRate} · {cell.growthRate === 'intensifying' ? 'Rising threat' : cell.growthRate === 'decaying' ? 'Weakening' : 'Steady state'}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Physics Analytics Quick Peek — only when cell selected */}
                {selectedCell && (
                  <div>
                    <button
                      onClick={() => setPhysicsOpen(!physicsOpen)}
                      style={{
                        width: '100%', background: '#0f172a', border: '1px solid #334155',
                        borderRadius: 8, padding: '7px 10px', cursor: 'pointer',
                        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                        color: '#94a3b8', fontSize: '11px', fontWeight: 600,
                      }}
                    >
                      <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                        <Gauge size={13} color="#38bdf8" /> Physics & Verification Analytics
                      </span>
                      {physicsOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                    </button>

                    {physicsOpen && (
                      <div style={{ marginTop: 6 }}>
                        <PhysicsMetricsPanel
                          thermodynamics={thermodynamics}
                          benchmarks={benchmarks}
                          selectedCell={selectedCell}
                        />
                      </div>
                    )}
                  </div>
                )}

                {/* Physics button when no cell selected */}
                {!selectedCell && (
                  <button
                    onClick={() => setPhysicsOpen(!physicsOpen)}
                    style={{
                      width: '100%', background: '#0f172a', border: '1px solid #334155',
                      borderRadius: 8, padding: '7px 10px', cursor: 'pointer',
                      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                      color: '#94a3b8', fontSize: '11px', fontWeight: 600,
                    }}
                  >
                    <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                      <Gauge size={13} color="#38bdf8" /> Physics & Verification Analytics
                    </span>
                    {physicsOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                  </button>
                )}
                {!selectedCell && physicsOpen && (
                  <PhysicsMetricsPanel thermodynamics={thermodynamics} benchmarks={benchmarks} selectedCell={null} />
                )}
              </div>
            )}

            {/* ══ TAB 2: MODEL & SENSOR LAB ══════════════════════════════ */}
            {sidebarTab === 'lab' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <SensorDropoutPanel
                  activeSensors={activeSensors}
                  onToggleSensor={handleToggleSensor}
                  attentionWeights={snapshot?.attentionWeights || { radar: 0.42, satellite: 0.28, lightning: 0.18, nwp: 0.12 }}
                  uncertaintyInflation={snapshot?.uncertaintyInflationFactor || 1.0}
                  onSetPreset={handleSetPreset}
                />

                {/* Full physics panel always visible in Lab tab */}
                <PhysicsMetricsPanel
                  thermodynamics={thermodynamics}
                  benchmarks={benchmarks}
                  selectedCell={selectedCell}
                />
              </div>
            )}
          </div>
        </aside>

        {/* ── RIGHT: MAP VIEWPORT + FLOATING HORIZON DOCK (Top on mobile via order: 1) ── */}
        <div className="map-viewport-section">
          {/* Full-bleed Map */}
          <ForecastMap
            cells={cells}
            stations={snapshot?.stations || []}
            horizon={currentHorizon}
            overlayMode={overlayMode}
            onOverlayModeChange={setOverlayMode}
            selectedCell={selectedCell}
            onSelectCell={handleSelectCell}
            uncertaintyInflation={snapshot?.uncertaintyInflationFactor || 1.0}
            onFlyToReady={(fn) => { flyToRef.current = fn; }}
          />

          {/* ── Floating Horizon Timeline Dock ────────────────────────── */}
          <div className="timeline-dock-wrapper">
            <HorizonController
              currentStep={horizonStep}
              onStepChange={setHorizonStep}
              alerts={snapshot?.alerts || []}
              peakFlashRate={peakFlashRate}
              isConvectiveActive={isConvectiveActive}
            />
          </div>

          {/* ── Selected Cluster Detail Card (Rendered Outside Leaflet MapContainer) ── */}
          {selectedCell && (
            <div
              ref={detailCardRef}
              className="selected-cell-inspector pointer-events-auto"
              onClick={(e) => e.stopPropagation()}
              onMouseDown={(e) => e.stopPropagation()}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <span style={{ fontWeight: 700, fontSize: '13px', color: '#38bdf8' }}>{selectedCell.name}</span>
                <button
                  type="button"
                  aria-label="Close cluster details"
                  className="pointer-events-auto"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    handleSelectCell(null);
                  }}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                  }}
                  onPointerDown={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                  }}
                  style={{
                    position: 'relative',
                    zIndex: 1101,
                    pointerEvents: 'auto',
                    background: 'transparent',
                    border: 'none',
                    color: '#94a3b8',
                    cursor: 'pointer',
                    fontSize: '16px',
                    lineHeight: 1,
                    padding: '4px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    borderRadius: '4px',
                    transition: 'color 0.15s, background-color 0.15s',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.color = '#f8fafc';
                    e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.color = '#94a3b8';
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }}
                >
                  ✕
                </button>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', margin: '8px 0', borderBottom: '1px solid #334155', paddingBottom: '8px' }}>
                <div>
                  <span style={{ color: '#94a3b8' }}>Coord:</span>
                  <div style={{ fontWeight: 600 }}>{selectedCell.lat.toFixed(2)}°N, {selectedCell.lon.toFixed(2)}°E</div>
                </div>
                <div>
                  <span style={{ color: '#94a3b8' }}>Radius:</span>
                  <div style={{ fontWeight: 600 }}>{selectedCell.radiusKm} km</div>
                </div>
                <div>
                  <span style={{ color: '#94a3b8' }}>Prob ({currentHorizon.label}):</span>
                  <div style={{ fontWeight: 700, color: '#facc15' }}>{(selectedCell.occurrenceProb * 100).toFixed(0)}%</div>
                </div>
                <div>
                  <span style={{ color: '#94a3b8' }}>Flash Rate:</span>
                  <div style={{ fontWeight: 700, color: '#f43f5e' }}>{selectedCell.flashDensity} fl/km²</div>
                </div>
              </div>

              <div style={{ fontSize: '10px', color: '#cbd5e1', lineHeight: '1.5' }}>
                <div><strong>CAPE / CIN:</strong> {selectedCell.cape} J/kg / {selectedCell.cin} J/kg</div>
                <div><strong>Aleatoric (data noise):</strong> ±{(selectedCell.aleatoricUncertainty * 100).toFixed(0)}%</div>
                <div><strong>Epistemic (model uncertainty):</strong> ±{(selectedCell.epistemicUncertainty * 100).toFixed(0)}%</div>
                <div><strong>Convective Regime:</strong> {selectedCell.growthRate.toUpperCase()}</div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* ── Footer ──────────────────────────────────────────────────────── */}
      <footer
        style={{
          borderTop: '1px solid #1e293b',
          padding: '6px 24px',
          fontSize: '10px',
          color: '#475569',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '6px',
          background: '#090d16',
          flexShrink: 0,
        }}
      >
        <span><strong style={{ color: '#64748b' }}>APM-Nowcast</strong> © 2026 — NVIDIA RTX 3050 Laptop GPU (4GB VRAM) · PyTorch FP16 AMP · FastAPI Port 8000</span>
        <div style={{ display: 'flex', gap: '14px' }}>
          <span>WMO/WWRP Standard Metrics</span>
          <span>INSAT-3D/3DR TIR-1/WV + IMD DWR + NRSC Lightning</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
