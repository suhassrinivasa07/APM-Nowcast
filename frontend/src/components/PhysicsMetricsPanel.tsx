/**
 * Physics & Verification Analytics Panel
 * Updated with:
 *   - Human-readable meteorological labels + severity badges
 *   - InfoTooltip (<?) next to every metric with plain-English explanation
 *   - selectedCell context (shows cell-specific CAPE/CIN when a cell is selected)
 *   - All scientific values, IDW interpolation, and regime classifications preserved
 */

import React, { useState } from 'react';
import {
  BenchmarkModelMetrics,
  ConvectiveStormCell,
  ThermodynamicState,
} from '../types/nowcast';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import {
  Flame,
  Gauge,
  TrendingUp,
  BarChart3,
  ShieldCheck,
  Sparkles,
  Info,
} from 'lucide-react';

// ─── Reusable InfoTooltip ────────────────────────────────────────────────────
interface InfoTooltipProps {
  text: string;
}
const InfoTooltip: React.FC<InfoTooltipProps> = ({ text }) => {
  const [show, setShow] = useState(false);
  return (
    <span
      style={{ position: 'relative', display: 'inline-flex', alignItems: 'center', marginLeft: 3, cursor: 'help' }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <Info size={11} color="#475569" />
      {show && (
        <span
          style={{
            position: 'absolute',
            bottom: '100%',
            left: '50%',
            transform: 'translateX(-50%)',
            marginBottom: 6,
            width: 220,
            background: '#0f172a',
            border: '1px solid #334155',
            borderRadius: 6,
            padding: '6px 8px',
            fontSize: '10px',
            lineHeight: 1.5,
            color: '#cbd5e1',
            zIndex: 9999,
            pointerEvents: 'none',
            boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
            whiteSpace: 'normal',
          }}
        >
          {text}
        </span>
      )}
    </span>
  );
};

// ─── Severity badge helpers ──────────────────────────────────────────────────
function capeSeverity(cape: number): { label: string; color: string } {
  if (cape >= 2500) return { label: 'Severe', color: '#f43f5e' };
  if (cape >= 1000) return { label: 'Moderate', color: '#facc15' };
  return { label: 'Low', color: '#34d399' };
}
function cinStatus(cin: number): { label: string; color: string } {
  if (cin > 100) return { label: 'Strong Cap', color: '#f43f5e' };
  if (cin > 50) return { label: 'Moderate Cap', color: '#facc15' };
  return { label: 'Weak Cap', color: '#34d399' };
}
function moistureStatus(mc: number): { label: string; color: string } {
  if (mc >= 6) return { label: 'Strong Feed', color: '#f43f5e' };
  if (mc >= 3) return { label: 'Moderate', color: '#facc15' };
  return { label: 'Weak', color: '#34d399' };
}
function filmStatus(gate: number): { label: string; color: string } {
  if (gate >= 0.75) return { label: 'Highly Active', color: '#c084fc' };
  if (gate >= 0.4) return { label: 'Moderate', color: '#a855f7' };
  return { label: 'Low', color: '#7c3aed' };
}

interface PhysicsMetricsPanelProps {
  thermodynamics: ThermodynamicState;
  benchmarks: BenchmarkModelMetrics[];
  selectedCell: ConvectiveStormCell | null;
}

export const PhysicsMetricsPanel: React.FC<PhysicsMetricsPanelProps> = ({
  thermodynamics,
  benchmarks,
  selectedCell,
}) => {
  const [activeTab, setActiveTab] = useState<'thermo' | 'benchmarks'>('thermo');
  const [selectedMetric, setSelectedMetric] = useState<'csi' | 'fss' | 'brier' | 'flashRmse'>('csi');

  // If a specific cell is selected, show its CAPE/CIN; otherwise show domain values
  const cape = selectedCell?.cape ?? thermodynamics.cape;
  const cin = selectedCell?.cin ?? thermodynamics.cin;
  const mc = thermodynamics.moistureConvergence;
  const film = thermodynamics.filmGateActivation;

  const capeSev = capeSeverity(cape);
  const cinStat = cinStatus(cin);
  const mcStat = moistureStatus(mc);
  const filmStat = filmStatus(film);

  // Recharts data
  const leadTimes = ['+15 min', '+30 min', '+45 min', '+60 min'];
  const chartData = leadTimes.map((lt, idx) => {
    const row: Record<string, any> = { leadTime: lt };
    benchmarks.forEach((b) => { row[b.modelName] = b[selectedMetric][idx]; });
    return row;
  });

  const MODEL_COLORS: Record<string, string> = {
    'APM-Nowcast (Proposed Core)': '#38bdf8',
    'Spatiotemporal U-Net': '#a855f7',
    'ConvLSTM Nowcaster': '#eab308',
    'Advection Persistence': '#ef4444',
  };

  const metricPill = (id: string, label: string) => (
    <button
      key={id}
      onClick={() => setSelectedMetric(id as any)}
      style={{
        background: selectedMetric === id ? '#0284c7' : '#1e293b',
        border: selectedMetric === id ? '1px solid #38bdf8' : '1px solid #334155',
        borderRadius: 4, padding: '3px 7px',
        color: selectedMetric === id ? '#fff' : '#94a3b8',
        fontSize: '10px', fontWeight: 600, cursor: 'pointer',
      }}
    >
      {label}
    </button>
  );

  return (
    <div
      style={{
        background: '#0f172a',
        border: '1px solid #334155',
        borderRadius: 10,
        overflow: 'hidden',
        fontSize: '11px',
      }}
    >
      {/* Header & Tabs */}
      <div
        style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '9px 12px', background: 'rgba(30,41,59,0.6)',
          borderBottom: '1px solid #334155',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Gauge size={15} color="#38bdf8" />
          <span style={{ fontWeight: 700, fontSize: '12px', color: '#f8fafc' }}>
            Physics &amp; Verification Analytics
          </span>
          {selectedCell && (
            <span style={{ fontSize: '10px', color: '#38bdf8', background: 'rgba(56,189,248,0.12)', padding: '1px 6px', borderRadius: 4, border: '1px solid rgba(56,189,248,0.25)' }}>
              {selectedCell.name.split(' ').slice(0, 2).join(' ')}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', background: '#090d16', padding: 2, borderRadius: 6, border: '1px solid #334155' }}>
          <button
            onClick={() => setActiveTab('thermo')}
            style={{ background: activeTab === 'thermo' ? '#0284c7' : 'transparent', color: activeTab === 'thermo' ? '#fff' : '#94a3b8', border: 'none', borderRadius: 4, padding: '3px 8px', fontSize: '10px', fontWeight: 600, cursor: 'pointer' }}
          >
            Thermodynamics
          </button>
          <button
            onClick={() => setActiveTab('benchmarks')}
            style={{ background: activeTab === 'benchmarks' ? '#7c3aed' : 'transparent', color: activeTab === 'benchmarks' ? '#fff' : '#94a3b8', border: 'none', borderRadius: 4, padding: '3px 8px', fontSize: '10px', fontWeight: 600, cursor: 'pointer' }}
          >
            Model Benchmarks
          </button>
        </div>
      </div>

      <div style={{ padding: 12 }}>
        {/* ── TAB 1: THERMODYNAMIC REGIME ───────────────────────────────── */}
        {activeTab === 'thermo' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 6 }}>
              <span style={{ fontSize: '10px', color: '#64748b' }}>
                ERA5 &amp; INSAT-3DS Sounder Assimilation
                {selectedCell ? ` · ${selectedCell.name}` : ' · Domain Average'}
              </span>
              <span style={{ fontSize: '10px', fontWeight: 700, color: '#facc15', background: 'rgba(234,179,8,0.12)', padding: '2px 7px', borderRadius: 4, border: '1px solid rgba(234,179,8,0.25)' }}>
                {thermodynamics.convectiveRegime}
              </span>
            </div>

            {/* 2×2 Metric Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>

              {/* CAPE */}
              <div style={{ background: '#1a2235', border: '1px solid #334155', borderRadius: 8, padding: '10px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                  <span style={{ color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 2 }}>
                    Storm Fuel (CAPE)
                    <InfoTooltip text="CAPE = Convective Available Potential Energy. Higher values mean more energy available to drive explosive thunderstorm updrafts. ≥2500 J/kg is 'Severe'." />
                  </span>
                  <Flame size={13} color="#ef4444" />
                </div>
                <div style={{ fontSize: '18px', fontWeight: 800, color: '#f87171' }}>
                  {cape} <span style={{ fontSize: '10px', fontWeight: 400, color: '#64748b' }}>J/kg</span>
                </div>
                <div style={{ width: '100%', height: 3, background: '#090d16', borderRadius: 2, margin: '6px 0 4px' }}>
                  <div style={{ width: `${Math.min(100, (cape / 4000) * 100)}%`, height: '100%', background: 'linear-gradient(to right,#eab308,#ef4444)', borderRadius: 2 }} />
                </div>
                <span style={{ fontSize: '9px', color: capeSev.color, fontWeight: 700 }}>{capeSev.label}</span>
                <span style={{ fontSize: '9px', color: '#475569' }}> · &gt;2500 J/kg = Severe</span>
              </div>

              {/* CIN */}
              <div style={{ background: '#1a2235', border: '1px solid #334155', borderRadius: 8, padding: '10px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                  <span style={{ color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 2 }}>
                    Atmosphere Lid (CIN)
                    <InfoTooltip text="CIN = Convective Inhibition. Acts as a 'lid' on the atmosphere. Weak CIN (&lt;50 J/kg) means storms can ignite easily; strong CIN suppresses initiation." />
                  </span>
                  <ShieldCheck size={13} color="#38bdf8" />
                </div>
                <div style={{ fontSize: '18px', fontWeight: 800, color: '#38bdf8' }}>
                  {cin} <span style={{ fontSize: '10px', fontWeight: 400, color: '#64748b' }}>J/kg</span>
                </div>
                <div style={{ width: '100%', height: 3, background: '#090d16', borderRadius: 2, margin: '6px 0 4px' }}>
                  <div style={{ width: `${Math.min(100, (cin / 100) * 100)}%`, height: '100%', background: '#38bdf8', borderRadius: 2 }} />
                </div>
                <span style={{ fontSize: '9px', color: cinStat.color, fontWeight: 700 }}>{cinStat.label}</span>
                <span style={{ fontSize: '9px', color: '#475569' }}> · &lt;50 = Initiation likely</span>
              </div>

              {/* Moisture Convergence */}
              <div style={{ background: '#1a2235', border: '1px solid #334155', borderRadius: 8, padding: '10px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                  <span style={{ color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 2 }}>
                    Updraft Moisture
                    <InfoTooltip text="Moisture Convergence measures how much humid air is being funnelled into the storm's base. High values sustain intense rainfall and lightning." />
                  </span>
                  <TrendingUp size={13} color="#34d399" />
                </div>
                <div style={{ fontSize: '18px', fontWeight: 800, color: '#34d399' }}>
                  {mc} <span style={{ fontSize: '10px', fontWeight: 400, color: '#64748b' }}>g/kg/s</span>
                </div>
                <div style={{ width: '100%', height: 3, background: '#090d16', borderRadius: 2, margin: '6px 0 4px' }}>
                  <div style={{ width: `${Math.min(100, (mc / 8) * 100)}%`, height: '100%', background: '#34d399', borderRadius: 2 }} />
                </div>
                <span style={{ fontSize: '9px', color: mcStat.color, fontWeight: 700 }}>{mcStat.label}</span>
                <span style={{ fontSize: '9px', color: '#475569' }}> · updraft fuelling</span>
              </div>

              {/* FiLM Gate */}
              <div style={{ background: '#1a2235', border: '1px solid #334155', borderRadius: 8, padding: '10px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                  <span style={{ color: '#94a3b8', display: 'flex', alignItems: 'center', gap: 2 }}>
                    Physics Filter (FiLM)
                    <InfoTooltip text="FiLM Gate Multiplier: the neural network's physics-guided attention gate. When active (&gt;75%), it actively suppresses false-alarm predictions based on atmospheric thermodynamics." />
                  </span>
                  <Sparkles size={13} color="#c084fc" />
                </div>
                <div style={{ fontSize: '18px', fontWeight: 800, color: '#c084fc' }}>
                  {(film * 100).toFixed(0)}% <span style={{ fontSize: '10px', fontWeight: 400, color: '#64748b' }}>gate</span>
                </div>
                <div style={{ width: '100%', height: 3, background: '#090d16', borderRadius: 2, margin: '6px 0 4px' }}>
                  <div style={{ width: `${film * 100}%`, height: '100%', background: '#c084fc', borderRadius: 2 }} />
                </div>
                <span style={{ fontSize: '9px', color: filmStat.color, fontWeight: 700 }}>{filmStat.label}</span>
                <span style={{ fontSize: '9px', color: '#475569' }}> · physics decoder gate</span>
              </div>
            </div>

            {/* Instability Synopsis */}
            <div style={{ background: 'rgba(30,41,59,0.4)', border: '1px solid #1e293b', borderRadius: 8, padding: '8px 10px', fontSize: '10px', color: '#94a3b8', lineHeight: 1.5 }}>
              <strong style={{ color: '#cbd5e1' }}>Interpretation:</strong> Lifted Index{' '}
              <strong>{thermodynamics.liftedIndex}°C</strong> with deep-layer shear{' '}
              <strong>{thermodynamics.shear06km} m/s</strong> — {thermodynamics.liftedIndex <= -4 ? 'explosive thermodynamic environment capable of sustaining long-lived supercellular thunderstorms' : 'moderate instability; shallow convection possible'}.
              The FiLM conditioning layer {film >= 0.75 ? 'is actively suppressing unphysical false alarms' : 'is operating at moderate gate strength'} when CIN &gt; 150 J/kg.
            </div>
          </div>
        )}

        {/* ── TAB 2: MODEL BENCHMARKS ───────────────────────────────────── */}
        {activeTab === 'benchmarks' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 6 }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 5, fontWeight: 600, color: '#f8fafc' }}>
                <BarChart3 size={13} color="#38bdf8" /> Verification Skill by Lead Time (+15m → +60m)
              </span>
              <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                {metricPill('csi', 'CSI')}
                {metricPill('fss', 'FSS')}
                {metricPill('brier', 'Brier ↓')}
                {metricPill('flashRmse', 'Flash RMSE')}
              </div>
            </div>

            <div style={{ width: '100%', height: 200, background: '#090d16', borderRadius: 8, padding: '8px 4px', border: '1px solid #334155' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 6, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="leadTime" stroke="#94a3b8" fontSize={10} />
                  <YAxis stroke="#94a3b8" fontSize={10} domain={selectedMetric === 'brier' || selectedMetric === 'flashRmse' ? ['auto', 'auto'] : [0, 1]} />
                  <Tooltip contentStyle={{ background: '#0f172a', borderColor: '#334155', borderRadius: 6, fontSize: '10px' }} />
                  <Legend wrapperStyle={{ fontSize: '10px', paddingTop: 4 }} />
                  {benchmarks.map((b) => (
                    <Line
                      key={b.modelName}
                      type="monotone"
                      dataKey={b.modelName}
                      stroke={MODEL_COLORS[b.modelName] || '#38bdf8'}
                      strokeWidth={b.isProposed ? 2.5 : 1.5}
                      strokeDasharray={b.isProposed ? undefined : '4 4'}
                      dot={{ r: b.isProposed ? 3 : 2 }}
                      activeDot={{ r: 5 }}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div style={{ fontSize: '9px', color: '#64748b', lineHeight: 1.4 }}>
              * Verified over Indian monsoon test partition. <strong style={{ color: '#94a3b8' }}>APM-Nowcast</strong> maintains &gt;0.45 CSI at +60 min via cross-modal attention and physics-guided loss constraints.
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
