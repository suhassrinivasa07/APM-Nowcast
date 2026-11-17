/**
 * Sensor Dropout & Modality Reliability Simulator
 * Allows operators and researchers to simulate real-time sensor outages
 * (INSAT-3D, DWR Radar, NRSC Lightning, ERA5 NWP) and observe the
 * Mask-Aware Cross-Modal Attention Weights dynamically re-balance
 * alongside Epistemic Uncertainty inflation.
 */

import React from 'react';
import { ModalityAttentionWeights, ModalityId } from '../types/nowcast';
import {
  Satellite,
  Radio,
  Zap,
  Wind,
  CheckCircle2,
  XCircle,
  AlertOctagon,
  Sparkles,
  ShieldAlert,
} from 'lucide-react';

interface SensorDropoutPanelProps {
  activeSensors: Record<ModalityId, boolean>;
  onToggleSensor: (modality: ModalityId) => void;
  attentionWeights: ModalityAttentionWeights;
  uncertaintyInflation: number;
  onSetPreset: (preset: 'all' | 'radar_blackout' | 'sat_blind' | 'severe') => void;
}

export const SensorDropoutPanel: React.FC<SensorDropoutPanelProps> = ({
  activeSensors,
  onToggleSensor,
  attentionWeights,
  uncertaintyInflation,
  onSetPreset,
}) => {
  const SENSOR_METADATA: Array<{
    id: ModalityId;
    name: string;
    description: string;
    icon: React.ReactNode;
    color: string;
    channels: string;
  }> = [
    {
      id: 'radar',
      name: 'DWR Radar Network',
      description: 'IMD S/C-band Doppler Weather Radars',
      icon: <Radio size={16} color="#38bdf8" />,
      color: '#38bdf8',
      channels: 'Reflectivity Z, Velocity V',
    },
    {
      id: 'satellite',
      name: 'INSAT-3D / 3DR / 3DS',
      description: 'ISRO MOSDAC Geostationary Imager & Sounder',
      icon: <Satellite size={16} color="#a855f7" />,
      color: '#a855f7',
      channels: 'TIR-1 (10.8µ), TIR-2 (12µ), WV (6.8µ)',
    },
    {
      id: 'lightning',
      name: 'NRSC Lightning Sensors',
      description: 'Ground LF/VHF Total Lightning Network',
      icon: <Zap size={16} color="#eab308" />,
      color: '#eab308',
      channels: 'Flash counts, event density history',
    },
    {
      id: 'nwp',
      name: 'ERA5 / NWP Dynamics',
      description: 'ECMWF / NCUM Reanalysis & Forecast Fields',
      icon: <Wind size={16} color="#10b981" />,
      color: '#10b981',
      channels: 'CAPE, CIN, 2m RH, 10m Wind shear',
    },
  ];

  return (
    <div
      style={{
        background: '#0f172a',
        border: '1px solid #334155',
        borderRadius: '12px',
        padding: '16px',
        boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4)',
        display: 'flex',
        flexDirection: 'column',
        gap: '14px',
      }}
    >
      {/* Title & Quick Presets */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <ShieldAlert size={18} color="#38bdf8" />
            <h3 style={{ margin: 0, fontSize: '14px', fontWeight: 700, color: '#f8fafc' }}>
              Sensor Dropout &amp; Reliability Simulator
            </h3>
          </div>
          <span style={{ fontSize: '10px', color: '#94a3b8', background: '#1e293b', padding: '2px 6px', borderRadius: '4px' }}>
            Adaptive Fusion
          </span>
        </div>
        <p style={{ margin: 0, fontSize: '11px', color: '#94a3b8', lineHeight: 1.4 }}>
          Toggle observational feeds to test how the neural attention network dynamically shifts weights and inflates epistemic confidence bounds.
        </p>
      </div>

      {/* Preset Buttons */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
        <button
          onClick={() => onSetPreset('all')}
          style={{
            background: '#1e293b',
            border: '1px solid #334155',
            borderRadius: '6px',
            padding: '6px 8px',
            color: '#34d399',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '4px',
          }}
        >
          <Sparkles size={12} /> All Online (Nominal)
        </button>

        <button
          onClick={() => onSetPreset('radar_blackout')}
          style={{
            background: '#1e293b',
            border: '1px solid #334155',
            borderRadius: '6px',
            padding: '6px 8px',
            color: '#f87171',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '4px',
          }}
        >
          <XCircle size={12} /> Radar Blackout
        </button>

        <button
          onClick={() => onSetPreset('sat_blind')}
          style={{
            background: '#1e293b',
            border: '1px solid #334155',
            borderRadius: '6px',
            padding: '6px 8px',
            color: '#c084fc',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '4px',
          }}
        >
          <Satellite size={12} /> Satellite Blind
        </button>

        <button
          onClick={() => onSetPreset('severe')}
          style={{
            background: '#1e293b',
            border: '1px solid #334155',
            borderRadius: '6px',
            padding: '6px 8px',
            color: '#facc15',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '4px',
          }}
        >
          <AlertOctagon size={12} /> Multi-Feed Failure
        </button>
      </div>

      {/* Sensor Feeds Toggle List */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {SENSOR_METADATA.map((sensor) => {
          const isOnline = activeSensors[sensor.id];
          const weight = attentionWeights[sensor.id] || 0;
          const weightPct = Math.round(weight * 100);

          return (
            <div
              key={sensor.id}
              onClick={() => onToggleSensor(sensor.id)}
              style={{
                background: isOnline ? 'rgba(30, 41, 59, 0.7)' : 'rgba(15, 23, 42, 0.4)',
                border: isOnline ? `1px solid ${sensor.color}55` : '1px solid #334155',
                borderRadius: '8px',
                padding: '8px 10px',
                cursor: 'pointer',
                transition: 'all 0.2s',
                opacity: isOnline ? 1.0 : 0.65,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  {sensor.icon}
                  <div>
                    <div style={{ fontSize: '12px', fontWeight: 600, color: isOnline ? '#f8fafc' : '#94a3b8' }}>
                      {sensor.name}
                    </div>
                    <div style={{ fontSize: '10px', color: '#64748b' }}>
                      {sensor.channels}
                    </div>
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div
                    style={{
                      fontSize: '10px',
                      fontWeight: 700,
                      color: isOnline ? sensor.color : '#ef4444',
                      background: isOnline ? `${sensor.color}22` : 'rgba(239, 68, 68, 0.15)',
                      padding: '2px 6px',
                      borderRadius: '4px',
                    }}
                  >
                    {isOnline ? `${weightPct}% ATTENTION` : 'DROPPED'}
                  </div>
                  {isOnline ? (
                    <CheckCircle2 size={16} color="#34d399" />
                  ) : (
                    <XCircle size={16} color="#f87171" />
                  )}
                </div>
              </div>

              {/* Individual Modality Attention Progress Bar */}
              <div style={{ width: '100%', height: '4px', background: '#090d16', borderRadius: '2px', overflow: 'hidden' }}>
                <div
                  style={{
                    width: `${isOnline ? weightPct : 0}%`,
                    height: '100%',
                    background: sensor.color,
                    transition: 'width 0.4s ease-out',
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Adaptive Cross-Modal Attention Balance Bar */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#cbd5e1', marginBottom: '4px' }}>
          <span style={{ fontWeight: 600 }}>Cross-Modal Attention Distribution:</span>
          <span style={{ color: '#38bdf8' }}>100% Normalized</span>
        </div>

        <div style={{ display: 'flex', width: '100%', height: '12px', borderRadius: '6px', overflow: 'hidden', background: '#1e293b' }}>
          <div
            title={`Radar: ${(attentionWeights.radar * 100).toFixed(1)}%`}
            style={{
              width: `${attentionWeights.radar * 100}%`,
              background: '#38bdf8',
              transition: 'width 0.4s',
            }}
          />
          <div
            title={`Satellite: ${(attentionWeights.satellite * 100).toFixed(1)}%`}
            style={{
              width: `${attentionWeights.satellite * 100}%`,
              background: '#a855f7',
              transition: 'width 0.4s',
            }}
          />
          <div
            title={`Lightning: ${(attentionWeights.lightning * 100).toFixed(1)}%`}
            style={{
              width: `${attentionWeights.lightning * 100}%`,
              background: '#eab308',
              transition: 'width 0.4s',
            }}
          />
          <div
            title={`NWP: ${(attentionWeights.nwp * 100).toFixed(1)}%`}
            style={{
              width: `${attentionWeights.nwp * 100}%`,
              background: '#10b981',
              transition: 'width 0.4s',
            }}
          />
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#94a3b8', marginTop: '4px' }}>
          <span style={{ color: '#38bdf8' }}>■ Radar</span>
          <span style={{ color: '#a855f7' }}>■ Sat</span>
          <span style={{ color: '#eab308' }}>■ Lgt</span>
          <span style={{ color: '#10b981' }}>■ NWP</span>
        </div>
      </div>

      {/* Uncertainty Inflation Meter */}
      <div
        style={{
          background: uncertaintyInflation > 1.2 ? 'rgba(239, 68, 68, 0.1)' : 'rgba(30, 41, 59, 0.5)',
          border: uncertaintyInflation > 1.2 ? '1px solid rgba(239, 68, 68, 0.4)' : '1px solid #334155',
          borderRadius: '8px',
          padding: '8px 12px',
          fontSize: '11px',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
          <span style={{ fontWeight: 600, color: '#f8fafc' }}>Predictive Uncertainty Inflation:</span>
          <strong style={{ color: uncertaintyInflation > 1.5 ? '#f43f5e' : '#34d399' }}>
            {uncertaintyInflation.toFixed(2)}x Baseline
          </strong>
        </div>
        <p style={{ margin: 0, fontSize: '10px', color: '#94a3b8', lineHeight: 1.4 }}>
          {uncertaintyInflation <= 1.05
            ? 'Optimal conditions: All multimodal sensors active. Epistemic uncertainty minimal.'
            : `Degraded observation regime: Missing modalities expand epistemic variance bounds by ${Math.round((uncertaintyInflation - 1) * 100)}%.`}
        </p>
      </div>
    </div>
  );
};
