/**
 * Forecast Lead-Time Scrubber & Horizon Controller
 * Compact floating glassmorphic dock for map bottom-center placement.
 * Steps interactively or auto-loops across +15m, +30m, +45m, +60m.
 */

import React, { useEffect, useState } from 'react';
import { HorizonInfo, HorizonStep, SevereAlert } from '../types/nowcast';
import { HORIZON_DEFINITIONS } from '../api/nowcastApi';
import { Play, Pause, SkipBack, SkipForward, AlertTriangle, Zap, Flame } from 'lucide-react';

interface HorizonControllerProps {
  currentStep: HorizonStep;
  onStepChange: (step: HorizonStep) => void;
  alerts: SevereAlert[];
  peakFlashRate: number;
  isConvectiveActive: boolean;
}

export const HorizonController: React.FC<HorizonControllerProps> = ({
  currentStep,
  onStepChange,
  alerts,
  peakFlashRate,
  isConvectiveActive,
}) => {
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(2000);

  useEffect(() => {
    if (!isPlaying) return;
    const timer = setInterval(() => {
      onStepChange(((currentStep + 1) % 4) as HorizonStep);
    }, playbackSpeed);
    return () => clearInterval(timer);
  }, [isPlaying, currentStep, playbackSpeed, onStepChange]);

  const currentHorizon: HorizonInfo = HORIZON_DEFINITIONS[currentStep];

  return (
    <div
      style={{
        background: 'rgba(9, 13, 22, 0.90)',
        backdropFilter: 'blur(14px)',
        WebkitBackdropFilter: 'blur(14px)',
        border: '1px solid rgba(51, 65, 85, 0.75)',
        borderRadius: 12,
        padding: '10px 14px',
        boxShadow: '0 12px 40px rgba(0,0,0,0.65)',
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
      }}
    >
      {/* Top Row: Playback + Speed + Horizon Steps */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap' }}>
        {/* Playback controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <button
            onClick={() => onStepChange((currentStep === 0 ? 3 : currentStep - 1) as HorizonStep)}
            title="Previous Horizon"
            style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6, padding: '5px 8px', color: '#cbd5e1', cursor: 'pointer' }}
          >
            <SkipBack size={13} />
          </button>

          <button
            onClick={() => setIsPlaying(!isPlaying)}
            title={isPlaying ? 'Pause' : 'Auto-Play Horizons'}
            style={{
              background: isPlaying ? '#ef4444' : '#0284c7',
              border: 'none', borderRadius: 7, padding: '6px 12px',
              color: '#fff', fontWeight: 700, fontSize: '11px', cursor: 'pointer',
              display: 'flex', alignItems: 'center', gap: 5, transition: 'all 0.2s',
            }}
          >
            {isPlaying ? <Pause size={13} /> : <Play size={13} />}
            {isPlaying ? 'Pause' : 'Play'}
          </button>

          <button
            onClick={() => onStepChange(((currentStep + 1) % 4) as HorizonStep)}
            title="Next Horizon"
            style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 6, padding: '5px 8px', color: '#cbd5e1', cursor: 'pointer' }}
          >
            <SkipForward size={13} />
          </button>

          {/* Speed pills */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 3, fontSize: '10px', color: '#64748b' }}>
            <span>Speed:</span>
            {[{ label: '1×', speed: 2000 }, { label: '2×', speed: 1000 }].map((sp) => (
              <button
                key={sp.label}
                onClick={() => setPlaybackSpeed(sp.speed)}
                style={{
                  background: playbackSpeed === sp.speed ? '#334155' : 'transparent',
                  border: '1px solid #475569', borderRadius: 4, padding: '1px 5px',
                  color: playbackSpeed === sp.speed ? '#38bdf8' : '#64748b',
                  fontSize: '10px', cursor: 'pointer',
                }}
              >
                {sp.label}
              </button>
            ))}
          </div>
        </div>

        {/* Horizon step buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          {([0, 1, 2, 3] as HorizonStep[]).map((step) => {
            const h = HORIZON_DEFINITIONS[step];
            const isSelected = currentStep === step;
            return (
              <button
                key={step}
                onClick={() => onStepChange(step)}
                style={{
                  background: isSelected ? 'linear-gradient(135deg,#0284c7,#2563eb)' : '#1e293b',
                  border: isSelected ? '1px solid #38bdf8' : '1px solid #334155',
                  borderRadius: 7, padding: '5px 10px',
                  color: isSelected ? '#fff' : '#64748b',
                  fontWeight: isSelected ? 700 : 500, fontSize: '11px', cursor: 'pointer',
                  boxShadow: isSelected ? '0 0 10px rgba(56,189,248,0.4)' : 'none',
                  transition: 'all 0.15s',
                }}
              >
                {h.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Scrubber slider */}
      <input
        type="range" min="0" max="3" step="1"
        value={currentStep}
        onChange={(e) => onStepChange(parseInt(e.target.value) as HorizonStep)}
        style={{ width: '100%', height: 4, accentColor: '#38bdf8', cursor: 'pointer', borderRadius: 2 }}
      />

      {/* Alert ticker */}
      <div
        style={{
          background: isConvectiveActive ? 'rgba(239,68,68,0.10)' : 'rgba(30,41,59,0.4)',
          border: isConvectiveActive ? '1px solid rgba(239,68,68,0.3)' : '1px solid #1e293b',
          borderRadius: 7, padding: '5px 10px',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '10px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#f8fafc' }}>
          <AlertTriangle size={12} color={isConvectiveActive ? '#ef4444' : '#eab308'} />
          <span style={{ fontWeight: 600 }}>
            {alerts.length > 0
              ? alerts[0].headline
              : `Nominal convective conditions for ${currentHorizon.label}.`}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: '#94a3b8' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
            <Zap size={11} color="#facc15" />
            Peak: <strong style={{ color: '#f43f5e', marginLeft: 2 }}>{peakFlashRate.toFixed(1)} fl/km²</strong>
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
            <Flame size={11} color="#fb923c" />
            <strong style={{ color: '#38bdf8' }}>{isConvectiveActive ? 'ACTIVE' : 'QUIET'}</strong>
          </span>
        </div>
      </div>
    </div>
  );
};
