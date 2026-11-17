/**
 * Forecast Lead-Time Scrubber & Horizon Controller
 * Steps interactively or automatically loops across +15m, +30m, +45m, and +60m
 * lead-time horizons with live peak risk alert indicators.
 */

import React, { useEffect, useState } from 'react';
import {
  HorizonInfo,
  HorizonStep,
  SevereAlert,
} from '../types/nowcast';
import { HORIZON_DEFINITIONS } from '../api/nowcastApi';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Clock,
  AlertTriangle,
  Flame,
  Zap,
} from 'lucide-react';

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
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(2000); // 2000ms per step

  // Auto-play loop
  useEffect(() => {
    if (!isPlaying) return;

    const timer = setInterval(() => {
      onStepChange(((currentStep + 1) % 4) as HorizonStep);
    }, playbackSpeed);

    return () => clearInterval(timer);
  }, [isPlaying, currentStep, playbackSpeed, onStepChange]);

  const handlePrev = () => {
    const prev = (currentStep === 0 ? 3 : currentStep - 1) as HorizonStep;
    onStepChange(prev);
  };

  const handleNext = () => {
    const next = ((currentStep + 1) % 4) as HorizonStep;
    onStepChange(next);
  };

  const currentHorizon: HorizonInfo = HORIZON_DEFINITIONS[currentStep];

  return (
    <div
      style={{
        background: '#0f172a',
        border: '1px solid #334155',
        borderRadius: '12px',
        padding: '14px 18px',
        boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4)',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
      }}
    >
      {/* Top Bar: Playback Controls & Timeline Scrubber */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
        {/* Playback Buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            title={isPlaying ? 'Pause Horizon Animation' : 'Auto-Play Forecast Horizons'}
            style={{
              background: isPlaying ? '#ef4444' : '#0284c7',
              border: 'none',
              borderRadius: '8px',
              padding: '8px 14px',
              color: '#ffffff',
              fontWeight: 600,
              fontSize: '12px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              transition: 'all 0.2s',
            }}
          >
            {isPlaying ? <Pause size={15} /> : <Play size={15} />}
            {isPlaying ? 'Pause' : 'Play Timeline'}
          </button>

          <button
            onClick={handlePrev}
            title="Previous Horizon (-15m)"
            style={{
              background: '#1e293b',
              border: '1px solid #334155',
              borderRadius: '6px',
              padding: '6px 10px',
              color: '#cbd5e1',
              cursor: 'pointer',
            }}
          >
            <SkipBack size={15} />
          </button>

          <button
            onClick={handleNext}
            title="Next Horizon (+15m)"
            style={{
              background: '#1e293b',
              border: '1px solid #334155',
              borderRadius: '6px',
              padding: '6px 10px',
              color: '#cbd5e1',
              cursor: 'pointer',
            }}
          >
            <SkipForward size={15} />
          </button>

          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginLeft: '6px', fontSize: '11px', color: '#94a3b8' }}>
            <span>Speed:</span>
            {[
              { label: '1x', speed: 2000 },
              { label: '2x', speed: 1000 },
            ].map((sp) => (
              <button
                key={sp.label}
                onClick={() => setPlaybackSpeed(sp.speed)}
                style={{
                  background: playbackSpeed === sp.speed ? '#334155' : 'transparent',
                  border: '1px solid #475569',
                  borderRadius: '4px',
                  padding: '2px 6px',
                  color: playbackSpeed === sp.speed ? '#38bdf8' : '#94a3b8',
                  fontSize: '10px',
                  cursor: 'pointer',
                }}
              >
                {sp.label}
              </button>
            ))}
          </div>
        </div>

        {/* Horizon Timeline Step Buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {([0, 1, 2, 3] as HorizonStep[]).map((step) => {
            const h = HORIZON_DEFINITIONS[step];
            const isSelected = currentStep === step;
            return (
              <button
                key={step}
                onClick={() => onStepChange(step)}
                style={{
                  background: isSelected ? 'linear-gradient(135deg, #0284c7, #2563eb)' : '#1e293b',
                  border: isSelected ? '1px solid #38bdf8' : '1px solid #334155',
                  borderRadius: '8px',
                  padding: '8px 14px',
                  color: isSelected ? '#ffffff' : '#94a3b8',
                  fontWeight: isSelected ? 700 : 500,
                  fontSize: '12px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  boxShadow: isSelected ? '0 0 12px rgba(56, 189, 248, 0.4)' : 'none',
                  transition: 'all 0.2s',
                }}
              >
                <Clock size={13} color={isSelected ? '#ffffff' : '#64748b'} />
                {h.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Progress Slider Track */}
      <div style={{ position: 'relative', width: '100%', padding: '4px 0' }}>
        <input
          type="range"
          min="0"
          max="3"
          step="1"
          value={currentStep}
          onChange={(e) => onStepChange(parseInt(e.target.value) as HorizonStep)}
          style={{
            width: '100%',
            height: '6px',
            accentColor: '#38bdf8',
            cursor: 'pointer',
            background: '#1e293b',
            borderRadius: '4px',
          }}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#64748b', marginTop: '2px' }}>
          <span>+15 min (T+15m)</span>
          <span>+30 min (T+30m)</span>
          <span>+45 min (T+45m)</span>
          <span>+60 min (T+60m)</span>
        </div>
      </div>

      {/* Peak Risk Alert Ticker for Current Horizon */}
      <div
        style={{
          background: isConvectiveActive ? 'rgba(239, 68, 68, 0.12)' : 'rgba(30, 41, 59, 0.5)',
          border: isConvectiveActive ? '1px solid rgba(239, 68, 68, 0.35)' : '1px solid #334155',
          borderRadius: '8px',
          padding: '8px 12px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: '11px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#f8fafc' }}>
          <AlertTriangle size={15} color={isConvectiveActive ? '#ef4444' : '#eab308'} />
          <span style={{ fontWeight: 600 }}>
            {alerts.length > 0
              ? alerts[0].headline
              : `Nominal convective conditions across monitored domain for ${currentHorizon.label}.`}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', color: '#cbd5e1' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Zap size={13} color="#facc15" />
            <span>Peak Rate:</span>
            <strong style={{ color: '#f43f5e' }}>{peakFlashRate.toFixed(1)} fl/km²</strong>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Flame size={13} color="#fb923c" />
            <span>Status:</span>
            <strong style={{ color: '#38bdf8' }}>{isConvectiveActive ? 'DEEP CONVECTION ACTIVE' : 'QUIET'}</strong>
          </div>
        </div>
      </div>
    </div>
  );
};
