/**
 * Meteorological Command Center Header / Navbar
 * Displays operational status, UTC clock, simulation mode toggle,
 * and system health indicators.
 */

import React, { useEffect, useState } from 'react';
import {
  CloudLightning,
  Clock,
  Cpu,
  Globe2,
  ToggleLeft,
  ToggleRight,
  Server,
} from 'lucide-react';

interface NavbarProps {
  isSimulationMode: boolean;
  onToggleSimulationMode: () => void;
  isBackendOnline: boolean;
  isLiveNwp?: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  isSimulationMode,
  onToggleSimulationMode,
  isBackendOnline,
  isLiveNwp,
}) => {
  const [utcTime, setUtcTime] = useState<string>('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setUtcTime(now.toUTCString().replace('GMT', 'UTC'));
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header
      style={{
        background: '#090d16',
        borderBottom: '1px solid #1e293b',
        padding: '10px 20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '12px',
        position: 'sticky',
        top: 0,
        zIndex: 2000,
      }}
    >
      {/* Brand & Subtitle */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div
          style={{
            width: '38px',
            height: '38px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #0284c7, #7c3aed)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 16px rgba(56, 189, 248, 0.4)',
          }}
        >
          <CloudLightning size={22} color="#ffffff" />
        </div>

        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <h1 style={{ margin: 0, fontSize: '18px', fontWeight: 800, letterSpacing: '-0.02em', color: '#f8fafc' }}>
              APM-Nowcast
            </h1>
            <span
              style={{
                fontSize: '10px',
                fontWeight: 700,
                color: '#38bdf8',
                background: 'rgba(56, 189, 248, 0.15)',
                border: '1px solid rgba(56, 189, 248, 0.3)',
                padding: '1px 6px',
                borderRadius: '4px',
              }}
            >
              v1.0-RC
            </span>
          </div>
          <p style={{ margin: 0, fontSize: '11px', color: '#94a3b8' }}>
            Adaptive Multimodal Thunderstorm &amp; Lightning Nowcasting Center • India Domain
          </p>
        </div>
      </div>

      {/* Center: Live UTC Clock & Operational Targets */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '18px', fontSize: '11px', color: '#cbd5e1' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Clock size={14} color="#38bdf8" />
          <span style={{ fontFamily: 'monospace', fontWeight: 600, color: '#f1f5f9' }}>{utcTime || 'Syncing UTC...'}</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Globe2 size={14} color="#34d399" />
          <span>Domain: <strong>Pan-India (6.0°N–38.0°N, 68.0°E–98.0°E)</strong></span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Cpu size={14} color="#a855f7" />
          <span>GPU: <strong>RTX 3050 (4GB AMP)</strong></span>
        </div>
      </div>

      {/* Right: Simulation Mode Toggle & Backend Health Badge */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        {/* Backend Online status */}
        <div
          title={isBackendOnline ? 'FastAPI Backend Online' : 'FastAPI Offline - Using Standalone/Simulation'}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '11px',
            background: '#1e293b',
            border: '1px solid #334155',
            padding: '4px 8px',
            borderRadius: '6px',
          }}
        >
          <Server size={13} color={isBackendOnline ? '#34d399' : '#eab308'} />
          <span style={{ color: isBackendOnline ? '#34d399' : '#cbd5e1' }}>
            {isBackendOnline ? 'API Connected (Port 8000)' : 'API Offline'}
          </span>
        </div>

        {/* Live vs Simulation Mode Toggle Button */}
        <button
          onClick={onToggleSimulationMode}
          title="Toggle between Live FastAPI Backend and Offline Simulation Mode"
          style={{
            background: isSimulationMode ? '#1e293b' : isLiveNwp ? '#0284c7' : '#1e293b',
            border: isSimulationMode
              ? '1px solid #facc15'
              : isLiveNwp
              ? '1px solid #38bdf8'
              : '1px solid #64748b',
            borderRadius: '6px',
            padding: '5px 12px',
            color: '#ffffff',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '11px',
            fontWeight: 600,
            transition: 'all 0.2s',
          }}
        >
          {isSimulationMode ? (
            <>
              <ToggleRight size={16} color="#facc15" />
              <span>Mode: <strong style={{ color: '#facc15' }}>Simulation Mode (Synthetic Data)</strong></span>
            </>
          ) : isLiveNwp ? (
            <>
              <ToggleLeft size={16} color="#38bdf8" />
              <span>Mode: <strong style={{ color: '#38bdf8' }}>Live FastAPI + Real-Time NWP</strong></span>
            </>
          ) : (
            <>
              <ToggleLeft size={16} color="#cbd5e1" />
              <span>Mode: <strong style={{ color: '#cbd5e1' }}>Benchmark Cube Fallback</strong></span>
            </>
          )}
        </button>
      </div>
    </header>
  );
};
