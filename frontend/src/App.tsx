import React from 'react';
import { HealthStatus } from './components/HealthStatus';
import { CloudLightning, Compass, ShieldCheck } from 'lucide-react';

export const App: React.FC = () => {
  return (
    <div style={{
      minHeight: '100vh',
      backgroundColor: '#0f172a',
      color: '#f8fafc',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
      padding: '2rem 1rem',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center'
    }}>
      <header style={{ textAlign: 'center', maxWidth: '750px', marginBottom: '2rem' }}>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.6rem', color: '#38bdf8', marginBottom: '0.5rem' }}>
          <CloudLightning size={36} />
          <h1 style={{ fontSize: '2.2rem', fontWeight: 700, margin: 0, letterSpacing: '-0.025em' }}>
            APM-Nowcast
          </h1>
        </div>
        <p style={{
          fontSize: '1rem',
          color: '#94a3b8',
          lineHeight: 1.5,
          marginTop: '0.5rem',
          fontWeight: 400
        }}>
          An Uncertainty-Aware Physics-Guided Adaptive Multimodal Deep Learning Framework for Thunderstorm Initiation &amp; Lightning Nowcasting over India
        </p>
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '1.5rem',
          marginTop: '1rem',
          fontSize: '0.85rem',
          color: '#cbd5e1'
        }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
            <ShieldCheck size={16} color="#34d399" /> Day 1: Project Foundation
          </span>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
            <Compass size={16} color="#60a5fa" /> Target: INSAT-3D/3DR + DWR Radar
          </span>
        </div>
      </header>

      <main style={{ width: '100%', maxWidth: '750px' }}>
        <HealthStatus />

        <div style={{
          marginTop: '2rem',
          padding: '1.25rem',
          borderRadius: '8px',
          backgroundColor: '#1e293b',
          border: '1px solid #334155',
          fontSize: '0.875rem'
        }}>
          <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '1rem', color: '#e2e8f0' }}>Architecture Overview</h3>
          <ul style={{ margin: 0, paddingLeft: '1.25rem', color: '#94a3b8', lineHeight: 1.6 }}>
            <li><strong>Frontend:</strong> React 18 + TypeScript + Vite dashboard with fast HMR.</li>
            <li><strong>Backend:</strong> High-performance FastAPI server with modular architecture and Pydantic validation.</li>
            <li><strong>Compute Target:</strong> NVIDIA RTX 3050 Laptop GPU (4GB VRAM) optimized with mixed precision &amp; memory checkpointing.</li>
            <li><strong>Future Pipelines:</strong> Multimodal alignment, convective parameter assimilation, and spatial-temporal transformer nowcasters.</li>
          </ul>
        </div>
      </main>

      <footer style={{ marginTop: 'auto', paddingTop: '2rem', fontSize: '0.75rem', color: '#64748b' }}>
        APM-Nowcast Research Prototype &copy; 2026. Built for India Meteorological Domain.
      </footer>
    </div>
  );
};

export default App;
