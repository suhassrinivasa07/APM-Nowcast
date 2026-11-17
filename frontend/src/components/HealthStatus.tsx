import React, { useEffect, useState } from 'react';
import { fetchHealth, HealthData } from '../api/client';
import { Activity, CheckCircle, AlertTriangle, RefreshCw, Cpu, Server } from 'lucide-react';

export const HealthStatus: React.FC = () => {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const checkHealth = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchHealth();
      setHealth(data);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Unknown error connecting to API');
      }
      setHealth(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  return (
    <div style={{
      padding: '1.5rem',
      borderRadius: '12px',
      backgroundColor: '#1e293b',
      color: '#f8fafc',
      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1)',
      border: '1px solid #334155',
      maxWidth: '560px',
      margin: '1.5rem auto',
      textAlign: 'left'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Activity size={22} color="#38bdf8" />
          <h2 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 600 }}>Backend Service Status</h2>
        </div>
        <button
          onClick={checkHealth}
          disabled={loading}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            padding: '0.4rem 0.8rem',
            fontSize: '0.85rem',
            borderRadius: '6px',
            border: 'none',
            backgroundColor: '#0284c7',
            color: '#fff',
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.7 : 1
          }}
        >
          <RefreshCw size={14} className={loading ? 'spin' : ''} />
          {loading ? 'Checking...' : 'Refresh'}
        </button>
      </div>

      {loading && !health && (
        <p style={{ color: '#94a3b8', fontStyle: 'italic', margin: 0 }}>Querying GET /health endpoint...</p>
      )}

      {error && (
        <div style={{
          display: 'flex',
          alignItems: 'flex-start',
          gap: '0.6rem',
          padding: '0.75rem',
          borderRadius: '8px',
          backgroundColor: 'rgba(239, 68, 68, 0.15)',
          border: '1px solid #ef4444',
          color: '#fca5a5'
        }}>
          <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <strong>Connection Warning:</strong>
            <div style={{ fontSize: '0.85rem', marginTop: '0.25rem' }}>{error}</div>
            <div style={{ fontSize: '0.75rem', marginTop: '0.4rem', color: '#cbd5e1' }}>
              Make sure the FastAPI backend is running on <code>http://localhost:8000</code>.
            </div>
          </div>
        </div>
      )}

      {health && (
        <div>
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.4rem',
            padding: '0.25rem 0.6rem',
            borderRadius: '9999px',
            backgroundColor: 'rgba(34, 197, 94, 0.2)',
            color: '#4ade80',
            fontSize: '0.85rem',
            fontWeight: 500,
            marginBottom: '1rem'
          }}>
            <CheckCircle size={15} />
            <span>Operational ({health.status.toUpperCase()})</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem', fontSize: '0.875rem' }}>
            <div style={{ backgroundColor: '#0f172a', padding: '0.6rem', borderRadius: '6px' }}>
              <div style={{ color: '#94a3b8', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                <Server size={13} /> Service
              </div>
              <div style={{ fontWeight: 600, marginTop: '2px' }}>{health.service}</div>
            </div>

            <div style={{ backgroundColor: '#0f172a', padding: '0.6rem', borderRadius: '6px' }}>
              <div style={{ color: '#94a3b8', fontSize: '0.75rem' }}>Version</div>
              <div style={{ fontWeight: 600, marginTop: '2px' }}>v{health.version}</div>
            </div>

            <div style={{ backgroundColor: '#0f172a', padding: '0.6rem', borderRadius: '6px' }}>
              <div style={{ color: '#94a3b8', fontSize: '0.75rem' }}>Environment</div>
              <div style={{ fontWeight: 600, marginTop: '2px', textTransform: 'capitalize' }}>{health.environment}</div>
            </div>

            <div style={{ backgroundColor: '#0f172a', padding: '0.6rem', borderRadius: '6px' }}>
              <div style={{ color: '#94a3b8', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                <Cpu size={13} /> Compute Target
              </div>
              <div style={{ fontWeight: 600, marginTop: '2px', textTransform: 'uppercase' }}>{health.device_target || 'N/A'}</div>
            </div>
          </div>

          <div style={{ marginTop: '0.8rem', fontSize: '0.75rem', color: '#64748b' }}>
            Last checked: {new Date(health.timestamp).toLocaleTimeString()} ({health.timestamp})
          </div>
        </div>
      )}
    </div>
  );
};
