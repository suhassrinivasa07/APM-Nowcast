/**
 * Geospatial Convective Forecast Map
 * Interactive Leaflet map centered over India (22.5°N, 79.0°E)
 * Renders spatial heatmaps/overlays for:
 *   - Predicted Lightning Occurrence Probability (0-100%)
 *   - Expected Flash Density (flashes / km² / 15min)
 *   - Total Predictive Uncertainty (Aleatoric + Epistemic)
 *   - IMD Doppler Weather Radar (DWR) 250km ranges and station markers
 */

import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import {
  ConvectiveStormCell,
  HorizonInfo,
  OverlayMode,
  SensorStation,
} from '../types/nowcast';
import {
  Layers,
  Radio,
  Zap,
  Eye,
  Compass,
  Activity,
  AlertTriangle,
} from 'lucide-react';

interface ForecastMapProps {
  cells: ConvectiveStormCell[];
  stations: SensorStation[];
  horizon: HorizonInfo;
  overlayMode: OverlayMode;
  onOverlayModeChange: (mode: OverlayMode) => void;
  selectedCell: ConvectiveStormCell | null;
  onSelectCell: (cell: ConvectiveStormCell | null) => void;
  uncertaintyInflation: number;
}

export const ForecastMap: React.FC<ForecastMapProps> = ({
  cells,
  stations,
  horizon,
  overlayMode,
  onOverlayModeChange,
  selectedCell,
  onSelectCell,
  uncertaintyInflation,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const cellLayerGroupRef = useRef<L.LayerGroup | null>(null);
  const stationLayerGroupRef = useRef<L.LayerGroup | null>(null);
  const rangeRingsGroupRef = useRef<L.LayerGroup | null>(null);

  const [showStations, setShowStations] = useState<boolean>(true);
  const [showRangeRings, setShowRangeRings] = useState<boolean>(true);
  const [showLabels, setShowLabels] = useState<boolean>(true);

  // Initialize Leaflet map
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    // Center over India (22.5°N, 79.0°E) with zoom level 5
    const map = L.map(mapContainerRef.current, {
      center: [22.5, 79.0],
      zoom: 5,
      minZoom: 4,
      maxZoom: 10,
      zoomControl: false,
    });

    // Dark theme CartoDB basemap tiles
    L.tileLayer(
      'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
      {
        attribution:
          '&copy; <a href="https://carto.com/">CARTO</a> | India Met Domain',
        subdomains: 'abcd',
        maxZoom: 19,
      }
    ).addTo(map);

    // Zoom control in bottom right
    L.control.zoom({ position: 'bottomright' }).addTo(map);

    // Create layer groups
    rangeRingsGroupRef.current = L.layerGroup().addTo(map);
    cellLayerGroupRef.current = L.layerGroup().addTo(map);
    stationLayerGroupRef.current = L.layerGroup().addTo(map);

    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Update Convective Cells & Overlays
  useEffect(() => {
    const layer = cellLayerGroupRef.current;
    if (!layer || !mapInstanceRef.current) return;

    layer.clearLayers();

    cells.forEach((cell) => {
      let fillColor = '#10b981';
      let strokeColor = '#34d399';
      let fillOpacity = 0.45;
      let radiusMeters = cell.radiusKm * 1000;

      if (overlayMode === 'probability') {
        const p = cell.occurrenceProb;
        if (p >= 0.8) {
          fillColor = '#ef4444'; // Red
          strokeColor = '#f87171';
          fillOpacity = 0.65;
        } else if (p >= 0.6) {
          fillColor = '#f97316'; // Orange
          strokeColor = '#fb923c';
          fillOpacity = 0.55;
        } else if (p >= 0.35) {
          fillColor = '#eab308'; // Amber
          strokeColor = '#fde047';
          fillOpacity = 0.45;
        } else {
          fillColor = '#10b981'; // Green
          strokeColor = '#34d399';
          fillOpacity = 0.35;
        }
      } else if (overlayMode === 'flash_density') {
        const fd = cell.flashDensity;
        if (fd >= 25) {
          fillColor = '#f43f5e'; // Bright Rose/Magenta
          strokeColor = '#fb7185';
          fillOpacity = 0.70;
        } else if (fd >= 15) {
          fillColor = '#a855f7'; // Purple
          strokeColor = '#c084fc';
          fillOpacity = 0.60;
        } else if (fd >= 8) {
          fillColor = '#3b82f6'; // Blue
          strokeColor = '#60a5fa';
          fillOpacity = 0.50;
        } else {
          fillColor = '#06b6d4'; // Cyan
          strokeColor = '#22d3ee';
          fillOpacity = 0.40;
        }
      } else if (overlayMode === 'uncertainty') {
        const u = cell.totalUncertainty;
        if (u >= 0.7) {
          fillColor = '#ec4899'; // High uncertainty warning pink
          strokeColor = '#f472b6';
          fillOpacity = 0.60;
        } else if (u >= 0.45) {
          fillColor = '#eab308';
          strokeColor = '#facc15';
          fillOpacity = 0.50;
        } else {
          fillColor = '#14b8a6'; // Low uncertainty teal
          strokeColor = '#2dd4bf';
          fillOpacity = 0.40;
        }
      }

      // Main convective cell circle
      const circle = L.circle([cell.lat, cell.lon], {
        radius: radiusMeters,
        color: strokeColor,
        weight: selectedCell?.id === cell.id ? 3 : 1.5,
        fillColor: fillColor,
        fillOpacity: fillOpacity,
        dashArray: overlayMode === 'uncertainty' ? '4, 4' : undefined,
      });

      // Inner core circle for high-intensity convection
      if (cell.occurrenceProb > 0.6 || cell.flashDensity > 15) {
        L.circle([cell.lat, cell.lon], {
          radius: radiusMeters * 0.4,
          color: '#ffffff',
          weight: 1,
          fillColor: '#ffffff',
          fillOpacity: 0.35,
        }).addTo(layer);
      }

      // Tooltip
      circle.bindTooltip(
        `<div style="font-family: ui-sans-serif, system-ui; font-size: 11px; padding: 4px 6px; background: #0f172a; color: #f8fafc; border: 1px solid #334155; border-radius: 4px;">
          <div style="font-weight: 700; color: #38bdf8;">${cell.name} (${horizon.label})</div>
          <div>Prob: <span style="font-weight: 600; color: #facc15;">${Math.round(cell.occurrenceProb * 100)}%</span></div>
          <div>Flash Rate: <span style="font-weight: 600; color: #f43f5e;">${cell.flashDensity} fl/km²/15m</span></div>
          <div>CAPE: ${cell.cape} J/kg | CIN: ${cell.cin} J/kg</div>
          <div>Uncertainty: ±${(cell.totalUncertainty * 100).toFixed(0)}%</div>
        </div>`,
        { sticky: true, opacity: 0.95 }
      );

      circle.on('click', () => {
        onSelectCell(cell);
      });

      circle.addTo(layer);

      // Label marker
      if (showLabels) {
        const textIcon = L.divIcon({
          className: 'storm-cell-label',
          html: `<div style="
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(56, 189, 248, 0.4);
            border-radius: 4px;
            padding: 2px 6px;
            color: #f1f5f9;
            font-size: 10px;
            font-weight: 600;
            white-space: nowrap;
            pointer-events: none;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
          ">
            ⚡ ${cell.name.split(' ')[0]} <span style="color: #facc15;">${Math.round(cell.occurrenceProb * 100)}%</span>
          </div>`,
          iconSize: [80, 20],
          iconAnchor: [40, 10],
        });
        L.marker([cell.lat, cell.lon], { icon: textIcon, interactive: false }).addTo(layer);
      }
    });
  }, [cells, overlayMode, horizon, selectedCell, showLabels, onSelectCell]);

  // Update Stations & DWR Range Rings
  useEffect(() => {
    const stGroup = stationLayerGroupRef.current;
    const ringGroup = rangeRingsGroupRef.current;
    if (!stGroup || !ringGroup) return;

    stGroup.clearLayers();
    ringGroup.clearLayers();

    if (!showStations) return;

    stations.forEach((st) => {
      // Station pin styling based on type
      let iconColor = '#38bdf8'; // Sky blue for radar
      let iconSymbol = '📡';

      if (st.type === 'satellite') {
        iconColor = '#a855f7';
        iconSymbol = '🛰️';
      } else if (st.type === 'lightning') {
        iconColor = '#eab308';
        iconSymbol = '⚡';
      }

      // Add DWR radar coverage 250km circle
      if (st.type === 'radar' && showRangeRings && st.status === 'online') {
        L.circle([st.lat, st.lon], {
          radius: st.rangeKm * 1000,
          color: '#0284c7',
          weight: 1,
          opacity: 0.35,
          fillColor: '#0284c7',
          fillOpacity: 0.04,
          dashArray: '5, 5',
        }).addTo(ringGroup);

        // 100km inner Doppler core range ring
        L.circle([st.lat, st.lon], {
          radius: 100 * 1000,
          color: '#38bdf8',
          weight: 0.8,
          opacity: 0.25,
          fillOpacity: 0,
          dashArray: '2, 4',
        }).addTo(ringGroup);
      }

      // Custom HTML Marker for sensor station
      const markerIcon = L.divIcon({
        className: 'sensor-station-icon',
        html: `<div style="
          width: 22px;
          height: 22px;
          border-radius: 50%;
          background: #0f172a;
          border: 2px solid ${iconColor};
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 11px;
          box-shadow: 0 0 10px ${iconColor}66;
          cursor: pointer;
        ">
          ${iconSymbol}
        </div>`,
        iconSize: [22, 22],
        iconAnchor: [11, 11],
      });

      const marker = L.marker([st.lat, st.lon], { icon: markerIcon });
      marker.bindTooltip(
        `<div style="font-family: ui-sans-serif, system-ui; font-size: 11px; padding: 4px; background: #0f172a; color: #f8fafc; border: 1px solid #334155; border-radius: 4px;">
          <div style="font-weight: 700; color: ${iconColor};">${st.name}</div>
          <div style="color: #94a3b8;">${st.description}</div>
          <div style="font-size: 10px; color: #22c55e;">Status: ${st.status.toUpperCase()}</div>
          ${st.band ? `<div style="font-size: 10px; color: #cbd5e1;">Band: ${st.band}</div>` : ''}
          ${st.type === 'radar' ? `<div style="font-size: 10px; color: #38bdf8;">Range: ${st.rangeKm} km</div>` : ''}
        </div>`,
        { sticky: true }
      );

      marker.addTo(stGroup);
    });
  }, [stations, showStations, showRangeRings]);

  const resetView = () => {
    if (mapInstanceRef.current) {
      mapInstanceRef.current.setView([22.5, 79.0], 5, { animate: true });
    }
  };

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: '560px', overflow: 'hidden', borderRadius: '12px', border: '1px solid #334155' }}>
      {/* The Leaflet Container */}
      <div ref={mapContainerRef} style={{ width: '100%', height: '100%', minHeight: '560px', background: '#090d16' }} />

      {/* Top Left: Overlay Switcher & Metrics Selector */}
      <div
        style={{
          position: 'absolute',
          top: '14px',
          left: '14px',
          zIndex: 1000,
          background: 'rgba(15, 23, 42, 0.88)',
          backdropFilter: 'blur(8px)',
          border: '1px solid rgba(51, 65, 85, 0.9)',
          borderRadius: '8px',
          padding: '6px 8px',
          display: 'flex',
          gap: '6px',
          boxShadow: '0 8px 24px rgba(0, 0, 0, 0.5)',
        }}
      >
        <button
          onClick={() => onOverlayModeChange('probability')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '6px 12px',
            fontSize: '12px',
            fontWeight: 600,
            borderRadius: '6px',
            border: 'none',
            cursor: 'pointer',
            transition: 'all 0.2s',
            background: overlayMode === 'probability' ? '#0284c7' : 'transparent',
            color: overlayMode === 'probability' ? '#ffffff' : '#94a3b8',
          }}
        >
          <Activity size={14} />
          Lightning Prob (%)
        </button>

        <button
          onClick={() => onOverlayModeChange('flash_density')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '6px 12px',
            fontSize: '12px',
            fontWeight: 600,
            borderRadius: '6px',
            border: 'none',
            cursor: 'pointer',
            transition: 'all 0.2s',
            background: overlayMode === 'flash_density' ? '#7c3aed' : 'transparent',
            color: overlayMode === 'flash_density' ? '#ffffff' : '#94a3b8',
          }}
        >
          <Zap size={14} />
          Flash Density (fl/km²)
        </button>

        <button
          onClick={() => onOverlayModeChange('uncertainty')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '6px 12px',
            fontSize: '12px',
            fontWeight: 600,
            borderRadius: '6px',
            border: 'none',
            cursor: 'pointer',
            transition: 'all 0.2s',
            background: overlayMode === 'uncertainty' ? '#db2777' : 'transparent',
            color: overlayMode === 'uncertainty' ? '#ffffff' : '#94a3b8',
          }}
        >
          <AlertTriangle size={14} />
          Uncertainty Bounds
        </button>
      </div>

      {/* Top Right: Layer Visibility & View Reset Controls */}
      <div
        style={{
          position: 'absolute',
          top: '14px',
          right: '14px',
          zIndex: 1000,
          background: 'rgba(15, 23, 42, 0.88)',
          backdropFilter: 'blur(8px)',
          border: '1px solid rgba(51, 65, 85, 0.9)',
          borderRadius: '8px',
          padding: '6px 10px',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          fontSize: '12px',
          color: '#cbd5e1',
          boxShadow: '0 8px 24px rgba(0, 0, 0, 0.5)',
        }}
      >
        <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={showStations}
            onChange={(e) => setShowStations(e.target.checked)}
            style={{ accentColor: '#38bdf8' }}
          />
          <Radio size={13} color="#38bdf8" /> Stations
        </label>

        <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={showRangeRings}
            onChange={(e) => setShowRangeRings(e.target.checked)}
            style={{ accentColor: '#0284c7' }}
          />
          <Layers size={13} color="#60a5fa" /> DWR 250km
        </label>

        <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={showLabels}
            onChange={(e) => setShowLabels(e.target.checked)}
            style={{ accentColor: '#facc15' }}
          />
          <Eye size={13} color="#facc15" /> Labels
        </label>

        <button
          onClick={resetView}
          title="Reset to India Overview (22.5°N, 79.0°E)"
          style={{
            background: '#1e293b',
            border: '1px solid #475569',
            borderRadius: '4px',
            padding: '4px 8px',
            color: '#f8fafc',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            fontSize: '11px',
          }}
        >
          <Compass size={13} />
          Reset Map
        </button>
      </div>

      {/* Bottom Left: Dynamic Colorbar Legend */}
      <div
        style={{
          position: 'absolute',
          bottom: '24px',
          left: '14px',
          zIndex: 1000,
          background: 'rgba(15, 23, 42, 0.90)',
          backdropFilter: 'blur(8px)',
          border: '1px solid rgba(51, 65, 85, 0.9)',
          borderRadius: '8px',
          padding: '8px 12px',
          minWidth: '220px',
          boxShadow: '0 8px 24px rgba(0, 0, 0, 0.5)',
          fontSize: '11px',
        }}
      >
        <div style={{ fontWeight: 600, color: '#e2e8f0', marginBottom: '6px', display: 'flex', justifyContent: 'space-between' }}>
          <span>
            {overlayMode === 'probability' && 'Occurrence Probability'}
            {overlayMode === 'flash_density' && 'Flash Density (fl/km²/15m)'}
            {overlayMode === 'uncertainty' && 'Total Predictive Uncertainty'}
          </span>
          <span style={{ color: '#38bdf8', fontWeight: 700 }}>{horizon.label}</span>
        </div>

        {overlayMode === 'probability' && (
          <div>
            <div style={{ height: '8px', borderRadius: '4px', background: 'linear-gradient(to right, #10b981, #eab308, #f97316, #ef4444)' }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', marginTop: '4px', fontSize: '10px' }}>
              <span>0%</span>
              <span>30%</span>
              <span>60%</span>
              <span>80%+</span>
            </div>
          </div>
        )}

        {overlayMode === 'flash_density' && (
          <div>
            <div style={{ height: '8px', borderRadius: '4px', background: 'linear-gradient(to right, #06b6d4, #3b82f6, #a855f7, #f43f5e)' }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', marginTop: '4px', fontSize: '10px' }}>
              <span>0</span>
              <span>8</span>
              <span>15</span>
              <span>30+</span>
            </div>
          </div>
        )}

        {overlayMode === 'uncertainty' && (
          <div>
            <div style={{ height: '8px', borderRadius: '4px', background: 'linear-gradient(to right, #14b8a6, #eab308, #ec4899)' }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', marginTop: '4px', fontSize: '10px' }}>
              <span>Low (±15%)</span>
              <span>Mod (±45%)</span>
              <span>High (±75%+)</span>
            </div>
            {uncertaintyInflation > 1.2 && (
              <div style={{ marginTop: '6px', color: '#f43f5e', fontSize: '10px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <AlertTriangle size={12} />
                <span>Sensors missing: Epistemic variance inflated {uncertaintyInflation}x</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Selected Convective Cell Inspector Card */}
      {selectedCell && (
        <div
          style={{
            position: 'absolute',
            bottom: '24px',
            right: '14px',
            zIndex: 1000,
            background: 'rgba(15, 23, 42, 0.94)',
            backdropFilter: 'blur(10px)',
            border: '1px solid #0284c7',
            borderRadius: '10px',
            padding: '12px 14px',
            width: '270px',
            boxShadow: '0 12px 30px rgba(0, 0, 0, 0.7)',
            fontSize: '11px',
            color: '#f8fafc',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
            <span style={{ fontWeight: 700, fontSize: '12px', color: '#38bdf8' }}>{selectedCell.name}</span>
            <button
              onClick={() => onSelectCell(null)}
              style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: '14px' }}
            >
              ✕
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', margin: '8px 0', borderBottom: '1px solid #334155', paddingBottom: '8px' }}>
            <div>
              <span style={{ color: '#94a3b8' }}>Coord:</span>
              <div style={{ fontWeight: 600 }}>{selectedCell.lat}°N, {selectedCell.lon}°E</div>
            </div>
            <div>
              <span style={{ color: '#94a3b8' }}>Radius:</span>
              <div style={{ fontWeight: 600 }}>{selectedCell.radiusKm} km</div>
            </div>
            <div>
              <span style={{ color: '#94a3b8' }}>Prob ({horizon.label}):</span>
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
  );
};
