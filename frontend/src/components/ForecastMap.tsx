/**
 * Geospatial Convective Forecast Map — Windy.com-Inspired 3D Satellite Edition
 *
 * Key Architecture:
 *   - Map initialized ONCE on mount (useEffect([], ...)); user zoom & pan strictly preserved across horizon ticks
 *   - Continuous 60fps requestAnimationFrame loop for 1,400 divergence-free wind streamline wisps (z-index: 450)
 *   - Continuous matte domain-warped organic scalar field heatmap in deep teal -> slate indigo -> purple -> rose-magenta (z-index: 430)
 *   - Darkened & desaturated satellite relief basemap (brightness 0.62, contrast 1.12, saturate 0.65)
 *   - Smooth in-place updates for storm cells, trajectories, and popups during Play mode
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import {
  ConvectiveStormCell,
  HorizonInfo,
  OverlayMode,
  SensorStation,
} from '../types/nowcast';
import {
  SUB_REGION_PRESETS,
  computeCellTrajectories,
} from '../api/nowcastApi';
import {
  Layers,
  Radio,
  Zap,
  Eye,
  Compass,
  Activity,
  AlertTriangle,
  TrendingUp,
  MapPin,
  Sparkles,
  Wind,
  Box,
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
  /** Called once the Leaflet map is mounted, with a flyTo(lat, lon) imperative handle */
  onFlyToReady?: (flyTo: (lat: number, lon: number) => void) => void;
}

interface WindParticle {
  lat: number;
  lon: number;
  prevLat?: number;
  prevLon?: number;
  age: number;
  maxAge: number;
  speedMultiplier: number;
}

const NUM_WIND_PARTICLES = 1400;
const GRID_W = 160;
const GRID_H = 120;

export const ForecastMap: React.FC<ForecastMapProps> = ({
  cells,
  stations,
  horizon,
  overlayMode,
  onOverlayModeChange,
  selectedCell,
  onSelectCell,
  uncertaintyInflation,
  onFlyToReady,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const heatmapCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const windCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);

  // Offscreen canvas and pixel buffer for smooth bilinear matte scalar field
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const offscreenCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const imageDataRef = useRef<ImageData | null>(null);

  const cellLayerGroupRef = useRef<L.LayerGroup | null>(null);
  const stationLayerGroupRef = useRef<L.LayerGroup | null>(null);
  const rangeRingsGroupRef = useRef<L.LayerGroup | null>(null);
  const trajectoryLayerGroupRef = useRef<L.LayerGroup | null>(null);

  // Floating HUD overlay refs for Leaflet click & scroll isolation
  const modeSwitcherRef = useRef<HTMLDivElement>(null);
  const layerControlsRef = useRef<HTMLDivElement>(null);
  const mobileLayersRef = useRef<HTMLDivElement>(null);
  const subRegionRef = useRef<HTMLDivElement>(null);
  const legendRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const overlays = [
      modeSwitcherRef.current,
      layerControlsRef.current,
      mobileLayersRef.current,
      subRegionRef.current,
      legendRef.current,
    ];
    overlays.forEach((el) => {
      if (el) {
        L.DomEvent.disableClickPropagation(el);
        L.DomEvent.disableScrollPropagation(el);
      }
    });
  }, []);

  // State refs to decouple active 60fps render loops from map re-initialization
  const cellsRef = useRef<ConvectiveStormCell[]>(cells);
  cellsRef.current = cells;

  const overlayModeRef = useRef<OverlayMode>(overlayMode);
  overlayModeRef.current = overlayMode;

  const [is3DMode, setIs3DMode] = useState<boolean>(false);
  const [showWindParticles, setShowWindParticles] = useState<boolean>(true);
  const [showPlumes, setShowPlumes] = useState<boolean>(true);
  const [showTrajectories, setShowTrajectories] = useState<boolean>(true);
  const isMobileInitial = typeof window !== 'undefined' && window.innerWidth < 768;
  const [showStations, setShowStations] = useState<boolean>(!isMobileInitial);
  const [showRangeRings, setShowRangeRings] = useState<boolean>(true);
  const [showLabels, setShowLabels] = useState<boolean>(true);
  const [mobileLayersOpen, setMobileLayersOpen] = useState<boolean>(false);
  const [activeRegionPreset, setActiveRegionPreset] = useState<string>('all_india');
  const [mapReady, setMapReady] = useState<boolean>(false);

  const showPlumesRef = useRef<boolean>(showPlumes);
  showPlumesRef.current = showPlumes;

  const particlesRef = useRef<WindParticle[]>([]);
  const animationFrameIdRef = useRef<number | null>(null);

  // =========================================================================
  // Initialize Wind Particles Array across Pan-India Domain (Reduced by 60% on mobile)
  // =========================================================================
  useEffect(() => {
    const isMobile = typeof window !== 'undefined' && window.innerWidth < 768;
    const particleCount = isMobile ? Math.round(NUM_WIND_PARTICLES * 0.4) : NUM_WIND_PARTICLES;
    const particles: WindParticle[] = [];
    for (let i = 0; i < particleCount; i++) {
      const lat = 6.0 + Math.random() * 32.0; // 6.0°N to 38.0°N
      const lon = 68.0 + Math.random() * 30.0; // 68.0°E to 98.0°E
      particles.push({
        lat,
        lon,
        prevLat: lat,
        prevLon: lon,
        age: Math.floor(Math.random() * 100),
        maxAge: 60 + Math.floor(Math.random() * 60),
        speedMultiplier: 0.85 + Math.random() * 0.35,
      });
    }
    particlesRef.current = particles;
  }, []);

  // Initialize offscreen grid canvas once
  useEffect(() => {
    if (typeof document !== 'undefined') {
      const oc = document.createElement('canvas');
      oc.width = GRID_W;
      oc.height = GRID_H;
      offscreenCanvasRef.current = oc;
      const octx = oc.getContext('2d');
      offscreenCtxRef.current = octx;
      if (octx) {
        imageDataRef.current = octx.createImageData(GRID_W, GRID_H);
      }
    }
  }, []);

  // =========================================================================
  // Synchronize Canvas Layer Offset & Dimensions with Leaflet (High-DPI)
  // containerPointToLayerPoint([0, 0]) keeps canvases locked to viewport
  // =========================================================================
  const syncCanvasSizeAndPosition = useCallback(() => {
    const map = mapInstanceRef.current;
    const hCanvas = heatmapCanvasRef.current;
    const wCanvas = windCanvasRef.current;
    if (!map) return;

    const container = map.getContainer();
    const w = container.clientWidth || map.getSize().x || 800;
    const h = container.clientHeight || map.getSize().y || 600;
    const dpr = window.devicePixelRatio || 1;

    const targetWidth = Math.round(w * dpr);
    const targetHeight = Math.round(h * dpr);

    const origin = map.containerPointToLayerPoint([0, 0]);

    if (hCanvas) {
      L.DomUtil.setPosition(hCanvas, origin);
      hCanvas.style.width = `${w}px`;
      hCanvas.style.height = `${h}px`;
      if (hCanvas.width !== targetWidth || hCanvas.height !== targetHeight) {
        hCanvas.width = targetWidth;
        hCanvas.height = targetHeight;
      }
    }

    if (wCanvas) {
      L.DomUtil.setPosition(wCanvas, origin);
      wCanvas.style.width = `${w}px`;
      wCanvas.style.height = `${h}px`;
      if (wCanvas.width !== targetWidth || wCanvas.height !== targetHeight) {
        wCanvas.width = targetWidth;
        wCanvas.height = targetHeight;
      }
    }
  }, []);

  const syncCanvasDimensions = syncCanvasSizeAndPosition;

  // =========================================================================
  // 1. Matte, Domain-Warped Organic Scalar Field Heatmap (z-index: 410)
  //    No radial circles, no hard bounding box; transparent when v < 0.08
  // =========================================================================
  const renderHeatmap = useCallback(() => {
    const canvas = heatmapCanvasRef.current;
    const map = mapInstanceRef.current;
    if (!canvas || !map) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    syncCanvasSizeAndPosition();

    const dpr = window.devicePixelRatio || 1;
    const width = canvas.width / dpr;
    const height = canvas.height / dpr;

    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    if (!showPlumesRef.current) {
      ctx.restore();
      return;
    }

    // Normal matte alpha blending (no neon screen blend)
    ctx.globalCompositeOperation = 'source-over';
    ctx.imageSmoothingEnabled = true;

    // Ensure offscreen grid canvas exists
    if (!offscreenCanvasRef.current && typeof document !== 'undefined') {
      const oc = document.createElement('canvas');
      oc.width = GRID_W;
      oc.height = GRID_H;
      offscreenCanvasRef.current = oc;
      const octx = oc.getContext('2d');
      offscreenCtxRef.current = octx;
      if (octx) {
        imageDataRef.current = octx.createImageData(GRID_W, GRID_H);
      }
    }

    const offCtx = offscreenCtxRef.current;
    const offCanvas = offscreenCanvasRef.current;
    const imgData = imageDataRef.current;
    if (!offCtx || !offCanvas || !imgData) {
      ctx.restore();
      return;
    }

    const bounds = map.getBounds();
    const southWest = bounds.getSouthWest();
    const northEast = bounds.getNorthEast();
    const minLat = southWest.lat;
    const maxLat = northEast.lat;
    const minLon = southWest.lng;
    const maxLon = northEast.lng;
    const dLat = (maxLat - minLat) / GRID_H;
    const dLon = (maxLon - minLon) / GRID_W;

    const activeCells = cellsRef.current;
    const trajectories = computeCellTrajectories(activeCells);
    const currentMode = overlayModeRef.current;

    // Pre-calculate storm cell parameters outside the grid loop for high performance
    const preparedCells = activeCells.map((cell) => {
      const traj = trajectories.find((t) => t.cellId.startsWith(cell.id.split('-h')[0]));
      const bearingRad = traj ? (traj.bearingDeg * Math.PI) / 180 : (68 * Math.PI) / 180;
      const cosLat = Math.cos((cell.lat * Math.PI) / 180);
      const rDeg = Math.max(cell.radiusKm / 111.32, 0.28);

      let weight = cell.occurrenceProb;
      // Hide or de-emphasize calm < 25% cells from high-severity visual halos so
      // the purple convective plume focuses on genuine active systems (Westcentral Bay of Bengal, South Bay, Coastal Andhra)
      if (cell.occurrenceProb < 0.25) {
        weight = cell.occurrenceProb * 0.10;
      }
      if (currentMode === 'flash_density') {
        weight = cell.occurrenceProb < 0.25 ? 0.0 : Math.min(1.0, (cell.flashDensity + 8) / 36);
      } else if (currentMode === 'uncertainty') {
        weight = cell.totalUncertainty;
      }

      return {
        lat: cell.lat,
        lon: cell.lon,
        cosLat,
        rDeg,
        cosTheta: Math.cos(bearingRad),
        sinTheta: Math.sin(bearingRad),
        weight,
      };
    });

    const data = imgData.data;

    for (let y = 0; y < GRID_H; y++) {
      const lat = maxLat - y * dLat;
      const yRowOffset = y * GRID_W * 4;

      for (let x = 0; x < GRID_W; x++) {
        const lon = minLon + x * dLon;

        // Smooth synoptic convergence trough (fades smoothly in all directions without any bounding box)
        const troughLat = 20.0 + 0.60 * (88.0 - lon);
        const distToTrough = Math.abs(lat - troughLat);
        const troughFactor = 0.14 * Math.exp(-(distToTrough * distToTrough) / 8.0) * Math.max(0, 1.0 - Math.abs(lon - 82.0) / 14.0);
        let field = troughFactor;

        // Storm cell plumes with multi-octave domain warping & elliptical wind-shear elongation
        for (let i = 0; i < preparedCells.length; i++) {
          const cell = preparedCells[i];
          const dLatDeg = lat - cell.lat;
          const dLonDeg = (lon - cell.lon) * cell.cosLat;

          // Fast bounding box rejection (skip points further than 4 radii)
          if (Math.abs(dLatDeg) > cell.rDeg * 3.8 || Math.abs(dLonDeg) > cell.rDeg * 4.5) {
            continue;
          }

          // Elliptical wind-shear coordinate rotation along storm velocity vector
          const xAlong = dLonDeg * cell.cosTheta + dLatDeg * cell.sinTheta;
          const yCross = -dLonDeg * cell.sinTheta + dLatDeg * cell.cosTheta;

          // Asymmetric downwind elongation (downwind anvil vs upwind core)
          const aRadius = xAlong >= 0 ? cell.rDeg * 1.65 : cell.rDeg * 0.95;
          const bRadius = cell.rDeg * 1.15;

          // Multi-octave trigonometric domain warping pseudo-noise
          const warp1 = 0.16 * Math.sin(2.4 * lat + 1.8 * lon) + 0.08 * Math.cos(4.8 * lat - 3.6 * lon);
          const warp2 = 0.16 * Math.cos(2.8 * lat - 1.5 * lon) + 0.08 * Math.sin(5.2 * lat + 4.2 * lon);

          const xNorm = xAlong / aRadius + warp1;
          const yNorm = yCross / bRadius + warp2;
          const distSq = xNorm * xNorm + yNorm * yNorm;

          if (distSq < 4.2) {
            // Smooth organic bell plume
            const plume = cell.weight * Math.exp(-distSq / 1.55);
            // Soft saturation accumulation
            field = field + plume * (1.0 - 0.45 * field);
          }
        }

        // Clamp field to [0, 1]
        const val = Math.min(1.0, Math.max(0.0, field));

        // Zero-intensity transparent cutoff (v < 0.08 => alpha = 0, no light-blue bounding box)
        let r = 0, g = 0, b = 0, a = 0;

        if (val >= 0.08) {
          // Smoothly ramp alpha from 0 at val = 0.08 up to 0.62 (158 in 0-255 scale) at val = 1.0
          const normAlpha = ((val - 0.08) / 0.92) * 0.62;
          a = Math.round(normAlpha * 255);

          const t = (val - 0.08) / 0.92;

          if (currentMode === 'uncertainty') {
            // Slate Indigo (#3b4ca8) -> Deep Teal (#1b6b88) -> Warm Amber (#d97706) -> Rose (#be185d)
            if (t <= 0.33) {
              const u = t / 0.33;
              r = Math.round(27 + (59 - 27) * u);
              g = Math.round(107 + (76 - 107) * u);
              b = Math.round(136 + (168 - 136) * u);
            } else if (t <= 0.67) {
              const u = (t - 0.33) / 0.34;
              r = Math.round(59 + (217 - 59) * u);
              g = Math.round(76 + (119 - 76) * u);
              b = Math.round(168 + (6 - 168) * u);
            } else {
              const u = (t - 0.67) / 0.33;
              r = Math.round(217 + (190 - 217) * u);
              g = Math.round(119 + (24 - 119) * u);
              b = Math.round(6 + (93 - 6) * u);
            }
          } else {
            // Natural Windy Matte Palette:
            // Deep Oceanic Teal (#1b6b88) -> Slate Indigo (#3b4ca8) -> Muted Purple (#7e3af2) -> Rich Rose-Magenta (#be185d)
            if (t <= 0.32) {
              const u = t / 0.32;
              r = Math.round(27 + (59 - 27) * u);
              g = Math.round(107 + (76 - 107) * u);
              b = Math.round(136 + (168 - 136) * u);
            } else if (t <= 0.68) {
              const u = (t - 0.32) / 0.36;
              r = Math.round(59 + (126 - 59) * u);
              g = Math.round(76 + (58 - 76) * u);
              b = Math.round(168 + (242 - 168) * u);
            } else {
              const u = (t - 0.68) / 0.32;
              r = Math.round(126 + (190 - 126) * u);
              g = Math.round(58 + (24 - 58) * u);
              b = Math.round(242 + (93 - 242) * u);
            }
          }
        }

        const pIdx = yRowOffset + x * 4;
        data[pIdx] = r;
        data[pIdx + 1] = g;
        data[pIdx + 2] = b;
        data[pIdx + 3] = a;
      }
    }

    offCtx.putImageData(imgData, 0, 0);
    ctx.drawImage(offCanvas, 0, 0, width, height);

    ctx.restore();
  }, [syncCanvasDimensions]);

  // =========================================================================
  // 2. Divergence-Free Rotational Wind Field via Streamfunction (z-index: 420)
  //    div(u, v) === 0: no collision seam, no vertical convergence line
  // =========================================================================
  useEffect(() => {
    const canvas = windCanvasRef.current;
    if (!canvas || !showWindParticles || !mapReady) {
      if (animationFrameIdRef.current) {
        cancelAnimationFrame(animationFrameIdRef.current);
        animationFrameIdRef.current = null;
      }
      return;
    }

    let isRunning = true;

    const animateWind = () => {
      if (!isRunning || !canvas || !mapInstanceRef.current) return;

      const mapInstance = mapInstanceRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      syncCanvasSizeAndPosition();

      const dpr = window.devicePixelRatio || 1;
      const width = canvas.width / dpr;
      const height = canvas.height / dpr;

      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      const bounds = mapInstance.getBounds();
      const southWest = bounds.getSouthWest();
      const northEast = bounds.getNorthEast();

      const particles = particlesRef.current;
      const activeCells = cellsRef.current;

      // Divergence-Free 2D Streamfunction psi(lat, lon)
      // Combines synoptic wave sin(lat * 0.12) * 1.8 - lon * 0.08
      // plus smooth Gaussian vortex potentials -strength * exp(-distSq / radiusSq)
      const samplePsi = (lat: number, lon: number): number => {
        let psi = Math.sin(lat * 0.12) * 1.8 - lon * 0.08;

        for (let cIdx = 0; cIdx < activeCells.length; cIdx++) {
          const cell = activeCells[cIdx];
          const cosLat = Math.cos((cell.lat * Math.PI) / 180);
          const dLon = (lon - cell.lon) * cosLat;
          const dLat = lat - cell.lat;
          const distSq = dLon * dLon + dLat * dLat;
          const radiusDeg = Math.max((cell.radiusKm * 2.8) / 111.32, 0.6);
          const radiusSq = radiusDeg * radiusDeg;

          if (distSq < radiusSq * 5.0) {
            const strength = 0.32 * Math.min(1.0, (cell.occurrenceProb || 0.6) * 1.25);
            psi -= strength * Math.exp(-distSq / radiusSq);
          }
        }

        return psi;
      };

      const eps = 0.04;
      const speedScale = 0.20;
      const minSpeedFloor = 0.024;

      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];

        // Central finite differences on divergence-free streamfunction
        // u = (psi(lat + eps, lon) - psi(lat - eps, lon)) / (2 * eps)
        // v = -(psi(lat, lon + eps) - psi(lat, lon - eps)) / (2 * eps)
        const uRaw = (samplePsi(p.lat + eps, p.lon) - samplePsi(p.lat - eps, p.lon)) / (2 * eps);
        const vRaw = -(samplePsi(p.lat, p.lon + eps) - samplePsi(p.lat, p.lon - eps)) / (2 * eps);

        let u = uRaw * speedScale;
        let v = vRaw * speedScale;

        // Enforce minimum particle speed floor so particles flow continuously
        const currentSpeed = Math.sqrt(u * u + v * v);
        if (currentSpeed < minSpeedFloor) {
          const boost = minSpeedFloor / (currentSpeed || 0.001);
          u *= boost;
          v *= boost;
        }

        // Advance particle position
        p.prevLat = p.lat;
        p.prevLon = p.lon;
        p.lon += u * p.speedMultiplier;
        p.lat += v * p.speedMultiplier;
        p.age++;

        // Random respawn across full viewport bounds so density stays completely uniform
        if (
          p.age >= p.maxAge ||
          p.lat < southWest.lat - 0.5 ||
          p.lat > northEast.lat + 0.5 ||
          p.lon < southWest.lng - 0.5 ||
          p.lon > northEast.lng + 0.5
        ) {
          p.lat = southWest.lat + Math.random() * (northEast.lat - southWest.lat);
          p.lon = southWest.lng + Math.random() * (northEast.lng - southWest.lng);
          p.prevLat = p.lat;
          p.prevLon = p.lon;
          p.age = 0;
          p.maxAge = 60 + Math.floor(Math.random() * 60);
          continue;
        }

        const pt1 = mapInstance.latLngToContainerPoint([p.lat, p.lon]);

        // Culling
        if (pt1.x < -30 || pt1.x > width + 30 || pt1.y < -30 || pt1.y > height + 30) {
          continue;
        }

        // Angle and streak length: 10–14px
        const screenSpeed = Math.sqrt(u * u + v * v) * 95;
        const streakLen = Math.min(14, Math.max(10, screenSpeed * 5.2 * p.speedMultiplier));
        const angle = Math.atan2(-v, u);

        const startX = pt1.x - Math.cos(angle) * streakLen;
        const startY = pt1.y - Math.sin(angle) * streakLen;

        // Smooth lifecycle opacity curve (peak 0.55)
        const lifeFraction = p.age / p.maxAge;
        const alpha = Math.sin(lifeFraction * Math.PI) * 0.55;

        // Tapered fading tail using linear gradient along the streak
        const grad = ctx.createLinearGradient(startX, startY, pt1.x, pt1.y);
        grad.addColorStop(0, 'rgba(255, 255, 255, 0)');
        grad.addColorStop(1, `rgba(255, 255, 255, ${alpha.toFixed(3)})`);

        // Draw visible 10–14px silky streamline wisp (lineWidth: 1.15)
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.15;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(pt1.x, pt1.y);
        ctx.stroke();

        // Subtle 3.5px directional chevron tip at the leading edge
        const chevLen = 3.5;
        const wing1X = pt1.x - chevLen * Math.cos(angle - 0.52);
        const wing1Y = pt1.y - chevLen * Math.sin(angle - 0.52);
        const wing2X = pt1.x - chevLen * Math.cos(angle + 0.52);
        const wing2Y = pt1.y - chevLen * Math.sin(angle + 0.52);
        ctx.strokeStyle = `rgba(255, 255, 255, ${(alpha * 0.95).toFixed(3)})`;
        ctx.lineWidth = 1.15;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(wing1X, wing1Y);
        ctx.lineTo(pt1.x, pt1.y);
        ctx.lineTo(wing2X, wing2Y);
        ctx.stroke();
      }

      ctx.restore();
      animationFrameIdRef.current = requestAnimationFrame(animateWind);
    };

    animationFrameIdRef.current = requestAnimationFrame(animateWind);

    return () => {
      isRunning = false;
      if (animationFrameIdRef.current) {
        cancelAnimationFrame(animationFrameIdRef.current);
        animationFrameIdRef.current = null;
      }
    };
  }, [showWindParticles, mapReady, syncCanvasSizeAndPosition]);

  // =========================================================================
  // Initialize Leaflet Map ONCE on Mount (Never re-initialize on horizon ticks)
  // =========================================================================
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    // Center over India (22.5°N, 79.0°E) with zoom level 5 (Pan-India Domain: 6.0°N–38.0°N, 68.0°E–98.0°E)
    const map = L.map(mapContainerRef.current, {
      center: [22.5, 79.0],
      zoom: 5,
      minZoom: 4,
      maxZoom: 14,
      zoomControl: false,
    });

    // 1. Create custom Leaflet panes inside single stacking context below markers & labels
    const heatmapPane = map.createPane('heatmapPane');
    heatmapPane.style.zIndex = '410';
    heatmapPane.style.pointerEvents = 'none';

    const windPane = map.createPane('windPane');
    windPane.style.zIndex = '420';
    windPane.style.pointerEvents = 'none';

    const labelsPane = map.createPane('labelsPane');
    labelsPane.style.zIndex = '520';
    labelsPane.style.pointerEvents = 'none';

    // Set Leaflet's built-in panes z-index order: markerPane 600, tooltipPane 900, popupPane 950
    map.getPane('markerPane')!.style.zIndex = '600';
    map.getPane('tooltipPane')!.style.zIndex = '900';
    map.getPane('popupPane')!.style.zIndex = '950';

    // 2. Mount Heatmap & Wind Canvases directly inside Leaflet panes with explicit pixel dimensions
    const container = map.getContainer();
    const w = container.clientWidth || map.getSize().x || 800;
    const h = container.clientHeight || map.getSize().y || 600;
    const dpr = window.devicePixelRatio || 1;

    const hCanvas = document.createElement('canvas');
    hCanvas.className = 'apm-heatmap-canvas';
    hCanvas.style.position = 'absolute';
    hCanvas.style.pointerEvents = 'none';
    hCanvas.style.zIndex = '410';
    hCanvas.style.width = `${w}px`;
    hCanvas.style.height = `${h}px`;
    hCanvas.width = Math.round(w * dpr);
    hCanvas.height = Math.round(h * dpr);
    heatmapPane.appendChild(hCanvas);
    heatmapCanvasRef.current = hCanvas;

    const wCanvas = document.createElement('canvas');
    wCanvas.className = 'apm-wind-canvas';
    wCanvas.style.position = 'absolute';
    wCanvas.style.pointerEvents = 'none';
    wCanvas.style.zIndex = '420';
    wCanvas.style.width = `${w}px`;
    wCanvas.style.height = `${h}px`;
    wCanvas.width = Math.round(w * dpr);
    wCanvas.height = Math.round(h * dpr);
    windPane.appendChild(wCanvas);
    windCanvasRef.current = wCanvas;

    // 3. Esri World Imagery Satellite Tiles (Darkened relief filter via CSS)
    L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      {
        attribution:
          'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP | Pan-India (6.0°N–38.0°N, 68.0°E–98.0°E)',
        maxZoom: 16,
      }
    ).addTo(map);

    // 4. Esri Reference Boundaries & Places (Mounted in top labelsPane at z-index: 520 with opacity 0.9)
    L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
      {
        maxZoom: 16,
        opacity: 0.9,
        pane: 'labelsPane',
      }
    ).addTo(map);

    // Zoom control in bottom right
    L.control.zoom({ position: 'bottomright' }).addTo(map);

    // Create LayerGroups for in-place data updates (markerPane: z-index 600)
    rangeRingsGroupRef.current = L.layerGroup().addTo(map);
    trajectoryLayerGroupRef.current = L.layerGroup().addTo(map);
    cellLayerGroupRef.current = L.layerGroup().addTo(map);
    stationLayerGroupRef.current = L.layerGroup().addTo(map);

    mapInstanceRef.current = map;

    // Register flyTo imperative handle for map-to-sidebar sync
    if (onFlyToReady) {
      onFlyToReady((lat: number, lon: number) => {
        map.flyTo([lat, lon], Math.max(map.getZoom(), 7), { animate: true, duration: 1.2 });
      });
    }

    // Immediate initial sync and render on mount
    syncCanvasSizeAndPosition();
    renderHeatmap();
    setMapReady(true);

    const handleMapResize = () => {
      if (!mapInstanceRef.current) return;
      mapInstanceRef.current.invalidateSize({ debounceMoveend: true });
      syncCanvasSizeAndPosition();
      renderHeatmap();
    };

    map.on('move', () => {
      syncCanvasSizeAndPosition();
      renderHeatmap();
    });
    map.on('zoom', () => {
      syncCanvasSizeAndPosition();
      renderHeatmap();
    });
    map.on('resize', () => {
      handleMapResize();
    });
    map.on('viewreset', () => {
      syncCanvasSizeAndPosition();
      renderHeatmap();
    });

    // ResizeObserver for zero-delay response to container resizing
    const ro = new ResizeObserver(() => {
      handleMapResize();
    });
    if (mapContainerRef.current) {
      ro.observe(mapContainerRef.current);
    }

    window.addEventListener('resize', handleMapResize);
    window.addEventListener('orientationchange', handleMapResize);

    const initialResizeTimer = setTimeout(() => {
      handleMapResize();
    }, 150);

    return () => {
      clearTimeout(initialResizeTimer);
      setMapReady(false);
      ro.disconnect();
      window.removeEventListener('resize', handleMapResize);
      window.removeEventListener('orientationchange', handleMapResize);
      map.off('move');
      map.off('zoom');
      map.off('resize');
      map.off('viewreset');
      if (heatmapCanvasRef.current && heatmapCanvasRef.current.parentNode) {
        heatmapCanvasRef.current.parentNode.removeChild(heatmapCanvasRef.current);
      }
      if (windCanvasRef.current && windCanvasRef.current.parentNode) {
        windCanvasRef.current.parentNode.removeChild(windCanvasRef.current);
      }
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []); // Mount ONCE! Strictly preserve user pan & zoom

  // Redraw heatmap whenever cells, overlayMode, or showPlumes change (without touching Leaflet map)
  useEffect(() => {
    renderHeatmap();
  }, [cells, overlayMode, showPlumes, renderHeatmap]);

  // =========================================================================
  // In-Place Smooth Update for Convective Cells & Floating Glass Badges
  // =========================================================================
  useEffect(() => {
    const layer = cellLayerGroupRef.current;
    if (!layer || !mapInstanceRef.current) return;

    layer.clearLayers();
    const trajectories = computeCellTrajectories(cells);

    cells.forEach((cell) => {
      let severityColor = '#10b981';
      let severityBg = 'rgba(16, 185, 129, 0.25)';
      const p = cell.occurrenceProb;
      const isSevere = p >= 0.8 || cell.severity === 'extreme';
      const traj = trajectories.find((t) => t.cellId.startsWith(cell.id.split('-h')[0]));

      if (p >= 0.8) {
        severityColor = '#ef4444';
        severityBg = 'rgba(239, 68, 68, 0.3)';
      } else if (p >= 0.6) {
        severityColor = '#f97316';
        severityBg = 'rgba(249, 115, 22, 0.3)';
      } else if (p >= 0.35) {
        severityColor = '#eab308';
        severityBg = 'rgba(234, 179, 8, 0.3)';
      }

      // Rich Interactive Hover Tooltip & Popup Content with Solid Dark Slate Surface
      const popupContent = `
        <div style="
          font-family: ui-sans-serif, system-ui;
          min-width: 230px;
          padding: 8px 10px;
          background-color: #090d16;
          border: 1px solid rgba(148, 163, 184, 0.35);
          border-radius: 8px;
          box-shadow: 0 16px 36px rgba(0, 0, 0, 0.85);
          color: #f8fafc;
        ">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; border-bottom: 1px solid rgba(51, 65, 85, 0.8); padding-bottom: 5px;">
            <div style="font-weight: 700; font-size: 13px; color: #38bdf8; display: flex; align-items: center; gap: 4px;">
              ⚡ ${cell.name}
            </div>
            <span style="font-size: 9px; font-weight: 800; text-transform: uppercase; padding: 2px 6px; border-radius: 4px; background: ${severityBg}; color: ${severityColor};">
              ${cell.severity}
            </span>
          </div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 11px; margin-bottom: 6px;">
            <div><span style="color: #94a3b8;">Horizon:</span> <strong style="color: #38bdf8;">${horizon.label}</strong></div>
            <div><span style="color: #94a3b8;">Location:</span> <strong>${cell.lat.toFixed(2)}°N, ${cell.lon.toFixed(2)}°E</strong></div>
            <div><span style="color: #94a3b8;">Occurrence:</span> <strong style="color: #facc15;">${Math.round(cell.occurrenceProb * 100)}%</strong></div>
            <div><span style="color: #94a3b8;">Flash Rate:</span> <strong style="color: #f43f5e;">${cell.flashDensity} fl/km²</strong></div>
          </div>
          <div style="font-size: 10px; background-color: #0d121f; border-radius: 6px; padding: 6px; border: 1px solid rgba(51, 65, 85, 0.6); line-height: 1.4;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
              <span style="color: #94a3b8;">Thermodynamics:</span>
              <span>CAPE ${cell.cape} J/kg | CIN ${cell.cin} J/kg</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
              <span style="color: #94a3b8;">Motion Vector:</span>
              <span style="color: #facc15;">${Math.round(cell.motionSpeedKmh !== undefined ? cell.motionSpeedKmh : (traj?.speedKmH || 18))} km/h towards ${traj?.directionLabel || 'ENE'} (${cell.bearingDeg !== undefined ? cell.bearingDeg : (traj?.bearingDeg || 68)}°)</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
              <span style="color: #94a3b8;">Uncertainty (Total):</span>
              <span>±${Math.round(cell.totalUncertainty * 100)}% (alea: ±${Math.round(cell.aleatoricUncertainty * 100)}%)</span>
            </div>
          </div>
        </div>
      `;

      // Compact floating glassmorphic badge marker
      if (showLabels) {
        const shortName = cell.name.split(' ')[0];
        const probPct = Math.round(cell.occurrenceProb * 100);

        const badgeIcon = L.divIcon({
          className: 'floating-cell-badge',
          html: `
            <div style="position: relative; display: inline-block;">
              ${isSevere ? '<div class="severe-pulse-ring"></div>' : ''}
              <div class="glass-cell-pill" style="
                display: flex;
                align-items: center;
                gap: 5px;
                background-color: #090d16;
                border: 1px solid rgba(148, 163, 184, 0.35);
                border-left: 3px solid ${severityColor};
                border-radius: 9999px;
                padding: 2px 7px 2px 6px;
                color: #f8fafc;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.6);
                font-size: 11px;
                font-weight: 600;
                white-space: nowrap;
                cursor: pointer;
              ">
                <span style="width: 5px; height: 5px; border-radius: 50%; background: ${severityColor}; display: inline-block; flex-shrink: 0;"></span>
                <span class="cell-badge-name">${shortName}</span>
                <span class="cell-badge-prob" style="color: #facc15; font-weight: 700; font-family: ui-monospace, monospace; margin-left: 2px;">${probPct}%</span>
              </div>
            </div>
          `,
          iconSize: [96, 22],
          iconAnchor: [48, 11],
        });

        const badgeMarker = L.marker([cell.lat, cell.lon], { icon: badgeIcon });
        badgeMarker.on('click', () => onSelectCell(cell));
        badgeMarker.bindTooltip(popupContent, { sticky: true, opacity: 1.0 });
        badgeMarker.addTo(layer);
      }
    });
  }, [cells, overlayMode, horizon, showLabels, onSelectCell]);

  // =========================================================================
  // In-Place Smooth Update for Trajectory Motion Vectors Overlay
  // =========================================================================
  useEffect(() => {
    const trajGroup = trajectoryLayerGroupRef.current;
    if (!trajGroup || !mapInstanceRef.current) return;

    trajGroup.clearLayers();
    if (!showTrajectories || cells.length === 0) return;

    const vectors = computeCellTrajectories(cells);

    vectors.forEach((vec) => {
      const originCell = cells.find((c) => c.id.startsWith(vec.cellId.split('-h')[0]));
      const fullPathLatLngs: L.LatLngExpression[] = [];

      if (originCell) {
        fullPathLatLngs.push([originCell.lat, originCell.lon]);
      }
      vec.points.forEach((p) => fullPathLatLngs.push([p.lat, p.lon]));

      // 1. Subtle glowing trail backdrop
      L.polyline(fullPathLatLngs, {
        color: '#facc15',
        weight: 6,
        opacity: 0.25,
      }).addTo(trajGroup);

      // 2. Animated dash-flow polyline
      const polyline = L.polyline(fullPathLatLngs, {
        className: 'animated-trajectory-path',
        color: '#facc15',
        weight: 2.5,
        opacity: 0.95,
      });

      polyline.bindTooltip(
        `
          <div style="font-family: ui-sans-serif, system-ui; font-size: 11px; padding: 6px 8px; background-color: #090d16; color: #f8fafc; border: 1px solid rgba(148, 163, 184, 0.35); border-radius: 6px;">
            <div style="font-weight: 700; color: #facc15;">⚡ ${vec.cellName} Trajectory</div>
            <div style="margin: 2px 0;">Velocity: <strong>${vec.speedKmH} km/h</strong> towards <strong>${vec.directionLabel} (${vec.bearingDeg}°)</strong></div>
            <div style="font-size: 10px; color: #94a3b8;">Extrapolated Path: T+0 → +15m → +30m → +45m → +60m</div>
          </div>
        `,
        { sticky: true, opacity: 1.0 }
      );
      polyline.addTo(trajGroup);

      // 3. Milestone dots (+15m, +30m, +45m, +60m)
      vec.points.forEach((pt, ptIdx) => {
        const isCurrent = pt.step === horizon.step;
        const radius = isCurrent ? 6 : 4;
        const fillColor = isCurrent ? '#f43f5e' : ptIdx === 3 ? '#ec4899' : '#facc15';

        const dot = L.circleMarker([pt.lat, pt.lon], {
          radius: radius,
          color: '#ffffff',
          weight: 1.5,
          fillColor: fillColor,
          fillOpacity: 1.0,
        });

        dot.bindTooltip(
          `
            <div style="font-size: 10px; font-weight: 600; padding: 3px 6px; background-color: #090d16; color: #f8fafc; border: 1px solid rgba(148, 163, 184, 0.35); border-radius: 4px;">
              ${vec.cellName.split(' ')[0]} @ <strong>${pt.label}</strong>
              <div style="color: #94a3b8; font-size: 9px;">${pt.lat.toFixed(2)}°N, ${pt.lon.toFixed(2)}°E</div>
            </div>
          `,
          { direction: 'top', offset: [0, -6], opacity: 1.0 }
        );

        dot.addTo(trajGroup);
      });

      // 4. Directional arrowhead at terminal point (+60m)
      const lastPoint = vec.points[vec.points.length - 1];
      const arrowIcon = L.divIcon({
        className: 'motion-vector-arrow',
        html: `
          <div style="
            transform: rotate(${vec.bearingDeg - 90}deg);
            color: #facc15;
            font-size: 16px;
            line-height: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            text-shadow: 0 0 8px rgba(0,0,0,0.9), 0 0 4px #eab308;
            pointer-events: none;
          ">
            ➤
          </div>
        `,
        iconSize: [18, 18],
        iconAnchor: [9, 9],
      });

      L.marker([lastPoint.lat, lastPoint.lon], { icon: arrowIcon, interactive: false }).addTo(trajGroup);
    });
  }, [cells, showTrajectories, horizon]);

// ---------------------------------------------------------------------------
// Canonical IMD Doppler Weather Radar Network (11 Strategic Radar Stations)
// ---------------------------------------------------------------------------
const IMD_DWR_DEFAULT_STATIONS: SensorStation[] = [
  { id: 'DWR-MUM', name: 'Mumbai Colaba DWR', type: 'radar', lat: 18.8987, lon: 72.8094, status: 'online', rangeKm: 250, description: 'IMD S-band Radar - Arabian Sea Coastal & Western Ghats', band: 'S-band (2.7 GHz)' },
  { id: 'DWR-CHE', name: 'Chennai Port DWR', type: 'radar', lat: 13.0827, lon: 80.2987, status: 'online', rangeKm: 250, description: 'IMD S-band Radar - Coromandel Coast convective monitoring', band: 'S-band (2.8 GHz)' },
  { id: 'DWR-VSK', name: 'Visakhapatnam DWR', type: 'radar', lat: 17.6868, lon: 83.2185, status: 'online', rangeKm: 250, description: 'IMD S-band Radar - Andhra Coast & Bay of Bengal Cyclone Warning', band: 'S-band (2.8 GHz)' },
  { id: 'DWR-MPT', name: 'Machilipatnam DWR', type: 'radar', lat: 16.1875, lon: 81.1389, status: 'online', rangeKm: 250, description: 'IMD S-band Radar - Krishna Basin & Central Bay of Bengal Coast', band: 'S-band (2.7 GHz)' },
  { id: 'DWR-KOL', name: 'Kolkata DWR', type: 'radar', lat: 22.6547, lon: 88.4467, status: 'online', rangeKm: 250, description: 'IMD S-band Radar - Bay of Bengal Norwester corridor', band: 'S-band (2.8 GHz)' },
  { id: 'DWR-PDP', name: 'Paradip DWR', type: 'radar', lat: 20.3167, lon: 86.6111, status: 'online', rangeKm: 250, description: 'IMD S-band Radar - Odisha Coast Cyclone Surveillance', band: 'S-band (2.8 GHz)' },
  { id: 'DWR-HYD', name: 'Hyderabad Begumpet DWR', type: 'radar', lat: 17.4531, lon: 78.4677, status: 'online', rangeKm: 250, description: 'IMD C-band Radar - Deccan Plateau thunderstorm detection', band: 'C-band (5.6 GHz)' },
  { id: 'DWR-BLR', name: 'Bengaluru DWR', type: 'radar', lat: 12.9538, lon: 77.5831, status: 'online', rangeKm: 250, description: 'IMD C-band Radar - South Peninsula Interior Convection', band: 'C-band (5.6 GHz)' },
  { id: 'DWR-NGP', name: 'Nagpur DWR', type: 'radar', lat: 21.0922, lon: 79.0617, status: 'online', rangeKm: 250, description: 'IMD S-band Radar - Central India Meteorological Radar', band: 'S-band (2.8 GHz)' },
  { id: 'DWR-DEL', name: 'New Delhi Palam DWR', type: 'radar', lat: 28.5665, lon: 77.1031, status: 'online', rangeKm: 250, description: 'IMD S-band Doppler Weather Radar - National Capital Region', band: 'S-band (2.7 GHz)' },
  { id: 'DWR-GHY', name: 'Guwahati DWR', type: 'radar', lat: 26.1061, lon: 91.5859, status: 'online', rangeKm: 250, description: 'IMD S-band Radar - Brahmaputra Valley & Northeast Convection', band: 'S-band (2.7 GHz)' },
];

  // =========================================================================
  // In-Place Update for Stations & DWR 250km Radar Rings
  // =========================================================================
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !mapReady) return;

    // Ensure layer groups exist and are mounted directly to the Leaflet map
    if (!stationLayerGroupRef.current) {
      stationLayerGroupRef.current = L.layerGroup().addTo(map);
    } else if (!map.hasLayer(stationLayerGroupRef.current)) {
      stationLayerGroupRef.current.addTo(map);
    }

    if (!rangeRingsGroupRef.current) {
      rangeRingsGroupRef.current = L.layerGroup().addTo(map);
    } else if (!map.hasLayer(rangeRingsGroupRef.current)) {
      rangeRingsGroupRef.current.addTo(map);
    }

    const stGroup = stationLayerGroupRef.current;
    const ringGroup = rangeRingsGroupRef.current;

    stGroup.clearLayers();
    ringGroup.clearLayers();

    // Combine prop stations with default IMD DWR network
    const stationMap = new Map<string, SensorStation>();
    IMD_DWR_DEFAULT_STATIONS.forEach((st) => stationMap.set(st.id, st));
    (stations || []).forEach((st) => stationMap.set(st.id, st));
    const effectiveStations = Array.from(stationMap.values());

    // 1. Render DWR 250km Range Rings whenever showRangeRings is true
    if (showRangeRings) {
      effectiveStations.forEach((st) => {
        if (st.type === 'radar' && st.status === 'online') {
          L.circle([st.lat, st.lon], {
            radius: 250000,
            color: '#38bdf8',
            weight: 1.2,
            dashArray: '5 5',
            opacity: 0.55,
            fillColor: '#38bdf8',
            fillOpacity: 0.04,
          }).addTo(ringGroup);
        }
      });
    }

    // 2. Render Station Markers whenever showStations is true
    if (showStations) {
      effectiveStations.forEach((st) => {
        let iconColor = '#38bdf8';
        let iconSymbol = '📡';

        if (st.type === 'satellite') {
          iconColor = '#a855f7';
          iconSymbol = '🛰️';
        } else if (st.type === 'lightning') {
          iconColor = '#eab308';
          iconSymbol = '⚡';
        }

        const markerIcon = L.divIcon({
          className: 'sensor-station-icon',
          html: `
            <div style="
              width: 22px;
              height: 22px;
              border-radius: 50%;
              background: rgba(15, 23, 42, 0.9);
              border: 1.5px solid ${iconColor};
              display: flex;
              align-items: center;
              justify-content: center;
              font-size: 11px;
              box-shadow: 0 0 10px ${iconColor}55;
              cursor: pointer;
              backdrop-filter: blur(4px);
            ">
              ${iconSymbol}
            </div>
          `,
          iconSize: [22, 22],
          iconAnchor: [11, 11],
        });

        const marker = L.marker([st.lat, st.lon], { icon: markerIcon });
        marker.bindTooltip(
          `
            <div style="font-family: ui-sans-serif, system-ui; font-size: 11px; padding: 6px 8px; background-color: #090d16; color: #f8fafc; border: 1px solid rgba(148, 163, 184, 0.35); border-radius: 6px;">
              <div style="font-weight: 700; color: ${iconColor}; font-size: 12px; margin-bottom: 2px;">${st.name}</div>
              <div style="color: #94a3b8; font-size: 10px; margin-bottom: 4px;">${st.description}</div>
              <div style="display: flex; gap: 8px; font-size: 10px;">
                <span style="color: #22c55e; font-weight: 600;">STATUS: ${st.status.toUpperCase()}</span>
                ${st.band ? `<span style="color: #cbd5e1;">BAND: ${st.band}</span>` : ''}
                ${st.type === 'radar' ? `<span style="color: #38bdf8;">RANGE: ${st.rangeKm} km</span>` : ''}
              </div>
            </div>
          `,
          { sticky: true, opacity: 1.0 }
        );

        marker.addTo(stGroup);
      });
    }
  }, [stations, showStations, showRangeRings, mapReady]);

  // Escape key listener to close selected cluster inspector and popups
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && selectedCell) {
        mapInstanceRef.current?.closePopup();
        onSelectCell(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedCell, onSelectCell]);

  // =========================================================================
  // Explicit User Navigation Handlers (Only places setView is called)
  // =========================================================================
  const handleSelectRegion = (presetId: string) => {
    setActiveRegionPreset(presetId);
    const preset = SUB_REGION_PRESETS.find((p) => p.id === presetId);
    if (preset && mapInstanceRef.current) {
      mapInstanceRef.current.setView(preset.center, preset.zoom, { animate: true });
    }
  };

  const resetView = () => {
    handleSelectRegion('all_india');
    const isMobile = typeof window !== 'undefined' && window.innerWidth < 768;
    setShowStations(!isMobile);
    setShowRangeRings(true);
  };

  return (
    <div
      className="map-3d-container satellite-terrain-tiles"
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        overflow: 'hidden',
        borderRadius: 0,
      }}
    >
      {/* 3D Atmospheric Horizon Vignette */}
      {is3DMode && <div className="atmospheric-horizon-vignette" />}

      {/* The Map Viewport (with smooth 3D tilt perspective support) */}
      <div
        className={`map-3d-viewport ${is3DMode ? 'tilted' : 'flat'}`}
        style={{
          position: 'relative',
          width: '100%',
          height: '100%',
        }}
      >
        {/* Leaflet Map Container */}
        <div
          ref={mapContainerRef}
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            background: '#040810',
          }}
        />
      </div>

      {/* Unified Top Controls Overlay: Responsive on mobile (< md), compact and never overlaps */}
      <div
        className="absolute top-2 sm:top-3 left-2 sm:left-3 right-2 sm:right-3 z-[1000] flex flex-col gap-1.5 sm:gap-2 pointer-events-none"
        style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          right: '10px',
          zIndex: 1000,
          pointerEvents: 'none',
          display: 'flex',
          flexDirection: 'column',
          gap: '6px',
        }}
      >
        {/* Row 1: Mode Switcher (Left) + Layers Toggle (< md) / Layer Bar (>= md) */}
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '6px',
            width: '100%',
          }}
        >
          {/* Mode Switcher Pill */}
          <div
            ref={modeSwitcherRef}
            className="no-scrollbar"
            style={{
              pointerEvents: 'auto',
              background: 'rgba(15, 23, 42, 0.92)',
              backdropFilter: 'blur(10px)',
              WebkitBackdropFilter: 'blur(10px)',
              border: '1px solid rgba(51, 65, 85, 0.9)',
              borderRadius: '8px',
              padding: '3px 4px',
              display: 'flex',
              flexWrap: 'nowrap',
              overflowX: 'auto',
              alignItems: 'center',
              gap: '4px',
              boxShadow: '0 8px 24px rgba(0, 0, 0, 0.55)',
            }}
          >
            <button
              onClick={() => onOverlayModeChange('probability')}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '5px',
                padding: '4px 7px',
                fontSize: '11px',
                fontWeight: 600,
                borderRadius: '6px',
                border: 'none',
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                transition: 'all 0.2s',
                background: overlayMode === 'probability' ? '#0284c7' : 'transparent',
                color: overlayMode === 'probability' ? '#ffffff' : '#94a3b8',
              }}
            >
              <Activity size={13} />
              Prob %
            </button>

            <button
              onClick={() => onOverlayModeChange('flash_density')}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '5px',
                padding: '4px 7px',
                fontSize: '11px',
                fontWeight: 600,
                borderRadius: '6px',
                border: 'none',
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                transition: 'all 0.2s',
                background: overlayMode === 'flash_density' ? '#7c3aed' : 'transparent',
                color: overlayMode === 'flash_density' ? '#ffffff' : '#94a3b8',
              }}
            >
              <Zap size={13} />
              Flash
            </button>

            <button
              onClick={() => onOverlayModeChange('uncertainty')}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '5px',
                padding: '4px 7px',
                fontSize: '11px',
                fontWeight: 600,
                borderRadius: '6px',
                border: 'none',
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                transition: 'all 0.2s',
                background: overlayMode === 'uncertainty' ? '#db2777' : 'transparent',
                color: overlayMode === 'uncertainty' ? '#ffffff' : '#94a3b8',
              }}
            >
              <AlertTriangle size={13} />
              Uncertainty
            </button>
          </div>

          {/* Mobile Layers Toggle Button (< md) */}
          <button
            type="button"
            className="hidden-md-up"
            onClick={() => setMobileLayersOpen(!mobileLayersOpen)}
            style={{
              pointerEvents: 'auto',
              display: 'flex',
              alignItems: 'center',
              gap: '5px',
              padding: '5px 10px',
              fontSize: '11px',
              fontWeight: 600,
              borderRadius: '8px',
              background: mobileLayersOpen ? '#0284c7' : 'rgba(15, 23, 42, 0.92)',
              border: mobileLayersOpen ? '1px solid #38bdf8' : '1px solid rgba(51, 65, 85, 0.9)',
              color: '#ffffff',
              cursor: 'pointer',
              boxShadow: '0 8px 24px rgba(0, 0, 0, 0.55)',
            }}
          >
            <Layers size={13} color={mobileLayersOpen ? '#fff' : '#38bdf8'} />
            Layers
            <span style={{ fontSize: '9px', opacity: 0.8 }}>{mobileLayersOpen ? '▲' : '▼'}</span>
          </button>

          {/* Desktop Layer Toggle Bar (>= md) */}
          <div
            ref={layerControlsRef}
            className="hidden-md-down no-scrollbar"
            style={{
              pointerEvents: 'auto',
              background: 'rgba(15, 23, 42, 0.92)',
              backdropFilter: 'blur(10px)',
              WebkitBackdropFilter: 'blur(10px)',
              border: '1px solid rgba(51, 65, 85, 0.9)',
              borderRadius: '8px',
              padding: '4px 8px',
              display: 'flex',
              flexWrap: 'nowrap',
              overflowX: 'auto',
              alignItems: 'center',
              gap: '10px',
              fontSize: '12px',
              color: '#cbd5e1',
              boxShadow: '0 8px 24px rgba(0, 0, 0, 0.55)',
            }}
          >
            {/* 3D Viewport Tilt Toggle */}
            <button
              onClick={() => setIs3DMode(!is3DMode)}
              title="Toggle 3D Perspective Tilt"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                background: is3DMode ? '#0284c7' : '#1e293b',
                border: '1px solid ' + (is3DMode ? '#38bdf8' : '#475569'),
                borderRadius: '4px',
                padding: '3px 7px',
                color: '#f8fafc',
                cursor: 'pointer',
                fontSize: '12px',
                fontWeight: 600,
                whiteSpace: 'nowrap',
                transition: 'all 0.2s',
              }}
            >
              <Box size={13} color={is3DMode ? '#ffffff' : '#38bdf8'} />
              3D Tilt
            </button>

            {/* Wind Particles Streamlines Toggle */}
            <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', whiteSpace: 'nowrap' }}>
              <input
                type="checkbox"
                checked={showWindParticles}
                onChange={(e) => setShowWindParticles(e.target.checked)}
                style={{ accentColor: '#38bdf8' }}
              />
              <Wind size={13} color="#38bdf8" /> Wind Particles
            </label>

            {/* Organic Convective Heatmap Toggle */}
            <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', whiteSpace: 'nowrap' }}>
              <input
                type="checkbox"
                checked={showPlumes}
                onChange={(e) => setShowPlumes(e.target.checked)}
                style={{ accentColor: '#38bdf8' }}
              />
              <Sparkles size={13} color="#38bdf8" /> Heatmap
            </label>

            {/* Motion Vectors Toggle */}
            <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', whiteSpace: 'nowrap' }}>
              <input
                type="checkbox"
                checked={showTrajectories}
                onChange={(e) => setShowTrajectories(e.target.checked)}
                style={{ accentColor: '#facc15' }}
              />
              <TrendingUp size={13} color="#facc15" /> Motion Vectors
            </label>

            {/* Stations Toggle */}
            <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', whiteSpace: 'nowrap' }}>
              <input
                type="checkbox"
                checked={showStations}
                onChange={(e) => setShowStations(e.target.checked)}
                style={{ accentColor: '#38bdf8' }}
              />
              <Radio size={13} color="#38bdf8" /> Stations
            </label>

            {/* DWR 250km Range Rings */}
            <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', whiteSpace: 'nowrap' }}>
              <input
                type="checkbox"
                checked={showRangeRings}
                onChange={(e) => setShowRangeRings(e.target.checked)}
                style={{ accentColor: '#38bdf8' }}
              />
              <Layers size={13} color="#38bdf8" /> DWR 250km
            </label>

            {/* Floating Labels Toggle */}
            <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', whiteSpace: 'nowrap' }}>
              <input
                type="checkbox"
                checked={showLabels}
                onChange={(e) => setShowLabels(e.target.checked)}
                style={{ accentColor: '#facc15' }}
              />
              <Eye size={13} color="#facc15" /> Labels
            </label>

            {/* Reset Map Button */}
            <button
              onClick={resetView}
              title="Reset to Pan-India Overview (22.5°N, 79.0°E, Zoom 5)"
              style={{
                background: '#1e293b',
                border: '1px solid #475569',
                borderRadius: '4px',
                padding: '3px 8px',
                color: '#f8fafc',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                fontSize: '12px',
                whiteSpace: 'nowrap',
                transition: 'background 0.2s',
              }}
            >
              <Compass size={13} />
              Reset
            </button>
          </div>
        </div>

        {/* Mobile Collapsible Layer & Sub-Region Dropdown Menu (< md, when mobileLayersOpen is true) */}
        {mobileLayersOpen && (
          <div
            ref={mobileLayersRef}
            className="hidden-md-up no-scrollbar"
            style={{
              pointerEvents: 'auto',
              background: 'rgba(15, 23, 42, 0.96)',
              backdropFilter: 'blur(14px)',
              WebkitBackdropFilter: 'blur(14px)',
              border: '1px solid rgba(56, 189, 248, 0.4)',
              borderRadius: '8px',
              padding: '8px 10px',
              display: 'flex',
              flexDirection: 'column',
              gap: '8px',
              boxShadow: '0 8px 24px rgba(0, 0, 0, 0.8)',
              width: '100%',
              maxHeight: '260px',
              overflowY: 'auto',
            }}
          >
            {/* Mobile Sub-Region Selector Row */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px', borderBottom: '1px solid #1e293b', paddingBottom: '6px' }}>
              <span style={{ color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '4px', fontWeight: 600, fontSize: '11px' }}>
                <MapPin size={12} color="#38bdf8" /> Sub-Region:
              </span>
              <select
                value={activeRegionPreset}
                onChange={(e) => handleSelectRegion(e.target.value)}
                style={{
                  background: '#1e293b',
                  border: '1px solid #334155',
                  color: '#f8fafc',
                  borderRadius: '4px',
                  padding: '2px 6px',
                  fontSize: '11px',
                  cursor: 'pointer',
                  outline: 'none',
                }}
              >
                {SUB_REGION_PRESETS.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name} {p.id === 'all_india' ? '[Default]' : ''}
                  </option>
                ))}
              </select>
            </div>

            {/* Mobile Layer Toggles Ribbon (horizontal scroll) */}
            <div
              className="no-scrollbar"
              style={{
                display: 'flex',
                flexWrap: 'nowrap',
                overflowX: 'auto',
                alignItems: 'center',
                gap: '10px',
                fontSize: '11px',
                color: '#cbd5e1',
                padding: '2px 0',
                WebkitOverflowScrolling: 'touch',
              }}
            >
              {/* 3D Viewport Tilt Toggle */}
              <button
                onClick={() => setIs3DMode(!is3DMode)}
                title="Toggle 3D Perspective Tilt"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  background: is3DMode ? '#0284c7' : '#1e293b',
                  border: '1px solid ' + (is3DMode ? '#38bdf8' : '#475569'),
                  borderRadius: '4px',
                  padding: '3px 7px',
                  color: '#f8fafc',
                  cursor: 'pointer',
                  fontSize: '11px',
                  fontWeight: 600,
                  whiteSpace: 'nowrap',
                  transition: 'all 0.2s',
                }}
              >
                <Box size={12} color={is3DMode ? '#ffffff' : '#38bdf8'} />
                3D Tilt
              </button>

              {/* Wind Particles Streamlines Toggle */}
              <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', whiteSpace: 'nowrap' }}>
                <input
                  type="checkbox"
                  checked={showWindParticles}
                  onChange={(e) => setShowWindParticles(e.target.checked)}
                  style={{ accentColor: '#38bdf8' }}
                />
                <Wind size={12} color="#38bdf8" /> Wind Particles
              </label>

              {/* Organic Convective Heatmap Toggle */}
              <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', whiteSpace: 'nowrap' }}>
                <input
                  type="checkbox"
                  checked={showPlumes}
                  onChange={(e) => setShowPlumes(e.target.checked)}
                  style={{ accentColor: '#38bdf8' }}
                />
                <Sparkles size={12} color="#38bdf8" /> Heatmap
              </label>

              {/* Motion Vectors Toggle */}
              <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', whiteSpace: 'nowrap' }}>
                <input
                  type="checkbox"
                  checked={showTrajectories}
                  onChange={(e) => setShowTrajectories(e.target.checked)}
                  style={{ accentColor: '#facc15' }}
                />
                <TrendingUp size={12} color="#facc15" /> Motion Vectors
              </label>

              {/* Stations Toggle */}
              <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', whiteSpace: 'nowrap' }}>
                <input
                  type="checkbox"
                  checked={showStations}
                  onChange={(e) => setShowStations(e.target.checked)}
                  style={{ accentColor: '#38bdf8' }}
                />
                <Radio size={12} color="#38bdf8" /> Stations
              </label>

              {/* DWR 250km Range Rings */}
              <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', whiteSpace: 'nowrap' }}>
                <input
                  type="checkbox"
                  checked={showRangeRings}
                  onChange={(e) => setShowRangeRings(e.target.checked)}
                  style={{ accentColor: '#38bdf8' }}
                />
                <Layers size={12} color="#38bdf8" /> DWR 250km
              </label>

              {/* Floating Labels Toggle */}
              <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', whiteSpace: 'nowrap' }}>
                <input
                  type="checkbox"
                  checked={showLabels}
                  onChange={(e) => setShowLabels(e.target.checked)}
                  style={{ accentColor: '#facc15' }}
                />
                <Eye size={12} color="#facc15" /> Labels
              </label>

              {/* Reset Map Button */}
              <button
                onClick={resetView}
                title="Reset to Pan-India Overview (22.5°N, 79.0°E, Zoom 5)"
                style={{
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: '4px',
                  padding: '3px 8px',
                  color: '#f8fafc',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  fontSize: '11px',
                  whiteSpace: 'nowrap',
                  transition: 'background 0.2s',
                }}
              >
                <Compass size={12} />
                Reset
              </button>
            </div>

            {/* Mobile 1-line Colorbar Legend */}
            <div style={{ borderTop: '1px solid #1e293b', paddingTop: '6px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#94a3b8', marginBottom: '3px' }}>
                <span>{overlayMode === 'probability' ? 'Occurrence Probability' : overlayMode === 'flash_density' ? 'Flash Density' : 'Uncertainty'}</span>
                <span style={{ color: '#38bdf8', fontWeight: 700 }}>{horizon.label}</span>
              </div>
              <div
                style={{
                  height: '6px',
                  borderRadius: '3px',
                  background: overlayMode === 'uncertainty'
                    ? 'linear-gradient(to right, #3b4ca8, #1b6b88, #d97706, #be185d)'
                    : 'linear-gradient(to right, #1b6b88, #3b4ca8, #7e3af2, #be185d)',
                }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#64748b', fontSize: '9px', marginTop: '2px' }}>
                <span>{overlayMode === 'probability' ? 'Ambient' : overlayMode === 'flash_density' ? '0' : 'Low'}</span>
                <span>{overlayMode === 'probability' ? 'Moderate' : overlayMode === 'flash_density' ? '8' : 'Mod'}</span>
                <span>{overlayMode === 'probability' ? 'Convective' : overlayMode === 'flash_density' ? '18' : 'Elevated'}</span>
                <span>{overlayMode === 'probability' ? 'Severe Core' : overlayMode === 'flash_density' ? '32+' : 'High'}</span>
              </div>
            </div>
          </div>
        )}

        {/* Row 2: Sub-Region Selector cleanly below on desktop (hidden on mobile) */}
        <div
          ref={subRegionRef}
          className="hidden-md-down pointer-events-auto mt-1 py-1 px-2 text-xs"
          style={{
            pointerEvents: 'auto',
            background: 'rgba(15, 23, 42, 0.90)',
            backdropFilter: 'blur(10px)',
            WebkitBackdropFilter: 'blur(10px)',
            border: '1px solid rgba(51, 65, 85, 0.9)',
            borderRadius: '8px',
            padding: '4px 8px',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '6px',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.4)',
            fontSize: '12px',
            width: 'fit-content',
          }}
        >
          <span style={{ color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '4px', fontWeight: 600 }}>
            <MapPin size={12} color="#38bdf8" /> Sub-Region:
          </span>
          <select
            value={activeRegionPreset}
            onChange={(e) => handleSelectRegion(e.target.value)}
            style={{
              background: '#1e293b',
              border: '1px solid #334155',
              color: '#f8fafc',
              borderRadius: '4px',
              padding: '3px 8px',
              fontSize: '11px',
              cursor: 'pointer',
              outline: 'none',
            }}
          >
            {SUB_REGION_PRESETS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} {p.id === 'all_india' ? '[Default]' : ''}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Bottom Left: Windy-Style Dynamic Colorbar Legend (Hidden on mobile < md) */}
      <div
        ref={legendRef}
        className="hidden-md-down"
        style={{
          position: 'absolute',
          bottom: '24px',
          left: '14px',
          zIndex: 1000,
          background: 'rgba(15, 23, 42, 0.94)',
          backdropFilter: 'blur(10px)',
          WebkitBackdropFilter: 'blur(10px)',
          border: '1px solid rgba(51, 65, 85, 0.9)',
          borderRadius: '8px',
          padding: '10px 14px',
          minWidth: '280px',
          boxShadow: '0 8px 24px rgba(0, 0, 0, 0.6)',
          fontSize: '11px',
        }}
      >
        <div style={{ fontWeight: 600, color: '#e2e8f0', marginBottom: '6px', display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
          <span>
            {overlayMode === 'probability' && 'Occurrence Probability'}
            {overlayMode === 'flash_density' && 'Flash Density (fl/km²/15m)'}
            {overlayMode === 'uncertainty' && 'Total Predictive Uncertainty'}
          </span>
          <span style={{ color: '#38bdf8', fontWeight: 700 }}>{horizon.label}</span>
        </div>

        {overlayMode === 'probability' && (
          <div>
            <div
              style={{
                height: '8px',
                borderRadius: '4px',
                background:
                  'linear-gradient(to right, #1b6b88, #3b4ca8, #7e3af2, #be185d)',
              }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', color: '#94a3b8', marginTop: '4px', fontSize: '10px', whiteSpace: 'nowrap' }}>
              <span>Ambient Ocean</span>
              <span>Moderate</span>
              <span>Convective</span>
              <span>Severe Core</span>
            </div>
          </div>
        )}

        {overlayMode === 'flash_density' && (
          <div>
            <div
              style={{
                height: '8px',
                borderRadius: '4px',
                background:
                  'linear-gradient(to right, #1b6b88, #3b4ca8, #7e3af2, #be185d)',
              }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', color: '#94a3b8', marginTop: '4px', fontSize: '10px', whiteSpace: 'nowrap' }}>
              <span>0</span>
              <span>8</span>
              <span>18</span>
              <span>32+ fl/km²</span>
            </div>
          </div>
        )}

        {overlayMode === 'uncertainty' && (
          <div>
            <div
              style={{
                height: '8px',
                borderRadius: '4px',
                background: 'linear-gradient(to right, #3b4ca8, #1b6b88, #d97706, #be185d)',
              }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px', color: '#94a3b8', marginTop: '4px', fontSize: '10px', whiteSpace: 'nowrap' }}>
              <span>Low (±15%)</span>
              <span>Mod (±45%)</span>
              <span>Elevated (±70%)</span>
              <span>High (±90%+)</span>
            </div>
            {uncertaintyInflation > 1.2 && (
              <div style={{ marginTop: '6px', color: '#f43f5e', fontSize: '10px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <AlertTriangle size={12} />
                <span>Sensors missing: Epistemic variance inflated {uncertaintyInflation}x</span>
              </div>
            )}
          </div>
        )}

        {showWindParticles && (
          <div style={{ marginTop: '8px', paddingTop: '6px', borderTop: '1px solid #334155', color: '#38bdf8', fontSize: '10px', display: 'flex', alignItems: 'center', gap: '5px' }}>
            <Wind size={12} color="#38bdf8" />
            <span>1,400 Wind Streamline Wisps Active</span>
          </div>
        )}

        {showTrajectories && (
          <div style={{ marginTop: '4px', color: '#facc15', fontSize: '10px', display: 'flex', alignItems: 'center', gap: '5px' }}>
            <TrendingUp size={12} color="#facc15" />
            <span>Extrapolated Storm Vectors (T+0 → +60m)</span>
          </div>
        )}
      </div>
    </div>
  );
};
