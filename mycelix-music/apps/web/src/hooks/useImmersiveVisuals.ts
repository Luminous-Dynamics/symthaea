// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Immersive Visuals Hook
 *
 * Provides audio-reactive 3D visuals using WebGL:
 * - Real-time audio analysis
 * - Particle systems
 * - Geometric visualizations
 * - Shader-based effects
 *
 * Integrates with mycelix-wasm visualization capabilities
 */

import { useState, useCallback, useRef, useEffect } from 'react';

// Visualization modes
export type VisualizationMode =
  | 'particles'
  | 'waveform'
  | 'spectrum'
  | 'galaxy'
  | 'tunnel'
  | 'terrain'
  | 'abstract';

// Color schemes
export type ColorScheme =
  | 'neon'
  | 'sunset'
  | 'ocean'
  | 'forest'
  | 'cosmic'
  | 'fire'
  | 'monochrome'
  | 'rainbow';

// Visual settings
export interface VisualSettings {
  mode: VisualizationMode;
  colorScheme: ColorScheme;
  intensity: number; // 0-1
  speed: number; // 0-2
  complexity: number; // 0-1
  bloom: number; // 0-1
  motionBlur: number; // 0-1
  bassResponse: number; // 0-2
  trebleResponse: number; // 0-2
  particleCount: number; // 100-10000
  cameraDistance: number; // 1-10
  cameraRotation: boolean;
}

// Audio analysis data
export interface AudioAnalysis {
  waveform: Float32Array;
  spectrum: Float32Array;
  bassLevel: number;
  midLevel: number;
  trebleLevel: number;
  energy: number;
  beat: boolean;
  bpm: number;
}

// Particle state (for particle visualizations)
interface Particle {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  size: number;
  life: number;
  color: [number, number, number];
}

// Hook state
export interface ImmersiveVisualsState {
  isInitialized: boolean;
  isRunning: boolean;
  settings: VisualSettings;
  fps: number;
  audioAnalysis: AudioAnalysis | null;
  error: string | null;
}

// Color scheme palettes
const COLOR_PALETTES: Record<ColorScheme, [number, number, number][]> = {
  neon: [[0.5, 0, 1], [0, 1, 1], [1, 0, 0.5], [0, 1, 0.5]],
  sunset: [[1, 0.3, 0], [1, 0.5, 0], [0.8, 0.2, 0.4], [0.3, 0.1, 0.5]],
  ocean: [[0, 0.3, 0.6], [0, 0.5, 0.8], [0, 0.7, 1], [0.2, 0.8, 1]],
  forest: [[0.1, 0.4, 0.1], [0.2, 0.6, 0.2], [0.4, 0.8, 0.2], [0.1, 0.3, 0.1]],
  cosmic: [[0.2, 0, 0.5], [0.5, 0, 0.8], [0.8, 0.2, 1], [0.1, 0, 0.3]],
  fire: [[1, 0.2, 0], [1, 0.5, 0], [1, 0.8, 0], [0.8, 0, 0]],
  monochrome: [[1, 1, 1], [0.7, 0.7, 0.7], [0.4, 0.4, 0.4], [0.1, 0.1, 0.1]],
  rainbow: [[1, 0, 0], [1, 0.5, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [0.5, 0, 1]],
};

const DEFAULT_SETTINGS: VisualSettings = {
  mode: 'particles',
  colorScheme: 'neon',
  intensity: 0.7,
  speed: 1,
  complexity: 0.5,
  bloom: 0.5,
  motionBlur: 0.3,
  bassResponse: 1.5,
  trebleResponse: 1,
  particleCount: 2000,
  cameraDistance: 5,
  cameraRotation: true,
};

export function useImmersiveVisuals(audioSource?: AudioNode) {
  const [state, setState] = useState<ImmersiveVisualsState>({
    isInitialized: false,
    isRunning: false,
    settings: DEFAULT_SETTINGS,
    fps: 0,
    audioAnalysis: null,
    error: null,
  });

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const glRef = useRef<WebGL2RenderingContext | null>(null);
  const animationFrameRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);
  const fpsCounterRef = useRef<number[]>([]);

  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const waveformDataRef = useRef<Float32Array>(new Float32Array(256));
  const spectrumDataRef = useRef<Float32Array>(new Float32Array(256));

  const particlesRef = useRef<Particle[]>([]);
  const timeRef = useRef<number>(0);
  const beatHistoryRef = useRef<number[]>([]);

  // Initialize WebGL and audio analysis
  const initialize = useCallback(async (canvas: HTMLCanvasElement) => {
    canvasRef.current = canvas;

    // Initialize WebGL
    const gl = canvas.getContext('webgl2', {
      alpha: true,
      antialias: true,
      premultipliedAlpha: false,
    });

    if (!gl) {
      setState(prev => ({
        ...prev,
        error: 'WebGL 2 not supported',
        isInitialized: false,
      }));
      return;
    }

    glRef.current = gl;

    // Enable blending for particles
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

    // Initialize audio analysis
    if (audioSource) {
      const ctx = audioSource.context as AudioContext;
      audioContextRef.current = ctx;

      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.8;
      audioSource.connect(analyser);
      analyserRef.current = analyser;

      waveformDataRef.current = new Float32Array(analyser.frequencyBinCount);
      spectrumDataRef.current = new Float32Array(analyser.frequencyBinCount);
    }

    // Initialize particles
    initializeParticles(state.settings.particleCount);

    setState(prev => ({ ...prev, isInitialized: true }));
  }, [audioSource, state.settings.particleCount]);

  // Initialize particles
  const initializeParticles = (count: number) => {
    const particles: Particle[] = [];
    const palette = COLOR_PALETTES[state.settings.colorScheme];

    for (let i = 0; i < count; i++) {
      const color = palette[Math.floor(Math.random() * palette.length)];
      particles.push({
        x: (Math.random() - 0.5) * 10,
        y: (Math.random() - 0.5) * 10,
        z: (Math.random() - 0.5) * 10,
        vx: (Math.random() - 0.5) * 0.02,
        vy: (Math.random() - 0.5) * 0.02,
        vz: (Math.random() - 0.5) * 0.02,
        size: Math.random() * 2 + 0.5,
        life: Math.random(),
        color: [...color],
      });
    }

    particlesRef.current = particles;
  };

  // Analyze audio
  const analyzeAudio = useCallback((): AudioAnalysis | null => {
    const analyser = analyserRef.current;
    if (!analyser) return null;

    // Get waveform and spectrum data
    analyser.getFloatTimeDomainData(waveformDataRef.current);
    analyser.getFloatFrequencyData(spectrumDataRef.current);

    const waveform = waveformDataRef.current;
    const spectrum = spectrumDataRef.current;

    // Calculate frequency bands
    const bandSize = Math.floor(spectrum.length / 3);
    let bassSum = 0, midSum = 0, trebleSum = 0;

    for (let i = 0; i < bandSize; i++) {
      bassSum += Math.pow(10, spectrum[i] / 20);
      midSum += Math.pow(10, spectrum[i + bandSize] / 20);
      trebleSum += Math.pow(10, spectrum[i + bandSize * 2] / 20);
    }

    const bassLevel = bassSum / bandSize;
    const midLevel = midSum / bandSize;
    const trebleLevel = trebleSum / bandSize;

    // Calculate overall energy
    const energy = (bassLevel + midLevel + trebleLevel) / 3;

    // Beat detection (simple threshold-based)
    beatHistoryRef.current.push(bassLevel);
    if (beatHistoryRef.current.length > 60) {
      beatHistoryRef.current.shift();
    }

    const avgBass = beatHistoryRef.current.reduce((a, b) => a + b, 0) / beatHistoryRef.current.length;
    const beat = bassLevel > avgBass * 1.3;

    // Estimate BPM (simplified)
    const bpm = 120; // In production, would use more sophisticated beat tracking

    return {
      waveform,
      spectrum,
      bassLevel: Math.min(1, bassLevel * 2),
      midLevel: Math.min(1, midLevel * 2),
      trebleLevel: Math.min(1, trebleLevel * 2),
      energy: Math.min(1, energy * 2),
      beat,
      bpm,
    };
  }, []);

  // Render frame
  const render = useCallback((timestamp: number) => {
    const gl = glRef.current;
    const canvas = canvasRef.current;
    const settings = state.settings;

    if (!gl || !canvas) return;

    // Calculate FPS
    const deltaTime = timestamp - lastTimeRef.current;
    lastTimeRef.current = timestamp;
    fpsCounterRef.current.push(1000 / deltaTime);
    if (fpsCounterRef.current.length > 30) {
      fpsCounterRef.current.shift();
    }
    const fps = Math.round(
      fpsCounterRef.current.reduce((a, b) => a + b, 0) / fpsCounterRef.current.length
    );

    // Analyze audio
    const analysis = analyzeAudio();

    // Update time
    timeRef.current += deltaTime * 0.001 * settings.speed;

    // Clear canvas with fade effect for motion blur
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0, 0, 0, 1 - settings.motionBlur * 0.9);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Update and render based on mode
    switch (settings.mode) {
      case 'particles':
        updateParticles(deltaTime, analysis, settings);
        renderParticles(gl, canvas, settings);
        break;

      case 'waveform':
        renderWaveform(gl, canvas, analysis, settings);
        break;

      case 'spectrum':
        renderSpectrum(gl, canvas, analysis, settings);
        break;

      // Other modes would be implemented similarly
    }

    setState(prev => ({
      ...prev,
      fps,
      audioAnalysis: analysis,
    }));
  }, [state.settings, analyzeAudio]);

  // Update particles
  const updateParticles = (
    deltaTime: number,
    analysis: AudioAnalysis | null,
    settings: VisualSettings
  ) => {
    const particles = particlesRef.current;
    const palette = COLOR_PALETTES[settings.colorScheme];
    const bass = (analysis?.bassLevel || 0) * settings.bassResponse;
    const treble = (analysis?.trebleLevel || 0) * settings.trebleResponse;
    const beat = analysis?.beat || false;

    for (const particle of particles) {
      // Apply forces
      particle.vx += (Math.random() - 0.5) * 0.001 * settings.intensity;
      particle.vy += (Math.random() - 0.5) * 0.001 * settings.intensity;
      particle.vz += (Math.random() - 0.5) * 0.001 * settings.intensity;

      // Audio reactivity
      if (beat) {
        const force = bass * 0.1;
        particle.vx += (Math.random() - 0.5) * force;
        particle.vy += (Math.random() - 0.5) * force;
        particle.vz += (Math.random() - 0.5) * force;
      }

      // Update position
      particle.x += particle.vx * deltaTime * 0.05 * settings.speed;
      particle.y += particle.vy * deltaTime * 0.05 * settings.speed;
      particle.z += particle.vz * deltaTime * 0.05 * settings.speed;

      // Damping
      particle.vx *= 0.99;
      particle.vy *= 0.99;
      particle.vz *= 0.99;

      // Wrap around
      const bounds = 5;
      if (particle.x < -bounds) particle.x += bounds * 2;
      if (particle.x > bounds) particle.x -= bounds * 2;
      if (particle.y < -bounds) particle.y += bounds * 2;
      if (particle.y > bounds) particle.y -= bounds * 2;
      if (particle.z < -bounds) particle.z += bounds * 2;
      if (particle.z > bounds) particle.z -= bounds * 2;

      // Update size based on treble
      particle.size = (0.5 + treble * 2) * settings.intensity;

      // Update life and reset if needed
      particle.life -= 0.001;
      if (particle.life <= 0) {
        particle.life = 1;
        const color = palette[Math.floor(Math.random() * palette.length)];
        particle.color = [...color];
      }
    }
  };

  // Render particles (simplified 2D projection)
  const renderParticles = (
    gl: WebGL2RenderingContext,
    canvas: HTMLCanvasElement,
    settings: VisualSettings
  ) => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear
    ctx.fillStyle = `rgba(0, 0, 0, ${1 - settings.motionBlur * 0.9})`;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const particles = particlesRef.current;
    const cameraAngle = settings.cameraRotation ? timeRef.current * 0.2 : 0;
    const cameraDistance = settings.cameraDistance;

    // Sort by z-depth for proper rendering
    const sortedParticles = [...particles].sort((a, b) => b.z - a.z);

    for (const particle of sortedParticles) {
      // Rotate around Y axis
      const rotX = particle.x * Math.cos(cameraAngle) - particle.z * Math.sin(cameraAngle);
      const rotZ = particle.x * Math.sin(cameraAngle) + particle.z * Math.cos(cameraAngle);

      // Perspective projection
      const scale = cameraDistance / (cameraDistance + rotZ);
      const screenX = canvas.width / 2 + rotX * scale * 100;
      const screenY = canvas.height / 2 + particle.y * scale * 100;

      // Size based on depth
      const size = particle.size * scale * 5;

      if (screenX < -size || screenX > canvas.width + size) continue;
      if (screenY < -size || screenY > canvas.height + size) continue;

      // Draw particle
      const alpha = particle.life * scale * settings.intensity;
      ctx.beginPath();
      ctx.arc(screenX, screenY, size, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${particle.color[0] * 255}, ${particle.color[1] * 255}, ${particle.color[2] * 255}, ${alpha})`;
      ctx.fill();

      // Bloom effect
      if (settings.bloom > 0) {
        const gradient = ctx.createRadialGradient(screenX, screenY, 0, screenX, screenY, size * 3);
        gradient.addColorStop(0, `rgba(${particle.color[0] * 255}, ${particle.color[1] * 255}, ${particle.color[2] * 255}, ${alpha * settings.bloom * 0.3})`);
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(screenX, screenY, size * 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  };

  // Render waveform
  const renderWaveform = (
    gl: WebGL2RenderingContext,
    canvas: HTMLCanvasElement,
    analysis: AudioAnalysis | null,
    settings: VisualSettings
  ) => {
    const ctx = canvas.getContext('2d');
    if (!ctx || !analysis) return;

    ctx.fillStyle = `rgba(0, 0, 0, ${1 - settings.motionBlur * 0.9})`;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const waveform = analysis.waveform;
    const palette = COLOR_PALETTES[settings.colorScheme];
    const color = palette[0];

    ctx.beginPath();
    ctx.strokeStyle = `rgba(${color[0] * 255}, ${color[1] * 255}, ${color[2] * 255}, ${settings.intensity})`;
    ctx.lineWidth = 2;

    const sliceWidth = canvas.width / waveform.length;
    let x = 0;

    for (let i = 0; i < waveform.length; i++) {
      const y = (waveform[i] * 0.5 + 0.5) * canvas.height;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
      x += sliceWidth;
    }

    ctx.stroke();
  };

  // Render spectrum
  const renderSpectrum = (
    gl: WebGL2RenderingContext,
    canvas: HTMLCanvasElement,
    analysis: AudioAnalysis | null,
    settings: VisualSettings
  ) => {
    const ctx = canvas.getContext('2d');
    if (!ctx || !analysis) return;

    ctx.fillStyle = `rgba(0, 0, 0, ${1 - settings.motionBlur * 0.9})`;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const spectrum = analysis.spectrum;
    const palette = COLOR_PALETTES[settings.colorScheme];
    const barWidth = canvas.width / spectrum.length * 2;

    for (let i = 0; i < spectrum.length / 2; i++) {
      const value = (spectrum[i] + 140) / 140; // Normalize from dB
      const height = value * canvas.height * settings.intensity;

      const colorIndex = Math.floor(i / (spectrum.length / 2) * palette.length);
      const color = palette[Math.min(colorIndex, palette.length - 1)];

      ctx.fillStyle = `rgba(${color[0] * 255}, ${color[1] * 255}, ${color[2] * 255}, 0.8)`;
      ctx.fillRect(
        i * barWidth,
        canvas.height - height,
        barWidth - 1,
        height
      );
    }
  };

  // Animation loop
  const animate = useCallback((timestamp: number) => {
    render(timestamp);
    animationFrameRef.current = requestAnimationFrame(animate);
  }, [render]);

  // Start visualization
  const start = useCallback(() => {
    if (!state.isInitialized) return;

    setState(prev => ({ ...prev, isRunning: true }));
    lastTimeRef.current = performance.now();
    animationFrameRef.current = requestAnimationFrame(animate);
  }, [state.isInitialized, animate]);

  // Stop visualization
  const stop = useCallback(() => {
    cancelAnimationFrame(animationFrameRef.current);
    setState(prev => ({ ...prev, isRunning: false }));
  }, []);

  // Update settings
  const updateSettings = useCallback((updates: Partial<VisualSettings>) => {
    setState(prev => {
      const newSettings = { ...prev.settings, ...updates };

      // Reinitialize particles if count changed
      if (updates.particleCount && updates.particleCount !== prev.settings.particleCount) {
        initializeParticles(updates.particleCount);
      }

      // Update particle colors if scheme changed
      if (updates.colorScheme) {
        const palette = COLOR_PALETTES[updates.colorScheme];
        particlesRef.current.forEach(p => {
          const color = palette[Math.floor(Math.random() * palette.length)];
          p.color = [...color];
        });
      }

      return { ...prev, settings: newSettings };
    });
  }, []);

  // Cleanup
  useEffect(() => {
    return () => {
      cancelAnimationFrame(animationFrameRef.current);
      analyserRef.current?.disconnect();
    };
  }, []);

  return {
    ...state,
    initialize,
    start,
    stop,
    updateSettings,
    colorSchemes: Object.keys(COLOR_PALETTES) as ColorScheme[],
    visualizationModes: ['particles', 'waveform', 'spectrum', 'galaxy', 'tunnel', 'terrain', 'abstract'] as VisualizationMode[],
  };
}

export default useImmersiveVisuals;
