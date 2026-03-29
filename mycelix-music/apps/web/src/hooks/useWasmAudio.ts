// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * useWasmAudio Hook
 *
 * React hook for loading and using the Mycelix WASM audio module.
 * Provides waveform generation, audio analysis, and real-time visualizations.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

// Types matching the WASM exports
export interface WaveformData {
  peaks: Float32Array;
  rms: Float32Array;
  duration_ms: number;
  sample_rate: number;
  channels: number;
}

export interface AudioAnalysisResult {
  loudness_lufs: number;
  peak_db: number;
  dynamic_range: number;
  estimated_bpm: number | null;
  spectral_centroid: number;
  zero_crossing_rate: number;
}

export type VisualizationType =
  | 'Waveform'
  | 'FrequencyBars'
  | 'FrequencyLine'
  | 'Circular'
  | 'Spectrogram';

export interface VisualizationConfig {
  viz_type: VisualizationType;
  fft_size: number;
  smoothing: number;
  bar_width: number;
  bar_gap: number;
  mirror: boolean;
  gradient: boolean;
}

// WASM module interface (dynamically loaded)
interface WasmModule {
  WaveformGenerator: new (targetPeaks: number) => WasmWaveformGenerator;
  AudioAnalyzer: new (fftSize: number) => WasmAudioAnalyzer;
  RealtimeVisualizer: new (config: any) => WasmVisualizer;
  AudioBufferUtils: {
    normalize: (data: Float32Array, targetPeak: number) => void;
    fade_in: (data: Float32Array, fadeSamples: number) => void;
    fade_out: (data: Float32Array, fadeSamples: number) => void;
    compute_rms: (data: Float32Array) => number;
    compute_peak: (data: Float32Array) => number;
  };
  db_to_linear: (db: number) => number;
  linear_to_db: (linear: number) => number;
  frequency_to_midi: (freq: number) => number;
  midi_to_frequency: (midi: number) => number;
  bpm_to_ms: (bpm: number) => number;
  samples_to_time: (samples: number, sampleRate: number) => number;
  time_to_samples: (time: number, sampleRate: number) => number;
}

interface WasmWaveformGenerator {
  generate: (audioData: Float32Array, sampleRate: number, channels: number) => WaveformData;
  generate_from_audio_buffer: (buffer: AudioBuffer) => WaveformData;
}

interface WasmAudioAnalyzer {
  analyze: (audioData: Float32Array, sampleRate: number) => AudioAnalysisResult;
  analyze_audio_buffer: (buffer: AudioBuffer) => AudioAnalysisResult;
}

interface WasmVisualizer {
  set_colors: (primary: string, secondary: string, background: string) => void;
  draw_waveform: (ctx: CanvasRenderingContext2D, data: Float32Array, width: number, height: number) => void;
  draw_frequency_bars: (ctx: CanvasRenderingContext2D, data: Uint8Array, width: number, height: number) => void;
  draw_circular: (ctx: CanvasRenderingContext2D, data: Uint8Array, width: number, height: number) => void;
  draw_spectrogram: (ctx: CanvasRenderingContext2D, data: Uint8Array, width: number, height: number) => void;
  draw: (ctx: CanvasRenderingContext2D, analyser: AnalyserNode) => void;
}

// Module state
let wasmModule: WasmModule | null = null;
let loadingPromise: Promise<WasmModule> | null = null;

/**
 * Load the WASM module (lazy, singleton)
 */
async function loadWasmModule(): Promise<WasmModule> {
  if (wasmModule) return wasmModule;

  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    try {
      // Dynamic import of the WASM package
      const wasm = await import('@mycelix/wasm');
      await wasm.default(); // Initialize WASM
      wasmModule = wasm as unknown as WasmModule;
      return wasmModule;
    } catch (error) {
      console.error('Failed to load WASM module:', error);
      throw error;
    }
  })();

  return loadingPromise;
}

export interface UseWasmAudioReturn {
  isLoaded: boolean;
  isLoading: boolean;
  error: Error | null;

  // Waveform generation
  generateWaveform: (audioBuffer: AudioBuffer, targetPeaks?: number) => WaveformData | null;
  generateWaveformFromSamples: (samples: Float32Array, sampleRate: number, channels: number, targetPeaks?: number) => WaveformData | null;

  // Audio analysis
  analyzeAudio: (audioBuffer: AudioBuffer) => AudioAnalysisResult | null;
  analyzeAudioSamples: (samples: Float32Array, sampleRate: number) => AudioAnalysisResult | null;

  // Utilities
  utils: {
    dbToLinear: (db: number) => number;
    linearToDb: (linear: number) => number;
    frequencyToMidi: (freq: number) => number;
    midiToFrequency: (midi: number) => number;
    bpmToMs: (bpm: number) => number;
    computeRms: (data: Float32Array) => number;
    computePeak: (data: Float32Array) => number;
  } | null;
}

/**
 * Hook for using WASM audio processing
 */
export function useWasmAudio(): UseWasmAudioReturn {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const moduleRef = useRef<WasmModule | null>(null);

  // Load WASM module on mount
  useEffect(() => {
    if (moduleRef.current) return;

    setIsLoading(true);
    loadWasmModule()
      .then((mod) => {
        moduleRef.current = mod;
        setIsLoaded(true);
        setIsLoading(false);
      })
      .catch((err) => {
        setError(err);
        setIsLoading(false);
      });
  }, []);

  // Waveform generation from AudioBuffer
  const generateWaveform = useCallback((audioBuffer: AudioBuffer, targetPeaks: number = 800): WaveformData | null => {
    if (!moduleRef.current) return null;

    try {
      const generator = new moduleRef.current.WaveformGenerator(targetPeaks);
      return generator.generate_from_audio_buffer(audioBuffer);
    } catch (err) {
      console.error('Waveform generation failed:', err);
      return null;
    }
  }, []);

  // Waveform generation from raw samples
  const generateWaveformFromSamples = useCallback(
    (samples: Float32Array, sampleRate: number, channels: number, targetPeaks: number = 800): WaveformData | null => {
      if (!moduleRef.current) return null;

      try {
        const generator = new moduleRef.current.WaveformGenerator(targetPeaks);
        return generator.generate(samples, sampleRate, channels);
      } catch (err) {
        console.error('Waveform generation failed:', err);
        return null;
      }
    },
    []
  );

  // Audio analysis from AudioBuffer
  const analyzeAudio = useCallback((audioBuffer: AudioBuffer): AudioAnalysisResult | null => {
    if (!moduleRef.current) return null;

    try {
      const analyzer = new moduleRef.current.AudioAnalyzer(2048);
      return analyzer.analyze_audio_buffer(audioBuffer);
    } catch (err) {
      console.error('Audio analysis failed:', err);
      return null;
    }
  }, []);

  // Audio analysis from raw samples
  const analyzeAudioSamples = useCallback((samples: Float32Array, sampleRate: number): AudioAnalysisResult | null => {
    if (!moduleRef.current) return null;

    try {
      const analyzer = new moduleRef.current.AudioAnalyzer(2048);
      return analyzer.analyze(samples, sampleRate);
    } catch (err) {
      console.error('Audio analysis failed:', err);
      return null;
    }
  }, []);

  // Utility functions
  const utils = moduleRef.current
    ? {
        dbToLinear: moduleRef.current.db_to_linear,
        linearToDb: moduleRef.current.linear_to_db,
        frequencyToMidi: moduleRef.current.frequency_to_midi,
        midiToFrequency: moduleRef.current.midi_to_frequency,
        bpmToMs: moduleRef.current.bpm_to_ms,
        computeRms: moduleRef.current.AudioBufferUtils.compute_rms,
        computePeak: moduleRef.current.AudioBufferUtils.compute_peak,
      }
    : null;

  return {
    isLoaded,
    isLoading,
    error,
    generateWaveform,
    generateWaveformFromSamples,
    analyzeAudio,
    analyzeAudioSamples,
    utils,
  };
}

/**
 * Hook for real-time audio visualization
 */
export interface UseWasmVisualizerReturn {
  isReady: boolean;
  setColors: (primary: string, secondary: string, background: string) => void;
  draw: (ctx: CanvasRenderingContext2D, analyser: AnalyserNode) => void;
  drawWaveform: (ctx: CanvasRenderingContext2D, data: Float32Array, width: number, height: number) => void;
  drawFrequencyBars: (ctx: CanvasRenderingContext2D, data: Uint8Array, width: number, height: number) => void;
  drawCircular: (ctx: CanvasRenderingContext2D, data: Uint8Array, width: number, height: number) => void;
  drawSpectrogram: (ctx: CanvasRenderingContext2D, data: Uint8Array, width: number, height: number) => void;
}

export function useWasmVisualizer(config?: Partial<VisualizationConfig>): UseWasmVisualizerReturn {
  const visualizerRef = useRef<WasmVisualizer | null>(null);
  const [isReady, setIsReady] = useState(false);

  // Initialize visualizer
  useEffect(() => {
    loadWasmModule()
      .then((mod) => {
        const defaultConfig = {
          viz_type: 'FrequencyBars',
          fft_size: 2048,
          smoothing: 0.8,
          bar_width: 3.0,
          bar_gap: 1.0,
          mirror: false,
          gradient: true,
          ...config,
        };

        visualizerRef.current = new mod.RealtimeVisualizer(defaultConfig);
        setIsReady(true);
      })
      .catch(console.error);
  }, []);

  const setColors = useCallback((primary: string, secondary: string, background: string) => {
    visualizerRef.current?.set_colors(primary, secondary, background);
  }, []);

  const draw = useCallback((ctx: CanvasRenderingContext2D, analyser: AnalyserNode) => {
    visualizerRef.current?.draw(ctx, analyser);
  }, []);

  const drawWaveform = useCallback((ctx: CanvasRenderingContext2D, data: Float32Array, width: number, height: number) => {
    visualizerRef.current?.draw_waveform(ctx, data, width, height);
  }, []);

  const drawFrequencyBars = useCallback((ctx: CanvasRenderingContext2D, data: Uint8Array, width: number, height: number) => {
    visualizerRef.current?.draw_frequency_bars(ctx, data, width, height);
  }, []);

  const drawCircular = useCallback((ctx: CanvasRenderingContext2D, data: Uint8Array, width: number, height: number) => {
    visualizerRef.current?.draw_circular(ctx, data, width, height);
  }, []);

  const drawSpectrogram = useCallback((ctx: CanvasRenderingContext2D, data: Uint8Array, width: number, height: number) => {
    visualizerRef.current?.draw_spectrogram(ctx, data, width, height);
  }, []);

  return {
    isReady,
    setColors,
    draw,
    drawWaveform,
    drawFrequencyBars,
    drawCircular,
    drawSpectrogram,
  };
}

export default useWasmAudio;
