// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * useAudioEffects Hook
 *
 * Provides real-time audio effects processing using WebAssembly DSP.
 * Includes EQ, reverb, spatial audio, compressor, and limiter.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

// Types for WASM effects (will be imported from @mycelix/wasm when built)
interface WasmEffects {
  ParametricEQ: new (sampleRate: number) => ParametricEQ;
  Reverb: new (sampleRate: number) => Reverb;
  SpatialAudio: new (sampleRate: number) => SpatialAudio;
  Compressor: new (sampleRate: number) => Compressor;
  Limiter: new (sampleRate: number) => Limiter;
  EffectsChain: new (sampleRate: number) => EffectsChain;
  BiquadFilter: new (sampleRate: number) => BiquadFilter;
}

interface BiquadFilter {
  set_lowpass(frequency: number, q: number): void;
  set_highpass(frequency: number, q: number): void;
  set_bandpass(frequency: number, q: number): void;
  set_notch(frequency: number, q: number): void;
  set_peak(frequency: number, q: number, gainDb: number): void;
  process(input: Float32Array): Float32Array;
  reset(): void;
}

interface ParametricEQ {
  set_low_shelf(frequency: number, gainDb: number): void;
  set_low_mid(frequency: number, q: number, gainDb: number): void;
  set_mid(frequency: number, q: number, gainDb: number): void;
  set_high_mid(frequency: number, q: number, gainDb: number): void;
  set_high_shelf(frequency: number, gainDb: number): void;
  process(input: Float32Array): Float32Array;
  reset(): void;
}

interface Reverb {
  set_room_size(size: number): void;
  set_damping(damping: number): void;
  set_wet(wet: number): void;
  set_dry(dry: number): void;
  set_mix(mix: number): void;
  process(input: Float32Array): Float32Array;
  process_stereo(left: Float32Array, right: Float32Array): Float32Array;
}

interface SpatialAudio {
  set_position(azimuth: number, elevation: number, distance: number): void;
  set_azimuth(azimuth: number): void;
  set_elevation(elevation: number): void;
  set_distance(distance: number): void;
  process(input: Float32Array): Float32Array;
}

interface Compressor {
  set_threshold(db: number): void;
  set_ratio(ratio: number): void;
  set_attack(ms: number): void;
  set_release(ms: number): void;
  set_knee(db: number): void;
  set_makeup_gain(db: number): void;
  process(input: Float32Array): Float32Array;
  get_gain_reduction(): number;
}

interface Limiter {
  set_ceiling(db: number): void;
  set_release(ms: number): void;
  process(input: Float32Array): Float32Array;
}

interface EffectsChain {
  enable_eq(enabled: boolean): void;
  enable_reverb(enabled: boolean): void;
  enable_compressor(enabled: boolean): void;
  enable_limiter(enabled: boolean): void;
  process(input: Float32Array): Float32Array;
}

// === State Types ===

export interface EQSettings {
  lowShelf: { frequency: number; gain: number };
  lowMid: { frequency: number; q: number; gain: number };
  mid: { frequency: number; q: number; gain: number };
  highMid: { frequency: number; q: number; gain: number };
  highShelf: { frequency: number; gain: number };
}

export interface ReverbSettings {
  roomSize: number;
  damping: number;
  mix: number;
}

export interface SpatialSettings {
  azimuth: number;
  elevation: number;
  distance: number;
}

export interface CompressorSettings {
  threshold: number;
  ratio: number;
  attack: number;
  release: number;
  knee: number;
  makeupGain: number;
}

export interface LimiterSettings {
  ceiling: number;
  release: number;
}

export interface EffectsState {
  isLoaded: boolean;
  isProcessing: boolean;
  eq: { enabled: boolean; settings: EQSettings };
  reverb: { enabled: boolean; settings: ReverbSettings };
  spatial: { enabled: boolean; settings: SpatialSettings };
  compressor: { enabled: boolean; settings: CompressorSettings };
  limiter: { enabled: boolean; settings: LimiterSettings };
  gainReduction: number;
}

// === Default Settings ===

const DEFAULT_EQ: EQSettings = {
  lowShelf: { frequency: 100, gain: 0 },
  lowMid: { frequency: 250, q: 1, gain: 0 },
  mid: { frequency: 1000, q: 1, gain: 0 },
  highMid: { frequency: 4000, q: 1, gain: 0 },
  highShelf: { frequency: 10000, gain: 0 },
};

const DEFAULT_REVERB: ReverbSettings = {
  roomSize: 0.5,
  damping: 0.5,
  mix: 0.3,
};

const DEFAULT_SPATIAL: SpatialSettings = {
  azimuth: 0,
  elevation: 0,
  distance: 1,
};

const DEFAULT_COMPRESSOR: CompressorSettings = {
  threshold: -20,
  ratio: 4,
  attack: 10,
  release: 100,
  knee: 6,
  makeupGain: 0,
};

const DEFAULT_LIMITER: LimiterSettings = {
  ceiling: -0.3,
  release: 50,
};

// === WASM Module Loader ===

let wasmModule: WasmEffects | null = null;
let wasmLoadPromise: Promise<WasmEffects> | null = null;

async function loadWasmEffects(): Promise<WasmEffects> {
  if (wasmModule) return wasmModule;

  if (!wasmLoadPromise) {
    wasmLoadPromise = (async () => {
      try {
        const module = await import('@mycelix/wasm');
        wasmModule = module as unknown as WasmEffects;
        return wasmModule;
      } catch (error) {
        console.error('[AudioEffects] Failed to load WASM module:', error);
        throw error;
      }
    })();
  }

  return wasmLoadPromise;
}

// === Hook ===

export interface UseAudioEffectsReturn extends EffectsState {
  // Initialization
  initialize: (sampleRate?: number) => Promise<void>;
  dispose: () => void;

  // EQ controls
  setEQEnabled: (enabled: boolean) => void;
  setEQBand: (band: keyof EQSettings, settings: EQSettings[typeof band]) => void;
  resetEQ: () => void;

  // Reverb controls
  setReverbEnabled: (enabled: boolean) => void;
  setReverbSettings: (settings: Partial<ReverbSettings>) => void;
  resetReverb: () => void;

  // Spatial controls
  setSpatialEnabled: (enabled: boolean) => void;
  setSpatialPosition: (azimuth: number, elevation: number, distance: number) => void;
  resetSpatial: () => void;

  // Compressor controls
  setCompressorEnabled: (enabled: boolean) => void;
  setCompressorSettings: (settings: Partial<CompressorSettings>) => void;
  resetCompressor: () => void;

  // Limiter controls
  setLimiterEnabled: (enabled: boolean) => void;
  setLimiterSettings: (settings: Partial<LimiterSettings>) => void;
  resetLimiter: () => void;

  // Processing
  processAudio: (input: Float32Array) => Float32Array;
  processAudioStereo: (left: Float32Array, right: Float32Array) => Float32Array;

  // Audio Worklet integration
  createWorkletProcessor: () => string;
}

export function useAudioEffects(): UseAudioEffectsReturn {
  const eqRef = useRef<ParametricEQ | null>(null);
  const reverbRef = useRef<Reverb | null>(null);
  const spatialRef = useRef<SpatialAudio | null>(null);
  const compressorRef = useRef<Compressor | null>(null);
  const limiterRef = useRef<Limiter | null>(null);
  const sampleRateRef = useRef<number>(44100);

  const [state, setState] = useState<EffectsState>({
    isLoaded: false,
    isProcessing: false,
    eq: { enabled: false, settings: DEFAULT_EQ },
    reverb: { enabled: false, settings: DEFAULT_REVERB },
    spatial: { enabled: false, settings: DEFAULT_SPATIAL },
    compressor: { enabled: false, settings: DEFAULT_COMPRESSOR },
    limiter: { enabled: true, settings: DEFAULT_LIMITER },
    gainReduction: 0,
  });

  // Initialize WASM effects
  const initialize = useCallback(async (sampleRate: number = 44100) => {
    try {
      const wasm = await loadWasmEffects();
      sampleRateRef.current = sampleRate;

      eqRef.current = new wasm.ParametricEQ(sampleRate);
      reverbRef.current = new wasm.Reverb(sampleRate);
      spatialRef.current = new wasm.SpatialAudio(sampleRate);
      compressorRef.current = new wasm.Compressor(sampleRate);
      limiterRef.current = new wasm.Limiter(sampleRate);

      setState((s) => ({ ...s, isLoaded: true }));
    } catch (error) {
      console.error('[AudioEffects] Initialization failed:', error);
    }
  }, []);

  // Dispose all effects
  const dispose = useCallback(() => {
    eqRef.current = null;
    reverbRef.current = null;
    spatialRef.current = null;
    compressorRef.current = null;
    limiterRef.current = null;

    setState((s) => ({ ...s, isLoaded: false }));
  }, []);

  // === EQ Controls ===

  const setEQEnabled = useCallback((enabled: boolean) => {
    setState((s) => ({ ...s, eq: { ...s.eq, enabled } }));
  }, []);

  const setEQBand = useCallback(
    <K extends keyof EQSettings>(band: K, settings: EQSettings[K]) => {
      if (!eqRef.current) return;

      const eq = eqRef.current;
      const newSettings = { ...state.eq.settings, [band]: settings };

      switch (band) {
        case 'lowShelf':
          eq.set_low_shelf(
            (settings as EQSettings['lowShelf']).frequency,
            (settings as EQSettings['lowShelf']).gain
          );
          break;
        case 'lowMid':
          eq.set_low_mid(
            (settings as EQSettings['lowMid']).frequency,
            (settings as EQSettings['lowMid']).q,
            (settings as EQSettings['lowMid']).gain
          );
          break;
        case 'mid':
          eq.set_mid(
            (settings as EQSettings['mid']).frequency,
            (settings as EQSettings['mid']).q,
            (settings as EQSettings['mid']).gain
          );
          break;
        case 'highMid':
          eq.set_high_mid(
            (settings as EQSettings['highMid']).frequency,
            (settings as EQSettings['highMid']).q,
            (settings as EQSettings['highMid']).gain
          );
          break;
        case 'highShelf':
          eq.set_high_shelf(
            (settings as EQSettings['highShelf']).frequency,
            (settings as EQSettings['highShelf']).gain
          );
          break;
      }

      setState((s) => ({ ...s, eq: { ...s.eq, settings: newSettings } }));
    },
    [state.eq.settings]
  );

  const resetEQ = useCallback(() => {
    if (!eqRef.current) return;

    const eq = eqRef.current;
    eq.reset();
    eq.set_low_shelf(DEFAULT_EQ.lowShelf.frequency, DEFAULT_EQ.lowShelf.gain);
    eq.set_low_mid(DEFAULT_EQ.lowMid.frequency, DEFAULT_EQ.lowMid.q, DEFAULT_EQ.lowMid.gain);
    eq.set_mid(DEFAULT_EQ.mid.frequency, DEFAULT_EQ.mid.q, DEFAULT_EQ.mid.gain);
    eq.set_high_mid(DEFAULT_EQ.highMid.frequency, DEFAULT_EQ.highMid.q, DEFAULT_EQ.highMid.gain);
    eq.set_high_shelf(DEFAULT_EQ.highShelf.frequency, DEFAULT_EQ.highShelf.gain);

    setState((s) => ({ ...s, eq: { ...s.eq, settings: DEFAULT_EQ } }));
  }, []);

  // === Reverb Controls ===

  const setReverbEnabled = useCallback((enabled: boolean) => {
    setState((s) => ({ ...s, reverb: { ...s.reverb, enabled } }));
  }, []);

  const setReverbSettings = useCallback((settings: Partial<ReverbSettings>) => {
    if (!reverbRef.current) return;

    const reverb = reverbRef.current;
    const newSettings = { ...state.reverb.settings, ...settings };

    if (settings.roomSize !== undefined) reverb.set_room_size(settings.roomSize);
    if (settings.damping !== undefined) reverb.set_damping(settings.damping);
    if (settings.mix !== undefined) reverb.set_mix(settings.mix);

    setState((s) => ({ ...s, reverb: { ...s.reverb, settings: newSettings } }));
  }, [state.reverb.settings]);

  const resetReverb = useCallback(() => {
    if (!reverbRef.current) return;

    const reverb = reverbRef.current;
    reverb.set_room_size(DEFAULT_REVERB.roomSize);
    reverb.set_damping(DEFAULT_REVERB.damping);
    reverb.set_mix(DEFAULT_REVERB.mix);

    setState((s) => ({ ...s, reverb: { ...s.reverb, settings: DEFAULT_REVERB } }));
  }, []);

  // === Spatial Controls ===

  const setSpatialEnabled = useCallback((enabled: boolean) => {
    setState((s) => ({ ...s, spatial: { ...s.spatial, enabled } }));
  }, []);

  const setSpatialPosition = useCallback(
    (azimuth: number, elevation: number, distance: number) => {
      if (!spatialRef.current) return;

      spatialRef.current.set_position(azimuth, elevation, distance);

      setState((s) => ({
        ...s,
        spatial: { ...s.spatial, settings: { azimuth, elevation, distance } },
      }));
    },
    []
  );

  const resetSpatial = useCallback(() => {
    if (!spatialRef.current) return;

    spatialRef.current.set_position(
      DEFAULT_SPATIAL.azimuth,
      DEFAULT_SPATIAL.elevation,
      DEFAULT_SPATIAL.distance
    );

    setState((s) => ({ ...s, spatial: { ...s.spatial, settings: DEFAULT_SPATIAL } }));
  }, []);

  // === Compressor Controls ===

  const setCompressorEnabled = useCallback((enabled: boolean) => {
    setState((s) => ({ ...s, compressor: { ...s.compressor, enabled } }));
  }, []);

  const setCompressorSettings = useCallback((settings: Partial<CompressorSettings>) => {
    if (!compressorRef.current) return;

    const comp = compressorRef.current;
    const newSettings = { ...state.compressor.settings, ...settings };

    if (settings.threshold !== undefined) comp.set_threshold(settings.threshold);
    if (settings.ratio !== undefined) comp.set_ratio(settings.ratio);
    if (settings.attack !== undefined) comp.set_attack(settings.attack);
    if (settings.release !== undefined) comp.set_release(settings.release);
    if (settings.knee !== undefined) comp.set_knee(settings.knee);
    if (settings.makeupGain !== undefined) comp.set_makeup_gain(settings.makeupGain);

    setState((s) => ({ ...s, compressor: { ...s.compressor, settings: newSettings } }));
  }, [state.compressor.settings]);

  const resetCompressor = useCallback(() => {
    if (!compressorRef.current) return;

    const comp = compressorRef.current;
    comp.set_threshold(DEFAULT_COMPRESSOR.threshold);
    comp.set_ratio(DEFAULT_COMPRESSOR.ratio);
    comp.set_attack(DEFAULT_COMPRESSOR.attack);
    comp.set_release(DEFAULT_COMPRESSOR.release);
    comp.set_knee(DEFAULT_COMPRESSOR.knee);
    comp.set_makeup_gain(DEFAULT_COMPRESSOR.makeupGain);

    setState((s) => ({ ...s, compressor: { ...s.compressor, settings: DEFAULT_COMPRESSOR } }));
  }, []);

  // === Limiter Controls ===

  const setLimiterEnabled = useCallback((enabled: boolean) => {
    setState((s) => ({ ...s, limiter: { ...s.limiter, enabled } }));
  }, []);

  const setLimiterSettings = useCallback((settings: Partial<LimiterSettings>) => {
    if (!limiterRef.current) return;

    const limiter = limiterRef.current;
    const newSettings = { ...state.limiter.settings, ...settings };

    if (settings.ceiling !== undefined) limiter.set_ceiling(settings.ceiling);
    if (settings.release !== undefined) limiter.set_release(settings.release);

    setState((s) => ({ ...s, limiter: { ...s.limiter, settings: newSettings } }));
  }, [state.limiter.settings]);

  const resetLimiter = useCallback(() => {
    if (!limiterRef.current) return;

    const limiter = limiterRef.current;
    limiter.set_ceiling(DEFAULT_LIMITER.ceiling);
    limiter.set_release(DEFAULT_LIMITER.release);

    setState((s) => ({ ...s, limiter: { ...s.limiter, settings: DEFAULT_LIMITER } }));
  }, []);

  // === Audio Processing ===

  const processAudio = useCallback(
    (input: Float32Array): Float32Array => {
      if (!state.isLoaded) return input;

      let output = input;

      // EQ
      if (state.eq.enabled && eqRef.current) {
        output = eqRef.current.process(output);
      }

      // Compressor
      if (state.compressor.enabled && compressorRef.current) {
        output = compressorRef.current.process(output);
        const gr = compressorRef.current.get_gain_reduction();
        if (gr !== state.gainReduction) {
          setState((s) => ({ ...s, gainReduction: gr }));
        }
      }

      // Reverb
      if (state.reverb.enabled && reverbRef.current) {
        output = reverbRef.current.process(output);
      }

      // Spatial
      if (state.spatial.enabled && spatialRef.current) {
        output = spatialRef.current.process(output);
      }

      // Limiter (always last)
      if (state.limiter.enabled && limiterRef.current) {
        output = limiterRef.current.process(output);
      }

      return output;
    },
    [state]
  );

  const processAudioStereo = useCallback(
    (left: Float32Array, right: Float32Array): Float32Array => {
      if (!state.isLoaded) {
        // Return interleaved stereo
        const output = new Float32Array(left.length * 2);
        for (let i = 0; i < left.length; i++) {
          output[i * 2] = left[i];
          output[i * 2 + 1] = right[i];
        }
        return output;
      }

      // Process left and right channels
      let leftOut = left;
      let rightOut = right;

      // EQ (process each channel)
      if (state.eq.enabled && eqRef.current) {
        leftOut = eqRef.current.process(leftOut);
        // Note: In production, you'd have separate EQ instances for L/R
        rightOut = eqRef.current.process(rightOut);
      }

      // Reverb (stereo processing)
      if (state.reverb.enabled && reverbRef.current) {
        return reverbRef.current.process_stereo(leftOut, rightOut);
      }

      // Return interleaved
      const output = new Float32Array(left.length * 2);
      for (let i = 0; i < left.length; i++) {
        output[i * 2] = leftOut[i];
        output[i * 2 + 1] = rightOut[i];
      }
      return output;
    },
    [state]
  );

  // Create Audio Worklet processor code
  const createWorkletProcessor = useCallback((): string => {
    return `
      class EffectsProcessor extends AudioWorkletProcessor {
        constructor() {
          super();
          this.port.onmessage = this.handleMessage.bind(this);
        }

        handleMessage(event) {
          // Handle settings updates from main thread
          const { type, settings } = event.data;
          // Apply settings...
        }

        process(inputs, outputs, parameters) {
          const input = inputs[0];
          const output = outputs[0];

          if (input.length > 0) {
            // Copy input to output (effects would be applied via WASM)
            for (let channel = 0; channel < output.length; channel++) {
              output[channel].set(input[channel] || new Float32Array(128));
            }
          }

          return true;
        }
      }

      registerProcessor('effects-processor', EffectsProcessor);
    `;
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      dispose();
    };
  }, [dispose]);

  return {
    ...state,
    initialize,
    dispose,
    setEQEnabled,
    setEQBand,
    resetEQ,
    setReverbEnabled,
    setReverbSettings,
    resetReverb,
    setSpatialEnabled,
    setSpatialPosition,
    resetSpatial,
    setCompressorEnabled,
    setCompressorSettings,
    resetCompressor,
    setLimiterEnabled,
    setLimiterSettings,
    resetLimiter,
    processAudio,
    processAudioStereo,
    createWorkletProcessor,
  };
}

export default useAudioEffects;
