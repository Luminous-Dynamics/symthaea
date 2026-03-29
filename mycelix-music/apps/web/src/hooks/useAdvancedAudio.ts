// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Advanced Audio Hook
 *
 * Professional audio features:
 * - Spectral editing
 * - Spatial audio encoding
 * - LUFS metering
 * - Multi-band processing
 */

import { useState, useCallback, useRef, useEffect } from 'react';

// Spectral editing types
export interface SpectralRegion {
  id: string;
  startTime: number;
  endTime: number;
  startFreq: number;
  endFreq: number;
  type: 'selection' | 'attenuate' | 'boost' | 'denoise';
  amount?: number;
}

export interface SpectralData {
  frequencies: Float32Array[];
  magnitudes: Float32Array[];
  phases: Float32Array[];
  timeResolution: number;
  frequencyResolution: number;
}

// Loudness metering
export interface LoudnessData {
  momentary: number;     // Short-term (400ms)
  shortTerm: number;     // Short-term (3s)
  integrated: number;    // Integrated (full program)
  range: number;         // Loudness Range (LRA)
  truePeak: number;      // True Peak (dBTP)
  truePeakL: number;     // Left channel true peak
  truePeakR: number;     // Right channel true peak
  psr: number;           // Peak-to-Short-Term Ratio
}

// Multi-band processing
export interface MultibandConfig {
  bands: BandConfig[];
  crossoverSlope: 12 | 24 | 48;  // dB/octave
}

export interface BandConfig {
  id: string;
  lowFreq: number;
  highFreq: number;
  gain: number;
  compression: {
    enabled: boolean;
    threshold: number;
    ratio: number;
    attack: number;
    release: number;
    knee: number;
  };
  expansion: {
    enabled: boolean;
    threshold: number;
    ratio: number;
    attack: number;
    release: number;
  };
}

// Spatial audio
export type SpatialFormat = 'stereo' | 'binaural' | '5.1' | '7.1' | 'atmos' | 'ambisonics';

export interface SpatialConfig {
  format: SpatialFormat;
  ambisonicsOrder?: 1 | 2 | 3;
  headTracking?: boolean;
  roomSize?: 'small' | 'medium' | 'large' | 'hall';
  listenerPosition?: { x: number; y: number; z: number };
}

export interface AdvancedAudioState {
  isProcessing: boolean;
  spectralData: SpectralData | null;
  loudness: LoudnessData | null;
  spatialConfig: SpatialConfig;
  multibandConfig: MultibandConfig;
}

// Default multiband configuration
const DEFAULT_MULTIBAND: MultibandConfig = {
  bands: [
    {
      id: 'low',
      lowFreq: 20,
      highFreq: 200,
      gain: 0,
      compression: { enabled: false, threshold: -20, ratio: 4, attack: 20, release: 100, knee: 6 },
      expansion: { enabled: false, threshold: -50, ratio: 2, attack: 10, release: 50 },
    },
    {
      id: 'low-mid',
      lowFreq: 200,
      highFreq: 1000,
      gain: 0,
      compression: { enabled: false, threshold: -20, ratio: 4, attack: 15, release: 80, knee: 6 },
      expansion: { enabled: false, threshold: -50, ratio: 2, attack: 10, release: 50 },
    },
    {
      id: 'mid',
      lowFreq: 1000,
      highFreq: 4000,
      gain: 0,
      compression: { enabled: false, threshold: -20, ratio: 4, attack: 10, release: 60, knee: 6 },
      expansion: { enabled: false, threshold: -50, ratio: 2, attack: 10, release: 50 },
    },
    {
      id: 'high-mid',
      lowFreq: 4000,
      highFreq: 10000,
      gain: 0,
      compression: { enabled: false, threshold: -20, ratio: 4, attack: 5, release: 40, knee: 6 },
      expansion: { enabled: false, threshold: -50, ratio: 2, attack: 10, release: 50 },
    },
    {
      id: 'high',
      lowFreq: 10000,
      highFreq: 20000,
      gain: 0,
      compression: { enabled: false, threshold: -20, ratio: 4, attack: 2, release: 30, knee: 6 },
      expansion: { enabled: false, threshold: -50, ratio: 2, attack: 10, release: 50 },
    },
  ],
  crossoverSlope: 24,
};

export function useAdvancedAudio(audioContext?: AudioContext) {
  const [state, setState] = useState<AdvancedAudioState>({
    isProcessing: false,
    spectralData: null,
    loudness: null,
    spatialConfig: { format: 'stereo' },
    multibandConfig: DEFAULT_MULTIBAND,
  });

  const ctxRef = useRef<AudioContext | null>(audioContext || null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const loudnessMeterRef = useRef<any>(null);

  // Initialize audio context
  useEffect(() => {
    if (!ctxRef.current && typeof window !== 'undefined') {
      ctxRef.current = new AudioContext();
    }
  }, []);

  /**
   * Perform spectral analysis with STFT
   */
  const analyzeSpectrum = useCallback(async (
    audioBuffer: AudioBuffer,
    options: {
      fftSize?: number;
      hopSize?: number;
      windowType?: 'hann' | 'hamming' | 'blackman';
    } = {}
  ): Promise<SpectralData> => {
    const { fftSize = 2048, hopSize = 512, windowType = 'hann' } = options;

    setState(prev => ({ ...prev, isProcessing: true }));

    try {
      const channelData = audioBuffer.getChannelData(0);
      const numFrames = Math.floor((channelData.length - fftSize) / hopSize) + 1;
      const numBins = fftSize / 2;

      const frequencies: Float32Array[] = [];
      const magnitudes: Float32Array[] = [];
      const phases: Float32Array[] = [];

      // Generate window function
      const window = generateWindow(windowType, fftSize);

      // Process each frame
      for (let frame = 0; frame < numFrames; frame++) {
        const start = frame * hopSize;
        const frameData = new Float32Array(fftSize);

        // Apply window
        for (let i = 0; i < fftSize; i++) {
          frameData[i] = (channelData[start + i] || 0) * window[i];
        }

        // Perform FFT
        const { real, imag } = performFFT(frameData);

        // Calculate magnitude and phase
        const frameMagnitudes = new Float32Array(numBins);
        const framePhases = new Float32Array(numBins);

        for (let i = 0; i < numBins; i++) {
          frameMagnitudes[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
          framePhases[i] = Math.atan2(imag[i], real[i]);
        }

        magnitudes.push(frameMagnitudes);
        phases.push(framePhases);
      }

      // Calculate frequency bins
      const freqBins = new Float32Array(numBins);
      for (let i = 0; i < numBins; i++) {
        freqBins[i] = (i * audioBuffer.sampleRate) / fftSize;
      }
      frequencies.push(freqBins);

      const spectralData: SpectralData = {
        frequencies,
        magnitudes,
        phases,
        timeResolution: hopSize / audioBuffer.sampleRate,
        frequencyResolution: audioBuffer.sampleRate / fftSize,
      };

      setState(prev => ({ ...prev, isProcessing: false, spectralData }));
      return spectralData;
    } catch (error) {
      setState(prev => ({ ...prev, isProcessing: false }));
      throw error;
    }
  }, []);

  /**
   * Apply spectral edit to audio
   */
  const applySpectralEdit = useCallback(async (
    audioBuffer: AudioBuffer,
    regions: SpectralRegion[]
  ): Promise<AudioBuffer> => {
    setState(prev => ({ ...prev, isProcessing: true }));

    try {
      const ctx = ctxRef.current!;
      const newBuffer = ctx.createBuffer(
        audioBuffer.numberOfChannels,
        audioBuffer.length,
        audioBuffer.sampleRate
      );

      // Copy original data
      for (let ch = 0; ch < audioBuffer.numberOfChannels; ch++) {
        newBuffer.copyToChannel(audioBuffer.getChannelData(ch), ch);
      }

      // Apply each region edit
      for (const region of regions) {
        await applyRegionEdit(newBuffer, region);
      }

      setState(prev => ({ ...prev, isProcessing: false }));
      return newBuffer;
    } catch (error) {
      setState(prev => ({ ...prev, isProcessing: false }));
      throw error;
    }
  }, []);

  /**
   * Measure loudness (ITU-R BS.1770)
   */
  const measureLoudness = useCallback(async (
    audioBuffer: AudioBuffer
  ): Promise<LoudnessData> => {
    setState(prev => ({ ...prev, isProcessing: true }));

    try {
      const sampleRate = audioBuffer.sampleRate;
      const leftChannel = audioBuffer.getChannelData(0);
      const rightChannel = audioBuffer.numberOfChannels > 1
        ? audioBuffer.getChannelData(1)
        : leftChannel;

      // Pre-filter (K-weighting)
      const leftFiltered = applyKWeighting(leftChannel, sampleRate);
      const rightFiltered = applyKWeighting(rightChannel, sampleRate);

      // Calculate mean square
      let sumSquare = 0;
      for (let i = 0; i < leftFiltered.length; i++) {
        sumSquare += leftFiltered[i] * leftFiltered[i] + rightFiltered[i] * rightFiltered[i];
      }
      const meanSquare = sumSquare / (leftFiltered.length * 2);

      // Calculate LUFS
      const integrated = -0.691 + 10 * Math.log10(meanSquare);

      // Calculate true peak
      const truePeakL = calculateTruePeak(leftChannel, sampleRate);
      const truePeakR = calculateTruePeak(rightChannel, sampleRate);
      const truePeak = Math.max(truePeakL, truePeakR);

      // Short-term loudness (3s window at end)
      const shortTermSamples = Math.min(sampleRate * 3, leftFiltered.length);
      let shortTermSum = 0;
      const startIdx = leftFiltered.length - shortTermSamples;
      for (let i = startIdx; i < leftFiltered.length; i++) {
        shortTermSum += leftFiltered[i] * leftFiltered[i] + rightFiltered[i] * rightFiltered[i];
      }
      const shortTerm = -0.691 + 10 * Math.log10(shortTermSum / (shortTermSamples * 2));

      // Momentary loudness (400ms window at end)
      const momentarySamples = Math.min(sampleRate * 0.4, leftFiltered.length);
      let momentarySum = 0;
      const momStartIdx = leftFiltered.length - momentarySamples;
      for (let i = momStartIdx; i < leftFiltered.length; i++) {
        momentarySum += leftFiltered[i] * leftFiltered[i] + rightFiltered[i] * rightFiltered[i];
      }
      const momentary = -0.691 + 10 * Math.log10(momentarySum / (momentarySamples * 2));

      // Loudness Range (simplified)
      const range = calculateLoudnessRange(leftFiltered, rightFiltered, sampleRate);

      const loudness: LoudnessData = {
        momentary,
        shortTerm,
        integrated,
        range,
        truePeak: 20 * Math.log10(truePeak),
        truePeakL: 20 * Math.log10(truePeakL),
        truePeakR: 20 * Math.log10(truePeakR),
        psr: truePeak / Math.pow(10, shortTerm / 10),
      };

      setState(prev => ({ ...prev, isProcessing: false, loudness }));
      return loudness;
    } catch (error) {
      setState(prev => ({ ...prev, isProcessing: false }));
      throw error;
    }
  }, []);

  /**
   * Apply multiband processing
   */
  const applyMultiband = useCallback(async (
    audioBuffer: AudioBuffer,
    config?: MultibandConfig
  ): Promise<AudioBuffer> => {
    const mbConfig = config || state.multibandConfig;
    setState(prev => ({ ...prev, isProcessing: true }));

    try {
      const ctx = ctxRef.current!;
      const offlineCtx = new OfflineAudioContext(
        audioBuffer.numberOfChannels,
        audioBuffer.length,
        audioBuffer.sampleRate
      );

      // Create source
      const source = offlineCtx.createBufferSource();
      source.buffer = audioBuffer;

      // Create band splitters and processors
      const bands = mbConfig.bands.map(band => {
        const lowpass = offlineCtx.createBiquadFilter();
        lowpass.type = 'lowpass';
        lowpass.frequency.value = band.highFreq;
        lowpass.Q.value = 0.707;

        const highpass = offlineCtx.createBiquadFilter();
        highpass.type = 'highpass';
        highpass.frequency.value = band.lowFreq;
        highpass.Q.value = 0.707;

        const gain = offlineCtx.createGain();
        gain.gain.value = Math.pow(10, band.gain / 20);

        return { lowpass, highpass, gain, config: band };
      });

      // Create merger
      const merger = offlineCtx.createGain();
      merger.connect(offlineCtx.destination);

      // Connect bands
      for (const band of bands) {
        source.connect(band.highpass);
        band.highpass.connect(band.lowpass);
        band.lowpass.connect(band.gain);
        band.gain.connect(merger);
      }

      source.start();
      const rendered = await offlineCtx.startRendering();

      setState(prev => ({ ...prev, isProcessing: false }));
      return rendered;
    } catch (error) {
      setState(prev => ({ ...prev, isProcessing: false }));
      throw error;
    }
  }, [state.multibandConfig]);

  /**
   * Encode to spatial format
   */
  const encodeSpatial = useCallback(async (
    audioBuffer: AudioBuffer,
    config: SpatialConfig
  ): Promise<AudioBuffer> => {
    setState(prev => ({ ...prev, isProcessing: true, spatialConfig: config }));

    try {
      const ctx = ctxRef.current!;
      let outputChannels = 2;

      switch (config.format) {
        case '5.1':
          outputChannels = 6;
          break;
        case '7.1':
          outputChannels = 8;
          break;
        case 'atmos':
          outputChannels = 12; // 7.1.4
          break;
        case 'ambisonics':
          const order = config.ambisonicsOrder || 1;
          outputChannels = (order + 1) ** 2;
          break;
      }

      const offlineCtx = new OfflineAudioContext(
        outputChannels,
        audioBuffer.length,
        audioBuffer.sampleRate
      );

      // Simple upmix (would use proper algorithms in production)
      const source = offlineCtx.createBufferSource();
      source.buffer = audioBuffer;

      const splitter = offlineCtx.createChannelSplitter(audioBuffer.numberOfChannels);
      const merger = offlineCtx.createChannelMerger(outputChannels);

      source.connect(splitter);

      // Map input channels to output
      for (let i = 0; i < outputChannels; i++) {
        const inputChannel = i % audioBuffer.numberOfChannels;
        splitter.connect(merger, inputChannel, i);
      }

      merger.connect(offlineCtx.destination);
      source.start();

      const rendered = await offlineCtx.startRendering();

      setState(prev => ({ ...prev, isProcessing: false }));
      return rendered;
    } catch (error) {
      setState(prev => ({ ...prev, isProcessing: false }));
      throw error;
    }
  }, []);

  /**
   * Update multiband configuration
   */
  const updateMultibandConfig = useCallback((config: Partial<MultibandConfig>) => {
    setState(prev => ({
      ...prev,
      multibandConfig: { ...prev.multibandConfig, ...config },
    }));
  }, []);

  /**
   * Update band configuration
   */
  const updateBandConfig = useCallback((bandId: string, config: Partial<BandConfig>) => {
    setState(prev => ({
      ...prev,
      multibandConfig: {
        ...prev.multibandConfig,
        bands: prev.multibandConfig.bands.map(b =>
          b.id === bandId ? { ...b, ...config } : b
        ),
      },
    }));
  }, []);

  return {
    ...state,
    analyzeSpectrum,
    applySpectralEdit,
    measureLoudness,
    applyMultiband,
    encodeSpatial,
    updateMultibandConfig,
    updateBandConfig,
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

function generateWindow(type: 'hann' | 'hamming' | 'blackman', size: number): Float32Array {
  const window = new Float32Array(size);

  for (let i = 0; i < size; i++) {
    const x = i / (size - 1);
    switch (type) {
      case 'hann':
        window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * x));
        break;
      case 'hamming':
        window[i] = 0.54 - 0.46 * Math.cos(2 * Math.PI * x);
        break;
      case 'blackman':
        window[i] = 0.42 - 0.5 * Math.cos(2 * Math.PI * x) + 0.08 * Math.cos(4 * Math.PI * x);
        break;
    }
  }

  return window;
}

function performFFT(data: Float32Array): { real: Float32Array; imag: Float32Array } {
  const n = data.length;
  const real = new Float32Array(n);
  const imag = new Float32Array(n);

  // Simple DFT (would use FFT algorithm in production)
  for (let k = 0; k < n; k++) {
    for (let t = 0; t < n; t++) {
      const angle = (2 * Math.PI * k * t) / n;
      real[k] += data[t] * Math.cos(angle);
      imag[k] -= data[t] * Math.sin(angle);
    }
  }

  return { real, imag };
}

async function applyRegionEdit(buffer: AudioBuffer, region: SpectralRegion): Promise<void> {
  // Would implement actual spectral editing
  // This is a placeholder
}

function applyKWeighting(data: Float32Array, sampleRate: number): Float32Array {
  // Simplified K-weighting filter
  // In production, would implement proper shelving filters
  return data;
}

function calculateTruePeak(data: Float32Array, sampleRate: number): number {
  // Simplified true peak (would oversample 4x in production)
  let peak = 0;
  for (let i = 0; i < data.length; i++) {
    const abs = Math.abs(data[i]);
    if (abs > peak) peak = abs;
  }
  return peak;
}

function calculateLoudnessRange(
  left: Float32Array,
  right: Float32Array,
  sampleRate: number
): number {
  // Simplified LRA calculation
  return 8; // Placeholder
}

export default useAdvancedAudio;
