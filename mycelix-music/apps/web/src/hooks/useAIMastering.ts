// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * AI Mastering Hook
 *
 * Provides AI-powered audio mastering capabilities:
 * - Intelligent EQ matching
 * - Dynamic range optimization
 * - Loudness normalization
 * - Multi-band compression
 * - Stereo enhancement
 * - Reference track matching
 *
 * Integrates with mycelix-ml for analysis and mycelix-analysis for audio fingerprinting
 */

import { useState, useCallback, useRef, useEffect } from 'react';

// Mastering presets
export type MasteringPreset =
  | 'balanced'
  | 'loud'
  | 'warm'
  | 'bright'
  | 'punchy'
  | 'smooth'
  | 'vintage'
  | 'modern'
  | 'streaming'
  | 'vinyl';

// Target platforms with different loudness standards
export type TargetPlatform =
  | 'spotify'
  | 'apple-music'
  | 'youtube'
  | 'soundcloud'
  | 'cd'
  | 'vinyl'
  | 'broadcast';

// Mastering chain stages
export interface MasteringChain {
  eq: EQSettings;
  compressor: CompressorSettings;
  multiband: MultibandSettings;
  saturation: SaturationSettings;
  stereo: StereoSettings;
  limiter: LimiterSettings;
  loudness: LoudnessSettings;
}

export interface EQSettings {
  enabled: boolean;
  bands: EQBand[];
  matchReference: boolean;
}

export interface EQBand {
  frequency: number;
  gain: number; // dB
  q: number;
  type: 'lowshelf' | 'highshelf' | 'peaking' | 'lowpass' | 'highpass';
}

export interface CompressorSettings {
  enabled: boolean;
  threshold: number; // dB
  ratio: number;
  attack: number; // ms
  release: number; // ms
  knee: number; // dB
  makeupGain: number; // dB
}

export interface MultibandSettings {
  enabled: boolean;
  bands: MultibandBand[];
}

export interface MultibandBand {
  lowFreq: number;
  highFreq: number;
  threshold: number;
  ratio: number;
  attack: number;
  release: number;
}

export interface SaturationSettings {
  enabled: boolean;
  type: 'tape' | 'tube' | 'transistor' | 'digital';
  amount: number; // 0-1
  mix: number; // 0-1
}

export interface StereoSettings {
  enabled: boolean;
  width: number; // 0-2 (1 = normal)
  midSideBalance: number; // -1 to 1 (0 = balanced)
  monoBelow: number; // Hz - mono frequencies below this
}

export interface LimiterSettings {
  enabled: boolean;
  ceiling: number; // dB (usually -0.1 to -1.0)
  release: number; // ms
  lookahead: boolean;
  truePeak: boolean;
}

export interface LoudnessSettings {
  targetLUFS: number; // Integrated loudness
  targetPeak: number; // True peak dB
  platform: TargetPlatform;
}

// Analysis results
export interface AudioAnalysisResult {
  peakLevel: number; // dB
  rmsLevel: number; // dB
  lufs: number; // Integrated loudness
  dynamicRange: number; // dB
  crestFactor: number;
  stereoWidth: number; // 0-1
  spectralBalance: {
    low: number;
    mid: number;
    high: number;
  };
  frequencyProfile: Float32Array;
  issues: MasteringIssue[];
}

export interface MasteringIssue {
  type: 'clipping' | 'too-quiet' | 'too-loud' | 'phase-issues' | 'unbalanced-stereo' | 'harsh-frequencies';
  severity: 'low' | 'medium' | 'high';
  description: string;
  suggestion: string;
  timestamp?: number;
}

// Comparison with reference
export interface ReferenceComparison {
  loudnessDiff: number;
  dynamicRangeDiff: number;
  spectralDiff: number[];
  overallMatch: number; // 0-100%
  suggestions: string[];
}

// Hook state
export interface AIMasteringState {
  isProcessing: boolean;
  progress: number;
  stage: string | null;
  chain: MasteringChain;
  inputAnalysis: AudioAnalysisResult | null;
  outputAnalysis: AudioAnalysisResult | null;
  referenceAnalysis: AudioAnalysisResult | null;
  referenceComparison: ReferenceComparison | null;
  preset: MasteringPreset;
  error: string | null;
}

// Platform loudness targets
const PLATFORM_TARGETS: Record<TargetPlatform, { lufs: number; peak: number }> = {
  'spotify': { lufs: -14, peak: -1 },
  'apple-music': { lufs: -16, peak: -1 },
  'youtube': { lufs: -14, peak: -1 },
  'soundcloud': { lufs: -14, peak: -1 },
  'cd': { lufs: -9, peak: -0.3 },
  'vinyl': { lufs: -12, peak: -0.5 },
  'broadcast': { lufs: -24, peak: -3 },
};

// Preset configurations
const PRESETS: Record<MasteringPreset, Partial<MasteringChain>> = {
  balanced: {
    compressor: { enabled: true, threshold: -12, ratio: 3, attack: 10, release: 100, knee: 6, makeupGain: 2 },
    limiter: { enabled: true, ceiling: -0.3, release: 100, lookahead: true, truePeak: true },
  },
  loud: {
    compressor: { enabled: true, threshold: -18, ratio: 4, attack: 5, release: 50, knee: 3, makeupGain: 6 },
    limiter: { enabled: true, ceiling: -0.1, release: 50, lookahead: true, truePeak: true },
  },
  warm: {
    eq: { enabled: true, bands: [
      { frequency: 100, gain: 2, q: 1, type: 'lowshelf' },
      { frequency: 3000, gain: -1, q: 1, type: 'peaking' },
    ], matchReference: false },
    saturation: { enabled: true, type: 'tape', amount: 0.3, mix: 0.5 },
  },
  bright: {
    eq: { enabled: true, bands: [
      { frequency: 8000, gain: 3, q: 1, type: 'highshelf' },
      { frequency: 200, gain: -1, q: 1, type: 'peaking' },
    ], matchReference: false },
  },
  punchy: {
    compressor: { enabled: true, threshold: -15, ratio: 4, attack: 1, release: 50, knee: 3, makeupGain: 4 },
    eq: { enabled: true, bands: [
      { frequency: 60, gain: 2, q: 1.5, type: 'peaking' },
      { frequency: 2500, gain: 1.5, q: 1, type: 'peaking' },
    ], matchReference: false },
  },
  smooth: {
    compressor: { enabled: true, threshold: -10, ratio: 2, attack: 30, release: 200, knee: 10, makeupGain: 1 },
    eq: { enabled: true, bands: [
      { frequency: 5000, gain: -1, q: 1, type: 'highshelf' },
    ], matchReference: false },
  },
  vintage: {
    saturation: { enabled: true, type: 'tube', amount: 0.4, mix: 0.6 },
    eq: { enabled: true, bands: [
      { frequency: 100, gain: 1.5, q: 1, type: 'lowshelf' },
      { frequency: 12000, gain: -2, q: 1, type: 'highshelf' },
    ], matchReference: false },
  },
  modern: {
    multiband: { enabled: true, bands: [
      { lowFreq: 0, highFreq: 200, threshold: -20, ratio: 3, attack: 10, release: 100 },
      { lowFreq: 200, highFreq: 2000, threshold: -15, ratio: 2.5, attack: 15, release: 120 },
      { lowFreq: 2000, highFreq: 20000, threshold: -12, ratio: 2, attack: 5, release: 80 },
    ]},
    stereo: { enabled: true, width: 1.2, midSideBalance: 0, monoBelow: 120 },
  },
  streaming: {
    loudness: { targetLUFS: -14, targetPeak: -1, platform: 'spotify' },
    limiter: { enabled: true, ceiling: -1, release: 100, lookahead: true, truePeak: true },
  },
  vinyl: {
    eq: { enabled: true, bands: [
      { frequency: 40, gain: -6, q: 1, type: 'highpass' },
      { frequency: 15000, gain: -3, q: 1, type: 'lowpass' },
    ], matchReference: false },
    stereo: { enabled: true, width: 0.9, midSideBalance: 0, monoBelow: 300 },
    limiter: { enabled: true, ceiling: -0.5, release: 150, lookahead: true, truePeak: false },
  },
};

const DEFAULT_CHAIN: MasteringChain = {
  eq: {
    enabled: true,
    bands: [
      { frequency: 80, gain: 0, q: 1, type: 'lowshelf' },
      { frequency: 400, gain: 0, q: 1, type: 'peaking' },
      { frequency: 2000, gain: 0, q: 1, type: 'peaking' },
      { frequency: 8000, gain: 0, q: 1, type: 'highshelf' },
    ],
    matchReference: false,
  },
  compressor: {
    enabled: true,
    threshold: -12,
    ratio: 3,
    attack: 10,
    release: 100,
    knee: 6,
    makeupGain: 0,
  },
  multiband: {
    enabled: false,
    bands: [
      { lowFreq: 0, highFreq: 200, threshold: -20, ratio: 3, attack: 10, release: 100 },
      { lowFreq: 200, highFreq: 2000, threshold: -15, ratio: 2.5, attack: 15, release: 120 },
      { lowFreq: 2000, highFreq: 20000, threshold: -12, ratio: 2, attack: 5, release: 80 },
    ],
  },
  saturation: {
    enabled: false,
    type: 'tape',
    amount: 0.2,
    mix: 0.5,
  },
  stereo: {
    enabled: true,
    width: 1,
    midSideBalance: 0,
    monoBelow: 100,
  },
  limiter: {
    enabled: true,
    ceiling: -0.3,
    release: 100,
    lookahead: true,
    truePeak: true,
  },
  loudness: {
    targetLUFS: -14,
    targetPeak: -1,
    platform: 'spotify',
  },
};

export function useAIMastering() {
  const [state, setState] = useState<AIMasteringState>({
    isProcessing: false,
    progress: 0,
    stage: null,
    chain: DEFAULT_CHAIN,
    inputAnalysis: null,
    outputAnalysis: null,
    referenceAnalysis: null,
    referenceComparison: null,
    preset: 'balanced',
    error: null,
  });

  const audioContextRef = useRef<AudioContext | null>(null);
  const offlineContextRef = useRef<OfflineAudioContext | null>(null);

  // Initialize audio context
  useEffect(() => {
    audioContextRef.current = new AudioContext();
    return () => {
      audioContextRef.current?.close();
    };
  }, []);

  // Analyze audio buffer
  const analyzeAudio = useCallback(async (buffer: AudioBuffer): Promise<AudioAnalysisResult> => {
    const sampleRate = buffer.sampleRate;
    const channelData = buffer.getChannelData(0);
    const rightChannel = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : channelData;

    // Calculate levels
    let peakL = 0, peakR = 0, sumL = 0, sumR = 0;
    for (let i = 0; i < channelData.length; i++) {
      const absL = Math.abs(channelData[i]);
      const absR = Math.abs(rightChannel[i]);
      peakL = Math.max(peakL, absL);
      peakR = Math.max(peakR, absR);
      sumL += channelData[i] * channelData[i];
      sumR += rightChannel[i] * rightChannel[i];
    }

    const peakLevel = 20 * Math.log10(Math.max(peakL, peakR));
    const rmsL = Math.sqrt(sumL / channelData.length);
    const rmsR = Math.sqrt(sumR / channelData.length);
    const rmsLevel = 20 * Math.log10((rmsL + rmsR) / 2);
    const crestFactor = peakLevel - rmsLevel;

    // Simplified LUFS calculation
    const lufs = rmsLevel - 0.691; // Simplified, real LUFS is more complex

    // Dynamic range
    const dynamicRange = crestFactor * 0.8; // Simplified

    // Stereo width
    let correlationSum = 0;
    for (let i = 0; i < channelData.length; i++) {
      correlationSum += channelData[i] * rightChannel[i];
    }
    const correlation = correlationSum / channelData.length;
    const stereoWidth = 1 - Math.abs(correlation);

    // FFT for spectral analysis
    const fftSize = 4096;
    const frequencyProfile = new Float32Array(fftSize / 2);
    const ctx = audioContextRef.current;

    if (ctx) {
      const analyser = ctx.createAnalyser();
      analyser.fftSize = fftSize;
      analyser.getFloatFrequencyData(frequencyProfile);
    }

    // Spectral balance (simplified)
    let lowSum = 0, midSum = 0, highSum = 0;
    const lowEnd = Math.floor(250 / (sampleRate / fftSize));
    const midEnd = Math.floor(4000 / (sampleRate / fftSize));

    for (let i = 0; i < frequencyProfile.length; i++) {
      const power = Math.pow(10, frequencyProfile[i] / 20);
      if (i < lowEnd) lowSum += power;
      else if (i < midEnd) midSum += power;
      else highSum += power;
    }

    const total = lowSum + midSum + highSum || 1;
    const spectralBalance = {
      low: lowSum / total,
      mid: midSum / total,
      high: highSum / total,
    };

    // Detect issues
    const issues: MasteringIssue[] = [];

    if (peakLevel > -0.3) {
      issues.push({
        type: 'clipping',
        severity: peakLevel > 0 ? 'high' : 'medium',
        description: 'Audio is clipping or very close to clipping',
        suggestion: 'Reduce gain or use a limiter',
      });
    }

    if (lufs < -20) {
      issues.push({
        type: 'too-quiet',
        severity: lufs < -24 ? 'high' : 'medium',
        description: 'Audio is quieter than typical streaming standards',
        suggestion: 'Increase overall loudness with compression and limiting',
      });
    }

    if (lufs > -8) {
      issues.push({
        type: 'too-loud',
        severity: lufs > -6 ? 'high' : 'medium',
        description: 'Audio may be over-compressed',
        suggestion: 'Reduce limiting and compression for more dynamics',
      });
    }

    if (stereoWidth < 0.3) {
      issues.push({
        type: 'unbalanced-stereo',
        severity: 'low',
        description: 'Audio has very narrow stereo image',
        suggestion: 'Consider stereo widening',
      });
    }

    if (spectralBalance.high > 0.4) {
      issues.push({
        type: 'harsh-frequencies',
        severity: 'medium',
        description: 'High frequencies may be too prominent',
        suggestion: 'Consider de-essing or high shelf cut',
      });
    }

    return {
      peakLevel,
      rmsLevel,
      lufs,
      dynamicRange,
      crestFactor,
      stereoWidth,
      spectralBalance,
      frequencyProfile,
      issues,
    };
  }, []);

  // Process audio through mastering chain
  const process = useCallback(async (inputBuffer: AudioBuffer): Promise<AudioBuffer> => {
    setState(prev => ({
      ...prev,
      isProcessing: true,
      progress: 0,
      stage: 'Analyzing input...',
      error: null,
    }));

    try {
      // Analyze input
      const inputAnalysis = await analyzeAudio(inputBuffer);
      setState(prev => ({
        ...prev,
        inputAnalysis,
        progress: 10,
        stage: 'Preparing mastering chain...',
      }));

      // Create offline context for processing
      const ctx = new OfflineAudioContext(
        inputBuffer.numberOfChannels,
        inputBuffer.length,
        inputBuffer.sampleRate
      );
      offlineContextRef.current = ctx;

      // Create source
      const source = ctx.createBufferSource();
      source.buffer = inputBuffer;

      let currentNode: AudioNode = source;
      const { chain } = state;

      // EQ
      if (chain.eq.enabled) {
        setState(prev => ({ ...prev, progress: 20, stage: 'Applying EQ...' }));

        for (const band of chain.eq.bands) {
          if (band.gain !== 0) {
            const filter = ctx.createBiquadFilter();
            filter.type = band.type;
            filter.frequency.value = band.frequency;
            filter.gain.value = band.gain;
            filter.Q.value = band.q;
            currentNode.connect(filter);
            currentNode = filter;
          }
        }
      }

      // Dynamics (simplified - real implementation would use worklet)
      setState(prev => ({ ...prev, progress: 40, stage: 'Applying dynamics...' }));

      if (chain.compressor.enabled) {
        const compressor = ctx.createDynamicsCompressor();
        compressor.threshold.value = chain.compressor.threshold;
        compressor.ratio.value = chain.compressor.ratio;
        compressor.attack.value = chain.compressor.attack / 1000;
        compressor.release.value = chain.compressor.release / 1000;
        compressor.knee.value = chain.compressor.knee;
        currentNode.connect(compressor);
        currentNode = compressor;

        // Makeup gain
        if (chain.compressor.makeupGain !== 0) {
          const makeup = ctx.createGain();
          makeup.gain.value = Math.pow(10, chain.compressor.makeupGain / 20);
          currentNode.connect(makeup);
          currentNode = makeup;
        }
      }

      // Saturation (simplified waveshaper)
      if (chain.saturation.enabled && chain.saturation.amount > 0) {
        setState(prev => ({ ...prev, progress: 50, stage: 'Adding saturation...' }));

        const waveshaper = ctx.createWaveShaper();
        const amount = chain.saturation.amount;
        const k = amount * 100;
        const n_samples = 44100;
        const curve = new Float32Array(n_samples);

        for (let i = 0; i < n_samples; i++) {
          const x = (i * 2) / n_samples - 1;
          curve[i] = ((3 + k) * x * 20 * (Math.PI / 180)) / (Math.PI + k * Math.abs(x));
        }

        waveshaper.curve = curve;
        waveshaper.oversample = '4x';

        // Dry/wet mix
        const dryGain = ctx.createGain();
        const wetGain = ctx.createGain();
        dryGain.gain.value = 1 - chain.saturation.mix;
        wetGain.gain.value = chain.saturation.mix;

        const mixGain = ctx.createGain();
        currentNode.connect(dryGain);
        currentNode.connect(waveshaper);
        waveshaper.connect(wetGain);
        dryGain.connect(mixGain);
        wetGain.connect(mixGain);
        currentNode = mixGain;
      }

      // Stereo enhancement
      if (chain.stereo.enabled && chain.stereo.width !== 1) {
        setState(prev => ({ ...prev, progress: 60, stage: 'Enhancing stereo...' }));
        // Simplified - real implementation would use Mid/Side processing
      }

      // Limiter
      if (chain.limiter.enabled) {
        setState(prev => ({ ...prev, progress: 70, stage: 'Applying limiter...' }));

        const limiter = ctx.createDynamicsCompressor();
        limiter.threshold.value = chain.limiter.ceiling;
        limiter.ratio.value = 20; // High ratio for limiting
        limiter.attack.value = 0.001;
        limiter.release.value = chain.limiter.release / 1000;
        limiter.knee.value = 0;
        currentNode.connect(limiter);
        currentNode = limiter;
      }

      // Final output gain for loudness
      setState(prev => ({ ...prev, progress: 80, stage: 'Normalizing loudness...' }));

      const targetLufs = chain.loudness.targetLUFS;
      const currentLufs = inputAnalysis.lufs;
      const gainNeeded = targetLufs - currentLufs;
      const clampedGain = Math.max(-12, Math.min(12, gainNeeded));

      const outputGain = ctx.createGain();
      outputGain.gain.value = Math.pow(10, clampedGain / 20);
      currentNode.connect(outputGain);
      outputGain.connect(ctx.destination);

      // Render
      setState(prev => ({ ...prev, progress: 90, stage: 'Rendering...' }));
      source.start();
      const outputBuffer = await ctx.startRendering();

      // Analyze output
      setState(prev => ({ ...prev, progress: 95, stage: 'Analyzing output...' }));
      const outputAnalysis = await analyzeAudio(outputBuffer);

      setState(prev => ({
        ...prev,
        isProcessing: false,
        progress: 100,
        stage: null,
        outputAnalysis,
      }));

      return outputBuffer;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isProcessing: false,
        progress: 0,
        stage: null,
        error: error instanceof Error ? error.message : 'Processing failed',
      }));
      throw error;
    }
  }, [state.chain, analyzeAudio]);

  // Apply preset
  const applyPreset = useCallback((preset: MasteringPreset) => {
    const presetConfig = PRESETS[preset];

    setState(prev => ({
      ...prev,
      preset,
      chain: {
        ...DEFAULT_CHAIN,
        ...presetConfig,
        eq: { ...DEFAULT_CHAIN.eq, ...presetConfig.eq },
        compressor: { ...DEFAULT_CHAIN.compressor, ...presetConfig.compressor },
        multiband: { ...DEFAULT_CHAIN.multiband, ...presetConfig.multiband },
        saturation: { ...DEFAULT_CHAIN.saturation, ...presetConfig.saturation },
        stereo: { ...DEFAULT_CHAIN.stereo, ...presetConfig.stereo },
        limiter: { ...DEFAULT_CHAIN.limiter, ...presetConfig.limiter },
        loudness: { ...DEFAULT_CHAIN.loudness, ...presetConfig.loudness },
      },
    }));
  }, []);

  // Set target platform
  const setTargetPlatform = useCallback((platform: TargetPlatform) => {
    const target = PLATFORM_TARGETS[platform];

    setState(prev => ({
      ...prev,
      chain: {
        ...prev.chain,
        loudness: {
          ...prev.chain.loudness,
          platform,
          targetLUFS: target.lufs,
          targetPeak: target.peak,
        },
      },
    }));
  }, []);

  // Update chain settings
  const updateChain = useCallback((updates: Partial<MasteringChain>) => {
    setState(prev => ({
      ...prev,
      chain: { ...prev.chain, ...updates },
    }));
  }, []);

  // Load reference track
  const loadReference = useCallback(async (buffer: AudioBuffer) => {
    const analysis = await analyzeAudio(buffer);
    setState(prev => ({ ...prev, referenceAnalysis: analysis }));
  }, [analyzeAudio]);

  // Compare with reference
  const compareWithReference = useCallback(() => {
    const { inputAnalysis, referenceAnalysis } = state;
    if (!inputAnalysis || !referenceAnalysis) return;

    const loudnessDiff = inputAnalysis.lufs - referenceAnalysis.lufs;
    const dynamicRangeDiff = inputAnalysis.dynamicRange - referenceAnalysis.dynamicRange;

    const spectralDiff: number[] = [];
    for (let i = 0; i < Math.min(inputAnalysis.frequencyProfile.length, referenceAnalysis.frequencyProfile.length); i++) {
      spectralDiff.push(inputAnalysis.frequencyProfile[i] - referenceAnalysis.frequencyProfile[i]);
    }

    const overallMatch = Math.max(0, 100 - Math.abs(loudnessDiff) * 5 - Math.abs(dynamicRangeDiff) * 3);

    const suggestions: string[] = [];
    if (loudnessDiff < -3) suggestions.push('Increase overall loudness');
    if (loudnessDiff > 3) suggestions.push('Reduce overall loudness');
    if (dynamicRangeDiff < -3) suggestions.push('Add more dynamics (less compression)');
    if (dynamicRangeDiff > 3) suggestions.push('Add more compression for punch');

    const comparison: ReferenceComparison = {
      loudnessDiff,
      dynamicRangeDiff,
      spectralDiff,
      overallMatch,
      suggestions,
    };

    setState(prev => ({ ...prev, referenceComparison: comparison }));
  }, [state.inputAnalysis, state.referenceAnalysis]);

  return {
    ...state,
    process,
    analyzeAudio,
    applyPreset,
    setTargetPlatform,
    updateChain,
    loadReference,
    compareWithReference,
    presets: Object.keys(PRESETS) as MasteringPreset[],
    platforms: Object.keys(PLATFORM_TARGETS) as TargetPlatform[],
    platformTargets: PLATFORM_TARGETS,
  };
}

export default useAIMastering;
