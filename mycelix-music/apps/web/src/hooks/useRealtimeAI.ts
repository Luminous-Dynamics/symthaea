// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Real-time AI Processing Hook
 *
 * Live AI-powered audio processing:
 * - Real-time pitch correction (auto-tune)
 * - AI DJ with automatic mixing
 * - Voice-to-instrument conversion
 * - Live style transfer
 */

import { useState, useCallback, useRef, useEffect } from 'react';

// Types
export interface PitchCorrectionConfig {
  enabled: boolean;
  key: string;
  scale: 'major' | 'minor' | 'chromatic' | 'pentatonic';
  speed: number;        // 0-1, how fast to correct
  humanize: number;     // 0-1, add natural variation
  formantShift: number; // -12 to 12 semitones
  mix: number;          // 0-1, wet/dry
}

export interface AIDJConfig {
  enabled: boolean;
  mode: 'party' | 'chill' | 'workout' | 'focus' | 'custom';
  energyCurve: 'build' | 'maintain' | 'wave' | 'decline';
  transitionStyle: 'smooth' | 'cut' | 'echo' | 'filter';
  transitionBars: 8 | 16 | 32;
  harmonyMatch: boolean;
  bpmRange: { min: number; max: number };
  autoEffects: boolean;
}

export interface AIDJState {
  isActive: boolean;
  currentTrack: TrackInfo | null;
  nextTrack: TrackInfo | null;
  crossfadePosition: number;
  currentBPM: number;
  currentKey: string;
  energyLevel: number;
  timeToTransition: number;
  suggestedTracks: TrackInfo[];
}

export interface TrackInfo {
  id: string;
  title: string;
  artist: string;
  bpm: number;
  key: string;
  energy: number;
  genre: string;
}

export interface VoiceToInstrumentConfig {
  enabled: boolean;
  instrument: 'piano' | 'synth' | 'guitar' | 'bass' | 'strings' | 'brass' | 'flute';
  octaveShift: number;
  attack: number;
  release: number;
  velocitySensitivity: number;
  legato: boolean;
}

export interface StyleTransferConfig {
  enabled: boolean;
  style: string;          // Artist or genre style
  intensity: number;      // 0-1
  preservePitch: boolean;
  preserveTiming: boolean;
}

export interface RealtimeAIState {
  isInitialized: boolean;
  isProcessing: boolean;
  pitchCorrection: PitchCorrectionConfig;
  aiDJ: AIDJState;
  voiceToInstrument: VoiceToInstrumentConfig;
  styleTransfer: StyleTransferConfig;
  latency: number;
  cpuUsage: number;
  error: string | null;
}

// Musical scales
const SCALES: Record<string, number[]> = {
  major: [0, 2, 4, 5, 7, 9, 11],
  minor: [0, 2, 3, 5, 7, 8, 10],
  chromatic: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
  pentatonic: [0, 2, 4, 7, 9],
  blues: [0, 3, 5, 6, 7, 10],
  dorian: [0, 2, 3, 5, 7, 9, 10],
  mixolydian: [0, 2, 4, 5, 7, 9, 10],
};

// Note frequencies
const NOTE_FREQUENCIES: Record<string, number> = {
  'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
  'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
  'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88,
};

// Default configurations
const DEFAULT_PITCH_CORRECTION: PitchCorrectionConfig = {
  enabled: false,
  key: 'C',
  scale: 'major',
  speed: 0.8,
  humanize: 0.2,
  formantShift: 0,
  mix: 1.0,
};

const DEFAULT_VOICE_TO_INSTRUMENT: VoiceToInstrumentConfig = {
  enabled: false,
  instrument: 'synth',
  octaveShift: 0,
  attack: 0.01,
  release: 0.3,
  velocitySensitivity: 0.8,
  legato: true,
};

const DEFAULT_STYLE_TRANSFER: StyleTransferConfig = {
  enabled: false,
  style: '',
  intensity: 0.5,
  preservePitch: true,
  preserveTiming: true,
};

export function useRealtimeAI() {
  const [state, setState] = useState<RealtimeAIState>({
    isInitialized: false,
    isProcessing: false,
    pitchCorrection: DEFAULT_PITCH_CORRECTION,
    aiDJ: {
      isActive: false,
      currentTrack: null,
      nextTrack: null,
      crossfadePosition: 0,
      currentBPM: 120,
      currentKey: 'C',
      energyLevel: 0.5,
      timeToTransition: 0,
      suggestedTracks: [],
    },
    voiceToInstrument: DEFAULT_VOICE_TO_INSTRUMENT,
    styleTransfer: DEFAULT_STYLE_TRANSFER,
    latency: 0,
    cpuUsage: 0,
    error: null,
  });

  const audioContextRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const pitchDetectorRef = useRef<any>(null);
  const animationFrameRef = useRef<number>(0);

  /**
   * Initialize the real-time AI processing pipeline
   */
  const initialize = useCallback(async (): Promise<boolean> => {
    try {
      // Create audio context
      audioContextRef.current = new AudioContext({ latencyHint: 'interactive' });

      // Load audio worklet for low-latency processing
      await audioContextRef.current.audioWorklet.addModule('/worklets/realtime-ai.js');

      // Create worklet node
      workletNodeRef.current = new AudioWorkletNode(
        audioContextRef.current,
        'realtime-ai-processor'
      );

      // Create analyser for pitch detection
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 2048;

      setState(prev => ({ ...prev, isInitialized: true }));
      return true;
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to initialize',
      }));
      return false;
    }
  }, []);

  /**
   * Start processing audio input
   */
  const startProcessing = useCallback(async (
    inputStream?: MediaStream
  ): Promise<boolean> => {
    if (!audioContextRef.current) {
      await initialize();
    }

    try {
      const ctx = audioContextRef.current!;

      // Get microphone input if not provided
      const stream = inputStream || await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      // Create source from stream
      const source = ctx.createMediaStreamSource(stream);

      // Connect processing chain
      source.connect(analyserRef.current!);
      analyserRef.current!.connect(workletNodeRef.current!);
      workletNodeRef.current!.connect(ctx.destination);

      // Start pitch detection loop
      startPitchDetection();

      setState(prev => ({ ...prev, isProcessing: true }));
      return true;
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to start processing',
      }));
      return false;
    }
  }, [initialize]);

  /**
   * Stop processing
   */
  const stopProcessing = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    workletNodeRef.current?.disconnect();
    setState(prev => ({ ...prev, isProcessing: false }));
  }, []);

  /**
   * Start pitch detection loop
   */
  const startPitchDetection = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const bufferLength = analyser.fftSize;
    const dataArray = new Float32Array(bufferLength);

    const detect = () => {
      analyser.getFloatTimeDomainData(dataArray);

      // Autocorrelation-based pitch detection
      const pitch = detectPitch(dataArray, audioContextRef.current!.sampleRate);

      if (pitch && state.pitchCorrection.enabled) {
        // Calculate correction
        const correctedPitch = correctPitch(
          pitch,
          state.pitchCorrection.key,
          state.pitchCorrection.scale,
          state.pitchCorrection.speed
        );

        // Send to worklet
        workletNodeRef.current?.port.postMessage({
          type: 'pitchCorrection',
          currentPitch: pitch,
          targetPitch: correctedPitch,
          config: state.pitchCorrection,
        });
      }

      animationFrameRef.current = requestAnimationFrame(detect);
    };

    detect();
  }, [state.pitchCorrection]);

  /**
   * Update pitch correction settings
   */
  const setPitchCorrection = useCallback((config: Partial<PitchCorrectionConfig>) => {
    setState(prev => ({
      ...prev,
      pitchCorrection: { ...prev.pitchCorrection, ...config },
    }));

    workletNodeRef.current?.port.postMessage({
      type: 'updatePitchCorrection',
      config: { ...state.pitchCorrection, ...config },
    });
  }, [state.pitchCorrection]);

  /**
   * Start AI DJ mode
   */
  const startAIDJ = useCallback(async (config: Partial<AIDJConfig> = {}): Promise<boolean> => {
    try {
      const djConfig: AIDJConfig = {
        enabled: true,
        mode: 'party',
        energyCurve: 'wave',
        transitionStyle: 'smooth',
        transitionBars: 16,
        harmonyMatch: true,
        bpmRange: { min: 120, max: 130 },
        autoEffects: true,
        ...config,
      };

      // Get initial track suggestions
      const suggestions = await getTrackSuggestions(djConfig);

      setState(prev => ({
        ...prev,
        aiDJ: {
          ...prev.aiDJ,
          isActive: true,
          currentTrack: suggestions[0] || null,
          nextTrack: suggestions[1] || null,
          suggestedTracks: suggestions,
        },
      }));

      // Start DJ loop
      startDJLoop(djConfig);

      return true;
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to start AI DJ',
      }));
      return false;
    }
  }, []);

  /**
   * Stop AI DJ mode
   */
  const stopAIDJ = useCallback(() => {
    setState(prev => ({
      ...prev,
      aiDJ: {
        ...prev.aiDJ,
        isActive: false,
      },
    }));
  }, []);

  /**
   * Skip to next track in AI DJ
   */
  const skipTrack = useCallback(async () => {
    const { suggestedTracks, nextTrack } = state.aiDJ;

    if (nextTrack) {
      const remaining = suggestedTracks.filter(t => t.id !== nextTrack.id);
      const newNext = remaining[0] || null;

      setState(prev => ({
        ...prev,
        aiDJ: {
          ...prev.aiDJ,
          currentTrack: nextTrack,
          nextTrack: newNext,
          suggestedTracks: remaining,
          crossfadePosition: 0,
        },
      }));
    }
  }, [state.aiDJ]);

  /**
   * Start voice-to-instrument conversion
   */
  const startVoiceToInstrument = useCallback(async (
    config: Partial<VoiceToInstrumentConfig> = {}
  ): Promise<boolean> => {
    const fullConfig: VoiceToInstrumentConfig = {
      ...DEFAULT_VOICE_TO_INSTRUMENT,
      ...config,
      enabled: true,
    };

    setState(prev => ({
      ...prev,
      voiceToInstrument: fullConfig,
    }));

    workletNodeRef.current?.port.postMessage({
      type: 'startVoiceToInstrument',
      config: fullConfig,
    });

    return true;
  }, []);

  /**
   * Stop voice-to-instrument conversion
   */
  const stopVoiceToInstrument = useCallback(() => {
    setState(prev => ({
      ...prev,
      voiceToInstrument: { ...prev.voiceToInstrument, enabled: false },
    }));

    workletNodeRef.current?.port.postMessage({
      type: 'stopVoiceToInstrument',
    });
  }, []);

  /**
   * Start live style transfer
   */
  const startStyleTransfer = useCallback(async (
    style: string,
    config: Partial<StyleTransferConfig> = {}
  ): Promise<boolean> => {
    const fullConfig: StyleTransferConfig = {
      ...DEFAULT_STYLE_TRANSFER,
      ...config,
      style,
      enabled: true,
    };

    // Load style model
    await loadStyleModel(style);

    setState(prev => ({
      ...prev,
      styleTransfer: fullConfig,
    }));

    workletNodeRef.current?.port.postMessage({
      type: 'startStyleTransfer',
      config: fullConfig,
    });

    return true;
  }, []);

  /**
   * Stop style transfer
   */
  const stopStyleTransfer = useCallback(() => {
    setState(prev => ({
      ...prev,
      styleTransfer: { ...prev.styleTransfer, enabled: false },
    }));

    workletNodeRef.current?.port.postMessage({
      type: 'stopStyleTransfer',
    });
  }, []);

  /**
   * Get available styles for transfer
   */
  const getAvailableStyles = useCallback((): string[] => {
    return [
      'daft-punk',
      'deadmau5',
      'skrillex',
      'flume',
      'odesza',
      'disclosure',
      'kaytranada',
      'jamie-xx',
      'four-tet',
      'bonobo',
      'tycho',
      'moderat',
    ];
  }, []);

  /**
   * Cleanup
   */
  useEffect(() => {
    return () => {
      stopProcessing();
      audioContextRef.current?.close();
    };
  }, [stopProcessing]);

  return {
    ...state,
    initialize,
    startProcessing,
    stopProcessing,
    setPitchCorrection,
    startAIDJ,
    stopAIDJ,
    skipTrack,
    startVoiceToInstrument,
    stopVoiceToInstrument,
    startStyleTransfer,
    stopStyleTransfer,
    getAvailableStyles,
    scales: Object.keys(SCALES),
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

function detectPitch(buffer: Float32Array, sampleRate: number): number | null {
  // Autocorrelation-based pitch detection
  const SIZE = buffer.length;
  const MAX_SAMPLES = Math.floor(SIZE / 2);
  let bestOffset = -1;
  let bestCorrelation = 0;
  let foundGoodCorrelation = false;

  const correlations = new Float32Array(MAX_SAMPLES);

  for (let offset = 0; offset < MAX_SAMPLES; offset++) {
    let correlation = 0;

    for (let i = 0; i < MAX_SAMPLES; i++) {
      correlation += Math.abs(buffer[i] - buffer[i + offset]);
    }

    correlation = 1 - correlation / MAX_SAMPLES;
    correlations[offset] = correlation;

    if (correlation > 0.9 && correlation > bestCorrelation) {
      bestCorrelation = correlation;
      bestOffset = offset;
      foundGoodCorrelation = true;
    } else if (foundGoodCorrelation && correlation < 0.8) {
      break;
    }
  }

  if (bestOffset === -1 || bestCorrelation < 0.8) {
    return null;
  }

  return sampleRate / bestOffset;
}

function correctPitch(
  currentPitch: number,
  key: string,
  scale: string,
  speed: number
): number {
  const scaleNotes = SCALES[scale] || SCALES.chromatic;
  const rootFreq = NOTE_FREQUENCIES[key] || 440;

  // Find the octave
  let octave = 0;
  let testFreq = rootFreq;
  while (testFreq * 2 < currentPitch) {
    testFreq *= 2;
    octave++;
  }
  while (testFreq > currentPitch) {
    testFreq /= 2;
    octave--;
  }

  // Find closest scale note
  const ratio = currentPitch / testFreq;
  const cents = 1200 * Math.log2(ratio);
  const semitones = Math.round(cents / 100);

  // Find nearest scale degree
  let nearestDegree = 0;
  let minDistance = 12;
  for (const degree of scaleNotes) {
    const distance = Math.abs(semitones - degree);
    if (distance < minDistance) {
      minDistance = distance;
      nearestDegree = degree;
    }
  }

  // Calculate target frequency
  const targetFreq = testFreq * Math.pow(2, nearestDegree / 12);

  // Apply speed (interpolate between current and target)
  return currentPitch + (targetFreq - currentPitch) * speed;
}

async function getTrackSuggestions(config: AIDJConfig): Promise<TrackInfo[]> {
  // Simulated track suggestions
  return [
    { id: '1', title: 'Energy Build', artist: 'DJ Producer', bpm: 128, key: 'Am', energy: 0.8, genre: 'house' },
    { id: '2', title: 'Peak Time', artist: 'Beat Maker', bpm: 130, key: 'Em', energy: 0.9, genre: 'techno' },
    { id: '3', title: 'Groove Flow', artist: 'Rhythm Master', bpm: 126, key: 'Dm', energy: 0.7, genre: 'deep-house' },
    { id: '4', title: 'Night Drive', artist: 'Synth Wave', bpm: 124, key: 'Fm', energy: 0.6, genre: 'synthwave' },
  ];
}

function startDJLoop(config: AIDJConfig) {
  // Would implement actual DJ automation logic
}

async function loadStyleModel(style: string): Promise<void> {
  // Would load ML model for style transfer
  await new Promise(resolve => setTimeout(resolve, 500));
}

export default useRealtimeAI;
