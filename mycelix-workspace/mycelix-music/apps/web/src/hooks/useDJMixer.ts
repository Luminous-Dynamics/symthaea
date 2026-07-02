// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * useDJMixer Hook
 *
 * Professional DJ mixing capabilities:
 * - Dual deck control
 * - Crossfader with curve options
 * - Beatmatching & sync
 * - Cue points & loops
 * - Seamless transitions
 */

import { useCallback, useEffect, useRef, useState } from 'react';

// === Types ===

export interface CuePoint {
  id: string;
  position: number; // seconds
  color: string;
  label?: string;
}

export interface Loop {
  start: number;
  end: number;
  active: boolean;
}

export interface DeckState {
  trackId: string | null;
  trackTitle: string;
  trackArtist: string;
  duration: number;
  position: number;
  bpm: number;
  originalBpm: number;
  pitch: number; // -100 to +100 percent
  isPlaying: boolean;
  isLoaded: boolean;
  waveformPeaks: number[];
  cuePoints: CuePoint[];
  loop: Loop | null;
  volume: number;
  eqLow: number;
  eqMid: number;
  eqHigh: number;
  filter: number; // -1 (lowpass) to 1 (highpass), 0 = off
}

export type CrossfaderCurve = 'linear' | 'constant' | 'smooth' | 'scratch';

export interface MixerState {
  crossfader: number; // 0 = deck A, 0.5 = center, 1 = deck B
  crossfaderCurve: CrossfaderCurve;
  masterVolume: number;
  headphoneMix: number; // 0 = cue, 1 = master
  deckACue: boolean;
  deckBCue: boolean;
  isSyncEnabled: boolean;
  syncSource: 'A' | 'B' | 'external';
}

export interface BeatGridInfo {
  bpm: number;
  firstBeatOffset: number;
  beatsPerBar: number;
}

// === Constants ===

const DEFAULT_DECK_STATE: DeckState = {
  trackId: null,
  trackTitle: '',
  trackArtist: '',
  duration: 0,
  position: 0,
  bpm: 120,
  originalBpm: 120,
  pitch: 0,
  isPlaying: false,
  isLoaded: false,
  waveformPeaks: [],
  cuePoints: [],
  loop: null,
  volume: 1,
  eqLow: 0,
  eqMid: 0,
  eqHigh: 0,
  filter: 0,
};

const CUE_COLORS = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4', '#3b82f6', '#8b5cf6', '#ec4899'];

// === Crossfader Curves ===

function applyCrossfaderCurve(position: number, curve: CrossfaderCurve): { deckA: number; deckB: number } {
  switch (curve) {
    case 'linear':
      return { deckA: 1 - position, deckB: position };

    case 'constant':
      // Constant power (equal loudness)
      return {
        deckA: Math.cos(position * Math.PI / 2),
        deckB: Math.sin(position * Math.PI / 2),
      };

    case 'smooth':
      // S-curve for smooth transitions
      const smooth = position * position * (3 - 2 * position);
      return { deckA: 1 - smooth, deckB: smooth };

    case 'scratch':
      // Sharp cut (DJ scratch style)
      if (position < 0.1) return { deckA: 1, deckB: 0 };
      if (position > 0.9) return { deckA: 0, deckB: 1 };
      const mid = (position - 0.1) / 0.8;
      return { deckA: 1 - mid, deckB: mid };

    default:
      return { deckA: 1 - position, deckB: position };
  }
}

// === BPM Detection ===

function detectBPM(peaks: number[], sampleRate: number = 44100, hopSize: number = 512): number {
  if (peaks.length < 100) return 120;

  // Simple onset detection
  const onsets: number[] = [];
  for (let i = 1; i < peaks.length; i++) {
    if (peaks[i] > peaks[i - 1] * 1.5 && peaks[i] > 0.3) {
      onsets.push(i);
    }
  }

  if (onsets.length < 4) return 120;

  // Calculate inter-onset intervals
  const intervals: number[] = [];
  for (let i = 1; i < onsets.length; i++) {
    intervals.push(onsets[i] - onsets[i - 1]);
  }

  // Find most common interval
  const counts = new Map<number, number>();
  for (const interval of intervals) {
    const quantized = Math.round(interval / 2) * 2;
    counts.set(quantized, (counts.get(quantized) || 0) + 1);
  }

  let bestInterval = 20;
  let bestCount = 0;
  counts.forEach((count, interval) => {
    if (count > bestCount) {
      bestCount = count;
      bestInterval = interval;
    }
  });

  const secondsPerBeat = (bestInterval * hopSize) / sampleRate;
  const bpm = 60 / secondsPerBeat;

  // Normalize to reasonable range
  let normalizedBpm = bpm;
  while (normalizedBpm < 70) normalizedBpm *= 2;
  while (normalizedBpm > 180) normalizedBpm /= 2;

  return Math.round(normalizedBpm * 10) / 10;
}

// === Beatmatching ===

function calculatePitchForSync(sourceBpm: number, targetBpm: number): number {
  if (sourceBpm === 0 || targetBpm === 0) return 0;
  const ratio = targetBpm / sourceBpm;
  return (ratio - 1) * 100;
}

function quantizeToNearestBeat(position: number, bpm: number, duration: number): number {
  const beatDuration = 60 / bpm;
  const beatNumber = Math.round(position / beatDuration);
  return Math.min(beatNumber * beatDuration, duration);
}

// === Hook ===

export interface UseDJMixerReturn {
  deckA: DeckState;
  deckB: DeckState;
  mixer: MixerState;
  audioContextRef: React.RefObject<AudioContext | null>;

  // Deck loading
  loadTrack: (deck: 'A' | 'B', trackId: string, audioUrl: string, metadata: {
    title: string;
    artist: string;
    bpm?: number;
    waveformPeaks?: number[];
  }) => Promise<void>;
  ejectTrack: (deck: 'A' | 'B') => void;

  // Playback control
  play: (deck: 'A' | 'B') => void;
  pause: (deck: 'A' | 'B') => void;
  stop: (deck: 'A' | 'B') => void;
  seek: (deck: 'A' | 'B', position: number) => void;

  // Tempo control
  setPitch: (deck: 'A' | 'B', pitch: number) => void;
  nudge: (deck: 'A' | 'B', direction: 'faster' | 'slower') => void;
  syncBpm: (deck: 'A' | 'B') => void;
  syncPhase: (deck: 'A' | 'B') => void;

  // Cue points
  setCuePoint: (deck: 'A' | 'B', index: number) => void;
  jumpToCue: (deck: 'A' | 'B', index: number) => void;
  deleteCuePoint: (deck: 'A' | 'B', index: number) => void;

  // Looping
  setLoop: (deck: 'A' | 'B', bars: number) => void;
  toggleLoop: (deck: 'A' | 'B') => void;
  clearLoop: (deck: 'A' | 'B') => void;
  moveLoop: (deck: 'A' | 'B', direction: 'forward' | 'backward') => void;
  halveLoop: (deck: 'A' | 'B') => void;
  doubleLoop: (deck: 'A' | 'B') => void;

  // EQ & Filter
  setEQ: (deck: 'A' | 'B', band: 'low' | 'mid' | 'high', value: number) => void;
  setFilter: (deck: 'A' | 'B', value: number) => void;
  killEQ: (deck: 'A' | 'B', band: 'low' | 'mid' | 'high') => void;

  // Volume
  setDeckVolume: (deck: 'A' | 'B', volume: number) => void;

  // Mixer
  setCrossfader: (position: number) => void;
  setCrossfaderCurve: (curve: CrossfaderCurve) => void;
  setMasterVolume: (volume: number) => void;
  setHeadphoneMix: (mix: number) => void;
  toggleCue: (deck: 'A' | 'B') => void;
  toggleSync: () => void;
  setSyncSource: (source: 'A' | 'B' | 'external') => void;

  // Utilities
  getBeatPosition: (deck: 'A' | 'B') => number;
  getPhaseOffset: (deck: 'A' | 'B') => number;
  getMixedOutput: () => { left: number; right: number };
}

export function useDJMixer(): UseDJMixerReturn {
  const audioContextRef = useRef<AudioContext | null>(null);
  const deckASourceRef = useRef<AudioBufferSourceNode | null>(null);
  const deckBSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const deckABufferRef = useRef<AudioBuffer | null>(null);
  const deckBBufferRef = useRef<AudioBuffer | null>(null);
  const deckAGainRef = useRef<GainNode | null>(null);
  const deckBGainRef = useRef<GainNode | null>(null);
  const masterGainRef = useRef<GainNode | null>(null);
  const animationFrameRef = useRef<number>(0);
  const startTimeRef = useRef<{ A: number; B: number }>({ A: 0, B: 0 });
  const pausePositionRef = useRef<{ A: number; B: number }>({ A: 0, B: 0 });

  const [deckA, setDeckA] = useState<DeckState>(DEFAULT_DECK_STATE);
  const [deckB, setDeckB] = useState<DeckState>(DEFAULT_DECK_STATE);
  const [mixer, setMixer] = useState<MixerState>({
    crossfader: 0.5,
    crossfaderCurve: 'constant',
    masterVolume: 1,
    headphoneMix: 0.5,
    deckACue: false,
    deckBCue: false,
    isSyncEnabled: false,
    syncSource: 'A',
  });

  // Initialize audio context
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  const initAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext();
      deckAGainRef.current = audioContextRef.current.createGain();
      deckBGainRef.current = audioContextRef.current.createGain();
      masterGainRef.current = audioContextRef.current.createGain();

      deckAGainRef.current.connect(masterGainRef.current);
      deckBGainRef.current.connect(masterGainRef.current);
      masterGainRef.current.connect(audioContextRef.current.destination);
    }
    return audioContextRef.current;
  }, []);

  // Position update loop
  useEffect(() => {
    const updatePositions = () => {
      const ctx = audioContextRef.current;
      if (!ctx) {
        animationFrameRef.current = requestAnimationFrame(updatePositions);
        return;
      }

      if (deckA.isPlaying && deckA.isLoaded) {
        const elapsed = ctx.currentTime - startTimeRef.current.A;
        const newPosition = pausePositionRef.current.A + elapsed * (1 + deckA.pitch / 100);

        // Handle looping
        if (deckA.loop?.active) {
          if (newPosition >= deckA.loop.end) {
            const loopDuration = deckA.loop.end - deckA.loop.start;
            const overflow = newPosition - deckA.loop.end;
            const loopPosition = deckA.loop.start + (overflow % loopDuration);
            setDeckA((s) => ({ ...s, position: loopPosition }));
          } else {
            setDeckA((s) => ({ ...s, position: newPosition }));
          }
        } else {
          setDeckA((s) => ({ ...s, position: Math.min(newPosition, s.duration) }));
        }
      }

      if (deckB.isPlaying && deckB.isLoaded) {
        const elapsed = ctx.currentTime - startTimeRef.current.B;
        const newPosition = pausePositionRef.current.B + elapsed * (1 + deckB.pitch / 100);

        if (deckB.loop?.active) {
          if (newPosition >= deckB.loop.end) {
            const loopDuration = deckB.loop.end - deckB.loop.start;
            const overflow = newPosition - deckB.loop.end;
            const loopPosition = deckB.loop.start + (overflow % loopDuration);
            setDeckB((s) => ({ ...s, position: loopPosition }));
          } else {
            setDeckB((s) => ({ ...s, position: newPosition }));
          }
        } else {
          setDeckB((s) => ({ ...s, position: Math.min(newPosition, s.duration) }));
        }
      }

      animationFrameRef.current = requestAnimationFrame(updatePositions);
    };

    animationFrameRef.current = requestAnimationFrame(updatePositions);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [deckA.isPlaying, deckA.isLoaded, deckA.pitch, deckA.loop, deckB.isPlaying, deckB.isLoaded, deckB.pitch, deckB.loop]);

  // Apply crossfader
  useEffect(() => {
    const { deckA: gainA, deckB: gainB } = applyCrossfaderCurve(mixer.crossfader, mixer.crossfaderCurve);

    if (deckAGainRef.current) {
      deckAGainRef.current.gain.value = gainA * deckA.volume;
    }
    if (deckBGainRef.current) {
      deckBGainRef.current.gain.value = gainB * deckB.volume;
    }
    if (masterGainRef.current) {
      masterGainRef.current.gain.value = mixer.masterVolume;
    }
  }, [mixer.crossfader, mixer.crossfaderCurve, mixer.masterVolume, deckA.volume, deckB.volume]);

  // === Deck Loading ===

  const loadTrack = useCallback(async (
    deck: 'A' | 'B',
    trackId: string,
    audioUrl: string,
    metadata: { title: string; artist: string; bpm?: number; waveformPeaks?: number[] }
  ) => {
    const ctx = initAudioContext();

    // Fetch and decode audio
    const response = await fetch(audioUrl);
    const arrayBuffer = await response.arrayBuffer();
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);

    // Calculate BPM if not provided
    let bpm = metadata.bpm || 120;
    if (!metadata.bpm && metadata.waveformPeaks) {
      bpm = detectBPM(metadata.waveformPeaks);
    }

    const newState: Partial<DeckState> = {
      trackId,
      trackTitle: metadata.title,
      trackArtist: metadata.artist,
      duration: audioBuffer.duration,
      position: 0,
      bpm,
      originalBpm: bpm,
      isLoaded: true,
      isPlaying: false,
      waveformPeaks: metadata.waveformPeaks || [],
      pitch: 0,
    };

    if (deck === 'A') {
      deckABufferRef.current = audioBuffer;
      setDeckA((s) => ({ ...s, ...newState }));
      pausePositionRef.current.A = 0;
    } else {
      deckBBufferRef.current = audioBuffer;
      setDeckB((s) => ({ ...s, ...newState }));
      pausePositionRef.current.B = 0;
    }
  }, [initAudioContext]);

  const ejectTrack = useCallback((deck: 'A' | 'B') => {
    if (deck === 'A') {
      deckASourceRef.current?.stop();
      deckASourceRef.current = null;
      deckABufferRef.current = null;
      setDeckA(DEFAULT_DECK_STATE);
    } else {
      deckBSourceRef.current?.stop();
      deckBSourceRef.current = null;
      deckBBufferRef.current = null;
      setDeckB(DEFAULT_DECK_STATE);
    }
  }, []);

  // === Playback ===

  const play = useCallback((deck: 'A' | 'B') => {
    const ctx = audioContextRef.current;
    if (!ctx) return;

    const buffer = deck === 'A' ? deckABufferRef.current : deckBBufferRef.current;
    const gainNode = deck === 'A' ? deckAGainRef.current : deckBGainRef.current;
    if (!buffer || !gainNode) return;

    // Stop existing source
    const existingSource = deck === 'A' ? deckASourceRef.current : deckBSourceRef.current;
    if (existingSource) {
      existingSource.stop();
    }

    // Create new source
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(gainNode);

    const deckState = deck === 'A' ? deckA : deckB;
    source.playbackRate.value = 1 + deckState.pitch / 100;

    const position = pausePositionRef.current[deck];
    source.start(0, position);

    startTimeRef.current[deck] = ctx.currentTime;

    if (deck === 'A') {
      deckASourceRef.current = source;
      setDeckA((s) => ({ ...s, isPlaying: true }));
    } else {
      deckBSourceRef.current = source;
      setDeckB((s) => ({ ...s, isPlaying: true }));
    }
  }, [deckA.pitch, deckB.pitch]);

  const pause = useCallback((deck: 'A' | 'B') => {
    const ctx = audioContextRef.current;
    const source = deck === 'A' ? deckASourceRef.current : deckBSourceRef.current;

    if (ctx && source) {
      const elapsed = ctx.currentTime - startTimeRef.current[deck];
      const deckState = deck === 'A' ? deckA : deckB;
      const newPosition = pausePositionRef.current[deck] + elapsed * (1 + deckState.pitch / 100);

      pausePositionRef.current[deck] = newPosition;
      source.stop();
    }

    if (deck === 'A') {
      deckASourceRef.current = null;
      setDeckA((s) => ({ ...s, isPlaying: false }));
    } else {
      deckBSourceRef.current = null;
      setDeckB((s) => ({ ...s, isPlaying: false }));
    }
  }, [deckA.pitch, deckB.pitch]);

  const stop = useCallback((deck: 'A' | 'B') => {
    pause(deck);
    pausePositionRef.current[deck] = 0;

    if (deck === 'A') {
      setDeckA((s) => ({ ...s, position: 0 }));
    } else {
      setDeckB((s) => ({ ...s, position: 0 }));
    }
  }, [pause]);

  const seek = useCallback((deck: 'A' | 'B', position: number) => {
    const deckState = deck === 'A' ? deckA : deckB;
    const wasPlaying = deckState.isPlaying;

    if (wasPlaying) {
      pause(deck);
    }

    pausePositionRef.current[deck] = position;

    if (deck === 'A') {
      setDeckA((s) => ({ ...s, position }));
    } else {
      setDeckB((s) => ({ ...s, position }));
    }

    if (wasPlaying) {
      play(deck);
    }
  }, [deckA.isPlaying, deckB.isPlaying, pause, play]);

  // === Tempo Control ===

  const setPitch = useCallback((deck: 'A' | 'B', pitch: number) => {
    const clampedPitch = Math.max(-50, Math.min(50, pitch));
    const setDeck = deck === 'A' ? setDeckA : setDeckB;

    setDeck((s) => ({
      ...s,
      pitch: clampedPitch,
      bpm: s.originalBpm * (1 + clampedPitch / 100),
    }));

    const source = deck === 'A' ? deckASourceRef.current : deckBSourceRef.current;
    if (source) {
      source.playbackRate.value = 1 + clampedPitch / 100;
    }
  }, []);

  const nudge = useCallback((deck: 'A' | 'B', direction: 'faster' | 'slower') => {
    const delta = direction === 'faster' ? 0.5 : -0.5;
    const deckState = deck === 'A' ? deckA : deckB;
    setPitch(deck, deckState.pitch + delta);

    // Auto-restore after brief nudge
    setTimeout(() => {
      setPitch(deck, deckState.pitch);
    }, 500);
  }, [deckA.pitch, deckB.pitch, setPitch]);

  const syncBpm = useCallback((deck: 'A' | 'B') => {
    const source = deck === 'A' ? deckB : deckA;
    const target = deck === 'A' ? deckA : deckB;

    if (!source.isLoaded || !target.isLoaded) return;

    const pitchAdjust = calculatePitchForSync(target.originalBpm, source.bpm);
    setPitch(deck, pitchAdjust);
  }, [deckA, deckB, setPitch]);

  const syncPhase = useCallback((deck: 'A' | 'B') => {
    const source = deck === 'A' ? deckB : deckA;
    const target = deck === 'A' ? deckA : deckB;

    if (!source.isLoaded || !target.isLoaded) return;

    // Find nearest beat in source
    const sourceBeatDuration = 60 / source.bpm;
    const sourceBeatPosition = source.position % sourceBeatDuration;

    // Adjust target position to match phase
    const targetBeatDuration = 60 / target.bpm;
    const targetBeatNumber = Math.floor(target.position / targetBeatDuration);
    const newPosition = targetBeatNumber * targetBeatDuration + sourceBeatPosition;

    seek(deck, newPosition);
  }, [deckA, deckB, seek]);

  // === Cue Points ===

  const setCuePoint = useCallback((deck: 'A' | 'B', index: number) => {
    const deckState = deck === 'A' ? deckA : deckB;
    const setDeck = deck === 'A' ? setDeckA : setDeckB;

    const newCue: CuePoint = {
      id: `cue-${index}`,
      position: deckState.position,
      color: CUE_COLORS[index % CUE_COLORS.length],
      label: `Cue ${index + 1}`,
    };

    setDeck((s) => {
      const cuePoints = [...s.cuePoints];
      cuePoints[index] = newCue;
      return { ...s, cuePoints };
    });
  }, [deckA.position, deckB.position]);

  const jumpToCue = useCallback((deck: 'A' | 'B', index: number) => {
    const deckState = deck === 'A' ? deckA : deckB;
    const cue = deckState.cuePoints[index];
    if (cue) {
      seek(deck, cue.position);
    }
  }, [deckA.cuePoints, deckB.cuePoints, seek]);

  const deleteCuePoint = useCallback((deck: 'A' | 'B', index: number) => {
    const setDeck = deck === 'A' ? setDeckA : setDeckB;
    setDeck((s) => {
      const cuePoints = [...s.cuePoints];
      cuePoints.splice(index, 1);
      return { ...s, cuePoints };
    });
  }, []);

  // === Looping ===

  const setLoop = useCallback((deck: 'A' | 'B', bars: number) => {
    const deckState = deck === 'A' ? deckA : deckB;
    const setDeck = deck === 'A' ? setDeckA : setDeckB;

    const beatsPerBar = 4;
    const beatDuration = 60 / deckState.bpm;
    const loopDuration = bars * beatsPerBar * beatDuration;

    // Quantize start to nearest beat
    const startBeat = Math.floor(deckState.position / beatDuration);
    const start = startBeat * beatDuration;
    const end = start + loopDuration;

    setDeck((s) => ({
      ...s,
      loop: { start, end, active: true },
    }));
  }, [deckA.bpm, deckA.position, deckB.bpm, deckB.position]);

  const toggleLoop = useCallback((deck: 'A' | 'B') => {
    const setDeck = deck === 'A' ? setDeckA : setDeckB;
    setDeck((s) => ({
      ...s,
      loop: s.loop ? { ...s.loop, active: !s.loop.active } : null,
    }));
  }, []);

  const clearLoop = useCallback((deck: 'A' | 'B') => {
    const setDeck = deck === 'A' ? setDeckA : setDeckB;
    setDeck((s) => ({ ...s, loop: null }));
  }, []);

  const moveLoop = useCallback((deck: 'A' | 'B', direction: 'forward' | 'backward') => {
    const deckState = deck === 'A' ? deckA : deckB;
    const setDeck = deck === 'A' ? setDeckA : setDeckB;

    if (!deckState.loop) return;

    const loopDuration = deckState.loop.end - deckState.loop.start;
    const offset = direction === 'forward' ? loopDuration : -loopDuration;

    setDeck((s) => ({
      ...s,
      loop: s.loop
        ? {
            ...s.loop,
            start: Math.max(0, s.loop.start + offset),
            end: Math.min(s.duration, s.loop.end + offset),
          }
        : null,
    }));
  }, [deckA.loop, deckA.duration, deckB.loop, deckB.duration]);

  const halveLoop = useCallback((deck: 'A' | 'B') => {
    const deckState = deck === 'A' ? deckA : deckB;
    const setDeck = deck === 'A' ? setDeckA : setDeckB;

    if (!deckState.loop) return;

    const loopDuration = deckState.loop.end - deckState.loop.start;
    const newDuration = loopDuration / 2;

    setDeck((s) => ({
      ...s,
      loop: s.loop ? { ...s.loop, end: s.loop.start + newDuration } : null,
    }));
  }, [deckA.loop, deckB.loop]);

  const doubleLoop = useCallback((deck: 'A' | 'B') => {
    const deckState = deck === 'A' ? deckA : deckB;
    const setDeck = deck === 'A' ? setDeckA : setDeckB;

    if (!deckState.loop) return;

    const loopDuration = deckState.loop.end - deckState.loop.start;
    const newEnd = Math.min(deckState.duration, deckState.loop.start + loopDuration * 2);

    setDeck((s) => ({
      ...s,
      loop: s.loop ? { ...s.loop, end: newEnd } : null,
    }));
  }, [deckA.loop, deckA.duration, deckB.loop, deckB.duration]);

  // === EQ & Filter ===

  const setEQ = useCallback((deck: 'A' | 'B', band: 'low' | 'mid' | 'high', value: number) => {
    const setDeck = deck === 'A' ? setDeckA : setDeckB;
    const key = `eq${band.charAt(0).toUpperCase() + band.slice(1)}` as 'eqLow' | 'eqMid' | 'eqHigh';
    setDeck((s) => ({ ...s, [key]: value }));
  }, []);

  const setFilter = useCallback((deck: 'A' | 'B', value: number) => {
    const setDeck = deck === 'A' ? setDeckA : setDeckB;
    setDeck((s) => ({ ...s, filter: value }));
  }, []);

  const killEQ = useCallback((deck: 'A' | 'B', band: 'low' | 'mid' | 'high') => {
    const deckState = deck === 'A' ? deckA : deckB;
    const setDeck = deck === 'A' ? setDeckA : setDeckB;
    const key = `eq${band.charAt(0).toUpperCase() + band.slice(1)}` as 'eqLow' | 'eqMid' | 'eqHigh';

    // Toggle between -∞ and 0
    const currentValue = deckState[key];
    setDeck((s) => ({ ...s, [key]: currentValue <= -24 ? 0 : -48 }));
  }, [deckA, deckB]);

  // === Volume ===

  const setDeckVolume = useCallback((deck: 'A' | 'B', volume: number) => {
    const setDeck = deck === 'A' ? setDeckA : setDeckB;
    setDeck((s) => ({ ...s, volume: Math.max(0, Math.min(1, volume)) }));
  }, []);

  // === Mixer Controls ===

  const setCrossfader = useCallback((position: number) => {
    setMixer((s) => ({ ...s, crossfader: Math.max(0, Math.min(1, position)) }));
  }, []);

  const setCrossfaderCurve = useCallback((curve: CrossfaderCurve) => {
    setMixer((s) => ({ ...s, crossfaderCurve: curve }));
  }, []);

  const setMasterVolume = useCallback((volume: number) => {
    setMixer((s) => ({ ...s, masterVolume: Math.max(0, Math.min(1, volume)) }));
  }, []);

  const setHeadphoneMix = useCallback((mix: number) => {
    setMixer((s) => ({ ...s, headphoneMix: Math.max(0, Math.min(1, mix)) }));
  }, []);

  const toggleCue = useCallback((deck: 'A' | 'B') => {
    setMixer((s) => ({
      ...s,
      deckACue: deck === 'A' ? !s.deckACue : s.deckACue,
      deckBCue: deck === 'B' ? !s.deckBCue : s.deckBCue,
    }));
  }, []);

  const toggleSync = useCallback(() => {
    setMixer((s) => ({ ...s, isSyncEnabled: !s.isSyncEnabled }));
  }, []);

  const setSyncSource = useCallback((source: 'A' | 'B' | 'external') => {
    setMixer((s) => ({ ...s, syncSource: source }));
  }, []);

  // === Utilities ===

  const getBeatPosition = useCallback((deck: 'A' | 'B'): number => {
    const deckState = deck === 'A' ? deckA : deckB;
    const beatDuration = 60 / deckState.bpm;
    return (deckState.position % beatDuration) / beatDuration;
  }, [deckA, deckB]);

  const getPhaseOffset = useCallback((deck: 'A' | 'B'): number => {
    const source = deck === 'A' ? deckB : deckA;
    const target = deck === 'A' ? deckA : deckB;

    const sourceBeatDuration = 60 / source.bpm;
    const targetBeatDuration = 60 / target.bpm;

    const sourcePhase = (source.position % sourceBeatDuration) / sourceBeatDuration;
    const targetPhase = (target.position % targetBeatDuration) / targetBeatDuration;

    let offset = sourcePhase - targetPhase;
    if (offset > 0.5) offset -= 1;
    if (offset < -0.5) offset += 1;

    return offset;
  }, [deckA, deckB]);

  const getMixedOutput = useCallback((): { left: number; right: number } => {
    const { deckA: gainA, deckB: gainB } = applyCrossfaderCurve(mixer.crossfader, mixer.crossfaderCurve);
    return {
      left: gainA * deckA.volume * mixer.masterVolume,
      right: gainB * deckB.volume * mixer.masterVolume,
    };
  }, [mixer, deckA.volume, deckB.volume]);

  return {
    deckA,
    deckB,
    mixer,
    audioContextRef,
    loadTrack,
    ejectTrack,
    play,
    pause,
    stop,
    seek,
    setPitch,
    nudge,
    syncBpm,
    syncPhase,
    setCuePoint,
    jumpToCue,
    deleteCuePoint,
    setLoop,
    toggleLoop,
    clearLoop,
    moveLoop,
    halveLoop,
    doubleLoop,
    setEQ,
    setFilter,
    killEQ,
    setDeckVolume,
    setCrossfader,
    setCrossfaderCurve,
    setMasterVolume,
    setHeadphoneMix,
    toggleCue,
    toggleSync,
    setSyncSource,
    getBeatPosition,
    getPhaseOffset,
    getMixedOutput,
  };
}

export default useDJMixer;
