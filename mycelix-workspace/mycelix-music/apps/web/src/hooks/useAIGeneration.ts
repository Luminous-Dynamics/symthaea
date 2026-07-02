// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * AI Music Generation Hook
 *
 * Provides AI-powered music generation capabilities including:
 * - Beat/rhythm generation with style presets
 * - Melody auto-completion
 * - Style transfer between tracks
 * - Remix suggestions based on analysis
 */

import { useState, useCallback, useRef, useEffect } from 'react';

// Generation modes
export type GenerationMode = 'beat' | 'melody' | 'harmony' | 'full';

// Musical styles for generation
export type MusicStyle =
  | 'electronic' | 'house' | 'techno' | 'trance' | 'dnb'
  | 'hiphop' | 'trap' | 'lofi'
  | 'rock' | 'jazz' | 'classical' | 'ambient';

// Beat pattern representation
export interface BeatPattern {
  id: string;
  name: string;
  tempo: number;
  timeSignature: [number, number];
  bars: number;
  tracks: DrumTrack[];
  style: MusicStyle;
}

export interface DrumTrack {
  name: string;
  instrument: 'kick' | 'snare' | 'hihat' | 'clap' | 'tom' | 'cymbal' | 'perc';
  pattern: boolean[]; // Steps where this instrument plays
  velocity: number[]; // Velocity per step (0-1)
}

// Melody representation
export interface MelodySequence {
  id: string;
  notes: NoteEvent[];
  scale: string;
  key: string;
  tempo: number;
}

export interface NoteEvent {
  pitch: number; // MIDI note number
  startTime: number; // In beats
  duration: number; // In beats
  velocity: number; // 0-1
}

// Style transfer parameters
export interface StyleTransferParams {
  sourceTrackId: string;
  targetStyle: MusicStyle;
  preserveRhythm: boolean;
  preserveMelody: boolean;
  preserveHarmony: boolean;
  intensity: number; // 0-1, how much to transform
}

// Generation parameters
export interface GenerationParams {
  mode: GenerationMode;
  style: MusicStyle;
  tempo?: number;
  key?: string;
  scale?: string;
  bars?: number;
  complexity?: number; // 0-1
  energy?: number; // 0-1
  seed?: number;
}

// Generation result
export interface GenerationResult {
  id: string;
  type: GenerationMode;
  audioBuffer?: AudioBuffer;
  midiData?: Uint8Array;
  beatPattern?: BeatPattern;
  melodySequence?: MelodySequence;
  metadata: {
    style: MusicStyle;
    tempo: number;
    key: string;
    duration: number;
    createdAt: number;
  };
}

// Hook state
export interface AIGenerationState {
  isGenerating: boolean;
  progress: number;
  currentTask: string | null;
  history: GenerationResult[];
  error: string | null;
}

// Preset beat patterns
const PRESET_PATTERNS: Record<MusicStyle, Partial<BeatPattern>> = {
  electronic: {
    tempo: 128,
    timeSignature: [4, 4],
    tracks: [
      { name: 'Kick', instrument: 'kick', pattern: [true, false, false, false, true, false, false, false, true, false, false, false, true, false, false, false], velocity: [1, 0, 0, 0, 0.9, 0, 0, 0, 1, 0, 0, 0, 0.9, 0, 0, 0] },
      { name: 'Snare', instrument: 'snare', pattern: [false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false], velocity: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] },
      { name: 'HiHat', instrument: 'hihat', pattern: [true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false], velocity: [0.8, 0, 0.6, 0, 0.8, 0, 0.6, 0, 0.8, 0, 0.6, 0, 0.8, 0, 0.6, 0] },
    ],
  },
  house: {
    tempo: 124,
    timeSignature: [4, 4],
    tracks: [
      { name: 'Kick', instrument: 'kick', pattern: [true, false, false, false, true, false, false, false, true, false, false, false, true, false, false, false], velocity: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0] },
      { name: 'Clap', instrument: 'clap', pattern: [false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false], velocity: [0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0, 0, 0, 0.9, 0, 0, 0] },
      { name: 'HiHat', instrument: 'hihat', pattern: [false, false, true, false, false, false, true, false, false, false, true, false, false, false, true, false], velocity: [0, 0, 0.7, 0, 0, 0, 0.7, 0, 0, 0, 0.7, 0, 0, 0, 0.7, 0] },
    ],
  },
  techno: {
    tempo: 138,
    timeSignature: [4, 4],
    tracks: [
      { name: 'Kick', instrument: 'kick', pattern: [true, false, false, false, true, false, false, false, true, false, false, false, true, false, false, false], velocity: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0] },
      { name: 'HiHat', instrument: 'hihat', pattern: [true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true], velocity: [0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3] },
      { name: 'Cymbal', instrument: 'cymbal', pattern: [false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false], velocity: [0, 0, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, 0] },
    ],
  },
  trance: {
    tempo: 140,
    timeSignature: [4, 4],
    tracks: [
      { name: 'Kick', instrument: 'kick', pattern: [true, false, false, false, true, false, false, false, true, false, false, false, true, false, false, false], velocity: [1, 0, 0, 0, 0.9, 0, 0, 0, 1, 0, 0, 0, 0.9, 0, 0, 0] },
      { name: 'Snare', instrument: 'snare', pattern: [false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, true], velocity: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.7] },
      { name: 'HiHat', instrument: 'hihat', pattern: [false, false, true, false, false, false, true, false, false, false, true, false, false, false, true, true], velocity: [0, 0, 0.8, 0, 0, 0, 0.8, 0, 0, 0, 0.8, 0, 0, 0, 0.8, 0.5] },
    ],
  },
  dnb: {
    tempo: 174,
    timeSignature: [4, 4],
    tracks: [
      { name: 'Kick', instrument: 'kick', pattern: [true, false, false, false, false, false, true, false, false, false, true, false, false, false, false, false], velocity: [1, 0, 0, 0, 0, 0, 0.9, 0, 0, 0, 1, 0, 0, 0, 0, 0] },
      { name: 'Snare', instrument: 'snare', pattern: [false, false, false, false, true, false, false, false, false, false, false, false, false, true, false, false], velocity: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] },
      { name: 'HiHat', instrument: 'hihat', pattern: [true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true], velocity: [0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4] },
    ],
  },
  hiphop: {
    tempo: 90,
    timeSignature: [4, 4],
    tracks: [
      { name: 'Kick', instrument: 'kick', pattern: [true, false, false, true, false, false, true, false, false, false, true, false, false, false, false, false], velocity: [1, 0, 0, 0.8, 0, 0, 0.9, 0, 0, 0, 1, 0, 0, 0, 0, 0] },
      { name: 'Snare', instrument: 'snare', pattern: [false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false], velocity: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] },
      { name: 'HiHat', instrument: 'hihat', pattern: [true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, true], velocity: [0.7, 0, 0.5, 0, 0.7, 0, 0.5, 0, 0.7, 0, 0.5, 0, 0.7, 0, 0.5, 0.4] },
    ],
  },
  trap: {
    tempo: 140,
    timeSignature: [4, 4],
    tracks: [
      { name: 'Kick', instrument: 'kick', pattern: [true, false, false, false, false, false, false, true, false, false, true, false, false, false, false, false], velocity: [1, 0, 0, 0, 0, 0, 0, 0.9, 0, 0, 1, 0, 0, 0, 0, 0] },
      { name: 'Snare', instrument: 'snare', pattern: [false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false], velocity: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] },
      { name: 'HiHat', instrument: 'hihat', pattern: [true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true], velocity: [0.7, 0.5, 0.6, 0.5, 0.7, 0.5, 0.6, 0.5, 0.7, 0.5, 0.6, 0.5, 0.7, 0.5, 0.6, 0.5] },
    ],
  },
  lofi: {
    tempo: 75,
    timeSignature: [4, 4],
    tracks: [
      { name: 'Kick', instrument: 'kick', pattern: [true, false, false, false, false, false, true, false, false, true, false, false, false, false, false, false], velocity: [0.8, 0, 0, 0, 0, 0, 0.7, 0, 0, 0.6, 0, 0, 0, 0, 0, 0] },
      { name: 'Snare', instrument: 'snare', pattern: [false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, true], velocity: [0, 0, 0, 0, 0.7, 0, 0, 0, 0, 0, 0, 0, 0.7, 0, 0, 0.4] },
      { name: 'HiHat', instrument: 'hihat', pattern: [false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true], velocity: [0, 0.5, 0, 0.4, 0, 0.5, 0, 0.4, 0, 0.5, 0, 0.4, 0, 0.5, 0, 0.4] },
    ],
  },
  rock: {
    tempo: 120,
    timeSignature: [4, 4],
    tracks: [
      { name: 'Kick', instrument: 'kick', pattern: [true, false, false, false, true, false, true, false, true, false, false, false, true, false, true, false], velocity: [1, 0, 0, 0, 0.9, 0, 0.8, 0, 1, 0, 0, 0, 0.9, 0, 0.8, 0] },
      { name: 'Snare', instrument: 'snare', pattern: [false, false, false, false, true, false, false, false, false, false, false, false, true, false, false, false], velocity: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] },
      { name: 'HiHat', instrument: 'hihat', pattern: [true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false], velocity: [0.8, 0, 0.6, 0, 0.8, 0, 0.6, 0, 0.8, 0, 0.6, 0, 0.8, 0, 0.6, 0] },
    ],
  },
  jazz: {
    tempo: 110,
    timeSignature: [4, 4],
    tracks: [
      { name: 'Kick', instrument: 'kick', pattern: [true, false, false, false, false, false, true, false, false, false, false, false, true, false, false, false], velocity: [0.7, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0.7, 0, 0, 0] },
      { name: 'Snare', instrument: 'snare', pattern: [false, false, true, false, false, true, false, false, true, false, false, true, false, false, true, false], velocity: [0, 0, 0.5, 0, 0, 0.4, 0, 0, 0.5, 0, 0, 0.4, 0, 0, 0.5, 0] },
      { name: 'Cymbal', instrument: 'cymbal', pattern: [true, false, true, true, true, false, true, true, true, false, true, true, true, false, true, true], velocity: [0.6, 0, 0.5, 0.4, 0.6, 0, 0.5, 0.4, 0.6, 0, 0.5, 0.4, 0.6, 0, 0.5, 0.4] },
    ],
  },
  classical: {
    tempo: 100,
    timeSignature: [4, 4],
    tracks: [
      { name: 'Bass', instrument: 'kick', pattern: [true, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false], velocity: [0.6, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0] },
      { name: 'Timpani', instrument: 'tom', pattern: [false, false, false, false, true, false, false, false, false, false, false, false, false, false, true, false], velocity: [0, 0, 0, 0, 0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6, 0] },
    ],
  },
  ambient: {
    tempo: 70,
    timeSignature: [4, 4],
    tracks: [
      { name: 'Kick', instrument: 'kick', pattern: [true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false], velocity: [0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
      { name: 'Perc', instrument: 'perc', pattern: [false, false, false, false, false, false, true, false, false, false, false, false, false, true, false, false], velocity: [0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0.2, 0, 0] },
    ],
  },
};

// Scale patterns (semitones from root)
const SCALES: Record<string, number[]> = {
  major: [0, 2, 4, 5, 7, 9, 11],
  minor: [0, 2, 3, 5, 7, 8, 10],
  pentatonic: [0, 2, 4, 7, 9],
  blues: [0, 3, 5, 6, 7, 10],
  dorian: [0, 2, 3, 5, 7, 9, 10],
  mixolydian: [0, 2, 4, 5, 7, 9, 10],
  phrygian: [0, 1, 3, 5, 7, 8, 10],
};

export function useAIGeneration() {
  const [state, setState] = useState<AIGenerationState>({
    isGenerating: false,
    progress: 0,
    currentTask: null,
    history: [],
    error: null,
  });

  const audioContextRef = useRef<AudioContext | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Initialize audio context
  useEffect(() => {
    audioContextRef.current = new AudioContext();
    return () => {
      audioContextRef.current?.close();
    };
  }, []);

  // Generate beat pattern
  const generateBeat = useCallback(async (params: GenerationParams): Promise<BeatPattern> => {
    const { style = 'electronic', tempo, bars = 4, complexity = 0.5, energy = 0.5, seed } = params;

    // Get base pattern for style
    const basePattern = PRESET_PATTERNS[style];
    const patternTempo = tempo || basePattern.tempo || 120;
    const stepsPerBar = 16;
    const totalSteps = stepsPerBar * bars;

    // Initialize random with seed if provided
    const random = seed !== undefined
      ? (() => {
          let s = seed;
          return () => {
            s = Math.sin(s) * 10000;
            return s - Math.floor(s);
          };
        })()
      : Math.random;

    // Create tracks based on complexity and energy
    const tracks: DrumTrack[] = (basePattern.tracks || []).map(track => {
      const pattern: boolean[] = [];
      const velocity: number[] = [];

      for (let i = 0; i < totalSteps; i++) {
        const stepInBar = i % stepsPerBar;
        const baseHit = track.pattern[stepInBar] || false;

        // Add variation based on complexity
        let shouldHit = baseHit;
        if (complexity > 0.3 && random() < complexity * 0.3) {
          shouldHit = !shouldHit; // Occasional variation
        }

        // Adjust velocity based on energy
        const baseVelocity = track.velocity[stepInBar] || 0.8;
        const adjustedVelocity = shouldHit
          ? Math.min(1, baseVelocity * (0.7 + energy * 0.6))
          : 0;

        pattern.push(shouldHit);
        velocity.push(adjustedVelocity);
      }

      return { ...track, pattern, velocity };
    });

    // Add extra percussion for high complexity
    if (complexity > 0.7) {
      const percPattern: boolean[] = [];
      const percVelocity: number[] = [];

      for (let i = 0; i < totalSteps; i++) {
        const hit = random() < 0.15;
        percPattern.push(hit);
        percVelocity.push(hit ? 0.3 + random() * 0.3 : 0);
      }

      tracks.push({
        name: 'Percussion',
        instrument: 'perc',
        pattern: percPattern,
        velocity: percVelocity,
      });
    }

    return {
      id: `beat-${Date.now()}`,
      name: `${style.charAt(0).toUpperCase() + style.slice(1)} Beat`,
      tempo: patternTempo,
      timeSignature: basePattern.timeSignature || [4, 4],
      bars,
      tracks,
      style,
    };
  }, []);

  // Generate melody
  const generateMelody = useCallback(async (params: GenerationParams): Promise<MelodySequence> => {
    const {
      style = 'electronic',
      tempo = 120,
      key = 'C',
      scale = 'minor',
      bars = 4,
      complexity = 0.5,
      energy = 0.5,
      seed
    } = params;

    const random = seed !== undefined
      ? (() => {
          let s = seed;
          return () => {
            s = Math.sin(s) * 10000;
            return s - Math.floor(s);
          };
        })()
      : Math.random;

    // Get scale intervals
    const scaleIntervals = SCALES[scale] || SCALES.minor;

    // Convert key to root note (C4 = 60)
    const keyToMidi: Record<string, number> = {
      'C': 60, 'C#': 61, 'Db': 61, 'D': 62, 'D#': 63, 'Eb': 63,
      'E': 64, 'F': 65, 'F#': 66, 'Gb': 66, 'G': 67, 'G#': 68,
      'Ab': 68, 'A': 69, 'A#': 70, 'Bb': 70, 'B': 71
    };
    const rootNote = keyToMidi[key] || 60;

    // Generate notes
    const notes: NoteEvent[] = [];
    const beatsPerBar = 4;
    const totalBeats = beatsPerBar * bars;

    let currentBeat = 0;
    let previousPitch = rootNote;

    while (currentBeat < totalBeats) {
      // Determine note duration based on complexity
      const durationOptions = complexity > 0.6
        ? [0.25, 0.5, 0.75, 1]
        : [0.5, 1, 2];
      const duration = durationOptions[Math.floor(random() * durationOptions.length)];

      // Rest probability
      if (random() < 0.2 - energy * 0.1) {
        currentBeat += duration;
        continue;
      }

      // Choose pitch from scale
      const scaleIndex = Math.floor(random() * scaleIntervals.length);
      const octaveOffset = Math.floor(random() * 2) * 12 - 12; // -1 to +1 octave
      let pitch = rootNote + scaleIntervals[scaleIndex] + octaveOffset;

      // Prefer stepwise motion (more musical)
      if (random() < 0.6) {
        const direction = pitch > previousPitch ? -1 : 1;
        const nearestScaleIndex = scaleIntervals.findIndex(
          interval => rootNote + interval >= previousPitch - 2 && rootNote + interval <= previousPitch + 2
        );
        if (nearestScaleIndex !== -1) {
          const nextIndex = nearestScaleIndex + direction;
          if (nextIndex >= 0 && nextIndex < scaleIntervals.length) {
            pitch = rootNote + scaleIntervals[nextIndex];
          }
        }
      }

      // Velocity based on beat position and energy
      const onBeat = currentBeat % 1 === 0;
      const velocity = (onBeat ? 0.8 : 0.6) * (0.6 + energy * 0.4);

      notes.push({
        pitch,
        startTime: currentBeat,
        duration: Math.min(duration, totalBeats - currentBeat),
        velocity: Math.min(1, velocity),
      });

      previousPitch = pitch;
      currentBeat += duration;
    }

    return {
      id: `melody-${Date.now()}`,
      notes,
      scale,
      key,
      tempo,
    };
  }, []);

  // Synthesize audio from beat pattern
  const synthesizeBeat = useCallback(async (pattern: BeatPattern): Promise<AudioBuffer> => {
    const ctx = audioContextRef.current;
    if (!ctx) throw new Error('AudioContext not initialized');

    const sampleRate = ctx.sampleRate;
    const secondsPerBeat = 60 / pattern.tempo;
    const secondsPerStep = secondsPerBeat / 4; // 16th notes
    const totalSteps = pattern.tracks[0]?.pattern.length || 64;
    const duration = totalSteps * secondsPerStep;

    const buffer = ctx.createBuffer(2, Math.ceil(duration * sampleRate), sampleRate);
    const leftChannel = buffer.getChannelData(0);
    const rightChannel = buffer.getChannelData(1);

    // Simple drum synthesis
    const synthesizeDrum = (instrument: DrumTrack['instrument'], velocity: number, startSample: number) => {
      const samplesPerHit = Math.floor(sampleRate * 0.2);

      for (let i = 0; i < samplesPerHit && startSample + i < leftChannel.length; i++) {
        const t = i / sampleRate;
        let sample = 0;

        switch (instrument) {
          case 'kick':
            // Sine wave with pitch envelope
            const kickFreq = 150 * Math.exp(-t * 30);
            sample = Math.sin(2 * Math.PI * kickFreq * t) * Math.exp(-t * 15);
            break;
          case 'snare':
            // Noise + tone
            sample = (Math.random() * 2 - 1) * Math.exp(-t * 20) * 0.7;
            sample += Math.sin(2 * Math.PI * 200 * t) * Math.exp(-t * 30) * 0.3;
            break;
          case 'hihat':
            // High-frequency noise
            sample = (Math.random() * 2 - 1) * Math.exp(-t * 50);
            break;
          case 'clap':
            // Multiple noise bursts
            const claps = [0, 0.01, 0.02];
            sample = claps.reduce((acc, offset) => {
              if (t >= offset) {
                return acc + (Math.random() * 2 - 1) * Math.exp(-(t - offset) * 30) / claps.length;
              }
              return acc;
            }, 0);
            break;
          case 'tom':
            const tomFreq = 100 * Math.exp(-t * 10);
            sample = Math.sin(2 * Math.PI * tomFreq * t) * Math.exp(-t * 10);
            break;
          case 'cymbal':
            sample = (Math.random() * 2 - 1) * Math.exp(-t * 5);
            break;
          case 'perc':
            sample = Math.sin(2 * Math.PI * 800 * t) * Math.exp(-t * 40);
            break;
        }

        const scaled = sample * velocity * 0.5;
        leftChannel[startSample + i] += scaled;
        rightChannel[startSample + i] += scaled;
      }
    };

    // Render each track
    pattern.tracks.forEach(track => {
      track.pattern.forEach((hit, step) => {
        if (hit && track.velocity[step] > 0) {
          const startTime = step * secondsPerStep;
          const startSample = Math.floor(startTime * sampleRate);
          synthesizeDrum(track.instrument, track.velocity[step], startSample);
        }
      });
    });

    return buffer;
  }, []);

  // Synthesize audio from melody
  const synthesizeMelody = useCallback(async (melody: MelodySequence): Promise<AudioBuffer> => {
    const ctx = audioContextRef.current;
    if (!ctx) throw new Error('AudioContext not initialized');

    const sampleRate = ctx.sampleRate;
    const secondsPerBeat = 60 / melody.tempo;
    const duration = melody.notes.reduce((max, note) =>
      Math.max(max, (note.startTime + note.duration) * secondsPerBeat), 0
    ) + 0.5;

    const buffer = ctx.createBuffer(2, Math.ceil(duration * sampleRate), sampleRate);
    const leftChannel = buffer.getChannelData(0);
    const rightChannel = buffer.getChannelData(1);

    // Simple synth
    melody.notes.forEach(note => {
      const startTime = note.startTime * secondsPerBeat;
      const noteDuration = note.duration * secondsPerBeat;
      const startSample = Math.floor(startTime * sampleRate);
      const numSamples = Math.floor(noteDuration * sampleRate);
      const frequency = 440 * Math.pow(2, (note.pitch - 69) / 12);

      for (let i = 0; i < numSamples && startSample + i < leftChannel.length; i++) {
        const t = i / sampleRate;
        // Saw wave with filter
        let sample = 0;
        for (let h = 1; h <= 8; h++) {
          sample += Math.sin(2 * Math.PI * frequency * h * t) / h;
        }

        // ADSR envelope
        const attack = 0.01, decay = 0.1, sustain = 0.7, release = 0.1;
        let envelope = 0;
        if (t < attack) {
          envelope = t / attack;
        } else if (t < attack + decay) {
          envelope = 1 - (1 - sustain) * (t - attack) / decay;
        } else if (t < noteDuration - release) {
          envelope = sustain;
        } else {
          envelope = sustain * (noteDuration - t) / release;
        }

        const scaled = sample * envelope * note.velocity * 0.2;
        leftChannel[startSample + i] += scaled;
        rightChannel[startSample + i] += scaled;
      }
    });

    return buffer;
  }, []);

  // Main generate function
  const generate = useCallback(async (params: GenerationParams): Promise<GenerationResult> => {
    setState(prev => ({
      ...prev,
      isGenerating: true,
      progress: 0,
      currentTask: `Generating ${params.mode}...`,
      error: null,
    }));

    abortControllerRef.current = new AbortController();

    try {
      let result: GenerationResult;

      switch (params.mode) {
        case 'beat': {
          setState(prev => ({ ...prev, progress: 20, currentTask: 'Creating beat pattern...' }));
          const beatPattern = await generateBeat(params);

          setState(prev => ({ ...prev, progress: 60, currentTask: 'Synthesizing audio...' }));
          const audioBuffer = await synthesizeBeat(beatPattern);

          result = {
            id: beatPattern.id,
            type: 'beat',
            audioBuffer,
            beatPattern,
            metadata: {
              style: params.style || 'electronic',
              tempo: beatPattern.tempo,
              key: 'N/A',
              duration: audioBuffer.duration,
              createdAt: Date.now(),
            },
          };
          break;
        }

        case 'melody': {
          setState(prev => ({ ...prev, progress: 20, currentTask: 'Composing melody...' }));
          const melodySequence = await generateMelody(params);

          setState(prev => ({ ...prev, progress: 60, currentTask: 'Synthesizing audio...' }));
          const audioBuffer = await synthesizeMelody(melodySequence);

          result = {
            id: melodySequence.id,
            type: 'melody',
            audioBuffer,
            melodySequence,
            metadata: {
              style: params.style || 'electronic',
              tempo: melodySequence.tempo,
              key: `${melodySequence.key} ${melodySequence.scale}`,
              duration: audioBuffer.duration,
              createdAt: Date.now(),
            },
          };
          break;
        }

        case 'full': {
          setState(prev => ({ ...prev, progress: 10, currentTask: 'Generating beat...' }));
          const beatPattern = await generateBeat(params);

          setState(prev => ({ ...prev, progress: 30, currentTask: 'Composing melody...' }));
          const melodySequence = await generateMelody({ ...params, tempo: beatPattern.tempo });

          setState(prev => ({ ...prev, progress: 50, currentTask: 'Synthesizing beat...' }));
          const beatBuffer = await synthesizeBeat(beatPattern);

          setState(prev => ({ ...prev, progress: 70, currentTask: 'Synthesizing melody...' }));
          const melodyBuffer = await synthesizeMelody(melodySequence);

          // Mix buffers
          setState(prev => ({ ...prev, progress: 90, currentTask: 'Mixing...' }));
          const ctx = audioContextRef.current!;
          const duration = Math.max(beatBuffer.duration, melodyBuffer.duration);
          const mixedBuffer = ctx.createBuffer(2, Math.ceil(duration * ctx.sampleRate), ctx.sampleRate);

          for (let channel = 0; channel < 2; channel++) {
            const mixedData = mixedBuffer.getChannelData(channel);
            const beatData = beatBuffer.getChannelData(channel);
            const melodyData = melodyBuffer.getChannelData(channel);

            for (let i = 0; i < mixedData.length; i++) {
              mixedData[i] = (beatData[i] || 0) * 0.6 + (melodyData[i] || 0) * 0.5;
            }
          }

          result = {
            id: `full-${Date.now()}`,
            type: 'full',
            audioBuffer: mixedBuffer,
            beatPattern,
            melodySequence,
            metadata: {
              style: params.style || 'electronic',
              tempo: beatPattern.tempo,
              key: `${melodySequence.key} ${melodySequence.scale}`,
              duration: mixedBuffer.duration,
              createdAt: Date.now(),
            },
          };
          break;
        }

        default:
          throw new Error(`Unknown generation mode: ${params.mode}`);
      }

      setState(prev => ({
        ...prev,
        isGenerating: false,
        progress: 100,
        currentTask: null,
        history: [result, ...prev.history].slice(0, 50),
      }));

      return result;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Generation failed';
      setState(prev => ({
        ...prev,
        isGenerating: false,
        progress: 0,
        currentTask: null,
        error: message,
      }));
      throw error;
    }
  }, [generateBeat, generateMelody, synthesizeBeat, synthesizeMelody]);

  // Style transfer
  const styleTransfer = useCallback(async (
    sourceBuffer: AudioBuffer,
    params: StyleTransferParams
  ): Promise<AudioBuffer> => {
    setState(prev => ({
      ...prev,
      isGenerating: true,
      progress: 0,
      currentTask: 'Analyzing source...',
      error: null,
    }));

    try {
      const ctx = audioContextRef.current;
      if (!ctx) throw new Error('AudioContext not initialized');

      // For now, apply style-based processing
      // In production, this would use ML models
      setState(prev => ({ ...prev, progress: 30, currentTask: 'Applying style...' }));

      const targetTempo = PRESET_PATTERNS[params.targetStyle]?.tempo || 120;
      const intensity = params.intensity;

      // Create output buffer
      const outputBuffer = ctx.createBuffer(
        sourceBuffer.numberOfChannels,
        sourceBuffer.length,
        sourceBuffer.sampleRate
      );

      // Process each channel
      for (let channel = 0; channel < sourceBuffer.numberOfChannels; channel++) {
        const inputData = sourceBuffer.getChannelData(channel);
        const outputData = outputBuffer.getChannelData(channel);

        // Apply style-based filtering
        for (let i = 0; i < inputData.length; i++) {
          let sample = inputData[i];

          // Style-specific processing
          switch (params.targetStyle) {
            case 'lofi':
              // Add bit crushing effect
              const bits = 12 - intensity * 4;
              const levels = Math.pow(2, bits);
              sample = Math.round(sample * levels) / levels;
              break;
            case 'electronic':
            case 'techno':
              // Add subtle distortion
              sample = Math.tanh(sample * (1 + intensity));
              break;
            case 'ambient':
              // Soften transients
              if (i > 0) {
                sample = sample * (1 - intensity * 0.5) + outputData[i - 1] * intensity * 0.5;
              }
              break;
          }

          outputData[i] = sample;
        }
      }

      setState(prev => ({
        ...prev,
        isGenerating: false,
        progress: 100,
        currentTask: null,
      }));

      return outputBuffer;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Style transfer failed';
      setState(prev => ({
        ...prev,
        isGenerating: false,
        error: message,
      }));
      throw error;
    }
  }, []);

  // Cancel generation
  const cancel = useCallback(() => {
    abortControllerRef.current?.abort();
    setState(prev => ({
      ...prev,
      isGenerating: false,
      progress: 0,
      currentTask: null,
    }));
  }, []);

  // Play generated audio
  const play = useCallback((result: GenerationResult) => {
    const ctx = audioContextRef.current;
    if (!ctx || !result.audioBuffer) return;

    const source = ctx.createBufferSource();
    source.buffer = result.audioBuffer;
    source.connect(ctx.destination);
    source.start();

    return source;
  }, []);

  // Clear history
  const clearHistory = useCallback(() => {
    setState(prev => ({ ...prev, history: [] }));
  }, []);

  return {
    ...state,
    generate,
    styleTransfer,
    cancel,
    play,
    clearHistory,
    presets: PRESET_PATTERNS,
    scales: Object.keys(SCALES),
  };
}

export default useAIGeneration;
