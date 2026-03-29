// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Music Theory Analysis Hook
 *
 * Provides music theory analysis and education:
 * - Key detection
 * - Chord recognition
 * - Scale suggestions
 * - Harmonic analysis
 * - Melody patterns
 * - Progression suggestions
 *
 * Integrates with mycelix-mir for beat/chord/key detection
 */

import { useState, useCallback, useRef, useEffect } from 'react';

// Musical keys
export type NoteName = 'C' | 'C#' | 'D' | 'D#' | 'E' | 'F' | 'F#' | 'G' | 'G#' | 'A' | 'A#' | 'B';
export type KeyMode = 'major' | 'minor';

export interface MusicalKey {
  root: NoteName;
  mode: KeyMode;
  confidence: number;
  alternates: Array<{ root: NoteName; mode: KeyMode; confidence: number }>;
}

// Chord types
export type ChordQuality =
  | 'major' | 'minor' | 'diminished' | 'augmented'
  | 'major7' | 'minor7' | 'dominant7' | 'diminished7'
  | 'sus2' | 'sus4' | 'add9' | 'add11';

export interface Chord {
  root: NoteName;
  quality: ChordQuality;
  bass?: NoteName; // For slash chords
  extensions?: number[]; // 9, 11, 13
  time: number; // Position in seconds
  duration: number;
  confidence: number;
}

// Scale information
export interface ScaleInfo {
  name: string;
  notes: NoteName[];
  intervals: number[];
  mood: string;
  genres: string[];
  relatedChords: string[];
}

// Chord progression
export interface ChordProgression {
  chords: Chord[];
  romanNumerals: string[];
  name?: string; // e.g., "I-V-vi-IV"
  commonNames?: string[]; // e.g., "Pop progression"
}

// Harmonic analysis
export interface HarmonicAnalysis {
  key: MusicalKey;
  chords: Chord[];
  progressions: ChordProgression[];
  tempo: number;
  timeSignature: [number, number];
  sections: Section[];
  suggestions: TheorySuggestion[];
}

export interface Section {
  name: string; // intro, verse, chorus, bridge, outro
  startTime: number;
  endTime: number;
  key?: MusicalKey;
  energy: number;
}

export interface TheorySuggestion {
  type: 'chord' | 'scale' | 'progression' | 'modulation';
  description: string;
  example?: string;
  theory?: string; // Theory explanation
}

// Scale patterns
const SCALE_PATTERNS: Record<string, { intervals: number[]; mood: string; genres: string[] }> = {
  'Major': { intervals: [0, 2, 4, 5, 7, 9, 11], mood: 'Happy, bright', genres: ['Pop', 'Classical'] },
  'Natural Minor': { intervals: [0, 2, 3, 5, 7, 8, 10], mood: 'Sad, dark', genres: ['Rock', 'Classical'] },
  'Harmonic Minor': { intervals: [0, 2, 3, 5, 7, 8, 11], mood: 'Exotic, tense', genres: ['Classical', 'Metal'] },
  'Melodic Minor': { intervals: [0, 2, 3, 5, 7, 9, 11], mood: 'Jazz, sophisticated', genres: ['Jazz', 'Fusion'] },
  'Pentatonic Major': { intervals: [0, 2, 4, 7, 9], mood: 'Universal, folk', genres: ['Blues', 'Rock', 'Folk'] },
  'Pentatonic Minor': { intervals: [0, 3, 5, 7, 10], mood: 'Bluesy, versatile', genres: ['Blues', 'Rock'] },
  'Blues': { intervals: [0, 3, 5, 6, 7, 10], mood: 'Soulful, expressive', genres: ['Blues', 'Rock', 'Jazz'] },
  'Dorian': { intervals: [0, 2, 3, 5, 7, 9, 10], mood: 'Minor but bright', genres: ['Jazz', 'Funk'] },
  'Phrygian': { intervals: [0, 1, 3, 5, 7, 8, 10], mood: 'Spanish, exotic', genres: ['Flamenco', 'Metal'] },
  'Lydian': { intervals: [0, 2, 4, 6, 7, 9, 11], mood: 'Dreamy, ethereal', genres: ['Film scores', 'Jazz'] },
  'Mixolydian': { intervals: [0, 2, 4, 5, 7, 9, 10], mood: 'Rock, bluesy major', genres: ['Rock', 'Blues'] },
  'Locrian': { intervals: [0, 1, 3, 5, 6, 8, 10], mood: 'Unstable, dark', genres: ['Metal', 'Jazz'] },
};

// Common chord progressions
const COMMON_PROGRESSIONS: Record<string, { numerals: string[]; name: string; genres: string[] }> = {
  'I-V-vi-IV': { numerals: ['I', 'V', 'vi', 'IV'], name: 'Pop Progression', genres: ['Pop', 'Rock'] },
  'I-IV-V-I': { numerals: ['I', 'IV', 'V', 'I'], name: 'Blues/Rock', genres: ['Blues', 'Rock'] },
  'ii-V-I': { numerals: ['ii', 'V', 'I'], name: 'Jazz Standard', genres: ['Jazz'] },
  'I-vi-IV-V': { numerals: ['I', 'vi', 'IV', 'V'], name: '50s Progression', genres: ['Doo-wop', 'Pop'] },
  'vi-IV-I-V': { numerals: ['vi', 'IV', 'I', 'V'], name: 'Sad Pop', genres: ['Pop', 'Ballads'] },
  'I-IV-vi-V': { numerals: ['I', 'IV', 'vi', 'V'], name: 'Optimistic', genres: ['Pop', 'Country'] },
  'i-VII-VI-VII': { numerals: ['i', 'VII', 'VI', 'VII'], name: 'Andalusian', genres: ['Flamenco', 'Metal'] },
  'I-bVII-IV-I': { numerals: ['I', 'bVII', 'IV', 'I'], name: 'Rock Progression', genres: ['Rock'] },
};

// Note names
const NOTE_NAMES: NoteName[] = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

// Chord templates (relative to root)
const CHORD_TEMPLATES: Record<ChordQuality, number[]> = {
  'major': [0, 4, 7],
  'minor': [0, 3, 7],
  'diminished': [0, 3, 6],
  'augmented': [0, 4, 8],
  'major7': [0, 4, 7, 11],
  'minor7': [0, 3, 7, 10],
  'dominant7': [0, 4, 7, 10],
  'diminished7': [0, 3, 6, 9],
  'sus2': [0, 2, 7],
  'sus4': [0, 5, 7],
  'add9': [0, 4, 7, 14],
  'add11': [0, 4, 7, 17],
};

// Hook state
export interface MusicTheoryState {
  isAnalyzing: boolean;
  progress: number;
  analysis: HarmonicAnalysis | null;
  selectedScale: ScaleInfo | null;
  availableScales: ScaleInfo[];
  error: string | null;
}

export function useMusicTheory() {
  const [state, setState] = useState<MusicTheoryState>({
    isAnalyzing: false,
    progress: 0,
    analysis: null,
    selectedScale: null,
    availableScales: [],
    error: null,
  });

  const audioContextRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    audioContextRef.current = new AudioContext();
    return () => {
      audioContextRef.current?.close();
    };
  }, []);

  // Detect key from audio
  const detectKey = useCallback(async (buffer: AudioBuffer): Promise<MusicalKey> => {
    // Krumhansl-Kessler key finding algorithm (simplified)
    const ctx = audioContextRef.current;
    if (!ctx) throw new Error('AudioContext not initialized');

    // Chroma feature extraction (simplified)
    const sampleRate = buffer.sampleRate;
    const channelData = buffer.getChannelData(0);

    // Use FFT to get frequency content
    const fftSize = 8192;
    const chromaBins = new Float32Array(12).fill(0);

    // Simplified chroma extraction
    for (let offset = 0; offset < channelData.length - fftSize; offset += fftSize) {
      const chunk = channelData.slice(offset, offset + fftSize);

      // Simple autocorrelation for pitch estimation
      for (let note = 0; note < 12; note++) {
        const freq = 440 * Math.pow(2, (note - 9) / 12); // A4 = 440Hz
        const period = sampleRate / freq;

        let correlation = 0;
        for (let i = 0; i < fftSize - period; i++) {
          correlation += chunk[i] * chunk[i + Math.floor(period)];
        }

        chromaBins[note] += Math.max(0, correlation);
      }
    }

    // Normalize
    const maxChroma = Math.max(...chromaBins);
    for (let i = 0; i < 12; i++) {
      chromaBins[i] /= maxChroma || 1;
    }

    // Key profiles (Krumhansl-Kessler)
    const majorProfile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
    const minorProfile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

    let bestKey = { root: 'C' as NoteName, mode: 'major' as KeyMode, confidence: 0 };
    const alternates: Array<{ root: NoteName; mode: KeyMode; confidence: number }> = [];

    for (let i = 0; i < 12; i++) {
      // Rotate chroma to test each key
      const rotatedChroma = [...chromaBins.slice(i), ...chromaBins.slice(0, i)];

      // Correlate with major profile
      let majorCorr = 0;
      for (let j = 0; j < 12; j++) {
        majorCorr += rotatedChroma[j] * majorProfile[j];
      }

      // Correlate with minor profile
      let minorCorr = 0;
      for (let j = 0; j < 12; j++) {
        minorCorr += rotatedChroma[j] * minorProfile[j];
      }

      const note = NOTE_NAMES[(12 - i) % 12];

      if (majorCorr > bestKey.confidence) {
        if (bestKey.confidence > 0) alternates.push(bestKey);
        bestKey = { root: note, mode: 'major', confidence: majorCorr };
      } else if (majorCorr > 0.5) {
        alternates.push({ root: note, mode: 'major', confidence: majorCorr });
      }

      if (minorCorr > bestKey.confidence) {
        if (bestKey.confidence > 0) alternates.push(bestKey);
        bestKey = { root: note, mode: 'minor', confidence: minorCorr };
      } else if (minorCorr > 0.5) {
        alternates.push({ root: note, mode: 'minor', confidence: minorCorr });
      }
    }

    // Normalize confidence
    const maxConf = Math.max(bestKey.confidence, ...alternates.map(a => a.confidence));
    bestKey.confidence = bestKey.confidence / maxConf;
    alternates.forEach(a => a.confidence = a.confidence / maxConf);

    return {
      ...bestKey,
      alternates: alternates.sort((a, b) => b.confidence - a.confidence).slice(0, 3),
    };
  }, []);

  // Detect chords from audio
  const detectChords = useCallback(async (buffer: AudioBuffer): Promise<Chord[]> => {
    const chords: Chord[] = [];
    const sampleRate = buffer.sampleRate;
    const channelData = buffer.getChannelData(0);

    // Analyze in windows
    const windowSize = Math.floor(sampleRate * 0.5); // 0.5 second windows
    const hopSize = Math.floor(windowSize / 2);

    for (let offset = 0; offset < channelData.length - windowSize; offset += hopSize) {
      const time = offset / sampleRate;

      // Simplified chord detection based on chroma
      const chromaBins = new Float32Array(12).fill(0);

      const chunk = channelData.slice(offset, offset + windowSize);
      for (let i = 0; i < chunk.length; i++) {
        const amplitude = Math.abs(chunk[i]);
        // Map to chroma (simplified)
        chromaBins[i % 12] += amplitude;
      }

      // Normalize
      const max = Math.max(...chromaBins);
      for (let i = 0; i < 12; i++) chromaBins[i] /= max || 1;

      // Find best matching chord
      let bestChord: Chord | null = null;
      let bestScore = 0;

      for (let root = 0; root < 12; root++) {
        for (const [quality, template] of Object.entries(CHORD_TEMPLATES)) {
          let score = 0;
          for (const interval of template) {
            score += chromaBins[(root + interval) % 12];
          }
          score /= template.length;

          if (score > bestScore && score > 0.3) {
            bestScore = score;
            bestChord = {
              root: NOTE_NAMES[root],
              quality: quality as ChordQuality,
              time,
              duration: hopSize / sampleRate,
              confidence: score,
            };
          }
        }
      }

      if (bestChord) {
        // Merge with previous chord if same
        if (chords.length > 0) {
          const prev = chords[chords.length - 1];
          if (prev.root === bestChord.root && prev.quality === bestChord.quality) {
            prev.duration += bestChord.duration;
            continue;
          }
        }
        chords.push(bestChord);
      }
    }

    return chords;
  }, []);

  // Get scale info
  const getScaleInfo = useCallback((root: NoteName, scaleName: string): ScaleInfo => {
    const pattern = SCALE_PATTERNS[scaleName];
    if (!pattern) {
      return {
        name: scaleName,
        notes: [],
        intervals: [],
        mood: 'Unknown',
        genres: [],
        relatedChords: [],
      };
    }

    const rootIndex = NOTE_NAMES.indexOf(root);
    const notes = pattern.intervals.map(i => NOTE_NAMES[(rootIndex + i) % 12]);

    // Get related chords
    const relatedChords: string[] = [];
    if (scaleName.includes('Major') || scaleName === 'Lydian' || scaleName === 'Mixolydian') {
      relatedChords.push(`${root}`, `${root}maj7`, `${notes[3]}`, `${notes[4]}`);
    } else {
      relatedChords.push(`${root}m`, `${root}m7`, `${notes[2]}`, `${notes[6]}`);
    }

    return {
      name: `${root} ${scaleName}`,
      notes,
      intervals: pattern.intervals,
      mood: pattern.mood,
      genres: pattern.genres,
      relatedChords,
    };
  }, []);

  // Analyze progression
  const analyzeProgression = useCallback((chords: Chord[], key: MusicalKey): ChordProgression[] => {
    if (chords.length < 2) return [];

    const progressions: ChordProgression[] = [];
    const rootIndex = NOTE_NAMES.indexOf(key.root);

    // Convert chords to roman numerals
    const romanNumerals = chords.map(chord => {
      const chordRoot = NOTE_NAMES.indexOf(chord.root);
      const degree = (chordRoot - rootIndex + 12) % 12;

      const degreeToNumeral: Record<number, string> = {
        0: 'I', 2: 'II', 4: 'III', 5: 'IV', 7: 'V', 9: 'VI', 11: 'VII',
        1: 'bII', 3: 'bIII', 6: 'bV', 8: 'bVI', 10: 'bVII',
      };

      let numeral = degreeToNumeral[degree] || '?';
      if (chord.quality === 'minor') numeral = numeral.toLowerCase();
      if (chord.quality.includes('7')) numeral += '7';

      return numeral;
    });

    // Find matching common progressions
    for (let i = 0; i <= romanNumerals.length - 4; i++) {
      const slice = romanNumerals.slice(i, i + 4);
      const sliceStr = slice.join('-').toUpperCase().replace(/7/g, '');

      for (const [pattern, info] of Object.entries(COMMON_PROGRESSIONS)) {
        const patternNumerals = info.numerals.join('-').toUpperCase();
        if (sliceStr === patternNumerals || sliceStr.includes(patternNumerals)) {
          progressions.push({
            chords: chords.slice(i, i + 4),
            romanNumerals: slice,
            name: pattern,
            commonNames: [info.name],
          });
        }
      }
    }

    // If no match found, return the progression as-is
    if (progressions.length === 0 && chords.length >= 2) {
      progressions.push({
        chords,
        romanNumerals,
      });
    }

    return progressions;
  }, []);

  // Full analysis
  const analyze = useCallback(async (buffer: AudioBuffer): Promise<HarmonicAnalysis> => {
    setState(prev => ({
      ...prev,
      isAnalyzing: true,
      progress: 0,
      error: null,
    }));

    try {
      // Detect key
      setState(prev => ({ ...prev, progress: 20 }));
      const key = await detectKey(buffer);

      // Detect chords
      setState(prev => ({ ...prev, progress: 50 }));
      const chords = await detectChords(buffer);

      // Analyze progressions
      setState(prev => ({ ...prev, progress: 70 }));
      const progressions = analyzeProgression(chords, key);

      // Generate suggestions
      const suggestions: TheorySuggestion[] = [];

      // Suggest relative key
      if (key.mode === 'major') {
        const relMinorRoot = NOTE_NAMES[(NOTE_NAMES.indexOf(key.root) + 9) % 12];
        suggestions.push({
          type: 'modulation',
          description: `Try modulating to the relative minor (${relMinorRoot}m)`,
          theory: 'Relative keys share the same notes, making modulation smooth',
        });
      } else {
        const relMajorRoot = NOTE_NAMES[(NOTE_NAMES.indexOf(key.root) + 3) % 12];
        suggestions.push({
          type: 'modulation',
          description: `Try modulating to the relative major (${relMajorRoot})`,
          theory: 'Relative keys share the same notes, making modulation smooth',
        });
      }

      // Suggest scale variations
      suggestions.push({
        type: 'scale',
        description: key.mode === 'major'
          ? `Try the Lydian mode for a dreamy sound (#4)`
          : `Try the Dorian mode for a brighter minor sound (natural 6)`,
        example: key.mode === 'major'
          ? `${key.root} Lydian: raise the 4th degree`
          : `${key.root} Dorian: raise the 6th degree`,
      });

      // Suggest chord substitutions
      if (chords.length > 0) {
        const lastChord = chords[chords.length - 1];
        if (lastChord.quality === 'major') {
          suggestions.push({
            type: 'chord',
            description: `Try substituting ${lastChord.root} with ${lastChord.root}maj7 or ${lastChord.root}add9`,
            theory: 'Extended chords add color and sophistication',
          });
        }
      }

      // Get available scales for this key
      const availableScales = Object.keys(SCALE_PATTERNS).map(name =>
        getScaleInfo(key.root, name)
      );

      const analysis: HarmonicAnalysis = {
        key,
        chords,
        progressions,
        tempo: 120, // Would be detected by mycelix-mir
        timeSignature: [4, 4],
        sections: [], // Would be detected by mycelix-mir
        suggestions,
      };

      setState(prev => ({
        ...prev,
        isAnalyzing: false,
        progress: 100,
        analysis,
        availableScales,
        selectedScale: getScaleInfo(key.root, key.mode === 'major' ? 'Major' : 'Natural Minor'),
      }));

      return analysis;
    } catch (error) {
      setState(prev => ({
        ...prev,
        isAnalyzing: false,
        progress: 0,
        error: error instanceof Error ? error.message : 'Analysis failed',
      }));
      throw error;
    }
  }, [detectKey, detectChords, analyzeProgression, getScaleInfo]);

  // Select a scale
  const selectScale = useCallback((scaleName: string) => {
    if (!state.analysis) return;

    const scaleInfo = getScaleInfo(state.analysis.key.root, scaleName);
    setState(prev => ({ ...prev, selectedScale: scaleInfo }));
  }, [state.analysis, getScaleInfo]);

  // Get chord for scale degree
  const getChordForDegree = useCallback((degree: number, quality: ChordQuality = 'major'): Chord | null => {
    if (!state.analysis) return null;

    const scaleNotes = state.selectedScale?.notes || [];
    if (degree < 1 || degree > scaleNotes.length) return null;

    return {
      root: scaleNotes[degree - 1],
      quality,
      time: 0,
      duration: 1,
      confidence: 1,
    };
  }, [state.analysis, state.selectedScale]);

  // Suggest next chord
  const suggestNextChord = useCallback((currentChord: Chord): Chord[] => {
    if (!state.analysis) return [];

    const { key } = state.analysis;
    const rootIndex = NOTE_NAMES.indexOf(currentChord.root);
    const suggestions: Chord[] = [];

    // Common movements based on function
    const movements = [
      { interval: 5, quality: 'major' as ChordQuality }, // V
      { interval: 7, quality: 'major' as ChordQuality }, // IV
      { interval: 9, quality: 'minor' as ChordQuality }, // vi
      { interval: 2, quality: 'minor' as ChordQuality }, // ii
    ];

    for (const { interval, quality } of movements) {
      const newRoot = NOTE_NAMES[(rootIndex + interval) % 12];
      suggestions.push({
        root: newRoot,
        quality,
        time: 0,
        duration: 1,
        confidence: 0.8,
      });
    }

    return suggestions;
  }, [state.analysis]);

  return {
    ...state,
    analyze,
    selectScale,
    getScaleInfo,
    getChordForDegree,
    suggestNextChord,
    scalePatterns: Object.keys(SCALE_PATTERNS),
    commonProgressions: COMMON_PROGRESSIONS,
  };
}

export default useMusicTheory;
