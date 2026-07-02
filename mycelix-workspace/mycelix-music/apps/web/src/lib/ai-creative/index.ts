// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * AI Creative Suite
 *
 * Generative AI for music creation:
 * - AI Composition (melody, chords, drums)
 * - Voice Synthesis (text-to-singing)
 * - Smart Remixing
 * - Cover Art Generation
 */

// ==================== Types ====================

export interface CompositionPrompt {
  style?: string;
  mood?: string;
  tempo?: number;
  key?: string;
  timeSignature?: string;
  instruments?: string[];
  duration?: number;
  reference?: string; // URL to reference track
}

export interface GeneratedMelody {
  notes: Array<{
    pitch: number;
    start: number;
    duration: number;
    velocity: number;
  }>;
  key: string;
  tempo: number;
  duration: number;
}

export interface GeneratedChords {
  progression: Array<{
    chord: string;
    notes: number[];
    start: number;
    duration: number;
  }>;
  key: string;
  tempo: number;
}

export interface GeneratedDrums {
  pattern: Array<{
    instrument: 'kick' | 'snare' | 'hihat' | 'tom' | 'crash' | 'ride';
    start: number;
    duration: number;
    velocity: number;
  }>;
  tempo: number;
  timeSignature: string;
}

export interface VoiceSynthesisParams {
  text: string;
  voice?: string;
  style?: 'speech' | 'singing' | 'rap';
  pitch?: number;
  speed?: number;
  emotion?: string;
  melody?: GeneratedMelody;
}

export interface SynthesizedVoice {
  audioUrl: string;
  duration: number;
  waveform: Float32Array;
}

export interface VoiceCloneParams {
  name: string;
  samples: File[];
  description?: string;
}

export interface ClonedVoice {
  id: string;
  name: string;
  previewUrl: string;
  createdAt: Date;
}

export interface RemixParams {
  trackId: string;
  style: string;
  keepVocals?: boolean;
  targetTempo?: number;
  targetKey?: string;
  intensity?: number;
}

export interface RemixResult {
  audioUrl: string;
  stems: {
    vocals?: string;
    drums?: string;
    bass?: string;
    other?: string;
  };
  analysis: {
    originalTempo: number;
    originalKey: string;
    appliedChanges: string[];
  };
}

export interface CoverArtParams {
  prompt?: string;
  style?: 'abstract' | 'realistic' | 'anime' | 'vintage' | 'minimalist' | 'psychedelic';
  colorScheme?: string[];
  mood?: string;
  referenceImage?: File;
  audioAnalysis?: {
    genre: string;
    mood: string;
    energy: number;
  };
}

export interface GeneratedArt {
  imageUrl: string;
  thumbnailUrl: string;
  variations: string[];
  prompt: string;
}

// ==================== AI Composition Service ====================

export class AICompositionService {
  private baseUrl: string;

  constructor(baseUrl = '/api/ai') {
    this.baseUrl = baseUrl;
  }

  async generateMelody(prompt: CompositionPrompt): Promise<GeneratedMelody> {
    const response = await fetch(`${this.baseUrl}/compose/melody`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(prompt),
    });
    return response.json();
  }

  async generateChords(prompt: CompositionPrompt): Promise<GeneratedChords> {
    const response = await fetch(`${this.baseUrl}/compose/chords`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(prompt),
    });
    return response.json();
  }

  async generateDrums(prompt: CompositionPrompt): Promise<GeneratedDrums> {
    const response = await fetch(`${this.baseUrl}/compose/drums`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(prompt),
    });
    return response.json();
  }

  async generateFullArrangement(prompt: CompositionPrompt): Promise<{
    melody: GeneratedMelody;
    chords: GeneratedChords;
    drums: GeneratedDrums;
    bassline: GeneratedMelody;
    midiUrl: string;
    audioPreviewUrl: string;
  }> {
    const response = await fetch(`${this.baseUrl}/compose/full`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(prompt),
    });
    return response.json();
  }

  async continueSection(
    existingMidi: ArrayBuffer,
    direction: 'forward' | 'variation' | 'bridge' | 'chorus'
  ): Promise<GeneratedMelody> {
    const formData = new FormData();
    formData.append('midi', new Blob([existingMidi]));
    formData.append('direction', direction);

    const response = await fetch(`${this.baseUrl}/compose/continue`, {
      method: 'POST',
      body: formData,
    });
    return response.json();
  }

  async harmonize(melody: GeneratedMelody): Promise<GeneratedChords> {
    const response = await fetch(`${this.baseUrl}/compose/harmonize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ melody }),
    });
    return response.json();
  }

  async suggestNextChord(currentProgression: string[]): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/compose/next-chord`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ progression: currentProgression }),
    });
    return response.json();
  }

  // On-device generation using local models
  async generateLocalMelody(prompt: CompositionPrompt): Promise<GeneratedMelody> {
    // This would use local TensorFlow.js model
    const notes: GeneratedMelody['notes'] = [];
    const tempo = prompt.tempo || 120;
    const duration = prompt.duration || 8;
    const beatsPerMeasure = 4;
    const totalBeats = duration * (tempo / 60);

    // Simple generative algorithm
    const scale = this.getScale(prompt.key || 'C major');
    let currentPitch = scale[Math.floor(scale.length / 2)];

    for (let beat = 0; beat < totalBeats; beat += 0.25) {
      if (Math.random() > 0.3) {
        const step = Math.floor(Math.random() * 5) - 2;
        const newIndex = Math.max(0, Math.min(scale.length - 1, scale.indexOf(currentPitch) + step));
        currentPitch = scale[newIndex];

        notes.push({
          pitch: currentPitch,
          start: beat * (60 / tempo),
          duration: (0.25 + Math.random() * 0.5) * (60 / tempo),
          velocity: 80 + Math.floor(Math.random() * 40),
        });
      }
    }

    return {
      notes,
      key: prompt.key || 'C major',
      tempo,
      duration,
    };
  }

  private getScale(key: string): number[] {
    const roots: Record<string, number> = {
      'C': 60, 'C#': 61, 'D': 62, 'D#': 63, 'E': 64, 'F': 65,
      'F#': 66, 'G': 67, 'G#': 68, 'A': 69, 'A#': 70, 'B': 71,
    };

    const patterns: Record<string, number[]> = {
      'major': [0, 2, 4, 5, 7, 9, 11],
      'minor': [0, 2, 3, 5, 7, 8, 10],
    };

    const [rootNote, mode] = key.split(' ');
    const root = roots[rootNote] || 60;
    const pattern = patterns[mode] || patterns['major'];

    return pattern.map(interval => root + interval);
  }
}

// ==================== Voice Synthesis Service ====================

export class VoiceSynthesisService {
  private baseUrl: string;

  constructor(baseUrl = '/api/ai') {
    this.baseUrl = baseUrl;
  }

  async getAvailableVoices(): Promise<Array<{
    id: string;
    name: string;
    language: string;
    style: string;
    previewUrl: string;
  }>> {
    const response = await fetch(`${this.baseUrl}/voice/voices`);
    return response.json();
  }

  async synthesize(params: VoiceSynthesisParams): Promise<SynthesizedVoice> {
    const response = await fetch(`${this.baseUrl}/voice/synthesize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return response.json();
  }

  async textToSinging(
    lyrics: string,
    melody: GeneratedMelody,
    voice: string
  ): Promise<SynthesizedVoice> {
    const response = await fetch(`${this.baseUrl}/voice/sing`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lyrics, melody, voice }),
    });
    return response.json();
  }

  async cloneVoice(params: VoiceCloneParams): Promise<ClonedVoice> {
    const formData = new FormData();
    formData.append('name', params.name);
    if (params.description) formData.append('description', params.description);
    params.samples.forEach((sample, i) => formData.append(`sample${i}`, sample));

    const response = await fetch(`${this.baseUrl}/voice/clone`, {
      method: 'POST',
      body: formData,
    });
    return response.json();
  }

  async getMyClonedVoices(): Promise<ClonedVoice[]> {
    const response = await fetch(`${this.baseUrl}/voice/clones`);
    return response.json();
  }

  async deleteClonedVoice(voiceId: string): Promise<void> {
    await fetch(`${this.baseUrl}/voice/clones/${voiceId}`, { method: 'DELETE' });
  }

  async styleTransfer(
    audioFile: File,
    targetVoice: string,
    preservePitch?: boolean
  ): Promise<SynthesizedVoice> {
    const formData = new FormData();
    formData.append('audio', audioFile);
    formData.append('targetVoice', targetVoice);
    if (preservePitch) formData.append('preservePitch', 'true');

    const response = await fetch(`${this.baseUrl}/voice/transfer`, {
      method: 'POST',
      body: formData,
    });
    return response.json();
  }
}

// ==================== Smart Remix Service ====================

export class SmartRemixService {
  private baseUrl: string;

  constructor(baseUrl = '/api/ai') {
    this.baseUrl = baseUrl;
  }

  async getRemixStyles(): Promise<Array<{
    id: string;
    name: string;
    description: string;
    previewUrl: string;
  }>> {
    const response = await fetch(`${this.baseUrl}/remix/styles`);
    return response.json();
  }

  async createRemix(params: RemixParams): Promise<RemixResult> {
    const response = await fetch(`${this.baseUrl}/remix/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return response.json();
  }

  async separateStems(audioFile: File): Promise<{
    vocals: string;
    drums: string;
    bass: string;
    other: string;
  }> {
    const formData = new FormData();
    formData.append('audio', audioFile);

    const response = await fetch(`${this.baseUrl}/remix/separate`, {
      method: 'POST',
      body: formData,
    });
    return response.json();
  }

  async matchTempo(audioFile: File, targetTempo: number): Promise<string> {
    const formData = new FormData();
    formData.append('audio', audioFile);
    formData.append('tempo', targetTempo.toString());

    const response = await fetch(`${this.baseUrl}/remix/tempo`, {
      method: 'POST',
      body: formData,
    });
    const { audioUrl } = await response.json();
    return audioUrl;
  }

  async matchKey(audioFile: File, targetKey: string): Promise<string> {
    const formData = new FormData();
    formData.append('audio', audioFile);
    formData.append('key', targetKey);

    const response = await fetch(`${this.baseUrl}/remix/key`, {
      method: 'POST',
      body: formData,
    });
    const { audioUrl } = await response.json();
    return audioUrl;
  }

  async mashup(
    tracks: Array<{ id: string; sections: Array<{ start: number; end: number }> }>,
    targetTempo: number,
    targetKey: string
  ): Promise<RemixResult> {
    const response = await fetch(`${this.baseUrl}/remix/mashup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tracks, targetTempo, targetKey }),
    });
    return response.json();
  }
}

// ==================== Cover Art Generator Service ====================

export class CoverArtService {
  private baseUrl: string;

  constructor(baseUrl = '/api/ai') {
    this.baseUrl = baseUrl;
  }

  async generate(params: CoverArtParams): Promise<GeneratedArt> {
    const formData = new FormData();

    if (params.prompt) formData.append('prompt', params.prompt);
    if (params.style) formData.append('style', params.style);
    if (params.colorScheme) formData.append('colorScheme', JSON.stringify(params.colorScheme));
    if (params.mood) formData.append('mood', params.mood);
    if (params.referenceImage) formData.append('reference', params.referenceImage);
    if (params.audioAnalysis) formData.append('audioAnalysis', JSON.stringify(params.audioAnalysis));

    const response = await fetch(`${this.baseUrl}/art/generate`, {
      method: 'POST',
      body: formData,
    });
    return response.json();
  }

  async generateFromTrack(trackId: string, style?: string): Promise<GeneratedArt> {
    const response = await fetch(`${this.baseUrl}/art/from-track`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ trackId, style }),
    });
    return response.json();
  }

  async generateVariations(imageUrl: string, count = 4): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/art/variations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imageUrl, count }),
    });
    const { variations } = await response.json();
    return variations;
  }

  async upscale(imageUrl: string, scale = 2): Promise<string> {
    const response = await fetch(`${this.baseUrl}/art/upscale`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imageUrl, scale }),
    });
    const { upscaledUrl } = await response.json();
    return upscaledUrl;
  }

  async removeBackground(imageUrl: string): Promise<string> {
    const response = await fetch(`${this.baseUrl}/art/remove-background`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imageUrl }),
    });
    const { transparentUrl } = await response.json();
    return transparentUrl;
  }

  async getStyles(): Promise<Array<{
    id: string;
    name: string;
    description: string;
    examples: string[];
  }>> {
    const response = await fetch(`${this.baseUrl}/art/styles`);
    return response.json();
  }
}

// ==================== AI Creative Manager ====================

export class AICreativeManager {
  public readonly composition: AICompositionService;
  public readonly voice: VoiceSynthesisService;
  public readonly remix: SmartRemixService;
  public readonly art: CoverArtService;

  constructor(baseUrl = '/api/ai') {
    this.composition = new AICompositionService(baseUrl);
    this.voice = new VoiceSynthesisService(baseUrl);
    this.remix = new SmartRemixService(baseUrl);
    this.art = new CoverArtService(baseUrl);
  }
}

// ==================== Singleton ====================

let aiCreativeManager: AICreativeManager | null = null;

export function getAICreativeManager(baseUrl?: string): AICreativeManager {
  if (!aiCreativeManager) {
    aiCreativeManager = new AICreativeManager(baseUrl);
  }
  return aiCreativeManager;
}

export default {
  AICreativeManager,
  getAICreativeManager,
  AICompositionService,
  VoiceSynthesisService,
  SmartRemixService,
  CoverArtService,
};
