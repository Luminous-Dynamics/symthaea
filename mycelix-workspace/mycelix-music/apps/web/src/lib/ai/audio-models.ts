// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Audio AI Models
 *
 * Music-specific AI capabilities:
 * - Genre classification
 * - Mood detection
 * - Instrument recognition
 * - Beat/tempo detection
 * - Key detection
 * - Source separation (stems)
 * - Audio transcription
 * - Similarity analysis
 */

import * as tf from '@tensorflow/tfjs';
import { getModelRuntime, preprocessors, postprocessors, ModelConfig, InferenceResult } from './model-runtime';

// ==================== Model Configurations ====================

export const AUDIO_MODEL_CONFIGS: Record<string, ModelConfig> = {
  genreClassifier: {
    id: 'genre-classifier-v1',
    name: 'Genre Classifier',
    version: '1.0.0',
    format: 'tfjs',
    url: '/models/genre-classifier/model.json',
    inputShape: [1, 128, 128, 1], // Mel spectrogram
    outputShape: [1, 10],
    labels: [
      'blues', 'classical', 'country', 'disco', 'hiphop',
      'jazz', 'metal', 'pop', 'reggae', 'rock'
    ],
    preprocessor: 'melSpectrogram',
    metadata: {
      sampleRate: 22050,
      duration: 3, // seconds
      fftSize: 2048,
      hopSize: 512,
      nMels: 128,
    },
  },

  moodDetector: {
    id: 'mood-detector-v1',
    name: 'Mood Detector',
    version: '1.0.0',
    format: 'tfjs',
    url: '/models/mood-detector/model.json',
    inputShape: [1, 128, 128, 1],
    outputShape: [1, 8],
    labels: [
      'happy', 'sad', 'energetic', 'calm',
      'angry', 'romantic', 'nostalgic', 'mysterious'
    ],
    preprocessor: 'melSpectrogram',
  },

  instrumentRecognizer: {
    id: 'instrument-recognizer-v1',
    name: 'Instrument Recognizer',
    version: '1.0.0',
    format: 'tfjs',
    url: '/models/instrument-recognizer/model.json',
    inputShape: [1, 128, 128, 1],
    outputShape: [1, 15],
    labels: [
      'acoustic_guitar', 'bass', 'drums', 'electric_guitar',
      'keyboard', 'piano', 'saxophone', 'strings', 'synth',
      'trumpet', 'violin', 'flute', 'voice', 'organ', 'percussion'
    ],
    preprocessor: 'melSpectrogram',
  },

  beatTracker: {
    id: 'beat-tracker-v1',
    name: 'Beat Tracker',
    version: '1.0.0',
    format: 'onnx',
    url: '/models/beat-tracker/model.onnx',
    inputShape: [1, 1, 22050 * 10], // 10 seconds of audio
    outputShape: [1, 2], // [beat_times, downbeat_times]
    preprocessor: 'audioNormalize',
  },

  keyDetector: {
    id: 'key-detector-v1',
    name: 'Key Detector',
    version: '1.0.0',
    format: 'tfjs',
    url: '/models/key-detector/model.json',
    inputShape: [1, 12, 128], // Chromagram
    outputShape: [1, 24],
    labels: [
      'C major', 'C# major', 'D major', 'D# major', 'E major', 'F major',
      'F# major', 'G major', 'G# major', 'A major', 'A# major', 'B major',
      'C minor', 'C# minor', 'D minor', 'D# minor', 'E minor', 'F minor',
      'F# minor', 'G minor', 'G# minor', 'A minor', 'A# minor', 'B minor'
    ],
    preprocessor: 'chromagram',
  },

  stemSeparator: {
    id: 'stem-separator-v1',
    name: 'Stem Separator',
    version: '1.0.0',
    format: 'onnx',
    url: '/models/stem-separator/model.onnx',
    inputShape: [1, 2, 44100 * 30], // 30 seconds stereo
    outputShape: [4, 2, 44100 * 30], // 4 stems: vocals, drums, bass, other
    metadata: {
      stems: ['vocals', 'drums', 'bass', 'other'],
      sampleRate: 44100,
    },
  },

  audioEmbedding: {
    id: 'audio-embedding-v1',
    name: 'Audio Embedding',
    version: '1.0.0',
    format: 'tfjs',
    url: '/models/audio-embedding/model.json',
    inputShape: [1, 128, 128, 1],
    outputShape: [1, 256], // 256-dimensional embedding
    preprocessor: 'melSpectrogram',
  },
};

// ==================== Genre Classification ====================

export interface GenreResult {
  genre: string;
  confidence: number;
  allGenres: Record<string, number>;
}

export async function classifyGenre(
  audio: Float32Array,
  sampleRate: number
): Promise<GenreResult> {
  const runtime = getModelRuntime();
  await runtime.initialize();

  const config = AUDIO_MODEL_CONFIGS.genreClassifier;
  await runtime.loadModel(config);

  // Preprocess audio to mel spectrogram
  const melSpec = await preprocessors.melSpectrogram(
    audio,
    sampleRate,
    config.metadata?.fftSize as number,
    config.metadata?.hopSize as number,
    config.metadata?.nMels as number
  );

  // Resize to model input shape
  const resized = tf.image.resizeBilinear(
    melSpec.expandDims(-1).expandDims(0) as tf.Tensor4D,
    [128, 128]
  );

  const result = await runtime.predict<{ label: string; probabilities: Record<string, number> }>(
    config.id,
    resized,
    { labels: config.labels }
  );

  melSpec.dispose();
  resized.dispose();

  return {
    genre: result.output.label,
    confidence: result.confidence || 0,
    allGenres: result.output.probabilities,
  };
}

// ==================== Mood Detection ====================

export interface MoodResult {
  primaryMood: string;
  confidence: number;
  moods: Record<string, number>;
  valence: number; // -1 (negative) to 1 (positive)
  energy: number; // 0 (calm) to 1 (energetic)
}

export async function detectMood(
  audio: Float32Array,
  sampleRate: number
): Promise<MoodResult> {
  const runtime = getModelRuntime();
  await runtime.initialize();

  const config = AUDIO_MODEL_CONFIGS.moodDetector;
  await runtime.loadModel(config);

  const melSpec = await preprocessors.melSpectrogram(audio, sampleRate);
  const resized = tf.image.resizeBilinear(
    melSpec.expandDims(-1).expandDims(0) as tf.Tensor4D,
    [128, 128]
  );

  const result = await runtime.predict<{ label: string; probabilities: Record<string, number> }>(
    config.id,
    resized,
    { labels: config.labels }
  );

  melSpec.dispose();
  resized.dispose();

  // Calculate valence and energy from mood probabilities
  const probs = result.output.probabilities;
  const valence =
    (probs.happy || 0) * 1 +
    (probs.romantic || 0) * 0.5 +
    (probs.energetic || 0) * 0.3 -
    (probs.sad || 0) * 1 -
    (probs.angry || 0) * 0.5 -
    (probs.nostalgic || 0) * 0.3;

  const energy =
    (probs.energetic || 0) * 1 +
    (probs.angry || 0) * 0.8 +
    (probs.happy || 0) * 0.6 -
    (probs.calm || 0) * 1 -
    (probs.sad || 0) * 0.5;

  return {
    primaryMood: result.output.label,
    confidence: result.confidence || 0,
    moods: probs,
    valence: Math.max(-1, Math.min(1, valence)),
    energy: Math.max(0, Math.min(1, (energy + 1) / 2)),
  };
}

// ==================== Instrument Recognition ====================

export interface InstrumentResult {
  instruments: Array<{ name: string; confidence: number }>;
  dominantInstrument: string;
}

export async function recognizeInstruments(
  audio: Float32Array,
  sampleRate: number,
  threshold = 0.3
): Promise<InstrumentResult> {
  const runtime = getModelRuntime();
  await runtime.initialize();

  const config = AUDIO_MODEL_CONFIGS.instrumentRecognizer;
  await runtime.loadModel(config);

  const melSpec = await preprocessors.melSpectrogram(audio, sampleRate);
  const resized = tf.image.resizeBilinear(
    melSpec.expandDims(-1).expandDims(0) as tf.Tensor4D,
    [128, 128]
  );

  const result = await runtime.predict<{ label: string; probabilities: Record<string, number> }>(
    config.id,
    resized,
    { labels: config.labels }
  );

  melSpec.dispose();
  resized.dispose();

  // Filter instruments above threshold
  const instruments = Object.entries(result.output.probabilities)
    .filter(([, conf]) => conf >= threshold)
    .sort((a, b) => b[1] - a[1])
    .map(([name, confidence]) => ({ name, confidence }));

  return {
    instruments,
    dominantInstrument: instruments[0]?.name || 'unknown',
  };
}

// ==================== Beat/Tempo Detection ====================

export interface BeatResult {
  tempo: number; // BPM
  beatTimes: number[]; // seconds
  downbeatTimes: number[];
  confidence: number;
  timeSignature: string;
}

export async function detectBeats(
  audio: Float32Array,
  sampleRate: number
): Promise<BeatResult> {
  // Simple onset detection for beats
  const hopSize = Math.floor(sampleRate / 100); // 10ms hops
  const frameSize = Math.floor(sampleRate / 10); // 100ms frames

  const energies: number[] = [];
  for (let i = 0; i + frameSize <= audio.length; i += hopSize) {
    let energy = 0;
    for (let j = 0; j < frameSize; j++) {
      energy += audio[i + j] ** 2;
    }
    energies.push(energy / frameSize);
  }

  // Compute onset strength (difference in energy)
  const onsetStrength: number[] = [];
  for (let i = 1; i < energies.length; i++) {
    onsetStrength.push(Math.max(0, energies[i] - energies[i - 1]));
  }

  // Find peaks (beats)
  const threshold = Math.max(...onsetStrength) * 0.3;
  const beatFrames: number[] = [];

  for (let i = 1; i < onsetStrength.length - 1; i++) {
    if (
      onsetStrength[i] > threshold &&
      onsetStrength[i] > onsetStrength[i - 1] &&
      onsetStrength[i] > onsetStrength[i + 1]
    ) {
      beatFrames.push(i);
    }
  }

  // Convert to time
  const beatTimes = beatFrames.map(f => (f * hopSize) / sampleRate);

  // Estimate tempo from inter-beat intervals
  const ibis: number[] = [];
  for (let i = 1; i < beatTimes.length; i++) {
    ibis.push(beatTimes[i] - beatTimes[i - 1]);
  }

  const medianIBI = ibis.sort((a, b) => a - b)[Math.floor(ibis.length / 2)] || 0.5;
  const tempo = medianIBI > 0 ? 60 / medianIBI : 120;

  // Estimate downbeats (every 4th beat for 4/4)
  const downbeatTimes = beatTimes.filter((_, i) => i % 4 === 0);

  return {
    tempo: Math.round(tempo),
    beatTimes,
    downbeatTimes,
    confidence: 0.8,
    timeSignature: '4/4',
  };
}

// ==================== Key Detection ====================

export interface KeyResult {
  key: string;
  mode: 'major' | 'minor';
  confidence: number;
  allKeys: Record<string, number>;
}

export async function detectKey(
  audio: Float32Array,
  sampleRate: number
): Promise<KeyResult> {
  const runtime = getModelRuntime();
  await runtime.initialize();

  const config = AUDIO_MODEL_CONFIGS.keyDetector;
  await runtime.loadModel(config);

  const chromagram = await preprocessors.chromagram(audio, sampleRate);

  const result = await runtime.predict<{ label: string; probabilities: Record<string, number> }>(
    config.id,
    chromagram,
    { labels: config.labels }
  );

  chromagram.dispose();

  const label = result.output.label;
  const [key, mode] = label.split(' ');

  return {
    key,
    mode: mode as 'major' | 'minor',
    confidence: result.confidence || 0,
    allKeys: result.output.probabilities,
  };
}

// ==================== Source Separation ====================

export interface StemResult {
  vocals: Float32Array;
  drums: Float32Array;
  bass: Float32Array;
  other: Float32Array;
}

export async function separateStems(
  audio: Float32Array,
  _sampleRate: number
): Promise<StemResult> {
  // Placeholder for stem separation
  // Real implementation would use a large model like Demucs/Spleeter

  const length = audio.length;

  // For now, return copies (real implementation would separate)
  return {
    vocals: new Float32Array(length),
    drums: new Float32Array(length),
    bass: new Float32Array(length),
    other: new Float32Array(audio), // Original as "other" stem
  };
}

// ==================== Audio Similarity ====================

export interface SimilarityResult {
  similarity: number; // 0-1
  embedding1: Float32Array;
  embedding2: Float32Array;
}

export async function computeAudioEmbedding(
  audio: Float32Array,
  sampleRate: number
): Promise<Float32Array> {
  const runtime = getModelRuntime();
  await runtime.initialize();

  const config = AUDIO_MODEL_CONFIGS.audioEmbedding;
  await runtime.loadModel(config);

  const melSpec = await preprocessors.melSpectrogram(audio, sampleRate);
  const resized = tf.image.resizeBilinear(
    melSpec.expandDims(-1).expandDims(0) as tf.Tensor4D,
    [128, 128]
  );

  const result = await runtime.predict<number[]>(config.id, resized);

  melSpec.dispose();
  resized.dispose();

  return new Float32Array(result.output);
}

export async function computeSimilarity(
  audio1: Float32Array,
  audio2: Float32Array,
  sampleRate: number
): Promise<SimilarityResult> {
  const [embedding1, embedding2] = await Promise.all([
    computeAudioEmbedding(audio1, sampleRate),
    computeAudioEmbedding(audio2, sampleRate),
  ]);

  // Cosine similarity
  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;

  for (let i = 0; i < embedding1.length; i++) {
    dotProduct += embedding1[i] * embedding2[i];
    norm1 += embedding1[i] ** 2;
    norm2 += embedding2[i] ** 2;
  }

  const similarity = dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2) + 1e-8);

  return {
    similarity: (similarity + 1) / 2, // Convert from [-1, 1] to [0, 1]
    embedding1,
    embedding2,
  };
}

// ==================== Comprehensive Analysis ====================

export interface AudioAnalysis {
  genre: GenreResult;
  mood: MoodResult;
  instruments: InstrumentResult;
  beats: BeatResult;
  key: KeyResult;
  duration: number;
  processingTime: number;
}

export async function analyzeAudio(
  audio: Float32Array,
  sampleRate: number
): Promise<AudioAnalysis> {
  const startTime = performance.now();
  const duration = audio.length / sampleRate;

  // Run all analyses in parallel
  const [genre, mood, instruments, beats, key] = await Promise.all([
    classifyGenre(audio, sampleRate),
    detectMood(audio, sampleRate),
    recognizeInstruments(audio, sampleRate),
    detectBeats(audio, sampleRate),
    detectKey(audio, sampleRate),
  ]);

  return {
    genre,
    mood,
    instruments,
    beats,
    key,
    duration,
    processingTime: performance.now() - startTime,
  };
}

export default {
  classifyGenre,
  detectMood,
  recognizeInstruments,
  detectBeats,
  detectKey,
  separateStems,
  computeAudioEmbedding,
  computeSimilarity,
  analyzeAudio,
  AUDIO_MODEL_CONFIGS,
};
