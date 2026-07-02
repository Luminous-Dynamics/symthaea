// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * AI Model Pipeline
 *
 * Complete AI system for Mycelix Music:
 * - TensorFlow.js runtime with WebGL/WebGPU acceleration
 * - ONNX runtime for cross-platform models
 * - Audio-specific models (genre, mood, instruments, beats, key)
 * - On-device inference via Web Workers
 * - Custom model training and fine-tuning
 * - Data augmentation and dataset management
 */

// Model Runtime
export {
  ModelRuntime,
  getModelRuntime,
  preprocessors,
  postprocessors,
  type ModelConfig,
  type LoadedModel,
  type InferenceResult,
  type RuntimeConfig,
  type ModelFormat,
  type ModelBackend,
  type AudioDevice,
  type AudioConfig,
} from './model-runtime';

// Audio Models
export {
  AUDIO_MODEL_CONFIGS,
  classifyGenre,
  detectMood,
  recognizeInstruments,
  detectBeats,
  detectKey,
  separateStems,
  computeAudioEmbedding,
  computeSimilarity,
  analyzeAudio,
  type GenreResult,
  type MoodResult,
  type InstrumentResult,
  type BeatResult,
  type KeyResult,
  type StemResult,
  type SimilarityResult,
  type AudioAnalysis,
} from './audio-models';

// React Hooks
export {
  useModelRuntime,
  useAudioAnalysis,
  useGenreClassification,
  useMoodDetection,
  useInstrumentRecognition,
  useBeatDetection,
  useKeyDetection,
  useStemSeparation,
  useAudioSimilarity,
  useAudioEmbedding,
  useRealtimeAnalysis,
  useSmartRecommendations,
  type UseAnalysisOptions,
  type AnalysisState,
  type RealtimeAnalysisOptions,
  type RecommendationOptions,
  type Recommendation,
} from './hooks';

// Worker Client
export {
  InferenceWorkerClient,
  getWorkerClient,
  type WorkerPoolConfig,
} from './worker-client';

// Model Training
export {
  ModelTrainer,
  DatasetManager,
  getModelTrainer,
  augmentation,
  type TrainingExample,
  type Dataset,
  type TrainingConfig,
  type TrainingProgress,
  type TrainingResult,
  type ModelExport,
} from './model-training';

// ==================== Convenience Functions ====================

import { getModelRuntime } from './model-runtime';
import { analyzeAudio as analyzeAudioFn } from './audio-models';
import { getWorkerClient } from './worker-client';
import { getModelTrainer } from './model-training';

/**
 * Initialize the AI system
 */
export async function initializeAI(options?: {
  backend?: 'webgl' | 'webgpu' | 'wasm' | 'cpu';
  useWorkers?: boolean;
  maxWorkers?: number;
}): Promise<void> {
  const { backend = 'webgl', useWorkers = true, maxWorkers } = options || {};

  // Initialize model runtime
  const runtime = getModelRuntime({ backend });
  await runtime.initialize();

  // Initialize workers if enabled
  if (useWorkers) {
    const client = getWorkerClient({ maxWorkers });
    await client.initialize();
  }

  // Initialize training system
  const trainer = getModelTrainer();
  await trainer.initialize();

  console.log('AI system initialized', { backend, useWorkers, maxWorkers });
}

/**
 * Quick audio analysis
 */
export async function quickAnalysis(audioFile: File): Promise<{
  genre: string;
  mood: string;
  tempo: number;
  key: string;
  instruments: string[];
}> {
  // Decode audio file
  const arrayBuffer = await audioFile.arrayBuffer();
  const audioContext = new AudioContext({ sampleRate: 22050 });
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  const audio = audioBuffer.getChannelData(0);
  await audioContext.close();

  // Run analysis
  const analysis = await analyzeAudioFn(audio, 22050);

  return {
    genre: analysis.genre.genre,
    mood: analysis.mood.primaryMood,
    tempo: analysis.beats.tempo,
    key: `${analysis.key.key} ${analysis.key.mode}`,
    instruments: analysis.instruments.instruments.map(i => i.name),
  };
}

/**
 * Get system stats
 */
export function getAIStats(): {
  runtime: { modelCount: number; totalMemory: number; backend: string };
  workers: { active: boolean };
  training: { datasetCount: number };
} {
  const runtime = getModelRuntime();
  const trainer = getModelTrainer();

  return {
    runtime: runtime.getStats(),
    workers: { active: true },
    training: { datasetCount: trainer.getDatasetManager().getAllDatasets().length },
  };
}

/**
 * Cleanup AI resources
 */
export function disposeAI(): void {
  getModelRuntime().dispose();
  getWorkerClient().dispose();
}

export default {
  initializeAI,
  quickAnalysis,
  getAIStats,
  disposeAI,
};
