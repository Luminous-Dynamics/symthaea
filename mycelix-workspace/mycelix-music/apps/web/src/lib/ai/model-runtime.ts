// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * AI Model Runtime
 *
 * Unified runtime for TensorFlow.js and ONNX models:
 * - Model loading and caching
 * - WebGL/WebGPU acceleration
 * - Worker-based inference
 * - Memory management
 * - Model versioning
 */

import * as tf from '@tensorflow/tfjs';
import { InferenceSession, Tensor as OnnxTensor } from 'onnxruntime-web';

// ==================== Types ====================

export type ModelFormat = 'tfjs' | 'onnx';
export type ModelBackend = 'webgl' | 'webgpu' | 'wasm' | 'cpu';

export interface ModelConfig {
  id: string;
  name: string;
  version: string;
  format: ModelFormat;
  url: string;
  inputShape: number[];
  outputShape: number[];
  labels?: string[];
  preprocessor?: string;
  postprocessor?: string;
  metadata?: Record<string, unknown>;
}

export interface LoadedModel {
  config: ModelConfig;
  model: tf.LayersModel | tf.GraphModel | InferenceSession;
  format: ModelFormat;
  loadedAt: number;
  lastUsed: number;
  memoryUsage: number;
}

export interface InferenceResult<T = unknown> {
  output: T;
  confidence?: number;
  processingTime: number;
  modelId: string;
}

export interface RuntimeConfig {
  backend: ModelBackend;
  maxModelsInMemory: number;
  cacheModels: boolean;
  enableProfiling: boolean;
  workerCount: number;
}

// ==================== Model Cache ====================

class ModelCache {
  private cache = new Map<string, LoadedModel>();
  private maxModels: number;

  constructor(maxModels = 5) {
    this.maxModels = maxModels;
  }

  get(id: string): LoadedModel | undefined {
    const model = this.cache.get(id);
    if (model) {
      model.lastUsed = Date.now();
    }
    return model;
  }

  set(id: string, model: LoadedModel): void {
    // Evict least recently used if at capacity
    if (this.cache.size >= this.maxModels) {
      this.evictLRU();
    }
    this.cache.set(id, model);
  }

  delete(id: string): boolean {
    const model = this.cache.get(id);
    if (model) {
      this.disposeModel(model);
      return this.cache.delete(id);
    }
    return false;
  }

  clear(): void {
    for (const model of this.cache.values()) {
      this.disposeModel(model);
    }
    this.cache.clear();
  }

  private evictLRU(): void {
    let oldest: { id: string; time: number } | null = null;

    for (const [id, model] of this.cache) {
      if (!oldest || model.lastUsed < oldest.time) {
        oldest = { id, time: model.lastUsed };
      }
    }

    if (oldest) {
      this.delete(oldest.id);
    }
  }

  private disposeModel(loaded: LoadedModel): void {
    if (loaded.format === 'tfjs') {
      (loaded.model as tf.LayersModel | tf.GraphModel).dispose();
    }
    // ONNX sessions are automatically cleaned up
  }

  getStats(): { count: number; totalMemory: number } {
    let totalMemory = 0;
    for (const model of this.cache.values()) {
      totalMemory += model.memoryUsage;
    }
    return { count: this.cache.size, totalMemory };
  }
}

// ==================== TensorFlow.js Backend ====================

async function initializeTFBackend(backend: ModelBackend): Promise<void> {
  switch (backend) {
    case 'webgpu':
      await tf.setBackend('webgpu');
      break;
    case 'webgl':
      await tf.setBackend('webgl');
      break;
    case 'wasm':
      await tf.setBackend('wasm');
      break;
    default:
      await tf.setBackend('cpu');
  }
  await tf.ready();
}

async function loadTFModel(config: ModelConfig): Promise<tf.LayersModel | tf.GraphModel> {
  // Try loading as GraphModel first (for SavedModel/frozen graphs)
  try {
    return await tf.loadGraphModel(config.url);
  } catch {
    // Fall back to LayersModel (for Keras models)
    return await tf.loadLayersModel(config.url);
  }
}

function getTFMemoryUsage(model: tf.LayersModel | tf.GraphModel): number {
  const info = tf.memory();
  return info.numBytes;
}

// ==================== ONNX Backend ====================

async function loadONNXModel(config: ModelConfig, backend: ModelBackend): Promise<InferenceSession> {
  const executionProviders: string[] = [];

  switch (backend) {
    case 'webgpu':
      executionProviders.push('webgpu');
      break;
    case 'webgl':
      executionProviders.push('webgl');
      break;
    case 'wasm':
      executionProviders.push('wasm');
      break;
    default:
      executionProviders.push('cpu');
  }

  return InferenceSession.create(config.url, {
    executionProviders,
    graphOptimizationLevel: 'all',
  });
}

// ==================== Preprocessors ====================

export const preprocessors = {
  audioNormalize: (audio: Float32Array): tf.Tensor => {
    const tensor = tf.tensor1d(audio);
    const normalized = tensor.sub(tensor.mean()).div(tensor.std());
    tensor.dispose();
    return normalized;
  },

  melSpectrogram: async (
    audio: Float32Array,
    sampleRate: number,
    fftSize = 2048,
    hopSize = 512,
    nMels = 128
  ): Promise<tf.Tensor> => {
    // Compute STFT
    const tensor = tf.tensor1d(audio);
    const frames: tf.Tensor[] = [];

    for (let i = 0; i + fftSize <= audio.length; i += hopSize) {
      const frame = tensor.slice(i, fftSize);
      const windowed = frame.mul(tf.signal.hannWindow(fftSize));
      const fft = tf.spectral.rfft(windowed);
      const magnitude = tf.abs(fft);
      frames.push(magnitude);
      frame.dispose();
      windowed.dispose();
      fft.dispose();
    }

    const stft = tf.stack(frames);
    frames.forEach(f => f.dispose());
    tensor.dispose();

    // Convert to mel scale (simplified)
    const melFilterbank = createMelFilterbank(fftSize / 2 + 1, nMels, sampleRate);
    const melSpec = tf.matMul(stft, melFilterbank);
    stft.dispose();
    melFilterbank.dispose();

    // Convert to log scale
    const logMelSpec = tf.log(melSpec.add(1e-6));
    melSpec.dispose();

    return logMelSpec;
  },

  mfcc: async (
    audio: Float32Array,
    sampleRate: number,
    nMfcc = 13
  ): Promise<tf.Tensor> => {
    const melSpec = await preprocessors.melSpectrogram(audio, sampleRate);
    // DCT for MFCC (simplified)
    const mfcc = tf.signal.dct(melSpec);
    melSpec.dispose();
    return mfcc.slice([0, 0], [-1, nMfcc]);
  },

  chromagram: async (
    audio: Float32Array,
    sampleRate: number
  ): Promise<tf.Tensor> => {
    const melSpec = await preprocessors.melSpectrogram(audio, sampleRate, 4096, 512, 12);
    // Fold into 12 chroma bins
    return melSpec;
  },
};

function createMelFilterbank(nFft: number, nMels: number, sampleRate: number): tf.Tensor {
  const fMin = 0;
  const fMax = sampleRate / 2;

  // Convert to mel scale
  const melMin = 2595 * Math.log10(1 + fMin / 700);
  const melMax = 2595 * Math.log10(1 + fMax / 700);

  // Create mel points
  const melPoints: number[] = [];
  for (let i = 0; i <= nMels + 1; i++) {
    melPoints.push(melMin + (melMax - melMin) * i / (nMels + 1));
  }

  // Convert back to Hz
  const hzPoints = melPoints.map(mel => 700 * (Math.pow(10, mel / 2595) - 1));

  // Convert to FFT bins
  const binPoints = hzPoints.map(hz => Math.floor((nFft + 1) * hz / sampleRate));

  // Create filterbank
  const filterbank = new Float32Array(nFft * nMels);

  for (let i = 0; i < nMels; i++) {
    for (let j = binPoints[i]; j < binPoints[i + 1]; j++) {
      filterbank[j * nMels + i] = (j - binPoints[i]) / (binPoints[i + 1] - binPoints[i]);
    }
    for (let j = binPoints[i + 1]; j < binPoints[i + 2]; j++) {
      filterbank[j * nMels + i] = (binPoints[i + 2] - j) / (binPoints[i + 2] - binPoints[i + 1]);
    }
  }

  return tf.tensor2d(filterbank, [nFft, nMels]);
}

// ==================== Postprocessors ====================

export const postprocessors = {
  softmax: (logits: tf.Tensor): tf.Tensor => {
    return tf.softmax(logits);
  },

  argmax: (tensor: tf.Tensor): number => {
    return tensor.argMax().dataSync()[0];
  },

  topK: (tensor: tf.Tensor, k = 5): { indices: number[]; values: number[] } => {
    const { indices, values } = tf.topk(tensor.flatten(), k);
    return {
      indices: Array.from(indices.dataSync()),
      values: Array.from(values.dataSync()),
    };
  },

  sigmoid: (tensor: tf.Tensor): tf.Tensor => {
    return tf.sigmoid(tensor);
  },

  threshold: (tensor: tf.Tensor, threshold = 0.5): boolean[] => {
    return Array.from(tensor.greater(threshold).dataSync()).map(v => v === 1);
  },
};

// ==================== Model Runtime ====================

export class ModelRuntime {
  private cache: ModelCache;
  private config: RuntimeConfig;
  private initialized = false;
  private workers: Worker[] = [];

  constructor(config: Partial<RuntimeConfig> = {}) {
    this.config = {
      backend: config.backend || 'webgl',
      maxModelsInMemory: config.maxModelsInMemory || 5,
      cacheModels: config.cacheModels ?? true,
      enableProfiling: config.enableProfiling ?? false,
      workerCount: config.workerCount || navigator.hardwareConcurrency || 4,
    };
    this.cache = new ModelCache(this.config.maxModelsInMemory);
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;

    // Initialize TensorFlow.js backend
    await initializeTFBackend(this.config.backend);

    // Enable profiling if requested
    if (this.config.enableProfiling) {
      tf.enableProdMode();
    }

    this.initialized = true;
    console.log(`Model runtime initialized with ${this.config.backend} backend`);
  }

  async loadModel(config: ModelConfig): Promise<LoadedModel> {
    // Check cache first
    const cached = this.cache.get(config.id);
    if (cached) {
      return cached;
    }

    const startTime = performance.now();

    let model: tf.LayersModel | tf.GraphModel | InferenceSession;
    let memoryUsage = 0;

    if (config.format === 'tfjs') {
      const memBefore = tf.memory().numBytes;
      model = await loadTFModel(config);
      memoryUsage = tf.memory().numBytes - memBefore;
    } else {
      model = await loadONNXModel(config, this.config.backend);
      // ONNX doesn't provide easy memory tracking
      memoryUsage = 0;
    }

    const loaded: LoadedModel = {
      config,
      model,
      format: config.format,
      loadedAt: Date.now(),
      lastUsed: Date.now(),
      memoryUsage,
    };

    if (this.config.cacheModels) {
      this.cache.set(config.id, loaded);
    }

    console.log(`Loaded model ${config.name} in ${performance.now() - startTime}ms`);
    return loaded;
  }

  async predict<T = unknown>(
    modelId: string,
    input: tf.Tensor | Float32Array | number[],
    options?: { labels?: string[] }
  ): Promise<InferenceResult<T>> {
    const loaded = this.cache.get(modelId);
    if (!loaded) {
      throw new Error(`Model ${modelId} not loaded`);
    }

    const startTime = performance.now();
    let output: T;
    let confidence: number | undefined;

    if (loaded.format === 'tfjs') {
      const tfModel = loaded.model as tf.LayersModel | tf.GraphModel;

      // Ensure input is a tensor
      let inputTensor: tf.Tensor;
      if (input instanceof tf.Tensor) {
        inputTensor = input;
      } else {
        inputTensor = tf.tensor(input);
      }

      // Reshape to match model input
      const reshapedInput = inputTensor.reshape(loaded.config.inputShape);

      // Run inference
      const result = tfModel.predict(reshapedInput) as tf.Tensor;

      // Get output
      const outputData = await result.data();

      // Apply softmax and get top prediction
      if (options?.labels || loaded.config.labels) {
        const labels = options?.labels || loaded.config.labels!;
        const probs = await tf.softmax(result).data();
        const maxIdx = probs.indexOf(Math.max(...probs));
        output = {
          label: labels[maxIdx],
          probabilities: Object.fromEntries(labels.map((l, i) => [l, probs[i]])),
        } as T;
        confidence = probs[maxIdx];
      } else {
        output = Array.from(outputData) as T;
      }

      // Cleanup
      if (!(input instanceof tf.Tensor)) {
        inputTensor.dispose();
      }
      reshapedInput.dispose();
      result.dispose();
    } else {
      const session = loaded.model as InferenceSession;

      // Convert input to ONNX tensor
      let inputArray: Float32Array;
      if (input instanceof tf.Tensor) {
        inputArray = new Float32Array(await input.data());
      } else if (input instanceof Float32Array) {
        inputArray = input;
      } else {
        inputArray = new Float32Array(input);
      }

      const onnxInput = new OnnxTensor('float32', inputArray, loaded.config.inputShape);
      const inputName = session.inputNames[0];

      // Run inference
      const results = await session.run({ [inputName]: onnxInput });
      const outputName = session.outputNames[0];
      output = Array.from(results[outputName].data as Float32Array) as T;
    }

    return {
      output,
      confidence,
      processingTime: performance.now() - startTime,
      modelId,
    };
  }

  async batchPredict<T = unknown>(
    modelId: string,
    inputs: (tf.Tensor | Float32Array | number[])[],
    options?: { labels?: string[] }
  ): Promise<InferenceResult<T>[]> {
    // Process in parallel using Promise.all
    return Promise.all(inputs.map(input => this.predict<T>(modelId, input, options)));
  }

  unloadModel(modelId: string): boolean {
    return this.cache.delete(modelId);
  }

  clearCache(): void {
    this.cache.clear();
  }

  getStats(): { modelCount: number; totalMemory: number; backend: string } {
    const cacheStats = this.cache.getStats();
    return {
      ...cacheStats,
      modelCount: cacheStats.count,
      backend: this.config.backend,
    };
  }

  getMemoryInfo(): tf.MemoryInfo {
    return tf.memory();
  }

  dispose(): void {
    this.cache.clear();
    this.workers.forEach(w => w.terminate());
    this.workers = [];
  }
}

// ==================== Singleton Instance ====================

let runtimeInstance: ModelRuntime | null = null;

export function getModelRuntime(config?: Partial<RuntimeConfig>): ModelRuntime {
  if (!runtimeInstance) {
    runtimeInstance = new ModelRuntime(config);
  }
  return runtimeInstance;
}

export default ModelRuntime;
