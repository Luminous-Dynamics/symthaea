// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Inference Worker
 *
 * Web Worker for background AI inference:
 * - Offloads computation from main thread
 * - Processes audio analysis in parallel
 * - Handles batch operations
 * - Supports model training
 */

import * as tf from '@tensorflow/tfjs';

// ==================== Types ====================

export type WorkerCommand =
  | 'initialize'
  | 'loadModel'
  | 'predict'
  | 'batchPredict'
  | 'analyze'
  | 'train'
  | 'dispose'
  | 'getStats';

export interface WorkerMessage {
  id: string;
  command: WorkerCommand;
  payload: unknown;
}

export interface WorkerResponse {
  id: string;
  success: boolean;
  result?: unknown;
  error?: string;
}

export interface ModelInfo {
  id: string;
  url: string;
  format: 'tfjs' | 'onnx';
  inputShape: number[];
}

export interface TrainingConfig {
  epochs: number;
  batchSize: number;
  learningRate: number;
  validationSplit: number;
  callbacks?: {
    onEpochEnd?: boolean;
    onBatchEnd?: boolean;
  };
}

// ==================== Worker State ====================

let initialized = false;
let loadedModels = new Map<string, tf.LayersModel | tf.GraphModel>();

// ==================== Initialization ====================

async function initialize(backend: string = 'webgl'): Promise<void> {
  if (initialized) return;

  await tf.setBackend(backend);
  await tf.ready();

  // Enable memory management
  tf.engine().startScope();

  initialized = true;
}

// ==================== Model Loading ====================

async function loadModel(info: ModelInfo): Promise<void> {
  if (loadedModels.has(info.id)) {
    return;
  }

  let model: tf.LayersModel | tf.GraphModel;

  if (info.format === 'tfjs') {
    try {
      model = await tf.loadGraphModel(info.url);
    } catch {
      model = await tf.loadLayersModel(info.url);
    }
  } else {
    throw new Error('ONNX not supported in worker yet');
  }

  loadedModels.set(info.id, model);
}

// ==================== Inference ====================

async function predict(
  modelId: string,
  input: number[] | Float32Array,
  inputShape: number[]
): Promise<number[]> {
  const model = loadedModels.get(modelId);
  if (!model) {
    throw new Error(`Model ${modelId} not loaded`);
  }

  const inputTensor = tf.tensor(input).reshape(inputShape);
  const output = model.predict(inputTensor) as tf.Tensor;
  const result = Array.from(await output.data());

  inputTensor.dispose();
  output.dispose();

  return result;
}

async function batchPredict(
  modelId: string,
  inputs: (number[] | Float32Array)[],
  inputShape: number[]
): Promise<number[][]> {
  const results: number[][] = [];

  for (const input of inputs) {
    results.push(await predict(modelId, input, inputShape));
  }

  return results;
}

// ==================== Audio Analysis ====================

async function analyzeAudio(audio: Float32Array, sampleRate: number): Promise<{
  rms: number;
  peak: number;
  zeroCrossings: number;
  spectralCentroid: number;
  spectralFlatness: number;
}> {
  const length = audio.length;

  // RMS (Root Mean Square)
  let sumSquares = 0;
  for (let i = 0; i < length; i++) {
    sumSquares += audio[i] ** 2;
  }
  const rms = Math.sqrt(sumSquares / length);

  // Peak
  let peak = 0;
  for (let i = 0; i < length; i++) {
    peak = Math.max(peak, Math.abs(audio[i]));
  }

  // Zero Crossings
  let zeroCrossings = 0;
  for (let i = 1; i < length; i++) {
    if ((audio[i] >= 0) !== (audio[i - 1] >= 0)) {
      zeroCrossings++;
    }
  }

  // Spectral analysis
  const fftSize = 2048;
  const audioTensor = tf.tensor1d(audio.slice(0, fftSize));
  const windowed = audioTensor.mul(tf.signal.hannWindow(fftSize));
  const fft = tf.spectral.rfft(windowed);
  const magnitude = tf.abs(fft);
  const magData = await magnitude.data();

  // Spectral Centroid
  let weightedSum = 0;
  let sum = 0;
  for (let i = 0; i < magData.length; i++) {
    const freq = (i * sampleRate) / fftSize;
    weightedSum += freq * magData[i];
    sum += magData[i];
  }
  const spectralCentroid = sum > 0 ? weightedSum / sum : 0;

  // Spectral Flatness
  let logSum = 0;
  let linearSum = 0;
  for (let i = 0; i < magData.length; i++) {
    if (magData[i] > 0) {
      logSum += Math.log(magData[i]);
      linearSum += magData[i];
    }
  }
  const geometricMean = Math.exp(logSum / magData.length);
  const arithmeticMean = linearSum / magData.length;
  const spectralFlatness = arithmeticMean > 0 ? geometricMean / arithmeticMean : 0;

  // Cleanup
  audioTensor.dispose();
  windowed.dispose();
  fft.dispose();
  magnitude.dispose();

  return {
    rms,
    peak,
    zeroCrossings,
    spectralCentroid,
    spectralFlatness,
  };
}

// ==================== Training ====================

async function trainModel(
  modelId: string,
  trainX: number[][],
  trainY: number[][],
  config: TrainingConfig,
  postMessage: (msg: unknown) => void
): Promise<{ loss: number; accuracy: number }> {
  const model = loadedModels.get(modelId);
  if (!model || !('fit' in model)) {
    throw new Error(`Model ${modelId} not found or not trainable`);
  }

  const layersModel = model as tf.LayersModel;

  // Compile model if not already
  if (!layersModel.optimizer) {
    layersModel.compile({
      optimizer: tf.train.adam(config.learningRate),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
  }

  const xs = tf.tensor2d(trainX);
  const ys = tf.tensor2d(trainY);

  const callbacks: tf.CustomCallbackArgs = {};

  if (config.callbacks?.onEpochEnd) {
    callbacks.onEpochEnd = (epoch, logs) => {
      postMessage({
        type: 'training_progress',
        epoch,
        logs,
      });
    };
  }

  if (config.callbacks?.onBatchEnd) {
    callbacks.onBatchEnd = (batch, logs) => {
      postMessage({
        type: 'batch_progress',
        batch,
        logs,
      });
    };
  }

  const history = await layersModel.fit(xs, ys, {
    epochs: config.epochs,
    batchSize: config.batchSize,
    validationSplit: config.validationSplit,
    callbacks,
  });

  xs.dispose();
  ys.dispose();

  const finalLoss = history.history.loss[history.history.loss.length - 1] as number;
  const finalAcc = (history.history.acc?.[history.history.acc.length - 1] ||
    history.history.accuracy?.[history.history.accuracy.length - 1] || 0) as number;

  return {
    loss: finalLoss,
    accuracy: finalAcc,
  };
}

// ==================== Stats ====================

function getStats(): {
  numModels: number;
  memoryInfo: tf.MemoryInfo;
  backend: string;
} {
  return {
    numModels: loadedModels.size,
    memoryInfo: tf.memory(),
    backend: tf.getBackend(),
  };
}

// ==================== Cleanup ====================

function dispose(modelId?: string): void {
  if (modelId) {
    const model = loadedModels.get(modelId);
    if (model) {
      model.dispose();
      loadedModels.delete(modelId);
    }
  } else {
    for (const model of loadedModels.values()) {
      model.dispose();
    }
    loadedModels.clear();
  }
}

// ==================== Message Handler ====================

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const { id, command, payload } = event.data;

  const respond = (success: boolean, result?: unknown, error?: string) => {
    const response: WorkerResponse = { id, success, result, error };
    self.postMessage(response);
  };

  try {
    switch (command) {
      case 'initialize': {
        await initialize(payload as string);
        respond(true);
        break;
      }

      case 'loadModel': {
        await loadModel(payload as ModelInfo);
        respond(true);
        break;
      }

      case 'predict': {
        const { modelId, input, inputShape } = payload as {
          modelId: string;
          input: number[];
          inputShape: number[];
        };
        const result = await predict(modelId, input, inputShape);
        respond(true, result);
        break;
      }

      case 'batchPredict': {
        const { modelId, inputs, inputShape } = payload as {
          modelId: string;
          inputs: number[][];
          inputShape: number[];
        };
        const results = await batchPredict(modelId, inputs, inputShape);
        respond(true, results);
        break;
      }

      case 'analyze': {
        const { audio, sampleRate } = payload as {
          audio: Float32Array;
          sampleRate: number;
        };
        const analysis = await analyzeAudio(audio, sampleRate);
        respond(true, analysis);
        break;
      }

      case 'train': {
        const { modelId, trainX, trainY, config } = payload as {
          modelId: string;
          trainX: number[][];
          trainY: number[][];
          config: TrainingConfig;
        };
        const result = await trainModel(modelId, trainX, trainY, config, self.postMessage);
        respond(true, result);
        break;
      }

      case 'dispose': {
        dispose(payload as string | undefined);
        respond(true);
        break;
      }

      case 'getStats': {
        const stats = getStats();
        respond(true, stats);
        break;
      }

      default:
        respond(false, undefined, `Unknown command: ${command}`);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    respond(false, undefined, message);
  }
};

// Signal that worker is ready
self.postMessage({ type: 'ready' });
