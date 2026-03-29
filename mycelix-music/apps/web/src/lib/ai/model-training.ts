// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Model Training System
 *
 * Custom model training for users:
 * - Genre classifier fine-tuning
 * - Mood model personalization
 * - Custom instrument detection
 * - Transfer learning
 * - Dataset management
 */

import * as tf from '@tensorflow/tfjs';
import { getModelRuntime, preprocessors, ModelConfig } from './model-runtime';

// ==================== Types ====================

export interface TrainingExample {
  id: string;
  audio: Float32Array;
  label: string;
  metadata?: Record<string, unknown>;
}

export interface Dataset {
  id: string;
  name: string;
  examples: TrainingExample[];
  labels: string[];
  createdAt: number;
  updatedAt: number;
}

export interface TrainingConfig {
  epochs: number;
  batchSize: number;
  learningRate: number;
  validationSplit: number;
  earlyStoppingPatience?: number;
  reduceOnPlateauFactor?: number;
  augmentation?: {
    timeStretch?: boolean;
    pitchShift?: boolean;
    noiseInjection?: boolean;
    gainVariation?: boolean;
  };
}

export interface TrainingProgress {
  epoch: number;
  totalEpochs: number;
  batch: number;
  totalBatches: number;
  loss: number;
  accuracy: number;
  valLoss?: number;
  valAccuracy?: number;
}

export interface TrainingResult {
  modelId: string;
  finalLoss: number;
  finalAccuracy: number;
  valLoss: number;
  valAccuracy: number;
  trainingTime: number;
  epochHistory: {
    loss: number[];
    accuracy: number[];
    valLoss: number[];
    valAccuracy: number[];
  };
}

export interface ModelExport {
  modelJson: unknown;
  weights: ArrayBuffer;
  metadata: {
    id: string;
    name: string;
    labels: string[];
    inputShape: number[];
    createdAt: number;
    accuracy: number;
  };
}

// ==================== Data Augmentation ====================

export const augmentation = {
  timeStretch(audio: Float32Array, rate: number): Float32Array {
    // Simple time stretching via resampling
    const newLength = Math.floor(audio.length / rate);
    const result = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const srcIdx = i * rate;
      const srcIdxFloor = Math.floor(srcIdx);
      const srcIdxCeil = Math.min(srcIdxFloor + 1, audio.length - 1);
      const frac = srcIdx - srcIdxFloor;
      result[i] = audio[srcIdxFloor] * (1 - frac) + audio[srcIdxCeil] * frac;
    }

    return result;
  },

  pitchShift(audio: Float32Array, semitones: number): Float32Array {
    // Simple pitch shift via resampling (changes duration)
    const rate = Math.pow(2, semitones / 12);
    return augmentation.timeStretch(audio, rate);
  },

  addNoise(audio: Float32Array, noiseLevel: number): Float32Array {
    const result = new Float32Array(audio.length);
    for (let i = 0; i < audio.length; i++) {
      result[i] = audio[i] + (Math.random() * 2 - 1) * noiseLevel;
    }
    return result;
  },

  adjustGain(audio: Float32Array, gain: number): Float32Array {
    const result = new Float32Array(audio.length);
    for (let i = 0; i < audio.length; i++) {
      result[i] = Math.max(-1, Math.min(1, audio[i] * gain));
    }
    return result;
  },

  augmentExample(
    audio: Float32Array,
    config: NonNullable<TrainingConfig['augmentation']>
  ): Float32Array[] {
    const augmented: Float32Array[] = [audio];

    if (config.timeStretch) {
      augmented.push(augmentation.timeStretch(audio, 0.9));
      augmented.push(augmentation.timeStretch(audio, 1.1));
    }

    if (config.pitchShift) {
      augmented.push(augmentation.pitchShift(audio, -2));
      augmented.push(augmentation.pitchShift(audio, 2));
    }

    if (config.noiseInjection) {
      augmented.push(augmentation.addNoise(audio, 0.01));
      augmented.push(augmentation.addNoise(audio, 0.02));
    }

    if (config.gainVariation) {
      augmented.push(augmentation.adjustGain(audio, 0.8));
      augmented.push(augmentation.adjustGain(audio, 1.2));
    }

    return augmented;
  },
};

// ==================== Dataset Management ====================

export class DatasetManager {
  private datasets = new Map<string, Dataset>();
  private dbName = 'mycelix-training-datasets';

  async loadFromIndexedDB(): Promise<void> {
    const db = await this.openDB();
    const tx = db.transaction('datasets', 'readonly');
    const store = tx.objectStore('datasets');

    return new Promise((resolve, reject) => {
      const request = store.getAll();
      request.onsuccess = () => {
        for (const dataset of request.result) {
          this.datasets.set(dataset.id, dataset);
        }
        resolve();
      };
      request.onerror = () => reject(request.error);
    });
  }

  private async openDB(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, 1);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains('datasets')) {
          db.createObjectStore('datasets', { keyPath: 'id' });
        }
      };
    });
  }

  async createDataset(name: string, labels: string[]): Promise<Dataset> {
    const dataset: Dataset = {
      id: `dataset-${Date.now()}`,
      name,
      examples: [],
      labels,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    this.datasets.set(dataset.id, dataset);
    await this.saveDataset(dataset);

    return dataset;
  }

  async addExample(datasetId: string, example: Omit<TrainingExample, 'id'>): Promise<TrainingExample> {
    const dataset = this.datasets.get(datasetId);
    if (!dataset) {
      throw new Error(`Dataset ${datasetId} not found`);
    }

    const fullExample: TrainingExample = {
      ...example,
      id: `example-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    };

    dataset.examples.push(fullExample);
    dataset.updatedAt = Date.now();

    await this.saveDataset(dataset);
    return fullExample;
  }

  async removeExample(datasetId: string, exampleId: string): Promise<void> {
    const dataset = this.datasets.get(datasetId);
    if (!dataset) return;

    dataset.examples = dataset.examples.filter(e => e.id !== exampleId);
    dataset.updatedAt = Date.now();

    await this.saveDataset(dataset);
  }

  private async saveDataset(dataset: Dataset): Promise<void> {
    const db = await this.openDB();
    const tx = db.transaction('datasets', 'readwrite');
    const store = tx.objectStore('datasets');

    return new Promise((resolve, reject) => {
      const request = store.put(dataset);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async deleteDataset(datasetId: string): Promise<void> {
    this.datasets.delete(datasetId);

    const db = await this.openDB();
    const tx = db.transaction('datasets', 'readwrite');
    const store = tx.objectStore('datasets');

    return new Promise((resolve, reject) => {
      const request = store.delete(datasetId);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  getDataset(id: string): Dataset | undefined {
    return this.datasets.get(id);
  }

  getAllDatasets(): Dataset[] {
    return Array.from(this.datasets.values());
  }
}

// ==================== Model Trainer ====================

export class ModelTrainer {
  private datasetManager: DatasetManager;
  private sampleRate = 22050;

  constructor() {
    this.datasetManager = new DatasetManager();
  }

  async initialize(): Promise<void> {
    await this.datasetManager.loadFromIndexedDB();
    await getModelRuntime().initialize();
  }

  async prepareTrainingData(
    dataset: Dataset,
    config: TrainingConfig
  ): Promise<{ xs: tf.Tensor; ys: tf.Tensor }> {
    const features: tf.Tensor[] = [];
    const labels: number[] = [];

    for (const example of dataset.examples) {
      // Get audio variations (with augmentation)
      let audioVariants: Float32Array[];

      if (config.augmentation) {
        audioVariants = augmentation.augmentExample(example.audio, config.augmentation);
      } else {
        audioVariants = [example.audio];
      }

      // Process each variant
      for (const audio of audioVariants) {
        // Convert to mel spectrogram
        const melSpec = await preprocessors.melSpectrogram(audio, this.sampleRate);

        // Resize to fixed size
        const resized = tf.image.resizeBilinear(
          melSpec.expandDims(-1) as tf.Tensor3D,
          [128, 128]
        );

        features.push(resized);
        labels.push(dataset.labels.indexOf(example.label));

        melSpec.dispose();
      }
    }

    // Stack features
    const xs = tf.stack(features);
    features.forEach(f => f.dispose());

    // One-hot encode labels
    const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), dataset.labels.length);

    return { xs, ys };
  }

  async createModel(numClasses: number): Promise<tf.LayersModel> {
    const model = tf.sequential();

    // Convolutional layers
    model.add(tf.layers.conv2d({
      inputShape: [128, 128, 1],
      filters: 32,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same',
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    model.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same',
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    model.add(tf.layers.conv2d({
      filters: 128,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same',
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    model.add(tf.layers.conv2d({
      filters: 256,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same',
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.globalAveragePooling2d({}));

    // Dense layers
    model.add(tf.layers.dense({
      units: 256,
      activation: 'relu',
    }));
    model.add(tf.layers.dropout({ rate: 0.5 }));

    model.add(tf.layers.dense({
      units: numClasses,
      activation: 'softmax',
    }));

    return model;
  }

  async train(
    datasetId: string,
    config: TrainingConfig,
    onProgress?: (progress: TrainingProgress) => void
  ): Promise<TrainingResult> {
    const dataset = this.datasetManager.getDataset(datasetId);
    if (!dataset) {
      throw new Error(`Dataset ${datasetId} not found`);
    }

    if (dataset.examples.length < 10) {
      throw new Error('Need at least 10 examples to train');
    }

    const startTime = performance.now();

    // Prepare data
    const { xs, ys } = await this.prepareTrainingData(dataset, config);

    // Create model
    const model = await this.createModel(dataset.labels.length);

    // Compile
    model.compile({
      optimizer: tf.train.adam(config.learningRate),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    // Training callbacks
    const callbacks: tf.CustomCallbackArgs[] = [];
    const epochHistory: TrainingResult['epochHistory'] = {
      loss: [],
      accuracy: [],
      valLoss: [],
      valAccuracy: [],
    };

    if (onProgress) {
      callbacks.push({
        onEpochEnd: (epoch, logs) => {
          epochHistory.loss.push(logs?.loss as number || 0);
          epochHistory.accuracy.push(logs?.acc as number || logs?.accuracy as number || 0);
          epochHistory.valLoss.push(logs?.val_loss as number || 0);
          epochHistory.valAccuracy.push(logs?.val_acc as number || logs?.val_accuracy as number || 0);

          onProgress({
            epoch: epoch + 1,
            totalEpochs: config.epochs,
            batch: 0,
            totalBatches: 0,
            loss: logs?.loss as number || 0,
            accuracy: logs?.acc as number || logs?.accuracy as number || 0,
            valLoss: logs?.val_loss as number,
            valAccuracy: logs?.val_acc as number || logs?.val_accuracy as number,
          });
        },
      });
    }

    // Early stopping
    if (config.earlyStoppingPatience) {
      callbacks.push(tf.callbacks.earlyStopping({
        monitor: 'val_loss',
        patience: config.earlyStoppingPatience,
      }));
    }

    // Train
    const history = await model.fit(xs, ys, {
      epochs: config.epochs,
      batchSize: config.batchSize,
      validationSplit: config.validationSplit,
      shuffle: true,
      callbacks,
    });

    // Cleanup
    xs.dispose();
    ys.dispose();

    const trainingTime = performance.now() - startTime;

    // Save model
    const modelId = `custom-model-${datasetId}-${Date.now()}`;
    await model.save(`indexeddb://${modelId}`);

    // Get final metrics
    const finalLoss = history.history.loss[history.history.loss.length - 1] as number;
    const finalAccuracy = (history.history.acc?.[history.history.acc.length - 1] ||
      history.history.accuracy?.[history.history.accuracy.length - 1] || 0) as number;
    const valLoss = (history.history.val_loss?.[history.history.val_loss.length - 1] || 0) as number;
    const valAccuracy = (history.history.val_acc?.[history.history.val_acc.length - 1] ||
      history.history.val_accuracy?.[history.history.val_accuracy.length - 1] || 0) as number;

    model.dispose();

    return {
      modelId,
      finalLoss,
      finalAccuracy,
      valLoss,
      valAccuracy,
      trainingTime,
      epochHistory,
    };
  }

  async fineTune(
    baseModelConfig: ModelConfig,
    datasetId: string,
    config: TrainingConfig,
    onProgress?: (progress: TrainingProgress) => void
  ): Promise<TrainingResult> {
    const dataset = this.datasetManager.getDataset(datasetId);
    if (!dataset) {
      throw new Error(`Dataset ${datasetId} not found`);
    }

    const startTime = performance.now();

    // Load base model
    const runtime = getModelRuntime();
    const loaded = await runtime.loadModel(baseModelConfig);

    if (loaded.format !== 'tfjs') {
      throw new Error('Fine-tuning only supported for TensorFlow.js models');
    }

    const baseModel = loaded.model as tf.LayersModel;

    // Freeze base layers (except last few)
    for (let i = 0; i < baseModel.layers.length - 3; i++) {
      baseModel.layers[i].trainable = false;
    }

    // Replace output layer if needed
    let model: tf.LayersModel;
    if (baseModel.outputs[0].shape[1] !== dataset.labels.length) {
      // Need to create new model with different output size
      const lastLayer = baseModel.layers[baseModel.layers.length - 2]; // Before softmax

      const newModel = tf.sequential();
      for (let i = 0; i < baseModel.layers.length - 1; i++) {
        newModel.add(baseModel.layers[i]);
      }
      newModel.add(tf.layers.dense({
        units: dataset.labels.length,
        activation: 'softmax',
        name: 'custom_output',
      }));

      model = newModel;
    } else {
      model = baseModel;
    }

    // Prepare data
    const { xs, ys } = await this.prepareTrainingData(dataset, config);

    // Compile with lower learning rate for fine-tuning
    model.compile({
      optimizer: tf.train.adam(config.learningRate * 0.1),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    // Training callbacks
    const epochHistory: TrainingResult['epochHistory'] = {
      loss: [],
      accuracy: [],
      valLoss: [],
      valAccuracy: [],
    };

    const callbacks: tf.CustomCallbackArgs[] = [];
    if (onProgress) {
      callbacks.push({
        onEpochEnd: (epoch, logs) => {
          epochHistory.loss.push(logs?.loss as number || 0);
          epochHistory.accuracy.push(logs?.acc as number || logs?.accuracy as number || 0);
          epochHistory.valLoss.push(logs?.val_loss as number || 0);
          epochHistory.valAccuracy.push(logs?.val_acc as number || logs?.val_accuracy as number || 0);

          onProgress({
            epoch: epoch + 1,
            totalEpochs: config.epochs,
            batch: 0,
            totalBatches: 0,
            loss: logs?.loss as number || 0,
            accuracy: logs?.acc as number || logs?.accuracy as number || 0,
            valLoss: logs?.val_loss as number,
            valAccuracy: logs?.val_acc as number || logs?.val_accuracy as number,
          });
        },
      });
    }

    // Train
    const history = await model.fit(xs, ys, {
      epochs: config.epochs,
      batchSize: config.batchSize,
      validationSplit: config.validationSplit,
      shuffle: true,
      callbacks,
    });

    // Cleanup
    xs.dispose();
    ys.dispose();

    const trainingTime = performance.now() - startTime;

    // Save model
    const modelId = `finetuned-${baseModelConfig.id}-${Date.now()}`;
    await model.save(`indexeddb://${modelId}`);

    const finalLoss = history.history.loss[history.history.loss.length - 1] as number;
    const finalAccuracy = (history.history.acc?.[history.history.acc.length - 1] ||
      history.history.accuracy?.[history.history.accuracy.length - 1] || 0) as number;
    const valLoss = (history.history.val_loss?.[history.history.val_loss.length - 1] || 0) as number;
    const valAccuracy = (history.history.val_acc?.[history.history.val_acc.length - 1] ||
      history.history.val_accuracy?.[history.history.val_accuracy.length - 1] || 0) as number;

    model.dispose();

    return {
      modelId,
      finalLoss,
      finalAccuracy,
      valLoss,
      valAccuracy,
      trainingTime,
      epochHistory,
    };
  }

  async exportModel(modelId: string, labels: string[]): Promise<ModelExport> {
    const model = await tf.loadLayersModel(`indexeddb://${modelId}`);

    // Get model topology
    const modelJson = model.toJSON();

    // Get weights
    const weightData = await model.save(tf.io.withSaveHandler(async (artifacts) => {
      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: 'JSON',
        },
      };
    }));

    // Serialize weights
    const weightBuffers: ArrayBuffer[] = [];
    for (const weight of model.weights) {
      const data = await weight.data();
      weightBuffers.push(data.buffer as ArrayBuffer);
    }

    // Combine weight buffers
    const totalSize = weightBuffers.reduce((sum, buf) => sum + buf.byteLength, 0);
    const combined = new ArrayBuffer(totalSize);
    const view = new Uint8Array(combined);
    let offset = 0;
    for (const buf of weightBuffers) {
      view.set(new Uint8Array(buf), offset);
      offset += buf.byteLength;
    }

    model.dispose();

    return {
      modelJson,
      weights: combined,
      metadata: {
        id: modelId,
        name: `Custom Model - ${modelId}`,
        labels,
        inputShape: [1, 128, 128, 1],
        createdAt: Date.now(),
        accuracy: 0, // Would be filled from training result
      },
    };
  }

  async importModel(exported: ModelExport): Promise<string> {
    // Create model from JSON
    const model = await tf.models.modelFromJSON(exported.modelJson as tf.io.ModelJSON);

    // Load weights
    const weightSpecs = model.weights.map(w => ({
      name: w.name,
      shape: w.shape,
      dtype: w.dtype,
    }));

    // This is simplified - real implementation would properly decode weights
    const modelId = `imported-${Date.now()}`;
    await model.save(`indexeddb://${modelId}`);

    model.dispose();
    return modelId;
  }

  getDatasetManager(): DatasetManager {
    return this.datasetManager;
  }
}

// ==================== Singleton Instance ====================

let trainerInstance: ModelTrainer | null = null;

export function getModelTrainer(): ModelTrainer {
  if (!trainerInstance) {
    trainerInstance = new ModelTrainer();
  }
  return trainerInstance;
}

export default ModelTrainer;
