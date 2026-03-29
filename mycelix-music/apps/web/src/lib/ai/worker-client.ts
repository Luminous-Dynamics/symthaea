// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Inference Worker Client
 *
 * Client for communicating with inference workers:
 * - Worker pool management
 * - Request queuing
 * - Load balancing
 * - Error handling
 */

import type { WorkerCommand, WorkerMessage, WorkerResponse } from './inference-worker';

// ==================== Types ====================

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timeout: NodeJS.Timeout;
}

interface WorkerInstance {
  worker: Worker;
  busy: boolean;
  id: number;
}

export interface WorkerPoolConfig {
  maxWorkers?: number;
  requestTimeout?: number;
  workerUrl?: string;
}

// ==================== Worker Client ====================

export class InferenceWorkerClient {
  private workers: WorkerInstance[] = [];
  private pendingRequests = new Map<string, PendingRequest>();
  private requestQueue: Array<{
    message: WorkerMessage;
    resolve: (value: unknown) => void;
    reject: (error: Error) => void;
  }> = [];
  private config: Required<WorkerPoolConfig>;
  private requestCounter = 0;
  private initialized = false;

  constructor(config: WorkerPoolConfig = {}) {
    this.config = {
      maxWorkers: config.maxWorkers || Math.max(2, navigator.hardwareConcurrency - 1),
      requestTimeout: config.requestTimeout || 30000,
      workerUrl: config.workerUrl || new URL('./inference-worker.ts', import.meta.url).href,
    };
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;

    // Create worker pool
    const workerPromises: Promise<void>[] = [];

    for (let i = 0; i < this.config.maxWorkers; i++) {
      const worker = new Worker(this.config.workerUrl, { type: 'module' });
      const instance: WorkerInstance = { worker, busy: false, id: i };

      worker.onmessage = (event) => this.handleWorkerMessage(instance, event);
      worker.onerror = (error) => this.handleWorkerError(instance, error);

      // Wait for worker to be ready
      workerPromises.push(
        new Promise((resolve) => {
          const handler = (event: MessageEvent) => {
            if (event.data.type === 'ready') {
              worker.removeEventListener('message', handler);
              resolve();
            }
          };
          worker.addEventListener('message', handler);
        })
      );

      this.workers.push(instance);
    }

    await Promise.all(workerPromises);

    // Initialize all workers with TensorFlow backend
    await Promise.all(
      this.workers.map((instance) =>
        this.sendToWorker(instance, 'initialize', 'webgl')
      )
    );

    this.initialized = true;
    console.log(`Inference worker pool initialized with ${this.workers.length} workers`);
  }

  private handleWorkerMessage(instance: WorkerInstance, event: MessageEvent): void {
    const data = event.data;

    // Handle training/progress updates
    if (data.type === 'training_progress' || data.type === 'batch_progress') {
      this.onTrainingProgress?.(data);
      return;
    }

    // Handle response
    const response = data as WorkerResponse;
    const pending = this.pendingRequests.get(response.id);

    if (pending) {
      clearTimeout(pending.timeout);
      this.pendingRequests.delete(response.id);

      if (response.success) {
        pending.resolve(response.result);
      } else {
        pending.reject(new Error(response.error || 'Worker error'));
      }
    }

    // Mark worker as available and process queue
    instance.busy = false;
    this.processQueue();
  }

  private handleWorkerError(instance: WorkerInstance, error: ErrorEvent): void {
    console.error(`Worker ${instance.id} error:`, error);
    instance.busy = false;

    // Reject any pending requests for this worker
    for (const [id, pending] of this.pendingRequests) {
      pending.reject(new Error(`Worker error: ${error.message}`));
      this.pendingRequests.delete(id);
    }
  }

  private getAvailableWorker(): WorkerInstance | null {
    return this.workers.find(w => !w.busy) || null;
  }

  private sendToWorker(
    instance: WorkerInstance,
    command: WorkerCommand,
    payload: unknown
  ): Promise<unknown> {
    return new Promise((resolve, reject) => {
      const id = `req-${++this.requestCounter}`;

      const timeout = setTimeout(() => {
        this.pendingRequests.delete(id);
        reject(new Error('Request timeout'));
      }, this.config.requestTimeout);

      this.pendingRequests.set(id, { resolve, reject, timeout });

      instance.busy = true;
      instance.worker.postMessage({ id, command, payload } as WorkerMessage);
    });
  }

  private processQueue(): void {
    while (this.requestQueue.length > 0) {
      const worker = this.getAvailableWorker();
      if (!worker) break;

      const request = this.requestQueue.shift()!;
      this.sendToWorker(worker, request.message.command, request.message.payload)
        .then(request.resolve)
        .catch(request.reject);
    }
  }

  async execute(command: WorkerCommand, payload: unknown): Promise<unknown> {
    if (!this.initialized) {
      await this.initialize();
    }

    const worker = this.getAvailableWorker();

    if (worker) {
      return this.sendToWorker(worker, command, payload);
    }

    // Queue the request
    return new Promise((resolve, reject) => {
      this.requestQueue.push({
        message: { id: '', command, payload },
        resolve,
        reject,
      });
    });
  }

  // Training progress callback
  onTrainingProgress?: (progress: { type: string; epoch?: number; batch?: number; logs?: unknown }) => void;

  // ==================== Public API ====================

  async loadModel(modelInfo: {
    id: string;
    url: string;
    format: 'tfjs' | 'onnx';
    inputShape: number[];
  }): Promise<void> {
    await this.execute('loadModel', modelInfo);
  }

  async predict(
    modelId: string,
    input: number[] | Float32Array,
    inputShape: number[]
  ): Promise<number[]> {
    return this.execute('predict', { modelId, input: Array.from(input), inputShape }) as Promise<number[]>;
  }

  async batchPredict(
    modelId: string,
    inputs: (number[] | Float32Array)[],
    inputShape: number[]
  ): Promise<number[][]> {
    return this.execute('batchPredict', {
      modelId,
      inputs: inputs.map(i => Array.from(i)),
      inputShape,
    }) as Promise<number[][]>;
  }

  async analyzeAudio(audio: Float32Array, sampleRate: number): Promise<{
    rms: number;
    peak: number;
    zeroCrossings: number;
    spectralCentroid: number;
    spectralFlatness: number;
  }> {
    return this.execute('analyze', { audio, sampleRate }) as Promise<{
      rms: number;
      peak: number;
      zeroCrossings: number;
      spectralCentroid: number;
      spectralFlatness: number;
    }>;
  }

  async train(
    modelId: string,
    trainX: number[][],
    trainY: number[][],
    config: {
      epochs: number;
      batchSize: number;
      learningRate: number;
      validationSplit: number;
      callbacks?: {
        onEpochEnd?: boolean;
        onBatchEnd?: boolean;
      };
    }
  ): Promise<{ loss: number; accuracy: number }> {
    return this.execute('train', { modelId, trainX, trainY, config }) as Promise<{
      loss: number;
      accuracy: number;
    }>;
  }

  async getStats(): Promise<{
    numModels: number;
    memoryInfo: unknown;
    backend: string;
  }> {
    return this.execute('getStats', undefined) as Promise<{
      numModels: number;
      memoryInfo: unknown;
      backend: string;
    }>;
  }

  async disposeModel(modelId?: string): Promise<void> {
    await this.execute('dispose', modelId);
  }

  dispose(): void {
    // Clear pending requests
    for (const pending of this.pendingRequests.values()) {
      clearTimeout(pending.timeout);
      pending.reject(new Error('Client disposed'));
    }
    this.pendingRequests.clear();

    // Clear queue
    for (const request of this.requestQueue) {
      request.reject(new Error('Client disposed'));
    }
    this.requestQueue = [];

    // Terminate workers
    for (const instance of this.workers) {
      instance.worker.terminate();
    }
    this.workers = [];
    this.initialized = false;
  }
}

// ==================== Singleton Instance ====================

let clientInstance: InferenceWorkerClient | null = null;

export function getWorkerClient(config?: WorkerPoolConfig): InferenceWorkerClient {
  if (!clientInstance) {
    clientInstance = new InferenceWorkerClient(config);
  }
  return clientInstance;
}

export default InferenceWorkerClient;
