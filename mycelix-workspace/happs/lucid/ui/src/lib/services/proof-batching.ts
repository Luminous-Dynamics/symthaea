/**
 * Proof Batching Service
 *
 * Efficiently batch and queue zero-knowledge proof generation requests.
 * Features:
 * - Queue management for proof requests
 * - Batch processing with configurable batch size
 * - Priority-based proof generation
 * - Caching of recent proofs
 */

import {
  generateAnonymousBeliefProof,
  generateReputationRangeProof,
  type AnonymousBeliefProofOutput,
  type ReputationRangeProofOutput,
} from './zkp';

// ============================================================================
// TYPES
// ============================================================================

export type ProofType = 'anonymous_belief' | 'reputation_range';

export interface ProofRequest {
  id: string;
  type: ProofType;
  params: BeliefProofParams | ReputationProofParams;
  priority: 'low' | 'normal' | 'high';
  createdAt: number;
  resolve: (result: ProofResult) => void;
  reject: (error: Error) => void;
}

export interface BeliefProofParams {
  beliefHash: string;
  agentSecret: string;
}

export interface ReputationProofParams {
  actualReputation: number;
  minThreshold: number;
}

export interface ProofResult {
  requestId: string;
  type: ProofType;
  proof: AnonymousBeliefProofOutput | ReputationRangeProofOutput;
  generatedAt: number;
  processingTimeMs: number;
}

export interface BatchStats {
  pendingCount: number;
  processingCount: number;
  completedCount: number;
  failedCount: number;
  averageProcessingTimeMs: number;
  cacheHitRate: number;
}

// ============================================================================
// PROOF CACHE
// ============================================================================

interface CacheEntry<T> {
  proof: T;
  generatedAt: number;
  expiresAt: number;
}

class ProofCache<T> {
  private cache = new Map<string, CacheEntry<T>>();
  private readonly ttlMs: number;
  private hits = 0;
  private misses = 0;

  constructor(ttlMs: number = 5 * 60 * 1000) {
    this.ttlMs = ttlMs;
  }

  generateKey(params: Record<string, unknown>): string {
    return JSON.stringify(params);
  }

  get(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) {
      this.misses++;
      return null;
    }

    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      this.misses++;
      return null;
    }

    this.hits++;
    return entry.proof;
  }

  set(key: string, proof: T): void {
    const now = Date.now();
    this.cache.set(key, {
      proof,
      generatedAt: now,
      expiresAt: now + this.ttlMs,
    });
  }

  getHitRate(): number {
    const total = this.hits + this.misses;
    return total > 0 ? this.hits / total : 0;
  }

  clear(): void {
    this.cache.clear();
    this.hits = 0;
    this.misses = 0;
  }

  cleanup(): void {
    const now = Date.now();
    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.expiresAt) {
        this.cache.delete(key);
      }
    }
  }
}

// ============================================================================
// PROOF BATCHING SERVICE
// ============================================================================

class ProofBatchingService {
  private queue: ProofRequest[] = [];
  private processing = new Set<string>();
  private readonly batchSize: number;
  private readonly processIntervalMs: number;
  private processTimer: number | null = null;

  private beliefProofCache = new ProofCache<AnonymousBeliefProofOutput>();
  private reputationProofCache = new ProofCache<ReputationRangeProofOutput>();

  private completedCount = 0;
  private failedCount = 0;
  private totalProcessingTimeMs = 0;

  constructor(batchSize: number = 5, processIntervalMs: number = 100) {
    this.batchSize = batchSize;
    this.processIntervalMs = processIntervalMs;
  }

  /**
   * Request an anonymous belief proof
   */
  async requestBeliefProof(
    beliefHash: string,
    agentSecret: string,
    priority: 'low' | 'normal' | 'high' = 'normal'
  ): Promise<AnonymousBeliefProofOutput> {
    const cacheKey = this.beliefProofCache.generateKey({ beliefHash, agentSecret });
    const cached = this.beliefProofCache.get(cacheKey);
    if (cached) {
      return cached;
    }

    return new Promise((resolve, reject) => {
      const request: ProofRequest = {
        id: `belief-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        type: 'anonymous_belief',
        params: { beliefHash, agentSecret },
        priority,
        createdAt: Date.now(),
        resolve: (result) => {
          const proof = result.proof as AnonymousBeliefProofOutput;
          this.beliefProofCache.set(cacheKey, proof);
          resolve(proof);
        },
        reject,
      };

      this.enqueue(request);
    });
  }

  /**
   * Request a reputation range proof
   */
  async requestReputationProof(
    actualReputation: number,
    minThreshold: number,
    priority: 'low' | 'normal' | 'high' = 'normal'
  ): Promise<ReputationRangeProofOutput> {
    const cacheKey = this.reputationProofCache.generateKey({ actualReputation, minThreshold });
    const cached = this.reputationProofCache.get(cacheKey);
    if (cached) {
      return cached;
    }

    return new Promise((resolve, reject) => {
      const request: ProofRequest = {
        id: `reputation-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        type: 'reputation_range',
        params: { actualReputation, minThreshold },
        priority,
        createdAt: Date.now(),
        resolve: (result) => {
          const proof = result.proof as ReputationRangeProofOutput;
          this.reputationProofCache.set(cacheKey, proof);
          resolve(proof);
        },
        reject,
      };

      this.enqueue(request);
    });
  }

  /**
   * Add a request to the queue
   */
  private enqueue(request: ProofRequest): void {
    // Insert by priority
    const priorityOrder = { high: 0, normal: 1, low: 2 };
    const insertIndex = this.queue.findIndex(
      (r) => priorityOrder[r.priority] > priorityOrder[request.priority]
    );

    if (insertIndex === -1) {
      this.queue.push(request);
    } else {
      this.queue.splice(insertIndex, 0, request);
    }

    this.startProcessing();
  }

  /**
   * Start the processing loop
   */
  private startProcessing(): void {
    if (this.processTimer !== null) return;

    this.processTimer = window.setInterval(() => {
      this.processBatch();
    }, this.processIntervalMs);

    // Process immediately
    this.processBatch();
  }

  /**
   * Stop the processing loop
   */
  private stopProcessing(): void {
    if (this.processTimer !== null) {
      clearInterval(this.processTimer);
      this.processTimer = null;
    }
  }

  /**
   * Process a batch of proof requests
   */
  private async processBatch(): Promise<void> {
    if (this.queue.length === 0 && this.processing.size === 0) {
      this.stopProcessing();
      return;
    }

    // Get next batch respecting max concurrent
    const available = this.batchSize - this.processing.size;
    if (available <= 0) return;

    const batch = this.queue.splice(0, available);

    // Process in parallel
    await Promise.all(batch.map((request) => this.processRequest(request)));
  }

  /**
   * Process a single proof request
   */
  private async processRequest(request: ProofRequest): Promise<void> {
    this.processing.add(request.id);
    const startTime = Date.now();

    try {
      let proof: AnonymousBeliefProofOutput | ReputationRangeProofOutput | null = null;

      if (request.type === 'anonymous_belief') {
        const params = request.params as BeliefProofParams;
        proof = await generateAnonymousBeliefProof(params.beliefHash, params.agentSecret);
      } else {
        const params = request.params as ReputationProofParams;
        proof = await generateReputationRangeProof(params.actualReputation, params.minThreshold);
      }

      if (!proof) {
        throw new Error(
          `Proof generation failed for ${request.type}: The ZKP backend returned null. ` +
          `This may indicate: (1) Tauri ZKP bridge not initialized, ` +
          `(2) Invalid input parameters, or (3) Proof circuit error. ` +
          `Check browser console for detailed errors.`
        );
      }

      const processingTimeMs = Date.now() - startTime;
      this.totalProcessingTimeMs += processingTimeMs;
      this.completedCount++;

      request.resolve({
        requestId: request.id,
        type: request.type,
        proof,
        generatedAt: Date.now(),
        processingTimeMs,
      });
    } catch (error) {
      this.failedCount++;
      request.reject(error instanceof Error ? error : new Error(String(error)));
    } finally {
      this.processing.delete(request.id);
    }
  }

  /**
   * Get current batch statistics
   */
  getStats(): BatchStats {
    return {
      pendingCount: this.queue.length,
      processingCount: this.processing.size,
      completedCount: this.completedCount,
      failedCount: this.failedCount,
      averageProcessingTimeMs:
        this.completedCount > 0 ? this.totalProcessingTimeMs / this.completedCount : 0,
      cacheHitRate:
        (this.beliefProofCache.getHitRate() + this.reputationProofCache.getHitRate()) / 2,
    };
  }

  /**
   * Cancel all pending requests
   */
  cancelAll(): void {
    for (const request of this.queue) {
      request.reject(new Error('Request cancelled'));
    }
    this.queue = [];
    this.stopProcessing();
  }

  /**
   * Clear caches
   */
  clearCaches(): void {
    this.beliefProofCache.clear();
    this.reputationProofCache.clear();
  }

  /**
   * Cleanup expired cache entries
   */
  cleanupCaches(): void {
    this.beliefProofCache.cleanup();
    this.reputationProofCache.cleanup();
  }
}

// ============================================================================
// SINGLETON EXPORT
// ============================================================================

export const proofBatcher = new ProofBatchingService();

// Convenience functions
export const requestBeliefProof = proofBatcher.requestBeliefProof.bind(proofBatcher);
export const requestReputationProof = proofBatcher.requestReputationProof.bind(proofBatcher);
export const getProofBatchStats = proofBatcher.getStats.bind(proofBatcher);
export const cancelAllProofs = proofBatcher.cancelAll.bind(proofBatcher);
export const clearProofCaches = proofBatcher.clearCaches.bind(proofBatcher);
