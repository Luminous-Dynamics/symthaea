// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Request Priority Queue
 *
 * Process heavy operations asynchronously with priority levels.
 * Supports job scheduling, retries, and progress tracking.
 */

import { EventEmitter } from 'events';
import { randomUUID } from 'crypto';
import { getLogger } from '../logging';
import { getMetrics } from '../metrics';

const logger = getLogger();

/**
 * Priority levels
 */
export enum Priority {
  CRITICAL = 0,
  HIGH = 1,
  NORMAL = 2,
  LOW = 3,
  BACKGROUND = 4,
}

/**
 * Job status
 */
export enum JobStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
  RETRYING = 'retrying',
}

/**
 * Job definition
 */
export interface Job<T = unknown, R = unknown> {
  id: string;
  type: string;
  priority: Priority;
  data: T;
  status: JobStatus;
  result?: R;
  error?: string;
  attempts: number;
  maxAttempts: number;
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
  progress?: number;
  metadata?: Record<string, unknown>;
}

/**
 * Job handler function
 */
export type JobHandler<T = unknown, R = unknown> = (
  job: Job<T, R>,
  context: JobContext
) => Promise<R>;

/**
 * Job context for handlers
 */
export interface JobContext {
  updateProgress: (progress: number) => void;
  log: (message: string, data?: Record<string, unknown>) => void;
  isCancelled: () => boolean;
}

/**
 * Queue configuration
 */
export interface QueueConfig {
  /** Maximum concurrent jobs */
  concurrency: number;
  /** Default max attempts */
  defaultMaxAttempts: number;
  /** Retry delay in ms */
  retryDelay: number;
  /** Job timeout in ms */
  jobTimeout: number;
  /** Enable persistence */
  persistent: boolean;
}

/**
 * Default configuration
 */
const defaultConfig: QueueConfig = {
  concurrency: 5,
  defaultMaxAttempts: 3,
  retryDelay: 5000,
  jobTimeout: 300000, // 5 minutes
  persistent: false,
};

/**
 * Priority Queue implementation
 */
export class PriorityQueue extends EventEmitter {
  private config: QueueConfig;
  private handlers: Map<string, JobHandler> = new Map();
  private jobs: Map<string, Job> = new Map();
  private pending: Job[][] = [[], [], [], [], []]; // One array per priority
  private running: Set<string> = new Set();
  private cancelled: Set<string> = new Set();
  private paused = false;
  private processing = false;

  constructor(config: Partial<QueueConfig> = {}) {
    super();
    this.config = { ...defaultConfig, ...config };

    // Register metrics
    const metrics = getMetrics();
    metrics.createGauge('queue_jobs_pending', 'Pending jobs in queue', ['priority']);
    metrics.createGauge('queue_jobs_running', 'Running jobs', []);
    metrics.createCounter('queue_jobs_completed_total', 'Completed jobs', ['type', 'status']);
    metrics.createHistogram('queue_job_duration_seconds', 'Job duration', ['type']);
  }

  /**
   * Register a job handler
   */
  register<T, R>(type: string, handler: JobHandler<T, R>): void {
    this.handlers.set(type, handler as JobHandler);
    logger.info(`Registered queue handler: ${type}`);
  }

  /**
   * Add a job to the queue
   */
  async add<T>(
    type: string,
    data: T,
    options: {
      priority?: Priority;
      maxAttempts?: number;
      metadata?: Record<string, unknown>;
    } = {}
  ): Promise<Job<T>> {
    if (!this.handlers.has(type)) {
      throw new Error(`No handler registered for job type: ${type}`);
    }

    const job: Job<T> = {
      id: randomUUID(),
      type,
      priority: options.priority ?? Priority.NORMAL,
      data,
      status: JobStatus.PENDING,
      attempts: 0,
      maxAttempts: options.maxAttempts ?? this.config.defaultMaxAttempts,
      createdAt: new Date(),
      metadata: options.metadata,
    };

    this.jobs.set(job.id, job);
    this.pending[job.priority].push(job);
    this.updateMetrics();

    logger.debug('Job added to queue', {
      jobId: job.id,
      type: job.type,
      priority: Priority[job.priority],
    });

    this.emit('job:added', job);

    // Start processing if not already
    this.process();

    return job;
  }

  /**
   * Get job by ID
   */
  getJob(id: string): Job | undefined {
    return this.jobs.get(id);
  }

  /**
   * Cancel a job
   */
  cancel(id: string): boolean {
    const job = this.jobs.get(id);
    if (!job) return false;

    if (job.status === JobStatus.PENDING) {
      job.status = JobStatus.CANCELLED;
      // Remove from pending queue
      const queue = this.pending[job.priority];
      const index = queue.findIndex(j => j.id === id);
      if (index !== -1) {
        queue.splice(index, 1);
      }
      this.emit('job:cancelled', job);
      return true;
    }

    if (job.status === JobStatus.RUNNING) {
      this.cancelled.add(id);
      return true;
    }

    return false;
  }

  /**
   * Pause the queue
   */
  pause(): void {
    this.paused = true;
    logger.info('Queue paused');
    this.emit('queue:paused');
  }

  /**
   * Resume the queue
   */
  resume(): void {
    this.paused = false;
    logger.info('Queue resumed');
    this.emit('queue:resumed');
    this.process();
  }

  /**
   * Process jobs
   */
  private async process(): Promise<void> {
    if (this.processing || this.paused) return;
    this.processing = true;

    try {
      while (!this.paused && this.running.size < this.config.concurrency) {
        const job = this.getNextJob();
        if (!job) break;

        this.running.add(job.id);
        this.updateMetrics();

        // Process job without awaiting (fire and forget)
        this.processJob(job).catch(error => {
          logger.error(`Error processing job ${job.id}`, error);
        });
      }
    } finally {
      this.processing = false;
    }
  }

  /**
   * Get next job to process (by priority)
   */
  private getNextJob(): Job | undefined {
    for (const queue of this.pending) {
      if (queue.length > 0) {
        return queue.shift();
      }
    }
    return undefined;
  }

  /**
   * Process a single job
   */
  private async processJob(job: Job): Promise<void> {
    const handler = this.handlers.get(job.type);
    if (!handler) {
      job.status = JobStatus.FAILED;
      job.error = 'No handler registered';
      this.running.delete(job.id);
      this.updateMetrics();
      return;
    }

    job.status = JobStatus.RUNNING;
    job.startedAt = new Date();
    job.attempts++;

    this.emit('job:started', job);

    const context: JobContext = {
      updateProgress: (progress: number) => {
        job.progress = Math.max(0, Math.min(100, progress));
        this.emit('job:progress', job);
      },
      log: (message: string, data?: Record<string, unknown>) => {
        logger.debug(`[Job ${job.id}] ${message}`, data);
      },
      isCancelled: () => this.cancelled.has(job.id),
    };

    const startTime = Date.now();

    try {
      // Set timeout
      const timeoutPromise = new Promise<never>((_, reject) => {
        setTimeout(() => reject(new Error('Job timeout')), this.config.jobTimeout);
      });

      // Execute handler
      const result = await Promise.race([
        handler(job, context),
        timeoutPromise,
      ]);

      // Check if cancelled during execution
      if (this.cancelled.has(job.id)) {
        job.status = JobStatus.CANCELLED;
        this.cancelled.delete(job.id);
        this.emit('job:cancelled', job);
      } else {
        job.status = JobStatus.COMPLETED;
        job.result = result;
        job.completedAt = new Date();
        job.progress = 100;
        this.emit('job:completed', job);

        getMetrics().incCounter('queue_jobs_completed_total', {
          type: job.type,
          status: 'success',
        });
      }
    } catch (error) {
      const err = error as Error;
      job.error = err.message;

      if (job.attempts < job.maxAttempts) {
        job.status = JobStatus.RETRYING;
        this.emit('job:retry', job);

        // Schedule retry
        setTimeout(() => {
          if (!this.cancelled.has(job.id)) {
            job.status = JobStatus.PENDING;
            this.pending[job.priority].push(job);
            this.updateMetrics();
            this.process();
          }
        }, this.config.retryDelay * job.attempts);
      } else {
        job.status = JobStatus.FAILED;
        job.completedAt = new Date();
        this.emit('job:failed', job);

        getMetrics().incCounter('queue_jobs_completed_total', {
          type: job.type,
          status: 'failed',
        });
      }
    } finally {
      this.running.delete(job.id);

      const duration = (Date.now() - startTime) / 1000;
      getMetrics().observeHistogram('queue_job_duration_seconds', duration, {
        type: job.type,
      });

      this.updateMetrics();

      // Continue processing
      this.process();
    }
  }

  /**
   * Update metrics
   */
  private updateMetrics(): void {
    const metrics = getMetrics();

    for (let i = 0; i < this.pending.length; i++) {
      metrics.setGauge('queue_jobs_pending', this.pending[i].length, {
        priority: Priority[i],
      });
    }

    metrics.setGauge('queue_jobs_running', this.running.size, {});
  }

  /**
   * Get queue statistics
   */
  getStats(): {
    pending: Record<string, number>;
    running: number;
    total: number;
    byStatus: Record<string, number>;
  } {
    const pending: Record<string, number> = {};
    for (let i = 0; i < this.pending.length; i++) {
      pending[Priority[i]] = this.pending[i].length;
    }

    const byStatus: Record<string, number> = {};
    for (const job of this.jobs.values()) {
      byStatus[job.status] = (byStatus[job.status] || 0) + 1;
    }

    return {
      pending,
      running: this.running.size,
      total: this.jobs.size,
      byStatus,
    };
  }

  /**
   * Clear completed/failed jobs
   */
  cleanup(olderThan?: Date): number {
    const cutoff = olderThan || new Date(Date.now() - 86400000); // 24 hours default
    let removed = 0;

    for (const [id, job] of this.jobs) {
      if (
        (job.status === JobStatus.COMPLETED || job.status === JobStatus.FAILED || job.status === JobStatus.CANCELLED) &&
        job.completedAt &&
        job.completedAt < cutoff
      ) {
        this.jobs.delete(id);
        removed++;
      }
    }

    logger.info(`Cleaned up ${removed} old jobs`);
    return removed;
  }

  /**
   * Wait for a job to complete
   */
  async waitFor<R>(jobId: string, timeout = 60000): Promise<Job<unknown, R>> {
    const job = this.jobs.get(jobId);
    if (!job) {
      throw new Error(`Job not found: ${jobId}`);
    }

    if (job.status === JobStatus.COMPLETED || job.status === JobStatus.FAILED) {
      return job as Job<unknown, R>;
    }

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        cleanup();
        reject(new Error('Job wait timeout'));
      }, timeout);

      const cleanup = () => {
        clearTimeout(timer);
        this.off('job:completed', onComplete);
        this.off('job:failed', onFailed);
        this.off('job:cancelled', onCancelled);
      };

      const onComplete = (j: Job) => {
        if (j.id === jobId) {
          cleanup();
          resolve(j as Job<unknown, R>);
        }
      };

      const onFailed = (j: Job) => {
        if (j.id === jobId) {
          cleanup();
          reject(new Error(j.error || 'Job failed'));
        }
      };

      const onCancelled = (j: Job) => {
        if (j.id === jobId) {
          cleanup();
          reject(new Error('Job cancelled'));
        }
      };

      this.on('job:completed', onComplete);
      this.on('job:failed', onFailed);
      this.on('job:cancelled', onCancelled);
    });
  }
}

/**
 * Global queue instance
 */
let queue: PriorityQueue | null = null;

export function getQueue(): PriorityQueue {
  if (!queue) {
    queue = new PriorityQueue();
  }
  return queue;
}

export function resetQueue(): void {
  queue = null;
}

/**
 * Common job types
 */
export const JobTypes = {
  EXPORT_DATA: 'export:data',
  AGGREGATE_ANALYTICS: 'analytics:aggregate',
  SEND_WEBHOOK: 'webhook:send',
  PROCESS_PAYMENT: 'payment:process',
  GENERATE_REPORT: 'report:generate',
  SYNC_BLOCKCHAIN: 'blockchain:sync',
  CLEANUP_OLD_DATA: 'maintenance:cleanup',
};

export default PriorityQueue;
