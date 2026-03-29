// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Queue Metrics
 *
 * Metrics for tracking Bull/Redis queue jobs.
 */

import { Registry, Counter, Gauge, Histogram } from 'prom-client';
import { createCounter, createGauge, createHistogram } from './registry';

export interface QueueMetrics {
  jobsTotal: Counter;
  jobsActive: Gauge;
  jobsWaiting: Gauge;
  jobsCompleted: Counter;
  jobsFailed: Counter;
  jobDuration: Histogram;
  jobAttempts: Histogram;
}

/**
 * Create queue metrics
 */
export function createQueueMetrics(registry: Registry, queueName: string): QueueMetrics {
  const labels = { queue: queueName };

  return {
    jobsTotal: createCounter(
      registry,
      `queue_${queueName}_jobs_total`,
      `Total jobs added to ${queueName} queue`,
      ['type']
    ),
    jobsActive: createGauge(
      registry,
      `queue_${queueName}_jobs_active`,
      `Active jobs in ${queueName} queue`
    ),
    jobsWaiting: createGauge(
      registry,
      `queue_${queueName}_jobs_waiting`,
      `Waiting jobs in ${queueName} queue`
    ),
    jobsCompleted: createCounter(
      registry,
      `queue_${queueName}_jobs_completed_total`,
      `Completed jobs in ${queueName} queue`,
      ['type']
    ),
    jobsFailed: createCounter(
      registry,
      `queue_${queueName}_jobs_failed_total`,
      `Failed jobs in ${queueName} queue`,
      ['type', 'error']
    ),
    jobDuration: createHistogram(
      registry,
      `queue_${queueName}_job_duration_seconds`,
      `Job duration in ${queueName} queue`,
      ['type'],
      [1, 5, 10, 30, 60, 120, 300, 600, 1800]
    ),
    jobAttempts: createHistogram(
      registry,
      `queue_${queueName}_job_attempts`,
      `Number of attempts per job in ${queueName} queue`,
      ['type', 'status'],
      [1, 2, 3, 4, 5]
    ),
  };
}

/**
 * Helper class to track queue metrics with Bull events
 */
export class QueueMetricsTracker {
  private jobStartTimes: Map<string, number> = new Map();

  constructor(private metrics: QueueMetrics) {}

  /**
   * Record a new job added
   */
  onJobAdded(jobId: string, type: string): void {
    this.metrics.jobsTotal.labels(type).inc();
    this.metrics.jobsWaiting.inc();
  }

  /**
   * Record job started processing
   */
  onJobActive(jobId: string): void {
    this.jobStartTimes.set(jobId, Date.now());
    this.metrics.jobsActive.inc();
    this.metrics.jobsWaiting.dec();
  }

  /**
   * Record job completed
   */
  onJobCompleted(jobId: string, type: string, attempts: number): void {
    const startTime = this.jobStartTimes.get(jobId);
    if (startTime) {
      const duration = (Date.now() - startTime) / 1000;
      this.metrics.jobDuration.labels(type).observe(duration);
      this.jobStartTimes.delete(jobId);
    }

    this.metrics.jobsCompleted.labels(type).inc();
    this.metrics.jobsActive.dec();
    this.metrics.jobAttempts.labels(type, 'success').observe(attempts);
  }

  /**
   * Record job failed
   */
  onJobFailed(jobId: string, type: string, error: string, attempts: number): void {
    const startTime = this.jobStartTimes.get(jobId);
    if (startTime) {
      const duration = (Date.now() - startTime) / 1000;
      this.metrics.jobDuration.labels(type).observe(duration);
      this.jobStartTimes.delete(jobId);
    }

    // Truncate error for label
    const errorLabel = error.substring(0, 50).replace(/[^a-zA-Z0-9_]/g, '_');
    this.metrics.jobsFailed.labels(type, errorLabel).inc();
    this.metrics.jobsActive.dec();
    this.metrics.jobAttempts.labels(type, 'failed').observe(attempts);
  }

  /**
   * Update waiting count (from queue.getWaitingCount())
   */
  setWaitingCount(count: number): void {
    this.metrics.jobsWaiting.set(count);
  }

  /**
   * Update active count (from queue.getActiveCount())
   */
  setActiveCount(count: number): void {
    this.metrics.jobsActive.set(count);
  }
}
