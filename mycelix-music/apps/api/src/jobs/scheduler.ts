// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Background Job Scheduler
 *
 * Simple but robust job scheduling system for:
 * - Analytics refresh
 * - Cleanup tasks
 * - Pending transaction processing
 * - Cache warming
 */

import { EventEmitter } from 'events';

/**
 * Job definition
 */
export interface JobDefinition {
  name: string;
  handler: () => Promise<void>;
  schedule: {
    type: 'interval' | 'cron';
    value: number | string; // ms for interval, cron expression for cron
  };
  timeout?: number;
  retries?: number;
  enabled?: boolean;
}

/**
 * Job execution result
 */
export interface JobResult {
  name: string;
  startedAt: Date;
  completedAt: Date;
  duration: number;
  success: boolean;
  error?: string;
  retryCount: number;
}

/**
 * Job status
 */
export interface JobStatus {
  name: string;
  enabled: boolean;
  running: boolean;
  lastRun?: JobResult;
  nextRun?: Date;
  runCount: number;
  errorCount: number;
}

/**
 * Job scheduler events
 */
export interface SchedulerEvents {
  'job:start': (name: string) => void;
  'job:complete': (result: JobResult) => void;
  'job:error': (name: string, error: Error) => void;
  'scheduler:start': () => void;
  'scheduler:stop': () => void;
}

/**
 * Job scheduler class
 */
export class JobScheduler extends EventEmitter {
  private jobs = new Map<string, {
    definition: JobDefinition;
    timer: NodeJS.Timeout | null;
    status: JobStatus;
  }>();
  private running = false;

  constructor() {
    super();
  }

  /**
   * Register a job
   */
  register(definition: JobDefinition): void {
    if (this.jobs.has(definition.name)) {
      throw new Error(`Job '${definition.name}' is already registered`);
    }

    this.jobs.set(definition.name, {
      definition: {
        timeout: 60000, // 1 minute default timeout
        retries: 0,
        enabled: true,
        ...definition,
      },
      timer: null,
      status: {
        name: definition.name,
        enabled: definition.enabled !== false,
        running: false,
        runCount: 0,
        errorCount: 0,
      },
    });

    console.log(`[Scheduler] Registered job: ${definition.name}`);
  }

  /**
   * Start the scheduler
   */
  start(): void {
    if (this.running) {
      console.log('[Scheduler] Already running');
      return;
    }

    this.running = true;
    console.log('[Scheduler] Starting...');

    for (const [name, job] of this.jobs) {
      if (job.definition.enabled !== false) {
        this.scheduleJob(name);
      }
    }

    this.emit('scheduler:start');
    console.log(`[Scheduler] Started with ${this.jobs.size} jobs`);
  }

  /**
   * Stop the scheduler
   */
  stop(): void {
    if (!this.running) {
      return;
    }

    this.running = false;
    console.log('[Scheduler] Stopping...');

    for (const [name, job] of this.jobs) {
      if (job.timer) {
        clearTimeout(job.timer);
        job.timer = null;
      }
    }

    this.emit('scheduler:stop');
    console.log('[Scheduler] Stopped');
  }

  /**
   * Run a job immediately
   */
  async runNow(name: string): Promise<JobResult> {
    const job = this.jobs.get(name);
    if (!job) {
      throw new Error(`Job '${name}' not found`);
    }

    return this.executeJob(name);
  }

  /**
   * Enable a job
   */
  enable(name: string): void {
    const job = this.jobs.get(name);
    if (!job) {
      throw new Error(`Job '${name}' not found`);
    }

    job.status.enabled = true;
    if (this.running && !job.timer) {
      this.scheduleJob(name);
    }
  }

  /**
   * Disable a job
   */
  disable(name: string): void {
    const job = this.jobs.get(name);
    if (!job) {
      throw new Error(`Job '${name}' not found`);
    }

    job.status.enabled = false;
    if (job.timer) {
      clearTimeout(job.timer);
      job.timer = null;
    }
  }

  /**
   * Get job status
   */
  getStatus(name: string): JobStatus | undefined {
    return this.jobs.get(name)?.status;
  }

  /**
   * Get all job statuses
   */
  getAllStatuses(): JobStatus[] {
    return Array.from(this.jobs.values()).map(j => j.status);
  }

  /**
   * Schedule a job
   */
  private scheduleJob(name: string): void {
    const job = this.jobs.get(name);
    if (!job || !job.status.enabled) {
      return;
    }

    const { definition } = job;

    if (definition.schedule.type === 'interval') {
      const intervalMs = definition.schedule.value as number;

      job.timer = setTimeout(async () => {
        await this.executeJob(name);
        if (this.running && job.status.enabled) {
          this.scheduleJob(name);
        }
      }, intervalMs);

      job.status.nextRun = new Date(Date.now() + intervalMs);
    }
    // TODO: Add cron support
  }

  /**
   * Execute a job
   */
  private async executeJob(name: string, retryCount = 0): Promise<JobResult> {
    const job = this.jobs.get(name);
    if (!job) {
      throw new Error(`Job '${name}' not found`);
    }

    const { definition, status } = job;
    const startedAt = new Date();

    status.running = true;
    this.emit('job:start', name);

    let result: JobResult;

    try {
      // Execute with timeout
      await Promise.race([
        definition.handler(),
        new Promise((_, reject) =>
          setTimeout(
            () => reject(new Error('Job timeout')),
            definition.timeout
          )
        ),
      ]);

      const completedAt = new Date();
      result = {
        name,
        startedAt,
        completedAt,
        duration: completedAt.getTime() - startedAt.getTime(),
        success: true,
        retryCount,
      };

      status.runCount++;
    } catch (error) {
      const completedAt = new Date();
      const err = error as Error;

      // Retry if configured
      if (retryCount < (definition.retries || 0)) {
        console.log(`[Scheduler] Job '${name}' failed, retrying (${retryCount + 1}/${definition.retries})...`);
        return this.executeJob(name, retryCount + 1);
      }

      result = {
        name,
        startedAt,
        completedAt,
        duration: completedAt.getTime() - startedAt.getTime(),
        success: false,
        error: err.message,
        retryCount,
      };

      status.errorCount++;
      this.emit('job:error', name, err);
      console.error(`[Scheduler] Job '${name}' failed:`, err.message);
    }

    status.running = false;
    status.lastRun = result;
    this.emit('job:complete', result);

    return result;
  }
}

// Singleton instance
let _scheduler: JobScheduler | null = null;

/**
 * Get or create scheduler
 */
export function getScheduler(): JobScheduler {
  if (!_scheduler) {
    _scheduler = new JobScheduler();
  }
  return _scheduler;
}

export default JobScheduler;
