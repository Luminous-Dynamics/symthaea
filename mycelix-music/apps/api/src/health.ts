// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Health Check System
 *
 * Comprehensive health checks for Kubernetes probes:
 * - /health/live - Liveness probe (is the process running?)
 * - /health/ready - Readiness probe (can it handle traffic?)
 * - /health/startup - Startup probe (has it finished initializing?)
 */

import { Router, Request, Response } from 'express';
import { Pool } from 'pg';
import Redis from 'ioredis';
import { getConfig } from './config';

/**
 * Health check status
 */
export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy';

/**
 * Individual check result
 */
export interface CheckResult {
  status: HealthStatus;
  latency?: number;
  message?: string;
  details?: Record<string, unknown>;
}

/**
 * Overall health response
 */
export interface HealthResponse {
  status: HealthStatus;
  version: string;
  uptime: number;
  timestamp: string;
  checks: Record<string, CheckResult>;
}

/**
 * Health checker class
 */
export class HealthChecker {
  private pool: Pool | null = null;
  private redis: Redis | null = null;
  private startupComplete = false;
  private startTime = Date.now();
  private version: string;

  constructor(version = '1.0.0') {
    this.version = version;
  }

  /**
   * Set database pool for health checks
   */
  setPool(pool: Pool): void {
    this.pool = pool;
  }

  /**
   * Set Redis client for health checks
   */
  setRedis(redis: Redis): void {
    this.redis = redis;
  }

  /**
   * Mark startup as complete
   */
  setStartupComplete(): void {
    this.startupComplete = true;
  }

  /**
   * Check database connectivity
   */
  async checkDatabase(): Promise<CheckResult> {
    if (!this.pool) {
      return { status: 'unhealthy', message: 'Database pool not configured' };
    }

    const start = Date.now();
    try {
      const client = await this.pool.connect();
      await client.query('SELECT 1');
      client.release();

      return {
        status: 'healthy',
        latency: Date.now() - start,
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        latency: Date.now() - start,
        message: (error as Error).message,
      };
    }
  }

  /**
   * Check Redis connectivity
   */
  async checkRedis(): Promise<CheckResult> {
    const config = getConfig();

    if (!config.redis.enabled) {
      return { status: 'healthy', message: 'Redis not configured (optional)' };
    }

    if (!this.redis) {
      return { status: 'degraded', message: 'Redis client not initialized' };
    }

    const start = Date.now();
    try {
      await this.redis.ping();

      return {
        status: 'healthy',
        latency: Date.now() - start,
      };
    } catch (error) {
      return {
        status: 'degraded', // Degraded, not unhealthy (Redis is optional)
        latency: Date.now() - start,
        message: (error as Error).message,
      };
    }
  }

  /**
   * Check memory usage
   */
  checkMemory(): CheckResult {
    const usage = process.memoryUsage();
    const heapUsedMB = Math.round(usage.heapUsed / 1024 / 1024);
    const heapTotalMB = Math.round(usage.heapTotal / 1024 / 1024);
    const heapUsagePercent = Math.round((usage.heapUsed / usage.heapTotal) * 100);

    let status: HealthStatus = 'healthy';
    if (heapUsagePercent > 90) {
      status = 'unhealthy';
    } else if (heapUsagePercent > 75) {
      status = 'degraded';
    }

    return {
      status,
      details: {
        heapUsedMB,
        heapTotalMB,
        heapUsagePercent,
        rssMB: Math.round(usage.rss / 1024 / 1024),
      },
    };
  }

  /**
   * Check event loop lag
   */
  async checkEventLoop(): Promise<CheckResult> {
    return new Promise((resolve) => {
      const start = Date.now();

      setImmediate(() => {
        const lag = Date.now() - start;
        let status: HealthStatus = 'healthy';

        if (lag > 100) {
          status = 'unhealthy';
        } else if (lag > 50) {
          status = 'degraded';
        }

        resolve({
          status,
          latency: lag,
          message: lag > 50 ? 'Event loop lag detected' : undefined,
        });
      });
    });
  }

  /**
   * Liveness check - is the process running?
   */
  async liveness(): Promise<HealthResponse> {
    const memory = this.checkMemory();
    const eventLoop = await this.checkEventLoop();

    const checks = { memory, eventLoop };
    const status = this.aggregateStatus(Object.values(checks));

    return {
      status,
      version: this.version,
      uptime: Math.floor((Date.now() - this.startTime) / 1000),
      timestamp: new Date().toISOString(),
      checks,
    };
  }

  /**
   * Readiness check - can it handle traffic?
   */
  async readiness(): Promise<HealthResponse> {
    const [database, redis, memory, eventLoop] = await Promise.all([
      this.checkDatabase(),
      this.checkRedis(),
      this.checkMemory(),
      this.checkEventLoop(),
    ]);

    const checks = { database, redis, memory, eventLoop };
    const status = this.aggregateStatus(Object.values(checks));

    return {
      status,
      version: this.version,
      uptime: Math.floor((Date.now() - this.startTime) / 1000),
      timestamp: new Date().toISOString(),
      checks,
    };
  }

  /**
   * Startup check - has initialization completed?
   */
  async startup(): Promise<HealthResponse> {
    const database = await this.checkDatabase();

    const checks = {
      database,
      initialized: {
        status: this.startupComplete ? 'healthy' : 'unhealthy',
        message: this.startupComplete ? undefined : 'Startup not complete',
      } as CheckResult,
    };

    const status = this.aggregateStatus(Object.values(checks));

    return {
      status,
      version: this.version,
      uptime: Math.floor((Date.now() - this.startTime) / 1000),
      timestamp: new Date().toISOString(),
      checks,
    };
  }

  /**
   * Aggregate check statuses
   */
  private aggregateStatus(results: CheckResult[]): HealthStatus {
    if (results.some(r => r.status === 'unhealthy')) {
      return 'unhealthy';
    }
    if (results.some(r => r.status === 'degraded')) {
      return 'degraded';
    }
    return 'healthy';
  }
}

/**
 * Create health check router
 */
export function createHealthRouter(checker: HealthChecker): Router {
  const router = Router();

  // Liveness probe
  router.get('/live', async (req: Request, res: Response) => {
    const health = await checker.liveness();
    const status = health.status === 'unhealthy' ? 503 : 200;
    res.status(status).json(health);
  });

  // Readiness probe
  router.get('/ready', async (req: Request, res: Response) => {
    const health = await checker.readiness();
    const status = health.status === 'unhealthy' ? 503 : 200;
    res.status(status).json(health);
  });

  // Startup probe
  router.get('/startup', async (req: Request, res: Response) => {
    const health = await checker.startup();
    const status = health.status === 'unhealthy' ? 503 : 200;
    res.status(status).json(health);
  });

  // Combined health check
  router.get('/', async (req: Request, res: Response) => {
    const health = await checker.readiness();
    const status = health.status === 'unhealthy' ? 503 : 200;
    res.status(status).json(health);
  });

  return router;
}

// Singleton instance
let _healthChecker: HealthChecker | null = null;

/**
 * Get or create health checker
 */
export function getHealthChecker(version?: string): HealthChecker {
  if (!_healthChecker) {
    _healthChecker = new HealthChecker(version);
  }
  return _healthChecker;
}

export default HealthChecker;
