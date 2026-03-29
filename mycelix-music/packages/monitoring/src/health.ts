// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Health Check Utilities
 *
 * Standardized health checks for services and dependencies.
 */

export interface HealthCheckResult {
  status: 'healthy' | 'degraded' | 'unhealthy';
  checks: Record<string, ComponentHealth>;
  timestamp: string;
  uptime: number;
}

export interface ComponentHealth {
  status: 'healthy' | 'unhealthy';
  latency?: number;
  message?: string;
  lastCheck?: string;
}

export type HealthChecker = () => Promise<ComponentHealth>;

interface HealthCheckOptions {
  timeout?: number;
}

/**
 * Create a health check runner
 */
export function healthCheck(
  checkers: Record<string, HealthChecker>,
  options: HealthCheckOptions = {}
): () => Promise<HealthCheckResult> {
  const startTime = Date.now();
  const timeout = options.timeout || 5000;

  return async (): Promise<HealthCheckResult> => {
    const checks: Record<string, ComponentHealth> = {};
    let overallStatus: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';

    const checkPromises = Object.entries(checkers).map(async ([name, checker]) => {
      const start = Date.now();
      try {
        const result = await Promise.race([
          checker(),
          new Promise<ComponentHealth>((_, reject) =>
            setTimeout(() => reject(new Error('Health check timeout')), timeout)
          ),
        ]);

        checks[name] = {
          ...result,
          latency: Date.now() - start,
          lastCheck: new Date().toISOString(),
        };

        if (result.status === 'unhealthy') {
          overallStatus = 'unhealthy';
        }
      } catch (error) {
        checks[name] = {
          status: 'unhealthy',
          latency: Date.now() - start,
          message: error instanceof Error ? error.message : 'Unknown error',
          lastCheck: new Date().toISOString(),
        };
        overallStatus = 'unhealthy';
      }
    });

    await Promise.all(checkPromises);

    // If some checks are unhealthy but not all, mark as degraded
    const statuses = Object.values(checks).map(c => c.status);
    const unhealthyCount = statuses.filter(s => s === 'unhealthy').length;
    if (unhealthyCount > 0 && unhealthyCount < statuses.length) {
      overallStatus = 'degraded';
    }

    return {
      status: overallStatus,
      checks,
      timestamp: new Date().toISOString(),
      uptime: Math.floor((Date.now() - startTime) / 1000),
    };
  };
}

/**
 * Common health checkers
 */
export const commonCheckers = {
  /**
   * Check Redis connection
   */
  redis: (client: { ping: () => Promise<string> }): HealthChecker => async () => {
    try {
      const result = await client.ping();
      return {
        status: result === 'PONG' ? 'healthy' : 'unhealthy',
        message: result === 'PONG' ? undefined : `Unexpected response: ${result}`,
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        message: error instanceof Error ? error.message : 'Redis connection failed',
      };
    }
  },

  /**
   * Check PostgreSQL connection
   */
  postgres: (client: { query: (q: string) => Promise<unknown> }): HealthChecker => async () => {
    try {
      await client.query('SELECT 1');
      return { status: 'healthy' };
    } catch (error) {
      return {
        status: 'unhealthy',
        message: error instanceof Error ? error.message : 'PostgreSQL connection failed',
      };
    }
  },

  /**
   * Check Elasticsearch connection
   */
  elasticsearch: (client: { ping: () => Promise<boolean> }): HealthChecker => async () => {
    try {
      const result = await client.ping();
      return {
        status: result ? 'healthy' : 'unhealthy',
        message: result ? undefined : 'Elasticsearch ping failed',
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        message: error instanceof Error ? error.message : 'Elasticsearch connection failed',
      };
    }
  },

  /**
   * Check external HTTP service
   */
  httpEndpoint: (url: string): HealthChecker => async () => {
    try {
      const response = await fetch(url, {
        method: 'HEAD',
        signal: AbortSignal.timeout(3000),
      });
      return {
        status: response.ok ? 'healthy' : 'unhealthy',
        message: response.ok ? undefined : `HTTP ${response.status}`,
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        message: error instanceof Error ? error.message : 'HTTP request failed',
      };
    }
  },

  /**
   * Check memory usage
   */
  memory: (thresholdPercent: number = 90): HealthChecker => async () => {
    const used = process.memoryUsage();
    const heapUsedPercent = (used.heapUsed / used.heapTotal) * 100;

    return {
      status: heapUsedPercent < thresholdPercent ? 'healthy' : 'unhealthy',
      message: `Heap: ${Math.round(heapUsedPercent)}% used (${Math.round(used.heapUsed / 1024 / 1024)}MB / ${Math.round(used.heapTotal / 1024 / 1024)}MB)`,
    };
  },
};
