// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Health Check Routes
 * Provides system health, readiness, and liveness probes
 */

import { Router, Request, Response } from 'express';
import { PrismaClient } from '@prisma/client';
import Redis from 'ioredis';
import { asyncHandler } from '../utils/async-handler';

const router = Router();

// Health check configuration
interface HealthCheck {
  name: string;
  check: () => Promise<HealthStatus>;
  critical: boolean;
}

interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  latency?: number;
  message?: string;
  details?: Record<string, any>;
}

interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  uptime: number;
  timestamp: string;
  checks: Record<string, HealthStatus>;
}

// Service instances (would be injected in production)
let prisma: PrismaClient;
let redis: Redis;

// Initialize services
export function initHealthServices(prismaClient: PrismaClient, redisClient: Redis) {
  prisma = prismaClient;
  redis = redisClient;
}

// Health checks
const healthChecks: HealthCheck[] = [
  {
    name: 'database',
    critical: true,
    check: async (): Promise<HealthStatus> => {
      const start = Date.now();
      try {
        await prisma.$queryRaw`SELECT 1`;
        return {
          status: 'healthy',
          latency: Date.now() - start,
        };
      } catch (error: any) {
        return {
          status: 'unhealthy',
          latency: Date.now() - start,
          message: error.message,
        };
      }
    },
  },
  {
    name: 'cache',
    critical: true,
    check: async (): Promise<HealthStatus> => {
      const start = Date.now();
      try {
        await redis.ping();
        const info = await redis.info('memory');
        const usedMemory = info.match(/used_memory:(\d+)/)?.[1];
        return {
          status: 'healthy',
          latency: Date.now() - start,
          details: {
            usedMemory: usedMemory ? parseInt(usedMemory) : undefined,
          },
        };
      } catch (error: any) {
        return {
          status: 'unhealthy',
          latency: Date.now() - start,
          message: error.message,
        };
      }
    },
  },
  {
    name: 'memory',
    critical: false,
    check: async (): Promise<HealthStatus> => {
      const used = process.memoryUsage();
      const heapUsedPercent = (used.heapUsed / used.heapTotal) * 100;

      let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
      if (heapUsedPercent > 90) status = 'unhealthy';
      else if (heapUsedPercent > 75) status = 'degraded';

      return {
        status,
        details: {
          heapUsed: Math.round(used.heapUsed / 1024 / 1024),
          heapTotal: Math.round(used.heapTotal / 1024 / 1024),
          heapUsedPercent: Math.round(heapUsedPercent),
          rss: Math.round(used.rss / 1024 / 1024),
          external: Math.round(used.external / 1024 / 1024),
        },
      };
    },
  },
  {
    name: 'eventLoop',
    critical: false,
    check: async (): Promise<HealthStatus> => {
      return new Promise((resolve) => {
        const start = Date.now();
        setImmediate(() => {
          const latency = Date.now() - start;
          let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
          if (latency > 100) status = 'unhealthy';
          else if (latency > 50) status = 'degraded';

          resolve({
            status,
            latency,
            message: latency > 50 ? 'Event loop lag detected' : undefined,
          });
        });
      });
    },
  },
];

/**
 * GET /health
 * Basic health check (for load balancers)
 */
router.get(
  '/',
  asyncHandler(async (req: Request, res: Response) => {
    res.json({
      status: 'ok',
      timestamp: new Date().toISOString(),
    });
  })
);

/**
 * GET /health/live
 * Liveness probe (is the service running?)
 */
router.get(
  '/live',
  asyncHandler(async (req: Request, res: Response) => {
    res.json({
      status: 'alive',
      uptime: process.uptime(),
      timestamp: new Date().toISOString(),
    });
  })
);

/**
 * GET /health/ready
 * Readiness probe (is the service ready to accept traffic?)
 */
router.get(
  '/ready',
  asyncHandler(async (req: Request, res: Response) => {
    const criticalChecks = healthChecks.filter(c => c.critical);
    const results: Record<string, HealthStatus> = {};
    let isReady = true;

    for (const check of criticalChecks) {
      try {
        results[check.name] = await check.check();
        if (results[check.name].status === 'unhealthy') {
          isReady = false;
        }
      } catch (error: any) {
        results[check.name] = {
          status: 'unhealthy',
          message: error.message,
        };
        isReady = false;
      }
    }

    const statusCode = isReady ? 200 : 503;
    res.status(statusCode).json({
      status: isReady ? 'ready' : 'not_ready',
      checks: results,
      timestamp: new Date().toISOString(),
    });
  })
);

/**
 * GET /health/detailed
 * Comprehensive health check with all systems
 */
router.get(
  '/detailed',
  asyncHandler(async (req: Request, res: Response) => {
    const results: Record<string, HealthStatus> = {};
    let overallStatus: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';

    // Run all checks in parallel
    await Promise.all(
      healthChecks.map(async (check) => {
        try {
          results[check.name] = await check.check();
        } catch (error: any) {
          results[check.name] = {
            status: 'unhealthy',
            message: error.message,
          };
        }

        // Determine overall status
        if (results[check.name].status === 'unhealthy') {
          if (check.critical) {
            overallStatus = 'unhealthy';
          } else if (overallStatus !== 'unhealthy') {
            overallStatus = 'degraded';
          }
        } else if (results[check.name].status === 'degraded' && overallStatus === 'healthy') {
          overallStatus = 'degraded';
        }
      })
    );

    const response: HealthResponse = {
      status: overallStatus,
      version: process.env.npm_package_version || '1.0.0',
      uptime: process.uptime(),
      timestamp: new Date().toISOString(),
      checks: results,
    };

    const statusCode = overallStatus === 'unhealthy' ? 503 : 200;
    res.status(statusCode).json(response);
  })
);

/**
 * GET /health/metrics
 * Prometheus-compatible metrics endpoint
 */
router.get(
  '/metrics',
  asyncHandler(async (req: Request, res: Response) => {
    const metrics: string[] = [];
    const timestamp = Date.now();

    // Process metrics
    const memory = process.memoryUsage();
    metrics.push(`# HELP process_heap_bytes Node.js heap size in bytes`);
    metrics.push(`# TYPE process_heap_bytes gauge`);
    metrics.push(`process_heap_bytes{type="used"} ${memory.heapUsed} ${timestamp}`);
    metrics.push(`process_heap_bytes{type="total"} ${memory.heapTotal} ${timestamp}`);

    metrics.push(`# HELP process_resident_memory_bytes Resident memory size in bytes`);
    metrics.push(`# TYPE process_resident_memory_bytes gauge`);
    metrics.push(`process_resident_memory_bytes ${memory.rss} ${timestamp}`);

    metrics.push(`# HELP process_uptime_seconds Process uptime in seconds`);
    metrics.push(`# TYPE process_uptime_seconds gauge`);
    metrics.push(`process_uptime_seconds ${process.uptime()} ${timestamp}`);

    // Run health checks and expose as metrics
    for (const check of healthChecks) {
      try {
        const result = await check.check();
        const statusValue = result.status === 'healthy' ? 1 : result.status === 'degraded' ? 0.5 : 0;

        metrics.push(`# HELP health_check_status Health check status (1=healthy, 0.5=degraded, 0=unhealthy)`);
        metrics.push(`# TYPE health_check_status gauge`);
        metrics.push(`health_check_status{name="${check.name}"} ${statusValue} ${timestamp}`);

        if (result.latency !== undefined) {
          metrics.push(`# HELP health_check_latency_ms Health check latency in milliseconds`);
          metrics.push(`# TYPE health_check_latency_ms gauge`);
          metrics.push(`health_check_latency_ms{name="${check.name}"} ${result.latency} ${timestamp}`);
        }
      } catch (error) {
        metrics.push(`health_check_status{name="${check.name}"} 0 ${timestamp}`);
      }
    }

    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.send(metrics.join('\n'));
  })
);

/**
 * GET /health/dependencies
 * Check external dependencies
 */
router.get(
  '/dependencies',
  asyncHandler(async (req: Request, res: Response) => {
    const dependencies: Record<string, { status: string; latency?: number; message?: string }> = {};

    // Check IPFS
    try {
      const start = Date.now();
      const ipfsResponse = await fetch(process.env.IPFS_API_URL || 'http://localhost:5001/api/v0/version', {
        method: 'POST',
        signal: AbortSignal.timeout(5000),
      });
      dependencies.ipfs = {
        status: ipfsResponse.ok ? 'healthy' : 'unhealthy',
        latency: Date.now() - start,
      };
    } catch (error: any) {
      dependencies.ipfs = {
        status: 'unhealthy',
        message: error.message,
      };
    }

    // Check blockchain RPC
    try {
      const start = Date.now();
      const rpcResponse = await fetch(process.env.RPC_URL || 'http://localhost:8545', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jsonrpc: '2.0', method: 'eth_blockNumber', params: [], id: 1 }),
        signal: AbortSignal.timeout(5000),
      });
      dependencies.blockchain = {
        status: rpcResponse.ok ? 'healthy' : 'unhealthy',
        latency: Date.now() - start,
      };
    } catch (error: any) {
      dependencies.blockchain = {
        status: 'unhealthy',
        message: error.message,
      };
    }

    const allHealthy = Object.values(dependencies).every(d => d.status === 'healthy');

    res.status(allHealthy ? 200 : 503).json({
      status: allHealthy ? 'healthy' : 'degraded',
      dependencies,
      timestamp: new Date().toISOString(),
    });
  })
);

export { router as healthRouter };
