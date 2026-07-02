// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Health Routes Integration Tests
 */

import request from 'supertest';
import express, { Express } from 'express';
import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest';
import { healthRouter, initHealthServices } from '../health.routes';

// Mock Prisma
const mockPrisma = {
  $queryRaw: vi.fn(),
};

// Mock Redis
const mockRedis = {
  ping: vi.fn(),
  info: vi.fn(),
};

describe('Health Routes', () => {
  let app: Express;

  beforeAll(() => {
    app = express();
    app.use(express.json());
    app.use('/health', healthRouter);

    // Initialize with mocks
    initHealthServices(mockPrisma as any, mockRedis as any);
  });

  afterAll(() => {
    vi.clearAllMocks();
  });

  describe('GET /health', () => {
    it('returns basic health status', async () => {
      const response = await request(app).get('/health');

      expect(response.status).toBe(200);
      expect(response.body).toEqual({
        status: 'ok',
        timestamp: expect.any(String),
      });
    });
  });

  describe('GET /health/live', () => {
    it('returns liveness probe', async () => {
      const response = await request(app).get('/health/live');

      expect(response.status).toBe(200);
      expect(response.body).toMatchObject({
        status: 'alive',
        uptime: expect.any(Number),
        timestamp: expect.any(String),
      });
    });

    it('returns positive uptime', async () => {
      const response = await request(app).get('/health/live');

      expect(response.body.uptime).toBeGreaterThan(0);
    });
  });

  describe('GET /health/ready', () => {
    it('returns ready when all critical services are healthy', async () => {
      mockPrisma.$queryRaw.mockResolvedValue([{ '?column?': 1 }]);
      mockRedis.ping.mockResolvedValue('PONG');
      mockRedis.info.mockResolvedValue('used_memory:1000000');

      const response = await request(app).get('/health/ready');

      expect(response.status).toBe(200);
      expect(response.body).toMatchObject({
        status: 'ready',
        timestamp: expect.any(String),
        checks: expect.any(Object),
      });
    });

    it('returns 503 when database is unhealthy', async () => {
      mockPrisma.$queryRaw.mockRejectedValue(new Error('Connection failed'));
      mockRedis.ping.mockResolvedValue('PONG');
      mockRedis.info.mockResolvedValue('used_memory:1000000');

      const response = await request(app).get('/health/ready');

      expect(response.status).toBe(503);
      expect(response.body.status).toBe('not_ready');
      expect(response.body.checks.database.status).toBe('unhealthy');
    });

    it('returns 503 when cache is unhealthy', async () => {
      mockPrisma.$queryRaw.mockResolvedValue([{ '?column?': 1 }]);
      mockRedis.ping.mockRejectedValue(new Error('Redis connection failed'));

      const response = await request(app).get('/health/ready');

      expect(response.status).toBe(503);
      expect(response.body.status).toBe('not_ready');
      expect(response.body.checks.cache.status).toBe('unhealthy');
    });
  });

  describe('GET /health/detailed', () => {
    it('returns detailed health status with all checks', async () => {
      mockPrisma.$queryRaw.mockResolvedValue([{ '?column?': 1 }]);
      mockRedis.ping.mockResolvedValue('PONG');
      mockRedis.info.mockResolvedValue('used_memory:1000000');

      const response = await request(app).get('/health/detailed');

      expect(response.status).toBe(200);
      expect(response.body).toMatchObject({
        status: expect.stringMatching(/healthy|degraded|unhealthy/),
        version: expect.any(String),
        uptime: expect.any(Number),
        timestamp: expect.any(String),
        checks: expect.objectContaining({
          database: expect.any(Object),
          cache: expect.any(Object),
          memory: expect.any(Object),
          eventLoop: expect.any(Object),
        }),
      });
    });

    it('includes latency for database check', async () => {
      mockPrisma.$queryRaw.mockResolvedValue([{ '?column?': 1 }]);
      mockRedis.ping.mockResolvedValue('PONG');
      mockRedis.info.mockResolvedValue('used_memory:1000000');

      const response = await request(app).get('/health/detailed');

      expect(response.body.checks.database.latency).toBeDefined();
      expect(typeof response.body.checks.database.latency).toBe('number');
    });

    it('includes memory usage details', async () => {
      mockPrisma.$queryRaw.mockResolvedValue([{ '?column?': 1 }]);
      mockRedis.ping.mockResolvedValue('PONG');
      mockRedis.info.mockResolvedValue('used_memory:1000000');

      const response = await request(app).get('/health/detailed');

      expect(response.body.checks.memory.details).toMatchObject({
        heapUsed: expect.any(Number),
        heapTotal: expect.any(Number),
        heapUsedPercent: expect.any(Number),
        rss: expect.any(Number),
      });
    });

    it('returns degraded status for non-critical unhealthy checks', async () => {
      mockPrisma.$queryRaw.mockResolvedValue([{ '?column?': 1 }]);
      mockRedis.ping.mockResolvedValue('PONG');
      mockRedis.info.mockResolvedValue('used_memory:1000000');

      // Mock high memory usage (non-critical degraded)
      const originalMemoryUsage = process.memoryUsage;
      process.memoryUsage = () => ({
        heapUsed: 900 * 1024 * 1024,
        heapTotal: 1000 * 1024 * 1024,
        external: 10 * 1024 * 1024,
        rss: 1000 * 1024 * 1024,
        arrayBuffers: 0,
      });

      const response = await request(app).get('/health/detailed');

      process.memoryUsage = originalMemoryUsage;

      // Memory is non-critical, so overall should be degraded not unhealthy
      expect(response.body.checks.memory.status).toBe('unhealthy');
    });
  });

  describe('GET /health/metrics', () => {
    it('returns Prometheus format by default', async () => {
      mockPrisma.$queryRaw.mockResolvedValue([{ '?column?': 1 }]);
      mockRedis.ping.mockResolvedValue('PONG');
      mockRedis.info.mockResolvedValue('used_memory:1000000');

      const response = await request(app).get('/health/metrics');

      expect(response.status).toBe(200);
      expect(response.headers['content-type']).toContain('text/plain');
      expect(response.text).toContain('# HELP');
      expect(response.text).toContain('# TYPE');
      expect(response.text).toContain('process_heap_bytes');
      expect(response.text).toContain('process_uptime_seconds');
    });

    it('includes health check metrics', async () => {
      mockPrisma.$queryRaw.mockResolvedValue([{ '?column?': 1 }]);
      mockRedis.ping.mockResolvedValue('PONG');
      mockRedis.info.mockResolvedValue('used_memory:1000000');

      const response = await request(app).get('/health/metrics');

      expect(response.text).toContain('health_check_status');
      expect(response.text).toContain('health_check_latency_ms');
    });
  });

  describe('GET /health/dependencies', () => {
    it('checks external dependencies', async () => {
      // Mock fetch for dependency checks
      global.fetch = vi.fn()
        .mockResolvedValueOnce({ ok: true }) // IPFS
        .mockResolvedValueOnce({ ok: true }); // Blockchain RPC

      const response = await request(app).get('/health/dependencies');

      expect(response.status).toBe(200);
      expect(response.body).toMatchObject({
        status: expect.stringMatching(/healthy|degraded/),
        dependencies: expect.any(Object),
        timestamp: expect.any(String),
      });
    });

    it('returns 503 when dependencies are unhealthy', async () => {
      global.fetch = vi.fn()
        .mockRejectedValueOnce(new Error('IPFS unavailable'))
        .mockRejectedValueOnce(new Error('RPC unavailable'));

      const response = await request(app).get('/health/dependencies');

      expect(response.status).toBe(503);
      expect(response.body.dependencies.ipfs.status).toBe('unhealthy');
      expect(response.body.dependencies.blockchain.status).toBe('unhealthy');
    });
  });
});
