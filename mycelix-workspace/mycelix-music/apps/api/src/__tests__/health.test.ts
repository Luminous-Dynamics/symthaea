// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import request from 'supertest';
import { app } from '../index';

describe('Health Endpoints', () => {
  describe('GET /health', () => {
    it('returns basic health status', async () => {
      const res = await request(app)
        .get('/health')
        .expect(200);

      expect(res.body).toHaveProperty('status');
      expect(res.body).toHaveProperty('timestamp');
    });
  });

  describe('GET /health/details', () => {
    it('returns detailed health status', async () => {
      const res = await request(app)
        .get('/health/details')
        .expect((r) => {
          // May be 200 (all healthy) or 503 (degraded)
          expect([200, 503]).toContain(r.status);
        });

      expect(res.body).toHaveProperty('status');
      expect(res.body).toHaveProperty('services');
      expect(res.body.services).toHaveProperty('database');
      expect(res.body.services).toHaveProperty('redis');
    });
  });

  describe('GET /health/ready', () => {
    it('returns readiness status', async () => {
      const res = await request(app)
        .get('/health/ready')
        .expect((r) => {
          expect([200, 503]).toContain(r.status);
        });

      expect(res.body).toHaveProperty('ready');
    });
  });

  describe('GET /health/live', () => {
    it('returns liveness status', async () => {
      const res = await request(app)
        .get('/health/live')
        .expect(200);

      expect(res.body).toHaveProperty('alive', true);
    });
  });

  describe('GET /health/startup', () => {
    it('returns startup status', async () => {
      const res = await request(app)
        .get('/health/startup')
        .expect((r) => {
          expect([200, 503]).toContain(r.status);
        });

      expect(res.body).toHaveProperty('started');
    });
  });
});

describe('Metrics Endpoint', () => {
  describe('GET /metrics', () => {
    it('returns Prometheus metrics', async () => {
      const res = await request(app)
        .get('/metrics')
        .expect(200);

      expect(res.headers['content-type']).toContain('text/plain');
      expect(res.text).toContain('process_');
    });
  });
});
