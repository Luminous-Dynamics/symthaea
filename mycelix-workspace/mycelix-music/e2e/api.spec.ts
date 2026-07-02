// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { test, expect } from '@playwright/test';

const API_URL = process.env.API_URL || 'http://localhost:3100';

test.describe('API Health Checks', () => {
  test('GET /health returns healthy status', async ({ request }) => {
    const response = await request.get(`${API_URL}/health`);

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body.status).toBe('ok');
    expect(body.timestamp).toBeDefined();
  });

  test('GET /health/ready returns readiness status', async ({ request }) => {
    const response = await request.get(`${API_URL}/health/ready`);

    // May be 200 or 503 depending on service state
    expect([200, 503]).toContain(response.status());

    const body = await response.json();
    expect(body).toHaveProperty('db');
    expect(body).toHaveProperty('redis');
  });

  test('GET /health/live returns liveness status', async ({ request }) => {
    const response = await request.get(`${API_URL}/health/live`);

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body.alive).toBe(true);
  });
});

test.describe('Songs API', () => {
  test('GET /api/songs returns paginated list', async ({ request }) => {
    const response = await request.get(`${API_URL}/api/songs`);

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(Array.isArray(body)).toBeTruthy();
  });

  test('GET /api/songs supports pagination', async ({ request }) => {
    const response = await request.get(`${API_URL}/api/songs?limit=5&offset=0`);

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body.length).toBeLessThanOrEqual(5);
  });

  test('GET /api/songs supports filtering by genre', async ({ request }) => {
    const response = await request.get(`${API_URL}/api/songs?genre=electronic`);

    expect(response.ok()).toBeTruthy();
  });

  test('GET /api/songs/export returns CSV', async ({ request }) => {
    const response = await request.get(`${API_URL}/api/songs/export`);

    expect(response.ok()).toBeTruthy();
    expect(response.headers()['content-type']).toContain('text/csv');
  });
});

test.describe('Analytics API', () => {
  test('GET /api/analytics/top-songs returns top songs', async ({ request }) => {
    const response = await request.get(`${API_URL}/api/analytics/top-songs`);

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(Array.isArray(body)).toBeTruthy();
  });
});

test.describe('API Documentation', () => {
  test('GET /openapi.json returns OpenAPI spec', async ({ request }) => {
    const response = await request.get(`${API_URL}/openapi.json`);

    // May be disabled in some environments
    if (response.ok()) {
      const body = await response.json();
      expect(body.openapi).toBeDefined();
      expect(body.paths).toBeDefined();
    }
  });

  test('GET /swagger returns Swagger UI', async ({ request }) => {
    const response = await request.get(`${API_URL}/swagger`);

    // May be disabled in some environments
    if (response.ok()) {
      const body = await response.text();
      expect(body).toContain('swagger');
    }
  });
});

test.describe('Metrics', () => {
  test('GET /metrics returns Prometheus metrics', async ({ request }) => {
    const response = await request.get(`${API_URL}/metrics`);

    expect(response.ok()).toBeTruthy();
    expect(response.headers()['content-type']).toContain('text/plain');

    const body = await response.text();
    expect(body).toContain('process_');
  });
});
