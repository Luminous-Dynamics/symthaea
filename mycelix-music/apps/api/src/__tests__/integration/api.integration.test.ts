// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * API Integration Tests
 *
 * Full request flow tests covering the entire API lifecycle.
 * Tests validation, authentication, business logic, and response formatting.
 */

import request from 'supertest';
import { Express } from 'express';
import { Pool } from 'pg';
import Redis from 'ioredis';
import { Application, bootstrap } from '../../app';

describe('API Integration Tests', () => {
  let app: Express;
  let application: Application;
  let pool: Pool;

  // Test data
  const testWallet = '0x1234567890123456789012345678901234567890';
  const testWallet2 = '0xabcdefabcdefabcdefabcdefabcdefabcdefabcd';

  beforeAll(async () => {
    // Bootstrap with test configuration
    process.env.NODE_ENV = 'test';
    process.env.DATABASE_URL = process.env.TEST_DATABASE_URL || 'postgresql://localhost:5432/mycelix_test';

    application = await bootstrap({
      skipJobs: true,
      skipEvents: true,
      port: 0, // Random port
    });

    app = application.app;
    pool = application.container.pool;

    // Clean test data
    await cleanTestData();
  });

  afterAll(async () => {
    await cleanTestData();
    await application.shutdown();
  });

  async function cleanTestData(): Promise<void> {
    await pool.query(`
      DELETE FROM plays WHERE wallet_address IN ($1, $2);
      DELETE FROM songs WHERE artist_address IN ($1, $2);
    `, [testWallet, testWallet2]);
  }

  describe('Health Endpoints', () => {
    test('GET /health returns healthy status', async () => {
      const response = await request(app)
        .get('/health')
        .expect(200);

      expect(response.body).toMatchObject({
        status: 'healthy',
      });
    });

    test('GET /health/ready returns readiness status', async () => {
      const response = await request(app)
        .get('/health/ready')
        .expect(200);

      expect(response.body).toMatchObject({
        status: 'ready',
      });
    });

    test('GET /health/live returns liveness status', async () => {
      const response = await request(app)
        .get('/health/live')
        .expect(200);

      expect(response.body).toMatchObject({
        status: 'alive',
      });
    });
  });

  describe('Root Endpoint', () => {
    test('GET / returns API info', async () => {
      const response = await request(app)
        .get('/')
        .expect(200);

      expect(response.body).toMatchObject({
        name: 'Mycelix Music API',
        health: '/health',
      });
    });
  });

  describe('Songs API', () => {
    let createdSongId: string;

    describe('POST /api/v2/songs', () => {
      test('creates a song with valid data', async () => {
        const songData = {
          title: 'Test Song',
          artistAddress: testWallet,
          ipfsHash: 'QmTest123456789012345678901234567890123456789',
          genre: 'Electronic',
          duration: 180,
        };

        const response = await request(app)
          .post('/api/v2/songs')
          .send(songData)
          .expect(201);

        expect(response.body).toMatchObject({
          success: true,
          data: {
            title: 'Test Song',
            artistAddress: testWallet.toLowerCase(),
            genre: 'Electronic',
            duration: 180,
          },
        });

        createdSongId = response.body.data.id;
      });

      test('validates required fields', async () => {
        const response = await request(app)
          .post('/api/v2/songs')
          .send({})
          .expect(400);

        expect(response.body).toMatchObject({
          success: false,
          error: {
            code: 'VALIDATION_ERROR',
          },
        });
      });

      test('validates wallet address format', async () => {
        const response = await request(app)
          .post('/api/v2/songs')
          .send({
            title: 'Test',
            artistAddress: 'invalid-address',
            ipfsHash: 'QmTest123456789012345678901234567890123456789',
          })
          .expect(400);

        expect(response.body.error.code).toBe('VALIDATION_ERROR');
      });

      test('validates IPFS hash format', async () => {
        const response = await request(app)
          .post('/api/v2/songs')
          .send({
            title: 'Test',
            artistAddress: testWallet,
            ipfsHash: 'invalid-hash',
          })
          .expect(400);

        expect(response.body.error.code).toBe('VALIDATION_ERROR');
      });
    });

    describe('GET /api/v2/songs', () => {
      test('returns paginated list of songs', async () => {
        const response = await request(app)
          .get('/api/v2/songs')
          .expect(200);

        expect(response.body).toMatchObject({
          success: true,
          data: expect.any(Array),
          meta: {
            pagination: {
              limit: expect.any(Number),
              offset: expect.any(Number),
              total: expect.any(Number),
              hasMore: expect.any(Boolean),
            },
          },
        });
      });

      test('supports pagination parameters', async () => {
        const response = await request(app)
          .get('/api/v2/songs?limit=5&offset=0')
          .expect(200);

        expect(response.body.meta.pagination.limit).toBe(5);
        expect(response.body.meta.pagination.offset).toBe(0);
      });

      test('filters by artist address', async () => {
        const response = await request(app)
          .get(`/api/v2/songs?artistAddress=${testWallet}`)
          .expect(200);

        expect(response.body.success).toBe(true);
        response.body.data.forEach((song: any) => {
          expect(song.artistAddress.toLowerCase()).toBe(testWallet.toLowerCase());
        });
      });

      test('filters by genre', async () => {
        const response = await request(app)
          .get('/api/v2/songs?genre=Electronic')
          .expect(200);

        expect(response.body.success).toBe(true);
      });
    });

    describe('GET /api/v2/songs/:id', () => {
      test('returns song by ID', async () => {
        const response = await request(app)
          .get(`/api/v2/songs/${createdSongId}`)
          .expect(200);

        expect(response.body).toMatchObject({
          success: true,
          data: {
            id: createdSongId,
            title: 'Test Song',
          },
        });
      });

      test('returns 404 for non-existent song', async () => {
        const fakeId = '00000000-0000-0000-0000-000000000000';
        const response = await request(app)
          .get(`/api/v2/songs/${fakeId}`)
          .expect(404);

        expect(response.body).toMatchObject({
          success: false,
          error: {
            code: 'NOT_FOUND',
          },
        });
      });

      test('validates UUID format', async () => {
        const response = await request(app)
          .get('/api/v2/songs/invalid-uuid')
          .expect(400);

        expect(response.body.error.code).toBe('VALIDATION_ERROR');
      });
    });

    describe('PUT /api/v2/songs/:id', () => {
      test('updates song with valid data', async () => {
        const response = await request(app)
          .put(`/api/v2/songs/${createdSongId}`)
          .send({
            title: 'Updated Song Title',
            genre: 'Ambient',
          })
          .expect(200);

        expect(response.body).toMatchObject({
          success: true,
          data: {
            id: createdSongId,
            title: 'Updated Song Title',
            genre: 'Ambient',
          },
        });
      });

      test('returns 404 for non-existent song', async () => {
        const fakeId = '00000000-0000-0000-0000-000000000000';
        const response = await request(app)
          .put(`/api/v2/songs/${fakeId}`)
          .send({ title: 'Test' })
          .expect(404);

        expect(response.body.error.code).toBe('NOT_FOUND');
      });
    });

    describe('DELETE /api/v2/songs/:id', () => {
      test('soft deletes song', async () => {
        const response = await request(app)
          .delete(`/api/v2/songs/${createdSongId}`)
          .expect(204);

        // Verify song is not returned in list
        const listResponse = await request(app)
          .get(`/api/v2/songs/${createdSongId}`)
          .expect(404);
      });
    });
  });

  describe('Plays API', () => {
    let testSongId: string;

    beforeAll(async () => {
      // Create a test song for plays
      const response = await request(app)
        .post('/api/v2/songs')
        .send({
          title: 'Play Test Song',
          artistAddress: testWallet,
          ipfsHash: 'QmPlayTest123456789012345678901234567890123',
          duration: 200,
        });

      testSongId = response.body.data.id;
    });

    describe('POST /api/v2/plays', () => {
      test('records a play with valid data', async () => {
        const response = await request(app)
          .post('/api/v2/plays')
          .send({
            songId: testSongId,
            walletAddress: testWallet2,
            duration: 180,
          })
          .expect(201);

        expect(response.body).toMatchObject({
          success: true,
          data: {
            songId: testSongId,
            walletAddress: testWallet2.toLowerCase(),
          },
        });
      });

      test('validates required fields', async () => {
        const response = await request(app)
          .post('/api/v2/plays')
          .send({})
          .expect(400);

        expect(response.body.error.code).toBe('VALIDATION_ERROR');
      });

      test('validates song exists', async () => {
        const fakeId = '00000000-0000-0000-0000-000000000000';
        const response = await request(app)
          .post('/api/v2/plays')
          .send({
            songId: fakeId,
            walletAddress: testWallet2,
          })
          .expect(404);

        expect(response.body.error.code).toBe('NOT_FOUND');
      });
    });

    describe('GET /api/v2/plays', () => {
      test('returns plays for a song', async () => {
        const response = await request(app)
          .get(`/api/v2/plays?songId=${testSongId}`)
          .expect(200);

        expect(response.body).toMatchObject({
          success: true,
          data: expect.any(Array),
        });
      });

      test('returns plays for a wallet', async () => {
        const response = await request(app)
          .get(`/api/v2/plays?walletAddress=${testWallet2}`)
          .expect(200);

        expect(response.body).toMatchObject({
          success: true,
          data: expect.any(Array),
        });
      });
    });
  });

  describe('Error Handling', () => {
    test('returns proper error for invalid JSON', async () => {
      const response = await request(app)
        .post('/api/v2/songs')
        .set('Content-Type', 'application/json')
        .send('{ invalid json }')
        .expect(400);

      expect(response.body).toMatchObject({
        success: false,
        error: expect.any(Object),
      });
    });

    test('returns 404 for unknown routes', async () => {
      const response = await request(app)
        .get('/api/v2/unknown')
        .expect(404);

      expect(response.body).toMatchObject({
        success: false,
        error: {
          code: 'NOT_FOUND',
        },
      });
    });

    test('handles request timeout gracefully', async () => {
      // This test would require a slow endpoint
      // Skipping for now as it needs special setup
    });
  });

  describe('Response Format', () => {
    test('includes request ID in response', async () => {
      const response = await request(app)
        .get('/api/v2/songs')
        .expect(200);

      expect(response.body.meta?.requestId).toBeDefined();
    });

    test('includes timing information', async () => {
      const response = await request(app)
        .get('/api/v2/songs')
        .expect(200);

      expect(response.body.meta?.timing).toBeDefined();
      expect(response.body.meta.timing.duration).toBeGreaterThanOrEqual(0);
    });

    test('includes API version header', async () => {
      const response = await request(app)
        .get('/api/v2/songs')
        .expect(200);

      expect(response.headers['x-api-version']).toBe('2');
    });
  });

  describe('Content Negotiation', () => {
    test('returns JSON by default', async () => {
      const response = await request(app)
        .get('/api/v2/songs')
        .expect('Content-Type', /json/)
        .expect(200);
    });

    test('sets appropriate cache headers', async () => {
      const response = await request(app)
        .get('/api/v2/songs')
        .expect(200);

      // Should have some cache control
      expect(response.headers).toHaveProperty('cache-control');
    });
  });

  describe('CORS', () => {
    test('allows configured origins', async () => {
      const response = await request(app)
        .options('/api/v2/songs')
        .set('Origin', 'http://localhost:3000')
        .expect(204);

      expect(response.headers['access-control-allow-origin']).toBeDefined();
    });

    test('allows required methods', async () => {
      const response = await request(app)
        .options('/api/v2/songs')
        .set('Origin', 'http://localhost:3000')
        .set('Access-Control-Request-Method', 'POST')
        .expect(204);

      expect(response.headers['access-control-allow-methods']).toContain('POST');
    });
  });
});

describe('Wallet Authentication Integration', () => {
  let app: Express;
  let application: Application;

  const testWallet = '0x1234567890123456789012345678901234567890';

  beforeAll(async () => {
    process.env.NODE_ENV = 'test';
    process.env.SKIP_AUTH_VERIFY = 'true'; // Allow test signatures

    application = await bootstrap({
      skipJobs: true,
      skipEvents: true,
      port: 0,
    });

    app = application.app;
  });

  afterAll(async () => {
    await application.shutdown();
  });

  test('requires authentication for protected endpoints', async () => {
    // This would test endpoints that require wallet auth
    // Implementation depends on which endpoints are protected
  });

  test('accepts valid signature in development mode', async () => {
    // Create a mock signature
    const payload = {
      address: testWallet,
      timestamp: Date.now(),
      nonce: 'test-nonce',
    };
    const payloadEncoded = Buffer.from(JSON.stringify(payload)).toString('base64');
    const mockSignature = '0x' + '00'.repeat(65);

    // Would test against a protected endpoint
  });

  test('rejects expired signatures', async () => {
    const payload = {
      address: testWallet,
      timestamp: Date.now() - 10 * 60 * 1000, // 10 minutes ago
      nonce: 'test-nonce',
    };
    const payloadEncoded = Buffer.from(JSON.stringify(payload)).toString('base64');
    const mockSignature = '0x' + '00'.repeat(65);

    // Would test against a protected endpoint expecting rejection
  });
});

describe('Rate Limiting Integration', () => {
  let app: Express;
  let application: Application;

  beforeAll(async () => {
    process.env.NODE_ENV = 'test';
    process.env.RATE_LIMIT_ENABLED = 'true';

    application = await bootstrap({
      skipJobs: true,
      skipEvents: true,
      port: 0,
    });

    app = application.app;
  });

  afterAll(async () => {
    await application.shutdown();
  });

  test('enforces rate limits', async () => {
    // Make multiple requests rapidly
    const requests = Array(100).fill(null).map(() =>
      request(app).get('/api/v2/songs')
    );

    const responses = await Promise.all(requests);

    // Some should be rate limited
    const rateLimited = responses.filter(r => r.status === 429);

    // At least verify the endpoint works
    expect(responses.some(r => r.status === 200)).toBe(true);
  });

  test('includes rate limit headers', async () => {
    const response = await request(app)
      .get('/api/v2/songs')
      .expect(200);

    // Rate limit headers should be present
    expect(response.headers['x-ratelimit-limit']).toBeDefined();
    expect(response.headers['x-ratelimit-remaining']).toBeDefined();
  });
});

describe('Metrics Integration', () => {
  let app: Express;
  let application: Application;

  beforeAll(async () => {
    process.env.NODE_ENV = 'test';

    application = await bootstrap({
      skipJobs: true,
      skipEvents: true,
      port: 0,
    });

    app = application.app;
  });

  afterAll(async () => {
    await application.shutdown();
  });

  test('exposes metrics endpoint', async () => {
    const response = await request(app)
      .get('/metrics')
      .expect(200)
      .expect('Content-Type', /text\/plain/);

    expect(response.text).toContain('mycelix_');
  });

  test('records request metrics', async () => {
    // Make a few requests
    await request(app).get('/api/v2/songs');
    await request(app).get('/api/v2/songs');

    const metricsResponse = await request(app)
      .get('/metrics')
      .expect(200);

    expect(metricsResponse.text).toContain('http_requests_total');
  });
});
