// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import request from 'supertest';
import { app } from '../index';

describe('Analytics API', () => {
  describe('GET /api/analytics/artist/:address', () => {
    it('returns artist analytics', async () => {
      const testAddress = '0x1234567890abcdef1234567890abcdef12345678';
      const res = await request(app)
        .get(`/api/analytics/artist/${testAddress}`)
        .expect((r) => {
          // May return 200 with data or 404 if no data
          expect([200, 404]).toContain(r.status);
        });

      if (res.status === 200) {
        expect(res.body).toHaveProperty('totalPlays');
        expect(res.body).toHaveProperty('totalEarnings');
      }
    });

    it('validates address format', async () => {
      const invalidAddress = 'not-an-address';
      const res = await request(app)
        .get(`/api/analytics/artist/${invalidAddress}`)
        .expect(400);

      expect(res.body).toHaveProperty('error');
    });

    it('supports date range filtering', async () => {
      const testAddress = '0x1234567890abcdef1234567890abcdef12345678';
      const startDate = '2024-01-01';
      const endDate = '2024-12-31';

      const res = await request(app)
        .get(`/api/analytics/artist/${testAddress}?startDate=${startDate}&endDate=${endDate}`)
        .expect((r) => {
          expect([200, 404]).toContain(r.status);
        });
    });
  });

  describe('GET /api/analytics/song/:id', () => {
    it('returns song analytics', async () => {
      // Get a song ID first
      const listRes = await request(app).get('/api/songs?pageSize=1');

      if (listRes.body.songs?.length > 0) {
        const songId = listRes.body.songs[0].id;
        const res = await request(app)
          .get(`/api/analytics/song/${songId}`)
          .expect((r) => {
            expect([200, 404]).toContain(r.status);
          });

        if (res.status === 200) {
          expect(res.body).toHaveProperty('totalPlays');
        }
      }
    });
  });

  describe('GET /api/analytics/platform', () => {
    it('returns platform-wide analytics', async () => {
      const res = await request(app)
        .get('/api/analytics/platform')
        .expect(200);

      expect(res.body).toHaveProperty('totalSongs');
      expect(res.body).toHaveProperty('totalArtists');
      expect(res.body).toHaveProperty('totalPlays');
    });
  });

  describe('CSV Export', () => {
    it('exports analytics as CSV', async () => {
      const testAddress = '0x1234567890abcdef1234567890abcdef12345678';
      const res = await request(app)
        .get(`/api/analytics/artist/${testAddress}/export`)
        .expect((r) => {
          // May return 200 with CSV or 404 if no data
          expect([200, 404]).toContain(r.status);
        });

      if (res.status === 200) {
        expect(res.headers['content-type']).toContain('text/csv');
      }
    });
  });
});

describe('Rate Limiting', () => {
  it('applies rate limits to analytics endpoints', async () => {
    const testAddress = '0x1234567890abcdef1234567890abcdef12345678';
    const requests = Array(10).fill(null).map(() =>
      request(app).get(`/api/analytics/artist/${testAddress}`)
    );

    const responses = await Promise.all(requests);

    // At least some should succeed
    const successCount = responses.filter(r => r.status === 200 || r.status === 404).length;
    expect(successCount).toBeGreaterThan(0);

    // If rate limited, should return 429
    const rateLimited = responses.filter(r => r.status === 429);
    if (rateLimited.length > 0) {
      expect(rateLimited[0].body).toHaveProperty('error');
    }
  });
});
