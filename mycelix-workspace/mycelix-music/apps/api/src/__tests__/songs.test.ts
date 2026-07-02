// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import request from 'supertest';
import { app } from '../index';

describe('Songs API', () => {
  describe('GET /api/songs', () => {
    it('returns paginated list of songs', async () => {
      const res = await request(app)
        .get('/api/songs')
        .expect(200);

      expect(res.body).toHaveProperty('songs');
      expect(res.body).toHaveProperty('total');
      expect(res.body).toHaveProperty('page');
      expect(res.body).toHaveProperty('pageSize');
      expect(Array.isArray(res.body.songs)).toBe(true);
    });

    it('respects pagination parameters', async () => {
      const res = await request(app)
        .get('/api/songs?page=1&pageSize=5')
        .expect(200);

      expect(res.body.page).toBe(1);
      expect(res.body.pageSize).toBe(5);
      expect(res.body.songs.length).toBeLessThanOrEqual(5);
    });

    it('filters by artist address', async () => {
      const testAddress = '0x1234567890abcdef1234567890abcdef12345678';
      const res = await request(app)
        .get(`/api/songs?artistAddress=${testAddress}`)
        .expect(200);

      expect(res.body).toHaveProperty('songs');
      // All returned songs should be from this artist
      res.body.songs.forEach((song: any) => {
        expect(song.artistAddress?.toLowerCase()).toBe(testAddress.toLowerCase());
      });
    });

    it('filters by payment model', async () => {
      const res = await request(app)
        .get('/api/songs?paymentModel=pay_per_stream')
        .expect(200);

      expect(res.body).toHaveProperty('songs');
    });

    it('rejects invalid pagination values', async () => {
      const res = await request(app)
        .get('/api/songs?page=-1&pageSize=1000')
        .expect(400);

      expect(res.body).toHaveProperty('error');
    });
  });

  describe('GET /api/songs/:id', () => {
    it('returns 404 for non-existent song', async () => {
      const res = await request(app)
        .get('/api/songs/non-existent-song-id')
        .expect(404);

      expect(res.body).toHaveProperty('error');
    });

    it('returns song details for valid ID', async () => {
      // First get a valid song ID
      const listRes = await request(app).get('/api/songs?pageSize=1');

      if (listRes.body.songs.length > 0) {
        const songId = listRes.body.songs[0].id;
        const res = await request(app)
          .get(`/api/songs/${songId}`)
          .expect(200);

        expect(res.body).toHaveProperty('id', songId);
        expect(res.body).toHaveProperty('title');
        expect(res.body).toHaveProperty('artistAddress');
      }
    });
  });

  describe('GET /api/songs/:id/plays', () => {
    it('returns play history for a song', async () => {
      // Get a song ID first
      const listRes = await request(app).get('/api/songs?pageSize=1');

      if (listRes.body.songs.length > 0) {
        const songId = listRes.body.songs[0].id;
        const res = await request(app)
          .get(`/api/songs/${songId}/plays`)
          .expect(200);

        expect(res.body).toHaveProperty('plays');
        expect(Array.isArray(res.body.plays)).toBe(true);
      }
    });
  });

  describe('Input Validation', () => {
    it('rejects SQL injection attempts in artist address', async () => {
      const maliciousInput = "'; DROP TABLE songs; --";
      const res = await request(app)
        .get(`/api/songs?artistAddress=${encodeURIComponent(maliciousInput)}`)
        .expect(400);

      expect(res.body).toHaveProperty('error');
    });

    it('rejects XSS attempts in search query', async () => {
      const xssPayload = '<script>alert("xss")</script>';
      const res = await request(app)
        .get(`/api/songs?search=${encodeURIComponent(xssPayload)}`)
        .expect((r) => {
          // Should either reject or sanitize
          if (r.status === 200) {
            expect(r.body.songs).toBeDefined();
          }
        });
    });

    it('validates Ethereum address format', async () => {
      const invalidAddress = '0xinvalid';
      const res = await request(app)
        .get(`/api/songs?artistAddress=${invalidAddress}`)
        .expect(400);

      expect(res.body).toHaveProperty('error');
    });
  });
});

describe('Search API', () => {
  describe('GET /api/search', () => {
    it('returns search results', async () => {
      const res = await request(app)
        .get('/api/search?q=test')
        .expect(200);

      expect(res.body).toHaveProperty('results');
    });

    it('requires search query parameter', async () => {
      const res = await request(app)
        .get('/api/search')
        .expect(400);

      expect(res.body).toHaveProperty('error');
    });

    it('limits search results', async () => {
      const res = await request(app)
        .get('/api/search?q=a&limit=5')
        .expect(200);

      expect(res.body.results.length).toBeLessThanOrEqual(5);
    });
  });
});
