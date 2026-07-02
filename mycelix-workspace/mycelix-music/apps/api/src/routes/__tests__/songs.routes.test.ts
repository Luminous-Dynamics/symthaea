// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Songs Routes Integration Tests
 */

import request from 'supertest';
import express, { Express } from 'express';
import { describe, it, expect, beforeAll, beforeEach, afterAll, vi } from 'vitest';
import { songsRouter } from '../songs.routes';

// Mock Prisma client
const mockPrisma = {
  song: {
    findMany: vi.fn(),
    findUnique: vi.fn(),
    create: vi.fn(),
    update: vi.fn(),
    delete: vi.fn(),
    count: vi.fn(),
  },
  $transaction: vi.fn(),
};

// Mock Redis
const mockRedis = {
  get: vi.fn(),
  set: vi.fn(),
  del: vi.fn(),
  keys: vi.fn(),
};

// Mock auth middleware
vi.mock('../../middleware/auth', () => ({
  authenticate: (req: any, res: any, next: any) => {
    req.user = { id: 'user-123', role: 'artist' };
    next();
  },
  requireRole: () => (req: any, res: any, next: any) => next(),
}));

describe('Songs Routes', () => {
  let app: Express;

  const mockSong = {
    id: 'song-123',
    title: 'Test Song',
    artist: 'Test Artist',
    artistId: 'user-123',
    album: 'Test Album',
    duration: 180,
    genre: 'Electronic',
    bpm: 128,
    key: 'C Major',
    audioUrl: 'https://cdn.example.com/song.mp3',
    coverArt: 'https://cdn.example.com/cover.jpg',
    waveformData: [0.1, 0.5, 0.8, 0.3],
    isPublic: true,
    playCount: 1000,
    likes: 50,
    createdAt: new Date('2024-01-01'),
    updatedAt: new Date('2024-01-01'),
  };

  beforeAll(() => {
    app = express();
    app.use(express.json());

    // Inject mock dependencies
    app.use((req: any, res, next) => {
      req.prisma = mockPrisma;
      req.redis = mockRedis;
      next();
    });

    app.use('/songs', songsRouter);
  });

  beforeEach(() => {
    vi.clearAllMocks();
    mockRedis.get.mockResolvedValue(null); // Default: no cache
  });

  afterAll(() => {
    vi.restoreAllMocks();
  });

  describe('GET /songs', () => {
    it('returns paginated list of songs', async () => {
      mockPrisma.song.findMany.mockResolvedValue([mockSong]);
      mockPrisma.song.count.mockResolvedValue(1);

      const response = await request(app)
        .get('/songs')
        .query({ page: 1, limit: 10 });

      expect(response.status).toBe(200);
      expect(response.body).toMatchObject({
        data: expect.arrayContaining([
          expect.objectContaining({ id: 'song-123' }),
        ]),
        pagination: {
          page: 1,
          limit: 10,
          total: 1,
          totalPages: 1,
        },
      });
    });

    it('filters by genre', async () => {
      mockPrisma.song.findMany.mockResolvedValue([mockSong]);
      mockPrisma.song.count.mockResolvedValue(1);

      const response = await request(app)
        .get('/songs')
        .query({ genre: 'Electronic' });

      expect(response.status).toBe(200);
      expect(mockPrisma.song.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          where: expect.objectContaining({
            genre: 'Electronic',
          }),
        })
      );
    });

    it('searches by title', async () => {
      mockPrisma.song.findMany.mockResolvedValue([mockSong]);
      mockPrisma.song.count.mockResolvedValue(1);

      const response = await request(app)
        .get('/songs')
        .query({ search: 'Test' });

      expect(response.status).toBe(200);
      expect(mockPrisma.song.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          where: expect.objectContaining({
            OR: expect.arrayContaining([
              { title: { contains: 'Test', mode: 'insensitive' } },
              { artist: { contains: 'Test', mode: 'insensitive' } },
            ]),
          }),
        })
      );
    });

    it('returns cached results when available', async () => {
      const cachedData = JSON.stringify({
        data: [mockSong],
        pagination: { page: 1, limit: 10, total: 1, totalPages: 1 },
      });
      mockRedis.get.mockResolvedValue(cachedData);

      const response = await request(app).get('/songs');

      expect(response.status).toBe(200);
      expect(mockPrisma.song.findMany).not.toHaveBeenCalled();
    });

    it('sorts by specified field', async () => {
      mockPrisma.song.findMany.mockResolvedValue([mockSong]);
      mockPrisma.song.count.mockResolvedValue(1);

      const response = await request(app)
        .get('/songs')
        .query({ sortBy: 'playCount', sortOrder: 'desc' });

      expect(response.status).toBe(200);
      expect(mockPrisma.song.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          orderBy: { playCount: 'desc' },
        })
      );
    });
  });

  describe('GET /songs/:id', () => {
    it('returns a single song', async () => {
      mockPrisma.song.findUnique.mockResolvedValue(mockSong);

      const response = await request(app).get('/songs/song-123');

      expect(response.status).toBe(200);
      expect(response.body).toMatchObject({
        id: 'song-123',
        title: 'Test Song',
      });
    });

    it('returns 404 for non-existent song', async () => {
      mockPrisma.song.findUnique.mockResolvedValue(null);

      const response = await request(app).get('/songs/non-existent');

      expect(response.status).toBe(404);
      expect(response.body).toMatchObject({
        error: 'Song not found',
      });
    });

    it('returns cached song when available', async () => {
      mockRedis.get.mockResolvedValue(JSON.stringify(mockSong));

      const response = await request(app).get('/songs/song-123');

      expect(response.status).toBe(200);
      expect(mockPrisma.song.findUnique).not.toHaveBeenCalled();
    });
  });

  describe('POST /songs', () => {
    const newSongData = {
      title: 'New Song',
      album: 'New Album',
      duration: 200,
      genre: 'House',
      bpm: 125,
      key: 'A Minor',
      audioUrl: 'https://cdn.example.com/new-song.mp3',
    };

    it('creates a new song', async () => {
      mockPrisma.song.create.mockResolvedValue({
        id: 'new-song-123',
        ...newSongData,
        artistId: 'user-123',
      });

      const response = await request(app)
        .post('/songs')
        .send(newSongData);

      expect(response.status).toBe(201);
      expect(response.body).toMatchObject({
        id: 'new-song-123',
        title: 'New Song',
      });
    });

    it('validates required fields', async () => {
      const response = await request(app)
        .post('/songs')
        .send({ title: 'Missing Fields' });

      expect(response.status).toBe(400);
      expect(response.body.error).toBeDefined();
    });

    it('validates BPM range', async () => {
      const response = await request(app)
        .post('/songs')
        .send({ ...newSongData, bpm: 500 });

      expect(response.status).toBe(400);
    });

    it('invalidates cache after creation', async () => {
      mockPrisma.song.create.mockResolvedValue({
        id: 'new-song-123',
        ...newSongData,
      });
      mockRedis.keys.mockResolvedValue(['songs:*']);

      await request(app).post('/songs').send(newSongData);

      expect(mockRedis.del).toHaveBeenCalled();
    });
  });

  describe('PUT /songs/:id', () => {
    it('updates an existing song', async () => {
      mockPrisma.song.findUnique.mockResolvedValue(mockSong);
      mockPrisma.song.update.mockResolvedValue({
        ...mockSong,
        title: 'Updated Title',
      });

      const response = await request(app)
        .put('/songs/song-123')
        .send({ title: 'Updated Title' });

      expect(response.status).toBe(200);
      expect(response.body.title).toBe('Updated Title');
    });

    it('returns 404 for non-existent song', async () => {
      mockPrisma.song.findUnique.mockResolvedValue(null);

      const response = await request(app)
        .put('/songs/non-existent')
        .send({ title: 'Updated' });

      expect(response.status).toBe(404);
    });

    it('returns 403 when user does not own the song', async () => {
      mockPrisma.song.findUnique.mockResolvedValue({
        ...mockSong,
        artistId: 'other-user',
      });

      const response = await request(app)
        .put('/songs/song-123')
        .send({ title: 'Updated' });

      expect(response.status).toBe(403);
    });

    it('invalidates cache after update', async () => {
      mockPrisma.song.findUnique.mockResolvedValue(mockSong);
      mockPrisma.song.update.mockResolvedValue(mockSong);

      await request(app)
        .put('/songs/song-123')
        .send({ title: 'Updated' });

      expect(mockRedis.del).toHaveBeenCalledWith('song:song-123');
    });
  });

  describe('DELETE /songs/:id', () => {
    it('deletes a song', async () => {
      mockPrisma.song.findUnique.mockResolvedValue(mockSong);
      mockPrisma.song.delete.mockResolvedValue(mockSong);

      const response = await request(app).delete('/songs/song-123');

      expect(response.status).toBe(204);
    });

    it('returns 404 for non-existent song', async () => {
      mockPrisma.song.findUnique.mockResolvedValue(null);

      const response = await request(app).delete('/songs/non-existent');

      expect(response.status).toBe(404);
    });

    it('returns 403 when user does not own the song', async () => {
      mockPrisma.song.findUnique.mockResolvedValue({
        ...mockSong,
        artistId: 'other-user',
      });

      const response = await request(app).delete('/songs/song-123');

      expect(response.status).toBe(403);
    });
  });

  describe('POST /songs/:id/play', () => {
    it('increments play count', async () => {
      mockPrisma.song.findUnique.mockResolvedValue(mockSong);
      mockPrisma.song.update.mockResolvedValue({
        ...mockSong,
        playCount: 1001,
      });

      const response = await request(app).post('/songs/song-123/play');

      expect(response.status).toBe(200);
      expect(response.body.playCount).toBe(1001);
    });

    it('records play analytics', async () => {
      mockPrisma.song.findUnique.mockResolvedValue(mockSong);
      mockPrisma.song.update.mockResolvedValue(mockSong);

      await request(app)
        .post('/songs/song-123/play')
        .send({ duration: 120, source: 'web' });

      // Verify analytics were recorded
      expect(mockPrisma.song.update).toHaveBeenCalled();
    });
  });

  describe('POST /songs/:id/like', () => {
    it('toggles like on a song', async () => {
      mockPrisma.song.findUnique.mockResolvedValue(mockSong);
      mockPrisma.$transaction.mockResolvedValue([
        { userId: 'user-123', songId: 'song-123' },
        { ...mockSong, likes: 51 },
      ]);

      const response = await request(app).post('/songs/song-123/like');

      expect(response.status).toBe(200);
      expect(response.body).toMatchObject({
        liked: true,
        likes: 51,
      });
    });
  });

  describe('GET /songs/:id/similar', () => {
    it('returns similar songs based on attributes', async () => {
      mockPrisma.song.findUnique.mockResolvedValue(mockSong);
      mockPrisma.song.findMany.mockResolvedValue([
        { ...mockSong, id: 'similar-1', title: 'Similar Song 1' },
        { ...mockSong, id: 'similar-2', title: 'Similar Song 2' },
      ]);

      const response = await request(app).get('/songs/song-123/similar');

      expect(response.status).toBe(200);
      expect(response.body).toHaveLength(2);
      expect(response.body[0].id).not.toBe('song-123');
    });

    it('filters similar songs by genre and BPM range', async () => {
      mockPrisma.song.findUnique.mockResolvedValue(mockSong);
      mockPrisma.song.findMany.mockResolvedValue([]);

      await request(app).get('/songs/song-123/similar');

      expect(mockPrisma.song.findMany).toHaveBeenCalledWith(
        expect.objectContaining({
          where: expect.objectContaining({
            genre: mockSong.genre,
            bpm: expect.objectContaining({
              gte: expect.any(Number),
              lte: expect.any(Number),
            }),
          }),
        })
      );
    });
  });
});
