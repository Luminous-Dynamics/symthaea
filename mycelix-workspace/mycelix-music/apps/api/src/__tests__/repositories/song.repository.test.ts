// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Song Repository Tests
 */

import { Pool } from 'pg';
import { SongRepository } from '../../repositories/song.repository';
import { FixedIds, SongFixtures, Factories } from '../../db/seeds/fixtures';

// Mock pg Pool
jest.mock('pg', () => {
  const mockClient = {
    query: jest.fn(),
    release: jest.fn(),
  };

  const mockPool = {
    query: jest.fn(),
    connect: jest.fn().mockResolvedValue(mockClient),
  };

  return { Pool: jest.fn(() => mockPool) };
});

describe('SongRepository', () => {
  let pool: jest.Mocked<Pool>;
  let repository: SongRepository;

  beforeEach(() => {
    pool = new Pool() as jest.Mocked<Pool>;
    repository = new SongRepository(pool);
    jest.clearAllMocks();
  });

  describe('findById', () => {
    it('should return a song when found', async () => {
      const mockSong = SongFixtures.validSong;
      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: [mockSong],
        rowCount: 1,
      });

      const result = await repository.findById(mockSong.id);

      expect(result).toEqual(mockSong);
      expect(pool.query).toHaveBeenCalledWith(
        expect.stringContaining('SELECT * FROM songs WHERE id = $1'),
        [mockSong.id]
      );
    });

    it('should return null when song not found', async () => {
      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: [],
        rowCount: 0,
      });

      const result = await repository.findById('non-existent-id');

      expect(result).toBeNull();
    });
  });

  describe('findByIds', () => {
    it('should return empty array for empty input', async () => {
      const result = await repository.findByIds([]);

      expect(result).toEqual([]);
      expect(pool.query).not.toHaveBeenCalled();
    });

    it('should return songs for given IDs', async () => {
      const mockSongs = [SongFixtures.validSong, SongFixtures.subscriptionSong];
      const ids = mockSongs.map(s => s.id);

      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: mockSongs,
        rowCount: mockSongs.length,
      });

      const result = await repository.findByIds(ids);

      expect(result).toEqual(mockSongs);
      expect(pool.query).toHaveBeenCalledWith(
        expect.stringContaining('WHERE id IN'),
        ids
      );
    });
  });

  describe('findByArtist', () => {
    it('should return songs for an artist', async () => {
      const artistAddress = FixedIds.artists.artist1;
      const mockSongs = [SongFixtures.validSong, SongFixtures.popularSong];

      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: mockSongs,
        rowCount: mockSongs.length,
      });

      const result = await repository.findByArtist(artistAddress);

      expect(result).toEqual(mockSongs);
      expect(pool.query).toHaveBeenCalledWith(
        expect.stringContaining('WHERE artist_address = $1'),
        [artistAddress]
      );
    });
  });

  describe('findByGenre', () => {
    it('should return songs for a genre with limit', async () => {
      const mockSongs = [SongFixtures.validSong];

      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: mockSongs,
        rowCount: 1,
      });

      const result = await repository.findByGenre('electronic', 10);

      expect(result).toEqual(mockSongs);
      expect(pool.query).toHaveBeenCalledWith(
        expect.stringContaining('WHERE genre = $1'),
        ['electronic', 10]
      );
    });
  });

  describe('search', () => {
    it('should search with query string', async () => {
      const mockSongs = [SongFixtures.validSong];

      // Mock count query
      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: [{ count: '1' }],
        rowCount: 1,
      });

      // Mock data query
      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: mockSongs,
        rowCount: 1,
      });

      const result = await repository.search({ query: 'test' }, 20, 0);

      expect(result.data).toEqual(mockSongs);
      expect(result.total).toBe(1);
      expect(result.hasMore).toBe(false);
    });

    it('should filter by genre', async () => {
      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: [{ count: '5' }],
        rowCount: 1,
      });

      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: [],
        rowCount: 0,
      });

      await repository.search({ genre: 'electronic' }, 20, 0);

      expect(pool.query).toHaveBeenCalledWith(
        expect.stringContaining('genre = $'),
        expect.arrayContaining(['electronic'])
      );
    });

    it('should handle pagination', async () => {
      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: [{ count: '100' }],
        rowCount: 1,
      });

      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: Array(20).fill(SongFixtures.validSong),
        rowCount: 20,
      });

      const result = await repository.search({}, 20, 40);

      expect(result.total).toBe(100);
      expect(result.offset).toBe(40);
      expect(result.hasMore).toBe(true);
    });
  });

  describe('getTopByPlays', () => {
    it('should return top songs ordered by plays', async () => {
      const mockSongs = [SongFixtures.popularSong, SongFixtures.validSong];

      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: mockSongs,
        rowCount: 2,
      });

      const result = await repository.getTopByPlays(10);

      expect(result).toEqual(mockSongs);
      expect(pool.query).toHaveBeenCalledWith(
        expect.stringContaining('ORDER BY plays DESC'),
        [10]
      );
    });
  });

  describe('create', () => {
    it('should create a new song', async () => {
      const input = {
        title: 'New Song',
        artist: 'New Artist',
        artist_address: FixedIds.artists.artist1,
        genre: 'electronic',
        ipfs_hash: 'QmTest123',
        payment_model: 'per_play' as const,
      };

      const mockCreated = { ...input, id: 'new-id', plays: 0, earnings: '0' };

      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: [mockCreated],
        rowCount: 1,
      });

      const result = await repository.create(input);

      expect(result).toEqual(mockCreated);
      expect(pool.query).toHaveBeenCalledWith(
        expect.stringContaining('INSERT INTO songs'),
        expect.arrayContaining([input.title, input.artist])
      );
    });
  });

  describe('update', () => {
    it('should update a song', async () => {
      const id = FixedIds.songs.song1;
      const updates = { title: 'Updated Title' };
      const mockUpdated = { ...SongFixtures.validSong, ...updates };

      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: [mockUpdated],
        rowCount: 1,
      });

      const result = await repository.update(id, updates);

      expect(result).toEqual(mockUpdated);
      expect(pool.query).toHaveBeenCalledWith(
        expect.stringContaining('UPDATE songs'),
        expect.arrayContaining(['Updated Title', id])
      );
    });

    it('should return existing song if no updates provided', async () => {
      const id = FixedIds.songs.song1;
      const mockSong = SongFixtures.validSong;

      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: [mockSong],
        rowCount: 1,
      });

      const result = await repository.update(id, {});

      expect(result).toEqual(mockSong);
    });
  });

  describe('delete', () => {
    it('should delete a song and return true', async () => {
      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: [],
        rowCount: 1,
      });

      const result = await repository.delete(FixedIds.songs.song1);

      expect(result).toBe(true);
    });

    it('should return false if song not found', async () => {
      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: [],
        rowCount: 0,
      });

      const result = await repository.delete('non-existent');

      expect(result).toBe(false);
    });
  });

  describe('recordPlay', () => {
    it('should increment plays and earnings', async () => {
      const mockUpdated = {
        ...SongFixtures.validSong,
        plays: 101,
        earnings: '0.101000',
      };

      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: [mockUpdated],
        rowCount: 1,
      });

      const result = await repository.recordPlay(FixedIds.songs.song1, '0.001000');

      expect(result).toEqual(mockUpdated);
      expect(pool.query).toHaveBeenCalledWith(
        expect.stringContaining('plays = plays + 1'),
        [FixedIds.songs.song1, '0.001000']
      );
    });
  });

  describe('getArtistStats', () => {
    it('should return aggregated artist statistics', async () => {
      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: [{
          total_songs: '5',
          total_plays: '1000',
          total_earnings: '10.000000',
        }],
        rowCount: 1,
      });

      const result = await repository.getArtistStats(FixedIds.artists.artist1);

      expect(result).toEqual({
        total_songs: 5,
        total_plays: 1000,
        total_earnings: '10.000000',
      });
    });

    it('should return null for artist with no songs', async () => {
      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: [{ total_songs: '0', total_plays: '0', total_earnings: '0' }],
        rowCount: 1,
      });

      const result = await repository.getArtistStats('0x0000000000000000000000000000000000000000');

      expect(result).toBeNull();
    });
  });

  describe('getGenreStats', () => {
    it('should return genre statistics', async () => {
      const mockStats = [
        { genre: 'electronic', count: '50', total_plays: '5000' },
        { genre: 'ambient', count: '30', total_plays: '2000' },
      ];

      (pool.query as jest.Mock).mockResolvedValueOnce({
        rows: mockStats,
        rowCount: 2,
      });

      const result = await repository.getGenreStats();

      expect(result).toEqual([
        { genre: 'electronic', count: 50, total_plays: 5000 },
        { genre: 'ambient', count: 30, total_plays: 2000 },
      ]);
    });
  });
});
