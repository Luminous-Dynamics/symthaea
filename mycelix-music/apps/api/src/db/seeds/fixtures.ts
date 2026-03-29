// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Test Fixtures
 *
 * Provides consistent test data for unit and integration tests.
 * These fixtures are deterministic - same data every time.
 */

import { randomUUID } from 'crypto';

/**
 * Fixed UUIDs for predictable testing
 */
export const FixedIds = {
  songs: {
    song1: '11111111-1111-1111-1111-111111111111',
    song2: '22222222-2222-2222-2222-222222222222',
    song3: '33333333-3333-3333-3333-333333333333',
    song4: '44444444-4444-4444-4444-444444444444',
    song5: '55555555-5555-5555-5555-555555555555',
  },
  plays: {
    play1: 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
    play2: 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb',
    play3: 'cccccccc-cccc-cccc-cccc-cccccccccccc',
  },
  artists: {
    artist1: '0x1111111111111111111111111111111111111111',
    artist2: '0x2222222222222222222222222222222222222222',
    artist3: '0x3333333333333333333333333333333333333333',
  },
  listeners: {
    listener1: '0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    listener2: '0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
  },
};

/**
 * Test song fixtures
 */
export const SongFixtures = {
  /**
   * Basic valid song
   */
  validSong: {
    id: FixedIds.songs.song1,
    title: 'Test Song One',
    artist: 'Test Artist',
    artist_address: FixedIds.artists.artist1,
    genre: 'electronic',
    description: 'A test song for unit testing',
    ipfs_hash: 'QmTestHash123456789012345678901234567890123456',
    payment_model: 'per_play' as const,
    plays: 100,
    earnings: '0.100000',
    created_at: new Date('2024-01-01T00:00:00Z'),
  },

  /**
   * Song with subscription model
   */
  subscriptionSong: {
    id: FixedIds.songs.song2,
    title: 'Subscription Track',
    artist: 'Premium Artist',
    artist_address: FixedIds.artists.artist2,
    genre: 'ambient',
    description: 'A premium subscription track',
    ipfs_hash: 'QmSubsHash987654321098765432109876543210987',
    payment_model: 'subscription' as const,
    plays: 500,
    earnings: '2.500000',
    created_at: new Date('2024-02-15T12:00:00Z'),
  },

  /**
   * Song with no plays
   */
  newSong: {
    id: FixedIds.songs.song3,
    title: 'Brand New Release',
    artist: 'Emerging Artist',
    artist_address: FixedIds.artists.artist3,
    genre: 'experimental',
    description: 'Just released today',
    ipfs_hash: 'QmNewHash111222333444555666777888999000111',
    payment_model: 'per_play' as const,
    plays: 0,
    earnings: '0.000000',
    created_at: new Date('2024-06-01T00:00:00Z'),
  },

  /**
   * Popular song with many plays
   */
  popularSong: {
    id: FixedIds.songs.song4,
    title: 'Viral Hit',
    artist: 'Famous Artist',
    artist_address: FixedIds.artists.artist1,
    genre: 'techno',
    description: 'The most played song on the platform',
    ipfs_hash: 'QmPopHash999888777666555444333222111000999',
    payment_model: 'per_play' as const,
    plays: 10000,
    earnings: '50.000000',
    created_at: new Date('2023-06-15T00:00:00Z'),
  },

  /**
   * Get all fixtures as array
   */
  all(): typeof SongFixtures.validSong[] {
    return [
      this.validSong,
      this.subscriptionSong,
      this.newSong,
      this.popularSong,
    ];
  },
};

/**
 * Test play fixtures
 */
export const PlayFixtures = {
  /**
   * Confirmed play
   */
  confirmedPlay: {
    id: FixedIds.plays.play1,
    song_id: FixedIds.songs.song1,
    listener_address: FixedIds.listeners.listener1,
    amount: '0.001000',
    status: 'confirmed' as const,
    transaction_hash: '0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef',
    created_at: new Date('2024-03-01T10:00:00Z'),
    confirmed_at: new Date('2024-03-01T10:01:00Z'),
  },

  /**
   * Pending play
   */
  pendingPlay: {
    id: FixedIds.plays.play2,
    song_id: FixedIds.songs.song2,
    listener_address: FixedIds.listeners.listener2,
    amount: '0.002000',
    status: 'pending' as const,
    transaction_hash: null,
    created_at: new Date('2024-03-15T14:30:00Z'),
    confirmed_at: null,
  },

  /**
   * Failed play
   */
  failedPlay: {
    id: FixedIds.plays.play3,
    song_id: FixedIds.songs.song1,
    listener_address: FixedIds.listeners.listener1,
    amount: '0.001000',
    status: 'failed' as const,
    transaction_hash: null,
    created_at: new Date('2024-02-20T08:00:00Z'),
    confirmed_at: null,
  },

  /**
   * Get all fixtures as array
   */
  all(): typeof PlayFixtures.confirmedPlay[] {
    return [
      this.confirmedPlay,
      this.pendingPlay,
      this.failedPlay,
    ];
  },
};

/**
 * Factory functions for generating test data
 */
export const Factories = {
  /**
   * Create a song with optional overrides
   */
  createSong(overrides: Partial<typeof SongFixtures.validSong> = {}) {
    return {
      ...SongFixtures.validSong,
      id: randomUUID(),
      created_at: new Date(),
      ...overrides,
    };
  },

  /**
   * Create a play with optional overrides
   */
  createPlay(overrides: Partial<typeof PlayFixtures.confirmedPlay> = {}) {
    return {
      ...PlayFixtures.confirmedPlay,
      id: randomUUID(),
      created_at: new Date(),
      confirmed_at: new Date(),
      ...overrides,
    };
  },

  /**
   * Create multiple songs
   */
  createSongs(count: number, overrides: Partial<typeof SongFixtures.validSong> = []) {
    return Array.from({ length: count }, (_, i) =>
      this.createSong({
        title: `Generated Song ${i + 1}`,
        ...overrides,
      })
    );
  },

  /**
   * Create multiple plays for a song
   */
  createPlaysForSong(songId: string, count: number) {
    return Array.from({ length: count }, () =>
      this.createPlay({ song_id: songId })
    );
  },
};

/**
 * Request fixtures for API testing
 */
export const RequestFixtures = {
  /**
   * Valid create song request
   */
  createSongRequest: {
    title: 'New Test Song',
    artist: 'Test Artist',
    artist_address: FixedIds.artists.artist1,
    genre: 'electronic',
    description: 'A new song being created',
    ipfs_hash: 'QmNewSongHash12345678901234567890123456789012',
    payment_model: 'per_play',
  },

  /**
   * Invalid create song request (missing required fields)
   */
  invalidCreateSongRequest: {
    title: 'Incomplete Song',
    // Missing artist, artist_address, etc.
  },

  /**
   * Valid play request
   */
  createPlayRequest: {
    song_id: FixedIds.songs.song1,
    listener_address: FixedIds.listeners.listener1,
    amount: '0.001000',
  },

  /**
   * Search filters
   */
  searchFilters: {
    genre: 'electronic',
    query: 'test',
    limit: 10,
    offset: 0,
  },
};

/**
 * Response fixtures for mocking
 */
export const ResponseFixtures = {
  /**
   * Paginated songs response
   */
  paginatedSongs: {
    success: true,
    data: SongFixtures.all(),
    meta: {
      pagination: {
        total: 4,
        limit: 20,
        offset: 0,
        hasMore: false,
      },
    },
  },

  /**
   * Single song response
   */
  singleSong: {
    success: true,
    data: SongFixtures.validSong,
  },

  /**
   * Not found response
   */
  notFound: {
    success: false,
    error: {
      code: 'NOT_FOUND',
      message: 'Resource not found',
    },
  },

  /**
   * Validation error response
   */
  validationError: {
    success: false,
    error: {
      code: 'VALIDATION_ERROR',
      message: 'Validation failed',
      details: {
        title: 'Title is required',
      },
    },
  },
};

export default {
  FixedIds,
  SongFixtures,
  PlayFixtures,
  Factories,
  RequestFixtures,
  ResponseFixtures,
};
