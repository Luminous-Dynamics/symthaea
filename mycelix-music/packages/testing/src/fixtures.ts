// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Test Fixtures
 *
 * Factory functions for generating test data.
 */

import { faker } from '@faker-js/faker';

// ==================== User Fixtures ====================

export interface TestUser {
  address: string;
  displayName: string;
  email: string;
  avatar: string;
  isArtist: boolean;
  createdAt: Date;
}

export function createUser(overrides: Partial<TestUser> = {}): TestUser {
  return {
    address: faker.string.hexadecimal({ length: 40, prefix: '0x' }),
    displayName: faker.person.fullName(),
    email: faker.internet.email(),
    avatar: faker.image.avatar(),
    isArtist: faker.datatype.boolean(),
    createdAt: faker.date.past(),
    ...overrides,
  };
}

export function createArtist(overrides: Partial<TestUser> = {}): TestUser {
  return createUser({ isArtist: true, ...overrides });
}

// ==================== Song Fixtures ====================

export interface TestSong {
  id: string;
  title: string;
  artist: string;
  artistAddress: string;
  genre: string;
  duration: number;
  coverArt: string;
  audioUrl: string;
  playCount: number;
  likeCount: number;
  isLiked: boolean;
  createdAt: Date;
}

const GENRES = ['Electronic', 'Hip Hop', 'Pop', 'Rock', 'Jazz', 'Classical', 'R&B', 'Lo-Fi'];

export function createSong(overrides: Partial<TestSong> = {}): TestSong {
  return {
    id: faker.string.uuid(),
    title: faker.music.songName(),
    artist: faker.person.fullName(),
    artistAddress: faker.string.hexadecimal({ length: 40, prefix: '0x' }),
    genre: faker.helpers.arrayElement(GENRES),
    duration: faker.number.int({ min: 120, max: 420 }),
    coverArt: faker.image.urlPicsumPhotos({ width: 500, height: 500 }),
    audioUrl: `https://cdn.mycelix.io/audio/${faker.string.uuid()}/master.m3u8`,
    playCount: faker.number.int({ min: 0, max: 1000000 }),
    likeCount: faker.number.int({ min: 0, max: 50000 }),
    isLiked: faker.datatype.boolean(),
    createdAt: faker.date.past(),
    ...overrides,
  };
}

export function createSongs(count: number, overrides: Partial<TestSong> = {}): TestSong[] {
  return Array.from({ length: count }, () => createSong(overrides));
}

// ==================== Playlist Fixtures ====================

export interface TestPlaylist {
  id: string;
  name: string;
  description: string;
  coverImage: string;
  ownerAddress: string;
  ownerName: string;
  songCount: number;
  duration: number;
  isPublic: boolean;
  createdAt: Date;
}

export function createPlaylist(overrides: Partial<TestPlaylist> = {}): TestPlaylist {
  return {
    id: faker.string.uuid(),
    name: faker.music.genre() + ' Vibes',
    description: faker.lorem.sentence(),
    coverImage: faker.image.urlPicsumPhotos({ width: 500, height: 500 }),
    ownerAddress: faker.string.hexadecimal({ length: 40, prefix: '0x' }),
    ownerName: faker.person.fullName(),
    songCount: faker.number.int({ min: 5, max: 50 }),
    duration: faker.number.int({ min: 600, max: 7200 }),
    isPublic: faker.datatype.boolean({ probability: 0.8 }),
    createdAt: faker.date.past(),
    ...overrides,
  };
}

// ==================== Dashboard Fixtures ====================

export interface TestDashboardData {
  streams: {
    totalStreams: number;
    uniqueListeners: number;
  };
  revenue: {
    totalRevenue: number;
    streamingRevenue: number;
    pendingPayout: number;
  };
  engagement: {
    followers: number;
    followerGrowthPercent: number;
  };
  topSongs: Array<{
    songId: string;
    title: string;
    streams: number;
    trend: 'up' | 'down' | 'stable';
  }>;
  streamHistory: Array<{
    date: string;
    value: number;
  }>;
}

export function createDashboardData(overrides: Partial<TestDashboardData> = {}): TestDashboardData {
  const baseStreams = faker.number.int({ min: 1000, max: 100000 });

  return {
    streams: {
      totalStreams: baseStreams,
      uniqueListeners: Math.floor(baseStreams * 0.4),
    },
    revenue: {
      totalRevenue: baseStreams * 0.004,
      streamingRevenue: baseStreams * 0.003,
      pendingPayout: baseStreams * 0.001,
    },
    engagement: {
      followers: faker.number.int({ min: 100, max: 50000 }),
      followerGrowthPercent: faker.number.float({ min: -10, max: 30, fractionDigits: 1 }),
    },
    topSongs: Array.from({ length: 5 }, () => ({
      songId: faker.string.uuid(),
      title: faker.music.songName(),
      streams: faker.number.int({ min: 100, max: 10000 }),
      trend: faker.helpers.arrayElement(['up', 'down', 'stable'] as const),
    })),
    streamHistory: Array.from({ length: 30 }, (_, i) => ({
      date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      value: faker.number.int({ min: 100, max: 5000 }),
    })),
    ...overrides,
  };
}

// ==================== Search Fixtures ====================

export interface TestSearchResults {
  songs: TestSong[];
  artists: Array<{
    address: string;
    name: string;
    avatar: string;
    followerCount: number;
  }>;
  playlists: TestPlaylist[];
}

export function createSearchResults(overrides: Partial<TestSearchResults> = {}): TestSearchResults {
  return {
    songs: createSongs(5),
    artists: Array.from({ length: 3 }, () => ({
      address: faker.string.hexadecimal({ length: 40, prefix: '0x' }),
      name: faker.person.fullName(),
      avatar: faker.image.avatar(),
      followerCount: faker.number.int({ min: 100, max: 100000 }),
    })),
    playlists: Array.from({ length: 3 }, () => createPlaylist()),
    ...overrides,
  };
}

// ==================== Transaction Fixtures ====================

export interface TestTransaction {
  hash: string;
  from: string;
  to: string;
  value: string;
  blockNumber: number;
  timestamp: number;
}

export function createTransaction(overrides: Partial<TestTransaction> = {}): TestTransaction {
  return {
    hash: faker.string.hexadecimal({ length: 64, prefix: '0x' }),
    from: faker.string.hexadecimal({ length: 40, prefix: '0x' }),
    to: faker.string.hexadecimal({ length: 40, prefix: '0x' }),
    value: faker.number.bigInt({ min: 0n, max: 1000000000000000000n }).toString(),
    blockNumber: faker.number.int({ min: 1000000, max: 20000000 }),
    timestamp: Math.floor(Date.now() / 1000) - faker.number.int({ min: 0, max: 86400 * 30 }),
    ...overrides,
  };
}
