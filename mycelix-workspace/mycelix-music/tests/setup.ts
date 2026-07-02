// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Test Setup & Utilities
 *
 * Common test configuration, mocks, and utilities for unit,
 * integration, and e2e testing.
 */

import { beforeAll, afterAll, beforeEach, afterEach, vi } from 'vitest';
import { PrismaClient } from '@prisma/client';
import Redis from 'ioredis';

// ============================================================================
// Test Database
// ============================================================================

let prisma: PrismaClient;
let redis: Redis;

export const getTestPrisma = () => prisma;
export const getTestRedis = () => redis;

beforeAll(async () => {
  // Initialize test database connection
  prisma = new PrismaClient({
    datasources: {
      db: {
        url: process.env.TEST_DATABASE_URL || 'postgresql://test:test@localhost:5432/mycelix_test',
      },
    },
  });

  // Initialize test Redis connection
  redis = new Redis(process.env.TEST_REDIS_URL || 'redis://localhost:6379/1');

  // Run migrations
  await prisma.$executeRawUnsafe(`
    CREATE SCHEMA IF NOT EXISTS test;
  `);
});

afterAll(async () => {
  await prisma.$disconnect();
  await redis.quit();
});

beforeEach(async () => {
  // Clear Redis before each test
  await redis.flushdb();
});

afterEach(() => {
  // Clear all mocks
  vi.clearAllMocks();
});

// ============================================================================
// Test Factories
// ============================================================================

export const createTestUser = async (overrides = {}) => {
  const defaultUser = {
    email: `test-${Date.now()}@example.com`,
    username: `testuser${Date.now()}`,
    displayName: 'Test User',
    passwordHash: '$2b$10$K7L1OJ45/4Y2nIvhRVpCe.FSmhDdWoXehVzJptJ/op0lSsvqNu',
  };

  return prisma.user.create({
    data: { ...defaultUser, ...overrides },
  });
};

export const createTestArtist = async (userId: string, overrides = {}) => {
  const defaultArtist = {
    userId,
    stageName: `Artist ${Date.now()}`,
    genres: ['pop', 'rock'],
    verified: false,
  };

  return prisma.artist.create({
    data: { ...defaultArtist, ...overrides },
  });
};

export const createTestTrack = async (artistId: string, overrides = {}) => {
  const defaultTrack = {
    title: `Track ${Date.now()}`,
    duration: 180,
    audioUrl: 'https://example.com/audio.mp3',
    primaryArtistId: artistId,
    genre: 'pop',
    bpm: 120,
    key: 'C major',
  };

  return prisma.track.create({
    data: { ...defaultTrack, ...overrides },
  });
};

export const createTestAlbum = async (artistId: string, overrides = {}) => {
  const defaultAlbum = {
    title: `Album ${Date.now()}`,
    type: 'album',
    artistId,
  };

  return prisma.album.create({
    data: { ...defaultAlbum, ...overrides },
  });
};

export const createTestPlaylist = async (userId: string, overrides = {}) => {
  const defaultPlaylist = {
    name: `Playlist ${Date.now()}`,
    ownerId: userId,
    isPublic: true,
  };

  return prisma.playlist.create({
    data: { ...defaultPlaylist, ...overrides },
  });
};

// ============================================================================
// Authentication Helpers
// ============================================================================

export const generateTestToken = (userId: string, scopes: string[] = ['user:read']) => {
  // Would use actual JWT signing in real tests
  return `test_token_${userId}_${scopes.join('_')}`;
};

export const createAuthenticatedRequest = (userId: string) => {
  const token = generateTestToken(userId);
  return {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  };
};

// ============================================================================
// API Test Helpers
// ============================================================================

export const createTestContext = (overrides = {}) => ({
  user: null,
  isAuthenticated: false,
  prisma,
  redis,
  ...overrides,
});

export const mockApiRequest = (method: string, path: string, body?: any) => ({
  method,
  url: path,
  body,
  headers: {},
  query: {},
});

// ============================================================================
// Event Testing
// ============================================================================

export const waitForEvent = <T>(
  emitter: { on: (event: string, handler: (data: T) => void) => void },
  event: string,
  timeout = 5000
): Promise<T> => {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`Timeout waiting for event: ${event}`));
    }, timeout);

    emitter.on(event, (data: T) => {
      clearTimeout(timer);
      resolve(data);
    });
  });
};

// ============================================================================
// Mock Services
// ============================================================================

export const mockS3 = {
  upload: vi.fn().mockResolvedValue({ Location: 'https://s3.example.com/file.mp3' }),
  getSignedUrl: vi.fn().mockResolvedValue('https://s3.example.com/signed-url'),
  deleteObject: vi.fn().mockResolvedValue({}),
};

export const mockStripe = {
  customers: {
    create: vi.fn().mockResolvedValue({ id: 'cus_test123' }),
  },
  subscriptions: {
    create: vi.fn().mockResolvedValue({ id: 'sub_test123', status: 'active' }),
  },
  paymentIntents: {
    create: vi.fn().mockResolvedValue({ id: 'pi_test123', client_secret: 'secret' }),
  },
};

export const mockPushNotifications = {
  send: vi.fn().mockResolvedValue({ success: true }),
  sendMultiple: vi.fn().mockResolvedValue({ successCount: 10 }),
};

export const mockEmailService = {
  send: vi.fn().mockResolvedValue({ messageId: 'msg_123' }),
  sendTemplate: vi.fn().mockResolvedValue({ messageId: 'msg_456' }),
};

// ============================================================================
// Snapshot Helpers
// ============================================================================

export const sanitizeForSnapshot = (obj: any) => {
  const sanitized = { ...obj };

  // Replace dynamic values
  if (sanitized.id) sanitized.id = '[ID]';
  if (sanitized.createdAt) sanitized.createdAt = '[TIMESTAMP]';
  if (sanitized.updatedAt) sanitized.updatedAt = '[TIMESTAMP]';
  if (sanitized.token) sanitized.token = '[TOKEN]';

  return sanitized;
};

// ============================================================================
// Performance Testing
// ============================================================================

export const measureExecutionTime = async <T>(
  fn: () => Promise<T>
): Promise<{ result: T; duration: number }> => {
  const start = performance.now();
  const result = await fn();
  const duration = performance.now() - start;
  return { result, duration };
};

export const assertPerformance = (
  duration: number,
  maxDuration: number,
  operation: string
) => {
  if (duration > maxDuration) {
    throw new Error(
      `Performance assertion failed: ${operation} took ${duration}ms (max: ${maxDuration}ms)`
    );
  }
};

// ============================================================================
// Database Cleanup
// ============================================================================

export const cleanupDatabase = async () => {
  const tablenames = await prisma.$queryRaw<
    Array<{ tablename: string }>
  >`SELECT tablename FROM pg_tables WHERE schemaname='public'`;

  const tables = tablenames
    .map(({ tablename }) => tablename)
    .filter((name) => name !== '_prisma_migrations')
    .map((name) => `"public"."${name}"`)
    .join(', ');

  if (tables.length > 0) {
    await prisma.$executeRawUnsafe(`TRUNCATE TABLE ${tables} CASCADE;`);
  }
};

// ============================================================================
// Exports
// ============================================================================

export { vi, expect, describe, it, test, beforeAll, afterAll, beforeEach, afterEach } from 'vitest';
