// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Database Package
 *
 * Prisma client and utilities for Mycelix platform.
 */

import { PrismaClient } from '@prisma/client';

// Global Prisma client for connection pooling
declare global {
  var prisma: PrismaClient | undefined;
}

export const prisma = global.prisma || new PrismaClient({
  log: process.env.NODE_ENV === 'development'
    ? ['query', 'error', 'warn']
    : ['error'],
});

if (process.env.NODE_ENV !== 'production') {
  global.prisma = prisma;
}

export * from '@prisma/client';

// Re-export types for convenience
export type {
  User,
  Song,
  Playlist,
  PlaylistTrack,
  Play,
  Follow,
  Comment,
  Notification,
  Report,
} from '@prisma/client';
