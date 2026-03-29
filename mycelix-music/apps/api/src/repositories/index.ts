// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Repository Index
 *
 * Central export point for all repositories.
 * Provides factory function for creating repositories with a shared pool.
 */

import { Pool } from 'pg';
import { SongRepository } from './song.repository';
import { PlayRepository } from './play.repository';

export { BaseRepository, RepositoryError } from './base.repository';
export { SongRepository } from './song.repository';
export { PlayRepository } from './play.repository';

export type { QueryOptions, PaginationOptions, SortOptions, FilterOptions, PaginatedResult } from './base.repository';
export type { Song, CreateSongInput, SongSearchOptions } from './song.repository';
export type { Play, CreatePlayInput, PlayWithSong, TimeRange } from './play.repository';

/**
 * Repository container for dependency injection
 */
export interface Repositories {
  songs: SongRepository;
  plays: PlayRepository;
}

/**
 * Create all repositories with a shared connection pool
 */
export function createRepositories(pool: Pool): Repositories {
  return {
    songs: new SongRepository(pool),
    plays: new PlayRepository(pool),
  };
}

/**
 * Singleton instance (initialized lazily)
 */
let _repositories: Repositories | null = null;

/**
 * Get or create the global repositories instance
 */
export function getRepositories(pool?: Pool): Repositories {
  if (!_repositories) {
    if (!pool) {
      throw new Error('Pool must be provided for initial repository creation');
    }
    _repositories = createRepositories(pool);
  }
  return _repositories;
}

/**
 * Reset repositories (useful for testing)
 */
export function resetRepositories(): void {
  _repositories = null;
}
