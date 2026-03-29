// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Cache Key Patterns
 *
 * Standardized cache key patterns for the Mycelix platform.
 */

export interface CachePattern {
  key: (...args: string[]) => string;
  ttl: number;
  invalidateOn?: string[];
}

export const keyPatterns = {
  // Song patterns
  song: (id: string) => `song:${id}`,
  songsByArtist: (artistId: string) => `artist:${artistId}:songs`,
  songsByGenre: (genre: string) => `genre:${genre}:songs`,
  songMetadata: (id: string) => `song:${id}:metadata`,
  songPlayCount: (id: string) => `song:${id}:plays`,

  // Artist patterns
  artist: (id: string) => `artist:${id}`,
  artistStats: (id: string) => `artist:${id}:stats`,
  artistFollowers: (id: string) => `artist:${id}:followers`,
  topArtists: () => 'artists:top',

  // User patterns
  user: (id: string) => `user:${id}`,
  userProfile: (id: string) => `user:${id}:profile`,
  userLibrary: (id: string) => `user:${id}:library`,
  userPlaylists: (id: string) => `user:${id}:playlists`,
  userLikes: (id: string) => `user:${id}:likes`,
  userFollowing: (id: string) => `user:${id}:following`,

  // Playlist patterns
  playlist: (id: string) => `playlist:${id}`,
  playlistTracks: (id: string) => `playlist:${id}:tracks`,
  featuredPlaylists: () => 'playlists:featured',
  trendingPlaylists: () => 'playlists:trending',

  // Search patterns
  searchResults: (query: string, type: string) => `search:${type}:${hashQuery(query)}`,
  autocomplete: (prefix: string) => `autocomplete:${prefix.toLowerCase()}`,

  // Feed patterns
  homeFeed: (userId: string) => `feed:home:${userId}`,
  activityFeed: (userId: string) => `feed:activity:${userId}`,
  trendingSongs: () => 'trending:songs',
  newReleases: () => 'releases:new',

  // Stats patterns
  globalStats: () => 'stats:global',
  genreStats: () => 'stats:genres',
  dailyStats: (date: string) => `stats:daily:${date}`,

  // Session patterns
  session: (sessionId: string) => `session:${sessionId}`,
  userSessions: (userId: string) => `user:${userId}:sessions`,
};

/**
 * TTL presets for different data types
 */
export const ttlPresets = {
  realtime: 10, // 10 seconds
  shortLived: 60, // 1 minute
  standard: 300, // 5 minutes
  longLived: 3600, // 1 hour
  daily: 86400, // 24 hours
  weekly: 604800, // 7 days
};

/**
 * Cache patterns with TTL and invalidation rules
 */
export const CachePatterns: Record<string, CachePattern> = {
  song: {
    key: keyPatterns.song,
    ttl: ttlPresets.longLived,
    invalidateOn: ['song:update', 'song:delete'],
  },
  artist: {
    key: keyPatterns.artist,
    ttl: ttlPresets.longLived,
    invalidateOn: ['artist:update'],
  },
  userProfile: {
    key: keyPatterns.userProfile,
    ttl: ttlPresets.standard,
    invalidateOn: ['user:update'],
  },
  playlist: {
    key: keyPatterns.playlist,
    ttl: ttlPresets.standard,
    invalidateOn: ['playlist:update', 'playlist:addTrack', 'playlist:removeTrack'],
  },
  searchResults: {
    key: keyPatterns.searchResults,
    ttl: ttlPresets.shortLived,
  },
  trendingSongs: {
    key: () => keyPatterns.trendingSongs(),
    ttl: ttlPresets.shortLived,
  },
  homeFeed: {
    key: keyPatterns.homeFeed,
    ttl: ttlPresets.standard,
    invalidateOn: ['follow', 'unfollow', 'song:upload'],
  },
};

/**
 * Build a cache key from pattern and arguments
 */
export function buildKey(pattern: string, ...args: string[]): string {
  const patternFn = keyPatterns[pattern as keyof typeof keyPatterns];
  if (!patternFn) {
    throw new Error(`Unknown cache pattern: ${pattern}`);
  }
  return (patternFn as (...args: string[]) => string)(...args);
}

/**
 * Parse a cache key to extract pattern and arguments
 */
export function parseKey(key: string): { type: string; id?: string; subtype?: string } {
  const parts = key.split(':');
  return {
    type: parts[0],
    id: parts[1],
    subtype: parts[2],
  };
}

/**
 * Get all keys that should be invalidated for an event
 */
export function getInvalidationKeys(
  event: string,
  context: Record<string, string>
): string[] {
  const keys: string[] = [];

  switch (event) {
    case 'song:update':
    case 'song:delete':
      if (context.songId) {
        keys.push(keyPatterns.song(context.songId));
        keys.push(keyPatterns.songMetadata(context.songId));
      }
      if (context.artistId) {
        keys.push(keyPatterns.songsByArtist(context.artistId));
      }
      break;

    case 'song:upload':
      if (context.artistId) {
        keys.push(keyPatterns.songsByArtist(context.artistId));
        keys.push(keyPatterns.artistStats(context.artistId));
      }
      keys.push(keyPatterns.newReleases());
      break;

    case 'user:update':
      if (context.userId) {
        keys.push(keyPatterns.user(context.userId));
        keys.push(keyPatterns.userProfile(context.userId));
      }
      break;

    case 'playlist:update':
    case 'playlist:addTrack':
    case 'playlist:removeTrack':
      if (context.playlistId) {
        keys.push(keyPatterns.playlist(context.playlistId));
        keys.push(keyPatterns.playlistTracks(context.playlistId));
      }
      if (context.userId) {
        keys.push(keyPatterns.userPlaylists(context.userId));
      }
      break;

    case 'follow':
    case 'unfollow':
      if (context.followerId) {
        keys.push(keyPatterns.userFollowing(context.followerId));
        keys.push(keyPatterns.homeFeed(context.followerId));
      }
      if (context.followingId) {
        keys.push(keyPatterns.artistFollowers(context.followingId));
        keys.push(keyPatterns.artistStats(context.followingId));
      }
      break;

    case 'like':
    case 'unlike':
      if (context.userId) {
        keys.push(keyPatterns.userLikes(context.userId));
      }
      break;
  }

  return keys;
}

// Helper to hash query strings for cache keys
function hashQuery(query: string): string {
  let hash = 0;
  for (let i = 0; i < query.length; i++) {
    const char = query.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(36);
}
