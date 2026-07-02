// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * React Hooks for Mycelix Music SDK
 *
 * Usage:
 *   import { useSongs, useSong, useArtistStats, useRealTimeUpdates } from '@mycelix/sdk/hooks';
 *
 *   function SongList() {
 *     const { songs, isLoading, error, refetch } = useSongs({ genre: 'electronic' });
 *     // ...
 *   }
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { io, Socket } from 'socket.io-client';

// Types
export interface Song {
  id: string;
  title: string;
  artist: string;
  artistAddress: string;
  genre: string;
  description?: string;
  ipfsHash: string;
  paymentModel: string;
  coverArt?: string;
  audioUrl?: string;
  plays: number;
  earnings: number;
  createdAt: string;
}

export interface ArtistStats {
  artistAddress: string;
  totalSongs: number;
  totalPlays: number;
  totalEarnings: number;
}

export interface PlayEvent {
  songId: string;
  songTitle: string;
  artistAddress: string;
  listenerAddress: string;
  amount: number;
  timestamp: string;
}

export interface UseSongsOptions {
  genre?: string;
  paymentModel?: string;
  search?: string;
  limit?: number;
  offset?: number;
  sort?: 'created_at' | 'plays' | 'earnings';
  order?: 'asc' | 'desc';
}

export interface UseSongsResult {
  songs: Song[];
  total: number;
  isLoading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
  hasMore: boolean;
  loadMore: () => Promise<void>;
}

export interface UseSongResult {
  song: Song | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

export interface UseArtistStatsResult {
  stats: ArtistStats | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

// Configuration
let apiBaseUrl = 'http://localhost:3100';

export function configureHooks(baseUrl: string) {
  apiBaseUrl = baseUrl;
}

// Fetch helper
async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: response.statusText }));
    throw new Error(error.error || error.message || 'API request failed');
  }

  return response.json();
}

/**
 * Hook for fetching a list of songs with filtering and pagination
 */
export function useSongs(options: UseSongsOptions = {}): UseSongsResult {
  const [songs, setSongs] = useState<Song[]>([]);
  const [total, setTotal] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [offset, setOffset] = useState(options.offset || 0);

  const limit = options.limit || 20;

  const fetchSongs = useCallback(async (append = false) => {
    setIsLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams();
      if (options.genre) params.set('genre', options.genre);
      if (options.paymentModel) params.set('model', options.paymentModel);
      if (options.search) params.set('q', options.search);
      if (options.sort) params.set('sort', options.sort);
      if (options.order) params.set('order', options.order);
      params.set('limit', String(limit));
      params.set('offset', String(append ? offset : 0));

      const response = await fetch(`${apiBaseUrl}/api/songs?${params}`);
      const totalHeader = response.headers.get('X-Total-Count');
      const data = await response.json();

      if (append) {
        setSongs(prev => [...prev, ...data]);
      } else {
        setSongs(data);
        setOffset(0);
      }

      if (totalHeader) {
        setTotal(parseInt(totalHeader, 10));
      }
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to fetch songs'));
    } finally {
      setIsLoading(false);
    }
  }, [options.genre, options.paymentModel, options.search, options.sort, options.order, limit, offset]);

  const loadMore = useCallback(async () => {
    const newOffset = offset + limit;
    setOffset(newOffset);
    await fetchSongs(true);
  }, [offset, limit, fetchSongs]);

  useEffect(() => {
    fetchSongs();
  }, [options.genre, options.paymentModel, options.search, options.sort, options.order]);

  return {
    songs,
    total,
    isLoading,
    error,
    refetch: () => fetchSongs(false),
    hasMore: songs.length < total,
    loadMore,
  };
}

/**
 * Hook for fetching a single song by ID
 */
export function useSong(songId: string | null): UseSongResult {
  const [song, setSong] = useState<Song | null>(null);
  const [isLoading, setIsLoading] = useState(!!songId);
  const [error, setError] = useState<Error | null>(null);

  const fetchSong = useCallback(async () => {
    if (!songId) {
      setSong(null);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const data = await fetchApi<Song>(`/api/songs/${songId}`);
      setSong(data);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to fetch song'));
      setSong(null);
    } finally {
      setIsLoading(false);
    }
  }, [songId]);

  useEffect(() => {
    fetchSong();
  }, [fetchSong]);

  return {
    song,
    isLoading,
    error,
    refetch: fetchSong,
  };
}

/**
 * Hook for fetching artist statistics
 */
export function useArtistStats(address: string | null): UseArtistStatsResult {
  const [stats, setStats] = useState<ArtistStats | null>(null);
  const [isLoading, setIsLoading] = useState(!!address);
  const [error, setError] = useState<Error | null>(null);

  const fetchStats = useCallback(async () => {
    if (!address) {
      setStats(null);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const data = await fetchApi<ArtistStats>(`/api/artists/${address}/stats`);
      setStats(data);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to fetch artist stats'));
      setStats(null);
    } finally {
      setIsLoading(false);
    }
  }, [address]);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  return {
    stats,
    isLoading,
    error,
    refetch: fetchStats,
  };
}

/**
 * Hook for real-time updates via WebSocket
 */
export function useRealTimeUpdates(options: {
  artistAddress?: string;
  songId?: string;
  onPlay?: (event: PlayEvent) => void;
  onSongUpdate?: (data: { songId: string; plays: number; earnings: number }) => void;
}) {
  const [connected, setConnected] = useState(false);
  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    // Connect to WebSocket
    const socket = io(apiBaseUrl, {
      path: '/socket.io',
      transports: ['websocket', 'polling'],
    });

    socketRef.current = socket;

    socket.on('connect', () => {
      setConnected(true);

      // Subscribe to channels
      if (options.artistAddress) {
        socket.emit('subscribe', { channel: 'artist', address: options.artistAddress });
      }
      if (options.songId) {
        socket.emit('subscribe', { channel: 'song', songId: options.songId });
      }
      if (!options.artistAddress && !options.songId) {
        socket.emit('subscribe', { channel: 'global' });
      }
    });

    socket.on('disconnect', () => {
      setConnected(false);
    });

    socket.on('play:new', (event: PlayEvent) => {
      options.onPlay?.(event);
    });

    socket.on('song:update', (data: { songId: string; plays: number; earnings: number }) => {
      options.onSongUpdate?.(data);
    });

    return () => {
      socket.disconnect();
    };
  }, [options.artistAddress, options.songId]);

  return { connected };
}

/**
 * Hook for search with debouncing
 */
export function useSearch(debounceMs = 300) {
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [results, setResults] = useState<Song[]>([]);
  const [isSearching, setIsSearching] = useState(false);

  // Debounce the query
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(query);
    }, debounceMs);

    return () => clearTimeout(timer);
  }, [query, debounceMs]);

  // Search when debounced query changes
  useEffect(() => {
    if (!debouncedQuery.trim()) {
      setResults([]);
      return;
    }

    setIsSearching(true);

    fetchApi<Song[]>(`/api/songs?q=${encodeURIComponent(debouncedQuery)}&limit=10`)
      .then(setResults)
      .catch(() => setResults([]))
      .finally(() => setIsSearching(false));
  }, [debouncedQuery]);

  return {
    query,
    setQuery,
    results,
    isSearching,
  };
}

/**
 * Hook for top songs/charts
 */
export function useTopSongs(limit = 10) {
  const [songs, setSongs] = useState<Song[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    setIsLoading(true);

    fetchApi<Song[]>(`/api/analytics/top-songs?limit=${limit}`)
      .then(setSongs)
      .catch(err => setError(err instanceof Error ? err : new Error('Failed to fetch top songs')))
      .finally(() => setIsLoading(false));
  }, [limit]);

  return { songs, isLoading, error };
}

export default {
  configureHooks,
  useSongs,
  useSong,
  useArtistStats,
  useRealTimeUpdates,
  useSearch,
  useTopSongs,
};
