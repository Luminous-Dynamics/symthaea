// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * TanStack Query Configuration
 *
 * Server state management with:
 * - Optimistic updates
 * - Background refetching
 * - Cache persistence
 * - Offline support
 * - Automatic retries
 */

import {
  QueryClient,
  QueryCache,
  MutationCache,
  defaultShouldDehydrateQuery,
  QueryClientProvider as TanStackQueryClientProvider,
} from '@tanstack/react-query';
import { persistQueryClient } from '@tanstack/react-query-persist-client';
import { createSyncStoragePersister } from '@tanstack/query-sync-storage-persister';
import { createAsyncStoragePersister } from '@tanstack/query-async-storage-persister';
import React, { ReactNode } from 'react';

// ==================== Types ====================

export interface APIError {
  message: string;
  code: string;
  status: number;
  details?: Record<string, unknown>;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
}

export interface APIResponse<T> {
  data: T;
  meta?: Record<string, unknown>;
}

// ==================== IndexedDB Persister ====================

async function openQueryDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('mycelix-query-cache', 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains('queries')) {
        db.createObjectStore('queries', { keyPath: 'key' });
      }
    };
  });
}

const indexedDBPersister = createAsyncStoragePersister({
  storage: {
    getItem: async (key: string) => {
      try {
        const db = await openQueryDB();
        return new Promise((resolve, reject) => {
          const tx = db.transaction('queries', 'readonly');
          const store = tx.objectStore('queries');
          const request = store.get(key);
          request.onsuccess = () => resolve(request.result?.value || null);
          request.onerror = () => reject(request.error);
        });
      } catch {
        return null;
      }
    },
    setItem: async (key: string, value: string) => {
      try {
        const db = await openQueryDB();
        return new Promise((resolve, reject) => {
          const tx = db.transaction('queries', 'readwrite');
          const store = tx.objectStore('queries');
          const request = store.put({ key, value });
          request.onsuccess = () => resolve();
          request.onerror = () => reject(request.error);
        });
      } catch {
        // Silently fail
      }
    },
    removeItem: async (key: string) => {
      try {
        const db = await openQueryDB();
        return new Promise((resolve, reject) => {
          const tx = db.transaction('queries', 'readwrite');
          const store = tx.objectStore('queries');
          const request = store.delete(key);
          request.onsuccess = () => resolve();
          request.onerror = () => reject(request.error);
        });
      } catch {
        // Silently fail
      }
    },
  },
  key: 'MYCELIX_QUERY_CACHE',
});

// ==================== Query Client Configuration ====================

// Global error handler
const handleQueryError = (error: unknown) => {
  const apiError = error as APIError;

  // Log errors in development
  if (process.env.NODE_ENV === 'development') {
    console.error('Query error:', apiError);
  }

  // Handle specific error codes
  if (apiError.status === 401) {
    // Redirect to login
    window.location.href = '/login';
  } else if (apiError.status === 403) {
    // Show forbidden toast
    console.warn('Access forbidden');
  } else if (apiError.status >= 500) {
    // Log server errors
    console.error('Server error:', apiError.message);
  }
};

// Create query client
export function createQueryClient(): QueryClient {
  return new QueryClient({
    queryCache: new QueryCache({
      onError: handleQueryError,
    }),
    mutationCache: new MutationCache({
      onError: handleQueryError,
    }),
    defaultOptions: {
      queries: {
        // Stale time: how long data is considered fresh
        staleTime: 1000 * 60 * 5, // 5 minutes

        // Cache time: how long inactive data stays in cache
        gcTime: 1000 * 60 * 60 * 24, // 24 hours

        // Retry configuration
        retry: (failureCount, error) => {
          const apiError = error as APIError;
          // Don't retry on 4xx errors
          if (apiError.status >= 400 && apiError.status < 500) {
            return false;
          }
          return failureCount < 3;
        },
        retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),

        // Refetch configuration
        refetchOnWindowFocus: true,
        refetchOnReconnect: true,
        refetchOnMount: true,

        // Network mode
        networkMode: 'offlineFirst',

        // Placeholder data
        placeholderData: (previousData: unknown) => previousData,
      },
      mutations: {
        // Retry mutations once
        retry: 1,
        retryDelay: 1000,

        // Network mode
        networkMode: 'offlineFirst',
      },
    },
  });
}

// Singleton query client
let queryClient: QueryClient | null = null;

export function getQueryClient(): QueryClient {
  if (!queryClient) {
    queryClient = createQueryClient();

    // Setup persistence
    if (typeof window !== 'undefined') {
      persistQueryClient({
        queryClient,
        persister: indexedDBPersister,
        maxAge: 1000 * 60 * 60 * 24 * 7, // 7 days
        dehydrateOptions: {
          shouldDehydrateQuery: (query) => {
            // Only persist successful queries
            return (
              defaultShouldDehydrateQuery(query) &&
              query.state.status === 'success'
            );
          },
        },
      });
    }
  }

  return queryClient;
}

// ==================== Query Key Factory ====================

export const queryKeys = {
  // Tracks
  tracks: {
    all: ['tracks'] as const,
    lists: () => [...queryKeys.tracks.all, 'list'] as const,
    list: (filters: Record<string, unknown>) => [...queryKeys.tracks.lists(), filters] as const,
    details: () => [...queryKeys.tracks.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.tracks.details(), id] as const,
    search: (query: string) => [...queryKeys.tracks.all, 'search', query] as const,
    trending: () => [...queryKeys.tracks.all, 'trending'] as const,
    recommended: () => [...queryKeys.tracks.all, 'recommended'] as const,
    byArtist: (artistId: string) => [...queryKeys.tracks.all, 'artist', artistId] as const,
    byAlbum: (albumId: string) => [...queryKeys.tracks.all, 'album', albumId] as const,
    byPlaylist: (playlistId: string) => [...queryKeys.tracks.all, 'playlist', playlistId] as const,
  },

  // Artists
  artists: {
    all: ['artists'] as const,
    lists: () => [...queryKeys.artists.all, 'list'] as const,
    list: (filters: Record<string, unknown>) => [...queryKeys.artists.lists(), filters] as const,
    details: () => [...queryKeys.artists.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.artists.details(), id] as const,
    search: (query: string) => [...queryKeys.artists.all, 'search', query] as const,
    following: () => [...queryKeys.artists.all, 'following'] as const,
    similar: (artistId: string) => [...queryKeys.artists.all, 'similar', artistId] as const,
  },

  // Albums
  albums: {
    all: ['albums'] as const,
    lists: () => [...queryKeys.albums.all, 'list'] as const,
    list: (filters: Record<string, unknown>) => [...queryKeys.albums.lists(), filters] as const,
    details: () => [...queryKeys.albums.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.albums.details(), id] as const,
    search: (query: string) => [...queryKeys.albums.all, 'search', query] as const,
    byArtist: (artistId: string) => [...queryKeys.albums.all, 'artist', artistId] as const,
    new: () => [...queryKeys.albums.all, 'new'] as const,
  },

  // Playlists
  playlists: {
    all: ['playlists'] as const,
    lists: () => [...queryKeys.playlists.all, 'list'] as const,
    list: (filters: Record<string, unknown>) => [...queryKeys.playlists.lists(), filters] as const,
    details: () => [...queryKeys.playlists.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.playlists.details(), id] as const,
    my: () => [...queryKeys.playlists.all, 'my'] as const,
    featured: () => [...queryKeys.playlists.all, 'featured'] as const,
  },

  // User
  user: {
    all: ['user'] as const,
    profile: () => [...queryKeys.user.all, 'profile'] as const,
    preferences: () => [...queryKeys.user.all, 'preferences'] as const,
    library: () => [...queryKeys.user.all, 'library'] as const,
    history: () => [...queryKeys.user.all, 'history'] as const,
    stats: () => [...queryKeys.user.all, 'stats'] as const,
  },

  // Analytics
  analytics: {
    all: ['analytics'] as const,
    streams: (range: string) => [...queryKeys.analytics.all, 'streams', range] as const,
    listeners: (range: string) => [...queryKeys.analytics.all, 'listeners', range] as const,
    revenue: (range: string) => [...queryKeys.analytics.all, 'revenue', range] as const,
    demographics: () => [...queryKeys.analytics.all, 'demographics'] as const,
  },

  // Projects (DAW)
  projects: {
    all: ['projects'] as const,
    lists: () => [...queryKeys.projects.all, 'list'] as const,
    list: (filters: Record<string, unknown>) => [...queryKeys.projects.lists(), filters] as const,
    details: () => [...queryKeys.projects.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.projects.details(), id] as const,
    recent: () => [...queryKeys.projects.all, 'recent'] as const,
    shared: () => [...queryKeys.projects.all, 'shared'] as const,
  },

  // Notifications
  notifications: {
    all: ['notifications'] as const,
    unread: () => [...queryKeys.notifications.all, 'unread'] as const,
    count: () => [...queryKeys.notifications.all, 'count'] as const,
  },
} as const;

// ==================== Fetch Utilities ====================

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '/api';

interface FetchOptions extends RequestInit {
  params?: Record<string, string | number | boolean | undefined>;
}

export async function apiFetch<T>(
  endpoint: string,
  options: FetchOptions = {}
): Promise<T> {
  const { params, ...fetchOptions } = options;

  // Build URL with query params
  let url = `${API_BASE_URL}${endpoint}`;
  if (params) {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        searchParams.append(key, String(value));
      }
    });
    const queryString = searchParams.toString();
    if (queryString) {
      url += `?${queryString}`;
    }
  }

  // Get auth token
  const token = typeof window !== 'undefined'
    ? localStorage.getItem('auth_token')
    : null;

  // Build headers
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...fetchOptions.headers,
  };

  if (token) {
    (headers as Record<string, string>)['Authorization'] = `Bearer ${token}`;
  }

  // Make request
  const response = await fetch(url, {
    ...fetchOptions,
    headers,
  });

  // Handle errors
  if (!response.ok) {
    const error: APIError = {
      message: 'An error occurred',
      code: 'UNKNOWN_ERROR',
      status: response.status,
    };

    try {
      const errorData = await response.json();
      error.message = errorData.message || error.message;
      error.code = errorData.code || error.code;
      error.details = errorData.details;
    } catch {
      // Response is not JSON
    }

    throw error;
  }

  // Parse response
  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
}

// ==================== Optimistic Update Helpers ====================

export function createOptimisticUpdate<T>(
  queryClient: QueryClient,
  queryKey: readonly unknown[],
  updater: (old: T | undefined) => T
) {
  // Snapshot previous value
  const previousData = queryClient.getQueryData<T>(queryKey);

  // Optimistically update
  queryClient.setQueryData<T>(queryKey, updater);

  // Return rollback function
  return {
    previousData,
    rollback: () => {
      queryClient.setQueryData(queryKey, previousData);
    },
  };
}

export function createOptimisticListUpdate<T extends { id: string }>(
  queryClient: QueryClient,
  queryKey: readonly unknown[],
  action: 'add' | 'update' | 'remove',
  item: T | string
) {
  const previousData = queryClient.getQueryData<T[]>(queryKey);

  queryClient.setQueryData<T[]>(queryKey, (old = []) => {
    switch (action) {
      case 'add':
        return [...old, item as T];
      case 'update':
        return old.map(i => i.id === (item as T).id ? item as T : i);
      case 'remove':
        return old.filter(i => i.id !== (typeof item === 'string' ? item : item.id));
      default:
        return old;
    }
  });

  return {
    previousData,
    rollback: () => {
      queryClient.setQueryData(queryKey, previousData);
    },
  };
}

// ==================== Provider Component ====================

interface QueryProviderProps {
  children: ReactNode;
}

export function QueryProvider({ children }: QueryProviderProps) {
  const client = getQueryClient();

  return React.createElement(
    TanStackQueryClientProvider,
    { client },
    children
  );
}

export default getQueryClient;
