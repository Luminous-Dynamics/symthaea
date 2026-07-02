// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * useOfflineStorage Hook
 *
 * Provides offline storage and sync capabilities:
 * - IndexedDB for track metadata and user data
 * - Service worker communication for audio caching
 * - Background sync queue
 * - Offline status detection
 */

import { useCallback, useEffect, useRef, useState } from 'react';

// === Types ===

export interface CachedTrack {
  id: string;
  title: string;
  artist: string;
  album?: string;
  duration: number;
  coverUrl?: string;
  audioUrl: string;
  cachedAt: number;
  size: number;
}

export interface SyncQueueItem {
  id: number;
  type: 'play-history' | 'favorite' | 'playlist' | 'reaction';
  data: Record<string, unknown>;
  timestamp: number;
  retries: number;
}

export interface OfflineState {
  isOnline: boolean;
  isServiceWorkerReady: boolean;
  isIndexedDBReady: boolean;
  cachedTracksCount: number;
  cacheSize: number;
  syncQueueCount: number;
  lastSyncTime: number | null;
}

// === IndexedDB Setup ===

const DB_NAME = 'mycelix-offline';
const DB_VERSION = 1;

interface MycelixDB extends IDBDatabase {
  objectStoreNames: DOMStringList;
}

const STORES = {
  tracks: 'tracks',
  playlists: 'playlists',
  favorites: 'favorites',
  history: 'history',
  syncQueue: 'syncQueue',
  settings: 'settings',
} as const;

function openDatabase(): Promise<MycelixDB> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result as MycelixDB);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;

      // Tracks store
      if (!db.objectStoreNames.contains(STORES.tracks)) {
        const tracksStore = db.createObjectStore(STORES.tracks, { keyPath: 'id' });
        tracksStore.createIndex('artist', 'artist', { unique: false });
        tracksStore.createIndex('album', 'album', { unique: false });
        tracksStore.createIndex('cachedAt', 'cachedAt', { unique: false });
      }

      // Playlists store
      if (!db.objectStoreNames.contains(STORES.playlists)) {
        db.createObjectStore(STORES.playlists, { keyPath: 'id' });
      }

      // Favorites store
      if (!db.objectStoreNames.contains(STORES.favorites)) {
        const favStore = db.createObjectStore(STORES.favorites, { keyPath: 'trackId' });
        favStore.createIndex('addedAt', 'addedAt', { unique: false });
      }

      // History store
      if (!db.objectStoreNames.contains(STORES.history)) {
        const historyStore = db.createObjectStore(STORES.history, { keyPath: 'id', autoIncrement: true });
        historyStore.createIndex('trackId', 'trackId', { unique: false });
        historyStore.createIndex('playedAt', 'playedAt', { unique: false });
      }

      // Sync queue store
      if (!db.objectStoreNames.contains(STORES.syncQueue)) {
        const syncStore = db.createObjectStore(STORES.syncQueue, { keyPath: 'id', autoIncrement: true });
        syncStore.createIndex('type', 'type', { unique: false });
        syncStore.createIndex('timestamp', 'timestamp', { unique: false });
      }

      // Settings store
      if (!db.objectStoreNames.contains(STORES.settings)) {
        db.createObjectStore(STORES.settings, { keyPath: 'key' });
      }
    };
  });
}

// === Service Worker Communication ===

function sendToServiceWorker(message: { type: string; payload?: unknown }): Promise<unknown> {
  return new Promise((resolve, reject) => {
    if (!navigator.serviceWorker.controller) {
      reject(new Error('Service worker not active'));
      return;
    }

    const channel = new MessageChannel();
    channel.port1.onmessage = (event) => resolve(event.data);

    navigator.serviceWorker.controller.postMessage(message, [channel.port2]);

    // Timeout after 10 seconds
    setTimeout(() => reject(new Error('Service worker timeout')), 10000);
  });
}

// === Hook ===

export interface UseOfflineStorageReturn extends OfflineState {
  // Track caching
  cacheTrack: (track: CachedTrack) => Promise<void>;
  uncacheTrack: (trackId: string) => Promise<void>;
  getCachedTrack: (trackId: string) => Promise<CachedTrack | undefined>;
  getAllCachedTracks: () => Promise<CachedTrack[]>;
  isTrackCached: (trackId: string) => Promise<boolean>;

  // Favorites (offline)
  addFavorite: (trackId: string) => Promise<void>;
  removeFavorite: (trackId: string) => Promise<void>;
  isFavorite: (trackId: string) => Promise<boolean>;
  getAllFavorites: () => Promise<string[]>;

  // Play history (offline)
  addToHistory: (trackId: string, duration: number) => Promise<void>;
  getRecentHistory: (limit?: number) => Promise<Array<{ trackId: string; playedAt: number }>>;

  // Sync queue
  queueSync: (type: SyncQueueItem['type'], data: Record<string, unknown>) => Promise<void>;
  processSyncQueue: () => Promise<void>;
  getSyncQueueCount: () => Promise<number>;

  // Cache management
  getCacheSize: () => Promise<number>;
  clearAllCaches: () => Promise<void>;
  clearOldCache: (maxAgeDays?: number) => Promise<number>;

  // Settings
  setSetting: (key: string, value: unknown) => Promise<void>;
  getSetting: <T>(key: string, defaultValue: T) => Promise<T>;
}

export function useOfflineStorage(): UseOfflineStorageReturn {
  const dbRef = useRef<MycelixDB | null>(null);

  const [state, setState] = useState<OfflineState>({
    isOnline: typeof navigator !== 'undefined' ? navigator.onLine : true,
    isServiceWorkerReady: false,
    isIndexedDBReady: false,
    cachedTracksCount: 0,
    cacheSize: 0,
    syncQueueCount: 0,
    lastSyncTime: null,
  });

  // Initialize IndexedDB
  useEffect(() => {
    openDatabase()
      .then((db) => {
        dbRef.current = db;
        setState((s) => ({ ...s, isIndexedDBReady: true }));
        updateStats();
      })
      .catch((error) => {
        console.error('[OfflineStorage] IndexedDB init failed:', error);
      });

    return () => {
      if (dbRef.current) {
        dbRef.current.close();
      }
    };
  }, []);

  // Monitor online status
  useEffect(() => {
    const handleOnline = () => setState((s) => ({ ...s, isOnline: true }));
    const handleOffline = () => setState((s) => ({ ...s, isOnline: false }));

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Check service worker
  useEffect(() => {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.ready.then(() => {
        setState((s) => ({ ...s, isServiceWorkerReady: true }));
      });

      // Listen for messages from service worker
      navigator.serviceWorker.addEventListener('message', (event) => {
        if (event.data.type === 'AUDIO_CACHED') {
          updateStats();
        }
      });
    }
  }, []);

  // Update stats
  const updateStats = useCallback(async () => {
    if (!dbRef.current) return;

    try {
      // Get cached tracks count
      const tx = dbRef.current.transaction(STORES.tracks, 'readonly');
      const store = tx.objectStore(STORES.tracks);
      const countRequest = store.count();

      countRequest.onsuccess = () => {
        setState((s) => ({ ...s, cachedTracksCount: countRequest.result }));
      };

      // Get sync queue count
      const syncTx = dbRef.current.transaction(STORES.syncQueue, 'readonly');
      const syncStore = syncTx.objectStore(STORES.syncQueue);
      const syncCountRequest = syncStore.count();

      syncCountRequest.onsuccess = () => {
        setState((s) => ({ ...s, syncQueueCount: syncCountRequest.result }));
      };

      // Get cache size from service worker
      if (state.isServiceWorkerReady) {
        try {
          const status = (await sendToServiceWorker({ type: 'GET_CACHE_STATUS' })) as {
            audioCacheSize: number;
          };
          setState((s) => ({ ...s, cacheSize: status.audioCacheSize }));
        } catch (e) {
          // Service worker not responding
        }
      }
    } catch (error) {
      console.error('[OfflineStorage] Failed to update stats:', error);
    }
  }, [state.isServiceWorkerReady]);

  // === Track Caching ===

  const cacheTrack = useCallback(async (track: CachedTrack) => {
    if (!dbRef.current) throw new Error('Database not ready');

    // Store metadata in IndexedDB
    const tx = dbRef.current.transaction(STORES.tracks, 'readwrite');
    const store = tx.objectStore(STORES.tracks);
    await new Promise<void>((resolve, reject) => {
      const request = store.put(track);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });

    // Cache audio file via service worker
    if (state.isServiceWorkerReady) {
      await sendToServiceWorker({
        type: 'CACHE_AUDIO',
        payload: { url: track.audioUrl, trackId: track.id },
      });
    }

    updateStats();
  }, [state.isServiceWorkerReady, updateStats]);

  const uncacheTrack = useCallback(async (trackId: string) => {
    if (!dbRef.current) throw new Error('Database not ready');

    // Get track info first
    const track = await getCachedTrack(trackId);

    // Remove from IndexedDB
    const tx = dbRef.current.transaction(STORES.tracks, 'readwrite');
    const store = tx.objectStore(STORES.tracks);
    await new Promise<void>((resolve, reject) => {
      const request = store.delete(trackId);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });

    // Remove from service worker cache
    if (track && state.isServiceWorkerReady) {
      await sendToServiceWorker({
        type: 'UNCACHE_AUDIO',
        payload: { url: track.audioUrl },
      });
    }

    updateStats();
  }, [state.isServiceWorkerReady, updateStats]);

  const getCachedTrack = useCallback(async (trackId: string): Promise<CachedTrack | undefined> => {
    if (!dbRef.current) return undefined;

    const tx = dbRef.current.transaction(STORES.tracks, 'readonly');
    const store = tx.objectStore(STORES.tracks);

    return new Promise((resolve, reject) => {
      const request = store.get(trackId);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }, []);

  const getAllCachedTracks = useCallback(async (): Promise<CachedTrack[]> => {
    if (!dbRef.current) return [];

    const tx = dbRef.current.transaction(STORES.tracks, 'readonly');
    const store = tx.objectStore(STORES.tracks);

    return new Promise((resolve, reject) => {
      const request = store.getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }, []);

  const isTrackCached = useCallback(async (trackId: string): Promise<boolean> => {
    const track = await getCachedTrack(trackId);
    return !!track;
  }, [getCachedTrack]);

  // === Favorites ===

  const addFavorite = useCallback(async (trackId: string) => {
    if (!dbRef.current) throw new Error('Database not ready');

    const tx = dbRef.current.transaction(STORES.favorites, 'readwrite');
    const store = tx.objectStore(STORES.favorites);

    await new Promise<void>((resolve, reject) => {
      const request = store.put({ trackId, addedAt: Date.now() });
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });

    // Queue sync
    await queueSync('favorite', { action: 'add', trackId });
  }, []);

  const removeFavorite = useCallback(async (trackId: string) => {
    if (!dbRef.current) throw new Error('Database not ready');

    const tx = dbRef.current.transaction(STORES.favorites, 'readwrite');
    const store = tx.objectStore(STORES.favorites);

    await new Promise<void>((resolve, reject) => {
      const request = store.delete(trackId);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });

    await queueSync('favorite', { action: 'remove', trackId });
  }, []);

  const isFavorite = useCallback(async (trackId: string): Promise<boolean> => {
    if (!dbRef.current) return false;

    const tx = dbRef.current.transaction(STORES.favorites, 'readonly');
    const store = tx.objectStore(STORES.favorites);

    return new Promise((resolve, reject) => {
      const request = store.get(trackId);
      request.onsuccess = () => resolve(!!request.result);
      request.onerror = () => reject(request.error);
    });
  }, []);

  const getAllFavorites = useCallback(async (): Promise<string[]> => {
    if (!dbRef.current) return [];

    const tx = dbRef.current.transaction(STORES.favorites, 'readonly');
    const store = tx.objectStore(STORES.favorites);

    return new Promise((resolve, reject) => {
      const request = store.getAll();
      request.onsuccess = () => resolve(request.result.map((f: { trackId: string }) => f.trackId));
      request.onerror = () => reject(request.error);
    });
  }, []);

  // === History ===

  const addToHistory = useCallback(async (trackId: string, duration: number) => {
    if (!dbRef.current) throw new Error('Database not ready');

    const tx = dbRef.current.transaction(STORES.history, 'readwrite');
    const store = tx.objectStore(STORES.history);

    await new Promise<void>((resolve, reject) => {
      const request = store.add({
        trackId,
        playedAt: Date.now(),
        duration,
      });
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });

    await queueSync('play-history', { trackId, playedAt: Date.now(), duration });
  }, []);

  const getRecentHistory = useCallback(
    async (limit: number = 50): Promise<Array<{ trackId: string; playedAt: number }>> => {
      if (!dbRef.current) return [];

      const tx = dbRef.current.transaction(STORES.history, 'readonly');
      const store = tx.objectStore(STORES.history);
      const index = store.index('playedAt');

      return new Promise((resolve, reject) => {
        const results: Array<{ trackId: string; playedAt: number }> = [];
        const request = index.openCursor(null, 'prev');

        request.onsuccess = () => {
          const cursor = request.result;
          if (cursor && results.length < limit) {
            results.push({
              trackId: cursor.value.trackId,
              playedAt: cursor.value.playedAt,
            });
            cursor.continue();
          } else {
            resolve(results);
          }
        };
        request.onerror = () => reject(request.error);
      });
    },
    []
  );

  // === Sync Queue ===

  const queueSync = useCallback(async (type: SyncQueueItem['type'], data: Record<string, unknown>) => {
    if (!dbRef.current) return;

    const tx = dbRef.current.transaction(STORES.syncQueue, 'readwrite');
    const store = tx.objectStore(STORES.syncQueue);

    await new Promise<void>((resolve, reject) => {
      const request = store.add({
        type,
        data,
        timestamp: Date.now(),
        retries: 0,
      });
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });

    updateStats();

    // Trigger background sync if available
    if ('serviceWorker' in navigator && 'sync' in (navigator.serviceWorker as unknown as { sync?: unknown })) {
      const registration = await navigator.serviceWorker.ready;
      await (registration as unknown as { sync: { register: (tag: string) => Promise<void> } }).sync.register(`sync-${type}`);
    }
  }, [updateStats]);

  const processSyncQueue = useCallback(async () => {
    if (!dbRef.current || !state.isOnline) return;

    const tx = dbRef.current.transaction(STORES.syncQueue, 'readwrite');
    const store = tx.objectStore(STORES.syncQueue);

    const items = await new Promise<SyncQueueItem[]>((resolve, reject) => {
      const request = store.getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });

    for (const item of items) {
      try {
        // Process based on type
        const endpoint = `/api/${item.type.replace('-', '/')}`;
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(item.data),
        });

        if (response.ok) {
          // Remove from queue
          const deleteTx = dbRef.current!.transaction(STORES.syncQueue, 'readwrite');
          const deleteStore = deleteTx.objectStore(STORES.syncQueue);
          deleteStore.delete(item.id);
        }
      } catch (error) {
        console.error('[OfflineStorage] Sync failed for item:', item.id, error);
      }
    }

    setState((s) => ({ ...s, lastSyncTime: Date.now() }));
    updateStats();
  }, [state.isOnline, updateStats]);

  const getSyncQueueCount = useCallback(async (): Promise<number> => {
    if (!dbRef.current) return 0;

    const tx = dbRef.current.transaction(STORES.syncQueue, 'readonly');
    const store = tx.objectStore(STORES.syncQueue);

    return new Promise((resolve, reject) => {
      const request = store.count();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }, []);

  // === Cache Management ===

  const getCacheSize = useCallback(async (): Promise<number> => {
    if (!state.isServiceWorkerReady) return 0;

    try {
      const status = (await sendToServiceWorker({ type: 'GET_CACHE_STATUS' })) as {
        audioCacheSize: number;
      };
      return status.audioCacheSize;
    } catch {
      return 0;
    }
  }, [state.isServiceWorkerReady]);

  const clearAllCaches = useCallback(async () => {
    // Clear IndexedDB stores
    if (dbRef.current) {
      const stores = [STORES.tracks, STORES.history];
      for (const storeName of stores) {
        const tx = dbRef.current.transaction(storeName, 'readwrite');
        const store = tx.objectStore(storeName);
        await new Promise<void>((resolve) => {
          const request = store.clear();
          request.onsuccess = () => resolve();
          request.onerror = () => resolve();
        });
      }
    }

    // Clear service worker cache
    if (state.isServiceWorkerReady) {
      await sendToServiceWorker({ type: 'CLEAR_AUDIO_CACHE' });
    }

    updateStats();
  }, [state.isServiceWorkerReady, updateStats]);

  const clearOldCache = useCallback(async (maxAgeDays: number = 30): Promise<number> => {
    if (!dbRef.current) return 0;

    const maxAge = Date.now() - maxAgeDays * 24 * 60 * 60 * 1000;
    let cleared = 0;

    const tx = dbRef.current.transaction(STORES.tracks, 'readwrite');
    const store = tx.objectStore(STORES.tracks);
    const index = store.index('cachedAt');

    const range = IDBKeyRange.upperBound(maxAge);

    await new Promise<void>((resolve) => {
      const request = index.openCursor(range);
      request.onsuccess = () => {
        const cursor = request.result;
        if (cursor) {
          cursor.delete();
          cleared++;
          cursor.continue();
        } else {
          resolve();
        }
      };
      request.onerror = () => resolve();
    });

    updateStats();
    return cleared;
  }, [updateStats]);

  // === Settings ===

  const setSetting = useCallback(async (key: string, value: unknown) => {
    if (!dbRef.current) throw new Error('Database not ready');

    const tx = dbRef.current.transaction(STORES.settings, 'readwrite');
    const store = tx.objectStore(STORES.settings);

    await new Promise<void>((resolve, reject) => {
      const request = store.put({ key, value });
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }, []);

  const getSetting = useCallback(async <T>(key: string, defaultValue: T): Promise<T> => {
    if (!dbRef.current) return defaultValue;

    const tx = dbRef.current.transaction(STORES.settings, 'readonly');
    const store = tx.objectStore(STORES.settings);

    return new Promise((resolve, reject) => {
      const request = store.get(key);
      request.onsuccess = () => resolve(request.result?.value ?? defaultValue);
      request.onerror = () => reject(request.error);
    });
  }, []);

  return {
    ...state,
    cacheTrack,
    uncacheTrack,
    getCachedTrack,
    getAllCachedTracks,
    isTrackCached,
    addFavorite,
    removeFavorite,
    isFavorite,
    getAllFavorites,
    addToHistory,
    getRecentHistory,
    queueSync,
    processSyncQueue,
    getSyncQueueCount,
    getCacheSize,
    clearAllCaches,
    clearOldCache,
    setSetting,
    getSetting,
  };
}

export default useOfflineStorage;
