// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Offline Support Hook for Mycelix Mail PWA
 *
 * Provides offline detection, data persistence, and sync capabilities
 */

import { useState, useEffect, useCallback, useRef } from 'react';

interface OfflineState {
  isOnline: boolean;
  isServiceWorkerReady: boolean;
  pendingActions: number;
  lastSyncTime: Date | null;
}

interface PendingAction {
  id: string;
  type: 'send' | 'draft' | 'delete' | 'read' | 'star';
  data: unknown;
  timestamp: Date;
}

interface UseOfflineReturn extends OfflineState {
  queueAction: (action: Omit<PendingAction, 'id' | 'timestamp'>) => Promise<void>;
  syncNow: () => Promise<void>;
  getCachedEmails: () => Promise<CachedEmail[]>;
  cacheEmail: (email: CachedEmail) => Promise<void>;
  clearCache: () => Promise<void>;
}

interface CachedEmail {
  id: string;
  subject: string;
  from: string;
  body: string;
  date: string;
  folder: string;
  isRead: boolean;
  isStarred: boolean;
}

const DB_NAME = 'mycelix-offline';
const DB_VERSION = 1;

// Open IndexedDB
function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;

      if (!db.objectStoreNames.contains('pending-sends')) {
        db.createObjectStore('pending-sends', { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains('pending-drafts')) {
        db.createObjectStore('pending-drafts', { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains('pending-actions')) {
        db.createObjectStore('pending-actions', { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains('cached-emails')) {
        const store = db.createObjectStore('cached-emails', { keyPath: 'id' });
        store.createIndex('folder', 'folder', { unique: false });
        store.createIndex('date', 'date', { unique: false });
      }
    };
  });
}

export function useOffline(): UseOfflineReturn {
  const [state, setState] = useState<OfflineState>({
    isOnline: typeof navigator !== 'undefined' ? navigator.onLine : true,
    isServiceWorkerReady: false,
    pendingActions: 0,
    lastSyncTime: null,
  });

  const dbRef = useRef<IDBDatabase | null>(null);

  // Initialize database
  useEffect(() => {
    openDB()
      .then((db) => {
        dbRef.current = db;
        updatePendingCount();
      })
      .catch(console.error);

    return () => {
      dbRef.current?.close();
    };
  }, []);

  // Online/offline detection
  useEffect(() => {
    const handleOnline = () => {
      setState((prev) => ({ ...prev, isOnline: true }));
      // Trigger sync when coming back online
      syncNow();
    };

    const handleOffline = () => {
      setState((prev) => ({ ...prev, isOnline: false }));
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Service worker registration
  useEffect(() => {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.ready.then(() => {
        setState((prev) => ({ ...prev, isServiceWorkerReady: true }));
      });

      // Listen for sync completion messages
      navigator.serviceWorker.addEventListener('message', (event) => {
        if (event.data.type === 'EMAIL_SYNCED') {
          updatePendingCount();
          setState((prev) => ({ ...prev, lastSyncTime: new Date() }));
        }
      });
    }
  }, []);

  // Update pending actions count
  const updatePendingCount = useCallback(async () => {
    if (!dbRef.current) return;

    try {
      const tx = dbRef.current.transaction(['pending-actions'], 'readonly');
      const store = tx.objectStore('pending-actions');
      const request = store.count();

      request.onsuccess = () => {
        setState((prev) => ({ ...prev, pendingActions: request.result }));
      };
    } catch (error) {
      console.error('Failed to count pending actions:', error);
    }
  }, []);

  // Queue an action for offline processing
  const queueAction = useCallback(async (action: Omit<PendingAction, 'id' | 'timestamp'>) => {
    if (!dbRef.current) {
      throw new Error('Database not initialized');
    }

    const pendingAction: PendingAction = {
      ...action,
      id: crypto.randomUUID(),
      timestamp: new Date(),
    };

    return new Promise<void>((resolve, reject) => {
      const tx = dbRef.current!.transaction(['pending-actions'], 'readwrite');
      const store = tx.objectStore('pending-actions');
      const request = store.add(pendingAction);

      request.onsuccess = () => {
        updatePendingCount();

        // Register background sync if available
        if ('serviceWorker' in navigator && 'sync' in self.registration) {
          navigator.serviceWorker.ready.then((registration) => {
            registration.sync.register('sync-emails');
          });
        }

        resolve();
      };

      request.onerror = () => reject(request.error);
    });
  }, [updatePendingCount]);

  // Manual sync trigger
  const syncNow = useCallback(async () => {
    if (!state.isOnline || !dbRef.current) return;

    const tx = dbRef.current.transaction(['pending-actions'], 'readonly');
    const store = tx.objectStore('pending-actions');

    return new Promise<void>((resolve, reject) => {
      const request = store.getAll();

      request.onsuccess = async () => {
        const actions = request.result as PendingAction[];

        for (const action of actions) {
          try {
            await processAction(action);
            await removeAction(action.id);
          } catch (error) {
            console.error('Failed to sync action:', action.id, error);
          }
        }

        await updatePendingCount();
        setState((prev) => ({ ...prev, lastSyncTime: new Date() }));
        resolve();
      };

      request.onerror = () => reject(request.error);
    });
  }, [state.isOnline, updatePendingCount]);

  // Process a single action
  async function processAction(action: PendingAction): Promise<void> {
    const endpoints: Record<string, { method: string; url: string }> = {
      send: { method: 'POST', url: '/api/v1/emails/send' },
      draft: { method: 'POST', url: '/api/v1/drafts' },
      delete: { method: 'DELETE', url: '/api/v1/emails' },
      read: { method: 'PATCH', url: '/api/v1/emails/read' },
      star: { method: 'PATCH', url: '/api/v1/emails/star' },
    };

    const endpoint = endpoints[action.type];
    if (!endpoint) throw new Error(`Unknown action type: ${action.type}`);

    const response = await fetch(endpoint.url, {
      method: endpoint.method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(action.data),
    });

    if (!response.ok) {
      throw new Error(`Sync failed: ${response.status}`);
    }
  }

  // Remove processed action
  async function removeAction(id: string): Promise<void> {
    if (!dbRef.current) return;

    return new Promise((resolve, reject) => {
      const tx = dbRef.current!.transaction(['pending-actions'], 'readwrite');
      const store = tx.objectStore('pending-actions');
      const request = store.delete(id);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  // Cache an email for offline viewing
  const cacheEmail = useCallback(async (email: CachedEmail) => {
    if (!dbRef.current) return;

    return new Promise<void>((resolve, reject) => {
      const tx = dbRef.current!.transaction(['cached-emails'], 'readwrite');
      const store = tx.objectStore('cached-emails');
      const request = store.put(email);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }, []);

  // Get cached emails
  const getCachedEmails = useCallback(async (): Promise<CachedEmail[]> => {
    if (!dbRef.current) return [];

    return new Promise((resolve, reject) => {
      const tx = dbRef.current!.transaction(['cached-emails'], 'readonly');
      const store = tx.objectStore('cached-emails');
      const request = store.getAll();

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }, []);

  // Clear all cached data
  const clearCache = useCallback(async () => {
    if (!dbRef.current) return;

    const stores = ['cached-emails', 'pending-actions', 'pending-sends', 'pending-drafts'];

    for (const storeName of stores) {
      await new Promise<void>((resolve, reject) => {
        const tx = dbRef.current!.transaction([storeName], 'readwrite');
        const store = tx.objectStore(storeName);
        const request = store.clear();

        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      });
    }

    updatePendingCount();
  }, [updatePendingCount]);

  return {
    ...state,
    queueAction,
    syncNow,
    getCachedEmails,
    cacheEmail,
    clearCache,
  };
}

// Hook for showing offline indicator
export function useOnlineStatus(): boolean {
  const [isOnline, setIsOnline] = useState(
    typeof navigator !== 'undefined' ? navigator.onLine : true
  );

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return isOnline;
}

// Hook for service worker updates
export function useServiceWorkerUpdate(): {
  updateAvailable: boolean;
  updateApp: () => void;
} {
  const [updateAvailable, setUpdateAvailable] = useState(false);
  const [waitingWorker, setWaitingWorker] = useState<ServiceWorker | null>(null);

  useEffect(() => {
    if (!('serviceWorker' in navigator)) return;

    navigator.serviceWorker.register('/sw.js').then((registration) => {
      registration.addEventListener('updatefound', () => {
        const newWorker = registration.installing;
        if (!newWorker) return;

        newWorker.addEventListener('statechange', () => {
          if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
            setUpdateAvailable(true);
            setWaitingWorker(newWorker);
          }
        });
      });
    });
  }, []);

  const updateApp = useCallback(() => {
    if (waitingWorker) {
      waitingWorker.postMessage({ type: 'SKIP_WAITING' });
      window.location.reload();
    }
  }, [waitingWorker]);

  return { updateAvailable, updateApp };
}
