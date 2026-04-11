// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Service Worker Registration and Management
 *
 * PWA functionality for Mycelix Mail:
 * - Offline support with caching
 * - Background sync for drafts
 * - Push notifications
 * - Periodic trust network sync
 */

// ============================================
// Types
// ============================================

interface ServiceWorkerConfig {
  onSuccess?: (registration: ServiceWorkerRegistration) => void;
  onUpdate?: (registration: ServiceWorkerRegistration) => void;
  onOffline?: () => void;
  onOnline?: () => void;
}

interface CacheConfig {
  name: string;
  maxAge: number;
  maxEntries: number;
}

// ============================================
// Service Worker Registration
// ============================================

export function register(config?: ServiceWorkerConfig): void {
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      const swUrl = '/sw.js';

      registerValidSW(swUrl, config);

      // Handle online/offline events
      window.addEventListener('online', () => {
        config?.onOnline?.();
        console.log('[SW] App is online');
      });

      window.addEventListener('offline', () => {
        config?.onOffline?.();
        console.log('[SW] App is offline');
      });
    });
  }
}

function registerValidSW(swUrl: string, config?: ServiceWorkerConfig): void {
  navigator.serviceWorker
    .register(swUrl)
    .then((registration) => {
      console.log('[SW] Service Worker registered:', registration.scope);

      registration.onupdatefound = () => {
        const installingWorker = registration.installing;
        if (installingWorker == null) return;

        installingWorker.onstatechange = () => {
          if (installingWorker.state === 'installed') {
            if (navigator.serviceWorker.controller) {
              // New content available
              console.log('[SW] New content available');
              config?.onUpdate?.(registration);
            } else {
              // Content cached for offline use
              console.log('[SW] Content cached for offline use');
              config?.onSuccess?.(registration);
            }
          }
        };
      };
    })
    .catch((error) => {
      console.error('[SW] Error during service worker registration:', error);
    });
}

export function unregister(): void {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.ready
      .then((registration) => {
        registration.unregister();
      })
      .catch((error) => {
        console.error('[SW] Error unregistering:', error);
      });
  }
}

// ============================================
// Service Worker File (sw.js content)
// ============================================

export const serviceWorkerScript = `
const CACHE_NAME = 'mycelix-mail-v1';
const RUNTIME_CACHE = 'mycelix-runtime-v1';
const DATA_CACHE = 'mycelix-data-v1';

// Static assets to cache on install
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
];

// API routes to cache with network-first strategy
const API_ROUTES = [
  '/api/emails',
  '/api/trust',
  '/api/credentials',
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[SW] Caching static assets');
      return cache.addAll(STATIC_ASSETS);
    })
  );
  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME && name !== RUNTIME_CACHE && name !== DATA_CACHE)
          .map((name) => caches.delete(name))
      );
    })
  );
  self.clients.claim();
});

// Fetch event - serve from cache or network
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // API requests - network first, fallback to cache
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirstStrategy(request));
    return;
  }

  // Static assets - cache first
  event.respondWith(cacheFirstStrategy(request));
});

// Network first strategy for API calls
async function networkFirstStrategy(request) {
  try {
    const response = await fetch(request);

    // Cache successful GET responses
    if (response.ok && request.method === 'GET') {
      const cache = await caches.open(DATA_CACHE);
      cache.put(request, response.clone());
    }

    return response;
  } catch (error) {
    // Fallback to cache
    const cached = await caches.match(request);
    if (cached) {
      return cached;
    }

    // Return offline response
    return new Response(JSON.stringify({ error: 'Offline' }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}

// Cache first strategy for static assets
async function cacheFirstStrategy(request) {
  const cached = await caches.match(request);
  if (cached) {
    return cached;
  }

  try {
    const response = await fetch(request);

    // Cache successful responses
    if (response.ok) {
      const cache = await caches.open(RUNTIME_CACHE);
      cache.put(request, response.clone());
    }

    return response;
  } catch (error) {
    // Return offline page for navigation requests
    if (request.mode === 'navigate') {
      return caches.match('/');
    }
    throw error;
  }
}

// Background sync for drafts
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-drafts') {
    event.waitUntil(syncDrafts());
  }
  if (event.tag === 'sync-attestations') {
    event.waitUntil(syncAttestations());
  }
});

async function syncDrafts() {
  const drafts = await getDraftsFromIndexedDB();
  for (const draft of drafts) {
    try {
      await fetch('/api/drafts', {
        method: 'POST',
        body: JSON.stringify(draft),
        headers: { 'Content-Type': 'application/json' },
      });
      await removeDraftFromIndexedDB(draft.id);
    } catch (error) {
      console.error('[SW] Failed to sync draft:', error);
    }
  }
}

async function syncAttestations() {
  const pending = await getPendingAttestationsFromIndexedDB();
  for (const attestation of pending) {
    try {
      await fetch('/api/attestations', {
        method: 'POST',
        body: JSON.stringify(attestation),
        headers: { 'Content-Type': 'application/json' },
      });
      await removeFromIndexedDB('pending-attestations', attestation.id);
    } catch (error) {
      console.error('[SW] Failed to sync attestation:', error);
    }
  }
}

// Push notifications
self.addEventListener('push', (event) => {
  const data = event.data?.json() || {};

  const options = {
    body: data.body || 'New notification',
    icon: '/icons/icon-192.png',
    badge: '/icons/badge.png',
    tag: data.tag || 'default',
    data: data,
    actions: data.actions || [],
    requireInteraction: data.requireInteraction || false,
  };

  event.waitUntil(
    self.registration.showNotification(data.title || 'Mycelix Mail', options)
  );
});

// Notification click handling
self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  const data = event.notification.data;
  let url = '/';

  if (data.type === 'email') {
    url = '/email/' + data.emailId;
  } else if (data.type === 'attestation') {
    url = '/trust';
  } else if (data.type === 'introduction') {
    url = '/trust/introductions';
  }

  event.waitUntil(
    clients.matchAll({ type: 'window' }).then((clientList) => {
      for (const client of clientList) {
        if (client.url === url && 'focus' in client) {
          return client.focus();
        }
      }
      if (clients.openWindow) {
        return clients.openWindow(url);
      }
    })
  );
});

// Periodic sync for trust network updates
self.addEventListener('periodicsync', (event) => {
  if (event.tag === 'trust-network-sync') {
    event.waitUntil(syncTrustNetwork());
  }
});

async function syncTrustNetwork() {
  try {
    const response = await fetch('/api/trust/sync');
    const data = await response.json();

    // Store in IndexedDB for offline access
    await storeTrustDataInIndexedDB(data);

    // Notify clients
    const clients = await self.clients.matchAll();
    clients.forEach((client) => {
      client.postMessage({
        type: 'TRUST_NETWORK_SYNCED',
        data: data,
      });
    });
  } catch (error) {
    console.error('[SW] Trust network sync failed:', error);
  }
}

// IndexedDB helpers (simplified)
function getDraftsFromIndexedDB() {
  return Promise.resolve([]);
}

function removeDraftFromIndexedDB(id) {
  return Promise.resolve();
}

function getPendingAttestationsFromIndexedDB() {
  return Promise.resolve([]);
}

function removeFromIndexedDB(store, id) {
  return Promise.resolve();
}

function storeTrustDataInIndexedDB(data) {
  return Promise.resolve();
}
`;

// ============================================
// Push Notification Support
// ============================================

export async function subscribeToPush(): Promise<PushSubscription | null> {
  if (!('PushManager' in window)) {
    console.warn('[SW] Push notifications not supported');
    return null;
  }

  try {
    const registration = await navigator.serviceWorker.ready;

    // Get existing subscription or create new one
    let subscription = await registration.pushManager.getSubscription();

    if (!subscription) {
      // Get VAPID public key from server
      const response = await fetch('/api/push/vapid-key');
      const { publicKey } = await response.json();

      subscription = await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: urlBase64ToUint8Array(publicKey),
      });

      // Send subscription to server
      await fetch('/api/push/subscribe', {
        method: 'POST',
        body: JSON.stringify(subscription),
        headers: { 'Content-Type': 'application/json' },
      });
    }

    return subscription;
  } catch (error) {
    console.error('[SW] Failed to subscribe to push:', error);
    return null;
  }
}

export async function unsubscribeFromPush(): Promise<boolean> {
  try {
    const registration = await navigator.serviceWorker.ready;
    const subscription = await registration.pushManager.getSubscription();

    if (subscription) {
      await subscription.unsubscribe();

      // Notify server
      await fetch('/api/push/unsubscribe', {
        method: 'POST',
        body: JSON.stringify({ endpoint: subscription.endpoint }),
        headers: { 'Content-Type': 'application/json' },
      });
    }

    return true;
  } catch (error) {
    console.error('[SW] Failed to unsubscribe from push:', error);
    return false;
  }
}

function urlBase64ToUint8Array(base64String: string): Uint8Array {
  const padding = '='.repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
  const rawData = window.atob(base64);
  const outputArray = new Uint8Array(rawData.length);

  for (let i = 0; i < rawData.length; ++i) {
    outputArray[i] = rawData.charCodeAt(i);
  }
  return outputArray;
}

// ============================================
// Background Sync Registration
// ============================================

export async function registerBackgroundSync(tag: string): Promise<void> {
  if (!('serviceWorker' in navigator) || !('SyncManager' in window)) {
    console.warn('[SW] Background sync not supported');
    return;
  }

  try {
    const registration = await navigator.serviceWorker.ready;
    await (registration as any).sync.register(tag);
    console.log('[SW] Background sync registered:', tag);
  } catch (error) {
    console.error('[SW] Failed to register background sync:', error);
  }
}

// ============================================
// Offline Detection Hook
// ============================================

export function useOfflineStatus(): boolean {
  const [isOffline, setIsOffline] = useState(!navigator.onLine);

  useEffect(() => {
    const handleOnline = () => setIsOffline(false);
    const handleOffline = () => setIsOffline(true);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return isOffline;
}

import { useState, useEffect } from 'react';

export default {
  register,
  unregister,
  subscribeToPush,
  unsubscribeFromPush,
  registerBackgroundSync,
  useOfflineStatus,
  serviceWorkerScript,
};
