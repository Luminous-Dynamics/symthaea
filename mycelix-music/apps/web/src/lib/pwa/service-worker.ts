// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Service Worker
 *
 * Offline-first PWA capabilities:
 * - Cache strategies
 * - Background sync
 * - Push notifications
 * - Offline audio playback
 */

/// <reference lib="webworker" />

declare const self: ServiceWorkerGlobalScope;

const CACHE_VERSION = 'v1';
const STATIC_CACHE = `static-${CACHE_VERSION}`;
const DYNAMIC_CACHE = `dynamic-${CACHE_VERSION}`;
const AUDIO_CACHE = `audio-${CACHE_VERSION}`;
const IMAGE_CACHE = `image-${CACHE_VERSION}`;

const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/offline.html',
  '/app.js',
  '/app.css',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
];

const CACHE_LIMITS = {
  dynamic: 50,
  audio: 100,
  image: 200,
};

// ==================== Install ====================

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(STATIC_CACHE).then((cache) => {
      return cache.addAll(STATIC_ASSETS);
    })
  );
  self.skipWaiting();
});

// ==================== Activate ====================

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys
          .filter((key) => {
            return key.startsWith('static-') || key.startsWith('dynamic-') ||
                   key.startsWith('audio-') || key.startsWith('image-');
          })
          .filter((key) => {
            return !key.includes(CACHE_VERSION);
          })
          .map((key) => caches.delete(key))
      );
    })
  );
  self.clients.claim();
});

// ==================== Fetch Strategies ====================

async function cacheFirst(request: Request, cacheName: string): Promise<Response> {
  const cached = await caches.match(request);
  if (cached) return cached;

  const response = await fetch(request);
  if (response.ok) {
    const cache = await caches.open(cacheName);
    cache.put(request, response.clone());
  }
  return response;
}

async function networkFirst(request: Request, cacheName: string): Promise<Response> {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(cacheName);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = await caches.match(request);
    if (cached) return cached;
    throw new Error('No cached response');
  }
}

async function staleWhileRevalidate(request: Request, cacheName: string): Promise<Response> {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);

  const fetchPromise = fetch(request).then((response) => {
    if (response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  });

  return cached || fetchPromise;
}

// ==================== Fetch Handler ====================

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Skip non-GET requests
  if (event.request.method !== 'GET') return;

  // API requests - network first
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      networkFirst(event.request, DYNAMIC_CACHE).catch(() => {
        return new Response(JSON.stringify({ error: 'Offline' }), {
          headers: { 'Content-Type': 'application/json' },
        });
      })
    );
    return;
  }

  // Audio files - cache first with range support
  if (url.pathname.match(/\.(mp3|m4a|ogg|flac|wav)$/)) {
    event.respondWith(handleAudioRequest(event.request));
    return;
  }

  // Images - cache first
  if (url.pathname.match(/\.(jpg|jpeg|png|gif|webp|svg)$/)) {
    event.respondWith(cacheFirst(event.request, IMAGE_CACHE));
    return;
  }

  // Static assets - cache first
  if (STATIC_ASSETS.includes(url.pathname)) {
    event.respondWith(cacheFirst(event.request, STATIC_CACHE));
    return;
  }

  // Everything else - stale while revalidate
  event.respondWith(
    staleWhileRevalidate(event.request, DYNAMIC_CACHE).catch(() => {
      if (event.request.destination === 'document') {
        return caches.match('/offline.html')!;
      }
      throw new Error('Offline');
    })
  );
});

// ==================== Audio Handling ====================

async function handleAudioRequest(request: Request): Promise<Response> {
  // Check cache first
  const cached = await caches.match(request);
  if (cached) return cached;

  // Handle range requests
  const rangeHeader = request.headers.get('range');
  if (rangeHeader) {
    return handleRangeRequest(request, rangeHeader);
  }

  // Fetch and cache
  const response = await fetch(request);
  if (response.ok) {
    const cache = await caches.open(AUDIO_CACHE);
    cache.put(request, response.clone());
    await trimCache(AUDIO_CACHE, CACHE_LIMITS.audio);
  }
  return response;
}

async function handleRangeRequest(request: Request, rangeHeader: string): Promise<Response> {
  const cache = await caches.open(AUDIO_CACHE);
  const cached = await cache.match(request.url);

  if (cached) {
    const blob = await cached.blob();
    const [, start, end] = rangeHeader.match(/bytes=(\d+)-(\d*)/) || [];

    const startByte = parseInt(start) || 0;
    const endByte = parseInt(end) || blob.size - 1;

    const slice = blob.slice(startByte, endByte + 1);

    return new Response(slice, {
      status: 206,
      headers: {
        'Content-Range': `bytes ${startByte}-${endByte}/${blob.size}`,
        'Content-Length': slice.size.toString(),
        'Content-Type': cached.headers.get('Content-Type') || 'audio/mpeg',
      },
    });
  }

  // Fetch with range
  return fetch(request);
}

// ==================== Cache Management ====================

async function trimCache(cacheName: string, maxItems: number): Promise<void> {
  const cache = await caches.open(cacheName);
  const keys = await cache.keys();

  if (keys.length > maxItems) {
    const toDelete = keys.slice(0, keys.length - maxItems);
    await Promise.all(toDelete.map((key) => cache.delete(key)));
  }
}

// ==================== Background Sync ====================

self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-plays') {
    event.waitUntil(syncPlayHistory());
  }
  if (event.tag === 'sync-likes') {
    event.waitUntil(syncLikes());
  }
  if (event.tag === 'sync-queue') {
    event.waitUntil(syncQueue());
  }
});

async function syncPlayHistory(): Promise<void> {
  const db = await openDB();
  const plays = await getFromStore(db, 'pending-plays');

  for (const play of plays) {
    try {
      await fetch('/api/plays', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(play),
      });
      await deleteFromStore(db, 'pending-plays', play.id);
    } catch {
      // Will retry on next sync
    }
  }
}

async function syncLikes(): Promise<void> {
  const db = await openDB();
  const likes = await getFromStore(db, 'pending-likes');

  for (const like of likes) {
    try {
      await fetch(`/api/tracks/${like.trackId}/like`, {
        method: like.action === 'like' ? 'POST' : 'DELETE',
      });
      await deleteFromStore(db, 'pending-likes', like.id);
    } catch {
      // Will retry
    }
  }
}

async function syncQueue(): Promise<void> {
  const db = await openDB();
  const queue = await getFromStore(db, 'queue');

  try {
    await fetch('/api/queue/sync', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(queue),
    });
  } catch {
    // Will retry
  }
}

// ==================== Push Notifications ====================

self.addEventListener('push', (event) => {
  if (!event.data) return;

  const data = event.data.json();

  const options: NotificationOptions = {
    body: data.body,
    icon: data.icon || '/icons/icon-192.png',
    badge: '/icons/badge.png',
    image: data.image,
    data: data.data,
    actions: data.actions,
    tag: data.tag,
    renotify: data.renotify,
  };

  event.waitUntil(
    self.registration.showNotification(data.title, options)
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  const data = event.notification.data;

  if (event.action === 'play') {
    event.waitUntil(
      clients.openWindow(`/play/${data.trackId}`)
    );
  } else if (event.action === 'view') {
    event.waitUntil(
      clients.openWindow(data.url || '/')
    );
  } else {
    event.waitUntil(
      clients.openWindow(data.url || '/')
    );
  }
});

// ==================== IndexedDB Helpers ====================

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('mycelix-sw', 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains('pending-plays')) {
        db.createObjectStore('pending-plays', { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains('pending-likes')) {
        db.createObjectStore('pending-likes', { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains('queue')) {
        db.createObjectStore('queue', { keyPath: 'id' });
      }
    };
  });
}

function getFromStore(db: IDBDatabase, storeName: string): Promise<any[]> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readonly');
    const store = tx.objectStore(storeName);
    const request = store.getAll();

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
  });
}

function deleteFromStore(db: IDBDatabase, storeName: string, id: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readwrite');
    const store = tx.objectStore(storeName);
    const request = store.delete(id);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}

// ==================== Message Handler ====================

self.addEventListener('message', (event) => {
  if (event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }

  if (event.data.type === 'CACHE_AUDIO') {
    event.waitUntil(
      caches.open(AUDIO_CACHE).then((cache) => {
        return cache.add(event.data.url);
      })
    );
  }

  if (event.data.type === 'CLEAR_AUDIO_CACHE') {
    event.waitUntil(caches.delete(AUDIO_CACHE));
  }
});

export {};
