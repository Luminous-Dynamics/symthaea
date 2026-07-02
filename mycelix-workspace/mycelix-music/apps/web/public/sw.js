// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Music Service Worker
 *
 * Provides offline functionality:
 * - App shell caching (critical assets)
 * - Audio file caching for offline playback
 * - Background sync for queued actions
 * - Push notifications for listening parties
 */

const CACHE_VERSION = 'v1';
const STATIC_CACHE = `mycelix-static-${CACHE_VERSION}`;
const AUDIO_CACHE = `mycelix-audio-${CACHE_VERSION}`;
const API_CACHE = `mycelix-api-${CACHE_VERSION}`;
const IMAGE_CACHE = `mycelix-images-${CACHE_VERSION}`;

// Files to cache immediately on install
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/offline.html',
  // Add critical JS/CSS bundles
];

// Cache size limits
const AUDIO_CACHE_LIMIT = 500 * 1024 * 1024; // 500MB for audio
const IMAGE_CACHE_LIMIT = 50 * 1024 * 1024;  // 50MB for images

// === Install Event ===

self.addEventListener('install', (event) => {
  console.log('[SW] Installing service worker...');

  event.waitUntil(
    caches.open(STATIC_CACHE).then((cache) => {
      console.log('[SW] Caching static assets');
      return cache.addAll(STATIC_ASSETS.filter(url => url !== '/offline.html'));
    }).then(() => {
      // Force activation
      return self.skipWaiting();
    })
  );
});

// === Activate Event ===

self.addEventListener('activate', (event) => {
  console.log('[SW] Activating service worker...');

  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          // Delete old cache versions
          if (cacheName.startsWith('mycelix-') && !cacheName.includes(CACHE_VERSION)) {
            console.log('[SW] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      // Take control of all pages immediately
      return self.clients.claim();
    })
  );
});

// === Fetch Event ===

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Skip non-GET requests
  if (event.request.method !== 'GET') {
    return;
  }

  // Skip cross-origin requests (except CDN)
  if (!url.origin.includes(self.location.origin) && !url.hostname.includes('cdn')) {
    return;
  }

  // Route to appropriate handler
  if (isAudioRequest(url)) {
    event.respondWith(handleAudioRequest(event.request));
  } else if (isImageRequest(url)) {
    event.respondWith(handleImageRequest(event.request));
  } else if (isAPIRequest(url)) {
    event.respondWith(handleAPIRequest(event.request));
  } else {
    event.respondWith(handleStaticRequest(event.request));
  }
});

// === Request Type Detection ===

function isAudioRequest(url) {
  return (
    url.pathname.endsWith('.mp3') ||
    url.pathname.endsWith('.flac') ||
    url.pathname.endsWith('.wav') ||
    url.pathname.endsWith('.ogg') ||
    url.pathname.endsWith('.m4a') ||
    url.pathname.includes('/audio/') ||
    url.pathname.includes('/stream/')
  );
}

function isImageRequest(url) {
  return (
    url.pathname.endsWith('.jpg') ||
    url.pathname.endsWith('.jpeg') ||
    url.pathname.endsWith('.png') ||
    url.pathname.endsWith('.webp') ||
    url.pathname.endsWith('.gif') ||
    url.pathname.includes('/covers/') ||
    url.pathname.includes('/avatars/')
  );
}

function isAPIRequest(url) {
  return url.pathname.startsWith('/api/');
}

// === Request Handlers ===

async function handleStaticRequest(request) {
  // Try cache first
  const cachedResponse = await caches.match(request);
  if (cachedResponse) {
    return cachedResponse;
  }

  // Network with cache update
  try {
    const networkResponse = await fetch(request);

    if (networkResponse.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, networkResponse.clone());
    }

    return networkResponse;
  } catch (error) {
    // Return offline page for navigation requests
    if (request.mode === 'navigate') {
      const offlinePage = await caches.match('/offline.html');
      if (offlinePage) {
        return offlinePage;
      }
    }

    throw error;
  }
}

async function handleAudioRequest(request) {
  const cache = await caches.open(AUDIO_CACHE);

  // Check cache first
  const cachedResponse = await cache.match(request);
  if (cachedResponse) {
    console.log('[SW] Serving audio from cache:', request.url);
    return cachedResponse;
  }

  // Fetch from network
  try {
    const networkResponse = await fetch(request);

    if (networkResponse.ok) {
      // Clone before caching
      const responseToCache = networkResponse.clone();

      // Cache in background
      cacheWithLimit(AUDIO_CACHE, request, responseToCache, AUDIO_CACHE_LIMIT);
    }

    return networkResponse;
  } catch (error) {
    console.error('[SW] Audio fetch failed:', error);
    throw error;
  }
}

async function handleImageRequest(request) {
  const cache = await caches.open(IMAGE_CACHE);

  // Cache first
  const cachedResponse = await cache.match(request);
  if (cachedResponse) {
    return cachedResponse;
  }

  // Network with cache
  try {
    const networkResponse = await fetch(request);

    if (networkResponse.ok) {
      cacheWithLimit(IMAGE_CACHE, request, networkResponse.clone(), IMAGE_CACHE_LIMIT);
    }

    return networkResponse;
  } catch (error) {
    // Return placeholder image if available
    const placeholder = await caches.match('/placeholder.png');
    if (placeholder) {
      return placeholder;
    }
    throw error;
  }
}

async function handleAPIRequest(request) {
  const url = new URL(request.url);

  // Network first for API requests
  try {
    const networkResponse = await fetch(request);

    // Cache successful GET responses for certain endpoints
    if (networkResponse.ok && shouldCacheAPI(url)) {
      const cache = await caches.open(API_CACHE);
      cache.put(request, networkResponse.clone());
    }

    return networkResponse;
  } catch (error) {
    // Fall back to cache for offline
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      console.log('[SW] Serving API from cache:', request.url);
      return cachedResponse;
    }

    // Return offline JSON response
    return new Response(
      JSON.stringify({ error: 'offline', message: 'You are currently offline' }),
      {
        status: 503,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}

function shouldCacheAPI(url) {
  // Cache these API endpoints
  const cacheableEndpoints = [
    '/api/tracks',
    '/api/playlists',
    '/api/user/library',
    '/api/user/favorites',
  ];

  return cacheableEndpoints.some((endpoint) => url.pathname.startsWith(endpoint));
}

// === Cache Management ===

async function cacheWithLimit(cacheName, request, response, maxSize) {
  const cache = await caches.open(cacheName);

  // Get current cache size
  const keys = await cache.keys();
  let totalSize = 0;

  for (const key of keys) {
    const cached = await cache.match(key);
    if (cached) {
      const blob = await cached.clone().blob();
      totalSize += blob.size;
    }
  }

  // Get new response size
  const newBlob = await response.clone().blob();
  const newSize = newBlob.size;

  // Evict old entries if needed
  if (totalSize + newSize > maxSize) {
    console.log('[SW] Cache limit reached, evicting old entries');

    // Simple FIFO eviction
    for (const key of keys) {
      await cache.delete(key);
      totalSize -= (await (await cache.match(key))?.clone().blob())?.size || 0;

      if (totalSize + newSize < maxSize * 0.9) {
        break;
      }
    }
  }

  // Add to cache
  await cache.put(request, response);
}

// === Background Sync ===

self.addEventListener('sync', (event) => {
  console.log('[SW] Background sync:', event.tag);

  if (event.tag === 'sync-play-history') {
    event.waitUntil(syncPlayHistory());
  } else if (event.tag === 'sync-favorites') {
    event.waitUntil(syncFavorites());
  } else if (event.tag === 'sync-playlists') {
    event.waitUntil(syncPlaylists());
  }
});

async function syncPlayHistory() {
  // Get queued play history from IndexedDB
  const db = await openDB();
  const tx = db.transaction('syncQueue', 'readwrite');
  const store = tx.objectStore('syncQueue');
  const items = await store.getAll();

  for (const item of items) {
    if (item.type === 'play-history') {
      try {
        await fetch('/api/history', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(item.data),
        });
        await store.delete(item.id);
      } catch (error) {
        console.error('[SW] Failed to sync play history:', error);
      }
    }
  }
}

async function syncFavorites() {
  const db = await openDB();
  const tx = db.transaction('syncQueue', 'readwrite');
  const store = tx.objectStore('syncQueue');
  const items = await store.getAll();

  for (const item of items) {
    if (item.type === 'favorite') {
      try {
        await fetch('/api/favorites', {
          method: item.data.action === 'add' ? 'POST' : 'DELETE',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(item.data),
        });
        await store.delete(item.id);
      } catch (error) {
        console.error('[SW] Failed to sync favorite:', error);
      }
    }
  }
}

async function syncPlaylists() {
  const db = await openDB();
  const tx = db.transaction('syncQueue', 'readwrite');
  const store = tx.objectStore('syncQueue');
  const items = await store.getAll();

  for (const item of items) {
    if (item.type === 'playlist') {
      try {
        await fetch('/api/playlists', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(item.data),
        });
        await store.delete(item.id);
      } catch (error) {
        console.error('[SW] Failed to sync playlist:', error);
      }
    }
  }
}

// === IndexedDB Helper ===

function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('mycelix-sw', 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('syncQueue')) {
        db.createObjectStore('syncQueue', { keyPath: 'id', autoIncrement: true });
      }
    };
  });
}

// === Push Notifications ===

self.addEventListener('push', (event) => {
  console.log('[SW] Push notification received');

  let data = { title: 'Mycelix', body: 'New notification' };

  if (event.data) {
    try {
      data = event.data.json();
    } catch (e) {
      data.body = event.data.text();
    }
  }

  const options = {
    body: data.body,
    icon: '/icons/icon-192.png',
    badge: '/icons/badge-72.png',
    vibrate: [100, 50, 100],
    data: data.url || '/',
    actions: data.actions || [],
  };

  event.waitUntil(self.registration.showNotification(data.title, options));
});

self.addEventListener('notificationclick', (event) => {
  console.log('[SW] Notification clicked');

  event.notification.close();

  event.waitUntil(
    clients.matchAll({ type: 'window' }).then((clientList) => {
      // Focus existing window or open new one
      for (const client of clientList) {
        if (client.url === event.notification.data && 'focus' in client) {
          return client.focus();
        }
      }
      if (clients.openWindow) {
        return clients.openWindow(event.notification.data);
      }
    })
  );
});

// === Messages from Client ===

self.addEventListener('message', (event) => {
  const { type, payload } = event.data;

  switch (type) {
    case 'CACHE_AUDIO':
      cacheAudioFile(payload.url, payload.trackId);
      break;

    case 'UNCACHE_AUDIO':
      uncacheAudioFile(payload.url);
      break;

    case 'GET_CACHE_STATUS':
      getCacheStatus().then((status) => {
        event.ports[0].postMessage(status);
      });
      break;

    case 'CLEAR_AUDIO_CACHE':
      clearAudioCache().then(() => {
        event.ports[0].postMessage({ success: true });
      });
      break;

    case 'SKIP_WAITING':
      self.skipWaiting();
      break;
  }
});

async function cacheAudioFile(url, trackId) {
  console.log('[SW] Caching audio file:', trackId);

  try {
    const response = await fetch(url);
    if (response.ok) {
      const cache = await caches.open(AUDIO_CACHE);
      await cache.put(url, response);

      // Notify clients
      const clients = await self.clients.matchAll();
      clients.forEach((client) => {
        client.postMessage({
          type: 'AUDIO_CACHED',
          payload: { trackId, url },
        });
      });
    }
  } catch (error) {
    console.error('[SW] Failed to cache audio:', error);
  }
}

async function uncacheAudioFile(url) {
  const cache = await caches.open(AUDIO_CACHE);
  await cache.delete(url);
}

async function getCacheStatus() {
  const audioCache = await caches.open(AUDIO_CACHE);
  const audioKeys = await audioCache.keys();

  let totalSize = 0;
  const cachedTracks = [];

  for (const request of audioKeys) {
    const response = await audioCache.match(request);
    if (response) {
      const blob = await response.clone().blob();
      totalSize += blob.size;
      cachedTracks.push({
        url: request.url,
        size: blob.size,
      });
    }
  }

  return {
    audioCacheSize: totalSize,
    audioCacheCount: cachedTracks.length,
    tracks: cachedTracks,
  };
}

async function clearAudioCache() {
  await caches.delete(AUDIO_CACHE);
  await caches.open(AUDIO_CACHE);
}

console.log('[SW] Service worker loaded');
