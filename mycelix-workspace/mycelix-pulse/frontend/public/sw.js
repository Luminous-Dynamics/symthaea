// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail Service Worker
 * Enables offline support, background sync, and push notifications
 */

const CACHE_NAME = 'mycelix-mail-v1';
const OFFLINE_URL = '/offline.html';

// Resources to cache immediately on install
const STATIC_CACHE = [
  '/',
  '/index.html',
  '/offline.html',
  '/manifest.json',
  '/icons/icon-192x192.png',
  '/icons/icon-512x512.png',
];

// API routes to cache with network-first strategy
const API_CACHE_ROUTES = [
  '/api/v1/emails',
  '/api/v1/folders',
  '/api/v1/contacts',
  '/api/v1/settings',
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('[SW] Installing service worker...');

  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[SW] Caching static assets');
      return cache.addAll(STATIC_CACHE);
    })
  );

  self.skipWaiting();
});

// Activate event - cleanup old caches
self.addEventListener('activate', (event) => {
  console.log('[SW] Activating service worker...');

  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME)
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

  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }

  // Handle API requests with network-first strategy
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirst(request));
    return;
  }

  // Handle static assets with cache-first strategy
  event.respondWith(cacheFirst(request));
});

// Cache-first strategy for static assets
async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) {
    return cached;
  }

  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    // Return offline page for navigation requests
    if (request.mode === 'navigate') {
      return caches.match(OFFLINE_URL);
    }
    throw error;
  }
}

// Network-first strategy for API requests
async function networkFirst(request) {
  try {
    const response = await fetch(request);

    // Cache successful API responses
    if (response.ok && shouldCacheApiResponse(request.url)) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }

    return response;
  } catch (error) {
    // Fallback to cache for offline
    const cached = await caches.match(request);
    if (cached) {
      return cached;
    }

    // Return offline JSON response for API requests
    return new Response(
      JSON.stringify({
        error: 'offline',
        message: 'You are currently offline. Changes will sync when you reconnect.',
      }),
      {
        status: 503,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}

function shouldCacheApiResponse(url) {
  return API_CACHE_ROUTES.some((route) => url.includes(route));
}

// Background sync for offline actions
self.addEventListener('sync', (event) => {
  console.log('[SW] Background sync triggered:', event.tag);

  if (event.tag === 'sync-emails') {
    event.waitUntil(syncPendingEmails());
  } else if (event.tag === 'sync-drafts') {
    event.waitUntil(syncDrafts());
  }
});

async function syncPendingEmails() {
  const db = await openOfflineDB();
  const pendingEmails = await db.getAll('pending-sends');

  for (const email of pendingEmails) {
    try {
      const response = await fetch('/api/v1/emails/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(email.data),
      });

      if (response.ok) {
        await db.delete('pending-sends', email.id);
        await notifyEmailSent(email.data.subject);
      }
    } catch (error) {
      console.error('[SW] Failed to sync email:', error);
    }
  }
}

async function syncDrafts() {
  const db = await openOfflineDB();
  const pendingDrafts = await db.getAll('pending-drafts');

  for (const draft of pendingDrafts) {
    try {
      const response = await fetch('/api/v1/drafts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(draft.data),
      });

      if (response.ok) {
        await db.delete('pending-drafts', draft.id);
      }
    } catch (error) {
      console.error('[SW] Failed to sync draft:', error);
    }
  }
}

// Push notifications
self.addEventListener('push', (event) => {
  console.log('[SW] Push notification received');

  let data = { title: 'Mycelix Mail', body: 'New message received' };

  if (event.data) {
    try {
      data = event.data.json();
    } catch (e) {
      data.body = event.data.text();
    }
  }

  const options = {
    body: data.body,
    icon: '/icons/icon-192x192.png',
    badge: '/icons/badge-72x72.png',
    vibrate: [100, 50, 100],
    data: data.payload || {},
    actions: [
      { action: 'view', title: 'View', icon: '/icons/view.png' },
      { action: 'dismiss', title: 'Dismiss', icon: '/icons/dismiss.png' },
    ],
    tag: data.tag || 'default',
    renotify: true,
  };

  event.waitUntil(self.registration.showNotification(data.title, options));
});

// Notification click handler
self.addEventListener('notificationclick', (event) => {
  console.log('[SW] Notification clicked:', event.action);

  event.notification.close();

  if (event.action === 'dismiss') {
    return;
  }

  const urlToOpen = event.notification.data.url || '/';

  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clientList) => {
      // Check if app is already open
      for (const client of clientList) {
        if (client.url.includes(urlToOpen) && 'focus' in client) {
          return client.focus();
        }
      }

      // Open new window
      if (clients.openWindow) {
        return clients.openWindow(urlToOpen);
      }
    })
  );
});

// IndexedDB helper for offline storage
function openOfflineDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('mycelix-offline', 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const db = request.result;
      resolve({
        async getAll(store) {
          return new Promise((res, rej) => {
            const tx = db.transaction(store, 'readonly');
            const req = tx.objectStore(store).getAll();
            req.onsuccess = () => res(req.result);
            req.onerror = () => rej(req.error);
          });
        },
        async delete(store, key) {
          return new Promise((res, rej) => {
            const tx = db.transaction(store, 'readwrite');
            const req = tx.objectStore(store).delete(key);
            req.onsuccess = () => res();
            req.onerror = () => rej(req.error);
          });
        },
      });
    };

    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('pending-sends')) {
        db.createObjectStore('pending-sends', { keyPath: 'id', autoIncrement: true });
      }
      if (!db.objectStoreNames.contains('pending-drafts')) {
        db.createObjectStore('pending-drafts', { keyPath: 'id', autoIncrement: true });
      }
      if (!db.objectStoreNames.contains('cached-emails')) {
        db.createObjectStore('cached-emails', { keyPath: 'id' });
      }
    };
  });
}

async function notifyEmailSent(subject) {
  const clients = await self.clients.matchAll();
  clients.forEach((client) => {
    client.postMessage({
      type: 'EMAIL_SYNCED',
      subject,
    });
  });
}

// Periodic background sync (if supported)
self.addEventListener('periodicsync', (event) => {
  if (event.tag === 'check-new-emails') {
    event.waitUntil(checkForNewEmails());
  }
});

async function checkForNewEmails() {
  try {
    const response = await fetch('/api/v1/emails/check-new');
    if (response.ok) {
      const data = await response.json();
      if (data.newCount > 0) {
        self.registration.showNotification('New Mail', {
          body: `You have ${data.newCount} new message(s)`,
          icon: '/icons/icon-192x192.png',
          tag: 'new-mail',
        });
      }
    }
  } catch (error) {
    console.log('[SW] Background check failed (offline)');
  }
}

console.log('[SW] Service worker loaded');
