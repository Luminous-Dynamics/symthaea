// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Service Worker for PWA Offline Support
 *
 * Handles caching, background sync, and offline email composition
 */

const CACHE_NAME = 'mycelix-mail-v1';
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/static/js/main.js',
  '/static/css/main.css',
];

const API_CACHE_NAME = 'mycelix-api-v1';
const OFFLINE_QUEUE_NAME = 'mycelix-offline-queue';

// Cache strategies
type CacheStrategy = 'cache-first' | 'network-first' | 'stale-while-revalidate';

interface CacheConfig {
  pattern: RegExp;
  strategy: CacheStrategy;
  maxAge?: number; // in seconds
}

const cacheConfigs: CacheConfig[] = [
  // Static assets - cache first
  { pattern: /\.(js|css|png|jpg|svg|woff2?)$/, strategy: 'cache-first' },
  // API emails list - network first with fallback
  { pattern: /\/api\/emails/, strategy: 'network-first', maxAge: 300 },
  // API contacts - stale while revalidate
  { pattern: /\/api\/contacts/, strategy: 'stale-while-revalidate', maxAge: 3600 },
  // API user settings - cache first
  { pattern: /\/api\/settings/, strategy: 'cache-first', maxAge: 86400 },
];

// Install event - cache static assets
self.addEventListener('install', (event: ExtendableEvent) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[SW] Caching static assets');
      return cache.addAll(STATIC_ASSETS);
    })
  );
  // Skip waiting to activate immediately
  (self as any).skipWaiting();
});

// Activate event - cleanup old caches
self.addEventListener('activate', (event: ExtendableEvent) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME && name !== API_CACHE_NAME)
          .map((name) => caches.delete(name))
      );
    })
  );
  // Claim all clients immediately
  (self as any).clients.claim();
});

// Fetch event - apply caching strategies
self.addEventListener('fetch', (event: FetchEvent) => {
  const url = new URL(event.request.url);

  // Skip non-GET requests (except for background sync)
  if (event.request.method !== 'GET') {
    // Queue POST/PUT/DELETE for background sync if offline
    if (!navigator.onLine) {
      event.respondWith(queueOfflineRequest(event.request));
      return;
    }
    return;
  }

  // Find matching cache config
  const config = cacheConfigs.find((c) => c.pattern.test(url.pathname));

  if (config) {
    switch (config.strategy) {
      case 'cache-first':
        event.respondWith(cacheFirst(event.request, config.maxAge));
        break;
      case 'network-first':
        event.respondWith(networkFirst(event.request, config.maxAge));
        break;
      case 'stale-while-revalidate':
        event.respondWith(staleWhileRevalidate(event.request, config.maxAge));
        break;
    }
  } else {
    // Default to network first for API, cache first for static
    if (url.pathname.startsWith('/api/')) {
      event.respondWith(networkFirst(event.request));
    } else {
      event.respondWith(cacheFirst(event.request));
    }
  }
});

// Cache-first strategy
async function cacheFirst(request: Request, maxAge?: number): Promise<Response> {
  const cached = await caches.match(request);

  if (cached && (!maxAge || !isExpired(cached, maxAge))) {
    return cached;
  }

  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    if (cached) return cached;
    return offlineResponse();
  }
}

// Network-first strategy
async function networkFirst(request: Request, maxAge?: number): Promise<Response> {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(API_CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = await caches.match(request);
    if (cached && (!maxAge || !isExpired(cached, maxAge))) {
      return cached;
    }
    return offlineResponse();
  }
}

// Stale-while-revalidate strategy
async function staleWhileRevalidate(request: Request, maxAge?: number): Promise<Response> {
  const cached = await caches.match(request);

  const fetchPromise = fetch(request).then((response) => {
    if (response.ok) {
      caches.open(API_CACHE_NAME).then((cache) => cache.put(request, response.clone()));
    }
    return response;
  });

  if (cached && (!maxAge || !isExpired(cached, maxAge))) {
    // Return cached immediately, update in background
    fetchPromise.catch(() => {}); // Ignore errors
    return cached;
  }

  return fetchPromise.catch(() => cached || offlineResponse());
}

// Check if cached response is expired
function isExpired(response: Response, maxAge: number): boolean {
  const dateHeader = response.headers.get('date');
  if (!dateHeader) return false;

  const date = new Date(dateHeader).getTime();
  const now = Date.now();
  return (now - date) / 1000 > maxAge;
}

// Offline response
function offlineResponse(): Response {
  return new Response(
    JSON.stringify({
      error: 'offline',
      message: 'You are currently offline. This action will be synced when you reconnect.',
    }),
    {
      status: 503,
      headers: { 'Content-Type': 'application/json' },
    }
  );
}

// Queue offline requests for background sync
async function queueOfflineRequest(request: Request): Promise<Response> {
  const db = await openOfflineDB();
  const tx = db.transaction(OFFLINE_QUEUE_NAME, 'readwrite');
  const store = tx.objectStore(OFFLINE_QUEUE_NAME);

  const body = await request.clone().text();

  await store.add({
    id: Date.now().toString(),
    url: request.url,
    method: request.method,
    headers: Object.fromEntries(request.headers.entries()),
    body,
    timestamp: Date.now(),
  });

  // Register for background sync
  if ('sync' in self.registration) {
    await (self.registration as any).sync.register('sync-offline-queue');
  }

  return new Response(
    JSON.stringify({
      queued: true,
      message: 'Request queued for sync when online',
    }),
    {
      status: 202,
      headers: { 'Content-Type': 'application/json' },
    }
  );
}

// Background sync event
self.addEventListener('sync', (event: any) => {
  if (event.tag === 'sync-offline-queue') {
    event.waitUntil(syncOfflineQueue());
  }
});

// Process offline queue
async function syncOfflineQueue(): Promise<void> {
  const db = await openOfflineDB();
  const tx = db.transaction(OFFLINE_QUEUE_NAME, 'readwrite');
  const store = tx.objectStore(OFFLINE_QUEUE_NAME);

  const requests = await store.getAll();

  for (const req of requests) {
    try {
      await fetch(req.url, {
        method: req.method,
        headers: req.headers,
        body: req.body,
      });

      // Remove from queue on success
      await store.delete(req.id);

      // Notify clients
      const clients = await (self as any).clients.matchAll();
      clients.forEach((client: any) => {
        client.postMessage({
          type: 'SYNC_SUCCESS',
          requestId: req.id,
          url: req.url,
        });
      });
    } catch (error) {
      console.error('[SW] Failed to sync request:', req.url, error);
    }
  }
}

// Open IndexedDB for offline queue
function openOfflineDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('mycelix-offline', 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(OFFLINE_QUEUE_NAME)) {
        db.createObjectStore(OFFLINE_QUEUE_NAME, { keyPath: 'id' });
      }
    };
  });
}

// Push notification handling
self.addEventListener('push', (event: PushEvent) => {
  const data = event.data?.json() || {};

  const options: NotificationOptions = {
    body: data.body || 'New email received',
    icon: '/icons/icon-192.png',
    badge: '/icons/badge-72.png',
    tag: data.tag || 'email-notification',
    data: data.url || '/',
    actions: [
      { action: 'open', title: 'Open' },
      { action: 'archive', title: 'Archive' },
    ],
  };

  event.waitUntil(
    (self as any).registration.showNotification(data.title || 'Mycelix Mail', options)
  );
});

// Notification click handling
self.addEventListener('notificationclick', (event: NotificationEvent) => {
  event.notification.close();

  if (event.action === 'archive') {
    // Archive the email
    event.waitUntil(
      fetch('/api/emails/archive', {
        method: 'POST',
        body: JSON.stringify({ emailId: event.notification.data.emailId }),
      })
    );
  } else {
    // Open the app
    event.waitUntil(
      (self as any).clients.openWindow(event.notification.data || '/')
    );
  }
});

// Message handling from main thread
self.addEventListener('message', (event: MessageEvent) => {
  if (event.data.type === 'SKIP_WAITING') {
    (self as any).skipWaiting();
  }

  if (event.data.type === 'CACHE_EMAILS') {
    // Pre-cache emails for offline access
    event.waitUntil(
      cacheEmails(event.data.emails)
    );
  }

  if (event.data.type === 'CLEAR_CACHE') {
    event.waitUntil(
      caches.delete(API_CACHE_NAME)
    );
  }
});

// Pre-cache emails for offline
async function cacheEmails(emails: any[]): Promise<void> {
  const cache = await caches.open(API_CACHE_NAME);

  for (const email of emails) {
    const response = new Response(JSON.stringify(email), {
      headers: { 'Content-Type': 'application/json' },
    });
    await cache.put(`/api/emails/${email.id}`, response);
  }
}

export {};
