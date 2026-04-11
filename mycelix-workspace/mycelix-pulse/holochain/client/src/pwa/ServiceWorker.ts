// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Advanced Service Worker for Mycelix Mail
 *
 * Features:
 * - Offline-first with IndexedDB sync
 * - Background sync for emails
 * - Push notifications
 * - Holochain-aware caching
 * - Periodic sync for trust updates
 */

/// <reference lib="webworker" />

declare const self: ServiceWorkerGlobalScope;

const CACHE_VERSION = 'mycelix-v1';
const STATIC_CACHE = `${CACHE_VERSION}-static`;
const DYNAMIC_CACHE = `${CACHE_VERSION}-dynamic`;
const EMAIL_CACHE = `${CACHE_VERSION}-emails`;

// Cache strategies
const CACHE_STRATEGIES = {
  // Static assets - cache first
  static: [
    '/',
    '/index.html',
    '/manifest.json',
    '/offline.html',
    /\.js$/,
    /\.css$/,
    /\.woff2?$/,
    /\.png$/,
    /\.svg$/,
    /\.ico$/,
  ],
  // API calls - network first with cache fallback
  network: [
    /\/api\//,
    /\/holochain\//,
  ],
  // Emails - stale while revalidate
  staleWhileRevalidate: [
    /\/emails\//,
    /\/contacts\//,
    /\/trust\//,
  ],
};

// Install event - cache static assets
self.addEventListener('install', (event: ExtendableEvent) => {
  event.waitUntil(
    caches.open(STATIC_CACHE).then((cache) => {
      return cache.addAll([
        '/',
        '/index.html',
        '/offline.html',
        '/manifest.json',
      ]);
    })
  );
  self.skipWaiting();
});

// Activate event - clean old caches
self.addEventListener('activate', (event: ExtendableEvent) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name.startsWith('mycelix-') && name !== CACHE_VERSION)
          .map((name) => caches.delete(name))
      );
    })
  );
  self.clients.claim();
});

// Fetch event - smart caching
self.addEventListener('fetch', (event: FetchEvent) => {
  const url = new URL(event.request.url);

  // Skip non-GET requests
  if (event.request.method !== 'GET') {
    return;
  }

  // Skip WebSocket connections
  if (url.protocol === 'ws:' || url.protocol === 'wss:') {
    return;
  }

  // Determine caching strategy
  const strategy = getCacheStrategy(url.pathname);

  switch (strategy) {
    case 'cache-first':
      event.respondWith(cacheFirst(event.request));
      break;
    case 'network-first':
      event.respondWith(networkFirst(event.request));
      break;
    case 'stale-while-revalidate':
      event.respondWith(staleWhileRevalidate(event.request));
      break;
    default:
      event.respondWith(networkFirst(event.request));
  }
});

function getCacheStrategy(pathname: string): string {
  // Check static patterns
  for (const pattern of CACHE_STRATEGIES.static) {
    if (typeof pattern === 'string') {
      if (pathname === pattern) return 'cache-first';
    } else if (pattern.test(pathname)) {
      return 'cache-first';
    }
  }

  // Check network patterns
  for (const pattern of CACHE_STRATEGIES.network) {
    if (pattern.test(pathname)) return 'network-first';
  }

  // Check stale-while-revalidate patterns
  for (const pattern of CACHE_STRATEGIES.staleWhileRevalidate) {
    if (pattern.test(pathname)) return 'stale-while-revalidate';
  }

  return 'network-first';
}

async function cacheFirst(request: Request): Promise<Response> {
  const cached = await caches.match(request);
  if (cached) {
    return cached;
  }

  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    return caches.match('/offline.html') as Promise<Response>;
  }
}

async function networkFirst(request: Request): Promise<Response> {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = await caches.match(request);
    if (cached) {
      return cached;
    }
    return caches.match('/offline.html') as Promise<Response>;
  }
}

async function staleWhileRevalidate(request: Request): Promise<Response> {
  const cache = await caches.open(EMAIL_CACHE);
  const cached = await cache.match(request);

  const fetchPromise = fetch(request).then((response) => {
    if (response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  });

  return cached || fetchPromise;
}

// Background sync for offline emails
self.addEventListener('sync', (event: SyncEvent) => {
  if (event.tag === 'sync-emails') {
    event.waitUntil(syncEmails());
  } else if (event.tag === 'sync-trust') {
    event.waitUntil(syncTrust());
  } else if (event.tag === 'send-queued-emails') {
    event.waitUntil(sendQueuedEmails());
  }
});

async function syncEmails(): Promise<void> {
  const db = await openIndexedDB();
  const pendingSync = await db.getAll('pending-sync');

  for (const item of pendingSync) {
    try {
      await fetch('/api/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(item),
      });
      await db.delete('pending-sync', item.id);
    } catch {
      // Will retry on next sync
    }
  }
}

async function syncTrust(): Promise<void> {
  try {
    const response = await fetch('/api/trust/sync');
    if (response.ok) {
      const data = await response.json();
      const db = await openIndexedDB();
      await db.put('trust-cache', { id: 'network', data, updatedAt: Date.now() });
    }
  } catch {
    // Will retry on next sync
  }
}

async function sendQueuedEmails(): Promise<void> {
  const db = await openIndexedDB();
  const queuedEmails = await db.getAll('email-queue');

  for (const email of queuedEmails) {
    try {
      await fetch('/api/emails/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(email),
      });
      await db.delete('email-queue', email.id);

      // Notify user
      self.registration.showNotification('Email Sent', {
        body: `"${email.subject}" was sent successfully`,
        icon: '/icons/sent.png',
        tag: `sent-${email.id}`,
      });
    } catch {
      // Will retry on next sync
    }
  }
}

// Push notifications
self.addEventListener('push', (event: PushEvent) => {
  if (!event.data) return;

  const data = event.data.json();

  const options: NotificationOptions = {
    body: data.body,
    icon: data.icon || '/icons/notification.png',
    badge: '/icons/badge.png',
    tag: data.tag,
    data: data.data,
    actions: data.actions || [],
    vibrate: [100, 50, 100],
    requireInteraction: data.requireInteraction || false,
  };

  event.waitUntil(
    self.registration.showNotification(data.title, options)
  );
});

// Notification click handler
self.addEventListener('notificationclick', (event: NotificationEvent) => {
  event.notification.close();

  const data = event.notification.data;

  if (event.action === 'reply') {
    event.waitUntil(
      self.clients.openWindow(`/compose?replyTo=${data.emailHash}`)
    );
  } else if (event.action === 'archive') {
    event.waitUntil(
      fetch(`/api/emails/${data.emailHash}/archive`, { method: 'POST' })
    );
  } else {
    event.waitUntil(
      self.clients.matchAll({ type: 'window' }).then((clients) => {
        if (clients.length > 0) {
          clients[0].focus();
          clients[0].postMessage({
            type: 'NOTIFICATION_CLICK',
            data: data,
          });
        } else {
          self.clients.openWindow(data.url || '/');
        }
      })
    );
  }
});

// Periodic background sync (for trust network updates)
self.addEventListener('periodicsync', (event: PeriodicSyncEvent) => {
  if (event.tag === 'trust-update') {
    event.waitUntil(syncTrust());
  } else if (event.tag === 'email-check') {
    event.waitUntil(checkNewEmails());
  }
});

async function checkNewEmails(): Promise<void> {
  try {
    const response = await fetch('/api/emails/unread-count');
    if (response.ok) {
      const { count, latest } = await response.json();

      if (count > 0 && latest) {
        self.registration.showNotification('New Emails', {
          body: `You have ${count} unread email${count > 1 ? 's' : ''}`,
          icon: '/icons/mail.png',
          tag: 'new-emails',
          data: { url: '/inbox' },
        });
      }
    }
  } catch {
    // Will retry on next periodic sync
  }
}

// Message handler for client communication
self.addEventListener('message', (event: ExtendableMessageEvent) => {
  const { type, payload } = event.data;

  switch (type) {
    case 'SKIP_WAITING':
      self.skipWaiting();
      break;

    case 'CACHE_URLS':
      event.waitUntil(
        caches.open(STATIC_CACHE).then((cache) => cache.addAll(payload.urls))
      );
      break;

    case 'CLEAR_CACHE':
      event.waitUntil(
        caches.keys().then((names) => Promise.all(names.map((n) => caches.delete(n))))
      );
      break;

    case 'GET_CACHE_STATUS':
      event.waitUntil(
        getCacheStatus().then((status) => {
          event.ports[0]?.postMessage(status);
        })
      );
      break;

    case 'QUEUE_EMAIL':
      event.waitUntil(queueEmail(payload));
      break;

    case 'REGISTER_PERIODIC_SYNC':
      event.waitUntil(registerPeriodicSync(payload.tag, payload.minInterval));
      break;
  }
});

async function getCacheStatus(): Promise<object> {
  const cacheNames = await caches.keys();
  const status: Record<string, number> = {};

  for (const name of cacheNames) {
    const cache = await caches.open(name);
    const keys = await cache.keys();
    status[name] = keys.length;
  }

  return status;
}

async function queueEmail(email: any): Promise<void> {
  const db = await openIndexedDB();
  await db.add('email-queue', { ...email, queuedAt: Date.now() });

  // Register sync
  if ('sync' in self.registration) {
    await self.registration.sync.register('send-queued-emails');
  }
}

async function registerPeriodicSync(tag: string, minInterval: number): Promise<void> {
  if ('periodicSync' in self.registration) {
    try {
      await (self.registration as any).periodicSync.register(tag, { minInterval });
    } catch {
      // Periodic sync not available
    }
  }
}

// IndexedDB helper
function openIndexedDB(): Promise<IDBDatabase & { getAll: (store: string) => Promise<any[]>; delete: (store: string, key: any) => Promise<void>; put: (store: string, value: any) => Promise<void>; add: (store: string, value: any) => Promise<void> }> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('mycelix-sw', 1);

    request.onerror = () => reject(request.error);

    request.onsuccess = () => {
      const db = request.result as any;

      db.getAll = (storeName: string) => {
        return new Promise((res, rej) => {
          const tx = db.transaction(storeName, 'readonly');
          const store = tx.objectStore(storeName);
          const req = store.getAll();
          req.onsuccess = () => res(req.result);
          req.onerror = () => rej(req.error);
        });
      };

      db.delete = (storeName: string, key: any) => {
        return new Promise((res, rej) => {
          const tx = db.transaction(storeName, 'readwrite');
          const store = tx.objectStore(storeName);
          const req = store.delete(key);
          req.onsuccess = () => res();
          req.onerror = () => rej(req.error);
        });
      };

      db.put = (storeName: string, value: any) => {
        return new Promise((res, rej) => {
          const tx = db.transaction(storeName, 'readwrite');
          const store = tx.objectStore(storeName);
          const req = store.put(value);
          req.onsuccess = () => res();
          req.onerror = () => rej(req.error);
        });
      };

      db.add = (storeName: string, value: any) => {
        return new Promise((res, rej) => {
          const tx = db.transaction(storeName, 'readwrite');
          const store = tx.objectStore(storeName);
          const req = store.add(value);
          req.onsuccess = () => res();
          req.onerror = () => rej(req.error);
        });
      };

      resolve(db);
    };

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains('pending-sync')) {
        db.createObjectStore('pending-sync', { keyPath: 'id', autoIncrement: true });
      }
      if (!db.objectStoreNames.contains('email-queue')) {
        db.createObjectStore('email-queue', { keyPath: 'id', autoIncrement: true });
      }
      if (!db.objectStoreNames.contains('trust-cache')) {
        db.createObjectStore('trust-cache', { keyPath: 'id' });
      }
    };
  });
}

// Type declarations for missing APIs
interface SyncEvent extends ExtendableEvent {
  tag: string;
}

interface PeriodicSyncEvent extends ExtendableEvent {
  tag: string;
}

export {};
