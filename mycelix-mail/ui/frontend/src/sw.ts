// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail Service Worker
 *
 * Enables offline functionality, background sync, and push notifications
 */

/// <reference lib="webworker" />
declare const self: ServiceWorkerGlobalScope;

import { precacheAndRoute, cleanupOutdatedCaches } from 'workbox-precaching';
import { registerRoute, NavigationRoute, Route } from 'workbox-routing';
import { CacheFirst, NetworkFirst, StaleWhileRevalidate } from 'workbox-strategies';
import { ExpirationPlugin } from 'workbox-expiration';
import { BackgroundSyncPlugin } from 'workbox-background-sync';
import { CacheableResponsePlugin } from 'workbox-cacheable-response';

// Version for cache busting
const VERSION = '1.0.0';
const CACHE_PREFIX = 'mycelix-mail';

// Cache names
const STATIC_CACHE = `${CACHE_PREFIX}-static-${VERSION}`;
const DYNAMIC_CACHE = `${CACHE_PREFIX}-dynamic-${VERSION}`;
const EMAIL_CACHE = `${CACHE_PREFIX}-emails-${VERSION}`;
const ATTACHMENT_CACHE = `${CACHE_PREFIX}-attachments-${VERSION}`;

// Precache static assets (injected by build tool)
precacheAndRoute(self.__WB_MANIFEST || []);
cleanupOutdatedCaches();

// Navigation route - network first with fallback
const navigationRoute = new NavigationRoute(
  new NetworkFirst({
    cacheName: STATIC_CACHE,
    plugins: [
      new CacheableResponsePlugin({ statuses: [200] }),
    ],
  }),
  {
    // Don't cache API routes
    denylist: [/^\/api\//],
  }
);
registerRoute(navigationRoute);

// Static assets - cache first
registerRoute(
  ({ request }) =>
    request.destination === 'style' ||
    request.destination === 'script' ||
    request.destination === 'font',
  new CacheFirst({
    cacheName: STATIC_CACHE,
    plugins: [
      new CacheableResponsePlugin({ statuses: [200] }),
      new ExpirationPlugin({
        maxEntries: 100,
        maxAgeSeconds: 30 * 24 * 60 * 60, // 30 days
      }),
    ],
  })
);

// Images - stale while revalidate
registerRoute(
  ({ request }) => request.destination === 'image',
  new StaleWhileRevalidate({
    cacheName: `${CACHE_PREFIX}-images-${VERSION}`,
    plugins: [
      new CacheableResponsePlugin({ statuses: [200] }),
      new ExpirationPlugin({
        maxEntries: 200,
        maxAgeSeconds: 7 * 24 * 60 * 60, // 7 days
      }),
    ],
  })
);

// API routes for emails - network first with cache fallback
registerRoute(
  ({ url }) => url.pathname.startsWith('/api/emails'),
  new NetworkFirst({
    cacheName: EMAIL_CACHE,
    plugins: [
      new CacheableResponsePlugin({ statuses: [200] }),
      new ExpirationPlugin({
        maxEntries: 500,
        maxAgeSeconds: 24 * 60 * 60, // 24 hours
      }),
    ],
  })
);

// Attachments - cache first (they don't change)
registerRoute(
  ({ url }) => url.pathname.includes('/attachments/'),
  new CacheFirst({
    cacheName: ATTACHMENT_CACHE,
    plugins: [
      new CacheableResponsePlugin({ statuses: [200] }),
      new ExpirationPlugin({
        maxEntries: 100,
        maxAgeSeconds: 7 * 24 * 60 * 60, // 7 days
        purgeOnQuotaError: true,
      }),
    ],
  })
);

// Background sync for sending emails
const bgSyncPlugin = new BackgroundSyncPlugin('emailQueue', {
  maxRetentionTime: 24 * 60, // 24 hours in minutes
  onSync: async ({ queue }) => {
    let entry;
    while ((entry = await queue.shiftRequest())) {
      try {
        await fetch(entry.request.clone());
        console.log('Background sync: Email sent successfully');
      } catch (error) {
        console.error('Background sync failed:', error);
        await queue.unshiftRequest(entry);
        throw error;
      }
    }
  },
});

// Register route for sending emails with background sync
registerRoute(
  ({ url, request }) =>
    url.pathname === '/api/emails/send' && request.method === 'POST',
  new NetworkFirst({
    cacheName: DYNAMIC_CACHE,
    plugins: [bgSyncPlugin],
  }),
  'POST'
);

// GraphQL endpoint - network first
registerRoute(
  ({ url }) => url.pathname === '/api/graphql',
  new NetworkFirst({
    cacheName: DYNAMIC_CACHE,
    networkTimeoutSeconds: 10,
    plugins: [
      new CacheableResponsePlugin({ statuses: [200] }),
      new ExpirationPlugin({
        maxEntries: 100,
        maxAgeSeconds: 60 * 60, // 1 hour
      }),
    ],
  }),
  'POST'
);

// Install event
self.addEventListener('install', (event) => {
  console.log(`[SW] Installing version ${VERSION}`);
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

// Activate event
self.addEventListener('activate', (event) => {
  console.log(`[SW] Activating version ${VERSION}`);
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name.startsWith(CACHE_PREFIX) && !name.includes(VERSION))
          .map((name) => {
            console.log(`[SW] Deleting old cache: ${name}`);
            return caches.delete(name);
          })
      );
    })
  );
  self.clients.claim();
});

// Push notifications
self.addEventListener('push', (event) => {
  if (!event.data) return;

  const data = event.data.json();
  const options: NotificationOptions = {
    body: data.body,
    icon: '/icons/icon-192x192.png',
    badge: '/icons/badge-72x72.png',
    tag: data.tag || 'default',
    data: data.data,
    actions: data.actions || [
      { action: 'open', title: 'Open' },
      { action: 'dismiss', title: 'Dismiss' },
    ],
    requireInteraction: data.requireInteraction || false,
  };

  event.waitUntil(
    self.registration.showNotification(data.title || 'Mycelix Mail', options)
  );
});

// Notification click handler
self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  if (event.action === 'dismiss') return;

  const urlToOpen = event.notification.data?.url || '/';

  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clientList) => {
      // Try to focus existing window
      for (const client of clientList) {
        if (client.url.includes(urlToOpen) && 'focus' in client) {
          return client.focus();
        }
      }
      // Open new window
      return self.clients.openWindow(urlToOpen);
    })
  );
});

// Message handler for client communication
self.addEventListener('message', (event) => {
  if (!event.data) return;

  switch (event.data.type) {
    case 'SKIP_WAITING':
      self.skipWaiting();
      break;

    case 'CACHE_EMAIL':
      // Cache specific email for offline access
      event.waitUntil(
        caches.open(EMAIL_CACHE).then((cache) => {
          return cache.put(
            new Request(`/api/emails/${event.data.emailId}`),
            new Response(JSON.stringify(event.data.email), {
              headers: { 'Content-Type': 'application/json' },
            })
          );
        })
      );
      break;

    case 'CLEAR_CACHE':
      event.waitUntil(
        caches.keys().then((names) =>
          Promise.all(names.map((name) => caches.delete(name)))
        )
      );
      break;

    case 'GET_CACHE_SIZE':
      event.waitUntil(
        (async () => {
          const cacheNames = await caches.keys();
          let totalSize = 0;

          for (const name of cacheNames) {
            const cache = await caches.open(name);
            const keys = await cache.keys();
            for (const request of keys) {
              const response = await cache.match(request);
              if (response) {
                const blob = await response.blob();
                totalSize += blob.size;
              }
            }
          }

          event.ports[0].postMessage({ size: totalSize });
        })()
      );
      break;
  }
});

// Periodic background sync for email checking
self.addEventListener('periodicsync', (event: any) => {
  if (event.tag === 'check-emails') {
    event.waitUntil(checkForNewEmails());
  }
});

async function checkForNewEmails() {
  try {
    const response = await fetch('/api/emails/check-new');
    const data = await response.json();

    if (data.newCount > 0) {
      await self.registration.showNotification('New Emails', {
        body: `You have ${data.newCount} new email${data.newCount > 1 ? 's' : ''}`,
        icon: '/icons/icon-192x192.png',
        tag: 'new-emails',
        data: { url: '/inbox' },
      });
    }
  } catch (error) {
    console.error('Failed to check for new emails:', error);
  }
}

// Offline fallback
self.addEventListener('fetch', (event) => {
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request).catch(() => {
        return caches.match('/offline.html') || new Response('Offline', {
          status: 503,
          statusText: 'Service Unavailable',
        });
      })
    );
  }
});

export {};
