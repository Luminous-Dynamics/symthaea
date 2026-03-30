#!/usr/bin/env node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

/**
 * Service Worker Generator
 *
 * Generates the service worker file (sw.js) for PWA support.
 * Run this after building to create the service worker in the dist folder.
 */

import { readFileSync, writeFileSync, readdirSync, statSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = join(__dirname, '..', 'dist');

// Generate cache version based on build time
const CACHE_VERSION = `mycelix-v${Date.now()}`;

// Get all files in dist for precaching
function getFilesToCache(dir, baseDir = dir, files = []) {
  try {
    const items = readdirSync(dir);
    for (const item of items) {
      const fullPath = join(dir, item);
      const stat = statSync(fullPath);
      if (stat.isDirectory()) {
        getFilesToCache(fullPath, baseDir, files);
      } else {
        const relativePath = '/' + fullPath.slice(baseDir.length + 1).replace(/\\/g, '/');
        // Skip source maps and hidden files
        if (!relativePath.endsWith('.map') && !item.startsWith('.')) {
          files.push(relativePath);
        }
      }
    }
  } catch (e) {
    console.log('Note: dist directory not found. Run build first.');
  }
  return files;
}

const filesToCache = getFilesToCache(distDir);

const swContent = `
// Generated Service Worker for Mycelix Mail PWA
// Cache version: ${CACHE_VERSION}

const CACHE_NAME = '${CACHE_VERSION}';
const STATIC_CACHE = '${CACHE_VERSION}-static';
const DYNAMIC_CACHE = '${CACHE_VERSION}-dynamic';
const API_CACHE = '${CACHE_VERSION}-api';

// Files to precache
const PRECACHE_URLS = ${JSON.stringify(filesToCache, null, 2)};

// Install event - precache static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames
            .filter((name) => name !== STATIC_CACHE && name !== DYNAMIC_CACHE && name !== API_CACHE)
            .map((name) => caches.delete(name))
        );
      })
      .then(() => self.clients.claim())
  );
});

// Fetch event - serve from cache with network fallback
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip cross-origin requests
  if (url.origin !== location.origin) {
    return;
  }

  // API requests - network first, cache fallback
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(request)
        .then((response) => {
          const clonedResponse = response.clone();
          caches.open(API_CACHE).then((cache) => {
            if (request.method === 'GET') {
              cache.put(request, clonedResponse);
            }
          });
          return response;
        })
        .catch(() => caches.match(request))
    );
    return;
  }

  // Static assets - cache first, network fallback
  event.respondWith(
    caches.match(request)
      .then((cachedResponse) => {
        if (cachedResponse) {
          return cachedResponse;
        }
        return fetch(request).then((response) => {
          const clonedResponse = response.clone();
          caches.open(DYNAMIC_CACHE).then((cache) => {
            cache.put(request, clonedResponse);
          });
          return response;
        });
      })
      .catch(() => {
        // Return offline page for navigation requests
        if (request.mode === 'navigate') {
          return caches.match('/offline.html');
        }
      })
  );
});

// Background sync for offline operations
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-emails') {
    event.waitUntil(syncEmails());
  } else if (event.tag === 'sync-attestations') {
    event.waitUntil(syncAttestations());
  }
});

async function syncEmails() {
  // Sync queued email operations when back online
  const cache = await caches.open('offline-queue');
  const requests = await cache.keys();

  for (const request of requests) {
    try {
      await fetch(request);
      await cache.delete(request);
    } catch (e) {
      console.log('Failed to sync:', request.url);
    }
  }
}

async function syncAttestations() {
  // Sync queued attestation operations when back online
  const cache = await caches.open('offline-queue');
  const requests = await cache.keys();

  for (const request of requests) {
    if (request.url.includes('/attestations')) {
      try {
        await fetch(request);
        await cache.delete(request);
      } catch (e) {
        console.log('Failed to sync attestation:', request.url);
      }
    }
  }
}

// Push notifications
self.addEventListener('push', (event) => {
  const data = event.data?.json() || {};
  const options = {
    body: data.body || 'New notification',
    icon: '/icons/icon-192.png',
    badge: '/icons/badge-72.png',
    vibrate: [100, 50, 100],
    data: { url: data.url || '/' },
    actions: data.actions || [],
  };

  event.waitUntil(
    self.registration.showNotification(data.title || 'Mycelix Mail', options)
  );
});

// Notification click handler
self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  event.waitUntil(
    clients.matchAll({ type: 'window' })
      .then((clientList) => {
        for (const client of clientList) {
          if (client.url === event.notification.data.url && 'focus' in client) {
            return client.focus();
          }
        }
        if (clients.openWindow) {
          return clients.openWindow(event.notification.data.url);
        }
      })
  );
});
`;

// Write to dist folder if it exists, otherwise to public
const outputPath = filesToCache.length > 0
  ? join(distDir, 'sw.js')
  : join(__dirname, '..', 'public', 'sw.js');

writeFileSync(outputPath, swContent.trim());
console.log(\`Service worker generated at: \${outputPath}\`);
console.log(\`Cache version: \${CACHE_VERSION}\`);
console.log(\`Precached files: \${filesToCache.length}\`);
