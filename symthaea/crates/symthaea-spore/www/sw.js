// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Service Worker for Sovereign Inoculation progressive decentralization.
//
// Strategy:
// 1. Cache core WASM + HTML on first visit (offline capability)
// 2. On subsequent loads, serve from cache, update in background
// 3. When Iroh mesh peers are discovered, prefer mesh over CDN
// 4. Heavy assets (Broca weights, Holochain zomes) fetched from nearest peer

const CACHE_NAME = 'symthaea-spore-v1';
const CORE_ASSETS = [
  '/',
  '/index.html',
  '/pkg/symthaea_spore_bg.wasm',
  '/pkg/symthaea_spore.js',
  '/spore-worker.js',
  '/sw.js',
];

// Install: cache core assets for offline capability
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(CORE_ASSETS);
    })
  );
  self.skipWaiting();
});

// Activate: clean old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) => {
      return Promise.all(
        names
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      );
    })
  );
  self.clients.claim();
});

// Fetch: cache-first for core assets, network-first for everything else
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Core assets: cache-first with background update
  if (CORE_ASSETS.some((asset) => url.pathname === asset || url.pathname.endsWith(asset))) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        const fetchPromise = fetch(event.request).then((response) => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          }
          return response;
        }).catch(() => cached);

        return cached || fetchPromise;
      })
    );
    return;
  }

  // Everything else: network-first with cache fallback
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        }
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});

// Message handler for Iroh mesh integration
self.addEventListener('message', (event) => {
  const { type, data } = event.data;

  switch (type) {
    case 'iroh-peer-discovered':
      // Future: register Iroh peer as alternative asset source
      // When mesh peers are available, heavy assets will be fetched from them
      console.log('[SW] Iroh peer discovered:', data.peerId);
      break;

    case 'cache-heavy-asset':
      // Cache a heavy asset (Broca weights, zome WASMs) fetched from mesh
      if (data.url && data.blob) {
        caches.open(CACHE_NAME).then((cache) => {
          const response = new Response(data.blob, {
            headers: { 'Content-Type': data.contentType || 'application/octet-stream' },
          });
          cache.put(data.url, response);
          console.log('[SW] Cached heavy asset from mesh:', data.url);
        });
      }
      break;

    case 'get-cache-status':
      // Report what's cached for the mesh status indicator
      caches.open(CACHE_NAME).then((cache) => {
        cache.keys().then((requests) => {
          event.source.postMessage({
            type: 'cache-status',
            cached: requests.map((r) => r.url),
            count: requests.length,
          });
        });
      });
      break;
  }
});
