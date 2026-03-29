// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/// <reference types="@sveltejs/kit" />
/// <reference no-default-lib="true"/>
/// <reference lib="esnext" />
/// <reference lib="webworker" />

/**
 * Observatory Service Worker — offline caching for resilience deployments.
 *
 * Caches the SvelteKit app shell and static assets so community members
 * can browse their last-fetched data when the device loses connectivity
 * to the local node (e.g., during load shedding or mesh-only operation).
 *
 * Strategy:
 * - Install: precache app shell (HTML, JS, CSS from SvelteKit build)
 * - Fetch: network-first for API calls, cache-first for static assets
 * - Activate: clean old caches
 */

import { build, files, version } from '$service-worker';

const sw = self as unknown as ServiceWorkerGlobalScope;
const CACHE_NAME = `observatory-v${version}`;

// Assets to precache: SvelteKit build output + static files
const PRECACHE_ASSETS = [...build, ...files];

// Install: precache app shell
sw.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(PRECACHE_ASSETS))
      .then(() => sw.skipWaiting()),
  );
});

// Activate: clean old caches
sw.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key !== CACHE_NAME)
            .map((key) => caches.delete(key)),
        ),
      )
      .then(() => sw.clients.claim()),
  );
});

// Fetch: network-first for navigation/API, cache-first for assets
sw.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests
  if (request.method !== 'GET') return;

  // Skip cross-origin requests
  if (url.origin !== sw.location.origin) return;

  // For build assets (hashed filenames) — cache-first (immutable)
  if (build.includes(url.pathname)) {
    event.respondWith(cacheFirst(request));
    return;
  }

  // For navigation (HTML pages) — network-first with cache fallback
  if (request.mode === 'navigate') {
    event.respondWith(networkFirst(request));
    return;
  }

  // For static files — cache-first
  if (files.includes(url.pathname)) {
    event.respondWith(cacheFirst(request));
    return;
  }

  // Everything else — network-first
  event.respondWith(networkFirst(request));
});

async function cacheFirst(request: Request): Promise<Response> {
  const cached = await caches.match(request);
  if (cached) return cached;

  const response = await fetch(request);
  if (response.ok) {
    const cache = await caches.open(CACHE_NAME);
    cache.put(request, response.clone());
  }
  return response;
}

async function networkFirst(request: Request): Promise<Response> {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = await caches.match(request);
    if (cached) return cached;

    // Last resort: return offline page for navigation
    if (request.mode === 'navigate') {
      const cached_index = await caches.match('/');
      if (cached_index) return cached_index;
    }

    return new Response('Offline', {
      status: 503,
      headers: { 'Content-Type': 'text/plain' },
    });
  }
}
