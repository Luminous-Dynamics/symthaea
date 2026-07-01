// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Symthaea Service Worker — caches large WASM assets for instant reload.
// Includes SRI integrity verification for binary checkpoints before caching.
// Cache name derived from build version. Trunk embeds content hashes in filenames,
// so each deploy creates a new cache name, automatically invalidating stale caches.
// The SW registration in index.html sets this via query param: sw.js?v=<hash>
const urlParams = new URL(self.location).searchParams;
const CACHE_VERSION = urlParams.get('v') || 'v2';
const CACHE_NAME = 'symthaea-' + CACHE_VERSION;

// SHA-384 integrity hashes for Broca binary checkpoints.
// Must match the hashes in spore-worker.js.
const INTEGRITY_HASHES = {
  'broca-spore-v1':  'zg2Z4NjVMxVkRCtcs66FQK0nCBSrghMQ3uh9gChjx6HTl2bYvuhvttRXJdBQkr4j',
  'broca-pipeline':  'zg2Z4NjVMxVkRCtcs66FQK0nCBSrghMQ3uh9gChjx6HTl2bYvuhvttRXJdBQkr4j',
};

// Compute SHA-384 base64 hash of an ArrayBuffer.
async function sha384Base64(buffer) {
  const hashBuf = await crypto.subtle.digest('SHA-384', buffer);
  const hashArr = new Uint8Array(hashBuf);
  let binary = '';
  for (let i = 0; i < hashArr.length; i++) binary += String.fromCharCode(hashArr[i]);
  return btoa(binary);
}

// Determine which integrity key (if any) matches this URL.
function integrityKeyFor(url) {
  if (url.includes('broca-spore')) return 'broca-spore-v1';
  if (url.includes('broca-pipeline')) return 'broca-pipeline';
  return null;
}

self.addEventListener('install', event => {
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  // Purge old cache versions
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys
          .filter(k => k.startsWith('symthaea-') && k !== CACHE_NAME)
          .map(k => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', event => {
  const url = event.request.url;
  // Cache-first for large binary assets and WASM
  if (url.includes('broca-spore') || url.includes('broca-pipeline') ||
      url.includes('.wasm') || url.includes('symthaea_spore')) {
    event.respondWith(
      caches.match(event.request).then(cached => {
        if (cached) return cached;
        return fetch(event.request).then(async response => {
          if (!response.ok) return response;

          // For Broca binaries: verify integrity before caching
          const key = integrityKeyFor(url);
          if (key && INTEGRITY_HASHES[key]) {
            const clone = response.clone();
            const buffer = await clone.arrayBuffer();
            const hash = await sha384Base64(buffer);
            if (hash !== INTEGRITY_HASHES[key]) {
              console.error(
                '[SW] Integrity check FAILED for', key,
                '— expected', INTEGRITY_HASHES[key],
                '— got', hash, '— NOT caching'
              );
              // Return the original response but do NOT cache it
              return response;
            }
            // Integrity verified — cache a new Response from the verified buffer
            const verifiedResponse = new Response(buffer, {
              status: response.status,
              statusText: response.statusText,
              headers: response.headers,
            });
            const cache = await caches.open(CACHE_NAME);
            await cache.put(event.request, verifiedResponse.clone());
            return verifiedResponse;
          }

          // Non-Broca assets (WASM, etc.): cache without integrity check
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
          return response;
        });
      })
    );
  }
});
