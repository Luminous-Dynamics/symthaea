// Symthaea Service Worker — caches large WASM assets for instant reload.
const CACHE_NAME = 'symthaea-v1';
const CACHED_ASSETS = [
  './assets/broca-spore-v1.bin',
  './assets/broca-pipeline.bin',
];

self.addEventListener('install', event => {
  // Pre-cache known large assets (ignore failures for optional files)
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache =>
      Promise.allSettled(
        CACHED_ASSETS.map(url => cache.add(url).catch(() => {}))
      )
    )
  );
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
  // Cache-first for large binary assets
  if (url.includes('broca-spore') || url.includes('broca-pipeline') ||
      url.includes('.wasm') || url.includes('symthaea_spore')) {
    event.respondWith(
      caches.match(event.request).then(cached => {
        if (cached) return cached;
        return fetch(event.request).then(response => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
          }
          return response;
        });
      })
    );
  }
});
