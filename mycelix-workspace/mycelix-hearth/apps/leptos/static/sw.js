// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// Mycelix Hearth — Service Worker (PWA offline app-shell cache)
//
// Trunk emits content-hashed JS/WASM/CSS filenames (e.g. main-<hash>.css),
// so we don't hardcode bundle names here (they'd go stale on every
// rebuild). Instead: precache the small set of stable-path shell files,
// then cache-first any other same-origin request as it's fetched.

const CACHE_NAME = 'hearth-shell-v1';
const PRECACHE_ASSETS = [
    '/',
    '/index.html',
    '/manifest.json',
    '/icon.svg'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE_ASSETS))
    );
    self.skipWaiting();
});

self.addEventListener('fetch', (event) => {
    const req = event.request;
    if (req.method !== 'GET' || new URL(req.url).origin !== self.location.origin) {
        return;
    }
    event.respondWith(
        caches.match(req).then((cached) => {
            if (cached) return cached;
            return fetch(req)
                .then((res) => {
                    if (res && res.ok) {
                        const clone = res.clone();
                        caches.open(CACHE_NAME).then((cache) => cache.put(req, clone));
                    }
                    return res;
                })
                .catch(() => {
                    if (req.mode === 'navigate') {
                        return caches.match('/index.html');
                    }
                });
        })
    );
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) =>
            Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
        ).then(() => self.clients.claim())
    );
});
