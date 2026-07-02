// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// Mycelix Praxis — Shadow Node Service Worker (PWA Offline Resilience)

const CACHE_NAME = 'praxis-shadow-v1';
const ASSETS = [
    '/',
    '/index.html',
    '/pkg/praxis_leptos.js',
    '/pkg/praxis_leptos_bg.wasm',
    '/static/manifest.json',
    '/static/style.css'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll(ASSETS);
        })
    );
});

self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request).then((response) => {
            // Serve from cache, fallback to network
            return response || fetch(event.request).catch(() => {
                // If both fail, and it's a navigation, return cached index.html
                if (event.request.mode === 'navigate') {
                    return caches.match('/');
                }
            });
        })
    );
});

// Self-update logic: activate immediately
self.addEventListener('activate', (event) => {
    event.waitUntil(clients.claim());
});
