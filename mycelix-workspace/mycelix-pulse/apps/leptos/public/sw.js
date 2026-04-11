// Mycelix Pulse — Service Worker v2
// Offline-first caching + background sync for PWA

const CACHE_NAME = 'mycelix-pulse-v2';
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/style/main.css',
];

// Install: cache static shell + prompt update
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

// Activate: clean old caches + claim clients immediately
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Fetch: cache-first for assets, network-first for navigation
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // WebSocket — pass through
  if (url.protocol === 'ws:' || url.protocol === 'wss:') return;

  // WASM/JS/CSS — cache-first (immutable hashed names)
  if (url.pathname.endsWith('.wasm') || url.pathname.endsWith('.js') || url.pathname.endsWith('.css')) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) return cached;
        return fetch(event.request).then((response) => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          }
          return response;
        });
      })
    );
    return;
  }

  // Images/fonts — cache-first with network fallback
  if (url.pathname.match(/\.(png|jpg|jpeg|svg|gif|woff2?|ttf|ico)$/)) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) return cached;
        return fetch(event.request).then((response) => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          }
          return response;
        });
      })
    );
    return;
  }

  // HTML navigation — network-first, fall back to cached index.html (SPA)
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          return response;
        })
        .catch(() => caches.match('/index.html'))
    );
    return;
  }

  // Everything else — network-first with cache fallback
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

// Background sync — process queued offline actions
self.addEventListener('sync', (event) => {
  if (event.tag === 'mycelix-offline-sync') {
    event.waitUntil(processOfflineQueue());
  }
});

async function processOfflineQueue() {
  // Read queued actions from IndexedDB or notify clients to flush localStorage queue
  const clients = await self.clients.matchAll();
  for (const client of clients) {
    client.postMessage({ type: 'flush-offline-queue' });
  }
}

// Push notifications
self.addEventListener('push', (event) => {
  const data = event.data ? event.data.json() : { title: 'Mycelix Pulse', body: 'New activity' };
  event.waitUntil(
    self.registration.showNotification(data.title || 'Mycelix Pulse', {
      body: data.body || 'You have new messages',
      icon: '/icons/icon-192.png',
      badge: '/icons/icon-72.png',
      tag: data.tag || 'mycelix-notification',
      data: { url: data.url || '/' },
    })
  );
});

// Notification click — focus or open the app
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const url = event.notification.data?.url || '/';
  event.waitUntil(
    self.clients.matchAll({ type: 'window' }).then((clients) => {
      for (const client of clients) {
        if (client.url.includes(self.location.origin) && 'focus' in client) {
          return client.focus();
        }
      }
      return self.clients.openWindow(url);
    })
  );
});

// Periodic background sync (if supported)
self.addEventListener('periodicsync', (event) => {
  if (event.tag === 'mycelix-check-mail') {
    event.waitUntil(checkForNewMail());
  }
});

async function checkForNewMail() {
  // In production: ping conductor for new messages
  // For now, just notify clients to refresh
  const clients = await self.clients.matchAll();
  for (const client of clients) {
    client.postMessage({ type: 'check-new-mail' });
  }
}
