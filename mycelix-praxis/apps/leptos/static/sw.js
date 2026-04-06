// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Praxis Service Worker v3 — offline-first with proactive lesson caching
const CACHE_NAME = 'praxis-v4';
const PRECACHE = [
  '/',
  '/index.html',
];

// Grade 12 lesson files to proactively cache (top exam topics)
const GR12_LESSONS = [
  '/caps/generated/math-12/caps.mathematics.gr12.p1.calc.json',
  '/caps/generated/math-12/caps.mathematics.gr12.p1.alg.json',
  '/caps/generated/math-12/caps.mathematics.gr12.p1.seq.json',
  '/caps/generated/math-12/caps.mathematics.gr12.p1.fin.json',
  '/caps/generated/math-12/caps.mathematics.gr12.p1.fn.json',
  '/caps/generated/math-12/caps.mathematics.gr12.p2.trig.json',
  '/caps/generated/math-12/caps.mathematics.gr12.p2.stat.json',
  '/caps/generated/math-12/caps.mathematics.gr12.p2.geom.json',
  '/caps/generated/math-12/caps.mathematics.gr12.p2.anag.json',
  '/caps/generated/math-12/caps.mathematics.gr12.p2.cnt.json',
  '/caps/generated/physics-12/caps.physicalsciences.gr12.p1.elec1.json',
  '/caps/generated/physics-12/caps.physicalsciences.gr12.p1.mech1.json',
  '/caps/generated/physics-12/caps.physicalsciences.gr12.p1.mech2.json',
  '/caps/generated/physics-12/caps.physicalsciences.gr12.p1.elec2.json',
  '/caps/generated/physics-12/caps.physicalsciences.gr12.p1.mod1.json',
  '/caps/generated/physics-12/caps.physicalsciences.gr12.p1.dop.json',
  '/caps/generated/physics-12/caps.physicalsciences.gr12.p2.acid.json',
  '/caps/generated/physics-12/caps.physicalsciences.gr12.p2.equil.json',
  '/caps/generated/physics-12/caps.physicalsciences.gr12.p2.org1.json',
  '/caps/generated/physics-12/caps.physicalsciences.gr12.p2.org2.json',
  '/caps/generated/physics-12/caps.physicalsciences.gr12.p2.rate.json',
  '/caps/generated/physics-12/caps.physicalsciences.gr12.p2.elchem.json',
  '/caps/generated/physics-12/caps.physicalsciences.gr12.p2.fert.json',
  '/caps/generated/physics-12/caps.physicalsciences.gr12.p1.mod2.json',
];

// Install: precache shell + Gr12 lessons
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      // Cache shell first (critical)
      cache.addAll(PRECACHE);
      // Then proactively cache Gr12 lessons (non-blocking)
      GR12_LESSONS.forEach(url => {
        fetch(url).then(resp => {
          if (resp.ok) cache.put(url, resp);
        }).catch(() => {}); // Silently skip if offline
      });
    })
  );
  self.skipWaiting();
});

// Activate: clean old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Fetch: cache-first for assets, network-first for navigation
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // Cache-first for static assets (wasm, js, css, json, fonts, images)
  if (url.pathname.match(/\.(wasm|js|css|json|woff2?|png|svg|ico)$/)) {
    event.respondWith(
      caches.match(event.request).then(cached => {
        if (cached) return cached;
        return fetch(event.request).then(response => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
          }
          return response;
        }).catch(() => {
          // Offline and not cached — return empty JSON for lesson files
          if (url.pathname.endsWith('.json')) {
            return new Response('{"error":"offline"}', {
              headers: { 'Content-Type': 'application/json' },
              status: 503
            });
          }
          return new Response('Offline', { status: 503 });
        });
      })
    );
    return;
  }

  // Network-first for HTML (SPA navigation)
  event.respondWith(
    fetch(event.request)
      .then(response => {
        const clone = response.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        return response;
      })
      .catch(() => caches.match('/index.html'))
  );
});

// Listen for messages from the app (e.g., "cache grade X content")
self.addEventListener('message', event => {
  if (event.data && event.data.type === 'CACHE_GRADE') {
    const grade = event.data.grade;
    const subjects = ['math', 'physics', 'natsci'];
    const cache_urls = [];
    subjects.forEach(subj => {
      // Try to cache all files for this grade
      for (let i = 0; i < 20; i++) {
        cache_urls.push(`/caps/generated/${subj}-${grade}/`);
      }
    });
    // Proactively cache
    caches.open(CACHE_NAME).then(cache => {
      cache_urls.forEach(url => {
        fetch(url).then(resp => { if (resp.ok) cache.put(url, resp); }).catch(() => {});
      });
    });
  }
});
