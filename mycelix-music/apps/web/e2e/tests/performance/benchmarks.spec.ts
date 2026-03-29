// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Performance Benchmark Tests
 *
 * Measures and validates performance metrics.
 */

import { test, expect, Page } from '@playwright/test';

// ==================== Types ====================

interface PerformanceMetrics {
  fcp: number;           // First Contentful Paint
  lcp: number;           // Largest Contentful Paint
  fid: number;           // First Input Delay
  cls: number;           // Cumulative Layout Shift
  ttfb: number;          // Time to First Byte
  tti: number;           // Time to Interactive
  totalBlockingTime: number;
  domContentLoaded: number;
  load: number;
  jsHeapSize: number;
}

interface ThresholdConfig {
  fcp: number;
  lcp: number;
  cls: number;
  ttfb: number;
  tti: number;
  totalBlockingTime: number;
}

// ==================== Thresholds ====================

const THRESHOLDS: ThresholdConfig = {
  fcp: 1800,              // 1.8s
  lcp: 2500,              // 2.5s
  cls: 0.1,               // 0.1
  ttfb: 600,              // 600ms
  tti: 3800,              // 3.8s
  totalBlockingTime: 300, // 300ms
};

// ==================== Helpers ====================

async function getPerformanceMetrics(page: Page): Promise<PerformanceMetrics> {
  return await page.evaluate(() => {
    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    const paint = performance.getEntriesByType('paint');
    const fcp = paint.find(e => e.name === 'first-contentful-paint');

    // Get LCP from PerformanceObserver
    let lcp = 0;
    const lcpEntries = performance.getEntriesByType('largest-contentful-paint');
    if (lcpEntries.length > 0) {
      lcp = lcpEntries[lcpEntries.length - 1].startTime;
    }

    // Get CLS
    let cls = 0;
    const layoutShiftEntries = performance.getEntriesByType('layout-shift') as (PerformanceEntry & { value: number; hadRecentInput: boolean })[];
    layoutShiftEntries.forEach(entry => {
      if (!entry.hadRecentInput) {
        cls += entry.value;
      }
    });

    // Memory (Chrome only)
    const memory = (performance as Performance & { memory?: { usedJSHeapSize: number } }).memory;

    return {
      fcp: fcp?.startTime || 0,
      lcp,
      fid: 0, // Requires user interaction
      cls,
      ttfb: navigation.responseStart - navigation.requestStart,
      tti: navigation.domInteractive - navigation.fetchStart,
      totalBlockingTime: 0, // Would need long task observer
      domContentLoaded: navigation.domContentLoadedEventEnd - navigation.fetchStart,
      load: navigation.loadEventEnd - navigation.fetchStart,
      jsHeapSize: memory?.usedJSHeapSize || 0,
    };
  });
}

async function measureInteraction(page: Page, action: () => Promise<void>): Promise<number> {
  const start = Date.now();
  await action();
  return Date.now() - start;
}

// ==================== Tests ====================

test.describe('Performance Benchmarks', () => {
  test.describe('Core Web Vitals', () => {
    test('home page meets FCP threshold', async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('load');

      const metrics = await getPerformanceMetrics(page);
      console.log(`FCP: ${metrics.fcp}ms (threshold: ${THRESHOLDS.fcp}ms)`);

      expect(metrics.fcp).toBeLessThan(THRESHOLDS.fcp);
    });

    test('home page meets LCP threshold', async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('networkidle');
      await page.waitForTimeout(1000); // Allow LCP to settle

      const metrics = await getPerformanceMetrics(page);
      console.log(`LCP: ${metrics.lcp}ms (threshold: ${THRESHOLDS.lcp}ms)`);

      expect(metrics.lcp).toBeLessThan(THRESHOLDS.lcp);
    });

    test('home page meets CLS threshold', async ({ page }) => {
      await page.goto('/');
      await page.waitForLoadState('networkidle');
      await page.waitForTimeout(2000); // Allow layout to stabilize

      const metrics = await getPerformanceMetrics(page);
      console.log(`CLS: ${metrics.cls} (threshold: ${THRESHOLDS.cls})`);

      expect(metrics.cls).toBeLessThan(THRESHOLDS.cls);
    });

    test('API response meets TTFB threshold', async ({ page }) => {
      await page.goto('/');

      const metrics = await getPerformanceMetrics(page);
      console.log(`TTFB: ${metrics.ttfb}ms (threshold: ${THRESHOLDS.ttfb}ms)`);

      expect(metrics.ttfb).toBeLessThan(THRESHOLDS.ttfb);
    });
  });

  test.describe('Page Load Performance', () => {
    const pages = [
      { name: 'Home', path: '/' },
      { name: 'Library', path: '/library' },
      { name: 'Search', path: '/search' },
      { name: 'Playlist', path: '/playlist/featured' },
      { name: 'Artist', path: '/artist/test-artist' },
      { name: 'Settings', path: '/settings' },
    ];

    for (const { name, path } of pages) {
      test(`${name} page loads within threshold`, async ({ page }) => {
        const start = Date.now();
        await page.goto(path);
        await page.waitForLoadState('networkidle');
        const loadTime = Date.now() - start;

        console.log(`${name} page load: ${loadTime}ms`);
        expect(loadTime).toBeLessThan(5000); // 5 second max
      });
    }
  });

  test.describe('Interaction Performance', () => {
    test('play button responds quickly', async ({ page }) => {
      await page.goto('/library');
      await page.waitForSelector('[data-testid^="track-"]');

      const responseTime = await measureInteraction(page, async () => {
        await page.click('[data-testid^="track-"] [data-testid="play-button"]:first-of-type');
        await page.waitForSelector('[data-testid="player-playing"]');
      });

      console.log(`Play button response: ${responseTime}ms`);
      expect(responseTime).toBeLessThan(500);
    });

    test('search responds quickly', async ({ page }) => {
      await page.goto('/search');
      await page.waitForSelector('[data-testid="search-input"]');

      const responseTime = await measureInteraction(page, async () => {
        await page.fill('[data-testid="search-input"]', 'test');
        await page.waitForSelector('[data-testid="search-results"]');
      });

      console.log(`Search response: ${responseTime}ms`);
      expect(responseTime).toBeLessThan(1000);
    });

    test('navigation is instant', async ({ page }) => {
      await page.goto('/');

      const responseTime = await measureInteraction(page, async () => {
        await page.click('[data-testid="nav-library"]');
        await page.waitForSelector('[data-testid="library-page"]');
      });

      console.log(`Navigation response: ${responseTime}ms`);
      expect(responseTime).toBeLessThan(300);
    });

    test('volume slider responds smoothly', async ({ page }) => {
      await page.goto('/');
      await page.waitForSelector('[data-testid="volume-slider"]');

      const slider = page.locator('[data-testid="volume-slider"]');
      const box = await slider.boundingBox();

      if (box) {
        const measurements: number[] = [];

        // Simulate smooth drag
        for (let i = 0; i < 10; i++) {
          const start = Date.now();
          await page.mouse.move(box.x + (box.width * i) / 10, box.y + box.height / 2);
          measurements.push(Date.now() - start);
        }

        const avgResponse = measurements.reduce((a, b) => a + b, 0) / measurements.length;
        console.log(`Volume slider avg response: ${avgResponse}ms`);
        expect(avgResponse).toBeLessThan(16); // 60fps target
      }
    });
  });

  test.describe('Memory Performance', () => {
    test('memory usage stays stable during playback', async ({ page }) => {
      await page.goto('/library');
      await page.click('[data-testid^="track-"] [data-testid="play-button"]:first-of-type');
      await page.waitForSelector('[data-testid="player-playing"]');

      // Initial memory measurement
      const initialMetrics = await getPerformanceMetrics(page);
      const initialHeap = initialMetrics.jsHeapSize;

      // Let it play for a while
      await page.waitForTimeout(10000);

      // Final memory measurement
      const finalMetrics = await getPerformanceMetrics(page);
      const finalHeap = finalMetrics.jsHeapSize;

      const memoryGrowth = ((finalHeap - initialHeap) / initialHeap) * 100;
      console.log(`Memory growth: ${memoryGrowth.toFixed(2)}%`);

      // Allow up to 20% memory growth
      expect(memoryGrowth).toBeLessThan(20);
    });

    test('no memory leaks during navigation', async ({ page }) => {
      await page.goto('/');

      const initialMetrics = await getPerformanceMetrics(page);
      const initialHeap = initialMetrics.jsHeapSize;

      // Navigate between pages multiple times
      const pages = ['/library', '/search', '/settings', '/'];
      for (let i = 0; i < 3; i++) {
        for (const path of pages) {
          await page.goto(path);
          await page.waitForLoadState('networkidle');
        }
      }

      // Force garbage collection if available
      await page.evaluate(() => {
        if ((window as Window & { gc?: () => void }).gc) {
          (window as Window & { gc?: () => void }).gc!();
        }
      });

      const finalMetrics = await getPerformanceMetrics(page);
      const finalHeap = finalMetrics.jsHeapSize;

      const memoryGrowth = ((finalHeap - initialHeap) / initialHeap) * 100;
      console.log(`Memory growth after navigation: ${memoryGrowth.toFixed(2)}%`);

      // Should not grow more than 50%
      expect(memoryGrowth).toBeLessThan(50);
    });
  });

  test.describe('Bundle Performance', () => {
    test('JavaScript bundle size is acceptable', async ({ page }) => {
      const jsSizes: number[] = [];

      page.on('response', async (response) => {
        const url = response.url();
        if (url.endsWith('.js') || url.includes('.js?')) {
          const headers = response.headers();
          const size = parseInt(headers['content-length'] || '0', 10);
          if (size > 0) {
            jsSizes.push(size);
          }
        }
      });

      await page.goto('/');
      await page.waitForLoadState('networkidle');

      const totalJS = jsSizes.reduce((a, b) => a + b, 0);
      const totalKB = totalJS / 1024;

      console.log(`Total JS size: ${totalKB.toFixed(2)}KB`);

      // Should be under 500KB
      expect(totalKB).toBeLessThan(500);
    });

    test('CSS bundle size is acceptable', async ({ page }) => {
      const cssSizes: number[] = [];

      page.on('response', async (response) => {
        const url = response.url();
        if (url.endsWith('.css') || url.includes('.css?')) {
          const headers = response.headers();
          const size = parseInt(headers['content-length'] || '0', 10);
          if (size > 0) {
            cssSizes.push(size);
          }
        }
      });

      await page.goto('/');
      await page.waitForLoadState('networkidle');

      const totalCSS = cssSizes.reduce((a, b) => a + b, 0);
      const totalKB = totalCSS / 1024;

      console.log(`Total CSS size: ${totalKB.toFixed(2)}KB`);

      // Should be under 100KB
      expect(totalKB).toBeLessThan(100);
    });
  });

  test.describe('Animation Performance', () => {
    test('waveform animation runs at 60fps', async ({ page }) => {
      await page.goto('/library');
      await page.click('[data-testid^="track-"] [data-testid="play-button"]:first-of-type');
      await page.waitForSelector('[data-testid="player-playing"]');

      // Measure frame rate
      const frameRates = await page.evaluate(async () => {
        return new Promise<number[]>((resolve) => {
          const rates: number[] = [];
          let lastTime = performance.now();
          let frameCount = 0;

          function measureFrame() {
            const now = performance.now();
            frameCount++;

            if (now - lastTime >= 1000) {
              rates.push(frameCount);
              frameCount = 0;
              lastTime = now;

              if (rates.length >= 3) {
                resolve(rates);
                return;
              }
            }

            requestAnimationFrame(measureFrame);
          }

          requestAnimationFrame(measureFrame);
        });
      });

      const avgFPS = frameRates.reduce((a, b) => a + b, 0) / frameRates.length;
      console.log(`Average FPS: ${avgFPS}`);

      // Should maintain at least 55 FPS
      expect(avgFPS).toBeGreaterThan(55);
    });
  });
});
