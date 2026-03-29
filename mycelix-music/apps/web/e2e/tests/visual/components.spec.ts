// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Visual Regression Tests
 *
 * Screenshot comparisons for UI consistency.
 */

import { test, expect } from '@playwright/test';

test.describe('Visual Regression - Components', () => {
  test.beforeEach(async ({ page }) => {
    // Disable animations for consistent screenshots
    await page.addStyleTag({
      content: `
        *, *::before, *::after {
          animation-duration: 0s !important;
          animation-delay: 0s !important;
          transition-duration: 0s !important;
          transition-delay: 0s !important;
        }
      `,
    });
  });

  test('player bar - idle state', async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('[data-testid="player"]');

    await expect(page.locator('[data-testid="player"]')).toHaveScreenshot('player-idle.png');
  });

  test('player bar - playing state', async ({ page }) => {
    await page.goto('/library');
    await page.click('[data-testid^="track-"] [data-testid="play-button"]:first-of-type');
    await page.waitForSelector('[data-testid="player-playing"]');

    // Wait for waveform to render
    await page.waitForTimeout(500);

    await expect(page.locator('[data-testid="player"]')).toHaveScreenshot('player-playing.png');
  });

  test('sidebar - collapsed and expanded', async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('[data-testid="sidebar"]');

    // Expanded state
    await expect(page.locator('[data-testid="sidebar"]')).toHaveScreenshot('sidebar-expanded.png');

    // Collapse sidebar
    await page.click('[data-testid="sidebar-toggle"]');
    await page.waitForTimeout(300);

    await expect(page.locator('[data-testid="sidebar"]')).toHaveScreenshot('sidebar-collapsed.png');
  });

  test('track card', async ({ page }) => {
    await page.goto('/library');
    await page.waitForSelector('[data-testid="track-card"]');

    // Normal state
    await expect(page.locator('[data-testid="track-card"]:first-of-type')).toHaveScreenshot('track-card.png');

    // Hover state
    await page.locator('[data-testid="track-card"]:first-of-type').hover();
    await expect(page.locator('[data-testid="track-card"]:first-of-type')).toHaveScreenshot('track-card-hover.png');
  });

  test('playlist grid', async ({ page }) => {
    await page.goto('/library/playlists');
    await page.waitForSelector('[data-testid="playlist-grid"]');

    await expect(page.locator('[data-testid="playlist-grid"]')).toHaveScreenshot('playlist-grid.png');
  });

  test('search results', async ({ page }) => {
    await page.goto('/search?q=test');
    await page.waitForSelector('[data-testid="search-results"]');

    await expect(page.locator('[data-testid="search-results"]')).toHaveScreenshot('search-results.png');
  });

  test('volume slider states', async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('[data-testid="volume-slider"]');

    // Full volume
    await expect(page.locator('[data-testid="volume-control"]')).toHaveScreenshot('volume-full.png');

    // Half volume
    await page.locator('[data-testid="volume-slider"]').click({ position: { x: 30, y: 5 } });
    await expect(page.locator('[data-testid="volume-control"]')).toHaveScreenshot('volume-half.png');

    // Muted
    await page.click('[data-testid="mute-button"]');
    await expect(page.locator('[data-testid="volume-control"]')).toHaveScreenshot('volume-muted.png');
  });

  test('modal dialogs', async ({ page }) => {
    await page.goto('/library');

    // Open settings modal
    await page.click('[data-testid="settings-button"]');
    await page.waitForSelector('[data-testid="settings-modal"]');

    await expect(page.locator('[data-testid="settings-modal"]')).toHaveScreenshot('settings-modal.png');
  });

  test('toast notifications', async ({ page }) => {
    await page.goto('/library');

    // Trigger a toast by liking a track
    await page.click('[data-testid^="track-"] [data-testid="like-button"]:first-of-type');
    await page.waitForSelector('[data-testid="toast"]');

    await expect(page.locator('[data-testid="toast"]')).toHaveScreenshot('toast-success.png');
  });
});

test.describe('Visual Regression - Pages', () => {
  test.beforeEach(async ({ page }) => {
    await page.addStyleTag({
      content: `
        *, *::before, *::after {
          animation-duration: 0s !important;
          transition-duration: 0s !important;
        }
      `,
    });
  });

  test('home page', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await expect(page).toHaveScreenshot('page-home.png', {
      fullPage: true,
      mask: [page.locator('[data-testid="dynamic-content"]')],
    });
  });

  test('library page', async ({ page }) => {
    await page.goto('/library');
    await page.waitForLoadState('networkidle');

    await expect(page).toHaveScreenshot('page-library.png', {
      fullPage: true,
    });
  });

  test('artist page', async ({ page }) => {
    await page.goto('/artist/test-artist');
    await page.waitForLoadState('networkidle');

    await expect(page).toHaveScreenshot('page-artist.png', {
      fullPage: true,
    });
  });

  test('playlist page', async ({ page }) => {
    await page.goto('/playlist/test-playlist');
    await page.waitForLoadState('networkidle');

    await expect(page).toHaveScreenshot('page-playlist.png', {
      fullPage: true,
    });
  });

  test('settings page', async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');

    await expect(page).toHaveScreenshot('page-settings.png', {
      fullPage: true,
    });
  });
});

test.describe('Visual Regression - Themes', () => {
  test('dark theme', async ({ page }) => {
    await page.goto('/');
    await page.evaluate(() => {
      document.documentElement.setAttribute('data-theme', 'dark');
    });
    await page.waitForTimeout(300);

    await expect(page).toHaveScreenshot('theme-dark.png');
  });

  test('light theme', async ({ page }) => {
    await page.goto('/');
    await page.evaluate(() => {
      document.documentElement.setAttribute('data-theme', 'light');
    });
    await page.waitForTimeout(300);

    await expect(page).toHaveScreenshot('theme-light.png');
  });

  test('studio theme', async ({ page }) => {
    await page.goto('/');
    await page.evaluate(() => {
      document.documentElement.setAttribute('data-theme', 'studio');
    });
    await page.waitForTimeout(300);

    await expect(page).toHaveScreenshot('theme-studio.png');
  });
});

test.describe('Visual Regression - Responsive', () => {
  test('mobile layout', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await expect(page).toHaveScreenshot('responsive-mobile.png');
  });

  test('tablet layout', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await expect(page).toHaveScreenshot('responsive-tablet.png');
  });

  test('desktop layout', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await expect(page).toHaveScreenshot('responsive-desktop.png');
  });
});
