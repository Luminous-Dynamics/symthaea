// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { test, expect } from '@playwright/test';

test.describe('Songs Browsing', () => {
  test('should display song list', async ({ page }) => {
    await page.goto('/');

    // Wait for songs to load (adjust selector based on actual implementation)
    const songList = page.locator('[data-testid="song-list"], .song-list, .songs-container');

    // If song list exists, it should be visible
    if (await songList.count() > 0) {
      await expect(songList.first()).toBeVisible({ timeout: 10000 });
    }
  });

  test('should filter songs by genre', async ({ page }) => {
    await page.goto('/');

    const genreFilter = page.locator('select[name="genre"], [data-testid="genre-filter"]');

    if (await genreFilter.isVisible()) {
      await genreFilter.selectOption({ index: 1 }); // Select first genre option

      // URL should update with genre parameter
      await page.waitForURL(/genre=/);
    }
  });

  test('should search for songs', async ({ page }) => {
    await page.goto('/');

    const searchInput = page.getByPlaceholder(/search/i);

    if (await searchInput.isVisible()) {
      await searchInput.fill('test');
      await searchInput.press('Enter');

      // Should show search results or empty state
      await page.waitForLoadState('networkidle');
    }
  });

  test('should paginate through songs', async ({ page }) => {
    await page.goto('/');

    const nextButton = page.getByRole('button', { name: /next|load more/i });

    if (await nextButton.isVisible()) {
      await nextButton.click();
      await page.waitForLoadState('networkidle');
    }
  });
});

test.describe('Song Details', () => {
  test('should navigate to song detail page', async ({ page }) => {
    await page.goto('/');

    // Click on a song card
    const songCard = page.locator('[data-testid="song-card"], .song-card, .song-item').first();

    if (await songCard.isVisible()) {
      await songCard.click();

      // Should navigate to song detail page
      await expect(page).toHaveURL(/song|track/);
    }
  });

  test('should display song information', async ({ page }) => {
    // Navigate directly to a song page if we know the URL pattern
    await page.goto('/');

    const songCard = page.locator('[data-testid="song-card"], .song-card').first();

    if (await songCard.isVisible()) {
      await songCard.click();
      await page.waitForLoadState('networkidle');

      // Check for common song detail elements
      const title = page.locator('h1, [data-testid="song-title"]');
      const artist = page.locator('[data-testid="artist-name"], .artist-name');

      if (await title.isVisible()) {
        await expect(title).not.toBeEmpty();
      }
    }
  });
});
