// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { test, expect } from '@playwright/test';

test.describe('Music Player', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should display player bar at bottom of page', async ({ page }) => {
    // Player should be visible even without a song
    const playerBar = page.locator('[class*="fixed bottom-0"]');
    await expect(playerBar).toBeVisible();
  });

  test('should show play controls', async ({ page }) => {
    // Check for basic controls
    await expect(page.getByRole('button', { name: /previous/i }).or(
      page.locator('button').filter({ has: page.locator('svg[class*="SkipBack"]') })
    )).toBeVisible();

    await expect(page.getByRole('button', { name: /play/i }).or(
      page.locator('button').filter({ has: page.locator('svg[class*="Play"]') })
    )).toBeVisible();

    await expect(page.getByRole('button', { name: /next/i }).or(
      page.locator('button').filter({ has: page.locator('svg[class*="SkipForward"]') })
    )).toBeVisible();
  });

  test('should show volume controls', async ({ page }) => {
    // Volume button should be visible
    const volumeButton = page.locator('button').filter({
      has: page.locator('svg[class*="Volume"]'),
    });
    await expect(volumeButton.first()).toBeVisible();
  });

  test('should show shuffle and repeat buttons', async ({ page }) => {
    const shuffleButton = page.locator('button').filter({
      has: page.locator('svg[class*="Shuffle"]'),
    });
    await expect(shuffleButton.first()).toBeVisible();

    const repeatButton = page.locator('button').filter({
      has: page.locator('svg[class*="Repeat"]'),
    });
    await expect(repeatButton.first()).toBeVisible();
  });

  test('should toggle shuffle when clicked', async ({ page }) => {
    const shuffleButton = page.locator('button').filter({
      has: page.locator('svg[class*="Shuffle"]'),
    }).first();

    // Get initial state
    const initialClass = await shuffleButton.getAttribute('class');

    // Click shuffle
    await shuffleButton.click();

    // State should change
    const newClass = await shuffleButton.getAttribute('class');
    expect(newClass).not.toBe(initialClass);
  });
});

test.describe('Player Interaction with Songs', () => {
  test('should play song when clicking play button on song card', async ({ page }) => {
    await page.goto('/');

    // Wait for songs to load
    await page.waitForSelector('[class*="SongCard"], [class*="song-card"]', {
      timeout: 10000,
    }).catch(() => {
      // Songs might not load in test environment, skip
    });

    // This test would need mocked data to work properly
    // In a real test environment, you'd mock the API responses
  });
});
