// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Player E2E Tests
 *
 * Tests critical user flows for the audio player.
 */

import { test, expect, Page } from '@playwright/test';

// ==================== Test Fixtures ====================

interface TestTrack {
  id: string;
  title: string;
  artist: string;
  duration: number;
}

const mockTrack: TestTrack = {
  id: 'test-track-1',
  title: 'Test Song',
  artist: 'Test Artist',
  duration: 180,
};

// ==================== Helper Functions ====================

async function waitForPlayerReady(page: Page): Promise<void> {
  await page.waitForSelector('[data-testid="player"]', { state: 'visible' });
  await page.waitForSelector('[data-testid="player-controls"]', { state: 'visible' });
}

async function playTrack(page: Page, trackId: string): Promise<void> {
  await page.click(`[data-testid="track-${trackId}"] [data-testid="play-button"]`);
  await page.waitForSelector('[data-testid="player-playing"]', { state: 'visible' });
}

async function pauseTrack(page: Page): Promise<void> {
  await page.click('[data-testid="pause-button"]');
  await page.waitForSelector('[data-testid="player-paused"]', { state: 'visible' });
}

// ==================== Tests ====================

test.describe('Audio Player', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to home page
    await page.goto('/');
    await waitForPlayerReady(page);
  });

  test('should display player controls', async ({ page }) => {
    // Verify all player controls are present
    await expect(page.locator('[data-testid="play-button"]')).toBeVisible();
    await expect(page.locator('[data-testid="prev-button"]')).toBeVisible();
    await expect(page.locator('[data-testid="next-button"]')).toBeVisible();
    await expect(page.locator('[data-testid="volume-slider"]')).toBeVisible();
    await expect(page.locator('[data-testid="progress-bar"]')).toBeVisible();
  });

  test('should play and pause track', async ({ page }) => {
    // Navigate to a track
    await page.goto('/library');
    await page.waitForSelector('[data-testid="track-list"]');

    // Click play on first track
    await page.click('[data-testid^="track-"] [data-testid="play-button"]:first-of-type');

    // Verify playing state
    await expect(page.locator('[data-testid="player-playing"]')).toBeVisible();
    await expect(page.locator('[data-testid="now-playing-title"]')).toBeVisible();

    // Pause
    await page.click('[data-testid="pause-button"]');
    await expect(page.locator('[data-testid="player-paused"]')).toBeVisible();

    // Resume
    await page.click('[data-testid="play-button"]');
    await expect(page.locator('[data-testid="player-playing"]')).toBeVisible();
  });

  test('should adjust volume', async ({ page }) => {
    const volumeSlider = page.locator('[data-testid="volume-slider"]');

    // Get initial volume
    const initialVolume = await volumeSlider.getAttribute('aria-valuenow');

    // Adjust volume
    await volumeSlider.click({ position: { x: 20, y: 5 } });

    // Verify volume changed
    const newVolume = await volumeSlider.getAttribute('aria-valuenow');
    expect(newVolume).not.toBe(initialVolume);
  });

  test('should mute and unmute', async ({ page }) => {
    // Click mute button
    await page.click('[data-testid="mute-button"]');
    await expect(page.locator('[data-testid="volume-muted"]')).toBeVisible();

    // Click again to unmute
    await page.click('[data-testid="mute-button"]');
    await expect(page.locator('[data-testid="volume-muted"]')).not.toBeVisible();
  });

  test('should seek within track', async ({ page }) => {
    // Start playing a track
    await page.goto('/library');
    await page.click('[data-testid^="track-"] [data-testid="play-button"]:first-of-type');
    await page.waitForSelector('[data-testid="player-playing"]');

    const progressBar = page.locator('[data-testid="progress-bar"]');

    // Get initial position
    const initialPosition = await page.locator('[data-testid="current-time"]').textContent();

    // Seek to middle
    const box = await progressBar.boundingBox();
    if (box) {
      await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
    }

    // Verify position changed
    await page.waitForTimeout(500);
    const newPosition = await page.locator('[data-testid="current-time"]').textContent();
    expect(newPosition).not.toBe(initialPosition);
  });

  test('should skip to next and previous tracks', async ({ page }) => {
    // Start playing from a playlist
    await page.goto('/playlist/featured');
    await page.click('[data-testid="play-all-button"]');
    await page.waitForSelector('[data-testid="player-playing"]');

    const currentTitle = await page.locator('[data-testid="now-playing-title"]').textContent();

    // Skip to next
    await page.click('[data-testid="next-button"]');
    await page.waitForTimeout(500);
    const nextTitle = await page.locator('[data-testid="now-playing-title"]').textContent();
    expect(nextTitle).not.toBe(currentTitle);

    // Skip to previous
    await page.click('[data-testid="prev-button"]');
    await page.waitForTimeout(500);
    const prevTitle = await page.locator('[data-testid="now-playing-title"]').textContent();
    expect(prevTitle).toBe(currentTitle);
  });

  test('should toggle shuffle mode', async ({ page }) => {
    await page.click('[data-testid="shuffle-button"]');
    await expect(page.locator('[data-testid="shuffle-active"]')).toBeVisible();

    await page.click('[data-testid="shuffle-button"]');
    await expect(page.locator('[data-testid="shuffle-active"]')).not.toBeVisible();
  });

  test('should cycle through repeat modes', async ({ page }) => {
    // Off -> All
    await page.click('[data-testid="repeat-button"]');
    await expect(page.locator('[data-testid="repeat-all"]')).toBeVisible();

    // All -> One
    await page.click('[data-testid="repeat-button"]');
    await expect(page.locator('[data-testid="repeat-one"]')).toBeVisible();

    // One -> Off
    await page.click('[data-testid="repeat-button"]');
    await expect(page.locator('[data-testid="repeat-off"]')).toBeVisible();
  });

  test('should show queue panel', async ({ page }) => {
    await page.click('[data-testid="queue-button"]');
    await expect(page.locator('[data-testid="queue-panel"]')).toBeVisible();

    // Close queue
    await page.click('[data-testid="queue-button"]');
    await expect(page.locator('[data-testid="queue-panel"]')).not.toBeVisible();
  });

  test('should reorder queue via drag and drop', async ({ page }) => {
    // Open queue
    await page.click('[data-testid="queue-button"]');
    await page.waitForSelector('[data-testid="queue-panel"]');

    const firstItem = page.locator('[data-testid="queue-item"]:nth-child(1)');
    const secondItem = page.locator('[data-testid="queue-item"]:nth-child(2)');

    const firstTitle = await firstItem.locator('[data-testid="queue-item-title"]').textContent();
    const secondTitle = await secondItem.locator('[data-testid="queue-item-title"]').textContent();

    // Drag first item to second position
    await firstItem.dragTo(secondItem);

    // Verify order changed
    const newFirstTitle = await page.locator('[data-testid="queue-item"]:nth-child(1) [data-testid="queue-item-title"]').textContent();
    expect(newFirstTitle).toBe(secondTitle);
  });
});

test.describe('Player Keyboard Shortcuts', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/library');
    await page.click('[data-testid^="track-"] [data-testid="play-button"]:first-of-type');
    await page.waitForSelector('[data-testid="player-playing"]');
  });

  test('should toggle play/pause with space', async ({ page }) => {
    await page.keyboard.press('Space');
    await expect(page.locator('[data-testid="player-paused"]')).toBeVisible();

    await page.keyboard.press('Space');
    await expect(page.locator('[data-testid="player-playing"]')).toBeVisible();
  });

  test('should skip tracks with arrow keys', async ({ page }) => {
    const currentTitle = await page.locator('[data-testid="now-playing-title"]').textContent();

    await page.keyboard.press('ArrowRight');
    await page.waitForTimeout(500);
    const nextTitle = await page.locator('[data-testid="now-playing-title"]').textContent();
    expect(nextTitle).not.toBe(currentTitle);
  });

  test('should adjust volume with up/down arrows', async ({ page }) => {
    const getVolume = async () => {
      return await page.locator('[data-testid="volume-slider"]').getAttribute('aria-valuenow');
    };

    const initialVolume = await getVolume();

    await page.keyboard.press('ArrowUp');
    const higherVolume = await getVolume();
    expect(Number(higherVolume)).toBeGreaterThan(Number(initialVolume));

    await page.keyboard.press('ArrowDown');
    const lowerVolume = await getVolume();
    expect(Number(lowerVolume)).toBeLessThan(Number(higherVolume));
  });

  test('should mute with M key', async ({ page }) => {
    await page.keyboard.press('m');
    await expect(page.locator('[data-testid="volume-muted"]')).toBeVisible();

    await page.keyboard.press('m');
    await expect(page.locator('[data-testid="volume-muted"]')).not.toBeVisible();
  });
});

test.describe('Player Persistence', () => {
  test('should remember volume after page reload', async ({ page }) => {
    await page.goto('/');

    // Set specific volume
    const volumeSlider = page.locator('[data-testid="volume-slider"]');
    await volumeSlider.click({ position: { x: 50, y: 5 } });
    const volumeBefore = await volumeSlider.getAttribute('aria-valuenow');

    // Reload page
    await page.reload();
    await waitForPlayerReady(page);

    // Verify volume persisted
    const volumeAfter = await page.locator('[data-testid="volume-slider"]').getAttribute('aria-valuenow');
    expect(volumeAfter).toBe(volumeBefore);
  });

  test('should remember shuffle/repeat state after reload', async ({ page }) => {
    await page.goto('/');

    // Enable shuffle and repeat
    await page.click('[data-testid="shuffle-button"]');
    await page.click('[data-testid="repeat-button"]');

    await page.reload();
    await waitForPlayerReady(page);

    // Verify states persisted
    await expect(page.locator('[data-testid="shuffle-active"]')).toBeVisible();
    await expect(page.locator('[data-testid="repeat-all"]')).toBeVisible();
  });
});
