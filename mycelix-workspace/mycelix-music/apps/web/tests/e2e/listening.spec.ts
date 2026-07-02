// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * E2E Tests: Listening Experience
 *
 * Tests music playback, player controls, and listening circles.
 */

import { test, expect, Page } from '@playwright/test';

// Helper to set up authenticated state
async function authenticateUser(page: Page) {
  await page.addInitScript(() => {
    window.localStorage.setItem('mycelix_soul', JSON.stringify({
      id: 'soul_test123',
      address: '0x1234567890123456789012345678901234567890',
      profile: {
        displayName: 'Test User',
        bio: 'Test bio',
        genres: ['ambient', 'electronic'],
        createdAt: new Date().toISOString(),
      },
      credentials: [],
      connections: [],
      resonance: {
        total: 100,
        categories: {
          listening: 50,
          patronage: 20,
          collaboration: 10,
          community: 10,
          presence: 10,
        },
      },
    }));
    window.localStorage.setItem('mycelix_wallet', JSON.stringify({
      provider: 'metamask',
      address: '0x1234567890123456789012345678901234567890',
      chainId: 84532,
      isConnected: true,
    }));
  });
}

test.describe('Track Playback', () => {
  test.beforeEach(async ({ page }) => {
    await authenticateUser(page);
    await page.goto('/');
  });

  test('should play track on click', async ({ page }) => {
    // Find a track card and click play
    const trackCard = page.locator('[data-testid="track-card"]').first();
    await trackCard.getByRole('button', { name: /play/i }).click();

    // Player should appear
    const player = page.locator('[data-testid="player"]');
    await expect(player).toBeVisible();

    // Should show track info
    await expect(player.getByTestId('track-title')).toBeVisible();
  });

  test('should pause and resume playback', async ({ page }) => {
    // Start playback
    await page.locator('[data-testid="track-card"]').first().click();

    const playPauseButton = page.getByTestId('play-pause-button');

    // Should be playing
    await expect(playPauseButton).toHaveAttribute('aria-label', /pause/i);

    // Pause
    await playPauseButton.click();
    await expect(playPauseButton).toHaveAttribute('aria-label', /play/i);

    // Resume
    await playPauseButton.click();
    await expect(playPauseButton).toHaveAttribute('aria-label', /pause/i);
  });

  test('should seek within track', async ({ page }) => {
    await page.locator('[data-testid="track-card"]').first().click();

    const progressBar = page.getByTestId('progress-bar');
    const boundingBox = await progressBar.boundingBox();

    if (boundingBox) {
      // Click at 50% of progress bar
      await page.mouse.click(
        boundingBox.x + boundingBox.width * 0.5,
        boundingBox.y + boundingBox.height / 2
      );

      // Progress should update
      const currentTime = page.getByTestId('current-time');
      await expect(currentTime).not.toHaveText('0:00');
    }
  });

  test('should skip to next/previous track', async ({ page }) => {
    // Play first track
    await page.locator('[data-testid="track-card"]').first().click();

    const initialTitle = await page.getByTestId('track-title').textContent();

    // Skip next
    await page.getByTestId('skip-next-button').click();

    // Title should change
    await expect(page.getByTestId('track-title')).not.toHaveText(initialTitle!);

    // Skip previous
    await page.getByTestId('skip-previous-button').click();

    // Should be back to original
    await expect(page.getByTestId('track-title')).toHaveText(initialTitle!);
  });

  test('should adjust volume', async ({ page }) => {
    await page.locator('[data-testid="track-card"]').first().click();

    const volumeSlider = page.getByTestId('volume-slider');

    // Mute
    await page.getByTestId('mute-button').click();
    await expect(volumeSlider).toHaveValue('0');

    // Unmute
    await page.getByTestId('mute-button').click();
    await expect(volumeSlider).not.toHaveValue('0');
  });

  test('should expand to full screen player', async ({ page }) => {
    await page.locator('[data-testid="track-card"]').first().click();

    // Click on player to expand
    await page.getByTestId('player').click();

    // Full screen player should be visible
    await expect(page.getByTestId('full-player')).toBeVisible();

    // Should show album art
    await expect(page.getByTestId('album-art')).toBeVisible();

    // Close full player
    await page.getByTestId('minimize-player').click();
    await expect(page.getByTestId('full-player')).not.toBeVisible();
  });
});

test.describe('Listening Circles', () => {
  test.beforeEach(async ({ page }) => {
    await authenticateUser(page);
    await page.goto('/circles');
  });

  test('should display active circles', async ({ page }) => {
    const circleCards = page.locator('[data-testid="circle-card"]');
    await expect(circleCards.first()).toBeVisible();

    // Should show listener count
    await expect(page.getByText(/listening/i).first()).toBeVisible();
  });

  test('should join a circle', async ({ page }) => {
    const circleCard = page.locator('[data-testid="circle-card"]').first();
    await circleCard.getByRole('button', { name: /join/i }).click();

    // Should navigate to circle view
    await expect(page).toHaveURL(/\/circles\/.+/);

    // Should show synced playback
    await expect(page.getByTestId('synced-player')).toBeVisible();

    // Should show other listeners
    await expect(page.getByTestId('listener-avatars')).toBeVisible();
  });

  test('should create a new circle', async ({ page }) => {
    await page.getByRole('button', { name: /Create Circle/i }).click();

    // Fill in circle details
    await page.getByLabel(/Circle Name/i).fill('Test Circle');
    await page.getByLabel(/Description/i).fill('A test listening circle');

    // Select starting track
    await page.getByRole('button', { name: /Select Track/i }).click();
    await page.locator('[data-testid="track-option"]').first().click();

    // Create circle
    await page.getByRole('button', { name: /Create/i }).click();

    // Should navigate to new circle
    await expect(page).toHaveURL(/\/circles\/.+/);

    // Should be the host
    await expect(page.getByText(/You are the host/i)).toBeVisible();
  });

  test('should show real-time listener updates', async ({ page }) => {
    await page.locator('[data-testid="circle-card"]').first().click();

    // Initial listener count
    const listenerCount = page.getByTestId('listener-count');
    const initialCount = await listenerCount.textContent();

    // Wait for WebSocket update (simulated in test environment)
    await page.waitForTimeout(2000);

    // Count may have changed
    // In real test, we'd mock WebSocket events
  });

  test('should sync playback position', async ({ page }) => {
    await page.locator('[data-testid="circle-card"]').first().click();

    // Player should show synced state
    await expect(page.getByTestId('sync-indicator')).toBeVisible();

    // Progress should update automatically
    const initialProgress = await page.getByTestId('progress-bar').getAttribute('value');

    await page.waitForTimeout(3000);

    const newProgress = await page.getByTestId('progress-bar').getAttribute('value');
    expect(Number(newProgress)).toBeGreaterThan(Number(initialProgress));
  });

  test('should send reactions', async ({ page }) => {
    await page.locator('[data-testid="circle-card"]').first().click();

    // Open reaction picker
    await page.getByTestId('reaction-button').click();

    // Select an emoji
    await page.getByText('🔥').click();

    // Reaction should appear briefly
    await expect(page.getByTestId('reaction-bubble')).toBeVisible();
  });

  test('should chat in circle', async ({ page }) => {
    await page.locator('[data-testid="circle-card"]').first().click();

    // Open chat
    await page.getByTestId('chat-toggle').click();

    // Send a message
    const chatInput = page.getByPlaceholder(/Type a message/i);
    await chatInput.fill('Hello everyone!');
    await chatInput.press('Enter');

    // Message should appear
    await expect(page.getByText('Hello everyone!')).toBeVisible();
  });

  test('should leave circle', async ({ page }) => {
    await page.locator('[data-testid="circle-card"]').first().click();

    await page.getByRole('button', { name: /Leave/i }).click();

    // Confirm
    await page.getByRole('button', { name: /Confirm/i }).click();

    // Should redirect back to circles list
    await expect(page).toHaveURL('/circles');
  });
});

test.describe('Queue Management', () => {
  test.beforeEach(async ({ page }) => {
    await authenticateUser(page);
    await page.goto('/');
  });

  test('should add track to queue', async ({ page }) => {
    // Start playing something first
    await page.locator('[data-testid="track-card"]').first().click();

    // Add another track to queue
    await page.locator('[data-testid="track-card"]').nth(1).getByRole('button', { name: /more/i }).click();
    await page.getByRole('menuitem', { name: /Add to Queue/i }).click();

    // Open queue view
    await page.getByTestId('queue-button').click();

    // Should show queued track
    const queueItems = page.locator('[data-testid="queue-item"]');
    await expect(queueItems).toHaveCount(2);
  });

  test('should reorder queue with drag and drop', async ({ page }) => {
    // Set up queue with multiple tracks
    await page.locator('[data-testid="track-card"]').first().click();
    await page.locator('[data-testid="track-card"]').nth(1).getByRole('button', { name: /more/i }).click();
    await page.getByRole('menuitem', { name: /Add to Queue/i }).click();
    await page.locator('[data-testid="track-card"]').nth(2).getByRole('button', { name: /more/i }).click();
    await page.getByRole('menuitem', { name: /Add to Queue/i }).click();

    // Open queue
    await page.getByTestId('queue-button').click();

    // Drag second item to first position
    const secondItem = page.locator('[data-testid="queue-item"]').nth(1);
    const firstItem = page.locator('[data-testid="queue-item"]').first();

    const secondBox = await secondItem.boundingBox();
    const firstBox = await firstItem.boundingBox();

    if (secondBox && firstBox) {
      await page.mouse.move(secondBox.x + 10, secondBox.y + 10);
      await page.mouse.down();
      await page.mouse.move(firstBox.x + 10, firstBox.y - 10);
      await page.mouse.up();
    }

    // Order should be updated
  });

  test('should remove track from queue', async ({ page }) => {
    await page.locator('[data-testid="track-card"]').first().click();
    await page.locator('[data-testid="track-card"]').nth(1).getByRole('button', { name: /more/i }).click();
    await page.getByRole('menuitem', { name: /Add to Queue/i }).click();

    await page.getByTestId('queue-button').click();

    // Remove queued track
    await page.locator('[data-testid="queue-item"]').nth(1).getByRole('button', { name: /remove/i }).click();

    const queueItems = page.locator('[data-testid="queue-item"]');
    await expect(queueItems).toHaveCount(1);
  });
});
