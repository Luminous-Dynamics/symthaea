// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Playlist E2E Tests
 *
 * Tests for playlist creation, management, and sharing features.
 */

import { test, expect } from '@playwright/test';

const testUser = {
  email: 'test-user@example.com',
  password: 'TestPassword123!',
};

test.describe('Playlist Management', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
    await page.getByLabel('Email').fill(testUser.email);
    await page.getByLabel('Password').fill(testUser.password);
    await page.getByRole('button', { name: 'Sign In' }).click();
    await page.waitForURL('/');
  });

  test.describe('Creation', () => {
    test('creates a new playlist', async ({ page }) => {
      await page.goto('/library');

      // Click create playlist button
      await page.getByRole('button', { name: /create playlist/i }).click();

      // Fill playlist details
      await page.getByLabel('Name').fill('E2E Test Playlist');
      await page.getByLabel('Description').fill('Created during E2E testing');

      // Set visibility
      await page.getByRole('radio', { name: /public/i }).check();

      // Create
      await page.getByRole('button', { name: /create/i }).click();

      // Verify redirect to new playlist
      await expect(page).toHaveURL(/\/playlist\/.+/);
      await expect(page.getByRole('heading', { name: 'E2E Test Playlist' })).toBeVisible();
    });

    test('creates playlist from selection', async ({ page }) => {
      await page.goto('/browse');

      // Select multiple songs
      await page.getByTestId('song-card').first().hover();
      await page.keyboard.down('Control');
      await page.getByTestId('song-card').first().click();
      await page.getByTestId('song-card').nth(1).click();
      await page.getByTestId('song-card').nth(2).click();
      await page.keyboard.up('Control');

      // Add to new playlist
      await page.getByRole('button', { name: /add to playlist/i }).click();
      await page.getByRole('menuitem', { name: /new playlist/i }).click();

      // Name the playlist
      await page.getByLabel('Name').fill('Quick Playlist');
      await page.getByRole('button', { name: /create/i }).click();

      // Verify songs were added
      await expect(page.getByText('3 tracks')).toBeVisible();
    });
  });

  test.describe('Editing', () => {
    test('renames a playlist', async ({ page }) => {
      await page.goto('/library');

      // Click on first playlist
      await page.getByTestId('playlist-item').first().click();

      // Open menu
      await page.getByRole('button', { name: /more options/i }).click();
      await page.getByRole('menuitem', { name: /rename/i }).click();

      // Enter new name
      const nameInput = page.getByLabel('Name');
      await nameInput.clear();
      await nameInput.fill('Renamed Playlist');
      await page.getByRole('button', { name: /save/i }).click();

      // Verify name changed
      await expect(page.getByRole('heading', { name: 'Renamed Playlist' })).toBeVisible();
    });

    test('updates playlist cover', async ({ page }) => {
      await page.goto('/library');
      await page.getByTestId('playlist-item').first().click();

      // Click edit cover
      await page.getByRole('button', { name: /edit cover/i }).click();

      // Upload new image
      const fileInput = page.locator('input[type="file"][accept*="image"]');
      await fileInput.setInputFiles({
        name: 'cover.jpg',
        mimeType: 'image/jpeg',
        buffer: Buffer.from('mock-image-data'),
      });

      // Save
      await page.getByRole('button', { name: /save/i }).click();

      // Verify cover updated
      await expect(page.getByText(/cover updated/i)).toBeVisible();
    });

    test('changes playlist visibility', async ({ page }) => {
      await page.goto('/library');
      await page.getByTestId('playlist-item').first().click();

      // Open settings
      await page.getByRole('button', { name: /settings/i }).click();

      // Toggle to private
      await page.getByRole('switch', { name: /public/i }).click();

      // Verify change
      await expect(page.getByText(/playlist is now private/i)).toBeVisible();
    });
  });

  test.describe('Track Management', () => {
    test('adds track to playlist', async ({ page }) => {
      // Go to a song page
      await page.goto('/browse');
      await page.getByTestId('song-card').first().click();

      // Add to playlist
      await page.getByRole('button', { name: /add to playlist/i }).click();
      await page.getByRole('menuitem').first().click();

      // Verify added
      await expect(page.getByText(/added to playlist/i)).toBeVisible();
    });

    test('removes track from playlist', async ({ page }) => {
      await page.goto('/library');
      await page.getByTestId('playlist-item').first().click();

      // Remove first track
      await page.getByTestId('track-row').first().hover();
      await page.getByRole('button', { name: /remove/i }).first().click();

      // Confirm
      await page.getByRole('button', { name: /confirm/i }).click();

      // Verify removed
      await expect(page.getByText(/track removed/i)).toBeVisible();
    });

    test('reorders tracks via drag and drop', async ({ page }) => {
      await page.goto('/library');
      await page.getByTestId('playlist-item').first().click();

      // Get first track title
      const firstTrack = page.getByTestId('track-row').first();
      const firstTrackTitle = await firstTrack.getByTestId('track-title').textContent();

      // Drag first track to third position
      const dragHandle = firstTrack.getByTestId('drag-handle');
      const thirdTrack = page.getByTestId('track-row').nth(2);

      await dragHandle.dragTo(thirdTrack);

      // Verify order changed
      const newThirdTrackTitle = await page.getByTestId('track-row').nth(2).getByTestId('track-title').textContent();
      expect(newThirdTrackTitle).toBe(firstTrackTitle);
    });
  });

  test.describe('Playback', () => {
    test('plays entire playlist', async ({ page }) => {
      await page.goto('/library');
      await page.getByTestId('playlist-item').first().click();

      // Click play button
      await page.getByRole('button', { name: /play all/i }).click();

      // Verify player is playing
      await expect(page.getByTestId('player-controls')).toBeVisible();
      await expect(page.getByTestId('play-pause-button')).toHaveAttribute('aria-pressed', 'true');
    });

    test('shuffles playlist', async ({ page }) => {
      await page.goto('/library');
      await page.getByTestId('playlist-item').first().click();

      // Enable shuffle
      await page.getByRole('button', { name: /shuffle/i }).click();

      // Start playback
      await page.getByRole('button', { name: /play all/i }).click();

      // Verify shuffle is active
      await expect(page.getByRole('button', { name: /shuffle/i })).toHaveAttribute('aria-pressed', 'true');
    });

    test('queues playlist', async ({ page }) => {
      await page.goto('/library');
      await page.getByTestId('playlist-item').first().click();

      // Add to queue
      await page.getByRole('button', { name: /add to queue/i }).click();

      // Verify queue updated
      await expect(page.getByText(/added to queue/i)).toBeVisible();
    });
  });

  test.describe('Sharing', () => {
    test('shares playlist link', async ({ page }) => {
      await page.goto('/library');
      await page.getByTestId('playlist-item').first().click();

      // Click share
      await page.getByRole('button', { name: /share/i }).click();

      // Verify share dialog
      await expect(page.getByRole('dialog')).toBeVisible();
      await expect(page.getByLabel('Share link')).toBeVisible();

      // Copy link
      await page.getByRole('button', { name: /copy link/i }).click();
      await expect(page.getByText(/link copied/i)).toBeVisible();
    });

    test('shares to social media', async ({ page }) => {
      await page.goto('/library');
      await page.getByTestId('playlist-item').first().click();

      await page.getByRole('button', { name: /share/i }).click();

      // Check social share buttons
      await expect(page.getByRole('button', { name: /twitter/i })).toBeVisible();
      await expect(page.getByRole('button', { name: /facebook/i })).toBeVisible();
    });

    test('makes playlist collaborative', async ({ page }) => {
      await page.goto('/library');
      await page.getByTestId('playlist-item').first().click();

      await page.getByRole('button', { name: /settings/i }).click();

      // Enable collaboration
      await page.getByRole('switch', { name: /collaborative/i }).click();

      // Verify collaboration enabled
      await expect(page.getByText(/collaboration enabled/i)).toBeVisible();

      // Invite collaborator
      await page.getByRole('button', { name: /invite/i }).click();
      await page.getByLabel('Email').fill('friend@example.com');
      await page.getByRole('button', { name: /send invite/i }).click();

      await expect(page.getByText(/invitation sent/i)).toBeVisible();
    });
  });

  test.describe('Smart Playlists', () => {
    test('creates smart playlist with filters', async ({ page }) => {
      await page.goto('/library');

      await page.getByRole('button', { name: /create playlist/i }).click();
      await page.getByRole('tab', { name: /smart playlist/i }).click();

      // Add rules
      await page.getByLabel('Name').fill('High Energy Electronic');

      await page.getByRole('button', { name: /add rule/i }).click();
      await page.getByRole('combobox', { name: /field/i }).selectOption('genre');
      await page.getByRole('combobox', { name: /operator/i }).selectOption('is');
      await page.getByRole('combobox', { name: /value/i }).selectOption('Electronic');

      await page.getByRole('button', { name: /add rule/i }).click();
      await page.locator('[data-rule-index="1"]').getByRole('combobox', { name: /field/i }).selectOption('bpm');
      await page.locator('[data-rule-index="1"]').getByRole('combobox', { name: /operator/i }).selectOption('greater than');
      await page.locator('[data-rule-index="1"]').getByLabel('Value').fill('120');

      // Create
      await page.getByRole('button', { name: /create/i }).click();

      // Verify playlist created with matched tracks
      await expect(page).toHaveURL(/\/playlist\/.+/);
      await expect(page.getByText(/auto-updating/i)).toBeVisible();
    });
  });

  test.describe('Offline Support', () => {
    test('downloads playlist for offline', async ({ page }) => {
      await page.goto('/library');
      await page.getByTestId('playlist-item').first().click();

      // Download for offline
      await page.getByRole('button', { name: /download/i }).click();

      // Verify download started
      await expect(page.getByText(/downloading/i)).toBeVisible();

      // Wait for completion (with timeout)
      await expect(page.getByText(/available offline/i)).toBeVisible({ timeout: 60000 });
    });
  });
});

test.describe('Public Playlist Discovery', () => {
  test('browses popular playlists', async ({ page }) => {
    await page.goto('/playlists');

    // Check popular section
    await expect(page.getByRole('heading', { name: /popular playlists/i })).toBeVisible();
    await expect(page.getByTestId('playlist-card')).toHaveCount({ min: 1 });
  });

  test('searches playlists', async ({ page }) => {
    await page.goto('/playlists');

    // Search
    await page.getByRole('searchbox').fill('chill');
    await page.keyboard.press('Enter');

    // Verify results
    await expect(page.getByText(/results for "chill"/i)).toBeVisible();
  });

  test('filters playlists by genre', async ({ page }) => {
    await page.goto('/playlists');

    // Filter by genre
    await page.getByRole('button', { name: /genre/i }).click();
    await page.getByRole('checkbox', { name: /electronic/i }).check();
    await page.getByRole('button', { name: /apply/i }).click();

    // Verify filter applied
    await expect(page.getByTestId('active-filter')).toContainText('Electronic');
  });

  test('follows a public playlist', async ({ page }) => {
    await page.goto('/login');
    await page.getByLabel('Email').fill(testUser.email);
    await page.getByLabel('Password').fill(testUser.password);
    await page.getByRole('button', { name: 'Sign In' }).click();

    await page.goto('/playlists');

    // Follow first playlist
    await page.getByTestId('playlist-card').first().click();
    await page.getByRole('button', { name: /follow/i }).click();

    // Verify followed
    await expect(page.getByRole('button', { name: /following/i })).toBeVisible();

    // Check in library
    await page.goto('/library');
    await expect(page.getByText(/following/i)).toBeVisible();
  });
});
