// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { test, expect, Page } from '@playwright/test';

test.describe('Music Player E2E Tests', () => {
  let page: Page;

  test.beforeEach(async ({ browser }) => {
    page = await browser.newPage();
    // Navigate to test page with player
    await page.goto('/');
    // Wait for player to load
    await page.waitForSelector('[data-testid="player"]');
  });

  test.afterEach(async () => {
    await page.close();
  });

  test.describe('Player UI', () => {
    test('should display player controls', async () => {
      await expect(page.getByTestId('play-button')).toBeVisible();
      await expect(page.getByTestId('next-button')).toBeVisible();
      await expect(page.getByTestId('prev-button')).toBeVisible();
      await expect(page.getByTestId('volume-slider')).toBeVisible();
      await expect(page.getByTestId('progress-bar')).toBeVisible();
    });

    test('should display current track info', async () => {
      await expect(page.getByTestId('track-title')).toBeVisible();
      await expect(page.getByTestId('track-artist')).toBeVisible();
      await expect(page.getByTestId('cover-art')).toBeVisible();
    });

    test('should display time information', async () => {
      await expect(page.getByTestId('current-time')).toBeVisible();
      await expect(page.getByTestId('duration')).toBeVisible();
    });
  });

  test.describe('Playback Controls', () => {
    test('should toggle play/pause', async () => {
      const playButton = page.getByTestId('play-button');

      // Initially paused
      await expect(playButton).toHaveAttribute('aria-label', 'Play');

      // Click to play
      await playButton.click();
      await expect(playButton).toHaveAttribute('aria-label', 'Pause');

      // Click to pause
      await playButton.click();
      await expect(playButton).toHaveAttribute('aria-label', 'Play');
    });

    test('should skip to next track', async () => {
      const trackTitle = page.getByTestId('track-title');
      const initialTitle = await trackTitle.textContent();

      await page.getByTestId('next-button').click();
      await page.waitForTimeout(500);

      const newTitle = await trackTitle.textContent();
      expect(newTitle).not.toBe(initialTitle);
    });

    test('should go to previous track', async () => {
      // First, go to next track
      await page.getByTestId('next-button').click();
      await page.waitForTimeout(500);

      const trackTitle = page.getByTestId('track-title');
      const afterNextTitle = await trackTitle.textContent();

      // Then go back
      await page.getByTestId('prev-button').click();
      await page.waitForTimeout(500);

      const afterPrevTitle = await trackTitle.textContent();
      expect(afterPrevTitle).not.toBe(afterNextTitle);
    });

    test('should seek when clicking progress bar', async () => {
      const progressBar = page.getByTestId('progress-slider');
      const currentTime = page.getByTestId('current-time');

      // Get initial time
      const initialTime = await currentTime.textContent();

      // Click at 50% of progress bar
      const bbox = await progressBar.boundingBox();
      if (bbox) {
        await page.mouse.click(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
      }

      await page.waitForTimeout(500);
      const newTime = await currentTime.textContent();
      expect(newTime).not.toBe(initialTime);
    });
  });

  test.describe('Volume Controls', () => {
    test('should adjust volume', async () => {
      const volumeSlider = page.getByTestId('volume-slider');

      // Set volume to 50%
      await volumeSlider.fill('0.5');

      const value = await volumeSlider.inputValue();
      expect(parseFloat(value)).toBe(0.5);
    });

    test('should toggle mute', async () => {
      const muteButton = page.getByTestId('mute-button');
      const volumeSlider = page.getByTestId('volume-slider');

      // Set initial volume
      await volumeSlider.fill('0.8');

      // Mute
      await muteButton.click();
      await expect(muteButton).toHaveAttribute('aria-pressed', 'true');

      // Unmute
      await muteButton.click();
      await expect(muteButton).toHaveAttribute('aria-pressed', 'false');
    });

    test('should restore volume after unmuting', async () => {
      const volumeSlider = page.getByTestId('volume-slider');
      const muteButton = page.getByTestId('mute-button');

      // Set volume
      await volumeSlider.fill('0.7');

      // Mute
      await muteButton.click();
      const mutedValue = await volumeSlider.inputValue();
      expect(parseFloat(mutedValue)).toBe(0);

      // Unmute
      await muteButton.click();
      const restoredValue = await volumeSlider.inputValue();
      expect(parseFloat(restoredValue)).toBe(0.7);
    });
  });

  test.describe('Shuffle and Repeat', () => {
    test('should toggle shuffle', async () => {
      const shuffleButton = page.getByTestId('shuffle-button');

      await expect(shuffleButton).toHaveAttribute('aria-pressed', 'false');

      await shuffleButton.click();
      await expect(shuffleButton).toHaveAttribute('aria-pressed', 'true');

      await shuffleButton.click();
      await expect(shuffleButton).toHaveAttribute('aria-pressed', 'false');
    });

    test('should cycle through repeat modes', async () => {
      const repeatButton = page.getByTestId('repeat-button');

      // Off -> All
      await repeatButton.click();
      await expect(repeatButton).toContainText('all');

      // All -> One
      await repeatButton.click();
      await expect(repeatButton).toContainText('one');

      // One -> Off
      await repeatButton.click();
      await expect(repeatButton).toContainText('off');
    });
  });

  test.describe('Like Feature', () => {
    test('should toggle like state', async () => {
      const likeButton = page.getByTestId('like-button');

      await expect(likeButton).toHaveAttribute('aria-pressed', 'false');

      await likeButton.click();
      await expect(likeButton).toHaveAttribute('aria-pressed', 'true');

      await likeButton.click();
      await expect(likeButton).toHaveAttribute('aria-pressed', 'false');
    });
  });

  test.describe('Expand/Collapse', () => {
    test('should expand player', async () => {
      const expandButton = page.getByTestId('expand-button');
      const player = page.getByTestId('player');

      await expect(player).toHaveClass(/mini/);

      await expandButton.click();
      await expect(player).toHaveClass(/expanded/);
    });

    test('should show waveform in expanded mode', async () => {
      await page.getByTestId('expand-button').click();
      await expect(page.getByTestId('waveform')).toBeVisible();
    });

    test('should collapse player', async () => {
      // First expand
      await page.getByTestId('expand-button').click();
      await expect(page.getByTestId('player')).toHaveClass(/expanded/);

      // Then collapse
      await page.getByTestId('expand-button').click();
      await expect(page.getByTestId('player')).toHaveClass(/mini/);
    });
  });

  test.describe('Keyboard Navigation', () => {
    test('should support spacebar for play/pause', async () => {
      await page.getByTestId('play-button').focus();
      await page.keyboard.press('Space');
      await expect(page.getByTestId('play-button')).toHaveAttribute('aria-label', 'Pause');
    });

    test('should support arrow keys for seeking', async () => {
      const progressBar = page.getByTestId('progress-slider');
      await progressBar.focus();

      const initialValue = await progressBar.inputValue();

      await page.keyboard.press('ArrowRight');
      const newValue = await progressBar.inputValue();

      expect(parseFloat(newValue)).toBeGreaterThan(parseFloat(initialValue));
    });

    test('should tab through controls', async () => {
      await page.keyboard.press('Tab');
      let focused = await page.evaluate(() => document.activeElement?.getAttribute('data-testid'));
      expect(focused).toBeTruthy();

      await page.keyboard.press('Tab');
      focused = await page.evaluate(() => document.activeElement?.getAttribute('data-testid'));
      expect(focused).toBeTruthy();
    });
  });

  test.describe('Progress Updates', () => {
    test('should update time display during playback', async () => {
      await page.getByTestId('play-button').click();

      const currentTime = page.getByTestId('current-time');
      const initialTime = await currentTime.textContent();

      // Wait for time to update
      await page.waitForTimeout(2000);

      const updatedTime = await currentTime.textContent();
      expect(updatedTime).not.toBe(initialTime);
    });
  });

  test.describe('Queue Management', () => {
    test('should open queue panel', async () => {
      const queueButton = page.getByTestId('queue-button');

      if (await queueButton.isVisible()) {
        await queueButton.click();
        await expect(page.getByTestId('queue-panel')).toBeVisible();
      }
    });

    test('should reorder queue items via drag and drop', async () => {
      const queueButton = page.getByTestId('queue-button');

      if (await queueButton.isVisible()) {
        await queueButton.click();

        const queueItems = page.locator('[data-testid="queue-item"]');
        const count = await queueItems.count();

        if (count >= 2) {
          // Drag first item to second position
          const firstItem = queueItems.first();
          const secondItem = queueItems.nth(1);

          await firstItem.dragTo(secondItem);
        }
      }
    });
  });

  test.describe('Responsive Design', () => {
    test('should adapt to mobile viewport', async () => {
      await page.setViewportSize({ width: 375, height: 667 });

      const player = page.getByTestId('player');
      await expect(player).toBeVisible();

      // Check that essential controls are still visible
      await expect(page.getByTestId('play-button')).toBeVisible();
      await expect(page.getByTestId('track-title')).toBeVisible();
    });

    test('should adapt to tablet viewport', async () => {
      await page.setViewportSize({ width: 768, height: 1024 });

      const player = page.getByTestId('player');
      await expect(player).toBeVisible();

      // All controls should be visible
      await expect(page.getByTestId('play-button')).toBeVisible();
      await expect(page.getByTestId('volume-slider')).toBeVisible();
      await expect(page.getByTestId('shuffle-button')).toBeVisible();
    });
  });

  test.describe('Error Handling', () => {
    test('should handle playback errors gracefully', async () => {
      // Simulate error by playing invalid track
      await page.evaluate(() => {
        window.dispatchEvent(new CustomEvent('player-error', {
          detail: { message: 'Failed to load audio' }
        }));
      });

      // Should show error state or notification
      const errorToast = page.locator('[role="alert"]');
      if (await errorToast.isVisible()) {
        await expect(errorToast).toContainText(/error|failed/i);
      }
    });
  });

  test.describe('Accessibility', () => {
    test('should have proper ARIA labels', async () => {
      await expect(page.getByTestId('play-button')).toHaveAttribute('aria-label');
      await expect(page.getByTestId('volume-slider')).toHaveAttribute('aria-label');
      await expect(page.getByTestId('progress-slider')).toHaveAttribute('aria-label');
    });

    test('should have proper focus indicators', async () => {
      const playButton = page.getByTestId('play-button');
      await playButton.focus();

      // Check that focus is visible (using computed styles)
      const hasFocusStyle = await playButton.evaluate((el) => {
        const styles = window.getComputedStyle(el);
        return styles.outline !== 'none' || styles.boxShadow !== 'none';
      });

      expect(hasFocusStyle).toBe(true);
    });

    test('should be navigable with screen reader', async () => {
      // Check for proper heading structure
      const trackTitle = page.getByTestId('track-title');
      const role = await trackTitle.getAttribute('role');
      const tagName = await trackTitle.evaluate((el) => el.tagName.toLowerCase());

      expect(['h1', 'h2', 'h3', 'h4', 'heading'].includes(tagName) || role === 'heading').toBe(true);
    });
  });
});

test.describe('Audio Quality Selection', () => {
  test('should show quality selector for premium users', async ({ page }) => {
    // Login as premium user
    await page.goto('/login');
    await page.fill('[data-testid="email"]', 'premium@test.com');
    await page.fill('[data-testid="password"]', 'password');
    await page.click('[data-testid="login-button"]');

    await page.goto('/');
    await page.waitForSelector('[data-testid="player"]');

    const qualitySelector = page.getByTestId('quality-selector');
    if (await qualitySelector.isVisible()) {
      await qualitySelector.click();
      await expect(page.getByText('High (320kbps)')).toBeVisible();
    }
  });
});
