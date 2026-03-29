// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Artist Studio E2E Tests
 *
 * Tests for the artist dashboard and studio features including
 * track uploads, analytics, and earnings management.
 */

import { test, expect, type Page } from '@playwright/test';

// Test fixtures
const testArtist = {
  email: 'test-artist@example.com',
  password: 'TestPassword123!',
  name: 'Test Artist',
};

const testTrack = {
  title: 'E2E Test Track',
  genre: 'Electronic',
  description: 'Test track uploaded via E2E tests',
};

test.describe('Artist Studio', () => {
  test.describe('Dashboard', () => {
    test.beforeEach(async ({ page }) => {
      // Login as artist
      await page.goto('/login');
      await page.getByLabel('Email').fill(testArtist.email);
      await page.getByLabel('Password').fill(testArtist.password);
      await page.getByRole('button', { name: 'Sign In' }).click();
      await page.waitForURL('/dashboard');
    });

    test('displays artist statistics', async ({ page }) => {
      await page.goto('/studio');

      // Verify dashboard widgets are visible
      await expect(page.getByTestId('total-plays')).toBeVisible();
      await expect(page.getByTestId('total-earnings')).toBeVisible();
      await expect(page.getByTestId('total-followers')).toBeVisible();
      await expect(page.getByTestId('recent-activity')).toBeVisible();
    });

    test('shows earnings breakdown', async ({ page }) => {
      await page.goto('/studio/earnings');

      await expect(page.getByRole('heading', { name: 'Earnings' })).toBeVisible();

      // Check for chart
      await expect(page.locator('[data-testid="earnings-chart"]')).toBeVisible();

      // Check for period selector
      await expect(page.getByRole('combobox', { name: /period/i })).toBeVisible();
    });

    test('displays analytics charts', async ({ page }) => {
      await page.goto('/studio/analytics');

      // Wait for charts to load
      await expect(page.getByTestId('plays-chart')).toBeVisible();
      await expect(page.getByTestId('listeners-chart')).toBeVisible();

      // Check date range selector
      const dateRange = page.getByRole('button', { name: /Last 30 days/i });
      await expect(dateRange).toBeVisible();

      // Change date range
      await dateRange.click();
      await page.getByRole('option', { name: /Last 7 days/i }).click();

      // Verify chart updates
      await expect(page.getByTestId('plays-chart')).toBeVisible();
    });
  });

  test.describe('Track Management', () => {
    test.beforeEach(async ({ page }) => {
      await page.goto('/login');
      await page.getByLabel('Email').fill(testArtist.email);
      await page.getByLabel('Password').fill(testArtist.password);
      await page.getByRole('button', { name: 'Sign In' }).click();
      await page.waitForURL('/dashboard');
    });

    test('lists all artist tracks', async ({ page }) => {
      await page.goto('/studio/tracks');

      // Should show tracks table
      await expect(page.getByRole('table')).toBeVisible();

      // Check for column headers
      await expect(page.getByRole('columnheader', { name: /title/i })).toBeVisible();
      await expect(page.getByRole('columnheader', { name: /plays/i })).toBeVisible();
      await expect(page.getByRole('columnheader', { name: /earnings/i })).toBeVisible();
    });

    test('uploads a new track', async ({ page }) => {
      await page.goto('/studio/upload');

      // Fill track details
      await page.getByLabel('Title').fill(testTrack.title);
      await page.getByLabel('Genre').selectOption(testTrack.genre);
      await page.getByLabel('Description').fill(testTrack.description);

      // Upload audio file (using mock file for E2E)
      const fileInput = page.locator('input[type="file"][accept*="audio"]');
      await fileInput.setInputFiles({
        name: 'test-track.mp3',
        mimeType: 'audio/mpeg',
        buffer: Buffer.from('mock-audio-data'),
      });

      // Wait for upload to complete
      await expect(page.getByText(/uploading/i)).toBeVisible();
      await expect(page.getByText(/upload complete/i)).toBeVisible({ timeout: 30000 });

      // Submit
      await page.getByRole('button', { name: /publish/i }).click();

      // Verify redirect to track page
      await expect(page).toHaveURL(/\/studio\/tracks\/.+/);
    });

    test('edits track metadata', async ({ page }) => {
      await page.goto('/studio/tracks');

      // Click on first track
      await page.getByRole('row').nth(1).click();

      // Click edit button
      await page.getByRole('button', { name: /edit/i }).click();

      // Update title
      const titleInput = page.getByLabel('Title');
      await titleInput.clear();
      await titleInput.fill('Updated Track Title');

      // Save changes
      await page.getByRole('button', { name: /save/i }).click();

      // Verify success message
      await expect(page.getByText(/track updated/i)).toBeVisible();
    });

    test('deletes a track with confirmation', async ({ page }) => {
      await page.goto('/studio/tracks');

      // Click delete on first track
      await page.getByRole('row').nth(1).getByRole('button', { name: /delete/i }).click();

      // Confirm deletion dialog
      await expect(page.getByRole('dialog')).toBeVisible();
      await expect(page.getByText(/are you sure/i)).toBeVisible();

      // Confirm
      await page.getByRole('button', { name: /confirm/i }).click();

      // Verify success
      await expect(page.getByText(/track deleted/i)).toBeVisible();
    });
  });

  test.describe('Collaboration', () => {
    test('invites collaborator to track', async ({ page }) => {
      await page.goto('/login');
      await page.getByLabel('Email').fill(testArtist.email);
      await page.getByLabel('Password').fill(testArtist.password);
      await page.getByRole('button', { name: 'Sign In' }).click();

      await page.goto('/studio/tracks');
      await page.getByRole('row').nth(1).click();

      // Open collaborators panel
      await page.getByRole('button', { name: /collaborators/i }).click();

      // Add collaborator
      await page.getByLabel('Email').fill('collaborator@example.com');
      await page.getByRole('combobox', { name: /role/i }).selectOption('Producer');
      await page.getByLabel('Split %').fill('25');

      await page.getByRole('button', { name: /invite/i }).click();

      // Verify collaborator added
      await expect(page.getByText('collaborator@example.com')).toBeVisible();
    });
  });
});

test.describe('DAW Integration', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
    await page.getByLabel('Email').fill(testArtist.email);
    await page.getByLabel('Password').fill(testArtist.password);
    await page.getByRole('button', { name: 'Sign In' }).click();
    await page.waitForURL('/dashboard');
  });

  test('opens DAW editor', async ({ page }) => {
    await page.goto('/studio/daw');

    // Wait for DAW to load
    await expect(page.getByTestId('daw-workspace')).toBeVisible({ timeout: 15000 });

    // Verify main components
    await expect(page.getByTestId('timeline')).toBeVisible();
    await expect(page.getByTestId('track-list')).toBeVisible();
    await expect(page.getByTestId('mixer')).toBeVisible();
  });

  test('creates new project', async ({ page }) => {
    await page.goto('/studio/daw');

    await page.getByRole('button', { name: /new project/i }).click();

    // Fill project details
    await page.getByLabel('Project Name').fill('E2E Test Project');
    await page.getByLabel('BPM').fill('128');
    await page.getByRole('button', { name: /create/i }).click();

    // Verify project created
    await expect(page.getByText('E2E Test Project')).toBeVisible();
  });

  test('adds audio track', async ({ page }) => {
    await page.goto('/studio/daw');

    // Click add track
    await page.getByRole('button', { name: /add track/i }).click();
    await page.getByRole('menuitem', { name: /audio track/i }).click();

    // Verify track added
    const trackCount = await page.getByTestId('audio-track').count();
    expect(trackCount).toBeGreaterThan(0);
  });

  test('uses keyboard shortcuts', async ({ page }) => {
    await page.goto('/studio/daw');

    // Space to play/pause
    await page.keyboard.press('Space');
    await expect(page.getByTestId('transport-play')).toHaveAttribute('aria-pressed', 'true');

    await page.keyboard.press('Space');
    await expect(page.getByTestId('transport-play')).toHaveAttribute('aria-pressed', 'false');

    // Ctrl+Z for undo
    await page.keyboard.press('Control+z');
    // Verify undo action (depends on state)
  });

  test('exports project', async ({ page }) => {
    await page.goto('/studio/daw');

    // Open export dialog
    await page.getByRole('button', { name: /export/i }).click();

    // Select format
    await expect(page.getByRole('dialog')).toBeVisible();
    await page.getByRole('radio', { name: /wav/i }).check();

    // Start export
    await page.getByRole('button', { name: /export/i }).click();

    // Wait for export to complete
    await expect(page.getByText(/export complete/i)).toBeVisible({ timeout: 30000 });
  });
});

test.describe('Royalty Management', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
    await page.getByLabel('Email').fill(testArtist.email);
    await page.getByLabel('Password').fill(testArtist.password);
    await page.getByRole('button', { name: 'Sign In' }).click();
  });

  test('displays royalty splits', async ({ page }) => {
    await page.goto('/studio/royalties');

    // Verify royalty table
    await expect(page.getByRole('table')).toBeVisible();
    await expect(page.getByRole('columnheader', { name: /track/i })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: /collaborators/i })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: /split/i })).toBeVisible();
  });

  test('initiates withdrawal', async ({ page }) => {
    await page.goto('/studio/earnings');

    // Check balance
    const balance = page.getByTestId('available-balance');
    await expect(balance).toBeVisible();

    // Click withdraw
    await page.getByRole('button', { name: /withdraw/i }).click();

    // Fill withdrawal details
    await expect(page.getByRole('dialog')).toBeVisible();
    await page.getByLabel('Amount').fill('10');

    // Confirm
    await page.getByRole('button', { name: /confirm withdrawal/i }).click();

    // Should show wallet connect or transaction
    await expect(page.getByText(/processing|confirm in wallet/i)).toBeVisible();
  });
});

test.describe('Studio Accessibility', () => {
  test('supports keyboard navigation', async ({ page }) => {
    await page.goto('/login');
    await page.getByLabel('Email').fill(testArtist.email);
    await page.getByLabel('Password').fill(testArtist.password);
    await page.getByRole('button', { name: 'Sign In' }).click();

    await page.goto('/studio');

    // Tab through navigation
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');

    // Should be able to navigate with keyboard
    const focusedElement = await page.evaluate(() => document.activeElement?.tagName);
    expect(focusedElement).toBeTruthy();
  });

  test('has proper ARIA labels', async ({ page }) => {
    await page.goto('/login');
    await page.getByLabel('Email').fill(testArtist.email);
    await page.getByLabel('Password').fill(testArtist.password);
    await page.getByRole('button', { name: 'Sign In' }).click();

    await page.goto('/studio');

    // Check for main navigation
    await expect(page.getByRole('navigation')).toBeVisible();

    // Check for main content area
    await expect(page.getByRole('main')).toBeVisible();

    // Check headings structure
    const h1Count = await page.locator('h1').count();
    expect(h1Count).toBe(1);
  });
});
