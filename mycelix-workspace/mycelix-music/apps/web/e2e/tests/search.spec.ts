// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { test, expect } from '@playwright/test';

test.describe('Search Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/search');
  });

  test('should display search input', async ({ page }) => {
    const searchInput = page.getByPlaceholder('What do you want to listen to?');
    await expect(searchInput).toBeVisible();
    await expect(searchInput).toBeFocused();
  });

  test('should display browse categories when no search query', async ({ page }) => {
    await expect(page.getByText('Browse All')).toBeVisible();
    await expect(page.getByRole('link', { name: 'Electronic' })).toBeVisible();
  });

  test('should show filter tabs when searching', async ({ page }) => {
    const searchInput = page.getByPlaceholder('What do you want to listen to?');
    await searchInput.fill('test query');

    // Wait for debounce
    await page.waitForTimeout(400);

    // Check filter tabs appear
    await expect(page.getByRole('button', { name: 'All' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Songs' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Artists' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Playlists' })).toBeVisible();
  });

  test('should clear search input with X button', async ({ page }) => {
    const searchInput = page.getByPlaceholder('What do you want to listen to?');
    await searchInput.fill('test query');

    const clearButton = page.locator('button').filter({ has: page.locator('svg') }).first();
    await clearButton.click();

    await expect(searchInput).toHaveValue('');
  });

  test('should update URL with search query', async ({ page }) => {
    const searchInput = page.getByPlaceholder('What do you want to listen to?');
    await searchInput.fill('electronic beats');

    // Wait for debounce and URL update
    await page.waitForTimeout(400);

    await expect(page).toHaveURL(/search\?q=electronic%20beats/);
  });

  test('should load search query from URL', async ({ page }) => {
    await page.goto('/search?q=lofi%20vibes');

    const searchInput = page.getByPlaceholder('What do you want to listen to?');
    await expect(searchInput).toHaveValue('lofi vibes');
  });

  test('should navigate to genre page when clicking genre card', async ({ page }) => {
    await page.getByRole('link', { name: 'Electronic' }).click();
    await expect(page).toHaveURL('/genre/electronic');
  });
});
