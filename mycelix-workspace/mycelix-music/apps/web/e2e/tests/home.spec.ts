// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { test, expect } from '@playwright/test';

test.describe('Home Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should display greeting based on time of day', async ({ page }) => {
    const greeting = page.locator('h1');
    await expect(greeting).toBeVisible();

    const text = await greeting.textContent();
    expect(
      text?.includes('Good morning') ||
      text?.includes('Good afternoon') ||
      text?.includes('Good evening')
    ).toBeTruthy();
  });

  test('should display sidebar navigation', async ({ page }) => {
    await expect(page.locator('nav')).toBeVisible();
    await expect(page.getByRole('link', { name: 'Home' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Search' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Library' })).toBeVisible();
  });

  test('should display trending section', async ({ page }) => {
    await expect(page.getByText('Trending Now')).toBeVisible();
  });

  test('should display new releases section', async ({ page }) => {
    await expect(page.getByText('New Releases')).toBeVisible();
  });

  test('should display genre browse section', async ({ page }) => {
    await expect(page.getByText('Browse by Genre')).toBeVisible();

    // Check for genre cards
    await expect(page.getByRole('link', { name: 'Electronic' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Hip Hop' })).toBeVisible();
  });

  test('should navigate to search page', async ({ page }) => {
    await page.getByRole('link', { name: 'Search' }).click();
    await expect(page).toHaveURL('/search');
  });

  test('should display login button when not authenticated', async ({ page }) => {
    await expect(page.getByRole('button', { name: 'Log in' })).toBeVisible();
  });
});
