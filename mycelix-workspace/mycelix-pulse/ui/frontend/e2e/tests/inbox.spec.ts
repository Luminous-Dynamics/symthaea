// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Inbox E2E Tests
 */

import { test, expect } from '@playwright/test';
import { mockEmails, setupAuthenticatedUser } from '../helpers';

test.describe('Inbox', () => {
  test.beforeEach(async ({ page, context }) => {
    await setupAuthenticatedUser(context);

    // Mock emails API
    await page.route('**/api/graphql', async (route) => {
      const body = JSON.parse(route.request().postData() || '{}');

      if (body.query?.includes('GetEmails')) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            data: {
              emails: {
                emails: mockEmails,
                nextCursor: null,
                hasMore: false,
                total: mockEmails.length,
              },
            },
          }),
        });
      } else {
        await route.continue();
      }
    });

    await page.goto('/inbox');
  });

  test('should display email list', async ({ page }) => {
    await expect(page.getByRole('listbox', { name: /email list/i })).toBeVisible();

    // Should show emails
    const emailItems = page.getByRole('option');
    await expect(emailItems).toHaveCount(mockEmails.length);
  });

  test('should show trust indicators on emails', async ({ page }) => {
    // Find email with high trust
    const trustedEmail = page.getByRole('option').filter({ hasText: 'Trusted Sender' });
    await expect(trustedEmail.getByText(/trusted/i)).toBeVisible();

    // Find email with low trust
    const unknownEmail = page.getByRole('option').filter({ hasText: 'Unknown Sender' });
    await expect(unknownEmail.getByText(/unknown|caution/i)).toBeVisible();
  });

  test('should mark email as read when opened', async ({ page }) => {
    const unreadEmail = page.getByRole('option').filter({ hasText: 'Unread Subject' });

    // Should have unread styling
    await expect(unreadEmail).toHaveCSS('font-weight', '700');

    // Click to open
    await unreadEmail.click();

    // Should navigate to email detail
    await expect(page).toHaveURL(/\/email\//);
  });

  test('should support keyboard navigation', async ({ page }) => {
    const list = page.getByRole('listbox');
    await list.focus();

    // Arrow down to select next email
    await page.keyboard.press('ArrowDown');
    await page.keyboard.press('ArrowDown');

    // Enter to open
    await page.keyboard.press('Enter');
    await expect(page).toHaveURL(/\/email\//);
  });

  test('should filter by folder', async ({ page }) => {
    await page.getByRole('link', { name: /sent/i }).click();
    await expect(page).toHaveURL(/\/sent/);

    await page.getByRole('link', { name: /archive/i }).click();
    await expect(page).toHaveURL(/\/archive/);
  });

  test('should star/unstar emails', async ({ page }) => {
    const firstEmail = page.getByRole('option').first();
    const starButton = firstEmail.getByRole('button', { name: /star/i });

    await starButton.click();
    await expect(starButton).toHaveAttribute('aria-pressed', 'true');

    await starButton.click();
    await expect(starButton).toHaveAttribute('aria-pressed', 'false');
  });

  test('should support bulk selection', async ({ page }) => {
    // Select all checkbox
    await page.getByRole('checkbox', { name: /select all/i }).check();

    const selectedCount = page.getByText(/\d+ selected/i);
    await expect(selectedCount).toBeVisible();

    // Bulk actions should appear
    await expect(page.getByRole('button', { name: /archive/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /delete/i })).toBeVisible();
  });
});

test.describe('Inbox - Search', () => {
  test.beforeEach(async ({ page, context }) => {
    await setupAuthenticatedUser(context);
    await page.goto('/inbox');
  });

  test('should search emails', async ({ page }) => {
    const searchInput = page.getByRole('searchbox', { name: /search/i });
    await searchInput.fill('important meeting');
    await page.keyboard.press('Enter');

    await expect(page).toHaveURL(/search.*important/i);
  });

  test('should show search suggestions', async ({ page }) => {
    const searchInput = page.getByRole('searchbox');
    await searchInput.fill('meet');

    // Autocomplete dropdown
    const suggestions = page.getByRole('listbox', { name: /suggestions/i });
    await expect(suggestions).toBeVisible();
  });

  test('should clear search', async ({ page }) => {
    const searchInput = page.getByRole('searchbox');
    await searchInput.fill('test query');

    await page.getByRole('button', { name: /clear/i }).click();
    await expect(searchInput).toHaveValue('');
  });
});

test.describe('Inbox - Accessibility', () => {
  test.beforeEach(async ({ page, context }) => {
    await setupAuthenticatedUser(context);
    await page.goto('/inbox');
  });

  test('should have proper ARIA labels', async ({ page }) => {
    await expect(page.getByRole('main')).toBeVisible();
    await expect(page.getByRole('navigation')).toBeVisible();
    await expect(page.getByRole('listbox')).toHaveAttribute('aria-label');
  });

  test('should announce loading states', async ({ page }) => {
    // Check for loading announcement
    const loadingStatus = page.getByRole('status');
    // Either shows loading or the content
    await expect(loadingStatus.or(page.getByRole('listbox'))).toBeVisible();
  });

  test('should support screen reader navigation', async ({ page }) => {
    // Skip to main content link
    await page.keyboard.press('Tab');
    const skipLink = page.getByRole('link', { name: /skip to/i });

    if (await skipLink.isVisible()) {
      await skipLink.click();
      await expect(page.getByRole('main')).toBeFocused();
    }
  });
});
