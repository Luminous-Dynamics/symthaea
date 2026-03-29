// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Authentication E2E Tests
 */

import { test, expect } from '@playwright/test';

test.describe('Authentication', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should show login page for unauthenticated users', async ({ page }) => {
    await expect(page.getByRole('heading', { name: /sign in/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /sign in with/i })).toBeVisible();
  });

  test('should redirect to inbox after successful login', async ({ page }) => {
    // Mock OAuth flow
    await page.route('**/api/auth/callback**', async (route) => {
      await route.fulfill({
        status: 200,
        body: JSON.stringify({
          token: 'mock-jwt-token',
          user: { id: '1', email: 'test@example.com', name: 'Test User' },
        }),
      });
    });

    await page.getByRole('button', { name: /sign in/i }).click();

    // Should redirect to inbox
    await expect(page).toHaveURL(/\/inbox/);
    await expect(page.getByRole('heading', { name: /inbox/i })).toBeVisible();
  });

  test('should persist session across page reloads', async ({ page, context }) => {
    // Set auth cookie
    await context.addCookies([
      {
        name: 'auth_token',
        value: 'mock-jwt-token',
        domain: 'localhost',
        path: '/',
      },
    ]);

    await page.reload();
    await expect(page).toHaveURL(/\/inbox/);
  });

  test('should logout and clear session', async ({ page, context }) => {
    // Setup authenticated state
    await context.addCookies([
      { name: 'auth_token', value: 'mock-jwt-token', domain: 'localhost', path: '/' },
    ]);

    await page.goto('/inbox');
    await page.getByRole('button', { name: /account/i }).click();
    await page.getByRole('menuitem', { name: /sign out/i }).click();

    await expect(page).toHaveURL(/\//);
    await expect(page.getByRole('heading', { name: /sign in/i })).toBeVisible();
  });
});

test.describe('Authentication - Accessibility', () => {
  test('login form should be keyboard navigable', async ({ page }) => {
    await page.goto('/');

    // Tab through form elements
    await page.keyboard.press('Tab');
    await expect(page.getByRole('button', { name: /sign in/i })).toBeFocused();
  });

  test('should announce authentication errors to screen readers', async ({ page }) => {
    await page.goto('/');

    // Mock failed auth
    await page.route('**/api/auth/**', (route) =>
      route.fulfill({ status: 401, body: JSON.stringify({ error: 'Invalid credentials' }) })
    );

    await page.getByRole('button', { name: /sign in/i }).click();

    // Error should have role="alert"
    const alert = page.getByRole('alert');
    await expect(alert).toBeVisible();
    await expect(alert).toContainText(/invalid/i);
  });
});
