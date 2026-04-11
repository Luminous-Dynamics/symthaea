// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Authentication Setup for E2E Tests
 *
 * This file handles user authentication before tests run.
 * The authenticated state is saved and reused by other test projects.
 */

import { test as setup, expect } from '@playwright/test';
import path from 'path';

const authFile = path.join(__dirname, '.auth/user.json');

// Test user credentials
const testUser = {
  email: 'test@mycelix.mail',
  password: 'TestPassword123!',
};

setup('authenticate', async ({ page }) => {
  // Navigate to login page
  await page.goto('/login');

  // Wait for the login form to be visible
  await expect(page.locator('form')).toBeVisible();

  // Fill in credentials
  await page.fill('input[name="email"], input[type="email"]', testUser.email);
  await page.fill('input[name="password"], input[type="password"]', testUser.password);

  // Click login button
  await page.click('button[type="submit"]');

  // Wait for successful login - should redirect to inbox or dashboard
  await page.waitForURL(/\/(inbox|dashboard|home)/, { timeout: 15000 });

  // Verify we're logged in by checking for user menu or similar element
  await expect(
    page.locator('[data-testid="user-menu"], [data-testid="user-avatar"], .user-menu')
  ).toBeVisible({ timeout: 10000 });

  // Save the authenticated state
  await page.context().storageState({ path: authFile });
});

setup('setup mock holochain connection', async ({ page }) => {
  // This setup ensures the mock Holochain connection is ready
  await page.goto('/');

  // Add initialization script for Holochain mock
  await page.addInitScript(() => {
    // Mark that we're in test mode
    (window as any).__PLAYWRIGHT_TEST__ = true;

    // Mock Holochain connection state
    (window as any).__HOLOCHAIN_MOCK_STATE__ = {
      connected: true,
      agentPubKey: 'uhCAktestuser123',
    };
  });
});
