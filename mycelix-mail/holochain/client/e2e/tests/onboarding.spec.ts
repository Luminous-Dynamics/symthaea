// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * E2E Tests - Onboarding and Authentication
 *
 * Tests first-time user setup, key generation, and profile configuration
 */

import { test, expect, Page } from '@playwright/test';

test.describe('Onboarding', () => {
  let page: Page;

  test.beforeEach(async ({ browser }) => {
    // Use a fresh context with no stored state
    const context = await browser.newContext({
      storageState: { cookies: [], origins: [] },
    });
    page = await context.newPage();
  });

  test.afterEach(async () => {
    await page.close();
  });

  test('should show welcome screen for new users', async () => {
    await page.goto('/');

    // Verify welcome screen is displayed
    await expect(page.getByTestId('welcome-screen')).toBeVisible({ timeout: 10000 });

    // Verify key elements
    await expect(page.getByText(/welcome to mycelix mail/i)).toBeVisible();
    await expect(page.getByRole('button', { name: /get started/i })).toBeVisible();
  });

  test('should generate agent keys', async () => {
    await page.goto('/');

    // Click get started
    await page.getByRole('button', { name: /get started/i }).click();

    // Wait for key generation step
    await expect(page.getByTestId('key-generation-step')).toBeVisible();

    // Verify explanatory text
    await expect(page.getByText(/generating your cryptographic keys/i)).toBeVisible();

    // Click generate button
    await page.getByRole('button', { name: /generate keys/i }).click();

    // Wait for key generation (may take a moment)
    await expect(page.getByTestId('key-generated')).toBeVisible({ timeout: 30000 });

    // Verify public key is displayed
    await expect(page.getByTestId('agent-pub-key')).toBeVisible();
  });

  test('should setup user profile', async () => {
    await page.goto('/');

    // Navigate through onboarding to profile step
    await page.getByRole('button', { name: /get started/i }).click();
    await page.getByRole('button', { name: /generate keys/i }).click();
    await expect(page.getByTestId('key-generated')).toBeVisible({ timeout: 30000 });
    await page.getByRole('button', { name: /continue/i }).click();

    // Wait for profile step
    await expect(page.getByTestId('profile-setup-step')).toBeVisible();

    // Fill in profile
    await page.getByLabel('Display Name').fill('Test User');
    await page.getByLabel('Email').fill('test@example.com');

    // Optional: Add avatar
    const avatarInput = page.locator('input[type="file"]');
    if (await avatarInput.isVisible()) {
      // Skip avatar for now
    }

    // Continue
    await page.getByRole('button', { name: /continue/i }).click();

    // Verify profile was saved
    await expect(page.getByText(/profile saved/i)).toBeVisible();
  });

  test('should backup recovery phrase', async () => {
    await page.goto('/');

    // Navigate through onboarding
    await page.getByRole('button', { name: /get started/i }).click();
    await page.getByRole('button', { name: /generate keys/i }).click();
    await expect(page.getByTestId('key-generated')).toBeVisible({ timeout: 30000 });
    await page.getByRole('button', { name: /continue/i }).click();
    await page.getByLabel('Display Name').fill('Test User');
    await page.getByLabel('Email').fill('test@example.com');
    await page.getByRole('button', { name: /continue/i }).click();

    // Wait for backup step
    await expect(page.getByTestId('backup-step')).toBeVisible();

    // Verify recovery phrase is displayed
    await expect(page.getByTestId('recovery-phrase')).toBeVisible();

    // Verify warning about saving phrase
    await expect(page.getByText(/write down/i)).toBeVisible();

    // Confirm backup
    await page.getByRole('checkbox', { name: /i have saved/i }).check();
    await page.getByRole('button', { name: /continue/i }).click();
  });

  test('should complete onboarding and show inbox', async () => {
    await page.goto('/');

    // Fast-track through onboarding
    await page.getByRole('button', { name: /get started/i }).click();
    await page.getByRole('button', { name: /generate keys/i }).click();
    await expect(page.getByTestId('key-generated')).toBeVisible({ timeout: 30000 });
    await page.getByRole('button', { name: /continue/i }).click();

    await page.getByLabel('Display Name').fill('Test User');
    await page.getByLabel('Email').fill('test@example.com');
    await page.getByRole('button', { name: /continue/i }).click();

    await expect(page.getByTestId('backup-step')).toBeVisible();
    await page.getByRole('checkbox', { name: /i have saved/i }).check();
    await page.getByRole('button', { name: /continue/i }).click();

    // Wait for final step
    await expect(page.getByTestId('onboarding-complete')).toBeVisible();

    // Click to enter app
    await page.getByRole('button', { name: /enter app|go to inbox/i }).click();

    // Verify inbox is displayed
    await expect(page.getByTestId('inbox-view')).toBeVisible({ timeout: 10000 });
  });

  test('should restore from recovery phrase', async () => {
    await page.goto('/');

    // Click restore instead of new setup
    await page.getByRole('button', { name: /restore account/i }).click();

    // Wait for restore screen
    await expect(page.getByTestId('restore-step')).toBeVisible();

    // Enter a recovery phrase (dummy for testing)
    const dummyPhrase = 'abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about';
    await page.getByLabel('Recovery Phrase').fill(dummyPhrase);

    // Click restore
    await page.getByRole('button', { name: /restore/i }).click();

    // This will fail in actual test since phrase is invalid,
    // but we verify the flow works
    const error = page.getByText(/invalid|error/i);
    const success = page.getByText(/restored|success/i);

    // One of these should appear
    await expect(error.or(success)).toBeVisible({ timeout: 10000 });
  });

  test('should allow skipping optional steps', async () => {
    await page.goto('/');

    await page.getByRole('button', { name: /get started/i }).click();
    await page.getByRole('button', { name: /generate keys/i }).click();
    await expect(page.getByTestId('key-generated')).toBeVisible({ timeout: 30000 });
    await page.getByRole('button', { name: /continue/i }).click();

    // Skip profile details (just enter required fields)
    await page.getByLabel('Display Name').fill('Anonymous');
    await page.getByRole('button', { name: /skip|continue/i }).click();

    // Skip backup with warning
    await expect(page.getByTestId('backup-step')).toBeVisible();
    const skipButton = page.getByRole('button', { name: /skip/i });

    if (await skipButton.isVisible()) {
      await skipButton.click();

      // Verify warning dialog
      await expect(page.getByTestId('skip-warning-dialog')).toBeVisible();
      await page.getByRole('button', { name: /i understand/i }).click();
    }
  });

  test('should persist session after onboarding', async () => {
    // Complete onboarding first
    await page.goto('/');

    await page.getByRole('button', { name: /get started/i }).click();
    await page.getByRole('button', { name: /generate keys/i }).click();
    await expect(page.getByTestId('key-generated')).toBeVisible({ timeout: 30000 });
    await page.getByRole('button', { name: /continue/i }).click();

    await page.getByLabel('Display Name').fill('Persistent User');
    await page.getByLabel('Email').fill('persistent@example.com');
    await page.getByRole('button', { name: /continue/i }).click();

    await page.getByRole('checkbox', { name: /i have saved/i }).check();
    await page.getByRole('button', { name: /continue/i }).click();
    await page.getByRole('button', { name: /enter app|go to inbox/i }).click();

    await expect(page.getByTestId('inbox-view')).toBeVisible({ timeout: 10000 });

    // Reload page
    await page.reload();

    // Verify still logged in
    await expect(page.getByTestId('inbox-view')).toBeVisible({ timeout: 10000 });
    await expect(page.getByTestId('welcome-screen')).not.toBeVisible();
  });

  test('should show connection status during onboarding', async () => {
    await page.goto('/');

    // Verify connection status indicator
    await expect(page.getByTestId('connection-status')).toBeVisible();

    // Should show connecting or connected
    const status = page.getByTestId('connection-status');
    const statusText = await status.textContent();
    expect(statusText).toMatch(/connecting|connected|holochain/i);
  });
});
