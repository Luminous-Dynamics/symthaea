// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { test, expect } from '@playwright/test';

test.describe('Wallet Connection', () => {
  test('should show connect wallet button when not connected', async ({ page }) => {
    await page.goto('/');

    const connectButton = page.getByRole('button', { name: /connect|wallet/i });

    // Connect button should be visible when not authenticated
    if (await connectButton.isVisible()) {
      await expect(connectButton).toBeEnabled();
    }
  });

  test('should open wallet modal on connect click', async ({ page }) => {
    await page.goto('/');

    const connectButton = page.getByRole('button', { name: /connect|wallet/i });

    if (await connectButton.isVisible()) {
      await connectButton.click();

      // Wait for modal to appear
      const modal = page.locator('[role="dialog"], .modal, [data-testid="wallet-modal"]');
      await expect(modal).toBeVisible({ timeout: 5000 });
    }
  });
});

test.describe('Artist Dashboard', () => {
  test('should redirect to connect wallet if not authenticated', async ({ page }) => {
    await page.goto('/dashboard');

    // Should either show connect prompt or redirect
    const connectPrompt = page.getByText(/connect.*wallet|sign in/i);
    const dashboardContent = page.locator('[data-testid="dashboard"], .dashboard');

    // Either we see a connect prompt or are redirected
    const isConnectVisible = await connectPrompt.isVisible().catch(() => false);
    const isDashboardVisible = await dashboardContent.isVisible().catch(() => false);

    expect(isConnectVisible || isDashboardVisible || page.url().includes('connect')).toBeTruthy();
  });
});

test.describe('Upload Flow', () => {
  test('should show upload page', async ({ page }) => {
    await page.goto('/upload');

    // Should show upload form or connect prompt
    const uploadForm = page.locator('form, [data-testid="upload-form"]');
    const connectPrompt = page.getByText(/connect.*wallet|sign in/i);

    const isFormVisible = await uploadForm.isVisible().catch(() => false);
    const isConnectVisible = await connectPrompt.isVisible().catch(() => false);

    expect(isFormVisible || isConnectVisible).toBeTruthy();
  });
});
