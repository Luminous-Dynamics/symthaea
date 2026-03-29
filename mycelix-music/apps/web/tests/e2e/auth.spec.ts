// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * E2E Tests: Authentication Flow
 *
 * Tests wallet connection, soul creation, and identity management.
 */

import { test, expect, Page } from '@playwright/test';

test.describe('Authentication', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should show connect button when not authenticated', async ({ page }) => {
    const connectButton = page.getByRole('button', { name: /connect/i });
    await expect(connectButton).toBeVisible();
  });

  test('should open wallet modal on connect click', async ({ page }) => {
    await page.getByRole('button', { name: /connect/i }).click();

    const modal = page.getByRole('dialog');
    await expect(modal).toBeVisible();

    // Should show wallet options
    await expect(page.getByText(/MetaMask/i)).toBeVisible();
    await expect(page.getByText(/WalletConnect/i)).toBeVisible();
    await expect(page.getByText(/Coinbase/i)).toBeVisible();
  });

  test('should show email option for account abstraction', async ({ page }) => {
    await page.getByRole('button', { name: /connect/i }).click();

    // Should have email/social login option
    await expect(page.getByText(/Continue with Email/i)).toBeVisible();
  });

  test('should close modal on backdrop click', async ({ page }) => {
    await page.getByRole('button', { name: /connect/i }).click();

    const modal = page.getByRole('dialog');
    await expect(modal).toBeVisible();

    // Click outside modal
    await page.locator('.modal-backdrop').click({ position: { x: 10, y: 10 } });

    await expect(modal).not.toBeVisible();
  });
});

test.describe('Soul Creation', () => {
  test.beforeEach(async ({ page }) => {
    // Mock wallet connection
    await page.addInitScript(() => {
      window.localStorage.setItem('mycelix_wallet', JSON.stringify({
        provider: 'metamask',
        address: '0x1234567890123456789012345678901234567890',
        chainId: 84532,
        isConnected: true,
      }));
    });
    await page.goto('/');
  });

  test('should prompt soul creation for new users', async ({ page }) => {
    await expect(page.getByText(/Create Your Soul/i)).toBeVisible();
  });

  test('should validate display name', async ({ page }) => {
    const nameInput = page.getByLabel(/Display Name/i);
    await nameInput.fill('AB'); // Too short

    await page.getByRole('button', { name: /Create Soul/i }).click();

    await expect(page.getByText(/at least 3 characters/i)).toBeVisible();
  });

  test('should create soul with valid data', async ({ page }) => {
    const nameInput = page.getByLabel(/Display Name/i);
    await nameInput.fill('Test User');

    const bioInput = page.getByLabel(/Bio/i);
    await bioInput.fill('Music enthusiast exploring new sounds');

    await page.getByRole('button', { name: /Create Soul/i }).click();

    // Should show loading state
    await expect(page.getByText(/Creating/i)).toBeVisible();

    // Should redirect to home after creation
    await expect(page).toHaveURL('/');
  });
});

test.describe('Authenticated User', () => {
  test.beforeEach(async ({ page }) => {
    // Mock authenticated state
    await page.addInitScript(() => {
      window.localStorage.setItem('mycelix_soul', JSON.stringify({
        id: 'soul_test123',
        address: '0x1234567890123456789012345678901234567890',
        profile: {
          displayName: 'Test User',
          bio: 'Test bio',
          genres: ['ambient', 'electronic'],
          createdAt: new Date().toISOString(),
        },
        credentials: [],
        connections: [],
        resonance: {
          total: 100,
          categories: {
            listening: 50,
            patronage: 20,
            collaboration: 10,
            community: 10,
            presence: 10,
          },
        },
      }));
      window.localStorage.setItem('mycelix_wallet', JSON.stringify({
        provider: 'metamask',
        address: '0x1234567890123456789012345678901234567890',
        chainId: 84532,
        isConnected: true,
      }));
    });
    await page.goto('/');
  });

  test('should show user avatar in header', async ({ page }) => {
    const avatar = page.getByTestId('user-avatar');
    await expect(avatar).toBeVisible();
  });

  test('should navigate to profile page', async ({ page }) => {
    await page.getByTestId('user-avatar').click();
    await page.getByRole('menuitem', { name: /Profile/i }).click();

    await expect(page).toHaveURL('/profile');
    await expect(page.getByText('Test User')).toBeVisible();
  });

  test('should show resonance breakdown on profile', async ({ page }) => {
    await page.goto('/profile');

    await expect(page.getByText(/Resonance/i)).toBeVisible();
    await expect(page.getByText('100')).toBeVisible(); // Total resonance
  });

  test('should allow profile editing', async ({ page }) => {
    await page.goto('/profile');

    await page.getByRole('button', { name: /Edit Profile/i }).click();

    const bioInput = page.getByLabel(/Bio/i);
    await bioInput.clear();
    await bioInput.fill('Updated bio text');

    await page.getByRole('button', { name: /Save/i }).click();

    await expect(page.getByText('Updated bio text')).toBeVisible();
  });

  test('should disconnect wallet', async ({ page }) => {
    await page.getByTestId('user-avatar').click();
    await page.getByRole('menuitem', { name: /Disconnect/i }).click();

    // Should show confirmation
    await page.getByRole('button', { name: /Confirm/i }).click();

    // Should redirect to home with connect button
    await expect(page.getByRole('button', { name: /connect/i })).toBeVisible();
  });
});

test.describe('Chain Switching', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem('mycelix_wallet', JSON.stringify({
        provider: 'metamask',
        address: '0x1234567890123456789012345678901234567890',
        chainId: 1, // Mainnet (unsupported for testing)
        isConnected: true,
      }));
    });
    await page.goto('/');
  });

  test('should show network warning for unsupported chain', async ({ page }) => {
    await expect(page.getByText(/Wrong Network/i)).toBeVisible();
  });

  test('should offer to switch network', async ({ page }) => {
    await page.getByRole('button', { name: /Switch Network/i }).click();

    // Should show network options
    await expect(page.getByText(/Base Sepolia/i)).toBeVisible();
  });
});
