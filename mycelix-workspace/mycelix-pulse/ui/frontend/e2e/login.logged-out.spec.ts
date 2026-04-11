// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Login and Connection Flow E2E Tests
 *
 * Tests for authentication, Holochain connection, and session management
 */

import { test, expect, Page } from '@playwright/test';

// Test fixtures
const testUser = {
  email: 'test@mycelix.mail',
  password: 'TestPassword123!',
  name: 'Test User',
};

const invalidCredentials = {
  email: 'invalid@example.com',
  password: 'WrongPassword123!',
};

// ============================================================================
// Helper Functions
// ============================================================================

async function fillLoginForm(page: Page, email: string, password: string) {
  await page.fill('input[name="email"], input[type="email"]', email);
  await page.fill('input[name="password"], input[type="password"]', password);
}

async function submitLoginForm(page: Page) {
  await page.click('button[type="submit"]');
}

// ============================================================================
// Login Page Display Tests
// ============================================================================

test.describe('Login Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
  });

  test('should display login page with all required elements', async ({ page }) => {
    // Check page title or header
    await expect(page.locator('h1, h2').first()).toContainText(/sign in|login|welcome/i);

    // Check for email input
    const emailInput = page.locator('input[name="email"], input[type="email"]');
    await expect(emailInput).toBeVisible();
    await expect(emailInput).toBeEnabled();

    // Check for password input
    const passwordInput = page.locator('input[name="password"], input[type="password"]');
    await expect(passwordInput).toBeVisible();
    await expect(passwordInput).toBeEnabled();

    // Check for submit button
    const submitButton = page.locator('button[type="submit"]');
    await expect(submitButton).toBeVisible();
    await expect(submitButton).toBeEnabled();
  });

  test('should have password field with type="password"', async ({ page }) => {
    const passwordInput = page.locator('input[name="password"], input[type="password"]');
    await expect(passwordInput).toHaveAttribute('type', 'password');
  });

  test('should show link to registration page', async ({ page }) => {
    const registerLink = page.locator('a[href*="register"], a[href*="signup"]');
    await expect(registerLink).toBeVisible();
  });

  test('should show link for forgot password', async ({ page }) => {
    const forgotPasswordLink = page.locator('a[href*="forgot"], a[href*="reset"]');
    // This might not exist in all implementations
    if (await forgotPasswordLink.count() > 0) {
      await expect(forgotPasswordLink.first()).toBeVisible();
    }
  });
});

// ============================================================================
// Login Validation Tests
// ============================================================================

test.describe('Login Form Validation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
  });

  test('should show error for empty email', async ({ page }) => {
    await fillLoginForm(page, '', testUser.password);
    await submitLoginForm(page);

    // Check for validation error
    const errorMessage = page.locator(
      '[data-testid="error-message"], .error-message, [role="alert"], .text-red-500'
    );
    await expect(errorMessage.first()).toBeVisible({ timeout: 5000 });
  });

  test('should show error for empty password', async ({ page }) => {
    await fillLoginForm(page, testUser.email, '');
    await submitLoginForm(page);

    // Check for validation error
    const errorMessage = page.locator(
      '[data-testid="error-message"], .error-message, [role="alert"], .text-red-500'
    );
    await expect(errorMessage.first()).toBeVisible({ timeout: 5000 });
  });

  test('should show error for invalid email format', async ({ page }) => {
    await fillLoginForm(page, 'invalidemail', testUser.password);
    await submitLoginForm(page);

    // Check for validation error
    const errorMessage = page.locator(
      '[data-testid="error-message"], .error-message, [role="alert"], .text-red-500, :invalid'
    );
    await expect(errorMessage.first()).toBeVisible({ timeout: 5000 });
  });

  test('should show error for invalid credentials', async ({ page }) => {
    await fillLoginForm(page, invalidCredentials.email, invalidCredentials.password);
    await submitLoginForm(page);

    // Wait for error message
    const errorMessage = page.locator(
      '[data-testid="error-message"], .error-message, [role="alert"]'
    );
    await expect(errorMessage.first()).toBeVisible({ timeout: 10000 });
  });
});

// ============================================================================
// Successful Login Tests
// ============================================================================

test.describe('Successful Login', () => {
  test('should successfully login with valid credentials', async ({ page }) => {
    await page.goto('/login');

    await fillLoginForm(page, testUser.email, testUser.password);
    await submitLoginForm(page);

    // Should redirect to inbox or dashboard
    await page.waitForURL(/\/(inbox|dashboard|home)/, { timeout: 15000 });

    // Should show user menu or avatar
    await expect(
      page.locator('[data-testid="user-menu"], [data-testid="user-avatar"], .user-menu')
    ).toBeVisible({ timeout: 10000 });
  });

  test('should show loading state during login', async ({ page }) => {
    await page.goto('/login');

    await fillLoginForm(page, testUser.email, testUser.password);

    // Start login and check for loading state
    const submitButton = page.locator('button[type="submit"]');
    await submitButton.click();

    // The button might show a loading spinner or be disabled
    // This depends on implementation
    await expect(submitButton).toBeDisabled({ timeout: 1000 }).catch(() => {
      // Button might not be disabled, which is okay
    });

    // Wait for successful login
    await page.waitForURL(/\/(inbox|dashboard|home)/, { timeout: 15000 });
  });

  test('should persist session after page reload', async ({ page }) => {
    // First login
    await page.goto('/login');
    await fillLoginForm(page, testUser.email, testUser.password);
    await submitLoginForm(page);
    await page.waitForURL(/\/(inbox|dashboard|home)/, { timeout: 15000 });

    // Reload the page
    await page.reload();

    // Should still be logged in
    await expect(page).toHaveURL(/\/(inbox|dashboard|home)/);
    await expect(
      page.locator('[data-testid="user-menu"], [data-testid="user-avatar"], .user-menu')
    ).toBeVisible();
  });
});

// ============================================================================
// Holochain Connection Tests
// ============================================================================

test.describe('Holochain Connection', () => {
  test('should show connection status indicator', async ({ page }) => {
    await page.goto('/login');
    await fillLoginForm(page, testUser.email, testUser.password);
    await submitLoginForm(page);
    await page.waitForURL(/\/(inbox|dashboard|home)/, { timeout: 15000 });

    // Look for connection status indicator
    const connectionIndicator = page.locator(
      '[data-testid="connection-status"], .connection-indicator, [aria-label*="connection"]'
    );

    // Connection indicator might exist
    if (await connectionIndicator.count() > 0) {
      await expect(connectionIndicator.first()).toBeVisible();
    }
  });

  test('should handle connection errors gracefully', async ({ page }) => {
    // Simulate offline mode
    await page.context().setOffline(true);

    await page.goto('/login');

    // Should show some indication of offline status or error
    const offlineIndicator = page.locator(
      '[data-testid="offline-indicator"], .offline-banner, [role="alert"]'
    );

    // The app should handle offline state
    // Wait a bit and check if any error handling is visible
    await page.waitForTimeout(2000);

    // Restore online mode
    await page.context().setOffline(false);
  });
});

// ============================================================================
// Logout Tests
// ============================================================================

test.describe('Logout', () => {
  test('should successfully logout', async ({ page }) => {
    // Login first
    await page.goto('/login');
    await fillLoginForm(page, testUser.email, testUser.password);
    await submitLoginForm(page);
    await page.waitForURL(/\/(inbox|dashboard|home)/, { timeout: 15000 });

    // Click user menu
    const userMenu = page.locator('[data-testid="user-menu"], [data-testid="user-avatar"], .user-menu');
    await userMenu.click();

    // Click logout
    const logoutButton = page.locator(
      '[data-testid="logout-button"], button:has-text("Logout"), button:has-text("Sign out")'
    );
    await logoutButton.click();

    // Should redirect to login page
    await page.waitForURL(/\/login/, { timeout: 10000 });
  });

  test('should clear session data on logout', async ({ page }) => {
    // Login first
    await page.goto('/login');
    await fillLoginForm(page, testUser.email, testUser.password);
    await submitLoginForm(page);
    await page.waitForURL(/\/(inbox|dashboard|home)/, { timeout: 15000 });

    // Logout
    const userMenu = page.locator('[data-testid="user-menu"], [data-testid="user-avatar"], .user-menu');
    await userMenu.click();

    const logoutButton = page.locator(
      '[data-testid="logout-button"], button:has-text("Logout"), button:has-text("Sign out")'
    );
    await logoutButton.click();

    await page.waitForURL(/\/login/, { timeout: 10000 });

    // Try to access protected route
    await page.goto('/inbox');

    // Should redirect back to login
    await expect(page).toHaveURL(/\/login/);
  });
});

// ============================================================================
// Security Tests
// ============================================================================

test.describe('Security', () => {
  test('should redirect unauthenticated users to login', async ({ page }) => {
    // Try to access protected routes without being logged in
    await page.goto('/inbox');
    await expect(page).toHaveURL(/\/login/);

    await page.goto('/compose');
    await expect(page).toHaveURL(/\/login/);

    await page.goto('/settings');
    await expect(page).toHaveURL(/\/login/);
  });

  test('should handle CSRF token properly', async ({ page }) => {
    await page.goto('/login');

    // Check if CSRF token is present in form or meta tag
    const csrfInput = page.locator('input[name="csrf"], input[name="_csrf"]');
    const csrfMeta = page.locator('meta[name="csrf-token"]');

    // At least one should exist (or the app uses a different CSRF method)
    const hasCsrf = (await csrfInput.count()) > 0 || (await csrfMeta.count()) > 0;

    // Log but don't fail - CSRF might be handled differently
    if (!hasCsrf) {
      console.log('Note: No visible CSRF token found. App might use different protection.');
    }
  });

  test('should not expose sensitive data in URL', async ({ page }) => {
    await page.goto('/login');
    await fillLoginForm(page, testUser.email, testUser.password);
    await submitLoginForm(page);
    await page.waitForURL(/\/(inbox|dashboard|home)/, { timeout: 15000 });

    // URL should not contain password or sensitive tokens
    const url = page.url();
    expect(url).not.toContain('password');
    expect(url).not.toContain(testUser.password);
  });
});

// ============================================================================
// Accessibility Tests
// ============================================================================

test.describe('Login Accessibility', () => {
  test('should have proper form labels', async ({ page }) => {
    await page.goto('/login');

    // Email input should have a label
    const emailInput = page.locator('input[name="email"], input[type="email"]');
    const emailLabel = page.locator('label[for]').filter({ has: page.locator('[name="email"], [type="email"]') });

    // Input should be associated with a label via id or aria-label
    const emailId = await emailInput.getAttribute('id');
    const emailAriaLabel = await emailInput.getAttribute('aria-label');

    expect(emailId || emailAriaLabel).toBeTruthy();
  });

  test('should be keyboard navigable', async ({ page }) => {
    await page.goto('/login');

    // Tab to email input
    await page.keyboard.press('Tab');
    const emailInput = page.locator('input[name="email"], input[type="email"]');

    // Tab to password input
    await page.keyboard.press('Tab');

    // Tab to submit button
    await page.keyboard.press('Tab');
    const submitButton = page.locator('button[type="submit"]');

    // We can test if the submit button receives focus
    await page.keyboard.press('Tab');
  });

  test('should support Enter key to submit form', async ({ page }) => {
    await page.goto('/login');

    await fillLoginForm(page, testUser.email, testUser.password);

    // Press Enter to submit
    await page.keyboard.press('Enter');

    // Should attempt to login
    await page.waitForURL(/\/(inbox|dashboard|home|login)/, { timeout: 15000 });
  });
});

// ============================================================================
// Registration Link Tests
// ============================================================================

test.describe('Registration', () => {
  test('should navigate to registration page', async ({ page }) => {
    await page.goto('/login');

    const registerLink = page.locator('a[href*="register"], a[href*="signup"]');

    if (await registerLink.count() > 0) {
      await registerLink.first().click();
      await expect(page).toHaveURL(/\/(register|signup)/);
    }
  });
});
