// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail E2E Tests
 *
 * Run with: npx playwright test
 */

import { test, expect, Page } from '@playwright/test';

// ============================================================================
// Test Configuration
// ============================================================================

const BASE_URL = process.env.TEST_URL || 'http://localhost:3000';

const testUser = {
  email: 'test@example.com',
  password: 'test_password_123',
};

// ============================================================================
// Helper Functions
// ============================================================================

async function login(page: Page) {
  await page.goto(`${BASE_URL}/login`);
  await page.fill('[name="email"]', testUser.email);
  await page.fill('[name="password"]', testUser.password);
  await page.click('button[type="submit"]');
  await page.waitForURL(`${BASE_URL}/inbox`);
}

async function composeEmail(page: Page, to: string, subject: string, body: string) {
  await page.click('[data-testid="compose-button"]');
  await page.waitForSelector('[data-testid="compose-modal"]');
  await page.fill('[name="to"]', to);
  await page.fill('[name="subject"]', subject);
  await page.fill('[data-testid="email-body"]', body);
}

// ============================================================================
// Authentication Tests
// ============================================================================

test.describe('Authentication', () => {
  test('should show login page', async ({ page }) => {
    await page.goto(`${BASE_URL}/login`);
    await expect(page.locator('h1')).toContainText('Sign In');
    await expect(page.locator('[name="email"]')).toBeVisible();
    await expect(page.locator('[name="password"]')).toBeVisible();
  });

  test('should show error for invalid credentials', async ({ page }) => {
    await page.goto(`${BASE_URL}/login`);
    await page.fill('[name="email"]', 'invalid@example.com');
    await page.fill('[name="password"]', 'wrong_password');
    await page.click('button[type="submit"]');
    await expect(page.locator('.error-message')).toBeVisible();
  });

  test('should login with valid credentials', async ({ page }) => {
    await login(page);
    await expect(page).toHaveURL(`${BASE_URL}/inbox`);
    await expect(page.locator('[data-testid="inbox-header"]')).toBeVisible();
  });

  test('should logout successfully', async ({ page }) => {
    await login(page);
    await page.click('[data-testid="user-menu"]');
    await page.click('[data-testid="logout-button"]');
    await expect(page).toHaveURL(`${BASE_URL}/login`);
  });
});

// ============================================================================
// Inbox Tests
// ============================================================================

test.describe('Inbox', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('should display email list', async ({ page }) => {
    await expect(page.locator('[data-testid="email-list"]')).toBeVisible();
    const emails = page.locator('[data-testid="email-item"]');
    await expect(emails.first()).toBeVisible();
  });

  test('should show unread count', async ({ page }) => {
    const unreadBadge = page.locator('[data-testid="unread-count"]');
    await expect(unreadBadge).toBeVisible();
  });

  test('should open email on click', async ({ page }) => {
    await page.click('[data-testid="email-item"]:first-child');
    await expect(page.locator('[data-testid="email-viewer"]')).toBeVisible();
    await expect(page.locator('[data-testid="email-subject"]')).toBeVisible();
    await expect(page.locator('[data-testid="email-body"]')).toBeVisible();
  });

  test('should mark email as read', async ({ page }) => {
    const firstEmail = page.locator('[data-testid="email-item"]:first-child');
    await expect(firstEmail).toHaveClass(/unread/);
    await firstEmail.click();
    await page.waitForTimeout(1000); // Wait for read status update
    await page.click('[data-testid="back-button"]');
    await expect(firstEmail).not.toHaveClass(/unread/);
  });

  test('should star email', async ({ page }) => {
    const starButton = page.locator('[data-testid="email-item"]:first-child [data-testid="star-button"]');
    await starButton.click();
    await expect(starButton).toHaveClass(/starred/);
  });

  test('should archive email', async ({ page }) => {
    const firstEmail = page.locator('[data-testid="email-item"]:first-child');
    const emailId = await firstEmail.getAttribute('data-email-id');

    await firstEmail.hover();
    await page.click('[data-testid="archive-button"]');

    await expect(page.locator(`[data-email-id="${emailId}"]`)).not.toBeVisible();
  });

  test('should delete email', async ({ page }) => {
    const firstEmail = page.locator('[data-testid="email-item"]:first-child');
    const emailId = await firstEmail.getAttribute('data-email-id');

    await firstEmail.hover();
    await page.click('[data-testid="delete-button"]');

    await expect(page.locator(`[data-email-id="${emailId}"]`)).not.toBeVisible();
  });
});

// ============================================================================
// Compose Tests
// ============================================================================

test.describe('Compose', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('should open compose modal', async ({ page }) => {
    await page.click('[data-testid="compose-button"]');
    await expect(page.locator('[data-testid="compose-modal"]')).toBeVisible();
  });

  test('should validate recipient email', async ({ page }) => {
    await composeEmail(page, 'invalid-email', 'Test', 'Body');
    await page.click('[data-testid="send-button"]');
    await expect(page.locator('.field-error')).toContainText('Invalid email');
  });

  test('should require subject', async ({ page }) => {
    await composeEmail(page, 'test@example.com', '', 'Body');
    await page.click('[data-testid="send-button"]');
    await expect(page.locator('.field-error')).toContainText('Subject required');
  });

  test('should send email', async ({ page }) => {
    await composeEmail(page, 'recipient@example.com', 'Test Subject', 'Test body content');
    await page.click('[data-testid="send-button"]');

    await expect(page.locator('[data-testid="compose-modal"]')).not.toBeVisible();
    await expect(page.locator('.toast-success')).toContainText('Email sent');
  });

  test('should save draft', async ({ page }) => {
    await composeEmail(page, 'recipient@example.com', 'Draft Subject', 'Draft content');
    await page.click('[data-testid="save-draft-button"]');

    await expect(page.locator('.toast-success')).toContainText('Draft saved');
  });

  test('should attach file', async ({ page }) => {
    await page.click('[data-testid="compose-button"]');

    const fileInput = page.locator('[data-testid="attachment-input"]');
    await fileInput.setInputFiles({
      name: 'test.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('Test file content'),
    });

    await expect(page.locator('[data-testid="attachment-list"]')).toContainText('test.txt');
  });
});

// ============================================================================
// Search Tests
// ============================================================================

test.describe('Search', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('should open search on click', async ({ page }) => {
    await page.click('[data-testid="search-input"]');
    await expect(page.locator('[data-testid="search-panel"]')).toBeVisible();
  });

  test('should search emails', async ({ page }) => {
    await page.fill('[data-testid="search-input"]', 'meeting');
    await page.press('[data-testid="search-input"]', 'Enter');

    await expect(page.locator('[data-testid="search-results"]')).toBeVisible();
  });

  test('should use search operators', async ({ page }) => {
    await page.fill('[data-testid="search-input"]', 'from:boss@company.com is:unread');
    await page.press('[data-testid="search-input"]', 'Enter');

    await expect(page.locator('[data-testid="search-results"]')).toBeVisible();
  });

  test('should show search suggestions', async ({ page }) => {
    await page.click('[data-testid="search-input"]');
    await page.fill('[data-testid="search-input"]', 'from:');

    await expect(page.locator('[data-testid="search-suggestions"]')).toBeVisible();
  });

  test('should clear search', async ({ page }) => {
    await page.fill('[data-testid="search-input"]', 'test query');
    await page.click('[data-testid="clear-search"]');

    await expect(page.locator('[data-testid="search-input"]')).toHaveValue('');
  });
});

// ============================================================================
// Keyboard Shortcuts Tests
// ============================================================================

test.describe('Keyboard Shortcuts', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('should open compose with C', async ({ page }) => {
    await page.keyboard.press('c');
    await expect(page.locator('[data-testid="compose-modal"]')).toBeVisible();
  });

  test('should navigate emails with J/K', async ({ page }) => {
    await page.keyboard.press('j');
    await expect(page.locator('[data-testid="email-item"].selected')).toBeVisible();

    await page.keyboard.press('j');
    await page.keyboard.press('k');
  });

  test('should open email with Enter', async ({ page }) => {
    await page.keyboard.press('j');
    await page.keyboard.press('Enter');
    await expect(page.locator('[data-testid="email-viewer"]')).toBeVisible();
  });

  test('should archive with E', async ({ page }) => {
    await page.keyboard.press('j');
    await page.keyboard.press('e');
    await expect(page.locator('.toast-success')).toContainText('Archived');
  });

  test('should search with /', async ({ page }) => {
    await page.keyboard.press('/');
    await expect(page.locator('[data-testid="search-input"]')).toBeFocused();
  });

  test('should show shortcuts help with ?', async ({ page }) => {
    await page.keyboard.press('Shift+/');
    await expect(page.locator('[data-testid="shortcuts-modal"]')).toBeVisible();
  });
});

// ============================================================================
// Trust System Tests
// ============================================================================

test.describe('Trust System', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('should display trust badge on emails', async ({ page }) => {
    const trustBadge = page.locator('[data-testid="email-item"]:first-child [data-testid="trust-badge"]');
    await expect(trustBadge).toBeVisible();
  });

  test('should show trust details on click', async ({ page }) => {
    await page.click('[data-testid="email-item"]:first-child');
    await page.click('[data-testid="trust-badge"]');
    await expect(page.locator('[data-testid="trust-panel"]')).toBeVisible();
  });

  test('should create attestation', async ({ page }) => {
    await page.click('[data-testid="email-item"]:first-child');
    await page.click('[data-testid="trust-badge"]');
    await page.click('[data-testid="create-attestation"]');

    await page.fill('[data-testid="trust-level-input"]', '4');
    await page.fill('[data-testid="trust-context"]', 'colleague');
    await page.click('[data-testid="save-attestation"]');

    await expect(page.locator('.toast-success')).toContainText('Attestation created');
  });
});

// ============================================================================
// Settings Tests
// ============================================================================

test.describe('Settings', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
    await page.goto(`${BASE_URL}/settings`);
  });

  test('should display settings page', async ({ page }) => {
    await expect(page.locator('[data-testid="settings-page"]')).toBeVisible();
  });

  test('should update display name', async ({ page }) => {
    await page.click('[data-testid="profile-tab"]');
    await page.fill('[name="displayName"]', 'New Display Name');
    await page.click('[data-testid="save-profile"]');

    await expect(page.locator('.toast-success')).toContainText('Profile updated');
  });

  test('should change theme', async ({ page }) => {
    await page.click('[data-testid="appearance-tab"]');
    await page.click('[data-testid="theme-dark"]');

    await expect(page.locator('body')).toHaveClass(/dark-theme/);
  });

  test('should configure keyboard shortcuts', async ({ page }) => {
    await page.click('[data-testid="shortcuts-tab"]');
    await expect(page.locator('[data-testid="shortcuts-list"]')).toBeVisible();
  });
});

// ============================================================================
// Mobile Responsiveness Tests
// ============================================================================

test.describe('Mobile Responsiveness', () => {
  test.use({ viewport: { width: 375, height: 667 } });

  test('should show mobile navigation', async ({ page }) => {
    await login(page);
    await expect(page.locator('[data-testid="mobile-nav"]')).toBeVisible();
    await expect(page.locator('[data-testid="desktop-sidebar"]')).not.toBeVisible();
  });

  test('should open mobile menu', async ({ page }) => {
    await login(page);
    await page.click('[data-testid="mobile-menu-button"]');
    await expect(page.locator('[data-testid="mobile-menu"]')).toBeVisible();
  });

  test('should show swipe actions on email', async ({ page }) => {
    await login(page);
    const email = page.locator('[data-testid="email-item"]:first-child');

    await email.hover();
    await page.mouse.down();
    await page.mouse.move(200, 0);
    await page.mouse.up();

    await expect(page.locator('[data-testid="swipe-actions"]')).toBeVisible();
  });
});

// ============================================================================
// Accessibility Tests
// ============================================================================

test.describe('Accessibility', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('should have proper heading structure', async ({ page }) => {
    const h1 = await page.locator('h1').count();
    expect(h1).toBe(1);
  });

  test('should have alt text on images', async ({ page }) => {
    const imagesWithoutAlt = await page.locator('img:not([alt])').count();
    expect(imagesWithoutAlt).toBe(0);
  });

  test('should be navigable with tab', async ({ page }) => {
    await page.keyboard.press('Tab');
    const focusedElement = await page.evaluate(() => document.activeElement?.tagName);
    expect(focusedElement).not.toBe('BODY');
  });

  test('should have visible focus indicators', async ({ page }) => {
    await page.keyboard.press('Tab');
    const focusedElement = page.locator(':focus');
    const outline = await focusedElement.evaluate(el => getComputedStyle(el).outline);
    expect(outline).not.toBe('none');
  });
});
