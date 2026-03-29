// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail E2E Tests
 *
 * End-to-end tests using Playwright for critical user flows.
 */

import { test, expect, Page } from '@playwright/test';

// ============================================================================
// Test Configuration
// ============================================================================

const BASE_URL = process.env.PLAYWRIGHT_BASE_URL || 'http://localhost:5173';

// ============================================================================
// Fixtures & Helpers
// ============================================================================

const testUser = {
  email: 'test@mycelix.mail',
  password: 'TestPassword123!',
  name: 'Test User',
};

const testEmail = {
  to: 'recipient@example.com',
  subject: 'Test Email Subject',
  body: 'This is a test email body content.',
};

async function login(page: Page) {
  await page.goto(`${BASE_URL}/login`);
  await page.fill('input[name="email"]', testUser.email);
  await page.fill('input[name="password"]', testUser.password);
  await page.click('button[type="submit"]');
  await page.waitForURL(`${BASE_URL}/inbox`);
}

async function waitForEmails(page: Page) {
  await page.waitForSelector('[data-testid="email-list"]', { timeout: 10000 });
}

// ============================================================================
// Authentication Tests
// ============================================================================

test.describe('Authentication', () => {
  test('should display login page', async ({ page }) => {
    await page.goto(`${BASE_URL}/login`);

    await expect(page.locator('h1')).toContainText('Sign In');
    await expect(page.locator('input[name="email"]')).toBeVisible();
    await expect(page.locator('input[name="password"]')).toBeVisible();
    await expect(page.locator('button[type="submit"]')).toBeVisible();
  });

  test('should show error for invalid credentials', async ({ page }) => {
    await page.goto(`${BASE_URL}/login`);

    await page.fill('input[name="email"]', 'invalid@test.com');
    await page.fill('input[name="password"]', 'wrongpassword');
    await page.click('button[type="submit"]');

    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
  });

  test('should successfully log in with valid credentials', async ({ page }) => {
    await login(page);

    await expect(page).toHaveURL(`${BASE_URL}/inbox`);
    await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
  });

  test('should successfully log out', async ({ page }) => {
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
    await waitForEmails(page);

    const emailList = page.locator('[data-testid="email-list"]');
    await expect(emailList).toBeVisible();
  });

  test('should show email preview on click', async ({ page }) => {
    await waitForEmails(page);

    const firstEmail = page.locator('[data-testid="email-item"]').first();
    await firstEmail.click();

    await expect(page.locator('[data-testid="email-preview"]')).toBeVisible();
  });

  test('should navigate between folders', async ({ page }) => {
    // Go to Sent
    await page.click('[data-testid="folder-sent"]');
    await expect(page).toHaveURL(/\/sent/);

    // Go to Drafts
    await page.click('[data-testid="folder-drafts"]');
    await expect(page).toHaveURL(/\/drafts/);

    // Go back to Inbox
    await page.click('[data-testid="folder-inbox"]');
    await expect(page).toHaveURL(/\/inbox/);
  });

  test('should mark email as read', async ({ page }) => {
    await waitForEmails(page);

    const unreadEmail = page.locator('[data-testid="email-item"][data-unread="true"]').first();

    if (await unreadEmail.count() > 0) {
      await unreadEmail.click();
      await page.waitForTimeout(500);

      await expect(unreadEmail).toHaveAttribute('data-unread', 'false');
    }
  });

  test('should archive email', async ({ page }) => {
    await waitForEmails(page);

    const firstEmail = page.locator('[data-testid="email-item"]').first();
    const emailId = await firstEmail.getAttribute('data-email-id');

    await firstEmail.hover();
    await page.click('[data-testid="archive-button"]');

    // Email should be removed from inbox
    await expect(page.locator(`[data-email-id="${emailId}"]`)).not.toBeVisible();
  });

  test('should delete email', async ({ page }) => {
    await waitForEmails(page);

    const firstEmail = page.locator('[data-testid="email-item"]').first();
    const emailId = await firstEmail.getAttribute('data-email-id');

    await firstEmail.hover();
    await page.click('[data-testid="delete-button"]');

    // Confirm deletion if dialog appears
    const confirmButton = page.locator('[data-testid="confirm-delete"]');
    if (await confirmButton.isVisible()) {
      await confirmButton.click();
    }

    await expect(page.locator(`[data-email-id="${emailId}"]`)).not.toBeVisible();
  });
});

// ============================================================================
// Compose Tests
// ============================================================================

test.describe('Compose Email', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('should open compose modal', async ({ page }) => {
    await page.click('[data-testid="compose-button"]');

    await expect(page.locator('[data-testid="compose-modal"]')).toBeVisible();
  });

  test('should send a new email', async ({ page }) => {
    await page.click('[data-testid="compose-button"]');

    await page.fill('input[name="to"]', testEmail.to);
    await page.fill('input[name="subject"]', testEmail.subject);
    await page.fill('[data-testid="email-body"]', testEmail.body);

    await page.click('[data-testid="send-button"]');

    // Should show success message
    await expect(page.locator('[data-testid="toast-success"]')).toBeVisible();

    // Modal should close
    await expect(page.locator('[data-testid="compose-modal"]')).not.toBeVisible();
  });

  test('should save draft', async ({ page }) => {
    await page.click('[data-testid="compose-button"]');

    await page.fill('input[name="to"]', testEmail.to);
    await page.fill('input[name="subject"]', 'Draft Test');
    await page.fill('[data-testid="email-body"]', 'This is a draft.');

    await page.click('[data-testid="save-draft-button"]');

    await expect(page.locator('[data-testid="toast-success"]')).toContainText('Draft saved');
  });

  test('should add attachments', async ({ page }) => {
    await page.click('[data-testid="compose-button"]');

    // Upload a file
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: 'test.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('Test file content'),
    });

    await expect(page.locator('[data-testid="attachment-item"]')).toBeVisible();
  });

  test('should show recipient suggestions', async ({ page }) => {
    await page.click('[data-testid="compose-button"]');

    await page.fill('input[name="to"]', 'john');

    await expect(page.locator('[data-testid="recipient-suggestions"]')).toBeVisible();
  });
});

// ============================================================================
// Search Tests
// ============================================================================

test.describe('Search', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('should focus search on keyboard shortcut', async ({ page }) => {
    await page.keyboard.press('/');

    await expect(page.locator('[data-testid="search-input"]')).toBeFocused();
  });

  test('should perform basic search', async ({ page }) => {
    const searchInput = page.locator('[data-testid="search-input"]');
    await searchInput.fill('important');
    await page.keyboard.press('Enter');

    await expect(page).toHaveURL(/search\?q=important/);
    await expect(page.locator('[data-testid="search-results"]')).toBeVisible();
  });

  test('should use natural language search', async ({ page }) => {
    const searchInput = page.locator('[data-testid="search-input"]');
    await searchInput.fill('emails from john last week');
    await page.keyboard.press('Enter');

    // Should parse and apply filters
    await expect(page.locator('[data-testid="active-filters"]')).toBeVisible();
  });

  test('should clear search', async ({ page }) => {
    await page.goto(`${BASE_URL}/search?q=test`);

    await page.click('[data-testid="clear-search"]');

    await expect(page).toHaveURL(`${BASE_URL}/inbox`);
  });
});

// ============================================================================
// Trust Network Tests
// ============================================================================

test.describe('Trust Network', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('should display trust score for email sender', async ({ page }) => {
    await waitForEmails(page);

    const firstEmail = page.locator('[data-testid="email-item"]').first();
    await firstEmail.click();

    await expect(page.locator('[data-testid="trust-indicator"]')).toBeVisible();
  });

  test('should open trust details panel', async ({ page }) => {
    await waitForEmails(page);

    const firstEmail = page.locator('[data-testid="email-item"]').first();
    await firstEmail.click();

    await page.click('[data-testid="trust-indicator"]');

    await expect(page.locator('[data-testid="trust-details-panel"]')).toBeVisible();
  });

  test('should create attestation', async ({ page }) => {
    await waitForEmails(page);

    const firstEmail = page.locator('[data-testid="email-item"]').first();
    await firstEmail.click();

    await page.click('[data-testid="trust-indicator"]');
    await page.click('[data-testid="create-attestation"]');

    // Fill attestation form
    await page.selectOption('[data-testid="attestation-type"]', 'professional');
    await page.fill('[data-testid="attestation-comment"]', 'Verified professional contact');
    await page.click('[data-testid="submit-attestation"]');

    await expect(page.locator('[data-testid="toast-success"]')).toContainText('Attestation created');
  });
});

// ============================================================================
// Workflow Tests
// ============================================================================

test.describe('Workflows', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
    await page.goto(`${BASE_URL}/settings/workflows`);
  });

  test('should display workflow list', async ({ page }) => {
    await expect(page.locator('[data-testid="workflow-list"]')).toBeVisible();
  });

  test('should open workflow builder', async ({ page }) => {
    await page.click('[data-testid="create-workflow"]');

    await expect(page.locator('[data-testid="workflow-builder"]')).toBeVisible();
  });

  test('should create simple workflow', async ({ page }) => {
    await page.click('[data-testid="create-workflow"]');

    // Name the workflow
    await page.fill('[data-testid="workflow-name"]', 'Test Workflow');

    // Add trigger
    await page.click('[data-testid="add-trigger"]');
    await page.click('[data-testid="trigger-email-received"]');

    // Add condition
    await page.click('[data-testid="add-condition"]');
    await page.selectOption('[data-testid="condition-field"]', 'from');
    await page.selectOption('[data-testid="condition-operator"]', 'contains');
    await page.fill('[data-testid="condition-value"]', '@company.com');

    // Add action
    await page.click('[data-testid="add-action"]');
    await page.click('[data-testid="action-add-label"]');
    await page.fill('[data-testid="action-label-name"]', 'Work');

    // Save
    await page.click('[data-testid="save-workflow"]');

    await expect(page.locator('[data-testid="toast-success"]')).toContainText('Workflow created');
  });
});

// ============================================================================
// Calendar Tests
// ============================================================================

test.describe('Calendar', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
    await page.goto(`${BASE_URL}/calendar`);
  });

  test('should display calendar view', async ({ page }) => {
    await expect(page.locator('[data-testid="calendar-view"]')).toBeVisible();
  });

  test('should switch between views', async ({ page }) => {
    // Day view
    await page.click('[data-testid="view-day"]');
    await expect(page.locator('[data-testid="day-view"]')).toBeVisible();

    // Week view
    await page.click('[data-testid="view-week"]');
    await expect(page.locator('[data-testid="week-view"]')).toBeVisible();

    // Month view
    await page.click('[data-testid="view-month"]');
    await expect(page.locator('[data-testid="month-view"]')).toBeVisible();
  });

  test('should create event from email', async ({ page }) => {
    await page.goto(`${BASE_URL}/inbox`);
    await waitForEmails(page);

    const firstEmail = page.locator('[data-testid="email-item"]').first();
    await firstEmail.click();

    await page.click('[data-testid="create-event-from-email"]');

    await expect(page.locator('[data-testid="event-modal"]')).toBeVisible();
  });
});

// ============================================================================
// Accessibility Tests
// ============================================================================

test.describe('Accessibility', () => {
  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('should navigate with keyboard', async ({ page }) => {
    await waitForEmails(page);

    // Tab to first email
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');

    // Navigate down with arrow keys
    await page.keyboard.press('ArrowDown');

    // Open with Enter
    await page.keyboard.press('Enter');

    await expect(page.locator('[data-testid="email-preview"]')).toBeVisible();
  });

  test('should support keyboard shortcuts', async ({ page }) => {
    await waitForEmails(page);

    // Compose new email
    await page.keyboard.press('c');
    await expect(page.locator('[data-testid="compose-modal"]')).toBeVisible();

    // Close with Escape
    await page.keyboard.press('Escape');
    await expect(page.locator('[data-testid="compose-modal"]')).not.toBeVisible();
  });

  test('should have proper ARIA labels', async ({ page }) => {
    await waitForEmails(page);

    // Check main navigation
    const nav = page.locator('nav[aria-label="Main navigation"]');
    await expect(nav).toBeVisible();

    // Check email list
    const emailList = page.locator('[role="listbox"]');
    await expect(emailList).toBeVisible();
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

  test('should save profile changes', async ({ page }) => {
    await page.click('[data-testid="settings-profile"]');

    await page.fill('input[name="displayName"]', 'Updated Name');
    await page.click('[data-testid="save-profile"]');

    await expect(page.locator('[data-testid="toast-success"]')).toContainText('Profile updated');
  });

  test('should update notification preferences', async ({ page }) => {
    await page.click('[data-testid="settings-notifications"]');

    await page.click('[data-testid="toggle-email-notifications"]');
    await page.click('[data-testid="save-notifications"]');

    await expect(page.locator('[data-testid="toast-success"]')).toBeVisible();
  });
});

// ============================================================================
// Performance Tests
// ============================================================================

test.describe('Performance', () => {
  test('should load inbox within 3 seconds', async ({ page }) => {
    const startTime = Date.now();

    await login(page);
    await waitForEmails(page);

    const loadTime = Date.now() - startTime;
    expect(loadTime).toBeLessThan(3000);
  });

  test('should handle large email list', async ({ page }) => {
    await login(page);

    // Scroll through list
    const emailList = page.locator('[data-testid="email-list"]');

    for (let i = 0; i < 5; i++) {
      await emailList.evaluate((el) => {
        el.scrollTop += 500;
      });
      await page.waitForTimeout(100);
    }

    // Should still be responsive
    const firstEmail = page.locator('[data-testid="email-item"]').first();
    await expect(firstEmail).toBeVisible();
  });
});

// ============================================================================
// Mobile Tests
// ============================================================================

test.describe('Mobile Responsiveness', () => {
  test.use({ viewport: { width: 375, height: 667 } });

  test.beforeEach(async ({ page }) => {
    await login(page);
  });

  test('should show mobile navigation', async ({ page }) => {
    await expect(page.locator('[data-testid="mobile-menu-button"]')).toBeVisible();
  });

  test('should open mobile menu', async ({ page }) => {
    await page.click('[data-testid="mobile-menu-button"]');

    await expect(page.locator('[data-testid="mobile-nav"]')).toBeVisible();
  });

  test('should have swipe actions on emails', async ({ page }) => {
    await waitForEmails(page);

    const firstEmail = page.locator('[data-testid="email-item"]').first();
    const box = await firstEmail.boundingBox();

    if (box) {
      // Simulate swipe left
      await page.mouse.move(box.x + box.width - 20, box.y + box.height / 2);
      await page.mouse.down();
      await page.mouse.move(box.x + 20, box.y + box.height / 2, { steps: 10 });
      await page.mouse.up();

      // Should show action buttons
      await expect(page.locator('[data-testid="swipe-actions"]')).toBeVisible();
    }
  });
});
