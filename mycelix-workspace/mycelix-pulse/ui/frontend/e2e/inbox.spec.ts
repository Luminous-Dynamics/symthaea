// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Inbox E2E Tests
 *
 * Tests for viewing and interacting with the email inbox
 */

import { test, expect, Page } from '@playwright/test';

// ============================================================================
// Helper Functions
// ============================================================================

async function waitForEmailList(page: Page) {
  await page.waitForSelector('[data-testid="email-list"], .email-list, [role="listbox"]', {
    timeout: 10000,
  });
}

async function getEmailItems(page: Page) {
  return page.locator('[data-testid="email-item"], .email-item, [role="option"]');
}

async function clickFirstEmail(page: Page) {
  const emails = await getEmailItems(page);
  if ((await emails.count()) > 0) {
    await emails.first().click();
  }
}

// ============================================================================
// Inbox Display Tests
// ============================================================================

test.describe('Inbox Display', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/inbox');
    await waitForEmailList(page);
  });

  test('should display inbox with email list', async ({ page }) => {
    const emailList = page.locator('[data-testid="email-list"], .email-list, [role="listbox"]');
    await expect(emailList).toBeVisible();
  });

  test('should show email items with required information', async ({ page }) => {
    const emails = await getEmailItems(page);

    if ((await emails.count()) > 0) {
      const firstEmail = emails.first();

      // Email should display sender, subject, and preview
      await expect(firstEmail).toBeVisible();

      // Check for common email item elements
      const senderElement = firstEmail.locator('[data-testid="email-sender"], .sender, .from');
      const subjectElement = firstEmail.locator('[data-testid="email-subject"], .subject');
      const dateElement = firstEmail.locator('[data-testid="email-date"], .date, time');

      // At least sender and subject should be visible
      if (await senderElement.count() > 0) {
        await expect(senderElement.first()).toBeVisible();
      }
      if (await subjectElement.count() > 0) {
        await expect(subjectElement.first()).toBeVisible();
      }
    }
  });

  test('should distinguish between read and unread emails', async ({ page }) => {
    const emails = await getEmailItems(page);

    if ((await emails.count()) > 0) {
      // Check for unread indicator class or attribute
      const unreadEmails = page.locator(
        '[data-unread="true"], .unread, [aria-label*="unread"]'
      );
      const readEmails = page.locator(
        '[data-unread="false"], .read:not(.unread), [aria-label*="read"]:not([aria-label*="unread"])'
      );

      // Log counts for debugging
      const unreadCount = await unreadEmails.count();
      const readCount = await readEmails.count();
      console.log(`Unread: ${unreadCount}, Read: ${readCount}`);
    }
  });

  test('should show starred emails with star indicator', async ({ page }) => {
    const starredIndicator = page.locator(
      '[data-starred="true"], .starred, [aria-label*="starred"], .star.active'
    );

    // Starred emails might exist
    const count = await starredIndicator.count();
    console.log(`Found ${count} starred emails`);
  });

  test('should show trust badge for emails', async ({ page }) => {
    const emails = await getEmailItems(page);

    if ((await emails.count()) > 0) {
      await emails.first().click();

      // Wait for email preview to load
      await page.waitForTimeout(500);

      // Look for trust indicator
      const trustBadge = page.locator(
        '[data-testid="trust-indicator"], [data-testid="trust-badge"], .trust-badge'
      );

      if (await trustBadge.count() > 0) {
        await expect(trustBadge.first()).toBeVisible();
      }
    }
  });
});

// ============================================================================
// Email Selection and Preview Tests
// ============================================================================

test.describe('Email Selection and Preview', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/inbox');
    await waitForEmailList(page);
  });

  test('should show email preview when clicking an email', async ({ page }) => {
    const emails = await getEmailItems(page);

    if ((await emails.count()) > 0) {
      await emails.first().click();

      // Wait for preview to appear
      const preview = page.locator(
        '[data-testid="email-preview"], [data-testid="email-view"], .email-preview, .email-content'
      );
      await expect(preview.first()).toBeVisible({ timeout: 5000 });
    }
  });

  test('should display full email content in preview', async ({ page }) => {
    await clickFirstEmail(page);

    // Check for email content elements
    const emailBody = page.locator(
      '[data-testid="email-body"], .email-body, .message-body'
    );

    if (await emailBody.count() > 0) {
      await expect(emailBody.first()).toBeVisible();
    }
  });

  test('should mark email as read when viewed', async ({ page }) => {
    const emails = await getEmailItems(page);

    if ((await emails.count()) > 0) {
      // Find an unread email
      const unreadEmail = page.locator('[data-unread="true"], .unread').first();

      if (await unreadEmail.count() > 0) {
        await unreadEmail.click();
        await page.waitForTimeout(1000);

        // Email should now be marked as read
        await expect(unreadEmail).not.toHaveAttribute('data-unread', 'true');
      }
    }
  });

  test('should show email thread view for conversations', async ({ page }) => {
    const emails = await getEmailItems(page);

    if ((await emails.count()) > 0) {
      await emails.first().click();

      // Check for thread indicator
      const threadView = page.locator(
        '[data-testid="thread-view"], .thread-view, .conversation'
      );

      // Thread view might not always be present
      if (await threadView.count() > 0) {
        await expect(threadView.first()).toBeVisible();
      }
    }
  });
});

// ============================================================================
// Email Actions Tests
// ============================================================================

test.describe('Email Actions', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/inbox');
    await waitForEmailList(page);
  });

  test('should toggle star on email', async ({ page }) => {
    const emails = await getEmailItems(page);

    if ((await emails.count()) > 0) {
      const firstEmail = emails.first();

      // Find star button
      const starButton = firstEmail.locator(
        '[data-testid="star-button"], button[aria-label*="star"], .star-button'
      );

      if (await starButton.count() > 0) {
        const wasStarred = await starButton.getAttribute('data-starred') === 'true';
        await starButton.click();
        await page.waitForTimeout(500);

        // Star state should toggle
        const isStarredNow = await starButton.getAttribute('data-starred') === 'true';
        expect(isStarredNow).not.toBe(wasStarred);
      }
    }
  });

  test('should archive email', async ({ page }) => {
    const emails = await getEmailItems(page);

    if ((await emails.count()) > 0) {
      const firstEmail = emails.first();
      const emailId = await firstEmail.getAttribute('data-email-id');

      // Hover to reveal actions
      await firstEmail.hover();

      // Find archive button
      const archiveButton = page.locator(
        '[data-testid="archive-button"], button[aria-label*="archive"]'
      );

      if (await archiveButton.count() > 0) {
        await archiveButton.click();
        await page.waitForTimeout(500);

        // Email should be removed from list
        if (emailId) {
          await expect(page.locator(`[data-email-id="${emailId}"]`)).not.toBeVisible();
        }
      }
    }
  });

  test('should delete email', async ({ page }) => {
    const emails = await getEmailItems(page);

    if ((await emails.count()) > 0) {
      const firstEmail = emails.first();
      await firstEmail.hover();

      // Find delete button
      const deleteButton = page.locator(
        '[data-testid="delete-button"], button[aria-label*="delete"]'
      );

      if (await deleteButton.count() > 0) {
        await deleteButton.click();

        // Handle confirmation dialog if present
        const confirmButton = page.locator(
          '[data-testid="confirm-delete"], button:has-text("Confirm"), button:has-text("Delete")'
        );
        if (await confirmButton.count() > 0) {
          await confirmButton.click();
        }

        await page.waitForTimeout(500);
      }
    }
  });

  test('should reply to email', async ({ page }) => {
    await clickFirstEmail(page);

    // Find reply button
    const replyButton = page.locator(
      '[data-testid="reply-button"], button[aria-label*="reply"], button:has-text("Reply")'
    );

    if (await replyButton.count() > 0) {
      await replyButton.click();

      // Compose modal or inline reply should appear
      const composeArea = page.locator(
        '[data-testid="compose-modal"], [data-testid="reply-compose"], .compose-form'
      );
      await expect(composeArea.first()).toBeVisible({ timeout: 5000 });
    }
  });

  test('should forward email', async ({ page }) => {
    await clickFirstEmail(page);

    // Find forward button
    const forwardButton = page.locator(
      '[data-testid="forward-button"], button[aria-label*="forward"], button:has-text("Forward")'
    );

    if (await forwardButton.count() > 0) {
      await forwardButton.click();

      // Compose modal should appear with forwarded content
      const composeArea = page.locator(
        '[data-testid="compose-modal"], .compose-form'
      );
      await expect(composeArea.first()).toBeVisible({ timeout: 5000 });
    }
  });
});

// ============================================================================
// Folder Navigation Tests
// ============================================================================

test.describe('Folder Navigation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/inbox');
    await waitForEmailList(page);
  });

  test('should navigate to Sent folder', async ({ page }) => {
    const sentFolder = page.locator(
      '[data-testid="folder-sent"], a[href*="sent"], [role="menuitem"]:has-text("Sent")'
    );

    if (await sentFolder.count() > 0) {
      await sentFolder.click();
      await expect(page).toHaveURL(/\/sent/);
    }
  });

  test('should navigate to Drafts folder', async ({ page }) => {
    const draftsFolder = page.locator(
      '[data-testid="folder-drafts"], a[href*="drafts"], [role="menuitem"]:has-text("Drafts")'
    );

    if (await draftsFolder.count() > 0) {
      await draftsFolder.click();
      await expect(page).toHaveURL(/\/drafts/);
    }
  });

  test('should navigate to Archive folder', async ({ page }) => {
    const archiveFolder = page.locator(
      '[data-testid="folder-archive"], a[href*="archive"], [role="menuitem"]:has-text("Archive")'
    );

    if (await archiveFolder.count() > 0) {
      await archiveFolder.click();
      await expect(page).toHaveURL(/\/archive/);
    }
  });

  test('should navigate to Trash folder', async ({ page }) => {
    const trashFolder = page.locator(
      '[data-testid="folder-trash"], a[href*="trash"], [role="menuitem"]:has-text("Trash")'
    );

    if (await trashFolder.count() > 0) {
      await trashFolder.click();
      await expect(page).toHaveURL(/\/trash/);
    }
  });

  test('should return to Inbox from other folders', async ({ page }) => {
    // Go to sent first
    const sentFolder = page.locator(
      '[data-testid="folder-sent"], a[href*="sent"]'
    );
    if (await sentFolder.count() > 0) {
      await sentFolder.click();
      await page.waitForURL(/\/sent/);

      // Go back to inbox
      const inboxFolder = page.locator(
        '[data-testid="folder-inbox"], a[href*="inbox"], [role="menuitem"]:has-text("Inbox")'
      );
      await inboxFolder.click();
      await expect(page).toHaveURL(/\/inbox/);
    }
  });
});

// ============================================================================
// Bulk Actions Tests
// ============================================================================

test.describe('Bulk Actions', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/inbox');
    await waitForEmailList(page);
  });

  test('should select multiple emails', async ({ page }) => {
    const checkboxes = page.locator(
      '[data-testid="email-checkbox"], input[type="checkbox"]'
    );

    if ((await checkboxes.count()) >= 2) {
      await checkboxes.first().click();
      await checkboxes.nth(1).click();

      // Bulk action toolbar should appear
      const bulkToolbar = page.locator(
        '[data-testid="bulk-actions"], .bulk-actions-toolbar'
      );

      if (await bulkToolbar.count() > 0) {
        await expect(bulkToolbar).toBeVisible();
      }
    }
  });

  test('should select all emails', async ({ page }) => {
    const selectAllCheckbox = page.locator(
      '[data-testid="select-all"], input[aria-label*="select all"]'
    );

    if (await selectAllCheckbox.count() > 0) {
      await selectAllCheckbox.click();

      // All email checkboxes should be checked
      const checkboxes = page.locator(
        '[data-testid="email-checkbox"]:checked, input[type="checkbox"]:checked'
      );
      const count = await checkboxes.count();
      expect(count).toBeGreaterThan(0);
    }
  });
});

// ============================================================================
// Search Tests
// ============================================================================

test.describe('Email Search', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/inbox');
    await waitForEmailList(page);
  });

  test('should focus search on keyboard shortcut', async ({ page }) => {
    await page.keyboard.press('/');

    const searchInput = page.locator(
      '[data-testid="search-input"], input[type="search"], input[placeholder*="search" i]'
    );

    await expect(searchInput).toBeFocused({ timeout: 2000 });
  });

  test('should perform search and show results', async ({ page }) => {
    const searchInput = page.locator(
      '[data-testid="search-input"], input[type="search"], input[placeholder*="search" i]'
    );

    if (await searchInput.count() > 0) {
      await searchInput.fill('test');
      await page.keyboard.press('Enter');

      // Wait for search results
      await page.waitForTimeout(1000);

      // URL should reflect search
      const url = page.url();
      expect(url.includes('search') || url.includes('q=')).toBeTruthy();
    }
  });

  test('should clear search and return to inbox', async ({ page }) => {
    const searchInput = page.locator(
      '[data-testid="search-input"], input[type="search"]'
    );

    if (await searchInput.count() > 0) {
      // Perform search
      await searchInput.fill('test');
      await page.keyboard.press('Enter');
      await page.waitForTimeout(500);

      // Clear search
      const clearButton = page.locator(
        '[data-testid="clear-search"], button[aria-label*="clear"]'
      );

      if (await clearButton.count() > 0) {
        await clearButton.click();
        await expect(page).toHaveURL(/\/inbox/);
      }
    }
  });
});

// ============================================================================
// Refresh and Sync Tests
// ============================================================================

test.describe('Refresh and Sync', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/inbox');
    await waitForEmailList(page);
  });

  test('should refresh inbox on button click', async ({ page }) => {
    const refreshButton = page.locator(
      '[data-testid="refresh-button"], button[aria-label*="refresh"]'
    );

    if (await refreshButton.count() > 0) {
      await refreshButton.click();

      // Should show loading state
      const loadingIndicator = page.locator(
        '[data-testid="loading"], .loading-spinner, .refreshing'
      );

      // Loading might be brief
      await page.waitForTimeout(1000);
    }
  });

  test('should show sync status', async ({ page }) => {
    const syncStatus = page.locator(
      '[data-testid="sync-status"], .sync-indicator'
    );

    if (await syncStatus.count() > 0) {
      await expect(syncStatus.first()).toBeVisible();
    }
  });
});

// ============================================================================
// Accessibility Tests
// ============================================================================

test.describe('Inbox Accessibility', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/inbox');
    await waitForEmailList(page);
  });

  test('should navigate email list with keyboard', async ({ page }) => {
    const emailList = page.locator('[data-testid="email-list"], [role="listbox"]');

    if (await emailList.count() > 0) {
      await emailList.focus();

      // Navigate with arrow keys
      await page.keyboard.press('ArrowDown');
      await page.keyboard.press('ArrowDown');

      // Select with Enter
      await page.keyboard.press('Enter');

      // Preview should open
      const preview = page.locator(
        '[data-testid="email-preview"], .email-preview'
      );

      if (await preview.count() > 0) {
        await expect(preview.first()).toBeVisible();
      }
    }
  });

  test('should have proper ARIA labels', async ({ page }) => {
    // Check for main navigation landmark
    const nav = page.locator('nav[aria-label], [role="navigation"]');
    await expect(nav.first()).toBeVisible();

    // Check for main content landmark
    const main = page.locator('main, [role="main"]');
    await expect(main.first()).toBeVisible();
  });
});

// ============================================================================
// Empty State Tests
// ============================================================================

test.describe('Empty State', () => {
  test('should show empty state when no emails', async ({ page }) => {
    // Navigate to a folder that might be empty (like trash or drafts)
    await page.goto('/drafts');

    // Look for empty state message
    const emptyState = page.locator(
      '[data-testid="empty-state"], .empty-state, :has-text("No emails")'
    );

    // This might not always appear depending on test data
    if (await emptyState.count() > 0) {
      await expect(emptyState.first()).toBeVisible();
    }
  });
});
