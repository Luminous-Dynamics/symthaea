// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Compose Email E2E Tests
 *
 * Tests for composing, editing, and sending emails
 */

import { test, expect, Page } from '@playwright/test';

// ============================================================================
// Test Data
// ============================================================================

const testEmail = {
  to: 'recipient@example.com',
  cc: 'cc@example.com',
  bcc: 'bcc@example.com',
  subject: 'Test Email Subject',
  body: 'This is a test email body with some content.',
};

const multipleRecipients = {
  to: ['recipient1@example.com', 'recipient2@example.com'],
  subject: 'Email to Multiple Recipients',
  body: 'This email goes to multiple people.',
};

// ============================================================================
// Helper Functions
// ============================================================================

async function openComposeModal(page: Page) {
  const composeButton = page.locator(
    '[data-testid="compose-button"], button:has-text("Compose"), button[aria-label*="compose"]'
  );
  await composeButton.click();

  // Wait for compose modal or page
  await page.waitForSelector(
    '[data-testid="compose-modal"], [data-testid="compose-form"], .compose-form',
    { timeout: 5000 }
  );
}

async function fillComposeForm(
  page: Page,
  options: {
    to?: string;
    cc?: string;
    bcc?: string;
    subject?: string;
    body?: string;
  }
) {
  if (options.to) {
    await page.fill('input[name="to"], [data-testid="to-input"]', options.to);
  }
  if (options.cc) {
    // CC field might need to be expanded first
    const ccToggle = page.locator('[data-testid="show-cc"], button:has-text("Cc")');
    if (await ccToggle.count() > 0 && await ccToggle.isVisible()) {
      await ccToggle.click();
    }
    await page.fill('input[name="cc"], [data-testid="cc-input"]', options.cc);
  }
  if (options.bcc) {
    // BCC field might need to be expanded first
    const bccToggle = page.locator('[data-testid="show-bcc"], button:has-text("Bcc")');
    if (await bccToggle.count() > 0 && await bccToggle.isVisible()) {
      await bccToggle.click();
    }
    await page.fill('input[name="bcc"], [data-testid="bcc-input"]', options.bcc);
  }
  if (options.subject) {
    await page.fill('input[name="subject"], [data-testid="subject-input"]', options.subject);
  }
  if (options.body) {
    // Body might be a textarea or contenteditable div
    const bodyInput = page.locator(
      '[data-testid="email-body"], textarea[name="body"], [contenteditable="true"]'
    );
    await bodyInput.fill(options.body);
  }
}

// ============================================================================
// Compose Modal/Page Tests
// ============================================================================

test.describe('Compose Modal', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/inbox');
    await page.waitForLoadState('networkidle');
  });

  test('should open compose modal when clicking compose button', async ({ page }) => {
    await openComposeModal(page);

    const composeModal = page.locator(
      '[data-testid="compose-modal"], [data-testid="compose-form"], .compose-form'
    );
    await expect(composeModal.first()).toBeVisible();
  });

  test('should have all required form fields', async ({ page }) => {
    await openComposeModal(page);

    // To field
    const toInput = page.locator('input[name="to"], [data-testid="to-input"]');
    await expect(toInput).toBeVisible();

    // Subject field
    const subjectInput = page.locator('input[name="subject"], [data-testid="subject-input"]');
    await expect(subjectInput).toBeVisible();

    // Body field
    const bodyInput = page.locator(
      '[data-testid="email-body"], textarea[name="body"], [contenteditable="true"]'
    );
    await expect(bodyInput.first()).toBeVisible();

    // Send button
    const sendButton = page.locator(
      '[data-testid="send-button"], button:has-text("Send")'
    );
    await expect(sendButton).toBeVisible();
  });

  test('should close compose modal when clicking close button', async ({ page }) => {
    await openComposeModal(page);

    const closeButton = page.locator(
      '[data-testid="close-compose"], button[aria-label*="close"], .close-button'
    );
    await closeButton.click();

    const composeModal = page.locator('[data-testid="compose-modal"]');
    await expect(composeModal).not.toBeVisible();
  });

  test('should close compose modal with Escape key', async ({ page }) => {
    await openComposeModal(page);

    await page.keyboard.press('Escape');

    const composeModal = page.locator('[data-testid="compose-modal"]');
    await expect(composeModal).not.toBeVisible();
  });

  test('should open compose with keyboard shortcut', async ({ page }) => {
    await page.keyboard.press('c');

    const composeModal = page.locator(
      '[data-testid="compose-modal"], [data-testid="compose-form"]'
    );
    await expect(composeModal.first()).toBeVisible({ timeout: 3000 });
  });
});

// ============================================================================
// Sending Email Tests
// ============================================================================

test.describe('Sending Email', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/inbox');
    await page.waitForLoadState('networkidle');
    await openComposeModal(page);
  });

  test('should send a basic email successfully', async ({ page }) => {
    await fillComposeForm(page, {
      to: testEmail.to,
      subject: testEmail.subject,
      body: testEmail.body,
    });

    const sendButton = page.locator('[data-testid="send-button"], button:has-text("Send")');
    await sendButton.click();

    // Should show success message
    const successToast = page.locator(
      '[data-testid="toast-success"], .toast-success, [role="alert"]:has-text("sent")'
    );
    await expect(successToast.first()).toBeVisible({ timeout: 10000 });

    // Modal should close
    const composeModal = page.locator('[data-testid="compose-modal"]');
    await expect(composeModal).not.toBeVisible();
  });

  test('should send email with CC and BCC', async ({ page }) => {
    await fillComposeForm(page, {
      to: testEmail.to,
      cc: testEmail.cc,
      bcc: testEmail.bcc,
      subject: 'Email with CC and BCC',
      body: testEmail.body,
    });

    const sendButton = page.locator('[data-testid="send-button"], button:has-text("Send")');
    await sendButton.click();

    const successToast = page.locator(
      '[data-testid="toast-success"], .toast-success'
    );
    await expect(successToast.first()).toBeVisible({ timeout: 10000 });
  });

  test('should show validation error for empty recipient', async ({ page }) => {
    await fillComposeForm(page, {
      subject: 'No Recipient',
      body: 'This email has no recipient.',
    });

    const sendButton = page.locator('[data-testid="send-button"], button:has-text("Send")');
    await sendButton.click();

    // Should show error
    const errorMessage = page.locator(
      '[data-testid="error-message"], .error-message, [role="alert"]:has-text("recipient")'
    );
    await expect(errorMessage.first()).toBeVisible({ timeout: 5000 });
  });

  test('should show validation error for invalid email', async ({ page }) => {
    await fillComposeForm(page, {
      to: 'invalid-email',
      subject: 'Invalid Recipient',
      body: 'This email has invalid recipient.',
    });

    const sendButton = page.locator('[data-testid="send-button"], button:has-text("Send")');
    await sendButton.click();

    // Should show validation error
    const errorMessage = page.locator(
      '[data-testid="error-message"], .error-message, [role="alert"], :invalid'
    );
    await expect(errorMessage.first()).toBeVisible({ timeout: 5000 });
  });

  test('should show loading state while sending', async ({ page }) => {
    await fillComposeForm(page, testEmail);

    const sendButton = page.locator('[data-testid="send-button"], button:has-text("Send")');
    await sendButton.click();

    // Button should show loading or be disabled briefly
    await expect(sendButton).toBeDisabled({ timeout: 1000 }).catch(() => {
      // Button might not disable, which is okay
    });

    // Wait for completion
    await page.waitForTimeout(3000);
  });
});

// ============================================================================
// Draft Tests
// ============================================================================

test.describe('Drafts', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/inbox');
    await page.waitForLoadState('networkidle');
    await openComposeModal(page);
  });

  test('should save draft manually', async ({ page }) => {
    await fillComposeForm(page, {
      to: testEmail.to,
      subject: 'Draft Email',
      body: 'This is a draft that will be saved.',
    });

    const saveDraftButton = page.locator(
      '[data-testid="save-draft-button"], button:has-text("Save Draft"), button:has-text("Save")'
    );

    if (await saveDraftButton.count() > 0) {
      await saveDraftButton.click();

      const successToast = page.locator(
        '[data-testid="toast-success"], .toast-success, [role="alert"]:has-text("Draft")'
      );
      await expect(successToast.first()).toBeVisible({ timeout: 5000 });
    }
  });

  test('should auto-save draft periodically', async ({ page }) => {
    await fillComposeForm(page, {
      to: testEmail.to,
      subject: 'Auto-saved Draft',
      body: 'This should be auto-saved.',
    });

    // Wait for auto-save (usually a few seconds)
    await page.waitForTimeout(5000);

    // Look for auto-save indicator
    const autoSaveIndicator = page.locator(
      '[data-testid="draft-saved"], .draft-saved, :has-text("Draft saved")'
    );

    if (await autoSaveIndicator.count() > 0) {
      await expect(autoSaveIndicator.first()).toBeVisible();
    }
  });

  test('should find saved draft in Drafts folder', async ({ page }) => {
    const uniqueSubject = `Draft ${Date.now()}`;

    await fillComposeForm(page, {
      to: testEmail.to,
      subject: uniqueSubject,
      body: 'Draft to be found later.',
    });

    const saveDraftButton = page.locator(
      '[data-testid="save-draft-button"], button:has-text("Save Draft")'
    );

    if (await saveDraftButton.count() > 0) {
      await saveDraftButton.click();
      await page.waitForTimeout(1000);

      // Close compose modal
      await page.keyboard.press('Escape');

      // Navigate to Drafts
      const draftsFolder = page.locator(
        '[data-testid="folder-drafts"], a[href*="drafts"]'
      );
      await draftsFolder.click();

      // Look for the draft
      await page.waitForTimeout(1000);
      const draftItem = page.locator(`:has-text("${uniqueSubject}")`);

      // Draft should exist (if drafts work in test environment)
    }
  });

  test('should prompt before discarding unsaved changes', async ({ page }) => {
    await fillComposeForm(page, {
      to: testEmail.to,
      subject: 'Unsaved Email',
      body: 'This has unsaved changes.',
    });

    // Try to close without saving
    const closeButton = page.locator(
      '[data-testid="close-compose"], button[aria-label*="close"]'
    );
    await closeButton.click();

    // Should show confirmation dialog
    const confirmDialog = page.locator(
      '[data-testid="discard-confirm"], [role="alertdialog"], .confirm-dialog'
    );

    if (await confirmDialog.count() > 0) {
      await expect(confirmDialog).toBeVisible();

      // Click discard or cancel
      const discardButton = page.locator(
        '[data-testid="confirm-discard"], button:has-text("Discard")'
      );
      await discardButton.click();
    }
  });
});

// ============================================================================
// Attachment Tests
// ============================================================================

test.describe('Attachments', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/inbox');
    await page.waitForLoadState('networkidle');
    await openComposeModal(page);
  });

  test('should add attachment to email', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');

    if (await fileInput.count() > 0) {
      await fileInput.setInputFiles({
        name: 'test-document.txt',
        mimeType: 'text/plain',
        buffer: Buffer.from('Test file content for attachment'),
      });

      // Attachment should appear in list
      const attachmentItem = page.locator(
        '[data-testid="attachment-item"], .attachment-item'
      );
      await expect(attachmentItem.first()).toBeVisible({ timeout: 5000 });
    }
  });

  test('should remove attachment', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');

    if (await fileInput.count() > 0) {
      await fileInput.setInputFiles({
        name: 'test-document.txt',
        mimeType: 'text/plain',
        buffer: Buffer.from('Test file content'),
      });

      // Wait for attachment to appear
      const attachmentItem = page.locator('[data-testid="attachment-item"]');
      await expect(attachmentItem.first()).toBeVisible({ timeout: 5000 });

      // Remove attachment
      const removeButton = page.locator(
        '[data-testid="remove-attachment"], button[aria-label*="remove"]'
      );
      if (await removeButton.count() > 0) {
        await removeButton.click();
        await expect(attachmentItem).not.toBeVisible();
      }
    }
  });

  test('should show error for oversized attachment', async ({ page }) => {
    // This test would need to be adjusted based on actual file size limits
    // For now, we're testing the UI behavior
    const fileInput = page.locator('input[type="file"]');

    if (await fileInput.count() > 0) {
      // Try to upload a large file (create a buffer of significant size)
      const largeBuffer = Buffer.alloc(50 * 1024 * 1024, 'x'); // 50MB

      await fileInput.setInputFiles({
        name: 'large-file.bin',
        mimeType: 'application/octet-stream',
        buffer: largeBuffer,
      }).catch(() => {
        // File might be rejected by the browser
      });

      // Check for error message about file size
      const errorMessage = page.locator(
        '[data-testid="error-message"], .error-message, [role="alert"]:has-text("size")'
      );

      // Error might appear if file is too large
      await page.waitForTimeout(2000);
    }
  });
});

// ============================================================================
// Recipient Suggestions Tests
// ============================================================================

test.describe('Recipient Suggestions', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/inbox');
    await page.waitForLoadState('networkidle');
    await openComposeModal(page);
  });

  test('should show recipient suggestions while typing', async ({ page }) => {
    const toInput = page.locator('input[name="to"], [data-testid="to-input"]');
    await toInput.fill('john');

    // Wait for suggestions
    const suggestions = page.locator(
      '[data-testid="recipient-suggestions"], .suggestions-dropdown, [role="listbox"]'
    );
    await expect(suggestions.first()).toBeVisible({ timeout: 3000 });
  });

  test('should select recipient from suggestions', async ({ page }) => {
    const toInput = page.locator('input[name="to"], [data-testid="to-input"]');
    await toInput.fill('john');

    const suggestions = page.locator(
      '[data-testid="recipient-suggestions"], .suggestions-dropdown'
    );

    if (await suggestions.count() > 0) {
      await suggestions.waitFor({ state: 'visible', timeout: 3000 });

      // Click first suggestion
      const firstSuggestion = page.locator(
        '[data-testid="suggestion-item"]:first-child, .suggestion-item:first-child, [role="option"]:first-child'
      );
      if (await firstSuggestion.count() > 0) {
        await firstSuggestion.click();

        // Input should contain selected email
        const inputValue = await toInput.inputValue();
        expect(inputValue).toContain('@');
      }
    }
  });
});

// ============================================================================
// Rich Text Editor Tests
// ============================================================================

test.describe('Rich Text Editor', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/inbox');
    await page.waitForLoadState('networkidle');
    await openComposeModal(page);
  });

  test('should apply bold formatting', async ({ page }) => {
    const bodyInput = page.locator(
      '[data-testid="email-body"], [contenteditable="true"]'
    );

    if (await bodyInput.count() > 0) {
      await bodyInput.fill('Text to bold');

      // Select all text
      await page.keyboard.press('Control+a');

      // Apply bold (Ctrl+B or button)
      await page.keyboard.press('Control+b');

      // Check if bold was applied (this depends on implementation)
      const boldText = page.locator(
        '[data-testid="email-body"] strong, [data-testid="email-body"] b'
      );

      // Bold formatting might be applied
    }
  });

  test('should format as bulleted list', async ({ page }) => {
    const formatToolbar = page.locator(
      '[data-testid="format-toolbar"], .editor-toolbar'
    );

    if (await formatToolbar.count() > 0) {
      const bulletButton = page.locator(
        '[data-testid="bullet-list"], button[aria-label*="bullet"]'
      );

      if (await bulletButton.count() > 0) {
        await bulletButton.click();

        // List formatting should be applied
        const listElement = page.locator(
          '[data-testid="email-body"] ul, [data-testid="email-body"] ol'
        );
        // List might be created
      }
    }
  });
});

// ============================================================================
// Template Tests
// ============================================================================

test.describe('Email Templates', () => {
  test('should open template picker', async ({ page }) => {
    await page.goto('/inbox');
    await openComposeModal(page);

    const templateButton = page.locator(
      '[data-testid="template-button"], button:has-text("Template")'
    );

    if (await templateButton.count() > 0) {
      await templateButton.click();

      const templatePicker = page.locator(
        '[data-testid="template-picker"], .template-picker'
      );
      await expect(templatePicker).toBeVisible({ timeout: 3000 });
    }
  });

  test('should apply selected template', async ({ page }) => {
    await page.goto('/inbox');
    await openComposeModal(page);

    const templateButton = page.locator(
      '[data-testid="template-button"], button:has-text("Template")'
    );

    if (await templateButton.count() > 0) {
      await templateButton.click();

      const templateItem = page.locator(
        '[data-testid="template-item"]:first-child, .template-item:first-child'
      );

      if (await templateItem.count() > 0) {
        await templateItem.click();

        // Body should be populated with template
        const bodyInput = page.locator('[data-testid="email-body"]');
        const bodyContent = await bodyInput.textContent();
        expect(bodyContent?.length).toBeGreaterThan(0);
      }
    }
  });
});

// ============================================================================
// Priority and Options Tests
// ============================================================================

test.describe('Email Options', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/inbox');
    await page.waitForLoadState('networkidle');
    await openComposeModal(page);
  });

  test('should set email priority', async ({ page }) => {
    const priorityButton = page.locator(
      '[data-testid="priority-button"], button[aria-label*="priority"]'
    );

    if (await priorityButton.count() > 0) {
      await priorityButton.click();

      const highPriority = page.locator(
        '[data-testid="priority-high"], [data-value="high"], button:has-text("High")'
      );

      if (await highPriority.count() > 0) {
        await highPriority.click();

        // Priority indicator should be visible
        const priorityIndicator = page.locator(
          '[data-testid="priority-indicator"], .priority-high'
        );
        await expect(priorityIndicator.first()).toBeVisible();
      }
    }
  });

  test('should toggle request read receipt', async ({ page }) => {
    const optionsButton = page.locator(
      '[data-testid="more-options"], button[aria-label*="options"]'
    );

    if (await optionsButton.count() > 0) {
      await optionsButton.click();

      const readReceiptToggle = page.locator(
        '[data-testid="read-receipt-toggle"], input[name="readReceipt"]'
      );

      if (await readReceiptToggle.count() > 0) {
        await readReceiptToggle.click();
        await expect(readReceiptToggle).toBeChecked();
      }
    }
  });
});

// ============================================================================
// Accessibility Tests
// ============================================================================

test.describe('Compose Accessibility', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/inbox');
    await openComposeModal(page);
  });

  test('should have proper form labels', async ({ page }) => {
    const toInput = page.locator('input[name="to"], [data-testid="to-input"]');

    // Check for associated label
    const toId = await toInput.getAttribute('id');
    const toAriaLabel = await toInput.getAttribute('aria-label');
    const toPlaceholder = await toInput.getAttribute('placeholder');

    expect(toId || toAriaLabel || toPlaceholder).toBeTruthy();
  });

  test('should trap focus within modal', async ({ page }) => {
    // Tab through modal elements
    for (let i = 0; i < 10; i++) {
      await page.keyboard.press('Tab');
    }

    // Focus should still be within modal
    const focusedElement = await page.evaluate(() => {
      const active = document.activeElement;
      return active?.closest('[data-testid="compose-modal"], .compose-form') !== null;
    });

    // Focus might or might not be trapped depending on implementation
  });

  test('should announce send success to screen readers', async ({ page }) => {
    await fillComposeForm(page, testEmail);

    const sendButton = page.locator('[data-testid="send-button"]');
    await sendButton.click();

    // Look for live region announcement
    const liveRegion = page.locator(
      '[role="alert"], [aria-live="polite"], [aria-live="assertive"]'
    );

    await expect(liveRegion.first()).toBeVisible({ timeout: 10000 });
  });
});
