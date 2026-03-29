// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Compose Email E2E Tests
 */

import { test, expect } from '@playwright/test';
import { setupAuthenticatedUser } from '../helpers';

test.describe('Compose Email', () => {
  test.beforeEach(async ({ page, context }) => {
    await setupAuthenticatedUser(context);
    await page.goto('/compose');
  });

  test('should display compose form', async ({ page }) => {
    await expect(page.getByRole('textbox', { name: /to/i })).toBeVisible();
    await expect(page.getByRole('textbox', { name: /subject/i })).toBeVisible();
    await expect(page.getByRole('textbox', { name: /body|message/i })).toBeVisible();
  });

  test('should add recipients with autocomplete', async ({ page }) => {
    const toField = page.getByRole('textbox', { name: /to/i });
    await toField.fill('john');

    // Wait for autocomplete
    const suggestions = page.getByRole('listbox', { name: /contacts/i });
    await expect(suggestions).toBeVisible();

    // Select first suggestion
    await page.keyboard.press('ArrowDown');
    await page.keyboard.press('Enter');

    // Should show as chip
    await expect(page.getByText(/john@example.com/i)).toBeVisible();
  });

  test('should show trust indicator for recipients', async ({ page }) => {
    const toField = page.getByRole('textbox', { name: /to/i });
    await toField.fill('trusted@example.com');
    await page.keyboard.press('Enter');

    // Trust badge should appear
    const chip = page.locator('[data-recipient]').filter({ hasText: 'trusted@example.com' });
    await expect(chip.getByText(/trusted|known/i)).toBeVisible();
  });

  test('should toggle encryption', async ({ page }) => {
    const encryptToggle = page.getByRole('switch', { name: /encrypt/i });

    await expect(encryptToggle).toHaveAttribute('aria-checked', 'false');
    await encryptToggle.click();
    await expect(encryptToggle).toHaveAttribute('aria-checked', 'true');

    // Encryption indicator should show
    await expect(page.getByText(/encrypted with.*quantum/i)).toBeVisible();
  });

  test('should attach files', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');

    await fileInput.setInputFiles({
      name: 'test-document.pdf',
      mimeType: 'application/pdf',
      buffer: Buffer.from('fake pdf content'),
    });

    // Attachment should appear
    await expect(page.getByText('test-document.pdf')).toBeVisible();
  });

  test('should send email', async ({ page }) => {
    // Mock send API
    await page.route('**/api/graphql', async (route) => {
      const body = JSON.parse(route.request().postData() || '{}');
      if (body.query?.includes('SendEmail')) {
        await route.fulfill({
          status: 200,
          body: JSON.stringify({ data: { sendEmail: { id: 'new-email-id' } } }),
        });
      } else {
        await route.continue();
      }
    });

    // Fill form
    await page.getByRole('textbox', { name: /to/i }).fill('recipient@example.com');
    await page.keyboard.press('Enter');
    await page.getByRole('textbox', { name: /subject/i }).fill('Test Subject');
    await page.getByRole('textbox', { name: /body|message/i }).fill('Test body content');

    // Send
    await page.getByRole('button', { name: /send/i }).click();

    // Should show success and redirect
    await expect(page.getByText(/sent successfully/i)).toBeVisible();
    await expect(page).toHaveURL(/\/inbox|\/sent/);
  });

  test('should save draft automatically', async ({ page }) => {
    // Fill some content
    await page.getByRole('textbox', { name: /subject/i }).fill('Draft Subject');
    await page.getByRole('textbox', { name: /body|message/i }).fill('Draft content');

    // Wait for auto-save
    await page.waitForTimeout(3000);

    // Draft indicator should show
    await expect(page.getByText(/saved|draft/i)).toBeVisible();
  });

  test('should use keyboard shortcut to send', async ({ page }) => {
    // Mock send
    let sendCalled = false;
    await page.route('**/api/graphql', async (route) => {
      const body = JSON.parse(route.request().postData() || '{}');
      if (body.query?.includes('SendEmail')) {
        sendCalled = true;
        await route.fulfill({
          status: 200,
          body: JSON.stringify({ data: { sendEmail: { id: 'email-id' } } }),
        });
      } else {
        await route.continue();
      }
    });

    // Fill required fields
    await page.getByRole('textbox', { name: /to/i }).fill('test@example.com');
    await page.keyboard.press('Enter');
    await page.getByRole('textbox', { name: /subject/i }).fill('Test');
    await page.getByRole('textbox', { name: /body|message/i }).fill('Content');

    // Ctrl+Enter to send
    await page.keyboard.press('Control+Enter');

    await page.waitForTimeout(1000);
    expect(sendCalled).toBe(true);
  });
});

test.describe('Compose - Reply/Forward', () => {
  test.beforeEach(async ({ page, context }) => {
    await setupAuthenticatedUser(context);
  });

  test('should pre-fill reply fields', async ({ page }) => {
    await page.goto('/compose?replyTo=sender@example.com&subject=Re: Original Subject');

    await expect(page.getByText('sender@example.com')).toBeVisible();
    await expect(page.getByRole('textbox', { name: /subject/i })).toHaveValue('Re: Original Subject');
  });

  test('should include original message in reply', async ({ page }) => {
    await page.goto('/compose?replyTo=sender@example.com&inReplyTo=original-email-id');

    // Should show quoted content
    await expect(page.getByText(/on .* wrote:/i)).toBeVisible();
  });
});

test.describe('Compose - Accessibility', () => {
  test.beforeEach(async ({ page, context }) => {
    await setupAuthenticatedUser(context);
    await page.goto('/compose');
  });

  test('should have proper form labels', async ({ page }) => {
    const toField = page.getByRole('textbox', { name: /to/i });
    const subjectField = page.getByRole('textbox', { name: /subject/i });

    await expect(toField).toBeVisible();
    await expect(subjectField).toBeVisible();
  });

  test('should announce validation errors', async ({ page }) => {
    // Try to send without required fields
    await page.getByRole('button', { name: /send/i }).click();

    // Error should be announced
    const error = page.getByRole('alert');
    await expect(error).toBeVisible();
  });

  test('should support keyboard-only operation', async ({ page }) => {
    // Tab through all fields
    await page.keyboard.press('Tab'); // To field
    await page.keyboard.type('test@example.com');
    await page.keyboard.press('Enter');

    await page.keyboard.press('Tab'); // CC (skip)
    await page.keyboard.press('Tab'); // Subject
    await page.keyboard.type('Test Subject');

    await page.keyboard.press('Tab'); // Body
    await page.keyboard.type('Test body');

    // Should be able to send
    await page.keyboard.press('Control+Enter');
  });
});
