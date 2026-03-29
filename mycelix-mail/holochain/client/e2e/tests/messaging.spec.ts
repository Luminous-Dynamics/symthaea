// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * E2E Tests - Messaging Flow
 *
 * Tests the complete email messaging flow between agents
 */

import { test, expect, Page } from '@playwright/test';

test.describe('Messaging', () => {
  let alicePage: Page;
  let bobPage: Page;

  test.beforeAll(async ({ browser }) => {
    // Create two browser contexts for Alice and Bob
    const aliceContext = await browser.newContext({
      storageState: { cookies: [], origins: [] },
    });
    const bobContext = await browser.newContext({
      storageState: { cookies: [], origins: [] },
    });

    alicePage = await aliceContext.newPage();
    bobPage = await bobContext.newPage();
  });

  test.afterAll(async () => {
    await alicePage.close();
    await bobPage.close();
  });

  test('should compose and send an email', async () => {
    await alicePage.goto('/');

    // Wait for app to load
    await expect(alicePage.getByTestId('inbox-view')).toBeVisible({ timeout: 10000 });

    // Click compose button
    await alicePage.getByRole('button', { name: /compose/i }).click();

    // Wait for compose modal
    await expect(alicePage.getByTestId('compose-modal')).toBeVisible();

    // Fill in email details
    await alicePage.getByLabel('To').fill('bob@example.com');
    await alicePage.getByLabel('Subject').fill('Test Email from E2E');
    await alicePage.getByLabel('Message').fill('This is a test email sent from Playwright E2E tests.');

    // Send the email
    await alicePage.getByRole('button', { name: /send/i }).click();

    // Verify success message
    await expect(alicePage.getByText(/email sent/i)).toBeVisible({ timeout: 5000 });

    // Verify email appears in sent folder
    await alicePage.getByRole('link', { name: /sent/i }).click();
    await expect(alicePage.getByText('Test Email from E2E')).toBeVisible();
  });

  test('should receive and display an email', async () => {
    await bobPage.goto('/');

    // Wait for inbox to load
    await expect(bobPage.getByTestId('inbox-view')).toBeVisible({ timeout: 10000 });

    // Refresh inbox
    await bobPage.getByRole('button', { name: /refresh/i }).click();

    // Wait for sync
    await bobPage.waitForTimeout(2000);

    // Check for the email from Alice
    await expect(bobPage.getByText('Test Email from E2E')).toBeVisible({ timeout: 10000 });
  });

  test('should open and read an email', async () => {
    await bobPage.goto('/');
    await expect(bobPage.getByTestId('inbox-view')).toBeVisible({ timeout: 10000 });

    // Click on the email
    await bobPage.getByText('Test Email from E2E').click();

    // Verify email content is displayed
    await expect(bobPage.getByTestId('email-view')).toBeVisible();
    await expect(bobPage.getByText('This is a test email sent from Playwright E2E tests.')).toBeVisible();

    // Verify sender info
    await expect(bobPage.getByText(/alice/i)).toBeVisible();
  });

  test('should reply to an email', async () => {
    await bobPage.goto('/');
    await bobPage.getByText('Test Email from E2E').click();

    // Click reply button
    await bobPage.getByRole('button', { name: /reply/i }).click();

    // Wait for compose modal with pre-filled data
    await expect(bobPage.getByTestId('compose-modal')).toBeVisible();
    await expect(bobPage.getByLabel('To')).toHaveValue(/alice/i);
    await expect(bobPage.getByLabel('Subject')).toHaveValue(/re: test email/i);

    // Add reply content
    await bobPage.getByLabel('Message').fill('Thanks for your email!');

    // Send reply
    await bobPage.getByRole('button', { name: /send/i }).click();
    await expect(bobPage.getByText(/email sent/i)).toBeVisible();
  });

  test('should forward an email', async () => {
    await bobPage.goto('/');
    await bobPage.getByText('Test Email from E2E').click();

    // Click forward button
    await bobPage.getByRole('button', { name: /forward/i }).click();

    // Wait for compose modal
    await expect(bobPage.getByTestId('compose-modal')).toBeVisible();
    await expect(bobPage.getByLabel('Subject')).toHaveValue(/fwd: test email/i);

    // Fill in recipient
    await bobPage.getByLabel('To').fill('carol@example.com');

    // Send forward
    await bobPage.getByRole('button', { name: /send/i }).click();
    await expect(bobPage.getByText(/email sent/i)).toBeVisible();
  });

  test('should archive an email', async () => {
    await bobPage.goto('/');
    await bobPage.getByText('Test Email from E2E').first().click();

    // Click archive button
    await bobPage.getByRole('button', { name: /archive/i }).click();

    // Verify email is removed from inbox
    await expect(bobPage.getByText('Test Email from E2E')).not.toBeVisible();

    // Navigate to archive folder
    await bobPage.getByRole('link', { name: /archive/i }).click();

    // Verify email is in archive
    await expect(bobPage.getByText('Test Email from E2E')).toBeVisible();
  });

  test('should delete an email', async () => {
    await bobPage.goto('/');

    // Check if there are any emails
    const emails = bobPage.getByTestId('email-list-item');
    const count = await emails.count();

    if (count > 0) {
      // Click on first email
      await emails.first().click();

      // Click delete button
      await bobPage.getByRole('button', { name: /delete/i }).click();

      // Confirm deletion if dialog appears
      const confirmButton = bobPage.getByRole('button', { name: /confirm/i });
      if (await confirmButton.isVisible()) {
        await confirmButton.click();
      }

      // Verify email is removed
      await expect(bobPage.getByText(/moved to trash/i)).toBeVisible();
    }
  });

  test('should search emails', async () => {
    await alicePage.goto('/');

    // Type in search box
    await alicePage.getByPlaceholder(/search/i).fill('test email');
    await alicePage.keyboard.press('Enter');

    // Wait for search results
    await alicePage.waitForTimeout(1000);

    // Verify search results contain the email
    await expect(alicePage.getByText('Test Email from E2E')).toBeVisible();
  });

  test('should star an email', async () => {
    await alicePage.goto('/');

    // Find star button on first email
    const starButton = alicePage.getByTestId('email-list-item').first().getByRole('button', { name: /star/i });

    // Click to star
    await starButton.click();

    // Verify email is starred
    await expect(starButton).toHaveAttribute('aria-pressed', 'true');

    // Navigate to starred folder
    await alicePage.getByRole('link', { name: /starred/i }).click();

    // Verify email appears in starred
    await expect(alicePage.getByTestId('email-list-item')).toBeVisible();
  });

  test('should apply labels to an email', async () => {
    await alicePage.goto('/');
    await alicePage.getByTestId('email-list-item').first().click();

    // Click labels/tags button
    await alicePage.getByRole('button', { name: /labels/i }).click();

    // Select a label
    await alicePage.getByRole('option', { name: /work/i }).click();

    // Verify label is applied
    await expect(alicePage.getByTestId('email-label').getByText(/work/i)).toBeVisible();
  });
});
