// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * E2E Tests - Contacts Management
 *
 * Tests contact CRUD operations, groups, and vCard import/export
 */

import { test, expect, Page } from '@playwright/test';

test.describe('Contacts', () => {
  let page: Page;

  test.beforeEach(async ({ browser }) => {
    const context = await browser.newContext();
    page = await context.newPage();
    await page.goto('/');
    await expect(page.getByTestId('inbox-view')).toBeVisible({ timeout: 10000 });

    // Navigate to contacts
    await page.getByRole('link', { name: /contacts/i }).click();
    await expect(page.getByTestId('contacts-view')).toBeVisible({ timeout: 5000 });
  });

  test.afterEach(async () => {
    await page.close();
  });

  test('should display contacts list', async () => {
    // Verify contacts list is visible
    await expect(page.getByTestId('contacts-list')).toBeVisible();

    // Verify search input
    await expect(page.getByPlaceholder(/search contacts/i)).toBeVisible();

    // Verify add contact button
    await expect(page.getByRole('button', { name: /add contact/i })).toBeVisible();
  });

  test('should create a new contact', async () => {
    // Click add contact button
    await page.getByRole('button', { name: /add contact/i }).click();

    // Wait for modal
    await expect(page.getByTestId('contact-modal')).toBeVisible();

    // Fill in contact details
    await page.getByLabel('Name').fill('John Doe');
    await page.getByLabel('Email').fill('john.doe@example.com');
    await page.getByLabel('Phone').fill('+1-555-123-4567');
    await page.getByLabel('Organization').fill('Acme Corp');
    await page.getByLabel('Notes').fill('Met at conference 2024');

    // Save contact
    await page.getByRole('button', { name: /save/i }).click();

    // Verify success
    await expect(page.getByText(/contact created/i)).toBeVisible();

    // Verify contact appears in list
    await expect(page.getByText('John Doe')).toBeVisible();
  });

  test('should edit an existing contact', async () => {
    // Click on a contact
    await page.getByText('John Doe').click();

    // Click edit button
    await page.getByRole('button', { name: /edit/i }).click();

    // Update organization
    await page.getByLabel('Organization').clear();
    await page.getByLabel('Organization').fill('New Company Inc');

    // Save changes
    await page.getByRole('button', { name: /save/i }).click();

    // Verify success
    await expect(page.getByText(/contact updated/i)).toBeVisible();

    // Verify changes are visible
    await expect(page.getByText('New Company Inc')).toBeVisible();
  });

  test('should delete a contact', async () => {
    // Click on a contact
    const contactName = 'John Doe';
    await page.getByText(contactName).click();

    // Click delete button
    await page.getByRole('button', { name: /delete/i }).click();

    // Confirm deletion
    await expect(page.getByTestId('confirm-dialog')).toBeVisible();
    await page.getByRole('button', { name: /confirm/i }).click();

    // Verify success
    await expect(page.getByText(/contact deleted/i)).toBeVisible();

    // Verify contact is removed from list
    await expect(page.getByText(contactName)).not.toBeVisible();
  });

  test('should search contacts', async () => {
    // Type in search box
    await page.getByPlaceholder(/search contacts/i).fill('john');

    // Wait for search results
    await page.waitForTimeout(500);

    // Verify search filters results
    const contacts = page.getByTestId('contact-item');
    const count = await contacts.count();

    for (let i = 0; i < count; i++) {
      const text = await contacts.nth(i).textContent();
      expect(text?.toLowerCase()).toContain('john');
    }
  });

  test('should create a contact group', async () => {
    // Click groups tab/button
    await page.getByRole('tab', { name: /groups/i }).click();

    // Click add group button
    await page.getByRole('button', { name: /add group/i }).click();

    // Fill in group name
    await page.getByLabel('Group Name').fill('Work Team');

    // Save group
    await page.getByRole('button', { name: /create/i }).click();

    // Verify success
    await expect(page.getByText(/group created/i)).toBeVisible();

    // Verify group appears in list
    await expect(page.getByText('Work Team')).toBeVisible();
  });

  test('should add contact to a group', async () => {
    // Click on a contact
    await page.getByTestId('contact-item').first().click();

    // Click add to group button
    await page.getByRole('button', { name: /add to group/i }).click();

    // Select a group
    await page.getByRole('option', { name: /work team/i }).click();

    // Verify success
    await expect(page.getByText(/added to group/i)).toBeVisible();
  });

  test('should import contacts from vCard', async () => {
    // Click import button
    await page.getByRole('button', { name: /import/i }).click();

    // Wait for import modal
    await expect(page.getByTestId('import-modal')).toBeVisible();

    // Create a sample vCard file
    const vCardContent = `BEGIN:VCARD
VERSION:3.0
FN:Jane Smith
EMAIL:jane.smith@example.com
TEL:+1-555-987-6543
ORG:Tech Company
END:VCARD`;

    // Upload the file
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: 'contacts.vcf',
      mimeType: 'text/vcard',
      buffer: Buffer.from(vCardContent),
    });

    // Click import
    await page.getByRole('button', { name: /import contacts/i }).click();

    // Verify success
    await expect(page.getByText(/contacts imported/i)).toBeVisible();

    // Verify contact was added
    await expect(page.getByText('Jane Smith')).toBeVisible();
  });

  test('should export contacts to vCard', async () => {
    // Select contacts
    await page.getByRole('checkbox', { name: /select all/i }).check();

    // Click export button
    await page.getByRole('button', { name: /export/i }).click();

    // Select format
    await page.getByRole('option', { name: /vcard/i }).click();

    // Wait for download
    const download = await page.waitForEvent('download');

    // Verify download
    expect(download.suggestedFilename()).toContain('.vcf');
  });

  test('should show contact trust level', async () => {
    // Click on a contact
    await page.getByTestId('contact-item').first().click();

    // Verify trust section is visible
    await expect(page.getByTestId('contact-trust')).toBeVisible();

    // Verify trust score is displayed
    await expect(page.getByText(/trust level/i)).toBeVisible();
  });

  test('should compose email to contact', async () => {
    // Click on a contact
    await page.getByTestId('contact-item').first().click();

    // Click compose button
    await page.getByRole('button', { name: /send email/i }).click();

    // Verify compose modal opens with pre-filled recipient
    await expect(page.getByTestId('compose-modal')).toBeVisible();
    const toField = page.getByLabel('To');
    const toValue = await toField.inputValue();
    expect(toValue).not.toBe('');
  });

  test('should view contact history', async () => {
    // Click on a contact
    await page.getByTestId('contact-item').first().click();

    // Click history tab
    await page.getByRole('tab', { name: /history/i }).click();

    // Verify history is displayed
    await expect(page.getByTestId('contact-history')).toBeVisible();

    // History shows emails exchanged
    await expect(page.getByText(/emails/i)).toBeVisible();
  });

  test('should display contact avatar', async () => {
    // Verify contact avatars are visible
    const avatars = page.getByTestId('contact-avatar');
    const count = await avatars.count();

    if (count > 0) {
      // First avatar should be visible
      await expect(avatars.first()).toBeVisible();

      // Avatars should have initials or image
      const avatar = avatars.first();
      const hasText = await avatar.textContent();
      const hasImage = await avatar.locator('img').count();

      expect(hasText || hasImage).toBeTruthy();
    }
  });
});
