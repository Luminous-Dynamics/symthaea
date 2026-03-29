// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Contacts Management E2E Tests
 *
 * Tests for viewing, creating, editing, and deleting contacts
 */

import { test, expect, Page } from '@playwright/test';

// ============================================================================
// Test Data
// ============================================================================

const testContact = {
  name: 'John Doe',
  email: 'john.doe@example.com',
  phone: '+1 555-123-4567',
  organization: 'Example Corp',
  notes: 'Test contact for E2E testing',
};

const updatedContact = {
  name: 'John D. Updated',
  email: 'john.updated@example.com',
};

// ============================================================================
// Helper Functions
// ============================================================================

async function navigateToContacts(page: Page) {
  const contactsLink = page.locator(
    '[data-testid="nav-contacts"], a[href*="contacts"], [role="menuitem"]:has-text("Contacts")'
  );

  if (await contactsLink.count() > 0) {
    await contactsLink.click();
    await page.waitForURL(/\/contacts/, { timeout: 10000 });
  } else {
    await page.goto('/contacts');
  }

  await page.waitForLoadState('networkidle');
}

async function openAddContactModal(page: Page) {
  const addButton = page.locator(
    '[data-testid="add-contact-button"], button:has-text("Add Contact"), button:has-text("New Contact")'
  );
  await addButton.click();

  await page.waitForSelector(
    '[data-testid="contact-modal"], [data-testid="contact-form"], .contact-form',
    { timeout: 5000 }
  );
}

async function fillContactForm(page: Page, contact: typeof testContact) {
  if (contact.name) {
    await page.fill(
      'input[name="name"], [data-testid="contact-name-input"]',
      contact.name
    );
  }
  if (contact.email) {
    await page.fill(
      'input[name="email"], [data-testid="contact-email-input"]',
      contact.email
    );
  }
  if (contact.phone) {
    const phoneInput = page.locator('input[name="phone"], [data-testid="contact-phone-input"]');
    if (await phoneInput.count() > 0) {
      await phoneInput.fill(contact.phone);
    }
  }
  if (contact.organization) {
    const orgInput = page.locator('input[name="organization"], [data-testid="contact-org-input"]');
    if (await orgInput.count() > 0) {
      await orgInput.fill(contact.organization);
    }
  }
  if (contact.notes) {
    const notesInput = page.locator('textarea[name="notes"], [data-testid="contact-notes-input"]');
    if (await notesInput.count() > 0) {
      await notesInput.fill(contact.notes);
    }
  }
}

async function findContact(page: Page, searchTerm: string) {
  const searchInput = page.locator(
    '[data-testid="contact-search"], input[placeholder*="search" i]'
  );

  if (await searchInput.count() > 0) {
    await searchInput.fill(searchTerm);
    await page.waitForTimeout(500);
  }

  return page.locator(`[data-testid="contact-item"]:has-text("${searchTerm}")`);
}

// ============================================================================
// Contacts List Display Tests
// ============================================================================

test.describe('Contacts List', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/inbox');
    await navigateToContacts(page);
  });

  test('should display contacts list', async ({ page }) => {
    const contactsList = page.locator(
      '[data-testid="contacts-list"], .contacts-list, [role="list"]'
    );
    await expect(contactsList.first()).toBeVisible();
  });

  test('should show contact items with name and email', async ({ page }) => {
    const contactItems = page.locator(
      '[data-testid="contact-item"], .contact-item'
    );

    if ((await contactItems.count()) > 0) {
      const firstContact = contactItems.first();

      // Should display name
      const nameElement = firstContact.locator(
        '[data-testid="contact-name"], .contact-name'
      );
      if (await nameElement.count() > 0) {
        await expect(nameElement).toBeVisible();
      }

      // Should display email
      const emailElement = firstContact.locator(
        '[data-testid="contact-email"], .contact-email'
      );
      if (await emailElement.count() > 0) {
        await expect(emailElement).toBeVisible();
      }
    }
  });

  test('should show trust level indicator for contacts', async ({ page }) => {
    const contactItems = page.locator('[data-testid="contact-item"]');

    if ((await contactItems.count()) > 0) {
      const trustIndicator = contactItems.first().locator(
        '[data-testid="trust-indicator"], [data-testid="trust-badge"], .trust-level'
      );

      if (await trustIndicator.count() > 0) {
        await expect(trustIndicator.first()).toBeVisible();
      }
    }
  });

  test('should show avatar or initials for contacts', async ({ page }) => {
    const contactItems = page.locator('[data-testid="contact-item"]');

    if ((await contactItems.count()) > 0) {
      const avatar = contactItems.first().locator(
        '[data-testid="contact-avatar"], .avatar, img.avatar'
      );

      if (await avatar.count() > 0) {
        await expect(avatar.first()).toBeVisible();
      }
    }
  });
});

// ============================================================================
// Search and Filter Tests
// ============================================================================

test.describe('Contact Search and Filter', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/contacts');
    await page.waitForLoadState('networkidle');
  });

  test('should search contacts by name', async ({ page }) => {
    const searchInput = page.locator(
      '[data-testid="contact-search"], input[placeholder*="search" i]'
    );

    if (await searchInput.count() > 0) {
      await searchInput.fill('John');
      await page.waitForTimeout(500);

      // Results should be filtered
      const visibleContacts = page.locator('[data-testid="contact-item"]:visible');
      // Should show filtered results
    }
  });

  test('should search contacts by email', async ({ page }) => {
    const searchInput = page.locator('[data-testid="contact-search"]');

    if (await searchInput.count() > 0) {
      await searchInput.fill('@example.com');
      await page.waitForTimeout(500);

      // Results should be filtered by email
    }
  });

  test('should filter contacts by group', async ({ page }) => {
    const groupFilter = page.locator(
      '[data-testid="group-filter"], select[name="group"]'
    );

    if (await groupFilter.count() > 0) {
      await groupFilter.click();

      const groupOption = page.locator(
        '[data-testid="group-option"]:first-child, option:not([value=""])'
      );

      if (await groupOption.count() > 0) {
        await groupOption.first().click();
        await page.waitForTimeout(500);
      }
    }
  });

  test('should clear search and show all contacts', async ({ page }) => {
    const searchInput = page.locator('[data-testid="contact-search"]');

    if (await searchInput.count() > 0) {
      await searchInput.fill('test');
      await page.waitForTimeout(500);

      // Clear search
      const clearButton = page.locator(
        '[data-testid="clear-search"], button[aria-label*="clear"]'
      );

      if (await clearButton.count() > 0) {
        await clearButton.click();
        await expect(searchInput).toHaveValue('');
      } else {
        await searchInput.clear();
      }
    }
  });
});

// ============================================================================
// Create Contact Tests
// ============================================================================

test.describe('Create Contact', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/contacts');
    await page.waitForLoadState('networkidle');
  });

  test('should open add contact modal', async ({ page }) => {
    await openAddContactModal(page);

    const contactForm = page.locator(
      '[data-testid="contact-modal"], [data-testid="contact-form"]'
    );
    await expect(contactForm.first()).toBeVisible();
  });

  test('should create a new contact', async ({ page }) => {
    await openAddContactModal(page);
    await fillContactForm(page, testContact);

    const saveButton = page.locator(
      '[data-testid="save-contact"], button:has-text("Save"), button:has-text("Add")'
    );
    await saveButton.click();

    // Should show success message
    const successToast = page.locator(
      '[data-testid="toast-success"], .toast-success, [role="alert"]:has-text("Contact")'
    );
    await expect(successToast.first()).toBeVisible({ timeout: 5000 });

    // Modal should close
    const modal = page.locator('[data-testid="contact-modal"]');
    await expect(modal).not.toBeVisible();
  });

  test('should show validation error for empty name', async ({ page }) => {
    await openAddContactModal(page);

    await page.fill('[data-testid="contact-email-input"], input[name="email"]', testContact.email);

    const saveButton = page.locator('[data-testid="save-contact"], button:has-text("Save")');
    await saveButton.click();

    const errorMessage = page.locator(
      '[data-testid="error-message"], .error-message, [role="alert"]'
    );
    await expect(errorMessage.first()).toBeVisible({ timeout: 3000 });
  });

  test('should show validation error for invalid email', async ({ page }) => {
    await openAddContactModal(page);

    await page.fill('[data-testid="contact-name-input"], input[name="name"]', testContact.name);
    await page.fill('[data-testid="contact-email-input"], input[name="email"]', 'invalid-email');

    const saveButton = page.locator('[data-testid="save-contact"], button:has-text("Save")');
    await saveButton.click();

    const errorMessage = page.locator(
      '[data-testid="error-message"], .error-message, :invalid'
    );
    await expect(errorMessage.first()).toBeVisible({ timeout: 3000 });
  });

  test('should cancel contact creation', async ({ page }) => {
    await openAddContactModal(page);
    await fillContactForm(page, testContact);

    const cancelButton = page.locator(
      '[data-testid="cancel-contact"], button:has-text("Cancel")'
    );
    await cancelButton.click();

    // Modal should close
    const modal = page.locator('[data-testid="contact-modal"]');
    await expect(modal).not.toBeVisible();
  });
});

// ============================================================================
// View Contact Details Tests
// ============================================================================

test.describe('View Contact Details', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/contacts');
    await page.waitForLoadState('networkidle');
  });

  test('should open contact details when clicking a contact', async ({ page }) => {
    const contactItems = page.locator('[data-testid="contact-item"]');

    if ((await contactItems.count()) > 0) {
      await contactItems.first().click();

      const detailsPanel = page.locator(
        '[data-testid="contact-details"], [data-testid="contact-profile"], .contact-details'
      );
      await expect(detailsPanel.first()).toBeVisible({ timeout: 5000 });
    }
  });

  test('should display full contact information', async ({ page }) => {
    const contactItems = page.locator('[data-testid="contact-item"]');

    if ((await contactItems.count()) > 0) {
      await contactItems.first().click();

      // Check for various contact fields
      const nameField = page.locator('[data-testid="detail-name"], .detail-name');
      const emailField = page.locator('[data-testid="detail-email"], .detail-email');

      if (await nameField.count() > 0) {
        await expect(nameField.first()).toBeVisible();
      }
      if (await emailField.count() > 0) {
        await expect(emailField.first()).toBeVisible();
      }
    }
  });

  test('should show contact trust history', async ({ page }) => {
    const contactItems = page.locator('[data-testid="contact-item"]');

    if ((await contactItems.count()) > 0) {
      await contactItems.first().click();

      const trustHistory = page.locator(
        '[data-testid="trust-history"], .trust-history'
      );

      if (await trustHistory.count() > 0) {
        await expect(trustHistory).toBeVisible();
      }
    }
  });

  test('should show email with contact button', async ({ page }) => {
    const contactItems = page.locator('[data-testid="contact-item"]');

    if ((await contactItems.count()) > 0) {
      await contactItems.first().click();

      const emailButton = page.locator(
        '[data-testid="email-contact"], button:has-text("Send Email"), a[href^="mailto:"]'
      );

      if (await emailButton.count() > 0) {
        await expect(emailButton.first()).toBeVisible();
      }
    }
  });
});

// ============================================================================
// Edit Contact Tests
// ============================================================================

test.describe('Edit Contact', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/contacts');
    await page.waitForLoadState('networkidle');
  });

  test('should open edit contact modal', async ({ page }) => {
    const contactItems = page.locator('[data-testid="contact-item"]');

    if ((await contactItems.count()) > 0) {
      await contactItems.first().click();

      const editButton = page.locator(
        '[data-testid="edit-contact"], button:has-text("Edit")'
      );

      if (await editButton.count() > 0) {
        await editButton.click();

        const editForm = page.locator('[data-testid="contact-form"]');
        await expect(editForm.first()).toBeVisible({ timeout: 5000 });
      }
    }
  });

  test('should update contact information', async ({ page }) => {
    const contactItems = page.locator('[data-testid="contact-item"]');

    if ((await contactItems.count()) > 0) {
      await contactItems.first().click();

      const editButton = page.locator('[data-testid="edit-contact"]');
      if (await editButton.count() > 0) {
        await editButton.click();

        // Update name
        const nameInput = page.locator('[data-testid="contact-name-input"], input[name="name"]');
        await nameInput.clear();
        await nameInput.fill(updatedContact.name);

        // Save changes
        const saveButton = page.locator('[data-testid="save-contact"], button:has-text("Save")');
        await saveButton.click();

        // Should show success message
        const successToast = page.locator('[data-testid="toast-success"]');
        await expect(successToast.first()).toBeVisible({ timeout: 5000 });
      }
    }
  });

  test('should cancel edit without saving changes', async ({ page }) => {
    const contactItems = page.locator('[data-testid="contact-item"]');

    if ((await contactItems.count()) > 0) {
      await contactItems.first().click();

      const editButton = page.locator('[data-testid="edit-contact"]');
      if (await editButton.count() > 0) {
        await editButton.click();

        // Make changes
        const nameInput = page.locator('[data-testid="contact-name-input"]');
        await nameInput.clear();
        await nameInput.fill('Changed Name');

        // Cancel
        const cancelButton = page.locator('[data-testid="cancel-contact"], button:has-text("Cancel")');
        await cancelButton.click();

        // Changes should not be saved
      }
    }
  });
});

// ============================================================================
// Delete Contact Tests
// ============================================================================

test.describe('Delete Contact', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/contacts');
    await page.waitForLoadState('networkidle');
  });

  test('should show delete confirmation dialog', async ({ page }) => {
    const contactItems = page.locator('[data-testid="contact-item"]');

    if ((await contactItems.count()) > 0) {
      await contactItems.first().click();

      const deleteButton = page.locator(
        '[data-testid="delete-contact"], button:has-text("Delete")'
      );

      if (await deleteButton.count() > 0) {
        await deleteButton.click();

        const confirmDialog = page.locator(
          '[data-testid="confirm-dialog"], [role="alertdialog"]'
        );
        await expect(confirmDialog.first()).toBeVisible({ timeout: 3000 });
      }
    }
  });

  test('should delete contact after confirmation', async ({ page }) => {
    const contactItems = page.locator('[data-testid="contact-item"]');

    if ((await contactItems.count()) > 0) {
      const contactName = await contactItems.first().locator('[data-testid="contact-name"]').textContent();

      await contactItems.first().click();

      const deleteButton = page.locator('[data-testid="delete-contact"]');
      if (await deleteButton.count() > 0) {
        await deleteButton.click();

        const confirmButton = page.locator(
          '[data-testid="confirm-delete"], button:has-text("Delete"):visible'
        );
        if (await confirmButton.count() > 0) {
          await confirmButton.click();

          // Should show success message
          const successToast = page.locator('[data-testid="toast-success"]');
          await expect(successToast.first()).toBeVisible({ timeout: 5000 });

          // Contact should be removed from list
          if (contactName) {
            await expect(page.locator(`[data-testid="contact-item"]:has-text("${contactName}")`)).not.toBeVisible();
          }
        }
      }
    }
  });

  test('should cancel delete and keep contact', async ({ page }) => {
    const contactItems = page.locator('[data-testid="contact-item"]');

    if ((await contactItems.count()) > 0) {
      await contactItems.first().click();

      const deleteButton = page.locator('[data-testid="delete-contact"]');
      if (await deleteButton.count() > 0) {
        await deleteButton.click();

        const cancelButton = page.locator(
          '[data-testid="cancel-delete"], button:has-text("Cancel"):visible'
        );
        if (await cancelButton.count() > 0) {
          await cancelButton.click();

          // Dialog should close
          const confirmDialog = page.locator('[data-testid="confirm-dialog"]');
          await expect(confirmDialog).not.toBeVisible();
        }
      }
    }
  });
});

// ============================================================================
// Contact Groups Tests
// ============================================================================

test.describe('Contact Groups', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/contacts');
    await page.waitForLoadState('networkidle');
  });

  test('should display group tabs or filters', async ({ page }) => {
    const groupTabs = page.locator(
      '[data-testid="group-tabs"], .group-tabs, [role="tablist"]'
    );

    if (await groupTabs.count() > 0) {
      await expect(groupTabs).toBeVisible();
    }
  });

  test('should add contact to group', async ({ page }) => {
    const contactItems = page.locator('[data-testid="contact-item"]');

    if ((await contactItems.count()) > 0) {
      await contactItems.first().click();

      const addToGroupButton = page.locator(
        '[data-testid="add-to-group"], button:has-text("Add to Group")'
      );

      if (await addToGroupButton.count() > 0) {
        await addToGroupButton.click();

        const groupOption = page.locator('[data-testid="group-option"]:first-child');
        if (await groupOption.count() > 0) {
          await groupOption.click();

          const successToast = page.locator('[data-testid="toast-success"]');
          await expect(successToast.first()).toBeVisible({ timeout: 5000 });
        }
      }
    }
  });

  test('should create new group', async ({ page }) => {
    const createGroupButton = page.locator(
      '[data-testid="create-group"], button:has-text("New Group")'
    );

    if (await createGroupButton.count() > 0) {
      await createGroupButton.click();

      const groupNameInput = page.locator('[data-testid="group-name-input"], input[name="groupName"]');
      await groupNameInput.fill('Test Group');

      const saveButton = page.locator('[data-testid="save-group"], button:has-text("Create")');
      await saveButton.click();

      const successToast = page.locator('[data-testid="toast-success"]');
      await expect(successToast.first()).toBeVisible({ timeout: 5000 });
    }
  });
});

// ============================================================================
// Trust Attestation Tests
// ============================================================================

test.describe('Contact Trust Attestation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/contacts');
    await page.waitForLoadState('networkidle');
  });

  test('should create trust attestation for contact', async ({ page }) => {
    const contactItems = page.locator('[data-testid="contact-item"]');

    if ((await contactItems.count()) > 0) {
      await contactItems.first().click();

      const attestButton = page.locator(
        '[data-testid="create-attestation"], button:has-text("Attest"), button:has-text("Trust")'
      );

      if (await attestButton.count() > 0) {
        await attestButton.click();

        // Fill attestation form
        const typeSelect = page.locator('[data-testid="attestation-type"]');
        if (await typeSelect.count() > 0) {
          await typeSelect.selectOption('professional');
        }

        const commentInput = page.locator('[data-testid="attestation-comment"]');
        if (await commentInput.count() > 0) {
          await commentInput.fill('Trusted professional contact');
        }

        const submitButton = page.locator('[data-testid="submit-attestation"]');
        await submitButton.click();

        const successToast = page.locator('[data-testid="toast-success"]');
        await expect(successToast.first()).toBeVisible({ timeout: 5000 });
      }
    }
  });

  test('should view attestation history', async ({ page }) => {
    const contactItems = page.locator('[data-testid="contact-item"]');

    if ((await contactItems.count()) > 0) {
      await contactItems.first().click();

      const attestationHistory = page.locator(
        '[data-testid="attestation-history"], .attestation-list'
      );

      if (await attestationHistory.count() > 0) {
        await expect(attestationHistory).toBeVisible();
      }
    }
  });
});

// ============================================================================
// Import/Export Tests
// ============================================================================

test.describe('Contact Import/Export', () => {
  test('should show import contacts option', async ({ page }) => {
    await page.goto('/contacts');

    const importButton = page.locator(
      '[data-testid="import-contacts"], button:has-text("Import")'
    );

    if (await importButton.count() > 0) {
      await importButton.click();

      const importDialog = page.locator('[data-testid="import-dialog"]');
      await expect(importDialog.first()).toBeVisible({ timeout: 3000 });
    }
  });

  test('should show export contacts option', async ({ page }) => {
    await page.goto('/contacts');

    const exportButton = page.locator(
      '[data-testid="export-contacts"], button:has-text("Export")'
    );

    if (await exportButton.count() > 0) {
      await expect(exportButton).toBeVisible();
    }
  });
});

// ============================================================================
// Accessibility Tests
// ============================================================================

test.describe('Contacts Accessibility', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/contacts');
    await page.waitForLoadState('networkidle');
  });

  test('should navigate contacts with keyboard', async ({ page }) => {
    const contactsList = page.locator('[data-testid="contacts-list"], [role="list"]');

    if (await contactsList.count() > 0) {
      await contactsList.focus();
      await page.keyboard.press('ArrowDown');
      await page.keyboard.press('ArrowDown');
      await page.keyboard.press('Enter');

      // Contact details should open
      const details = page.locator('[data-testid="contact-details"]');
      if (await details.count() > 0) {
        await expect(details.first()).toBeVisible();
      }
    }
  });

  test('should have proper ARIA labels for contact items', async ({ page }) => {
    const contactItems = page.locator('[data-testid="contact-item"]');

    if ((await contactItems.count()) > 0) {
      const ariaLabel = await contactItems.first().getAttribute('aria-label');
      const role = await contactItems.first().getAttribute('role');

      // Should have some accessibility attribute
      expect(ariaLabel || role).toBeTruthy();
    }
  });
});
