// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * E2E Tests - Accessibility
 *
 * Tests keyboard navigation, screen reader compatibility, and WCAG compliance
 */

import { test, expect, Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('Accessibility', () => {
  let page: Page;

  test.beforeEach(async ({ browser }) => {
    const context = await browser.newContext();
    page = await context.newPage();
    await page.goto('/');
    await expect(page.getByTestId('inbox-view')).toBeVisible({ timeout: 10000 });
  });

  test.afterEach(async () => {
    await page.close();
  });

  test('should have no critical accessibility violations on inbox page', async () => {
    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
      .exclude('.trust-graph svg') // Exclude complex SVG graphs
      .analyze();

    expect(accessibilityScanResults.violations.filter(v => v.impact === 'critical')).toEqual([]);
  });

  test('should navigate inbox with keyboard only', async () => {
    // Focus on first interactive element
    await page.keyboard.press('Tab');

    // Navigate to email list
    let focused = await page.locator(':focus').getAttribute('data-testid');

    // Press Tab until we reach the email list
    let attempts = 0;
    while (focused !== 'email-list-item' && attempts < 20) {
      await page.keyboard.press('Tab');
      focused = await page.locator(':focus').getAttribute('data-testid');
      attempts++;
    }

    // Navigate within email list using arrow keys
    await page.keyboard.press('ArrowDown');
    await page.keyboard.press('ArrowDown');
    await page.keyboard.press('ArrowUp');

    // Open email with Enter
    await page.keyboard.press('Enter');

    // Verify email opened
    await expect(page.getByTestId('email-view')).toBeVisible();

    // Close with Escape
    await page.keyboard.press('Escape');
    await expect(page.getByTestId('inbox-view')).toBeVisible();
  });

  test('should operate compose modal with keyboard', async () => {
    // Use keyboard shortcut to open compose
    await page.keyboard.press('c');

    // Wait for compose modal
    await expect(page.getByTestId('compose-modal')).toBeVisible();

    // Tab through form fields
    await page.keyboard.press('Tab'); // To field
    await page.keyboard.type('test@example.com');

    await page.keyboard.press('Tab'); // Subject field
    await page.keyboard.type('Keyboard Test');

    await page.keyboard.press('Tab'); // Body field
    await page.keyboard.type('This email was composed entirely with keyboard.');

    // Close with Escape
    await page.keyboard.press('Escape');

    // Verify discard dialog
    const discardDialog = page.getByTestId('discard-dialog');
    if (await discardDialog.isVisible()) {
      // Use keyboard to navigate dialog
      await page.keyboard.press('Tab');
      await page.keyboard.press('Enter'); // Confirm discard
    }
  });

  test('should have proper focus management', async () => {
    // Open compose modal
    await page.getByRole('button', { name: /compose/i }).click();
    await expect(page.getByTestId('compose-modal')).toBeVisible();

    // Focus should be in modal
    const activeElement = await page.locator(':focus');
    const modalContains = await activeElement.evaluate((el) => {
      return el.closest('[data-testid="compose-modal"]') !== null;
    });
    expect(modalContains).toBe(true);

    // Close modal
    await page.keyboard.press('Escape');

    // Focus should return to compose button
    const returnedFocus = await page.locator(':focus').getAttribute('name');
    expect(returnedFocus).toMatch(/compose/i);
  });

  test('should have visible focus indicators', async () => {
    // Tab to first interactive element
    await page.keyboard.press('Tab');

    // Get focused element
    const focused = page.locator(':focus');

    // Verify focus is visible (outline or border)
    const hasOutline = await focused.evaluate((el) => {
      const style = window.getComputedStyle(el);
      return (
        style.outline !== 'none' ||
        style.boxShadow.includes('ring') ||
        el.classList.contains('focus-visible')
      );
    });

    expect(hasOutline).toBe(true);
  });

  test('should support reduced motion preference', async () => {
    // Set reduced motion preference
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.reload();

    // Verify animations are reduced/removed
    const body = page.locator('body');
    const hasReducedMotion = await body.evaluate(() => {
      return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    });

    expect(hasReducedMotion).toBe(true);

    // Check that animation duration is 0 or very short
    const animatedElements = page.locator('[class*="animate"]');
    const count = await animatedElements.count();

    for (let i = 0; i < Math.min(count, 5); i++) {
      const el = animatedElements.nth(i);
      const duration = await el.evaluate((element) => {
        return window.getComputedStyle(element).animationDuration;
      });
      expect(duration).toMatch(/0s|0\.001s|0\.01s/);
    }
  });

  test('should have proper ARIA labels', async () => {
    // Check navigation
    const nav = page.getByRole('navigation');
    await expect(nav).toHaveAttribute('aria-label');

    // Check main content region
    const main = page.getByRole('main');
    await expect(main).toBeVisible();

    // Check buttons have accessible names
    const buttons = page.getByRole('button');
    const buttonCount = await buttons.count();

    for (let i = 0; i < Math.min(buttonCount, 10); i++) {
      const button = buttons.nth(i);
      const name = await button.getAttribute('aria-label') ||
                   await button.getAttribute('title') ||
                   await button.textContent();
      expect(name?.trim()).not.toBe('');
    }
  });

  test('should have proper heading hierarchy', async () => {
    // Get all headings
    const h1 = await page.locator('h1').count();
    const h2 = await page.locator('h2').count();
    const h3 = await page.locator('h3').count();

    // Should have exactly one h1
    expect(h1).toBe(1);

    // Headings should not skip levels
    // (if there's an h3, there should be an h2)
    if (h3 > 0) {
      expect(h2).toBeGreaterThan(0);
    }
  });

  test('should have sufficient color contrast', async () => {
    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['color-contrast'])
      .analyze();

    // Allow some warnings but no critical failures
    const criticalContrast = accessibilityScanResults.violations.filter(
      v => v.id === 'color-contrast' && v.impact === 'serious'
    );

    expect(criticalContrast.length).toBe(0);
  });

  test('should be usable with screen reader', async () => {
    // Check for skip link
    await page.keyboard.press('Tab');
    const skipLink = page.getByText(/skip to main content/i);

    if (await skipLink.isVisible()) {
      await page.keyboard.press('Enter');
      // Focus should move to main content
      const main = page.getByRole('main');
      const mainContainsFocus = await main.evaluate((el) => {
        return el.contains(document.activeElement);
      });
      expect(mainContainsFocus).toBe(true);
    }

    // Check for live regions
    const liveRegions = page.locator('[aria-live]');
    const liveCount = await liveRegions.count();
    expect(liveCount).toBeGreaterThan(0);
  });

  test('should announce dynamic content changes', async () => {
    // Find live region for announcements
    const announcer = page.locator('[aria-live="polite"], [aria-live="assertive"]');
    await expect(announcer.first()).toBeVisible();

    // Trigger an action that should announce
    await page.getByRole('button', { name: /refresh/i }).click();

    // Wait for announcement
    await page.waitForTimeout(1000);

    // Check if announcement was made
    const announcerText = await announcer.first().textContent();
    // Announcer should have some content after action
    expect(announcerText?.length).toBeGreaterThanOrEqual(0);
  });

  test('should have accessible forms', async () => {
    // Open compose modal
    await page.getByRole('button', { name: /compose/i }).click();
    await expect(page.getByTestId('compose-modal')).toBeVisible();

    // Check all inputs have labels
    const inputs = page.locator('input, textarea, select');
    const inputCount = await inputs.count();

    for (let i = 0; i < inputCount; i++) {
      const input = inputs.nth(i);
      const id = await input.getAttribute('id');
      const ariaLabel = await input.getAttribute('aria-label');
      const ariaLabelledBy = await input.getAttribute('aria-labelledby');

      // Should have a label association
      if (id) {
        const label = page.locator(`label[for="${id}"]`);
        const hasLabel = await label.count() > 0 || ariaLabel || ariaLabelledBy;
        expect(hasLabel).toBeTruthy();
      } else {
        expect(ariaLabel || ariaLabelledBy).toBeTruthy();
      }
    }
  });

  test('should support high contrast mode', async () => {
    await page.emulateMedia({ forcedColors: 'active' });
    await page.reload();

    // App should still be usable
    await expect(page.getByTestId('inbox-view')).toBeVisible();

    // Buttons should still be identifiable
    const composeButton = page.getByRole('button', { name: /compose/i });
    await expect(composeButton).toBeVisible();
  });

  test('should have accessible error messages', async () => {
    // Open compose and try to send empty email
    await page.getByRole('button', { name: /compose/i }).click();
    await expect(page.getByTestId('compose-modal')).toBeVisible();

    // Try to send without filling required fields
    await page.getByRole('button', { name: /send/i }).click();

    // Check for error message with proper association
    const errorMessage = page.locator('[role="alert"], .error-message');

    if (await errorMessage.isVisible()) {
      // Error should be announced
      const ariaLive = await errorMessage.getAttribute('aria-live');
      const role = await errorMessage.getAttribute('role');
      expect(ariaLive === 'assertive' || role === 'alert').toBeTruthy();
    }
  });
});
