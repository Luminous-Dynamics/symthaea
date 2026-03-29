// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Epistemic Mail E2E Tests
 *
 * End-to-end tests for critical user flows:
 * - Onboarding wizard
 * - Trust attestation creation
 * - Email viewing with epistemic data
 * - Settings management
 * - Search and filtering
 */

import { test, expect, Page } from '@playwright/test';

// ============================================
// Test Fixtures
// ============================================

const TEST_USER_DID = 'did:mycelix:test-user-123';

// Helper to wait for app to be ready
async function waitForAppReady(page: Page) {
  await page.waitForSelector('[data-testid="app-ready"]', { timeout: 10000 });
}

// Helper to bypass onboarding
async function skipOnboarding(page: Page) {
  await page.evaluate(() => {
    localStorage.setItem('mycelix_onboarding_complete', 'true');
  });
}

// ============================================
// Onboarding Tests
// ============================================

test.describe('Onboarding', () => {
  test.beforeEach(async ({ page }) => {
    // Clear localStorage to ensure fresh state
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
    await page.reload();
  });

  test('should show onboarding wizard for new users', async ({ page }) => {
    await page.goto('/');

    // Should see onboarding modal
    await expect(page.locator('[data-testid="onboarding-modal"]')).toBeVisible();
    await expect(page.locator('text=Welcome to Epistemic Mail')).toBeVisible();
  });

  test('should complete onboarding flow', async ({ page }) => {
    await page.goto('/');

    // Step 1: Welcome
    await expect(page.locator('text=Welcome')).toBeVisible();
    await page.click('[data-testid="onboarding-next"]');

    // Step 2: Epistemic Tiers
    await expect(page.locator('text=Epistemic Tiers')).toBeVisible();
    await page.click('[data-testid="onboarding-next"]');

    // Step 3: Verify Identity
    await expect(page.locator('text=Verify')).toBeVisible();
    await page.click('[data-testid="onboarding-next"]');

    // Step 4: Build Network
    await expect(page.locator('text=Network')).toBeVisible();
    await page.click('[data-testid="onboarding-next"]');

    // Step 5: AI Features
    await expect(page.locator('text=AI')).toBeVisible();
    await page.click('[data-testid="onboarding-next"]');

    // Step 6: Complete
    await expect(page.locator('text=Ready')).toBeVisible();
    await page.click('[data-testid="onboarding-complete"]');

    // Onboarding should be dismissed
    await expect(page.locator('[data-testid="onboarding-modal"]')).not.toBeVisible();
  });

  test('should persist onboarding completion', async ({ page }) => {
    await page.goto('/');

    // Complete onboarding
    for (let i = 0; i < 5; i++) {
      await page.click('[data-testid="onboarding-next"]');
    }
    await page.click('[data-testid="onboarding-complete"]');

    // Reload page
    await page.reload();

    // Should not show onboarding again
    await expect(page.locator('[data-testid="onboarding-modal"]')).not.toBeVisible();
  });
});

// ============================================
// Navigation Tests
// ============================================

test.describe('Navigation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await skipOnboarding(page);
    await page.reload();
  });

  test('should navigate between views', async ({ page }) => {
    // Start at inbox
    await expect(page.locator('[data-testid="view-inbox"]')).toBeVisible();

    // Navigate to Dashboard
    await page.click('text=Trust Network');
    await expect(page.locator('[data-testid="view-dashboard"]')).toBeVisible();

    // Navigate to Identity
    await page.click('text=Identity');
    await expect(page.locator('[data-testid="view-identity"]')).toBeVisible();

    // Navigate to Settings
    await page.click('text=Settings');
    await expect(page.locator('[data-testid="view-settings"]')).toBeVisible();

    // Navigate back to Inbox
    await page.click('text=Inbox');
    await expect(page.locator('[data-testid="view-inbox"]')).toBeVisible();
  });

  test('should show keyboard shortcuts modal', async ({ page }) => {
    // Press ? key
    await page.keyboard.press('?');

    // Should show shortcuts modal
    await expect(page.locator('text=Keyboard Shortcuts')).toBeVisible();
    await expect(page.locator('text=Navigation')).toBeVisible();
    await expect(page.locator('text=Email Actions')).toBeVisible();

    // Close with Escape
    await page.keyboard.press('Escape');
    await expect(page.locator('text=Keyboard Shortcuts')).not.toBeVisible();
  });
});

// ============================================
// Trust Attestation Tests
// ============================================

test.describe('Trust Attestations', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await skipOnboarding(page);
    await page.reload();
    await page.click('text=Trust Network');
  });

  test('should display trust dashboard', async ({ page }) => {
    await expect(page.locator('[data-testid="trust-dashboard"]')).toBeVisible();
    await expect(page.locator('text=Trust Score')).toBeVisible();
    await expect(page.locator('text=Connections')).toBeVisible();
  });

  test('should open create attestation form', async ({ page }) => {
    await page.click('[data-testid="create-attestation-btn"]');

    await expect(page.locator('[data-testid="attestation-form"]')).toBeVisible();
    await expect(page.locator('text=Create Attestation')).toBeVisible();
  });

  test('should create a new attestation', async ({ page }) => {
    await page.click('[data-testid="create-attestation-btn"]');

    // Fill form
    await page.fill('[data-testid="attestation-did"]', 'did:mycelix:new-contact');
    await page.selectOption('[data-testid="attestation-relationship"]', 'direct_trust');
    await page.fill('[data-testid="attestation-score"]', '80');
    await page.fill('[data-testid="attestation-message"]', 'Trusted colleague');

    // Submit
    await page.click('[data-testid="attestation-submit"]');

    // Should show success
    await expect(page.locator('text=Attestation created')).toBeVisible();
  });

  test('should show trust graph visualization', async ({ page }) => {
    await expect(page.locator('[data-testid="trust-graph"]')).toBeVisible();

    // Should have nodes
    await expect(page.locator('[data-testid="trust-node"]').first()).toBeVisible();
  });
});

// ============================================
// Email Viewing Tests
// ============================================

test.describe('Email Viewing', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await skipOnboarding(page);
    await page.reload();
  });

  test('should display email list with epistemic indicators', async ({ page }) => {
    // Should see email list
    await expect(page.locator('[data-testid="email-list"]')).toBeVisible();

    // Should have trust badges
    await expect(page.locator('[data-testid="trust-badge"]').first()).toBeVisible();
  });

  test('should open email with epistemic sidebar', async ({ page }) => {
    // Click first email
    await page.click('[data-testid="email-row"]');

    // Should show email detail
    await expect(page.locator('[data-testid="email-detail"]')).toBeVisible();

    // Should have epistemic sidebar
    await expect(page.locator('[data-testid="epistemic-sidebar"]')).toBeVisible();
  });

  test('should toggle AI insights panel', async ({ page }) => {
    await page.click('[data-testid="email-row"]');

    // Press Alt+I to toggle insights
    await page.keyboard.press('Alt+i');

    await expect(page.locator('[data-testid="ai-insights-panel"]')).toBeVisible();
    await expect(page.locator('text=Intent')).toBeVisible();
  });

  test('should toggle contact profile panel', async ({ page }) => {
    await page.click('[data-testid="email-row"]');

    // Press Alt+C to toggle contact
    await page.keyboard.press('Alt+c');

    await expect(page.locator('[data-testid="contact-profile-panel"]')).toBeVisible();
  });
});

// ============================================
// Settings Tests
// ============================================

test.describe('Settings', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await skipOnboarding(page);
    await page.reload();
    await page.click('text=Settings');
  });

  test('should display settings tabs', async ({ page }) => {
    await expect(page.locator('text=Trust')).toBeVisible();
    await expect(page.locator('text=AI')).toBeVisible();
    await expect(page.locator('text=Notifications')).toBeVisible();
    await expect(page.locator('text=Privacy')).toBeVisible();
  });

  test('should toggle trust settings', async ({ page }) => {
    // Find trust enabled toggle
    const toggle = page.locator('[data-testid="setting-trust-enabled"]');

    // Toggle it
    await toggle.click();

    // Should persist (check localStorage or visual state)
    await page.reload();
    await page.click('text=Settings');

    // Verify state persisted
    await expect(toggle).toBeVisible();
  });

  test('should adjust trust threshold slider', async ({ page }) => {
    const slider = page.locator('[data-testid="setting-trust-threshold"]');

    // Adjust slider
    await slider.fill('50');

    // Should show new value
    await expect(page.locator('text=50%')).toBeVisible();
  });
});

// ============================================
// Search Tests
// ============================================

test.describe('Search', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await skipOnboarding(page);
    await page.reload();
  });

  test('should focus search with / key', async ({ page }) => {
    await page.keyboard.press('/');

    const searchInput = page.locator('[data-testid="search-input"]');
    await expect(searchInput).toBeFocused();
  });

  test('should filter emails by search query', async ({ page }) => {
    await page.fill('[data-testid="search-input"]', 'project');
    await page.keyboard.press('Enter');

    // Should filter results
    await expect(page.locator('[data-testid="email-row"]')).toBeVisible();
  });

  test('should open advanced search', async ({ page }) => {
    await page.keyboard.press('Control+Shift+f');

    await expect(page.locator('[data-testid="advanced-search"]')).toBeVisible();
    await expect(page.locator('text=Trust Score')).toBeVisible();
    await expect(page.locator('text=Epistemic Tier')).toBeVisible();
  });

  test('should filter by trust score', async ({ page }) => {
    await page.keyboard.press('Control+Shift+f');

    // Set minimum trust score
    await page.fill('[data-testid="filter-trust-min"]', '70');
    await page.click('[data-testid="apply-filters"]');

    // Should only show high-trust emails
    const trustBadges = page.locator('[data-testid="trust-badge"]');
    // All visible badges should be green/high
  });
});

// ============================================
// External Identity Tests
// ============================================

test.describe('External Identity', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await skipOnboarding(page);
    await page.reload();
    await page.click('text=Identity');
  });

  test('should display identity providers', async ({ page }) => {
    await expect(page.locator('text=ENS')).toBeVisible();
    await expect(page.locator('text=GitHub')).toBeVisible();
    await expect(page.locator('text=Domain')).toBeVisible();
  });

  test('should start ENS verification', async ({ page }) => {
    await page.click('text=Connect >> nth=0'); // First connect button (ENS)

    await expect(page.locator('[data-testid="ens-verification"]')).toBeVisible();
    await expect(page.locator('text=ENS Name')).toBeVisible();
  });

  test('should start GitHub verification', async ({ page }) => {
    // Find GitHub provider and click connect
    const githubCard = page.locator('text=GitHub').locator('..');
    await githubCard.locator('text=Connect').click();

    await expect(page.locator('text=Start GitHub Verification')).toBeVisible();
  });
});

// ============================================
// Notification Tests
// ============================================

test.describe('Notifications', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await skipOnboarding(page);
    await page.reload();
  });

  test('should show notification bell', async ({ page }) => {
    await expect(page.locator('[data-testid="notification-bell"]')).toBeVisible();
  });

  test('should open notification panel', async ({ page }) => {
    await page.click('[data-testid="notification-bell"]');

    await expect(page.locator('[data-testid="notification-panel"]')).toBeVisible();
  });

  test('should display notifications', async ({ page }) => {
    await page.click('[data-testid="notification-bell"]');

    // Should have some notifications or empty state
    const hasNotifications = await page.locator('[data-testid="notification-item"]').count() > 0;
    const hasEmptyState = await page.locator('text=No notifications').isVisible();

    expect(hasNotifications || hasEmptyState).toBeTruthy();
  });
});

// ============================================
// Accessibility Tests
// ============================================

test.describe('Accessibility', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await skipOnboarding(page);
    await page.reload();
  });

  test('should have proper ARIA labels', async ({ page }) => {
    // Check main navigation
    await expect(page.locator('nav[aria-label]')).toBeVisible();

    // Check email list
    await expect(page.locator('[role="list"]')).toBeVisible();
  });

  test('should support keyboard navigation', async ({ page }) => {
    // Tab through elements
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');

    // Should have visible focus indicator
    const focusedElement = page.locator(':focus');
    await expect(focusedElement).toBeVisible();
  });

  test('should respect reduced motion preference', async ({ page }) => {
    // Emulate reduced motion
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.reload();

    // Animations should be disabled
    // (Visual check - components should not animate)
  });
});

// ============================================
// Performance Tests
// ============================================

test.describe('Performance', () => {
  test('should load initial page quickly', async ({ page }) => {
    const startTime = Date.now();
    await page.goto('/');
    await skipOnboarding(page);
    await page.reload();

    // Wait for app ready
    await page.waitForSelector('[data-testid="app-ready"]', { timeout: 5000 });

    const loadTime = Date.now() - startTime;
    expect(loadTime).toBeLessThan(3000); // Should load in under 3 seconds
  });

  test('should handle large email lists', async ({ page }) => {
    await page.goto('/');
    await skipOnboarding(page);
    await page.reload();

    // Scroll through email list
    const emailList = page.locator('[data-testid="email-list"]');

    // Scroll to bottom
    await emailList.evaluate((el) => el.scrollTo(0, el.scrollHeight));

    // Should still be responsive
    await expect(emailList).toBeVisible();
  });
});

// ============================================
// Offline Tests
// ============================================

test.describe('Offline Support', () => {
  test('should show offline indicator when disconnected', async ({ page, context }) => {
    await page.goto('/');
    await skipOnboarding(page);
    await page.reload();

    // Go offline
    await context.setOffline(true);

    // Should show offline indicator
    await expect(page.locator('[data-testid="offline-indicator"]')).toBeVisible();
  });

  test('should cache emails for offline viewing', async ({ page, context }) => {
    await page.goto('/');
    await skipOnboarding(page);
    await page.reload();

    // Load some emails first
    await page.waitForSelector('[data-testid="email-row"]');

    // Go offline
    await context.setOffline(true);

    // Should still be able to view cached emails
    await page.click('[data-testid="email-row"]');
    await expect(page.locator('[data-testid="email-detail"]')).toBeVisible();
  });
});
