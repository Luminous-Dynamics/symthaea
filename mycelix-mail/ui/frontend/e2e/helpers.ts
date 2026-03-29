// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * E2E Test Helpers
 */

import { BrowserContext } from '@playwright/test';

// Mock email data
export const mockEmails = [
  {
    id: 'email-1',
    subject: 'Trusted Sender Email',
    bodyPreview: 'This is a preview of the email content...',
    from: { email: 'trusted@company.com', name: 'Trusted Sender' },
    isRead: true,
    isStarred: false,
    isEncrypted: true,
    trustScore: 0.95,
    labels: ['important'],
    receivedAt: new Date().toISOString(),
    hasAttachments: false,
  },
  {
    id: 'email-2',
    subject: 'Unread Subject',
    bodyPreview: 'Another email preview...',
    from: { email: 'known@example.com', name: 'Known Contact' },
    isRead: false,
    isStarred: true,
    isEncrypted: false,
    trustScore: 0.65,
    labels: [],
    receivedAt: new Date(Date.now() - 3600000).toISOString(),
    hasAttachments: true,
  },
  {
    id: 'email-3',
    subject: 'Unknown Sender Message',
    bodyPreview: 'Be careful with this email...',
    from: { email: 'unknown@suspicious.net', name: 'Unknown Sender' },
    isRead: false,
    isStarred: false,
    isEncrypted: false,
    trustScore: 0.15,
    labels: ['spam'],
    receivedAt: new Date(Date.now() - 86400000).toISOString(),
    hasAttachments: false,
  },
];

// Mock contacts
export const mockContacts = [
  {
    id: 'contact-1',
    email: 'john@example.com',
    displayName: 'John Doe',
    trustScore: 0.9,
    interactionCount: 150,
  },
  {
    id: 'contact-2',
    email: 'jane@example.com',
    displayName: 'Jane Smith',
    trustScore: 0.75,
    interactionCount: 45,
  },
  {
    id: 'contact-3',
    email: 'trusted@example.com',
    displayName: 'Trusted Contact',
    trustScore: 0.95,
    interactionCount: 200,
  },
];

// Setup authenticated user
export async function setupAuthenticatedUser(context: BrowserContext): Promise<void> {
  await context.addCookies([
    {
      name: 'auth_token',
      value: 'mock-jwt-token-for-testing',
      domain: 'localhost',
      path: '/',
      httpOnly: true,
      secure: false,
      sameSite: 'Lax',
    },
  ]);

  // Add user to localStorage
  await context.addInitScript(() => {
    localStorage.setItem(
      'user',
      JSON.stringify({
        id: 'user-1',
        email: 'testuser@example.com',
        name: 'Test User',
      })
    );
  });
}

// Setup mock API responses
export async function setupMockAPI(context: BrowserContext): Promise<void> {
  await context.route('**/api/graphql', async (route) => {
    const body = JSON.parse(route.request().postData() || '{}');
    const query = body.query || '';

    // Route based on query content
    if (query.includes('GetEmails')) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          data: {
            emails: {
              emails: mockEmails,
              nextCursor: null,
              hasMore: false,
              total: mockEmails.length,
            },
          },
        }),
      });
    } else if (query.includes('GetContacts')) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          data: { contacts: mockContacts },
        }),
      });
    } else if (query.includes('GetTrustNetwork')) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          data: {
            trustNetwork: {
              nodes: mockContacts.map((c) => ({
                id: c.id,
                label: c.displayName,
                email: c.email,
                trustScore: c.trustScore,
              })),
              edges: [
                { from: 'contact-1', to: 'contact-2', weight: 0.8, context: 'professional' },
              ],
            },
          },
        }),
      });
    } else {
      await route.continue();
    }
  });
}

// Wait for network idle
export async function waitForNetworkIdle(
  page: import('@playwright/test').Page,
  timeout = 5000
): Promise<void> {
  await page.waitForLoadState('networkidle', { timeout });
}

// Accessibility check helper
export async function checkA11y(page: import('@playwright/test').Page): Promise<void> {
  // Basic accessibility checks
  const violations: string[] = [];

  // Check for images without alt text
  const imagesWithoutAlt = await page.locator('img:not([alt])').count();
  if (imagesWithoutAlt > 0) {
    violations.push(`${imagesWithoutAlt} images without alt text`);
  }

  // Check for buttons without accessible names
  const buttonsWithoutName = await page.locator('button:not([aria-label]):not(:has-text(*))').count();
  if (buttonsWithoutName > 0) {
    violations.push(`${buttonsWithoutName} buttons without accessible name`);
  }

  // Check for form inputs without labels
  const inputsWithoutLabels = await page.locator('input:not([aria-label]):not([id])').count();
  if (inputsWithoutLabels > 0) {
    violations.push(`${inputsWithoutLabels} inputs without labels`);
  }

  if (violations.length > 0) {
    console.warn('Accessibility violations:', violations);
  }
}

// Take screenshot for visual regression
export async function captureScreenshot(
  page: import('@playwright/test').Page,
  name: string
): Promise<void> {
  await page.screenshot({
    path: `screenshots/${name}.png`,
    fullPage: true,
  });
}

// Mock service worker for offline testing
export async function enableOfflineMode(context: BrowserContext): Promise<void> {
  await context.setOffline(true);
}

export async function disableOfflineMode(context: BrowserContext): Promise<void> {
  await context.setOffline(false);
}
