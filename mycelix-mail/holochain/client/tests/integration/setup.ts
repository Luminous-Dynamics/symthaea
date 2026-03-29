// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Integration Test Setup
 *
 * Global configuration and utilities for integration tests.
 */

import { beforeAll, afterAll } from 'vitest';

// Global test configuration
const TEST_TIMEOUT = 30000; // 30 seconds for Holochain operations

// Set default timeout for all tests
beforeAll(() => {
  // Holochain operations can be slow, especially on first run
  // when zomes are being compiled
});

afterAll(() => {
  // Global cleanup
});

/**
 * Wait for a condition to be true
 */
export async function waitFor(
  condition: () => Promise<boolean> | boolean,
  timeout = 5000,
  interval = 100
): Promise<void> {
  const start = Date.now();

  while (Date.now() - start < timeout) {
    if (await condition()) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, interval));
  }

  throw new Error('Timeout waiting for condition');
}

/**
 * Generate a unique test identifier
 */
export function uniqueId(prefix = 'test'): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Generate a unique email for testing
 */
export function uniqueEmail(domain = 'test.local'): string {
  return `${uniqueId('user')}@${domain}`;
}

/**
 * Sleep for a specified duration
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Retry a function with exponential backoff
 */
export async function retry<T>(
  fn: () => Promise<T>,
  maxAttempts = 3,
  baseDelay = 1000
): Promise<T> {
  let lastError: Error | undefined;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      const delay = baseDelay * Math.pow(2, attempt);
      await sleep(delay);
    }
  }

  throw lastError;
}

/**
 * Mock agent public key generator
 */
export function mockAgentPubKey(): string {
  const bytes = new Uint8Array(39);
  crypto.getRandomValues(bytes);
  // Prefix with 'uhCAk' to simulate Holochain format
  return 'uhCAk' + Buffer.from(bytes).toString('base64').replace(/[+/=]/g, 'x');
}

/**
 * Create test fixtures
 */
export function createTestFixtures() {
  return {
    emails: [
      {
        to: [{ address: 'recipient1@test.local', name: 'Recipient One' }],
        cc: [],
        bcc: [],
        subject: 'Test Email 1',
        body: 'This is test email 1.',
        attachments: [],
        encrypted: false,
      },
      {
        to: [{ address: 'recipient2@test.local', name: 'Recipient Two' }],
        cc: [{ address: 'cc@test.local' }],
        bcc: [],
        subject: 'Test Email 2',
        body: 'This is test email 2 with CC.',
        attachments: [],
        encrypted: true,
      },
    ],
    contacts: [
      {
        email: 'contact1@test.local',
        name: 'Contact One',
        organization: 'Org One',
        tags: ['work'],
        favorite: false,
        blocked: false,
      },
      {
        email: 'contact2@test.local',
        name: 'Contact Two',
        organization: 'Org Two',
        tags: ['personal'],
        favorite: true,
        blocked: false,
      },
    ],
    groups: [
      { name: 'Work', color: '#0066cc' },
      { name: 'Personal', color: '#00cc66' },
      { name: 'Project X', color: '#cc6600' },
    ],
  };
}

export default {
  waitFor,
  uniqueId,
  uniqueEmail,
  sleep,
  retry,
  mockAgentPubKey,
  createTestFixtures,
};
