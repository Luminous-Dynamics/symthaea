// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Test Utilities
 *
 * Common utilities and render helpers for testing components
 */

import React, { ReactElement, ReactNode } from 'react';
import { render, RenderOptions, RenderResult } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { HolochainTestProvider } from './mocks/HolochainTestProvider';
import type { Email, Contact, TrustNode, TrustEdge, MailboxStatus } from '@/lib/holochain/hooks';
import type { MockConnectionState } from './mocks/holochain';

// ============================================================================
// Types
// ============================================================================

interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  // Router options
  initialEntries?: string[];
  // Holochain mock options
  initialEmails?: Email[];
  initialContacts?: Contact[];
  initialTrustNodes?: TrustNode[];
  initialTrustEdges?: TrustEdge[];
  initialMailboxStatus?: MailboxStatus;
  initialConnectionState?: MockConnectionState;
  initialConnectionError?: Error;
  sendEmailShouldFail?: boolean;
  sendEmailFailReason?: string;
  loadingShouldNeverResolve?: boolean;
  // Query client options
  queryClient?: QueryClient;
}

interface CustomRenderResult extends RenderResult {
  user: ReturnType<typeof userEvent.setup>;
}

// ============================================================================
// Provider Wrapper
// ============================================================================

function createWrapper(options: CustomRenderOptions) {
  const queryClient = options.queryClient || createTestQueryClient();

  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <HolochainTestProvider
            initialEmails={options.initialEmails}
            initialContacts={options.initialContacts}
            initialTrustNodes={options.initialTrustNodes}
            initialTrustEdges={options.initialTrustEdges}
            initialMailboxStatus={options.initialMailboxStatus}
            initialConnectionState={options.initialConnectionState}
            initialConnectionError={options.initialConnectionError}
            sendEmailShouldFail={options.sendEmailShouldFail}
            sendEmailFailReason={options.sendEmailFailReason}
            loadingShouldNeverResolve={options.loadingShouldNeverResolve}
          >
            {children}
          </HolochainTestProvider>
        </BrowserRouter>
      </QueryClientProvider>
    );
  };
}

// ============================================================================
// Custom Render Function
// ============================================================================

/**
 * Custom render function that wraps components with necessary providers
 *
 * @example
 * const { user } = renderWithProviders(<MyComponent />);
 * await user.click(screen.getByRole('button'));
 */
export function renderWithProviders(
  ui: ReactElement,
  options: CustomRenderOptions = {}
): CustomRenderResult {
  const user = userEvent.setup();

  const renderResult = render(ui, {
    wrapper: createWrapper(options),
    ...options,
  });

  return {
    ...renderResult,
    user,
  };
}

// ============================================================================
// Query Client for Tests
// ============================================================================

/**
 * Creates a QueryClient configured for testing
 * - No retries
 * - No caching between tests
 * - Errors thrown to test handlers
 */
export function createTestQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
        staleTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
    logger: {
      log: console.log,
      warn: console.warn,
      error: () => {}, // Suppress errors in test output
    },
  });
}

// ============================================================================
// Async Utilities
// ============================================================================

/**
 * Wait for a specified amount of time
 */
export function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Wait for loading state to resolve
 */
export async function waitForLoadingToFinish(container: HTMLElement): Promise<void> {
  const loadingIndicators = container.querySelectorAll('[data-loading="true"], .loading, .skeleton');

  for (const indicator of Array.from(loadingIndicators)) {
    await new Promise<void>((resolve) => {
      const observer = new MutationObserver((mutations, obs) => {
        if (!indicator.parentNode) {
          obs.disconnect();
          resolve();
        }
      });

      observer.observe(indicator.parentNode!, { childList: true });

      // Timeout fallback
      setTimeout(() => {
        observer.disconnect();
        resolve();
      }, 5000);
    });
  }
}

// ============================================================================
// Mock Data Generators
// ============================================================================

/**
 * Generate a mock email
 */
export function createMockEmail(overrides: Partial<Email> = {}): Email {
  const timestamp = Date.now();
  return {
    hash: `uhCkk${Math.random().toString(36).substr(2, 9)}`,
    sender: 'sender@example.com',
    recipient: 'recipient@example.com',
    subject: 'Test Email Subject',
    body: 'This is a test email body.',
    timestamp,
    isRead: false,
    isStarred: false,
    trustLevel: 0.5,
    attachments: [],
    labels: [],
    ...overrides,
  };
}

/**
 * Generate multiple mock emails
 */
export function createMockEmails(count: number, baseOverrides: Partial<Email> = {}): Email[] {
  return Array.from({ length: count }, (_, i) =>
    createMockEmail({
      ...baseOverrides,
      hash: `uhCkk${Math.random().toString(36).substr(2, 9)}${i}`,
      subject: `${baseOverrides.subject || 'Test Email'} ${i + 1}`,
      timestamp: Date.now() - i * 3600000, // Each email 1 hour older
    })
  );
}

/**
 * Generate a mock contact
 */
export function createMockContact(overrides: Partial<Contact> = {}): Contact {
  return {
    id: `contact-${Math.random().toString(36).substr(2, 9)}`,
    name: 'Test Contact',
    email: 'contact@example.com',
    agentPubKey: `uhCAk${Math.random().toString(36).substr(2, 9)}`,
    trustLevel: 0.5,
    groups: [],
    ...overrides,
  };
}

/**
 * Generate multiple mock contacts
 */
export function createMockContacts(count: number, baseOverrides: Partial<Contact> = {}): Contact[] {
  return Array.from({ length: count }, (_, i) =>
    createMockContact({
      ...baseOverrides,
      id: `contact-${i}`,
      name: `${baseOverrides.name || 'Contact'} ${i + 1}`,
      email: `contact${i + 1}@example.com`,
    })
  );
}

/**
 * Generate a mock trust node
 */
export function createMockTrustNode(overrides: Partial<TrustNode> = {}): TrustNode {
  return {
    id: `uhCAk${Math.random().toString(36).substr(2, 9)}`,
    name: 'Test Node',
    email: 'node@example.com',
    trustLevel: 0.5,
    attestationCount: 0,
    isMe: false,
    ...overrides,
  };
}

/**
 * Generate a mock trust edge
 */
export function createMockTrustEdge(overrides: Partial<TrustEdge> = {}): TrustEdge {
  return {
    source: `uhCAk${Math.random().toString(36).substr(2, 9)}`,
    target: `uhCAk${Math.random().toString(36).substr(2, 9)}`,
    trustLevel: 0.5,
    createdAt: Date.now(),
    ...overrides,
  };
}

// ============================================================================
// Assertion Helpers
// ============================================================================

/**
 * Assert that an element has the expected text content
 */
export function expectTextContent(element: HTMLElement, text: string): void {
  expect(element.textContent).toContain(text);
}

/**
 * Assert that an element is visible
 */
export function expectVisible(element: HTMLElement): void {
  expect(element).toBeVisible();
}

/**
 * Assert that an element is not in the document
 */
export function expectNotInDocument(element: HTMLElement | null): void {
  expect(element).not.toBeInTheDocument();
}

// ============================================================================
// Event Helpers
// ============================================================================

/**
 * Dispatch a custom holochain signal event
 */
export function dispatchHolochainSignal(type: string, payload: Record<string, unknown>): void {
  window.dispatchEvent(
    new CustomEvent('holochain-signal', {
      detail: { type, payload },
    })
  );
}

/**
 * Simulate receiving a new email signal
 */
export function simulateNewEmailSignal(email: Partial<Email> = {}): void {
  dispatchHolochainSignal('EmailReceived', {
    email_hash: email.hash || `uhCkk${Date.now()}`,
    subject: email.subject || 'New Email',
    sender: email.sender || 'sender@example.com',
  });
}

/**
 * Simulate trust attestation created signal
 */
export function simulateTrustAttestationSignal(target: string, trustLevel: number): void {
  dispatchHolochainSignal('TrustAttestationCreated', {
    attestation_hash: `uhCkkatt${Date.now()}`,
    target,
    trust_level: trustLevel,
  });
}

// ============================================================================
// Re-exports
// ============================================================================

export * from '@testing-library/react';
export { default as userEvent } from '@testing-library/user-event';
