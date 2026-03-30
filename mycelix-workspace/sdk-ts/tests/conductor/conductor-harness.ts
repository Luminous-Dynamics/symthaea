// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Holochain Conductor Test Harness
 *
 * Provides utilities for testing against a real Holochain conductor.
 * Requires a running conductor with the Mycelix hApp installed.
 *
 * Uses dynamic imports to avoid libsodium ESM issues when conductor is not available.
 *
 * Environment variables:
 * - HOLOCHAIN_APP_URL: WebSocket URL for app interface (default: ws://localhost:8888)
 * - HOLOCHAIN_ADMIN_URL: WebSocket URL for admin interface (default: ws://localhost:8889)
 * - MYCELIX_APP_ID: Installed app ID (default: mycelix_ecosystem)
 */

export interface ConductorConfig {
  appUrl: string;
  adminUrl: string;
  appId: string;
}

export interface TestContext {
  appClient: any; // AppClient type from @holochain/client (dynamically loaded)
  adminClient: any; // AdminWebsocket type from @holochain/client (dynamically loaded)
  config: ConductorConfig;
}

/**
 * Get conductor configuration from environment
 */
export function getConductorConfig(): ConductorConfig {
  return {
    appUrl: process.env.HOLOCHAIN_APP_URL || 'ws://localhost:8888',
    adminUrl: process.env.HOLOCHAIN_ADMIN_URL || 'ws://localhost:8889',
    appId: process.env.MYCELIX_APP_ID || 'mycelix_ecosystem',
  };
}

/**
 * Check if conductor is available
 * Uses dynamic import to avoid loading @holochain/client until needed
 */
export async function isConductorAvailable(config: ConductorConfig): Promise<boolean> {
  try {
    // Dynamic import to avoid libsodium ESM issues when not running conductor tests
    const { AdminWebsocket } = await import('@holochain/client');
    const admin = await AdminWebsocket.connect({ url: new URL(config.adminUrl) });
    await admin.listApps({});
    await admin.client.close();
    return true;
  } catch {
    return false;
  }
}

/**
 * Setup test context with connected clients
 * Uses dynamic import to avoid loading @holochain/client until needed
 */
export async function setupTestContext(): Promise<TestContext> {
  const config = getConductorConfig();

  // Dynamic import to avoid libsodium ESM issues
  const { AdminWebsocket, AppWebsocket } = await import('@holochain/client');

  const adminClient = await AdminWebsocket.connect({ url: new URL(config.adminUrl) });
  const appClient = await AppWebsocket.connect({ url: new URL(config.appUrl) });

  return {
    appClient,
    adminClient,
    config,
  };
}

/**
 * Teardown test context
 */
export async function teardownTestContext(ctx: TestContext): Promise<void> {
  await ctx.appClient.client.close();
  await ctx.adminClient.client.close();
}

/**
 * Skip test if conductor is not available
 */
export function skipIfNoConductor(available: boolean): void {
  if (!available) {
    console.log('⚠️  Skipping conductor tests - no conductor available');
    console.log('   Start conductor with: ./mycelix-workspace/scripts/start-conductor.sh');
  }
}

/**
 * Retry helper for flaky network operations
 */
export async function retry<T>(
  fn: () => Promise<T>,
  maxRetries = 3,
  delayMs = 1000
): Promise<T> {
  let lastError: Error | undefined;

  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      if (i < maxRetries - 1) {
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }
    }
  }

  throw lastError;
}

/**
 * Generate a unique agent key for testing
 */
export function generateTestAgentId(): string {
  return `test_agent_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Wait for DHT sync (useful after writes)
 */
export async function waitForSync(ms = 500): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms));
}
