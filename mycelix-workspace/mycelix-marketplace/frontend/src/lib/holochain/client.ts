// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Holochain Client Wrapper
 *
 * Centralized Holochain connection and zome call functions
 *
 * Usage:
 * import { initHolochainClient, getListing, createTransaction } from '$lib/holochain/client';
 *
 * const client = await initHolochainClient();
 * const listing = await getListing(client, listingHash);
 */

import { AppWebsocket, type AppClient, type AppInfo } from '@holochain/client';
import { holochain } from '$lib/stores';
import { get } from 'svelte/store';
import { handleSignal } from './signals';

/**
 * Default WebSocket URL
 */
const FALLBACK_WS_URL = 'ws://localhost:8888';

/**
 * Resolved WebSocket URL from env or fallback
 */
const DEFAULT_WS_URL = import.meta.env?.VITE_HOLOCHAIN_URL || FALLBACK_WS_URL;

/**
 * Maximum reconnection attempts
 */
const MAX_RECONNECT_ATTEMPTS = 5;

/**
 * Reconnection delay (milliseconds)
 */
const RECONNECT_DELAY = 1000;

/**
 * App info cache
 */
let appInfoCache: AppInfo | null = null;

/**
 * Initialize Holochain client with retry logic
 *
 * @param url - WebSocket URL (default: ws://localhost:8888)
 * @returns AppClient instance
 */
export async function initHolochainClient(url: string = DEFAULT_WS_URL): Promise<AppClient> {
  const currentState = get(holochain);
  const resolvedUrl = url || DEFAULT_WS_URL;

  // Return existing client if already connected
  if (currentState.client && currentState.status === 'connected') {
    return currentState.client;
  }

  holochain.setStatus('connecting');
  holochain.setUrl(resolvedUrl);

  let lastError: Error | null = null;

  for (let attempt = 0; attempt < MAX_RECONNECT_ATTEMPTS; attempt++) {
    try {
      holochain.incrementAttempts();

      // Connect to Holochain conductor
      // AppWebsocket.connect expects an options object or string URL
      const client = await AppWebsocket.connect({
        url: resolvedUrl as any,
        signalHandler: handleSignal as any,
      } as any);

      // Get app info and cache it
      appInfoCache = await client.appInfo();

      // Store client in global state
      holochain.setClient(client);

      console.log('Holochain client connected successfully');
      return client;
    } catch (error) {
      lastError = error as Error;
      console.error(`Connection attempt ${attempt + 1} failed:`, error);

      if (attempt < MAX_RECONNECT_ATTEMPTS - 1) {
        // Wait before retrying
        await new Promise((resolve) => setTimeout(resolve, RECONNECT_DELAY * (attempt + 1)));
      }
    }
  }

  // All attempts failed
  const errorMessage = `Failed to connect to Holochain at ${resolvedUrl} after ${MAX_RECONNECT_ATTEMPTS} attempts: ${lastError?.message}. Check your conductor and VITE_HOLOCHAIN_URL.`;
  holochain.setError(errorMessage);
  throw new Error(errorMessage);
}

/**
 * Get cached app info
 */
export function getAppInfo(): AppInfo | null {
  return appInfoCache;
}

/**
 * Get cell ID for a zome
 *
 * @param roleName - Role name (default: first role)
 * @returns Cell ID [dna_hash, agent_pub_key]
 */
export function getCellId(roleName?: string): [Uint8Array, Uint8Array] {
  if (!appInfoCache) {
    throw new Error('App info not loaded. Call initHolochainClient first.');
  }

  const cellData = roleName
    ? appInfoCache.cell_info[roleName]?.[0]
    : appInfoCache.cell_info[Object.keys(appInfoCache.cell_info)[0]][0];

  if (!cellData || !('provisioned' in cellData)) {
    throw new Error('No provisioned cell found');
  }

  return cellData.provisioned.cell_id;
}

/**
 * Generic zome call wrapper
 *
 * @param client - Holochain client
 * @param zomeName - Zome name
 * @param fnName - Function name
 * @param payload - Function payload
 * @returns Function result
 */
export async function callZome<T = any>(
  client: AppClient,
  zomeName: string,
  fnName: string,
  payload: any
): Promise<T> {
  const cellId = getCellId();

  try {
    const result = await client.callZome({
      cap_secret: null,
      cell_id: cellId,
      zome_name: zomeName,
      fn_name: fnName,
      payload,
    });

    return result as T;
  } catch (error) {
    console.error(`Zome call failed: ${zomeName}.${fnName}`, error);
    throw error;
  }
}

/**
 * Disconnect Holochain client
 */
export function disconnectHolochainClient(): void {
  const currentState = get(holochain);

  if (currentState.client) {
    // Note: AppClient in @holochain/client v0.17 doesn't have a direct close method
    // The WebSocket connection is managed internally
    console.log('Holochain client disconnecting');
  }

  holochain.reset();
  appInfoCache = null;
}

/**
 * Check if client is connected
 */
export function isClientConnected(): boolean {
  const currentState = get(holochain);
  return currentState.status === 'connected' && currentState.client !== null;
}

/**
 * Get current client or throw error
 */
export function getClient(): AppClient {
  const currentState = get(holochain);

  if (!currentState.client || currentState.status !== 'connected') {
    throw new Error('Holochain client not connected. Call initHolochainClient first.');
  }

  return currentState.client;
}
