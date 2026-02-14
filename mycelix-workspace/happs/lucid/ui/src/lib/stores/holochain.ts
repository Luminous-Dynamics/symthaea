/**
 * Holochain Connection Store
 */

import { writable, derived } from 'svelte/store';
import { AppWebsocket } from '@holochain/client';
import { LucidClient } from '@mycelix/lucid-client';
import { isDemoMode } from '$lib/demo-data';

// Connection state
export const appWebsocket = writable<AppWebsocket | null>(null);
export const lucidClient = writable<LucidClient | null>(null);
export const connectionStatus = writable<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
export const connectionError = writable<string | null>(null);
export const demoMode = writable(false);

// Derived store for checking if connected
export const isConnected = derived(connectionStatus, ($status) => $status === 'connected');

/**
 * Connect to Holochain conductor
 */
export async function connect(url: string = 'ws://localhost:8888'): Promise<void> {
  connectionStatus.set('connecting');
  connectionError.set(null);

  try {
    const client = await AppWebsocket.connect({ url: new URL(url) });
    appWebsocket.set(client);

    // Cast to any to satisfy LucidClient constructor - the API is compatible
    const lucid = new LucidClient(client as any);
    lucidClient.set(lucid);

    connectionStatus.set('connected');
    console.log('Connected to Holochain conductor');
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    connectionError.set(message);
    connectionStatus.set('error');
    console.error('Failed to connect:', err);
  }
}

/**
 * Disconnect from Holochain conductor
 */
export async function disconnect(): Promise<void> {
  const ws = await new Promise<AppWebsocket | null>((resolve) => {
    appWebsocket.subscribe((value) => resolve(value))();
  });

  if (ws) {
    await ws.client.close();
  }

  appWebsocket.set(null);
  lucidClient.set(null);
  connectionStatus.set('disconnected');
}

/**
 * Initialize Holochain connection on app startup
 * Attempts to connect with retries
 */
export async function initializeHolochain(
  url: string = 'ws://localhost:8888',
  retries: number = 3
): Promise<boolean> {
  // Check for demo mode first
  if (isDemoMode()) {
    return initializeDemoMode();
  }

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      await connect(url);
      return true;
    } catch (err) {
      console.log(`Connection attempt ${attempt}/${retries} failed`);
      if (attempt < retries) {
        // Wait before retry
        await new Promise((resolve) => setTimeout(resolve, 1000 * attempt));
      }
    }
  }
  return false;
}

/**
 * Initialize demo mode — sets connected state without a real conductor.
 * The thoughts store detects demo mode and loads demo data instead of querying Holochain.
 */
export function initializeDemoMode(): boolean {
  console.log('LUCID running in demo mode');
  demoMode.set(true);
  connectionStatus.set('connected');
  return true;
}
