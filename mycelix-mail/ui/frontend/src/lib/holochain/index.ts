// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Holochain Integration for UI Frontend
 *
 * Connects the Mycelix Mail UI to the Holochain client
 */

import { AppWebsocket, AdminWebsocket, AppClient } from '@holochain/client';
import {
  MycelixMailClient,
  createMycelixMailClient,
  MycelixClient,
  SignalHub,
} from '@mycelix/holochain-client';

// Re-export client types
export type { MycelixMailClient } from '@mycelix/holochain-client';
export * from '@mycelix/holochain-client/types';

// Connection configuration
export interface HolochainConfig {
  appWebsocketUrl?: string;
  adminWebsocketUrl?: string;
  appId?: string;
  roleName?: string;
  timeout?: number;
}

const DEFAULT_CONFIG: Required<HolochainConfig> = {
  appWebsocketUrl: 'ws://localhost:8888',
  adminWebsocketUrl: 'ws://localhost:8889',
  appId: 'mycelix-mail',
  roleName: 'mycelix_mail',
  timeout: 30000,
};

// Singleton client instance
let holochainClient: MycelixMailClient | null = null;
let mycelixClient: MycelixClient | null = null;
let appWebsocket: AppWebsocket | null = null;
let connectionPromise: Promise<MycelixMailClient> | null = null;

/**
 * Connection state
 */
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error';

let connectionState: ConnectionState = 'disconnected';
let connectionError: Error | null = null;
const connectionListeners: Set<(state: ConnectionState, error?: Error) => void> = new Set();

function setConnectionState(state: ConnectionState, error?: Error) {
  connectionState = state;
  connectionError = error || null;
  connectionListeners.forEach((listener) => listener(state, error));
}

/**
 * Subscribe to connection state changes
 */
export function onConnectionStateChange(
  listener: (state: ConnectionState, error?: Error) => void
): () => void {
  connectionListeners.add(listener);
  // Immediately call with current state
  listener(connectionState, connectionError || undefined);
  return () => connectionListeners.delete(listener);
}

/**
 * Get current connection state
 */
export function getConnectionState(): { state: ConnectionState; error: Error | null } {
  return { state: connectionState, error: connectionError };
}

/**
 * Connect to Holochain
 */
export async function connectToHolochain(
  config: HolochainConfig = {}
): Promise<MycelixMailClient> {
  // Return existing connection if available
  if (holochainClient && connectionState === 'connected') {
    return holochainClient;
  }

  // Return pending connection if in progress
  if (connectionPromise) {
    return connectionPromise;
  }

  const fullConfig = { ...DEFAULT_CONFIG, ...config };

  connectionPromise = (async () => {
    try {
      setConnectionState('connecting');
      console.log('[Holochain] Connecting to', fullConfig.appWebsocketUrl);

      // Connect to app websocket
      appWebsocket = await AppWebsocket.connect(fullConfig.appWebsocketUrl, fullConfig.timeout);

      // Create the basic mail client
      holochainClient = createMycelixMailClient(appWebsocket as AppClient, fullConfig.roleName);

      // Initialize the client
      await holochainClient.initialize();

      // Create the enhanced Mycelix client with all services
      mycelixClient = await MycelixClient.create({
        appId: fullConfig.appId,
        roleName: fullConfig.roleName,
        websocketUrl: fullConfig.appWebsocketUrl,
      });

      // Setup signal handling
      setupSignalHandlers(holochainClient.signals);

      // Setup reconnection logic
      setupReconnection(fullConfig);

      setConnectionState('connected');
      console.log('[Holochain] Connected successfully');

      return holochainClient;
    } catch (error) {
      console.error('[Holochain] Connection failed:', error);
      setConnectionState('error', error as Error);
      connectionPromise = null;
      throw error;
    }
  })();

  return connectionPromise;
}

/**
 * Disconnect from Holochain
 */
export async function disconnectFromHolochain(): Promise<void> {
  if (appWebsocket) {
    await appWebsocket.client.close();
    appWebsocket = null;
  }

  if (mycelixClient) {
    await mycelixClient.shutdown();
    mycelixClient = null;
  }

  holochainClient = null;
  connectionPromise = null;
  setConnectionState('disconnected');
  console.log('[Holochain] Disconnected');
}

/**
 * Get the Holochain client (throws if not connected)
 */
export function getHolochainClient(): MycelixMailClient {
  if (!holochainClient) {
    throw new Error('Holochain client not connected. Call connectToHolochain() first.');
  }
  return holochainClient;
}

/**
 * Get the enhanced Mycelix client (throws if not connected)
 */
export function getMycelixClient(): MycelixClient {
  if (!mycelixClient) {
    throw new Error('Mycelix client not connected. Call connectToHolochain() first.');
  }
  return mycelixClient;
}

/**
 * Check if connected
 */
export function isConnected(): boolean {
  return connectionState === 'connected' && holochainClient !== null;
}

// ==========================================
// Signal Handlers
// ==========================================

function setupSignalHandlers(signals: SignalHub): void {
  // Forward all signals to global event system
  signals.onAny((type, payload) => {
    window.dispatchEvent(
      new CustomEvent('holochain-signal', {
        detail: { type, payload },
      })
    );
  });

  // Handle specific signals
  signals.on('EmailReceived', (signal) => {
    console.log('[Holochain] New email received:', signal.email_hash);
    showNotification('New Email', signal.subject);
  });

  signals.on('TrustAttestationCreated', (signal) => {
    console.log('[Holochain] Trust attestation created:', signal.attestation_hash);
  });

  signals.on('SyncCompleted', (signal) => {
    console.log('[Holochain] Sync completed:', signal.operations_synced, 'operations');
  });
}

function showNotification(title: string, body: string): void {
  if ('Notification' in window && Notification.permission === 'granted') {
    new Notification(title, { body });
  }
}

// ==========================================
// Reconnection Logic
// ==========================================

let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY_MS = 3000;

function setupReconnection(config: Required<HolochainConfig>): void {
  if (!appWebsocket) return;

  // Handle WebSocket close
  appWebsocket.client.socket.addEventListener('close', () => {
    if (connectionState === 'connected') {
      console.log('[Holochain] Connection lost, attempting reconnection...');
      setConnectionState('connecting');
      attemptReconnection(config);
    }
  });

  // Handle WebSocket error
  appWebsocket.client.socket.addEventListener('error', (error) => {
    console.error('[Holochain] WebSocket error:', error);
  });
}

async function attemptReconnection(config: Required<HolochainConfig>): Promise<void> {
  if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
    console.error('[Holochain] Max reconnection attempts reached');
    setConnectionState('error', new Error('Failed to reconnect after multiple attempts'));
    return;
  }

  reconnectAttempts++;
  console.log(`[Holochain] Reconnection attempt ${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS}`);

  await new Promise((resolve) => setTimeout(resolve, RECONNECT_DELAY_MS * reconnectAttempts));

  try {
    // Clear existing connection
    holochainClient = null;
    connectionPromise = null;

    // Attempt to reconnect
    await connectToHolochain(config);
    reconnectAttempts = 0;
  } catch (error) {
    console.error('[Holochain] Reconnection failed:', error);
    attemptReconnection(config);
  }
}

// ==========================================
// React Integration Helpers
// ==========================================

/**
 * Initialize Holochain on app startup
 */
export async function initializeHolochain(config?: HolochainConfig): Promise<void> {
  try {
    await connectToHolochain(config);

    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
      await Notification.requestPermission();
    }

    // Handle online/offline events
    window.addEventListener('online', async () => {
      if (holochainClient) {
        await holochainClient.goOnline();
      }
    });

    window.addEventListener('offline', async () => {
      if (holochainClient) {
        await holochainClient.goOffline();
      }
    });
  } catch (error) {
    console.error('[Holochain] Initialization failed:', error);
    throw error;
  }
}

/**
 * Cleanup on app unmount
 */
export async function cleanupHolochain(): Promise<void> {
  await disconnectFromHolochain();
}

// Export default connection function
export default connectToHolochain;
