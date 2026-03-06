import { AppWebsocket } from '@holochain/client';
import { browser } from '$app/environment';
import { writable, type Readable } from 'svelte/store';

type ZomeCallParams = {
  role_name: string;
  zome_name: string;
  fn_name: string;
  payload: unknown;
  cap_secret?: Uint8Array | null;
};

export type ConductorStatusValue = 'connecting' | 'connected' | 'disconnected' | 'demo';

const APP_WS_URL =
  import.meta.env.VITE_CONDUCTOR_URL ??
  import.meta.env.VITE_HOLOCHAIN_APP_WS_URL ??
  'ws://localhost:8888';

let client: AppWebsocket | null = null;
let pendingConnect: Promise<AppWebsocket | null> | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let reconnectAttempt = 0;
let reconnectStopped = false;

const MAX_BACKOFF_MS = 30_000;
const BASE_BACKOFF_MS = 1_000;

/** Reactive conductor connection status. */
export const conductorStatus = writable<ConductorStatusValue>('disconnected');

/** Read-only accessor for use in components. */
export const conductorStatus$: Readable<ConductorStatusValue> = { subscribe: conductorStatus.subscribe };

function scheduleReconnect(): void {
  if (reconnectStopped || reconnectTimer) return;
  const delay = Math.min(BASE_BACKOFF_MS * Math.pow(2, reconnectAttempt), MAX_BACKOFF_MS);
  reconnectAttempt++;
  console.log(`[Observatory] Reconnecting in ${delay}ms (attempt ${reconnectAttempt})`);
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    pendingConnect = null;
    connectToConductor();
  }, delay);
}

function clearReconnect(): void {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
}

export async function connectToConductor(): Promise<boolean> {
  if (!browser) return false;
  if (client) return true;
  if (!pendingConnect) {
    conductorStatus.set('connecting');
    pendingConnect = AppWebsocket.connect(APP_WS_URL)
      .then((connected) => {
        client = connected;
        reconnectAttempt = 0;
        clearReconnect();
        conductorStatus.set('connected');
        console.log('[Observatory] Connected to conductor at', APP_WS_URL);
        return connected;
      })
      .catch((error) => {
        console.warn('[Observatory] Conductor connection failed:', error);
        client = null;
        conductorStatus.set('demo');
        scheduleReconnect();
        return null;
      });
  }
  const result = await pendingConnect;
  if (result) return true;
  pendingConnect = null;
  return false;
}

/**
 * Stop the auto-reconnect loop. Call on app teardown.
 */
export function stopReconnect(): void {
  reconnectStopped = true;
  clearReconnect();
}

/**
 * Resume the auto-reconnect loop after stopReconnect().
 */
export function resumeReconnect(): void {
  reconnectStopped = false;
}

export function isConnected(): boolean {
  return client !== null;
}

export function getClient(): AppWebsocket {
  if (!client) {
    throw new Error('Holochain conductor is not connected');
  }
  return client;
}

export async function callZome<T>(params: ZomeCallParams): Promise<T> {
  const connected = await connectToConductor();
  if (!connected || !client) {
    throw new Error('Holochain conductor is not connected');
  }
  return client.callZome(params);
}
