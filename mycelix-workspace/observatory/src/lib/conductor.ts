import { AppWebsocket } from '@holochain/client';
import { browser } from '$app/environment';
import { writable, type Readable } from 'svelte/store';
import { flush } from './offline-queue';
import { toasts } from './toast';

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
        // Auto-flush any queued offline submissions
        flushOfflineQueue().catch((err) =>
          console.warn('[Observatory] Auto-flush failed:', err),
        );
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

// ============================================================================
// Offline Queue Flush — domain/action → zome call mapping
// ============================================================================

/**
 * Map queued domain+action to the correct zome call parameters.
 * Each entry corresponds to an enqueue() call in resilience-client.ts.
 */
function resolveZomeCall(domain: string, action: string, payload: unknown): ZomeCallParams {
  const p = payload as Record<string, unknown>;

  switch (`${domain}/${action}`) {
    case 'tend/recordExchange':
      return {
        role_name: 'finance',
        zome_name: 'tend',
        fn_name: 'record_exchange',
        payload: {
          receiver_did: p.receiverDid,
          hours: p.hours,
          service_description: p.serviceDescription,
          service_category: p.serviceCategory,
          dao_did: p.daoDid,
        },
      };
    case 'food/registerPlot':
      return {
        role_name: 'commons_care',
        zome_name: 'food_production',
        fn_name: 'register_plot',
        payload: { name: p.name, location: p.location, area_sqm: p.areaSqm, plot_type: p.plotType, steward: null },
      };
    case 'food/recordHarvest':
      return {
        role_name: 'commons_care',
        zome_name: 'food_production',
        fn_name: 'record_harvest',
        payload: { crop_hash: p.plotId, quantity_kg: p.quantityKg, quality: 'good', notes: p.notes },
      };
    case 'food/logResourceInput':
      return {
        role_name: 'commons_care',
        zome_name: 'food_production',
        fn_name: 'log_resource_input',
        payload: { plot_hash: p.plotId, input_type: p.inputType, quantity_kg: p.quantityKg, notes: p.notes },
      };
    case 'mutual-aid/createServiceOffer':
      return {
        role_name: 'commons_care',
        zome_name: 'mutualaid_timebank',
        fn_name: 'create_service_offer',
        payload: { title: p.title, description: p.description, category: p.category, hours_available: p.hoursAvailable },
      };
    case 'mutual-aid/createServiceRequest':
      return {
        role_name: 'commons_care',
        zome_name: 'mutualaid_timebank',
        fn_name: 'create_service_request',
        payload: { title: p.title, description: p.description, category: p.category, urgency: p.urgency, hours_needed: p.hoursNeeded },
      };
    case 'emergency/createChannel':
      return {
        role_name: 'civic',
        zome_name: 'emergency_comms',
        fn_name: 'create_channel',
        payload: { name: p.name, description: p.description },
      };
    case 'emergency/sendMessage':
      return {
        role_name: 'civic',
        zome_name: 'emergency_comms',
        fn_name: 'send_message',
        payload: { channel_id: p.channelId, content: p.content, priority: p.priority },
      };
    case 'water/registerWaterSystem':
      return {
        role_name: 'commons_care',
        zome_name: 'water',
        fn_name: 'register_water_system',
        payload: p.system,
      };
    case 'water/submitWaterReading':
      return {
        role_name: 'commons_care',
        zome_name: 'water',
        fn_name: 'submit_water_reading',
        payload: p.reading,
      };
    case 'supplies/addInventoryItem':
      return {
        role_name: 'commons_care',
        zome_name: 'supplies',
        fn_name: 'add_inventory_item',
        payload: p.item,
      };
    case 'supplies/updateStockLevel':
      return {
        role_name: 'commons_care',
        zome_name: 'supplies',
        fn_name: 'update_stock_level',
        payload: { item_hash: p.itemHash, quantity_change: p.quantityChange, reason: p.reason },
      };
    default:
      throw new Error(`Unknown queued action: ${domain}/${action}`);
  }
}

/**
 * Execute a single queued submission by mapping it to a zome call.
 * Used as the executor callback for flush().
 */
export async function flushExecutor(domain: string, action: string, payload: unknown): Promise<void> {
  const params = resolveZomeCall(domain, action, payload);
  await callZome(params);
}

/**
 * Flush the offline queue and show toast notifications with results.
 * Called automatically on reconnect and available for manual "Sync Now".
 */
export async function flushOfflineQueue(): Promise<void> {
  const results = await flush(flushExecutor);
  const total = results.succeeded + results.failed;
  if (total === 0) return;

  if (results.succeeded > 0) {
    toasts.success(`${results.succeeded} item${results.succeeded !== 1 ? 's' : ''} synced`);
  }
  if (results.failed > 0) {
    toasts.error(`${results.failed} item${results.failed !== 1 ? 's' : ''} failed to sync`);
  }
}
