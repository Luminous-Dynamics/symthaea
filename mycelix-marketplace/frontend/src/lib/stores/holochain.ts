/**
 * Holochain Connection Store
 *
 * Centralized state management for Holochain client connection
 */

import { writable, derived } from 'svelte/store';
import type { AppClient } from '@holochain/client';

/**
 * Connection status
 */
export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

/**
 * Holochain connection state
 */
export interface HolochainState {
  /** WebSocket client instance */
  client: AppClient | null;

  /** Connection status */
  status: ConnectionStatus;

  /** Error message if status is 'error' */
  error: string | null;

  /** Connection attempts */
  attempts: number;

  /** Last connection attempt timestamp */
  lastAttempt: number | null;

  /** WebSocket URL */
  url: string;
}

/**
 * Initial state
 */
const initialState: HolochainState = {
  client: null,
  status: 'disconnected',
  error: null,
  attempts: 0,
  lastAttempt: null,
  url: 'ws://localhost:8888',
};

/**
 * Create Holochain connection store
 */
function createHolochainStore() {
  const { subscribe, set, update } = writable<HolochainState>(initialState);

  return {
    subscribe,

    /**
     * Set connection status
     */
    setStatus: (status: ConnectionStatus) => {
      update((state) => ({
        ...state,
        status,
        error: status === 'error' ? state.error : null,
      }));
    },

    /**
     * Set client instance
     */
    setClient: (client: AppClient) => {
      update((state) => ({
        ...state,
        client,
        status: 'connected',
        error: null,
      }));
    },

    /**
     * Set error
     */
    setError: (error: string) => {
      update((state) => ({
        ...state,
        error,
        status: 'error',
      }));
    },

    /**
     * Increment connection attempts
     */
    incrementAttempts: () => {
      update((state) => ({
        ...state,
        attempts: state.attempts + 1,
        lastAttempt: Date.now(),
      }));
    },

    /**
     * Reset connection state
     */
    reset: () => {
      set(initialState);
    },

    /**
     * Set WebSocket URL
     */
    setUrl: (url: string) => {
      update((state) => ({
        ...state,
        url,
      }));
    },

    /**
     * Disconnect client
     */
    disconnect: () => {
      update((state) => {
        if (state.client) {
          // Note: AppClient in @holochain/client v0.17 doesn't have a direct close method
          // The WebSocket connection is managed internally
          console.log('Holochain client disconnecting');
        }

        return {
          ...initialState,
          url: state.url, // Preserve URL
        };
      });
    },
  };
}

/**
 * Holochain connection store
 */
export const holochain = createHolochainStore();

/**
 * Derived store: is connected
 */
export const isConnected = derived(holochain, ($holochain) => $holochain.status === 'connected');

/**
 * Derived store: is connecting
 */
export const isConnecting = derived(holochain, ($holochain) => $holochain.status === 'connecting');

/**
 * Derived store: has error
 */
export const hasConnectionError = derived(holochain, ($holochain) => $holochain.status === 'error');

/**
 * Derived store: client instance
 */
export const client = derived(holochain, ($holochain) => $holochain.client);

/**
 * Derived store: connection error message
 */
export const connectionError = derived(holochain, ($holochain) => $holochain.error);
