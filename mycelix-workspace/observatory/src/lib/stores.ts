/**
 * Observatory Svelte Stores
 *
 * Provides reactive state for ecosystem status, alerts, and connection mode.
 * Bridges the EcosystemService (simulation) with the conductor (live data).
 */

import { writable, derived } from 'svelte/store';
import { browser } from '$app/environment';
import {
  conductorStatus,
  connectToConductor,
  isConnected,
  stopReconnect,
  type ConductorStatusValue,
} from './conductor';
import {
  getEcosystemService,
  type EcosystemStatus,
  type ByzantineAlert,
} from './ecosystem';

// Re-export conductor status for convenience
export { conductorStatus } from './conductor';
export type { ConductorStatusValue } from './conductor';

// ============================================================================
// Core Stores
// ============================================================================

const ecosystemService = getEcosystemService();

/** Current ecosystem status (live or simulated). */
export const ecosystemStatus = writable<EcosystemStatus>(ecosystemService.getStatus());

/** Byzantine detection alerts. */
export const byzantineAlerts = writable<ByzantineAlert[]>(ecosystemService.getAlerts());

/** Whether the observatory is showing live conductor data. */
export const isLiveMode = derived(conductorStatus, ($status: ConductorStatusValue) => $status === 'connected');

// ============================================================================
// Initialization
// ============================================================================

let simulationInterval: ReturnType<typeof setInterval> | null = null;
let unsubEcosystem: (() => void) | null = null;

/**
 * Initialize stores: attempt conductor connection, fall back to simulation.
 * Returns a cleanup function for onDestroy.
 */
export async function initializeStores(): Promise<() => void> {
  if (!browser) return () => {};

  // Subscribe ecosystem service updates to stores
  unsubEcosystem = ecosystemService.subscribe((status) => {
    ecosystemStatus.set(status);
    byzantineAlerts.set(ecosystemService.getAlerts());
  });

  // Attempt live connection
  const connected = await connectToConductor();

  if (!connected) {
    // Start simulation for demo mode
    startSimulation();
  }

  // Watch for connection status changes to toggle simulation
  const unsubStatus = conductorStatus.subscribe((status: ConductorStatusValue) => {
    if (status === 'connected') {
      stopSimulation();
    } else if (status === 'demo' && !simulationInterval) {
      startSimulation();
    }
  });

  return () => {
    stopSimulation();
    stopReconnect();
    if (unsubEcosystem) {
      unsubEcosystem();
      unsubEcosystem = null;
    }
    unsubStatus();
  };
}

function startSimulation(): void {
  if (simulationInterval) return;
  // Initial simulation tick
  ecosystemService.simulateMetricsUpdate();
  simulationInterval = setInterval(() => {
    ecosystemService.simulateMetricsUpdate();
  }, 3000);
}

function stopSimulation(): void {
  if (simulationInterval) {
    clearInterval(simulationInterval);
    simulationInterval = null;
  }
}
