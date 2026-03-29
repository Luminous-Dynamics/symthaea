// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Connection health tracking — richer signal than binary online/offline.
 *
 * Tracks: internet connectivity, conductor status, reconnect attempts,
 * last successful zome call, and mesh bridge status.
 */

import { writable, derived } from 'svelte/store';
import { conductorStatus, type ConductorStatusValue } from './conductor';

export interface ConnectionHealth {
  internet: boolean;
  conductor: ConductorStatusValue;
  lastSuccessfulCall: number;
  reconnectAttempts: number;
  meshBridgeDetected: boolean;
}

const _health = writable<ConnectionHealth>({
  internet: true,
  conductor: 'disconnected',
  lastSuccessfulCall: 0,
  reconnectAttempts: 0,
  meshBridgeDetected: false,
});

export const connectionHealth = { subscribe: _health.subscribe };

/** Overall quality level derived from health signals. */
export const connectionQuality = derived(_health, ($h): 'excellent' | 'good' | 'degraded' | 'offline' => {
  if (!$h.internet) return 'offline';
  if ($h.conductor === 'connected') {
    const staleMs = $h.lastSuccessfulCall > 0 ? Date.now() - $h.lastSuccessfulCall : 0;
    if (staleMs > 0 && staleMs > 120_000) return 'degraded'; // connected but no successful calls in 2min
    return 'excellent';
  }
  if ($h.conductor === 'demo' || $h.conductor === 'connecting') return 'degraded';
  return 'offline';
});

/** Human-readable status label. */
export const connectionLabel = derived(
  [_health, connectionQuality],
  ([$h, $q]): string => {
    switch ($q) {
      case 'excellent': return $h.meshBridgeDetected ? 'Connected + Mesh' : 'Connected';
      case 'good': return 'Connected';
      case 'degraded':
        if ($h.conductor === 'demo') return 'Demo Mode';
        if ($h.conductor === 'connecting') return `Connecting (attempt ${$h.reconnectAttempts})`;
        return 'Degraded';
      case 'offline': return 'Offline';
    }
  }
);

/** Color class for the quality indicator. */
export const qualityColor = derived(connectionQuality, ($q): string => {
  switch ($q) {
    case 'excellent': return 'bg-green-500';
    case 'good': return 'bg-green-400';
    case 'degraded': return 'bg-yellow-500';
    case 'offline': return 'bg-red-500';
  }
});

// ============================================================================
// Update functions (called from conductor.ts, layout, etc.)
// ============================================================================

export function setInternet(online: boolean): void {
  _health.update(h => ({ ...h, internet: online }));
}

export function setConductorStatus(status: ConductorStatusValue): void {
  _health.update(h => ({ ...h, conductor: status }));
}

export function recordSuccessfulCall(): void {
  _health.update(h => ({ ...h, lastSuccessfulCall: Date.now() }));
}

export function setReconnectAttempts(n: number): void {
  _health.update(h => ({ ...h, reconnectAttempts: n }));
}

export function setMeshBridgeDetected(detected: boolean): void {
  _health.update(h => ({ ...h, meshBridgeDetected: detected }));
}

// ============================================================================
// Auto-subscribe to conductor status changes
// ============================================================================

conductorStatus.subscribe((status) => {
  setConductorStatus(status);
  if (status === 'connected') {
    setReconnectAttempts(0);
  }
});
