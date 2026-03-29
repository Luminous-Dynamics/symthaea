// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Tests for connection-health module — derived quality, labels, colors,
 * and update functions.
 *
 * The module auto-subscribes to conductorStatus on import, so we mock
 * the conductor module to provide a controllable writable store.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { writable, get } from 'svelte/store';

// vi.mock factory is hoisted — cannot reference outer variables.
// We create the store inside the factory and re-export it via a
// shared reference that we retrieve after the mock is set up.
vi.mock('./conductor', async () => {
  const { writable: w } = await import('svelte/store');
  const store = w<'connecting' | 'connected' | 'disconnected' | 'demo'>('disconnected');
  return { conductorStatus: store, __mockStore: store };
});

// Import the mock store reference after mocking is established.
const { __mockStore: mockConductorStatus } = await import('./conductor') as any;

import {
  connectionHealth,
  connectionQuality,
  connectionLabel,
  qualityColor,
  setInternet,
  setConductorStatus,
  recordSuccessfulCall,
  setReconnectAttempts,
  setMeshBridgeDetected,
  type ConnectionHealth,
} from './connection-health';

/** Reset health state to defaults before each test. */
function resetHealth(): void {
  setInternet(true);
  setConductorStatus('disconnected');
  setReconnectAttempts(0);
  setMeshBridgeDetected(false);
  // Reset lastSuccessfulCall by setting conductor to something and back
  // — there is no direct setter, so we rely on the initial state after
  // the module-level subscription fires. We just need to ensure
  // a known conductor state.
  mockConductorStatus.set('disconnected');
}

beforeEach(() => {
  resetHealth();
});

// ---------------------------------------------------------------------------
// 1. Initial / default state
// ---------------------------------------------------------------------------
describe('initial state', () => {
  it('has internet true by default', () => {
    const h = get(connectionHealth);
    expect(h.internet).toBe(true);
  });

  it('has conductor disconnected by default', () => {
    const h = get(connectionHealth);
    expect(h.conductor).toBe('disconnected');
  });

  it('has no mesh bridge by default', () => {
    const h = get(connectionHealth);
    expect(h.meshBridgeDetected).toBe(false);
  });

  it('has zero reconnect attempts by default', () => {
    const h = get(connectionHealth);
    expect(h.reconnectAttempts).toBe(0);
  });

  it('has lastSuccessfulCall at 0 by default', () => {
    const h = get(connectionHealth);
    expect(h.lastSuccessfulCall).toBe(0);
  });

  it('connectionQuality is offline when conductor is disconnected', () => {
    expect(get(connectionQuality)).toBe('offline');
  });
});

// ---------------------------------------------------------------------------
// 2. setInternet(false) -> offline
// ---------------------------------------------------------------------------
describe('setInternet', () => {
  it('setInternet(false) makes connectionQuality offline', () => {
    setConductorStatus('connected');
    expect(get(connectionQuality)).toBe('excellent');

    setInternet(false);
    expect(get(connectionQuality)).toBe('offline');
  });

  it('setInternet(false) overrides even a connected conductor', () => {
    setConductorStatus('connected');
    setInternet(false);
    expect(get(connectionQuality)).toBe('offline');
    expect(get(connectionLabel)).toBe('Offline');
  });

  it('setInternet(true) restores quality when conductor is connected', () => {
    setConductorStatus('connected');
    setInternet(false);
    setInternet(true);
    expect(get(connectionQuality)).toBe('excellent');
  });
});

// ---------------------------------------------------------------------------
// 3. internet + conductor connected -> excellent
// ---------------------------------------------------------------------------
describe('excellent quality', () => {
  it('internet true + conductor connected = excellent', () => {
    setInternet(true);
    setConductorStatus('connected');
    expect(get(connectionQuality)).toBe('excellent');
  });

  it('label is Connected when excellent without mesh', () => {
    setInternet(true);
    setConductorStatus('connected');
    expect(get(connectionLabel)).toBe('Connected');
  });
});

// ---------------------------------------------------------------------------
// 4. internet + conductor demo -> degraded
// ---------------------------------------------------------------------------
describe('demo mode', () => {
  it('internet true + conductor demo = degraded', () => {
    setInternet(true);
    setConductorStatus('demo');
    expect(get(connectionQuality)).toBe('degraded');
  });

  it('label is Demo Mode when conductor is demo', () => {
    setInternet(true);
    setConductorStatus('demo');
    expect(get(connectionLabel)).toBe('Demo Mode');
  });
});

// ---------------------------------------------------------------------------
// 5. internet + conductor connecting -> degraded
// ---------------------------------------------------------------------------
describe('connecting state', () => {
  it('internet true + conductor connecting = degraded', () => {
    setInternet(true);
    setConductorStatus('connecting');
    expect(get(connectionQuality)).toBe('degraded');
  });

  it('label shows Connecting with attempt 0', () => {
    setInternet(true);
    setConductorStatus('connecting');
    expect(get(connectionLabel)).toBe('Connecting (attempt 0)');
  });
});

// ---------------------------------------------------------------------------
// 6. mesh bridge detection
// ---------------------------------------------------------------------------
describe('mesh bridge', () => {
  it('Connected + Mesh label when mesh bridge detected and excellent', () => {
    setInternet(true);
    setConductorStatus('connected');
    setMeshBridgeDetected(true);
    expect(get(connectionLabel)).toBe('Connected + Mesh');
  });

  it('mesh bridge does not affect quality level', () => {
    setInternet(true);
    setConductorStatus('connected');
    setMeshBridgeDetected(true);
    expect(get(connectionQuality)).toBe('excellent');
  });

  it('mesh bridge flag is false after reset', () => {
    setMeshBridgeDetected(true);
    expect(get(connectionHealth).meshBridgeDetected).toBe(true);
    setMeshBridgeDetected(false);
    expect(get(connectionHealth).meshBridgeDetected).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// 7. reconnect attempts in label
// ---------------------------------------------------------------------------
describe('reconnect attempts', () => {
  it('label includes attempt count when connecting', () => {
    setInternet(true);
    setConductorStatus('connecting');
    setReconnectAttempts(3);
    expect(get(connectionLabel)).toBe('Connecting (attempt 3)');
  });

  it('label includes attempt count 1', () => {
    setInternet(true);
    setConductorStatus('connecting');
    setReconnectAttempts(1);
    expect(get(connectionLabel)).toBe('Connecting (attempt 1)');
  });

  it('reconnect attempts are stored in health state', () => {
    setReconnectAttempts(5);
    expect(get(connectionHealth).reconnectAttempts).toBe(5);
  });
});

// ---------------------------------------------------------------------------
// 8. recordSuccessfulCall
// ---------------------------------------------------------------------------
describe('recordSuccessfulCall', () => {
  it('updates lastSuccessfulCall to a recent timestamp', () => {
    const before = Date.now();
    recordSuccessfulCall();
    const after = Date.now();
    const h = get(connectionHealth);
    expect(h.lastSuccessfulCall).toBeGreaterThanOrEqual(before);
    expect(h.lastSuccessfulCall).toBeLessThanOrEqual(after);
  });

  it('successive calls update the timestamp', () => {
    recordSuccessfulCall();
    const first = get(connectionHealth).lastSuccessfulCall;
    // Small delay not needed — Date.now() is monotonic enough
    recordSuccessfulCall();
    const second = get(connectionHealth).lastSuccessfulCall;
    expect(second).toBeGreaterThanOrEqual(first);
  });
});

// ---------------------------------------------------------------------------
// 9. qualityColor mapping
// ---------------------------------------------------------------------------
describe('qualityColor', () => {
  it('excellent -> bg-green-500', () => {
    setInternet(true);
    setConductorStatus('connected');
    expect(get(qualityColor)).toBe('bg-green-500');
  });

  it('degraded -> bg-yellow-500', () => {
    setInternet(true);
    setConductorStatus('demo');
    expect(get(qualityColor)).toBe('bg-yellow-500');
  });

  it('offline -> bg-red-500', () => {
    setInternet(false);
    expect(get(qualityColor)).toBe('bg-red-500');
  });

  it('offline via disconnected conductor -> bg-red-500', () => {
    setInternet(true);
    setConductorStatus('disconnected');
    expect(get(qualityColor)).toBe('bg-red-500');
  });
});

// ---------------------------------------------------------------------------
// 10. conductorStatus subscription auto-updates health store
// ---------------------------------------------------------------------------
describe('conductorStatus auto-subscription', () => {
  it('setting mockConductorStatus to connected updates health', () => {
    mockConductorStatus.set('connected');
    const h = get(connectionHealth);
    expect(h.conductor).toBe('connected');
  });

  it('setting mockConductorStatus to demo updates health', () => {
    mockConductorStatus.set('demo');
    expect(get(connectionHealth).conductor).toBe('demo');
  });

  it('setting mockConductorStatus to connecting updates health', () => {
    mockConductorStatus.set('connecting');
    expect(get(connectionHealth).conductor).toBe('connecting');
  });

  it('connected via subscription resets reconnectAttempts to 0', () => {
    setReconnectAttempts(5);
    expect(get(connectionHealth).reconnectAttempts).toBe(5);

    mockConductorStatus.set('connected');
    expect(get(connectionHealth).reconnectAttempts).toBe(0);
  });

  it('demo via subscription does NOT reset reconnectAttempts', () => {
    setReconnectAttempts(3);
    mockConductorStatus.set('demo');
    expect(get(connectionHealth).reconnectAttempts).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// Additional edge cases
// ---------------------------------------------------------------------------
describe('stale connection detection', () => {
  it('connected with stale lastSuccessfulCall (>2min) becomes degraded', () => {
    setInternet(true);
    setConductorStatus('connected');
    // Simulate a call that happened 3 minutes ago
    const threeMinAgo = Date.now() - 180_000;
    // We need to set lastSuccessfulCall directly — use the internal update
    // by calling recordSuccessfulCall and then manipulating time.
    // Instead, we can use setConductorStatus + the store's update path.
    // The simplest approach: record a call, then check with a mocked Date.now.
    const realNow = Date.now;
    let fakeTime = Date.now();

    // Record a call at current time
    recordSuccessfulCall();
    expect(get(connectionQuality)).toBe('excellent');

    // Advance fake time by 3 minutes
    fakeTime += 180_000;
    vi.spyOn(Date, 'now').mockReturnValue(fakeTime);

    // Force a re-derivation by toggling a health field
    setConductorStatus('connected');
    expect(get(connectionQuality)).toBe('degraded');

    vi.restoreAllMocks();
  });

  it('connected with recent lastSuccessfulCall stays excellent', () => {
    setInternet(true);
    setConductorStatus('connected');
    recordSuccessfulCall();
    expect(get(connectionQuality)).toBe('excellent');
  });
});

describe('connectionHealth is read-only (no set method)', () => {
  it('connectionHealth exposes subscribe but not set or update', () => {
    expect(connectionHealth).toHaveProperty('subscribe');
    expect(connectionHealth).not.toHaveProperty('set');
    expect(connectionHealth).not.toHaveProperty('update');
  });
});
