// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Negative Validation Tests
 *
 * Tests validation rejection paths to ensure invalid inputs
 * are properly rejected by the SDK.
 */

import { describe, it, expect } from 'vitest';
import {
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
} from '../../src/matl/index.js';

import {
  LocalBridge,
  createReputationQuery,
  createBroadcastEvent,
  BridgeMessageType,
} from '../../src/bridge/index.js';

import {
  krum,
  coordinateMedian,
  type GradientUpdate,
} from '../../src/fl/index.js';

// ============================================================================
// Test Utilities
// ============================================================================

function createGradientUpdate(
  participantId: string,
  gradients: Float64Array
): GradientUpdate {
  return {
    participantId,
    modelVersion: 1,
    gradients,
    metadata: {
      batchSize: 32,
      loss: 0.1,
      timestamp: Date.now(),
    },
  };
}

// ============================================================================
// MATL Validation Tests
// ============================================================================

describe('MATL Negative Validation', () => {
  describe('Reputation Creation', () => {
    it('should reject empty agent ID', () => {
      expect(() => createReputation('')).toThrow();
    });

    it('should reject whitespace-only agent ID', () => {
      expect(() => createReputation('   ')).toThrow();
    });
  });

  describe('Reputation Bounds', () => {
    it('should never return score below 0', () => {
      let rep = createReputation('agent-1');
      for (let i = 0; i < 1000; i++) {
        rep = recordNegative(rep);
      }
      expect(reputationValue(rep)).toBeGreaterThanOrEqual(0);
    });

    it('should never return score above 1', () => {
      let rep = createReputation('agent-1');
      for (let i = 0; i < 1000; i++) {
        rep = recordPositive(rep);
      }
      expect(reputationValue(rep)).toBeLessThanOrEqual(1);
    });

    it('should return finite score', () => {
      let rep = createReputation('agent-1');
      for (let i = 0; i < 100; i++) {
        rep = Math.random() > 0.5 ? recordPositive(rep) : recordNegative(rep);
      }
      expect(Number.isFinite(reputationValue(rep))).toBe(true);
    });
  });
});

// ============================================================================
// Bridge Validation Tests
// ============================================================================

describe('Bridge Negative Validation', () => {
  describe('hApp Registration', () => {
    it('should reject empty hApp ID', () => {
      const bridge = new LocalBridge();
      expect(() => bridge.registerHapp('')).toThrow();
    });

    it('should reject whitespace hApp ID', () => {
      const bridge = new LocalBridge();
      expect(() => bridge.registerHapp('   ')).toThrow();
    });
  });

  describe('Message Creation', () => {
    it('should reject query with empty source', () => {
      expect(() => createReputationQuery('', 'agent-1')).toThrow();
    });

    it('should reject query with empty agent', () => {
      expect(() => createReputationQuery('source', '')).toThrow();
    });

    it('should reject broadcast with empty source', () => {
      expect(() => createBroadcastEvent('', 'event', new Uint8Array())).toThrow();
    });

    it('should reject broadcast with empty event type', () => {
      expect(() => createBroadcastEvent('source', '', new Uint8Array())).toThrow();
    });
  });

  describe('Message Sending', () => {
    it('should reject null message', () => {
      const bridge = new LocalBridge();
      bridge.registerHapp('test');
      expect(() => bridge.send('test', null as any)).toThrow();
    });

    it('should reject undefined message', () => {
      const bridge = new LocalBridge();
      bridge.registerHapp('test');
      expect(() => bridge.send('test', undefined as any)).toThrow();
    });
  });
});

// ============================================================================
// Federated Learning Validation Tests
// ============================================================================

describe('FL Negative Validation', () => {
  describe('Gradient Arrays', () => {
    it('should reject empty gradient array for krum', () => {
      expect(() => krum([], 1)).toThrow();
    });

    it('should reject empty gradient array for coordinateMedian', () => {
      expect(() => coordinateMedian([])).toThrow();
    });

    it('should reject mismatched gradient lengths', () => {
      const updates = [
        createGradientUpdate('a', new Float64Array([1, 2, 3])),
        createGradientUpdate('b', new Float64Array([1, 2])), // Different length
      ];
      expect(() => coordinateMedian(updates)).toThrow();
    });
  });

  describe('Krum Parameters', () => {
    it('should reject numSelect < 1', () => {
      const updates = [
        createGradientUpdate('a', new Float64Array([1, 2, 3])),
        createGradientUpdate('b', new Float64Array([1, 2, 3])),
        createGradientUpdate('c', new Float64Array([1, 2, 3])),
      ];
      expect(() => krum(updates, 0)).toThrow();
    });

    it('should require at least 3 updates for Byzantine tolerance', () => {
      const updates = [
        createGradientUpdate('a', new Float64Array([1, 2, 3])),
        createGradientUpdate('b', new Float64Array([1, 2, 3])),
      ];
      expect(() => krum(updates, 1)).toThrow();
    });
  });
});

// ============================================================================
// Edge Cases
// ============================================================================

describe('Edge Cases', () => {
  it('should handle unicode in agent IDs', () => {
    // Unicode should be handled gracefully
    const rep = createReputation('agent-\u00e9\u00e0\u00fc');
    expect(reputationValue(rep)).toBeGreaterThanOrEqual(0);
  });

  it('should handle very long valid agent IDs', () => {
    const longId = 'a'.repeat(200);
    const rep = createReputation(longId);
    expect(reputationValue(rep)).toBeGreaterThanOrEqual(0);
  });

  it('should handle special characters in valid IDs', () => {
    const validIds = ['happ-with-dash', 'happ_with_underscore', 'happ123'];

    for (const id of validIds) {
      expect(() => createReputation(id)).not.toThrow();
    }
  });

  it('should handle rapid positive/negative alternation', () => {
    let rep = createReputation('agent-stress');

    for (let i = 0; i < 100; i++) {
      rep = recordPositive(rep);
      rep = recordNegative(rep);
    }

    const score = reputationValue(rep);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
    expect(Number.isFinite(score)).toBe(true);
  });
});
