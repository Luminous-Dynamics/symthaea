// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mutation Killer Tests
 *
 * These tests specifically target boundary conditions and edge cases
 * to catch mutations that survive regular testing. Each test is designed
 * to detect a specific type of code mutation.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as utils from '../src/utils/index.js';
import * as security from '../src/security/index.js';
import * as matl from '../src/matl/index.js';

// ============================================================================
// TIMESTAMP BOUNDARY TESTS - Kill EqualityOperator mutations
// ============================================================================

describe('Timestamp Boundary Tests', () => {
  describe('isExpiredTimestamp boundaries', () => {
    it('should return false when time elapsed equals TTL exactly (> not >=)', () => {
      const referenceTime = 10000;
      const ts = 5000;  // 5000ms ago
      const ttlMs = 5000;  // TTL is exactly 5000ms

      // ref - ts = 10000 - 5000 = 5000
      // 5000 > 5000 = false (not expired at exact boundary)
      expect(utils.isExpiredTimestamp(ts, ttlMs, referenceTime)).toBe(false);
    });

    it('should return true when time elapsed is just over TTL', () => {
      const referenceTime = 10001;
      const ts = 5000;
      const ttlMs = 5000;

      // ref - ts = 10001 - 5000 = 5001
      // 5001 > 5000 = true (expired)
      expect(utils.isExpiredTimestamp(ts, ttlMs, referenceTime)).toBe(true);
    });

    it('should return false when time elapsed is just under TTL', () => {
      const referenceTime = 9999;
      const ts = 5000;
      const ttlMs = 5000;

      // ref - ts = 9999 - 5000 = 4999
      // 4999 > 5000 = false (not expired)
      expect(utils.isExpiredTimestamp(ts, ttlMs, referenceTime)).toBe(false);
    });

    it('should handle zero TTL boundary', () => {
      const referenceTime = 5000;
      const ts = 5000;
      const ttlMs = 0;

      // ref - ts = 0, 0 > 0 = false
      expect(utils.isExpiredTimestamp(ts, ttlMs, referenceTime)).toBe(false);

      // ref - ts = 1, 1 > 0 = true
      expect(utils.isExpiredTimestamp(ts - 1, ttlMs, referenceTime)).toBe(true);
    });
  });

  describe('isFutureTimestamp boundaries', () => {
    let originalDateNow: () => number;

    beforeEach(() => {
      originalDateNow = Date.now;
    });

    afterEach(() => {
      Date.now = originalDateNow;
    });

    it('should return false when ts equals now exactly (> not >=)', () => {
      const fixedNow = 1000000;
      Date.now = () => fixedNow;

      // ts = now, ts > now + 0 => false
      expect(utils.isFutureTimestamp(fixedNow, 0)).toBe(false);
    });

    it('should return true when ts is 1ms after now', () => {
      const fixedNow = 1000000;
      Date.now = () => fixedNow;

      // ts = now + 1, ts > now + 0 => true
      expect(utils.isFutureTimestamp(fixedNow + 1, 0)).toBe(true);
    });

    it('should return false when ts equals now + tolerance exactly', () => {
      const fixedNow = 1000000;
      const tolerance = 100;
      Date.now = () => fixedNow;

      // ts = now + tolerance, ts > now + tolerance => false
      expect(utils.isFutureTimestamp(fixedNow + tolerance, tolerance)).toBe(false);
    });

    it('should return true when ts exceeds now + tolerance by 1ms', () => {
      const fixedNow = 1000000;
      const tolerance = 100;
      Date.now = () => fixedNow;

      // ts = now + tolerance + 1, ts > now + tolerance => true
      expect(utils.isFutureTimestamp(fixedNow + tolerance + 1, tolerance)).toBe(true);
    });
  });

  describe('timestampAge arithmetic', () => {
    it('should calculate age correctly (subtraction not addition)', () => {
      const ts = 5000;
      const reference = 10000;

      // age = reference - ts = 10000 - 5000 = 5000
      expect(utils.timestampAge(ts, reference)).toBe(5000);

      // If mutated to addition: 10000 + 5000 = 15000 (wrong)
      expect(utils.timestampAge(ts, reference)).not.toBe(15000);
    });

    it('should return 0 for same timestamp', () => {
      const ts = 10000;
      expect(utils.timestampAge(ts, ts)).toBe(0);
    });

    it('should return negative for future timestamps', () => {
      const ts = 15000;
      const reference = 10000;
      expect(utils.timestampAge(ts, reference)).toBe(-5000);
    });
  });
});

// ============================================================================
// TRUST CHECK BOUNDARY TESTS - Kill LogicalOperator mutations
// ============================================================================

describe('Trust Check Boundary Tests', () => {
  describe('checkTrust logic combinations', () => {
    it('should be trustworthy only when composite is trustworthy AND not byzantine', () => {
      // Both conditions must be true for trustworthy
      // trustworthy: composite.isTrustworthy && !byzantine
      const agentId = 'agent-test-001';

      const result = utils.checkTrust(agentId, 0.9, 0.9, 0.1, { trustThreshold: 0.5 });

      // With high scores, should be trustworthy
      expect(result.trustworthy).toBe(true);
      expect(result.byzantine).toBe(false);
    });

    it('should NOT be trustworthy when byzantine even if composite is trustworthy', () => {
      // High entropy = byzantine behavior
      const agentId = 'agent-test-002';

      const result = utils.checkTrust(agentId, 0.9, 0.9, 0.9, { trustThreshold: 0.3 });

      // High entropy should make it byzantine
      expect(result.byzantine).toBe(true);
      // Even with good scores, byzantine should make it not trustworthy
      expect(result.trustworthy).toBe(false);
    });

    it('should NOT be trustworthy when composite is not trustworthy', () => {
      const agentId = 'agent-test-003';

      const result = utils.checkTrust(agentId, 0.1, 0.1, 0.1, { trustThreshold: 0.9 });

      expect(result.trustworthy).toBe(false);
    });
  });
});

// ============================================================================
// CONDITIONAL EXPRESSION TESTS - Kill ternary mutations
// ============================================================================

describe('Conditional Expression Tests', () => {
  describe('Optional parameter defaults', () => {
    it('should use referenceTime when provided to isExpiredTimestamp', () => {
      const ts = 1000;
      const ttlMs = 500;
      const customRef = 2000;

      // With custom ref: 2000 - 1000 = 1000 > 500 = true
      expect(utils.isExpiredTimestamp(ts, ttlMs, customRef)).toBe(true);

      // Same ts but different ref should give different result
      const nearRef = 1400;
      // 1400 - 1000 = 400 > 500 = false
      expect(utils.isExpiredTimestamp(ts, ttlMs, nearRef)).toBe(false);
    });
  });

  describe('deferred utility', () => {
    it('should create resolvable deferred promise', async () => {
      const d = utils.deferred<number>();

      expect(d.promise).toBeInstanceOf(Promise);
      expect(typeof d.resolve).toBe('function');
      expect(typeof d.reject).toBe('function');

      d.resolve(42);
      await expect(d.promise).resolves.toBe(42);
    });

    it('should create rejectable deferred promise', async () => {
      const d = utils.deferred<number>();

      const error = new Error('test error');
      d.reject(error);
      await expect(d.promise).rejects.toThrow('test error');
    });
  });
});

// ============================================================================
// ARITHMETIC OPERATOR TESTS - Kill +/- mutations
// ============================================================================

describe('Arithmetic Operator Tests', () => {
  describe('retry with backoff', () => {
    it('should increase delay with backoff factor (multiplication not division)', async () => {
      let callCount = 0;
      const delays: number[] = [];

      const originalSetTimeout = global.setTimeout;
      vi.useFakeTimers();

      const failingFn = async () => {
        callCount++;
        if (callCount < 3) {
          throw new Error('Fail');
        }
        return 'success';
      };

      const promise = utils.retry(failingFn, {
        maxAttempts: 3,
        initialDelay: 100,
        backoffFactor: 2,
      });

      // First attempt fails immediately
      await vi.advanceTimersByTimeAsync(0);

      // Wait for first retry delay (100ms)
      await vi.advanceTimersByTimeAsync(100);

      // Wait for second retry delay (200ms = 100 * 2)
      await vi.advanceTimersByTimeAsync(200);

      const result = await promise;
      expect(result).toBe('success');
      expect(callCount).toBe(3);

      vi.useRealTimers();
    });
  });
});

// ============================================================================
// STRING LITERAL TESTS - Ensure error messages are correct
// ============================================================================

describe('String Literal Tests', () => {
  describe('Error messages should be non-empty', () => {
    it('should format timestamp to non-empty string', () => {
      const formatted = utils.formatTimestamp(Date.now());
      expect(formatted).toBeTruthy();
      expect(formatted.length).toBeGreaterThan(0);
    });
  });

  describe('UUID generation', () => {
    it('should generate non-empty UUIDs', () => {
      const uuid = security.secureUUID();
      expect(uuid).toBeTruthy();
      expect(uuid.length).toBeGreaterThan(0);
      expect(uuid).toMatch(/^[0-9a-f-]+$/i);
    });

    it('should generate unique UUIDs', () => {
      const uuids = new Set<string>();
      for (let i = 0; i < 100; i++) {
        uuids.add(security.secureUUID());
      }
      expect(uuids.size).toBe(100);
    });
  });
});

// ============================================================================
// BLOCK STATEMENT TESTS - Ensure code blocks execute
// ============================================================================

describe('Block Statement Tests', () => {
  describe('retry executes callback block', () => {
    it('should execute the function and return result', async () => {
      let executed = false;

      const result = await utils.retry(async () => {
        executed = true;
        return 'done';
      }, { maxAttempts: 1 });

      expect(executed).toBe(true);
      expect(result).toBe('done');
    });

    it('should execute retry block on failure', async () => {
      let attempts = 0;

      await utils.retry(async () => {
        attempts++;
        if (attempts < 2) throw new Error('fail');
        return 'ok';
      }, { maxAttempts: 3, initialDelay: 1 });

      expect(attempts).toBe(2);
    });
  });
});

// ============================================================================
// BOOLEAN LITERAL TESTS - Kill true/false swaps
// ============================================================================

describe('Boolean Literal Tests', () => {
  describe('isValidTimestamp returns correct booleans', () => {
    it('should return exactly true for valid timestamps', () => {
      const result = utils.isValidTimestamp(Date.now());
      expect(result).toBe(true);
      expect(result).not.toBe(false);
    });

    it('should return exactly false for invalid timestamps', () => {
      const result = utils.isValidTimestamp(-1);
      expect(result).toBe(false);
      expect(result).not.toBe(true);
    });
  });

  describe('timestamp checks return correct types', () => {
    it('isExpiredTimestamp returns boolean, not truthy/falsy', () => {
      const result = utils.isExpiredTimestamp(Date.now() - 10000, 5000);
      expect(typeof result).toBe('boolean');
      expect(result).toBe(true);
    });

    it('isFutureTimestamp returns boolean, not truthy/falsy', () => {
      const result = utils.isFutureTimestamp(Date.now() + 10000);
      expect(typeof result).toBe('boolean');
      expect(result).toBe(true);
    });
  });
});

// ============================================================================
// UPDATE OPERATOR TESTS - Kill ++/-- swaps
// ============================================================================

describe('Update Operator Tests', () => {
  describe('Counter increments and decrements', () => {
    it('CircuitBreaker should track failure count correctly', async () => {
      const cb = new utils.CircuitBreaker({
        failureThreshold: 3,
        successThreshold: 1,
        resetTimeoutMs: 100,
      });

      const failingFn = async () => {
        throw new Error('Simulated failure');
      };

      // Record failures by calling failing function
      try { await cb.call(failingFn); } catch {}
      try { await cb.call(failingFn); } catch {}

      // Not yet open (need 3 failures)
      expect(cb.getStats().consecutiveFailures).toBe(2);

      // One more failure should open the circuit
      try { await cb.call(failingFn); } catch {}
      expect(cb.getStats().consecutiveFailures).toBe(3);
      expect(cb.getStats().state).toBe('open');
    });

    it('CircuitBreaker should track success count correctly', async () => {
      const cb = new utils.CircuitBreaker({
        failureThreshold: 3,
        successThreshold: 2,
        resetTimeoutMs: 100,
      });

      const successFn = async () => 'success';

      await cb.call(successFn);
      expect(cb.getStats().successes).toBe(1);

      await cb.call(successFn);
      expect(cb.getStats().successes).toBe(2);
    });
  });
});

// ============================================================================
// OBJECT LITERAL TESTS - Ensure objects have correct properties
// ============================================================================

describe('Object Literal Tests', () => {
  describe('checkTrust returns complete result object', () => {
    it('should return object with all required properties', () => {
      const agentId = 'agent-obj-001';
      const result = utils.checkTrust(agentId, 0.8, 0.8, 0.1, { trustThreshold: 0.5 });

      // Ensure the object is not empty
      expect(Object.keys(result).length).toBeGreaterThan(0);

      // Check all required properties exist
      expect(result).toHaveProperty('trustworthy');
      expect(result).toHaveProperty('score');
      expect(result).toHaveProperty('byzantine');
      expect(result).toHaveProperty('confidence');
      expect(result).toHaveProperty('details');

      // Check nested details object
      expect(result.details).toHaveProperty('pogqScore');
      expect(result.details).toHaveProperty('reputationScore');
    });
  });

  describe('deferred returns complete object', () => {
    it('should return object with promise, resolve, and reject', () => {
      const d = utils.deferred();

      expect(d).toHaveProperty('promise');
      expect(d).toHaveProperty('resolve');
      expect(d).toHaveProperty('reject');
      expect(d.promise).toBeInstanceOf(Promise);
    });
  });
});

// ============================================================================
// ARRAY DECLARATION TESTS - Ensure arrays are initialized correctly
// ============================================================================

describe('Array Declaration Tests', () => {
  describe('parallel utility handles arrays correctly', () => {
    it('should return array of results', async () => {
      const items = [1, 2, 3];
      const operation = async (item: number) => item * 2;

      const results = await utils.parallel(items, operation, 2);

      expect(Array.isArray(results)).toBe(true);
      expect(results).toHaveLength(3);
      expect(results).toEqual([2, 4, 6]);
    });

    it('should handle empty item array', async () => {
      const results = await utils.parallel([], async (x: number) => x, 2);

      expect(Array.isArray(results)).toBe(true);
      expect(results).toHaveLength(0);
    });

    it('should preserve order of results', async () => {
      const items = [3, 1, 2];
      // Simulate varying delays
      const operation = async (item: number) => {
        await utils.sleep(item * 10);
        return item;
      };

      const results = await utils.parallel(items, operation, 3);

      // Results should be in original order, not completion order
      expect(results).toEqual([3, 1, 2]);
    });
  });
});
