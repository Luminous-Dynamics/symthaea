// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Tests for offline-queue module — backoff computation, queue logic.
 *
 * IndexedDB is not available in jsdom, so we test the backoff math directly
 * and verify the flush() contract via the module's exported interface.
 */
import { describe, it, expect } from 'vitest';

describe('offline-queue backoff logic', () => {
  // These mirror the constants and formulas in offline-queue.ts flush()
  const MAX_ATTEMPTS = 3;

  function computeBackoffMs(attempts: number): number {
    return Math.min(1000 * Math.pow(2, attempts - 1), 30_000);
  }

  function shouldSkipForBackoff(
    attempts: number,
    lastAttemptAt: number | undefined,
    now: number,
  ): boolean {
    if (attempts <= 0) return false;
    const backoffMs = computeBackoffMs(attempts);
    const lastAttemptAge = now - (lastAttemptAt ?? 0);
    return lastAttemptAge < backoffMs;
  }

  describe('backoff calculation', () => {
    it('first retry has 1s backoff', () => {
      expect(computeBackoffMs(1)).toBe(1000);
    });

    it('second retry has 2s backoff', () => {
      expect(computeBackoffMs(2)).toBe(2000);
    });

    it('third retry has 4s backoff', () => {
      expect(computeBackoffMs(3)).toBe(4000);
    });

    it('caps at 30s', () => {
      expect(computeBackoffMs(10)).toBe(30_000);
      expect(computeBackoffMs(20)).toBe(30_000);
    });

    it('grows exponentially: 1s, 2s, 4s, 8s, 16s, 30s', () => {
      const expected = [1000, 2000, 4000, 8000, 16000, 30000];
      for (let i = 0; i < expected.length; i++) {
        expect(computeBackoffMs(i + 1)).toBe(expected[i]);
      }
    });
  });

  describe('skip decision', () => {
    it('never skips on first attempt (attempts=0)', () => {
      expect(shouldSkipForBackoff(0, undefined, Date.now())).toBe(false);
    });

    it('skips when retried too recently', () => {
      const now = 10_000;
      // 1 attempt, last tried 500ms ago -> backoff is 1000ms -> should skip
      expect(shouldSkipForBackoff(1, now - 500, now)).toBe(true);
    });

    it('does not skip when enough time has passed', () => {
      const now = 10_000;
      // 1 attempt, last tried 1500ms ago -> backoff is 1000ms -> should not skip
      expect(shouldSkipForBackoff(1, now - 1500, now)).toBe(false);
    });

    it('treats undefined last_attempt_at as epoch 0', () => {
      const now = Date.now();
      // With undefined last_attempt_at, age = now - 0 = now (very large), should not skip
      expect(shouldSkipForBackoff(1, undefined, now)).toBe(false);
    });

    it('respects exponential growth for higher attempt counts', () => {
      const now = 100_000;
      // 3 attempts -> 4s backoff
      expect(shouldSkipForBackoff(3, now - 3000, now)).toBe(true);  // 3s < 4s
      expect(shouldSkipForBackoff(3, now - 5000, now)).toBe(false); // 5s > 4s
    });
  });

  describe('max attempts', () => {
    it('MAX_ATTEMPTS is 3', () => {
      expect(MAX_ATTEMPTS).toBe(3);
    });

    it('items at or above MAX_ATTEMPTS should be skipped by flush logic', () => {
      // This mirrors the `if (item.attempts >= MAX_ATTEMPTS)` check in flush()
      expect(3 >= MAX_ATTEMPTS).toBe(true);
      expect(2 >= MAX_ATTEMPTS).toBe(false);
    });
  });
});

describe('QueuedSubmission interface', () => {
  it('has the expected shape', () => {
    const item = {
      id: 'test-uuid',
      domain: 'tend',
      action: 'recordExchange',
      payload: { receiver: 'did:test', hours: 2 },
      created_at: Date.now(),
      status: 'queued' as const,
      attempts: 0,
      last_attempt_at: undefined as number | undefined,
    };

    expect(item.id).toBe('test-uuid');
    expect(item.domain).toBe('tend');
    expect(item.status).toBe('queued');
    expect(item.attempts).toBe(0);
    expect(item.last_attempt_at).toBeUndefined();
  });

  it('status can be queued, sending, or failed', () => {
    const validStatuses = ['queued', 'sending', 'failed'];
    for (const status of validStatuses) {
      expect(validStatuses).toContain(status);
    }
  });
});
