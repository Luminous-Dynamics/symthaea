// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Property-Based Tests for Security Module
 *
 * Uses fast-check to verify security properties:
 * - Random number generation uniformity
 * - Rate limiting correctness
 * - Input sanitization safety
 * - BFT parameter validation
 */

import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import {
  secureRandomBytes,
  secureRandomFloat,
  secureRandomInt,
  secureUUID,
  secureShuffle,
  constantTimeEqual,
  createRateLimiter,
  checkRateLimit,
  sanitizeString,
  sanitizeId,
  validateBftParams,
  maxByzantineFailures,
  hasQuorum,
} from '../../src/security/index.js';

describe('Security Property-Based Tests', () => {
  describe('Secure Random Number Generation Properties', () => {
    it('secureRandomBytes should return correct length', () => {
      fc.assert(
        fc.property(fc.integer({ min: 1, max: 1000 }), (length) => {
          const bytes = secureRandomBytes(length);
          expect(bytes.length).toBe(length);
          expect(bytes).toBeInstanceOf(Uint8Array);
        }),
        { numRuns: 100 }
      );
    });

    it('secureRandomFloat should be in [0, 1)', () => {
      fc.assert(
        fc.property(fc.nat({ max: 1000 }), () => {
          const value = secureRandomFloat();
          expect(value).toBeGreaterThanOrEqual(0);
          expect(value).toBeLessThan(1);
        }),
        { numRuns: 500 }
      );
    });

    it('secureRandomInt should be within bounds', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 0, max: 1000 }),
          fc.integer({ min: 1, max: 1000 }),
          (min, range) => {
            const max = min + range;
            const value = secureRandomInt(min, max);
            expect(value).toBeGreaterThanOrEqual(min);
            // secureRandomInt is inclusive on both ends (range = max - min + 1)
            expect(value).toBeLessThanOrEqual(max);
            expect(Number.isInteger(value)).toBe(true);
          }
        ),
        { numRuns: 200 }
      );
    });

    it('secureUUID should follow UUID v4 format', () => {
      fc.assert(
        fc.property(fc.nat({ max: 100 }), () => {
          const uuid = secureUUID();
          // UUID v4 format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
          expect(uuid).toMatch(
            /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
          );
        }),
        { numRuns: 100 }
      );
    });

    it('secureUUID should be unique', () => {
      const uuids = new Set<string>();
      for (let i = 0; i < 1000; i++) {
        uuids.add(secureUUID());
      }
      expect(uuids.size).toBe(1000);
    });
  });

  describe('Secure Shuffle Properties', () => {
    it('shuffle should preserve array length', () => {
      fc.assert(
        fc.property(fc.array(fc.integer(), { minLength: 0, maxLength: 100 }), (arr) => {
          const shuffled = secureShuffle([...arr]);
          expect(shuffled.length).toBe(arr.length);
        }),
        { numRuns: 100 }
      );
    });

    it('shuffle should preserve array elements', () => {
      fc.assert(
        fc.property(fc.array(fc.integer(), { minLength: 0, maxLength: 100 }), (arr) => {
          const shuffled = secureShuffle([...arr]);
          const sortedOriginal = [...arr].sort((a, b) => a - b);
          const sortedShuffled = [...shuffled].sort((a, b) => a - b);
          expect(sortedShuffled).toEqual(sortedOriginal);
        }),
        { numRuns: 100 }
      );
    });

    it('shuffle should not always return same order (statistical)', () => {
      const arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      let sameCount = 0;
      const runs = 100;

      for (let i = 0; i < runs; i++) {
        const shuffled = secureShuffle([...arr]);
        if (shuffled.every((v, idx) => v === arr[idx])) {
          sameCount++;
        }
      }

      // Statistically, almost none should match
      expect(sameCount).toBeLessThan(runs * 0.05);
    });
  });

  describe('Constant Time Comparison Properties', () => {
    it('equal arrays should return true', () => {
      fc.assert(
        fc.property(fc.uint8Array({ minLength: 1, maxLength: 100 }), (arr) => {
          const copy = new Uint8Array(arr);
          expect(constantTimeEqual(arr, copy)).toBe(true);
        }),
        { numRuns: 100 }
      );
    });

    it('different arrays should return false', () => {
      fc.assert(
        fc.property(
          fc.uint8Array({ minLength: 1, maxLength: 100 }),
          fc.integer({ min: 0, max: 99 }),
          (arr, idx) => {
            if (idx >= arr.length) return;

            const modified = new Uint8Array(arr);
            modified[idx] = (modified[idx] + 1) % 256;
            expect(constantTimeEqual(arr, modified)).toBe(false);
          }
        ),
        { numRuns: 100 }
      );
    });

    it('different length arrays should return false', () => {
      fc.assert(
        fc.property(
          fc.uint8Array({ minLength: 1, maxLength: 50 }),
          fc.uint8Array({ minLength: 51, maxLength: 100 }),
          (arr1, arr2) => {
            expect(constantTimeEqual(arr1, arr2)).toBe(false);
          }
        ),
        { numRuns: 50 }
      );
    });

    it('empty arrays should be equal', () => {
      expect(constantTimeEqual(new Uint8Array(0), new Uint8Array(0))).toBe(true);
    });
  });

  describe('Rate Limiting Properties', () => {
    it('should allow requests under limit', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 5, max: 100 }),
          fc.integer({ min: 1000, max: 60000 }),
          (maxRequests, windowMs) => {
            let limiter = createRateLimiter('test', { maxRequests, windowMs });

            // Make fewer requests than limit
            const requestCount = Math.floor(maxRequests / 2);
            let allAllowed = true;

            for (let i = 0; i < requestCount; i++) {
              const result = checkRateLimit(limiter);
              limiter = result.state;
              if (!result.allowed) {
                allAllowed = false;
                break;
              }
            }

            expect(allAllowed).toBe(true);
          }
        ),
        { numRuns: 30 }
      );
    });

    it('should block requests over limit', () => {
      const maxRequests = 5;
      const windowMs = 60000;
      let limiter = createRateLimiter('test', { maxRequests, windowMs });

      // Use all requests
      for (let i = 0; i < maxRequests; i++) {
        const result = checkRateLimit(limiter);
        limiter = result.state;
        expect(result.allowed).toBe(true);
      }

      // Next request should be blocked
      const result = checkRateLimit(limiter);
      expect(result.allowed).toBe(false);
    });

    it('rate limiter should have correct initial state', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: 1, max: 100 }),
          fc.integer({ min: 100, max: 60000 }),
          (maxRequests, windowMs) => {
            const limiter = createRateLimiter('test-id', { maxRequests, windowMs });

            expect(limiter.id).toBe('test-id');
            expect(limiter.config.maxRequests).toBe(maxRequests);
            expect(limiter.config.windowMs).toBe(windowMs);
            expect(limiter.timestamps).toEqual([]);
            expect(limiter.blocked).toBe(false);
          }
        ),
        { numRuns: 30 }
      );
    });
  });

  describe('Input Sanitization Properties', () => {
    it('sanitizeString should remove dangerous characters', () => {
      fc.assert(
        fc.property(fc.string({ minLength: 0, maxLength: 200 }), (input) => {
          const sanitized = sanitizeString(input);

          // Should not contain script tags
          expect(sanitized).not.toMatch(/<script/i);
          // Should not contain event handlers
          expect(sanitized).not.toMatch(/on\w+=/i);
          // Should be a string
          expect(typeof sanitized).toBe('string');
        }),
        { numRuns: 100 }
      );
    });

    it('sanitizeString should preserve safe content', () => {
      fc.assert(
        fc.property(
          fc.array(fc.constantFrom(...'abcdefghijklmnopqrstuvwxyz0123456789 '.split('')), {
            minLength: 1,
            maxLength: 50,
          }),
          (chars) => {
            const input = chars.join('');
            const sanitized = sanitizeString(input);
            // Safe alphanumeric content should be preserved
            expect(sanitized).toBe(input);
          }
        ),
        { numRuns: 50 }
      );
    });

    it('sanitizeId should produce valid identifiers', () => {
      fc.assert(
        fc.property(fc.string({ minLength: 1, maxLength: 100 }), (input) => {
          try {
            const sanitized = sanitizeId(input);

            // Should only contain alphanumeric, underscore, hyphen
            expect(sanitized).toMatch(/^[a-zA-Z0-9_-]*$/);
            // Should be a string
            expect(typeof sanitized).toBe('string');
          } catch (e) {
            // sanitizeId throws SecurityError for inputs that produce empty identifiers
            // (e.g., whitespace-only strings) - this is valid behavior
            expect((e as Error).message).toMatch(/Invalid identifier/);
          }
        }),
        { numRuns: 100 }
      );
    });

    it('sanitizeId should be idempotent', () => {
      fc.assert(
        fc.property(fc.string({ minLength: 1, maxLength: 100 }), (input) => {
          try {
            const once = sanitizeId(input);
            const twice = sanitizeId(once);
            expect(twice).toBe(once);
          } catch (e) {
            // sanitizeId throws for invalid inputs - this is expected
            expect((e as Error).message).toMatch(/Invalid identifier/);
          }
        }),
        { numRuns: 50 }
      );
    });
  });

  describe('BFT Parameter Validation Properties', () => {
    it('valid BFT params should pass validation', () => {
      fc.assert(
        fc.property(fc.integer({ min: 4, max: 100 }), (n) => {
          // f < n/3 for Byzantine tolerance
          const f = Math.floor((n - 1) / 3);
          const result = validateBftParams(n, f);
          expect(result.valid).toBe(true);
        }),
        { numRuns: 50 }
      );
    });

    it('too many Byzantine nodes should fail validation', () => {
      fc.assert(
        fc.property(fc.integer({ min: 4, max: 100 }), (n) => {
          // f >= n/3 should fail
          const f = Math.ceil(n / 3);
          const result = validateBftParams(n, f);
          expect(result.valid).toBe(false);
        }),
        { numRuns: 50 }
      );
    });

    it('negative values should fail validation', () => {
      fc.assert(
        fc.property(
          fc.integer({ min: -100, max: -1 }),
          fc.integer({ min: 0, max: 100 }),
          (n, f) => {
            const result = validateBftParams(n, f);
            expect(result.valid).toBe(false);
          }
        ),
        { numRuns: 30 }
      );
    });

    it('maxByzantineFailures should satisfy BFT bound', () => {
      fc.assert(
        fc.property(fc.integer({ min: 1, max: 100 }), (n) => {
          const f = maxByzantineFailures(n);

          // Should satisfy 3f + 1 <= n
          expect(3 * f + 1).toBeLessThanOrEqual(n);

          // Should be maximum possible
          expect(3 * (f + 1) + 1).toBeGreaterThan(n);
        }),
        { numRuns: 50 }
      );
    });
  });

  describe('Quorum Properties', () => {
    // hasQuorum(totalNodes, byzantineNodes, requiredHonest) checks:
    // (totalNodes - byzantineNodes) >= requiredHonest

    it('quorum should require sufficient honest nodes', () => {
      fc.assert(
        fc.property(fc.integer({ min: 4, max: 50 }), (n) => {
          const f = Math.floor((n - 1) / 3);
          const honestNodes = n - f;
          // For BFT, quorum typically requires 2f+1 honest nodes
          const quorumSize = 2 * f + 1;

          // With n total nodes and f Byzantine, we have (n-f) honest
          // Quorum achieved if (n - f) >= 2f+1
          expect(hasQuorum(n, f, quorumSize)).toBe(honestNodes >= quorumSize);
        }),
        { numRuns: 50 }
      );
    });

    it('zero Byzantine nodes should always have quorum if total >= required', () => {
      fc.assert(
        fc.property(fc.integer({ min: 4, max: 50 }), (n) => {
          // With 0 Byzantine nodes, all n nodes are honest
          const requiredHonest = Math.floor(n / 2) + 1; // Simple majority
          expect(hasQuorum(n, 0, requiredHonest)).toBe(true);
        }),
        { numRuns: 30 }
      );
    });

    it('all nodes Byzantine should never have quorum', () => {
      fc.assert(
        fc.property(fc.integer({ min: 4, max: 50 }), (n) => {
          // If all n nodes are Byzantine, 0 honest nodes remain
          expect(hasQuorum(n, n, 1)).toBe(false);
        }),
        { numRuns: 30 }
      );
    });
  });

  describe('Determinism Properties', () => {
    it('sanitization should be deterministic', () => {
      fc.assert(
        fc.property(fc.string({ minLength: 0, maxLength: 100 }), (input) => {
          const result1 = sanitizeString(input);
          const result2 = sanitizeString(input);
          expect(result1).toBe(result2);
        }),
        { numRuns: 50 }
      );
    });

    it('BFT validation should be deterministic', () => {
      fc.assert(
        fc.property(fc.integer({ min: 1, max: 100 }), fc.integer({ min: 0, max: 50 }), (n, f) => {
          const result1 = validateBftParams(n, f);
          const result2 = validateBftParams(n, f);
          expect(result1.valid).toBe(result2.valid);
        }),
        { numRuns: 50 }
      );
    });

    it('constantTimeEqual should be deterministic', () => {
      fc.assert(
        fc.property(
          fc.uint8Array({ minLength: 1, maxLength: 50 }),
          fc.uint8Array({ minLength: 1, maxLength: 50 }),
          (a, b) => {
            const result1 = constantTimeEqual(a, b);
            const result2 = constantTimeEqual(a, b);
            expect(result1).toBe(result2);
          }
        ),
        { numRuns: 50 }
      );
    });
  });

  describe('Edge Cases', () => {
    it('should handle single byte generation', () => {
      const bytes = secureRandomBytes(1);
      expect(bytes.length).toBe(1);
    });

    it('should handle min == max-1 for secureRandomInt', () => {
      const value = secureRandomInt(5, 6);
      // secureRandomInt is inclusive on both ends, so can return 5 or 6
      expect(value).toBeGreaterThanOrEqual(5);
      expect(value).toBeLessThanOrEqual(6);
    });

    it('should handle empty string sanitization', () => {
      const sanitized = sanitizeString('');
      expect(sanitized).toBe('');
    });

    it('should handle empty array shuffle', () => {
      const shuffled = secureShuffle([]);
      expect(shuffled).toEqual([]);
    });

    it('should handle single element shuffle', () => {
      const shuffled = secureShuffle([42]);
      expect(shuffled).toEqual([42]);
    });
  });
});
