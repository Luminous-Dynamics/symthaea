// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Property-Based Tests for Validation Module
 *
 * Uses fast-check to fuzz test all validators with random inputs.
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import {
  validateDID,
  validateAmount,
  validatePercentage,
  validateURL,
  validateHash,
  validateRange,
  validateNonEmpty,
  combineResults,
  validResult,
  invalidResult,
  type ValidationResult,
} from '../../src/utils/validation.js';

describe('Validation Property-Based Tests', () => {
  // Safe alphanumeric string arbitrary - start with letter, rest alphanumeric
  const alphanumericCharset = 'abcdefghijklmnopqrstuvwxyz0123456789';
  const letterCharset = 'abcdefghijklmnopqrstuvwxyz';

  const alphanumericArb = (minLen: number, maxLen: number) => {
    const firstChar = fc.constantFrom(...letterCharset.split(''));
    const restChars = fc.array(fc.constantFrom(...alphanumericCharset.split('')), {
      minLength: minLen - 1,
      maxLength: maxLen - 1,
    });
    return fc.tuple(firstChar, restChars).map(([first, rest]) => first + rest.join(''));
  };

  // Hex string arbitrary - use hex characters only
  const hexCharset = '0123456789abcdef';

  const hexArb = (minLen: number, maxLen: number) =>
    fc
      .array(fc.constantFrom(...hexCharset.split('')), { minLength: minLen, maxLength: maxLen })
      .map((chars) => chars.join(''));

  describe('DID Validation Properties', () => {
    // Valid DID arbitrary
    const validDIDArb = fc
      .tuple(fc.constantFrom('mycelix', 'key', 'web', 'example'), alphanumericArb(10, 64))
      .map(([method, id]) => `did:${method}:${id}`);

    it('valid DIDs should always pass validation', () => {
      fc.assert(
        fc.property(validDIDArb, (did) => {
          const result = validateDID(did);
          expect(result.valid).toBe(true);
        }),
        { numRuns: 50 }
      );
    });

    it('empty strings should fail validation', () => {
      const result = validateDID('');
      expect(result.valid).toBe(false);
    });

    it('strings without did: prefix should fail', () => {
      fc.assert(
        fc.property(
          // Use alphanumericArb to generate strings that definitely don't start with 'did:'
          alphanumericArb(5, 50),
          (str) => {
            const result = validateDID(str);
            expect(result.valid).toBe(false);
          }
        ),
        { numRuns: 30 }
      );
    });

    it('validation result should be consistent', () => {
      fc.assert(
        fc.property(fc.string({ minLength: 0, maxLength: 100 }), (str) => {
          const result1 = validateDID(str);
          const result2 = validateDID(str);
          expect(result1.valid).toBe(result2.valid);
        }),
        { numRuns: 30 }
      );
    });
  });

  describe('Amount Validation Properties', () => {
    it('positive amounts should be valid', () => {
      fc.assert(
        fc.property(
          fc.float({ min: Math.fround(0.01), max: Math.fround(1e9), noNaN: true }),
          (amount) => {
            const result = validateAmount(amount, 'Test');
            expect(result.valid).toBe(true);
          }
        ),
        { numRuns: 100 }
      );
    });

    it('zero should be invalid for amounts', () => {
      const result = validateAmount(0, 'Test');
      expect(result.valid).toBe(false);
    });

    it('negative amounts should be invalid', () => {
      fc.assert(
        fc.property(
          fc.float({ min: Math.fround(-1e9), max: Math.fround(-0.01), noNaN: true }),
          (amount) => {
            const result = validateAmount(amount, 'Test');
            expect(result.valid).toBe(false);
          }
        ),
        { numRuns: 50 }
      );
    });

    it('NaN should be invalid', () => {
      const result = validateAmount(NaN, 'Test');
      expect(result.valid).toBe(false);
    });

    it('Infinity should be handled appropriately', () => {
      // Infinity is positive, so it's considered valid by validatePositive
      expect(validateAmount(Infinity, 'Test').valid).toBe(true);
      // -Infinity is not positive, so it's invalid
      expect(validateAmount(-Infinity, 'Test').valid).toBe(false);
    });
  });

  describe('Percentage Validation Properties', () => {
    it('values 0-1 should be valid percentages', () => {
      fc.assert(
        fc.property(fc.float({ min: Math.fround(0), max: Math.fround(1), noNaN: true }), (pct) => {
          const result = validatePercentage(pct, 'Test');
          expect(result.valid).toBe(true);
        }),
        { numRuns: 100 }
      );
    });

    it('values outside 0-1 should be invalid', () => {
      fc.assert(
        fc.property(
          fc.oneof(
            fc.float({ min: Math.fround(-1000), max: Math.fround(-0.001), noNaN: true }),
            fc.float({ min: Math.fround(1.001), max: Math.fround(1000), noNaN: true })
          ),
          (pct) => {
            const result = validatePercentage(pct, 'Test');
            expect(result.valid).toBe(false);
          }
        ),
        { numRuns: 50 }
      );
    });

    it('boundary values should be valid', () => {
      expect(validatePercentage(0, 'Test').valid).toBe(true);
      expect(validatePercentage(1, 'Test').valid).toBe(true);
      expect(validatePercentage(0.5, 'Test').valid).toBe(true);
    });
  });

  describe('URL Validation Properties', () => {
    // Valid URL arbitrary
    const validURLArb = fc
      .tuple(
        fc.constantFrom('http', 'https'),
        alphanumericArb(3, 20),
        fc.constantFrom('.com', '.org', '.net', '.io', '.dev')
      )
      .map(([protocol, domain, tld]) => `${protocol}://${domain}${tld}`);

    it('valid URLs should pass validation', () => {
      fc.assert(
        fc.property(validURLArb, (url) => {
          const result = validateURL(url, 'Test');
          expect(result.valid).toBe(true);
        }),
        { numRuns: 50 }
      );
    });

    it('empty strings should fail', () => {
      const result = validateURL('', 'Test');
      expect(result.valid).toBe(false);
    });

    it('random strings without protocol should fail', () => {
      fc.assert(
        fc.property(alphanumericArb(5, 20), (str) => {
          // Only test strings that don't accidentally form valid URLs
          if (!str.includes('://')) {
            const result = validateURL(str, 'Test');
            expect(result.valid).toBe(false);
          }
        }),
        { numRuns: 30 }
      );
    });
  });

  describe('Hash Validation Properties', () => {
    // Valid hash arbitrary (hex string of even length)
    const validHashArb = hexArb(32, 64).map((s) => (s.length % 2 === 0 ? s : s + '0'));

    it('valid hex hashes should pass', () => {
      fc.assert(
        fc.property(validHashArb, (hash) => {
          const result = validateHash(hash, 'Test');
          expect(result.valid).toBe(true);
        }),
        { numRuns: 50 }
      );
    });

    it('empty hash should fail', () => {
      const result = validateHash('', 'Test');
      expect(result.valid).toBe(false);
    });

    it('non-hex characters should fail', () => {
      // Generate strings with only g-z characters (no hex digits)
      const nonHexCharset = 'ghijklmnopqrstuvwxyz';
      const nonHexArb = fc
        .array(fc.constantFrom(...nonHexCharset.split('')), { minLength: 32, maxLength: 64 })
        .map((chars) => chars.join(''));

      fc.assert(
        fc.property(nonHexArb, (str) => {
          const result = validateHash(str, 'Test');
          expect(result.valid).toBe(false);
        }),
        { numRuns: 30 }
      );
    });
  });

  describe('Range Validation Properties', () => {
    it('values within range should be valid', () => {
      fc.assert(
        fc.property(
          fc.float({ min: Math.fround(-1000), max: Math.fround(1000), noNaN: true }),
          fc.float({ min: Math.fround(-1000), max: Math.fround(1000), noNaN: true }),
          fc.float({ min: Math.fround(-1000), max: Math.fround(1000), noNaN: true }),
          (a, b, c) => {
            const [min, max] = [Math.min(a, b), Math.max(a, b)];
            const value = min + ((max - min) * Math.abs(c)) / 1000;

            if (min < max && value >= min && value <= max) {
              const result = validateRange(value, 'Test', { min, max });
              expect(result.valid).toBe(true);
            }
          }
        ),
        { numRuns: 50 }
      );
    });

    it('values outside range should be invalid', () => {
      fc.assert(
        fc.property(
          fc.float({ min: Math.fround(0), max: Math.fround(100), noNaN: true }),
          fc.float({ min: Math.fround(100), max: Math.fround(200), noNaN: true }),
          fc.float({ min: Math.fround(200), max: Math.fround(1000), noNaN: true }),
          (min, max, value) => {
            const result = validateRange(value, 'Test', { min, max });
            expect(result.valid).toBe(false);
          }
        ),
        { numRuns: 50 }
      );
    });
  });

  describe('NonEmpty Validation Properties', () => {
    it('non-empty strings should be valid', () => {
      fc.assert(
        fc.property(fc.string({ minLength: 1, maxLength: 1000 }), (str) => {
          if (str.trim().length > 0) {
            const result = validateNonEmpty(str, 'Test');
            expect(result.valid).toBe(true);
          }
        }),
        { numRuns: 50 }
      );
    });

    it('empty strings should be invalid', () => {
      const result = validateNonEmpty('', 'Test');
      expect(result.valid).toBe(false);
    });

    it('whitespace-only strings should be invalid', () => {
      const whitespaceStrings = ['   ', '\t\t', '\n\n', '  \t\n  '];
      for (const str of whitespaceStrings) {
        const result = validateNonEmpty(str, 'Test');
        expect(result.valid).toBe(false);
      }
    });
  });

  describe('Result Combination Properties', () => {
    it('combining all valid results should be valid', () => {
      fc.assert(
        fc.property(
          fc.array(fc.constant(validResult()), { minLength: 1, maxLength: 20 }),
          (results) => {
            // combineResults takes rest parameters, so spread the array
            const combined = combineResults(...results);
            expect(combined.valid).toBe(true);
          }
        ),
        { numRuns: 100 }
      );
    });

    it('combining with any invalid result should be invalid', () => {
      fc.assert(
        fc.property(
          fc.array(fc.constant(validResult()), { minLength: 0, maxLength: 10 }),
          alphanumericArb(1, 50),
          fc.array(fc.constant(validResult()), { minLength: 0, maxLength: 10 }),
          (before, errorMsg, after) => {
            const results = [...before, invalidResult(errorMsg), ...after];
            // combineResults takes rest parameters, so spread the array
            const combined = combineResults(...results);
            expect(combined.valid).toBe(false);
          }
        ),
        { numRuns: 30 }
      );
    });

    it('empty combination should be valid', () => {
      // combineResults() with no arguments returns valid because no errors exist
      const combined = combineResults();
      expect(combined.valid).toBe(true);
    });

    it('combined errors should include all error messages', () => {
      fc.assert(
        fc.property(
          fc.array(alphanumericArb(1, 30), { minLength: 2, maxLength: 5 }),
          (messages) => {
            const results = messages.map((msg) => invalidResult(msg));
            // combineResults takes rest parameters, so spread the array
            const combined = combineResults(...results);

            expect(combined.valid).toBe(false);
            for (const msg of messages) {
              expect(combined.errors).toContain(msg);
            }
          }
        ),
        { numRuns: 100 }
      );
    });
  });
});
