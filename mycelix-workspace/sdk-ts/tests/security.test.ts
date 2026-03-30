// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Security Module Tests
 *
 * Comprehensive tests for cryptographic utilities, rate limiting,
 * input sanitization, and Byzantine resistance utilities.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as security from '../src/security/index.js';

describe('Security Module', () => {
  // ============================================================================
  // Secure Random Number Generation
  // ============================================================================

  describe('Secure Random Generation', () => {
    it('should generate random bytes of correct length', () => {
      const bytes16 = security.secureRandomBytes(16);
      const bytes32 = security.secureRandomBytes(32);
      const bytes64 = security.secureRandomBytes(64);

      expect(bytes16.length).toBe(16);
      expect(bytes32.length).toBe(32);
      expect(bytes64.length).toBe(64);
    });

    it('should generate different random bytes each time', () => {
      const bytes1 = security.secureRandomBytes(32);
      const bytes2 = security.secureRandomBytes(32);
      const bytes3 = security.secureRandomBytes(32);

      // Extremely unlikely to be equal
      expect(bytes1).not.toEqual(bytes2);
      expect(bytes2).not.toEqual(bytes3);
      expect(bytes1).not.toEqual(bytes3);
    });

    it('should generate random floats in range [0, 1)', () => {
      for (let i = 0; i < 100; i++) {
        const value = security.secureRandomFloat();
        expect(value).toBeGreaterThanOrEqual(0);
        expect(value).toBeLessThan(1);
      }
    });

    it('should generate random integers in range [min, max]', () => {
      for (let i = 0; i < 100; i++) {
        const value = security.secureRandomInt(10, 20);
        expect(value).toBeGreaterThanOrEqual(10);
        expect(value).toBeLessThanOrEqual(20);
        expect(Number.isInteger(value)).toBe(true);
      }
    });

    it('should handle edge cases for random int range', () => {
      // Same min and max
      expect(security.secureRandomInt(5, 5)).toBe(5);

      // Large range
      const largeRangeValue = security.secureRandomInt(0, 1000000);
      expect(largeRangeValue).toBeGreaterThanOrEqual(0);
      expect(largeRangeValue).toBeLessThanOrEqual(1000000);
    });

    it('should generate valid UUIDs', () => {
      const uuid1 = security.secureUUID();
      const uuid2 = security.secureUUID();

      // Should match UUID v4 format
      const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
      expect(uuid1).toMatch(uuidRegex);
      expect(uuid2).toMatch(uuidRegex);

      // Should be unique
      expect(uuid1).not.toBe(uuid2);
    });

    it('should shuffle arrays securely', () => {
      const original = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const shuffled = security.secureShufffle(original);

      // Should be same length
      expect(shuffled.length).toBe(original.length);

      // Should contain same elements (sort copy to preserve shuffle order)
      expect([...shuffled].sort((a, b) => a - b)).toEqual(original);

      // Run multiple shuffles to verify randomness
      // At least one of 5 shuffles should differ from original
      let anyDifferent = false;
      for (let i = 0; i < 5; i++) {
        const attempt = security.secureShufffle(original);
        if (attempt.some((v, idx) => v !== original[idx])) {
          anyDifferent = true;
          break;
        }
      }
      expect(anyDifferent).toBe(true);
    });

    it('should throw on invalid length for random bytes', () => {
      expect(() => security.secureRandomBytes(0)).toThrow();
      expect(() => security.secureRandomBytes(-1)).toThrow();
    });
  });

  // ============================================================================
  // Cryptographic Hashing
  // ============================================================================

  describe('Cryptographic Hashing', () => {
    it('should compute SHA-256 hash', async () => {
      const hash = await security.hash('hello world');
      expect(hash.length).toBe(32); // 256 bits = 32 bytes

      const hashHex = await security.hashHex('hello world');
      expect(hashHex.length).toBe(64); // 32 bytes * 2 hex chars
    });

    it('should produce consistent hashes for same input', async () => {
      const hash1 = await security.hashHex('test data');
      const hash2 = await security.hashHex('test data');
      expect(hash1).toBe(hash2);
    });

    it('should produce different hashes for different input', async () => {
      const hash1 = await security.hashHex('test1');
      const hash2 = await security.hashHex('test2');
      expect(hash1).not.toBe(hash2);
    });

    it('should support different hash algorithms', async () => {
      const sha256 = await security.hash('data', 'SHA-256');
      const sha384 = await security.hash('data', 'SHA-384');
      const sha512 = await security.hash('data', 'SHA-512');

      expect(sha256.length).toBe(32);
      expect(sha384.length).toBe(48);
      expect(sha512.length).toBe(64);
    });

    it('should compute HMAC correctly', async () => {
      const key = security.secureRandomBytes(32);
      const hmac = await security.hmac(key, 'message');

      expect(hmac.length).toBe(32); // SHA-256

      // Same key and message should produce same HMAC
      const hmac2 = await security.hmac(key, 'message');
      expect(security.constantTimeEqual(hmac, hmac2)).toBe(true);
    });

    it('should verify HMAC with constant-time comparison', async () => {
      const key = security.secureRandomBytes(32);
      const message = 'important data';
      const mac = await security.hmac(key, message);

      const isValid = await security.verifyHmac(key, message, mac);
      expect(isValid).toBe(true);

      // Tampered message should fail
      const isInvalid = await security.verifyHmac(key, 'tampered data', mac);
      expect(isInvalid).toBe(false);
    });

    it('should perform constant-time comparison', () => {
      const a = new Uint8Array([1, 2, 3, 4, 5]);
      const b = new Uint8Array([1, 2, 3, 4, 5]);
      const c = new Uint8Array([1, 2, 3, 4, 6]);
      const d = new Uint8Array([1, 2, 3]);

      expect(security.constantTimeEqual(a, b)).toBe(true);
      expect(security.constantTimeEqual(a, c)).toBe(false);
      expect(security.constantTimeEqual(a, d)).toBe(false);
    });
  });

  // ============================================================================
  // Rate Limiting
  // ============================================================================

  describe('Rate Limiting', () => {
    it('should create rate limiter with config', () => {
      const limiter = security.createRateLimiter('test', {
        maxRequests: 10,
        windowMs: 1000,
      });

      expect(limiter.id).toBe('test');
      expect(limiter.config.maxRequests).toBe(10);
      expect(limiter.config.windowMs).toBe(1000);
    });

    it('should allow requests within limit', () => {
      let state = security.createRateLimiter('test', {
        maxRequests: 3,
        windowMs: 10000,
      });

      // First 3 requests should be allowed
      for (let i = 0; i < 3; i++) {
        const result = security.checkRateLimit(state);
        expect(result.allowed).toBe(true);
        expect(result.remaining).toBe(2 - i);
        state = result.state;
      }
    });

    it('should block requests exceeding limit', () => {
      let state = security.createRateLimiter('test', {
        maxRequests: 2,
        windowMs: 10000,
      });

      // First 2 requests allowed
      const r1 = security.checkRateLimit(state);
      state = r1.state;
      const r2 = security.checkRateLimit(state);
      state = r2.state;

      expect(r1.allowed).toBe(true);
      expect(r2.allowed).toBe(true);

      // Third request blocked
      const r3 = security.checkRateLimit(state);
      expect(r3.allowed).toBe(false);
      expect(r3.remaining).toBe(0);
    });

    it('should use RateLimiterRegistry for multiple entities', () => {
      const registry = new security.RateLimiterRegistry({
        maxRequests: 5,
        windowMs: 10000,
      });

      // Different entities have separate limits
      for (let i = 0; i < 5; i++) {
        expect(registry.check('user1').allowed).toBe(true);
        expect(registry.check('user2').allowed).toBe(true);
      }

      // Both should now be rate limited
      expect(registry.check('user1').allowed).toBe(false);
      expect(registry.check('user2').allowed).toBe(false);
    });

    it('should reset rate limiter', () => {
      const registry = new security.RateLimiterRegistry({
        maxRequests: 1,
        windowMs: 10000,
      });

      registry.check('user1');
      expect(registry.check('user1').allowed).toBe(false);

      registry.reset('user1');
      expect(registry.check('user1').allowed).toBe(true);
    });

    it('should throw on invalid rate limiter config', () => {
      expect(() =>
        security.createRateLimiter('test', {
          maxRequests: 0,
          windowMs: 1000,
        })
      ).toThrow();

      expect(() =>
        security.createRateLimiter('', {
          maxRequests: 10,
          windowMs: 1000,
        })
      ).toThrow();
    });

    it('should track metrics for rate limited entities', () => {
      const registry = new security.RateLimiterRegistry({
        maxRequests: 3,
        windowMs: 10000,
      });

      // Make some requests
      registry.check('user1');
      registry.check('user1');
      registry.check('user1');
      registry.check('user1'); // This should be rejected

      const metrics = registry.getMetrics('user1');
      expect(metrics).not.toBeNull();
      expect(metrics!.totalRequests).toBe(4);
      expect(metrics!.rejectedRequests).toBe(1);
      expect(metrics!.peakRequestsInWindow).toBe(3);
    });

    it('should return null for unknown entity metrics', () => {
      const registry = new security.RateLimiterRegistry({
        maxRequests: 5,
        windowMs: 10000,
      });

      expect(registry.getMetrics('unknown')).toBeNull();
    });

    it('should get all metrics', () => {
      const registry = new security.RateLimiterRegistry({
        maxRequests: 5,
        windowMs: 10000,
      });

      registry.check('user1');
      registry.check('user2');
      registry.check('user3');

      const allMetrics = registry.getAllMetrics();
      expect(allMetrics.size).toBe(3);
      expect(allMetrics.has('user1')).toBe(true);
      expect(allMetrics.has('user2')).toBe(true);
      expect(allMetrics.has('user3')).toBe(true);
    });

    it('should provide summary statistics', () => {
      const registry = new security.RateLimiterRegistry({
        maxRequests: 2,
        windowMs: 10000,
      });

      registry.check('user1');
      registry.check('user1');
      registry.check('user1'); // rejected
      registry.check('user2');

      const summary = registry.getSummary();
      expect(summary.totalEntities).toBe(2);
      expect(summary.totalRequests).toBe(4);
      expect(summary.totalRejected).toBe(1);
      expect(summary.rejectionRate).toBe(0.25);
    });

    it('should export metrics as JSON', () => {
      const registry = new security.RateLimiterRegistry({
        maxRequests: 5,
        windowMs: 10000,
      });

      registry.check('user1');
      registry.check('user2');

      const exported = registry.exportMetrics();
      const parsed = JSON.parse(exported);

      expect(parsed.exportedAt).toBeGreaterThan(0);
      expect(parsed.summary).toBeDefined();
      expect(parsed.entities.user1).toBeDefined();
      expect(parsed.entities.user2).toBeDefined();
    });

    it('should reset and clear metrics', () => {
      const registry = new security.RateLimiterRegistry({
        maxRequests: 5,
        windowMs: 10000,
      });

      registry.check('user1');
      registry.check('user2');

      expect(registry.getMetrics('user1')).not.toBeNull();

      registry.resetMetrics('user1');
      expect(registry.getMetrics('user1')).toBeNull();
      expect(registry.getMetrics('user2')).not.toBeNull();

      registry.clearMetrics();
      expect(registry.getAllMetrics().size).toBe(0);
    });
  });

  // ============================================================================
  // Input Sanitization
  // ============================================================================

  describe('Input Sanitization', () => {
    it('should sanitize strings by stripping HTML', () => {
      const input = '<script>alert("xss")</script>Hello';
      const sanitized = security.sanitizeString(input);
      expect(sanitized).toBe('alert("xss")Hello');
    });

    it('should strip control characters', () => {
      const input = 'Hello\x00World\x1F!';
      const sanitized = security.sanitizeString(input);
      expect(sanitized).toBe('HelloWorld!');
    });

    it('should preserve newlines and tabs', () => {
      const input = 'Line1\nLine2\tTabbed';
      const sanitized = security.sanitizeString(input);
      expect(sanitized).toBe('Line1\nLine2\tTabbed');
    });

    it('should enforce max length', () => {
      const input = 'a'.repeat(200);
      const sanitized = security.sanitizeString(input, { maxLength: 100 });
      expect(sanitized.length).toBe(100);
    });

    it('should sanitize identifiers', () => {
      expect(security.sanitizeId('valid_id-123')).toBe('valid_id-123');
      expect(security.sanitizeId('hello world!')).toBe('helloworld');
      expect(security.sanitizeId('<script>')).toBe('script');
    });

    it('should throw on empty identifier after sanitization', () => {
      expect(() => security.sanitizeId('!!!@@@###')).toThrow();
    });

    it('should validate and sanitize JSON', () => {
      const validJson = '{"name": "test", "value": 123}';
      const parsed = security.sanitizeJson<{ name: string; value: number }>(validJson);
      expect(parsed.name).toBe('test');
      expect(parsed.value).toBe(123);
    });

    it('should reject JSON exceeding max size', () => {
      const largeJson = JSON.stringify({ data: 'x'.repeat(10000) });
      expect(() => security.sanitizeJson(largeJson, 10, 100)).toThrow();
    });

    it('should reject deeply nested JSON', () => {
      let nested: any = { value: 1 };
      for (let i = 0; i < 20; i++) {
        nested = { child: nested };
      }
      const deepJson = JSON.stringify(nested);
      expect(() => security.sanitizeJson(deepJson, 5)).toThrow();
    });

    it('should reject invalid JSON', () => {
      expect(() => security.sanitizeJson('not valid json')).toThrow();
    });
  });

  // ============================================================================
  // Secure Secret Storage
  // ============================================================================

  describe('Secure Secret Storage', () => {
    it('should create and expose secret', () => {
      const secret = new security.SecureSecret('my-secret-key');
      const exposed = secret.exposeString();
      expect(exposed).toBe('my-secret-key');
    });

    it('should clear secret from memory', () => {
      const secret = new security.SecureSecret('my-secret-key');
      expect(secret.isValid()).toBe(true);

      secret.clear();
      expect(secret.isValid()).toBe(false);
      expect(() => secret.expose()).toThrow();
    });

    it('should use secret with callback and auto-clear', async () => {
      const secret = new security.SecureSecret('my-key');

      let usedValue: string = '';
      await secret.use((bytes) => {
        usedValue = new TextDecoder().decode(bytes);
      });

      expect(usedValue).toBe('my-key');
      expect(secret.isValid()).toBe(false);
    });

    it('should support binary secrets', () => {
      const binarySecret = new Uint8Array([1, 2, 3, 4, 5]);
      const secret = new security.SecureSecret(binarySecret);

      const exposed = secret.expose();
      expect(exposed).toEqual(binarySecret);
    });

    it('should expire secrets after TTL', async () => {
      const secret = new security.SecureSecret('short-lived', 50);

      expect(secret.isValid()).toBe(true);

      await new Promise((resolve) => setTimeout(resolve, 100));

      expect(secret.isValid()).toBe(false);
      expect(() => secret.expose()).toThrow();
    });
  });

  // ============================================================================
  // Byzantine Resistance Utilities
  // ============================================================================

  describe('Byzantine Resistance Utilities', () => {
    it('should validate BFT parameters (n >= 3f + 1)', () => {
      // n=4, f=1: 4 >= 3(1)+1 = 4 >= 4 ✓
      expect(security.validateBftParams(4, 1).valid).toBe(true);

      // n=7, f=2: 7 >= 3(2)+1 = 7 >= 7 ✓
      expect(security.validateBftParams(7, 2).valid).toBe(true);

      // n=10, f=3: 10 >= 3(3)+1 = 10 >= 10 ✓
      expect(security.validateBftParams(10, 3).valid).toBe(true);

      // n=3, f=1: 3 >= 4 ✗
      expect(security.validateBftParams(3, 1).valid).toBe(false);

      // n=5, f=2: 5 >= 7 ✗
      expect(security.validateBftParams(5, 2).valid).toBe(false);
    });

    it('should calculate max Byzantine failures', () => {
      expect(security.maxByzantineFailures(4)).toBe(1);
      expect(security.maxByzantineFailures(7)).toBe(2);
      expect(security.maxByzantineFailures(10)).toBe(3);
      expect(security.maxByzantineFailures(100)).toBe(33);
    });

    it('should check quorum requirements', () => {
      // 10 nodes, 2 Byzantine, need 6 honest -> 8 honest available
      expect(security.hasQuorum(10, 2, 6)).toBe(true);

      // 10 nodes, 5 Byzantine, need 6 honest -> 5 honest available
      expect(security.hasQuorum(10, 5, 6)).toBe(false);

      // 7 nodes, 2 Byzantine, need 5 honest -> 5 honest available
      expect(security.hasQuorum(7, 2, 5)).toBe(true);
    });
  });

  // ============================================================================
  // Timing Attack Protection
  // ============================================================================

  describe('Timing Attack Protection', () => {
    it('should add random delay', async () => {
      const start = Date.now();
      await security.randomDelay(10, 50);
      const elapsed = Date.now() - start;

      expect(elapsed).toBeGreaterThanOrEqual(10);
      expect(elapsed).toBeLessThan(500); // Large buffer for slow CI/systems
    });

    it('should ensure constant time execution', async () => {
      const fastFn = () => 'fast';

      const start = Date.now();
      const result = await security.constantTime(fastFn, 50);
      const elapsed = Date.now() - start;

      expect(result).toBe('fast');
      expect(elapsed).toBeGreaterThanOrEqual(50);
    });

    it('should not extend time for slow operations', async () => {
      const slowFn = async () => {
        await new Promise((resolve) => setTimeout(resolve, 100));
        return 'slow';
      };

      const start = Date.now();
      const result = await security.constantTime(slowFn, 50);
      const elapsed = Date.now() - start;

      expect(result).toBe('slow');
      // Allow 5ms tolerance for timer accuracy in CI environments
      expect(elapsed).toBeGreaterThanOrEqual(95);
      expect(elapsed).toBeLessThan(200);
    });
  });

  // ============================================================================
  // Security Audit Logging
  // ============================================================================

  describe('Security Audit Logging', () => {
    let auditLog: security.SecurityAuditLog;

    beforeEach(() => {
      auditLog = new security.SecurityAuditLog();
    });

    it('should log security events', () => {
      auditLog.log(
        security.SecurityEventType.RATE_LIMIT_EXCEEDED,
        { userId: 'test-user', requests: 100 },
        'medium',
        'test-user'
      );

      const events = auditLog.getEvents();
      expect(events.length).toBe(1);
      expect(events[0].type).toBe(security.SecurityEventType.RATE_LIMIT_EXCEEDED);
      expect(events[0].entityId).toBe('test-user');
      expect(events[0].severity).toBe('medium');
    });

    it('should filter events by type', () => {
      auditLog.log(security.SecurityEventType.RATE_LIMIT_EXCEEDED, {}, 'medium');
      auditLog.log(security.SecurityEventType.INVALID_INPUT, {}, 'low');
      auditLog.log(security.SecurityEventType.RATE_LIMIT_EXCEEDED, {}, 'high');

      const rateLimitEvents = auditLog.getEvents({
        type: security.SecurityEventType.RATE_LIMIT_EXCEEDED,
      });
      expect(rateLimitEvents.length).toBe(2);
    });

    it('should filter events by severity', () => {
      auditLog.log(security.SecurityEventType.INVALID_INPUT, {}, 'low');
      auditLog.log(security.SecurityEventType.INVALID_INPUT, {}, 'critical');
      auditLog.log(security.SecurityEventType.INVALID_INPUT, {}, 'low');

      const criticalEvents = auditLog.getEvents({ severity: 'critical' });
      expect(criticalEvents.length).toBe(1);
    });

    it('should call event handlers', () => {
      const handler = vi.fn();
      auditLog.onEvent(handler);

      auditLog.log(security.SecurityEventType.SUSPICIOUS_ACTIVITY, { ip: '1.2.3.4' }, 'high');

      expect(handler).toHaveBeenCalledTimes(1);
      expect(handler.mock.calls[0][0].type).toBe(security.SecurityEventType.SUSPICIOUS_ACTIVITY);
    });

    it('should limit max events', () => {
      const smallLog = new security.SecurityAuditLog(5);

      for (let i = 0; i < 10; i++) {
        smallLog.log(security.SecurityEventType.INVALID_INPUT, { index: i }, 'low');
      }

      const events = smallLog.getEvents();
      expect(events.length).toBe(5);
      // Should have the latest events (indices 5-9)
      expect((events[0].details as any).index).toBe(5);
      expect((events[4].details as any).index).toBe(9);
    });

    it('should clear all events', () => {
      auditLog.log(security.SecurityEventType.INVALID_INPUT, {}, 'low');
      auditLog.log(security.SecurityEventType.INVALID_INPUT, {}, 'low');

      expect(auditLog.getEvents().length).toBe(2);

      auditLog.clear();
      expect(auditLog.getEvents().length).toBe(0);
    });
  });

  // ============================================================================
  // Security Error Types
  // ============================================================================

  describe('Security Error Types', () => {
    it('should create SecurityError with correct properties', () => {
      const error = new security.SecurityError(
        'Rate limit exceeded',
        security.SecurityErrorCode.RATE_LIMITED,
        { userId: 'test' }
      );

      expect(error.name).toBe('SecurityError');
      expect(error.message).toBe('Rate limit exceeded');
      expect(error.code).toBe(security.SecurityErrorCode.RATE_LIMITED);
      expect(error.context.userId).toBe('test');
    });
  });

  // ============================================================================
  // Namespace Export
  // ============================================================================

  describe('Security Namespace', () => {
    it('should export all functions via security namespace', () => {
      expect(typeof security.security.randomBytes).toBe('function');
      expect(typeof security.security.randomFloat).toBe('function');
      expect(typeof security.security.randomInt).toBe('function');
      expect(typeof security.security.uuid).toBe('function');
      expect(typeof security.security.shuffle).toBe('function');
      expect(typeof security.security.hash).toBe('function');
      expect(typeof security.security.hashHex).toBe('function');
      expect(typeof security.security.hmac).toBe('function');
      expect(typeof security.security.verifyHmac).toBe('function');
      expect(typeof security.security.constantTimeEqual).toBe('function');
      expect(typeof security.security.createRateLimiter).toBe('function');
      expect(typeof security.security.checkRateLimit).toBe('function');
      expect(typeof security.security.sanitizeString).toBe('function');
      expect(typeof security.security.sanitizeId).toBe('function');
      expect(typeof security.security.sanitizeJson).toBe('function');
      expect(typeof security.security.validateBftParams).toBe('function');
      expect(typeof security.security.maxByzantineFailures).toBe('function');
      expect(typeof security.security.hasQuorum).toBe('function');
      expect(typeof security.security.randomDelay).toBe('function');
      expect(typeof security.security.constantTime).toBe('function');
    });
  });
});
