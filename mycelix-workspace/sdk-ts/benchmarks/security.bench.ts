/**
 * Security Module Benchmarks
 *
 * Performance benchmarks for cryptographic operations, rate limiting,
 * and other security utilities.
 */

import { describe, bench, beforeAll } from 'vitest';
import * as security from '../src/security/index.js';
import * as matl from '../src/matl/index.js';
import * as fl from '../src/fl/index.js';

// ============================================================================
// Secure Random Generation Benchmarks
// ============================================================================

describe('Secure Random Generation', () => {
  bench('secureRandomBytes(16)', () => {
    security.secureRandomBytes(16);
  });

  bench('secureRandomBytes(32)', () => {
    security.secureRandomBytes(32);
  });

  bench('secureRandomBytes(64)', () => {
    security.secureRandomBytes(64);
  });

  bench('secureRandomBytes(256)', () => {
    security.secureRandomBytes(256);
  });

  bench('secureRandomFloat()', () => {
    security.secureRandomFloat();
  });

  bench('secureRandomInt(0, 1000)', () => {
    security.secureRandomInt(0, 1000);
  });

  bench('secureUUID()', () => {
    security.secureUUID();
  });

  bench('secureShufffle([1..100])', () => {
    const arr = Array.from({ length: 100 }, (_, i) => i);
    security.secureShufffle(arr);
  });
});

// ============================================================================
// Cryptographic Hashing Benchmarks
// ============================================================================

describe('Cryptographic Hashing', () => {
  const testData = 'x'.repeat(1000);
  const largeData = 'x'.repeat(100000);

  bench('hash(1KB) SHA-256', async () => {
    await security.hash(testData, 'SHA-256');
  });

  bench('hash(100KB) SHA-256', async () => {
    await security.hash(largeData, 'SHA-256');
  });

  bench('hash(1KB) SHA-512', async () => {
    await security.hash(testData, 'SHA-512');
  });

  bench('hashHex(1KB)', async () => {
    await security.hashHex(testData);
  });
});

// ============================================================================
// HMAC Benchmarks
// ============================================================================

describe('HMAC Operations', () => {
  let key: Uint8Array;
  const testMessage = 'This is a test message for HMAC benchmarking';
  let existingMac: Uint8Array;

  beforeAll(async () => {
    key = security.secureRandomBytes(32);
    existingMac = await security.hmac(key, testMessage);
  });

  bench('hmac(32-byte key, message)', async () => {
    await security.hmac(key, testMessage);
  });

  bench('verifyHmac(valid)', async () => {
    await security.verifyHmac(key, testMessage, existingMac);
  });

  bench('verifyHmac(invalid)', async () => {
    const invalidMac = security.secureRandomBytes(32);
    await security.verifyHmac(key, testMessage, invalidMac);
  });
});

// ============================================================================
// Constant-Time Comparison Benchmarks
// ============================================================================

describe('Constant-Time Comparison', () => {
  const a = security.secureRandomBytes(32);
  const b = new Uint8Array(a); // Copy
  const c = security.secureRandomBytes(32); // Different

  bench('constantTimeEqual(32 bytes, equal)', () => {
    security.constantTimeEqual(a, b);
  });

  bench('constantTimeEqual(32 bytes, different)', () => {
    security.constantTimeEqual(a, c);
  });

  bench('constantTimeEqual(256 bytes)', () => {
    const large1 = security.secureRandomBytes(256);
    const large2 = new Uint8Array(large1);
    security.constantTimeEqual(large1, large2);
  });
});

// ============================================================================
// Rate Limiting Benchmarks
// ============================================================================

describe('Rate Limiting', () => {
  bench('createRateLimiter()', () => {
    security.createRateLimiter('test', {
      maxRequests: 100,
      windowMs: 60000,
    });
  });

  bench('checkRateLimit()', () => {
    let state = security.createRateLimiter('test', {
      maxRequests: 1000,
      windowMs: 60000,
    });
    const result = security.checkRateLimit(state);
    state = result.state;
  });

  bench('RateLimiterRegistry.check()', () => {
    const registry = new security.RateLimiterRegistry({
      maxRequests: 1000,
      windowMs: 60000,
    });
    registry.check('user-1');
  });

  bench('RateLimiterRegistry.check(100 users)', () => {
    const registry = new security.RateLimiterRegistry({
      maxRequests: 1000,
      windowMs: 60000,
    });
    for (let i = 0; i < 100; i++) {
      registry.check(`user-${i}`);
    }
  });
});

// ============================================================================
// Input Sanitization Benchmarks
// ============================================================================

describe('Input Sanitization', () => {
  const cleanString = 'This is a clean test string';
  const dirtyString = '<script>alert("xss")</script>Hello\x00World\x1F!';
  const jsonData = JSON.stringify({
    name: 'test',
    value: 123,
    nested: { a: 1, b: 2 },
  });

  bench('sanitizeString(clean)', () => {
    security.sanitizeString(cleanString);
  });

  bench('sanitizeString(dirty)', () => {
    security.sanitizeString(dirtyString);
  });

  bench('sanitizeId()', () => {
    security.sanitizeId('valid_id-123');
  });

  bench('sanitizeJson()', () => {
    security.sanitizeJson(jsonData);
  });
});

// ============================================================================
// Byzantine Fault Tolerance Benchmarks
// ============================================================================

describe('Byzantine Fault Tolerance', () => {
  bench('validateBftParams()', () => {
    security.validateBftParams(10, 3);
  });

  bench('maxByzantineFailures()', () => {
    security.maxByzantineFailures(100);
  });

  bench('hasQuorum()', () => {
    security.hasQuorum(10, 3, 5);
  });
});

// ============================================================================
// Timing Attack Protection Benchmarks
// ============================================================================

describe('Timing Attack Protection', () => {
  bench('randomDelay(1, 5)', async () => {
    await security.randomDelay(1, 5);
  });

  bench('constantTime(fast fn, 10ms)', async () => {
    await security.constantTime(() => 'fast', 10);
  });
});

// ============================================================================
// Secure Secret Storage Benchmarks
// ============================================================================

describe('Secure Secret Storage', () => {
  bench('SecureSecret creation', () => {
    new security.SecureSecret('my-secret-key-value');
  });

  bench('SecureSecret.expose()', () => {
    const secret = new security.SecureSecret('my-secret-key-value');
    secret.expose();
  });

  bench('SecureSecret.exposeString()', () => {
    const secret = new security.SecureSecret('my-secret-key-value');
    secret.exposeString();
  });

  bench('SecureSecret.clear()', () => {
    const secret = new security.SecureSecret('my-secret-key-value');
    secret.clear();
  });
});

// ============================================================================
// Audit Logging Benchmarks
// ============================================================================

describe('Security Audit Logging', () => {
  bench('SecurityAuditLog.log()', () => {
    const log = new security.SecurityAuditLog();
    log.log(
      security.SecurityEventType.RATE_LIMIT_EXCEEDED,
      { userId: 'test' },
      'medium',
      'test-entity'
    );
  });

  bench('SecurityAuditLog.getEvents(1000 events)', () => {
    const log = new security.SecurityAuditLog(2000);
    for (let i = 0; i < 1000; i++) {
      log.log(security.SecurityEventType.INVALID_INPUT, { index: i }, 'low');
    }
    log.getEvents();
  });

  bench('SecurityAuditLog.getEvents(filtered)', () => {
    const log = new security.SecurityAuditLog(2000);
    for (let i = 0; i < 1000; i++) {
      const type =
        i % 2 === 0
          ? security.SecurityEventType.INVALID_INPUT
          : security.SecurityEventType.RATE_LIMIT_EXCEEDED;
      log.log(type, { index: i }, 'low');
    }
    log.getEvents({ type: security.SecurityEventType.RATE_LIMIT_EXCEEDED });
  });
});

// ============================================================================
// Integration Benchmarks (Security + Other Modules)
// ============================================================================

describe('Security Integration with MATL', () => {
  bench('PoGQ hash for integrity', async () => {
    const pogq = matl.createPoGQ(0.85, 0.9, 0.1);
    const data = JSON.stringify(pogq);
    await security.hashHex(data);
  });

  bench('Composite score with HMAC', async () => {
    const pogq = matl.createPoGQ(0.85, 0.9, 0.1);
    const reputation = matl.createReputation('agent');
    const composite = matl.calculateComposite(pogq, reputation);
    const key = security.secureRandomBytes(32);
    await security.hmac(key, JSON.stringify(composite));
  });

  bench('Byzantine check + audit log', () => {
    const log = new security.SecurityAuditLog();
    const pogq = matl.createPoGQ(0.2, 0.3, 0.9);
    if (matl.isByzantine(pogq)) {
      log.log(
        security.SecurityEventType.SUSPICIOUS_ACTIVITY,
        { pogq },
        'high'
      );
    }
  });
});

describe('Security Integration with FL', () => {
  bench('Gradient serialization + hash', async () => {
    const gradients = new Float64Array(100).map(() => Math.random() - 0.5);
    const serialized = fl.serializeGradients(gradients);
    await security.hash(serialized);
  });

  bench('Gradient HMAC signature', async () => {
    const gradients = new Float64Array(100).map(() => Math.random() - 0.5);
    const serialized = fl.serializeGradients(gradients);
    const key = security.secureRandomBytes(32);
    await security.hmac(key, serialized);
  });

  bench('Full gradient verification pipeline', async () => {
    // Generate gradients
    const gradients = new Float64Array(100).map(() => Math.random() - 0.5);
    const serialized = fl.serializeGradients(gradients);

    // Sign
    const key = security.secureRandomBytes(32);
    const signature = await security.hmac(key, serialized);

    // Verify
    await security.verifyHmac(key, serialized, signature);
  });
});

// ============================================================================
// End-to-End Security Pipeline Benchmark
// ============================================================================

describe('End-to-End Security Pipeline', () => {
  bench('Full participant verification pipeline', async () => {
    // 1. Generate secure ID
    const participantId = security.secureUUID();

    // 2. Rate limit check
    const registry = new security.RateLimiterRegistry({
      maxRequests: 100,
      windowMs: 60000,
    });
    registry.check(participantId);

    // 3. Create and verify PoGQ
    const pogq = matl.createPoGQ(0.85, 0.9, 0.1);

    // 4. Sign PoGQ
    const key = security.secureRandomBytes(32);
    const pogqData = JSON.stringify(pogq);
    const signature = await security.hmac(key, pogqData);

    // 5. Verify signature
    await security.verifyHmac(key, pogqData, signature);

    // 6. Byzantine check
    matl.isByzantine(pogq);

    // 7. Calculate composite score
    const reputation = matl.createReputation(participantId);
    matl.calculateComposite(pogq, reputation);
  });
});
