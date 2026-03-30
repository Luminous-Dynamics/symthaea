// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * PQC Signature Service Tests
 *
 * Tests key generation, registration, signing, verification, replay prevention,
 * key revocation, and the SignedMessageBridge.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  PQCKeyManager,
  PQCSignatureService,
  SignedMessageBridge,
  getPQCKeyManager,
  getPQCSignatureService,
  getSignedMessageBridge,
  resetPQCSingletons,
  getRecommendedAlgorithm,
  isAlgorithmSecure,
  estimateSignatureSize,
  type PQCAlgorithm,
  type PQCKeyPair,
} from '../../src/bridge/pqc-signatures.js';

// ============================================================================
// PQCKeyManager
// ============================================================================

describe('PQCKeyManager', () => {
  let km: PQCKeyManager;

  beforeEach(() => {
    km = new PQCKeyManager();
  });

  describe('Key Generation', () => {
    it('should generate a Dilithium3 key pair by default', () => {
      const kp = km.generateKeyPair();
      expect(kp.algorithm).toBe('Dilithium3');
      expect(kp.publicKey).toBeInstanceOf(Uint8Array);
      expect(kp.privateKey).toBeInstanceOf(Uint8Array);
      expect(kp.publicKey.length).toBe(1952);
      expect(kp.privateKey.length).toBe(4032);
      expect(kp.keyId).toMatch(/^pqc-dilithium3-/);
    });

    it('should generate keys for each algorithm', () => {
      const algorithms: PQCAlgorithm[] = ['Dilithium3', 'Dilithium5', 'Falcon512', 'SPHINCS+', 'Hybrid-Ed25519-Dilithium3'];
      for (const algo of algorithms) {
        const kp = km.generateKeyPair(algo);
        expect(kp.algorithm).toBe(algo);
        expect(kp.publicKey.length).toBeGreaterThan(0);
        expect(kp.privateKey.length).toBeGreaterThan(0);
      }
    });

    it('should set ownerDid when provided', () => {
      const kp = km.generateKeyPair('Dilithium3', 'did:test:owner');
      expect(kp.ownerDid).toBe('did:test:owner');
    });

    it('should set expiration with TTL', () => {
      const kp = km.generateKeyPair('Dilithium3', undefined, 60000);
      expect(kp.expiresAt).toBeDefined();
      expect(kp.expiresAt!).toBeGreaterThan(kp.createdAt);
    });

    it('should not set expiration without TTL', () => {
      const kp = km.generateKeyPair();
      expect(kp.expiresAt).toBeUndefined();
    });
  });

  describe('Key Registration', () => {
    it('should register and retrieve a key', () => {
      const kp = km.generateKeyPair('Dilithium3', 'did:test:1');
      const registered = km.registerKey(kp.keyId, kp.publicKey, kp.algorithm, 'did:test:1', 'governance');

      expect(registered.keyId).toBe(kp.keyId);
      expect(registered.revoked).toBe(false);
      expect(km.getKey(kp.keyId)).not.toBeNull();
      expect(km.keyCount).toBe(1);
    });

    it('should reject duplicate key registration', () => {
      const kp = km.generateKeyPair('Dilithium3', 'did:test:1');
      km.registerKey(kp.keyId, kp.publicKey, kp.algorithm, 'did:test:1', 'governance');

      expect(() =>
        km.registerKey(kp.keyId, kp.publicKey, kp.algorithm, 'did:test:1', 'governance')
      ).toThrow();
    });

    it('should index keys by DID', () => {
      const kp1 = km.generateKeyPair('Dilithium3', 'did:test:1');
      const kp2 = km.generateKeyPair('Dilithium5', 'did:test:1');
      km.registerKey(kp1.keyId, kp1.publicKey, kp1.algorithm, 'did:test:1', 'governance');
      km.registerKey(kp2.keyId, kp2.publicKey, kp2.algorithm, 'did:test:1', 'finance');

      const keys = km.getKeysByDid('did:test:1');
      expect(keys).toHaveLength(2);
    });

    it('should index keys by hApp', () => {
      const kp1 = km.generateKeyPair('Dilithium3', 'did:test:1');
      const kp2 = km.generateKeyPair('Dilithium3', 'did:test:2');
      km.registerKey(kp1.keyId, kp1.publicKey, kp1.algorithm, 'did:test:1', 'governance');
      km.registerKey(kp2.keyId, kp2.publicKey, kp2.algorithm, 'did:test:2', 'governance');

      expect(km.getKeysByHapp('governance')).toHaveLength(2);
      expect(km.getKeysByHapp('finance')).toHaveLength(0);
    });

    it('should return null for missing key', () => {
      expect(km.getKey('nonexistent')).toBeNull();
    });

    it('should return empty for unknown DID', () => {
      expect(km.getKeysByDid('did:unknown')).toHaveLength(0);
    });
  });

  describe('Key Validity', () => {
    it('should report valid key as valid', () => {
      const kp = km.generateKeyPair();
      km.registerKey(kp.keyId, kp.publicKey, kp.algorithm, 'did:test:1', 'governance');
      expect(km.isKeyValid(kp.keyId)).toBe(true);
    });

    it('should report revoked key as invalid', () => {
      const kp = km.generateKeyPair();
      km.registerKey(kp.keyId, kp.publicKey, kp.algorithm, 'did:test:1', 'governance');
      km.revokeKey(kp.keyId, 'compromised');
      expect(km.isKeyValid(kp.keyId)).toBe(false);
    });

    it('should report expired key as invalid', () => {
      const kp = km.generateKeyPair('Dilithium3', undefined, 1); // 1ms TTL
      km.registerKey(kp.keyId, kp.publicKey, kp.algorithm, 'did:test:1', 'governance', kp.expiresAt);
      // Wait for expiry
      const start = Date.now();
      while (Date.now() - start < 5) { /* busy wait */ }
      expect(km.isKeyValid(kp.keyId)).toBe(false);
    });

    it('should report unknown key as invalid', () => {
      expect(km.isKeyValid('nonexistent')).toBe(false);
    });
  });

  describe('Key Revocation', () => {
    it('should revoke a key with reason', () => {
      const kp = km.generateKeyPair();
      km.registerKey(kp.keyId, kp.publicKey, kp.algorithm, 'did:test:1', 'governance');

      const result = km.revokeKey(kp.keyId, 'compromised');
      expect(result).toBe(true);

      const key = km.getKey(kp.keyId);
      expect(key?.revoked).toBe(true);
      expect(key?.revokedReason).toBe('compromised');
      expect(key?.revokedAt).toBeDefined();
    });

    it('should return false for unknown key revocation', () => {
      expect(km.revokeKey('nonexistent')).toBe(false);
    });
  });

  describe('Cleanup', () => {
    it('should remove expired keys', () => {
      const kp = km.generateKeyPair('Dilithium3', undefined, 1); // 1ms TTL
      km.registerKey(kp.keyId, kp.publicKey, kp.algorithm, 'did:test:1', 'governance', kp.expiresAt);

      const start = Date.now();
      while (Date.now() - start < 5) { /* busy wait */ }

      const removed = km.cleanup();
      expect(removed).toBe(1);
      expect(km.keyCount).toBe(0);
    });

    it('should clear all keys', () => {
      const kp = km.generateKeyPair();
      km.registerKey(kp.keyId, kp.publicKey, kp.algorithm, 'did:test:1', 'governance');
      km.clear();
      expect(km.keyCount).toBe(0);
    });
  });
});

// ============================================================================
// PQCSignatureService
// ============================================================================

describe('PQCSignatureService', () => {
  let service: PQCSignatureService;
  let km: PQCKeyManager;

  beforeEach(() => {
    km = new PQCKeyManager();
    service = new PQCSignatureService(km);
  });

  it('should sign and verify string data', async () => {
    const kp = km.generateKeyPair('Dilithium3', 'did:test:signer');
    km.registerKey(kp.keyId, kp.publicKey, kp.algorithm, 'did:test:signer', 'governance');

    const sig = await service.sign('Hello, world!', kp, 'did:test:signer');
    expect(sig.signature).toBeInstanceOf(Uint8Array);
    expect(sig.algorithm).toBe('Dilithium3');
    expect(sig.keyId).toBe(kp.keyId);
    expect(sig.contentHash).toBeTruthy();
    expect(sig.nonce).toBeTruthy();

    const result = await service.verify('Hello, world!', sig);
    expect(result.valid).toBe(true);
    expect(result.signerDid).toBe('did:test:signer');
  });

  it('should sign and verify Uint8Array data', async () => {
    const kp = km.generateKeyPair('Dilithium3', 'did:test:signer');
    km.registerKey(kp.keyId, kp.publicKey, kp.algorithm, 'did:test:signer', 'governance');

    const data = new Uint8Array([1, 2, 3, 4, 5]);
    const sig = await service.sign(data, kp, 'did:test:signer');
    const result = await service.verify(data, sig);
    expect(result.valid).toBe(true);
  });

  it('should reject unregistered key', async () => {
    const kp = km.generateKeyPair('Dilithium3', 'did:test:signer');
    // NOT registering the key

    const sig = await service.sign('data', kp, 'did:test:signer');
    const result = await service.verify('data', sig);
    expect(result.valid).toBe(false);
    expect(result.failureReason).toContain('not registered');
  });

  it('should reject revoked key', async () => {
    const kp = km.generateKeyPair('Dilithium3', 'did:test:signer');
    km.registerKey(kp.keyId, kp.publicKey, kp.algorithm, 'did:test:signer', 'governance');

    const sig = await service.sign('data', kp, 'did:test:signer');

    km.revokeKey(kp.keyId, 'compromised');

    const result = await service.verify('data', sig);
    expect(result.valid).toBe(false);
    expect(result.failureReason).toContain('revoked');
  });

  it('should reject expired key', async () => {
    const kp = km.generateKeyPair('Dilithium3', 'did:test:signer', 1); // 1ms TTL
    km.registerKey(kp.keyId, kp.publicKey, kp.algorithm, 'did:test:signer', 'governance', kp.expiresAt);

    const sig = await service.sign('data', kp, 'did:test:signer');

    // Wait for expiry
    await new Promise(resolve => setTimeout(resolve, 10));

    const result = await service.verify('data', sig);
    expect(result.valid).toBe(false);
    expect(result.failureReason).toContain('expired');
  });

  it('should reject algorithm mismatch', async () => {
    const kp = km.generateKeyPair('Dilithium3', 'did:test:signer');
    km.registerKey(kp.keyId, kp.publicKey, kp.algorithm, 'did:test:signer', 'governance');

    const sig = await service.sign('data', kp, 'did:test:signer');
    // Tamper with algorithm
    (sig as any).algorithm = 'Dilithium5';

    const result = await service.verify('data', sig);
    expect(result.valid).toBe(false);
    expect(result.failureReason).toContain('Algorithm mismatch');
  });

  it('should reject DID mismatch', async () => {
    const kp = km.generateKeyPair('Dilithium3', 'did:test:signer');
    km.registerKey(kp.keyId, kp.publicKey, kp.algorithm, 'did:test:signer', 'governance');

    const sig = await service.sign('data', kp, 'did:test:signer');
    // Tamper with signer DID
    (sig as any).signerDid = 'did:test:impostor';

    const result = await service.verify('data', sig);
    expect(result.valid).toBe(false);
    expect(result.failureReason).toContain('does not match');
  });

  it('should reject replay (duplicate nonce)', async () => {
    const kp = km.generateKeyPair('Dilithium3', 'did:test:signer');
    km.registerKey(kp.keyId, kp.publicKey, kp.algorithm, 'did:test:signer', 'governance');

    const sig = await service.sign('data', kp, 'did:test:signer');

    // First verify should succeed
    const result1 = await service.verify('data', sig);
    expect(result1.valid).toBe(true);

    // Second verify with same nonce should fail
    const result2 = await service.verify('data', sig);
    expect(result2.valid).toBe(false);
    expect(result2.failureReason).toContain('Nonce already used');
  });

  it('should reject tampered content', async () => {
    const kp = km.generateKeyPair('Dilithium3', 'did:test:signer');
    km.registerKey(kp.keyId, kp.publicKey, kp.algorithm, 'did:test:signer', 'governance');

    const sig = await service.sign('original data', kp, 'did:test:signer');
    const result = await service.verify('tampered data', sig);
    expect(result.valid).toBe(false);
    expect(result.failureReason).toContain('Content hash mismatch');
  });

  it('should reject old signatures beyond nonce window', async () => {
    const kp = km.generateKeyPair('Dilithium3', 'did:test:signer');
    km.registerKey(kp.keyId, kp.publicKey, kp.algorithm, 'did:test:signer', 'governance');

    service.setNonceWindow(50); // 50ms window

    const sig = await service.sign('data', kp, 'did:test:signer');

    // Wait beyond nonce window
    await new Promise(resolve => setTimeout(resolve, 100));

    const result = await service.verify('data', sig);
    expect(result.valid).toBe(false);
    expect(result.failureReason).toContain('too old');
  });

  it('should add warning for Falcon512', async () => {
    const kp = km.generateKeyPair('Falcon512', 'did:test:signer');
    km.registerKey(kp.keyId, kp.publicKey, kp.algorithm, 'did:test:signer', 'governance');

    const sig = await service.sign('data', kp, 'did:test:signer');
    const result = await service.verify('data', sig);
    expect(result.valid).toBe(true);
    expect(result.warnings).toBeDefined();
    expect(result.warnings!.some(w => w.includes('Falcon512'))).toBe(true);
  });

  it('should handle hybrid Ed25519-Dilithium3 signing', async () => {
    const kp = km.generateKeyPair('Hybrid-Ed25519-Dilithium3', 'did:test:signer');
    km.registerKey(kp.keyId, kp.publicKey, kp.algorithm, 'did:test:signer', 'governance');

    const sig = await service.sign('data', kp, 'did:test:signer');
    // Hybrid signatures are larger (combined)
    expect(sig.signature.length).toBeGreaterThan(32);
    expect(sig.algorithm).toBe('Hybrid-Ed25519-Dilithium3');

    const result = await service.verify('data', sig);
    expect(result.valid).toBe(true);
  });

  it('should cleanup nonces when threshold exceeded', () => {
    // Manually set up large nonce set
    const usedNonces = (service as any).usedNonces as Set<string>;
    for (let i = 0; i < 10001; i++) {
      usedNonces.add(`nonce-${i}`);
    }
    service.cleanupNonces();
    expect(usedNonces.size).toBe(0);
  });
});

// ============================================================================
// SignedMessageBridge
// ============================================================================

describe('SignedMessageBridge', () => {
  let bridge: SignedMessageBridge;

  beforeEach(() => {
    bridge = new SignedMessageBridge();
  });

  it('should generate and register a key pair for a hApp', () => {
    const kp = bridge.generateHappKeyPair('governance', 'did:test:gov');
    expect(kp.algorithm).toBe('Dilithium3');
    expect(bridge.getRegisteredHapps()).toContain('governance');
  });

  it('should sign and verify a message', async () => {
    bridge.generateHappKeyPair('governance', 'did:test:gov');

    const message = {
      id: 'msg-1',
      type: 'reputation_update' as const,
      sourceHapp: 'governance' as const,
      targetHapp: 'finance' as const,
      payload: { score: 0.9 },
      timestamp: Date.now(),
    };

    const signed = await bridge.signMessage(message, 'did:test:gov');
    expect(signed.signature).toBeDefined();
    expect(signed.signerDid).toBe('did:test:gov');

    const result = await bridge.verifyMessage(signed);
    expect(result.valid).toBe(true);
  });

  it('should reject signing for unregistered hApp', async () => {
    const message = {
      id: 'msg-1',
      type: 'reputation_update' as const,
      sourceHapp: 'finance' as const,
      targetHapp: 'governance' as const,
      payload: {},
      timestamp: Date.now(),
    };

    await expect(bridge.signMessage(message, 'did:test:fin')).rejects.toThrow();
  });

  it('should support custom algorithm', () => {
    const kp = bridge.generateHappKeyPair('justice', 'did:test:justice', 'Dilithium5');
    expect(kp.algorithm).toBe('Dilithium5');
  });
});

// ============================================================================
// Singletons
// ============================================================================

describe('PQC Singletons', () => {
  afterEach(() => {
    resetPQCSingletons();
  });

  it('should return same instances', () => {
    expect(getPQCKeyManager()).toBe(getPQCKeyManager());
    expect(getPQCSignatureService()).toBe(getPQCSignatureService());
    expect(getSignedMessageBridge()).toBe(getSignedMessageBridge());
  });

  it('should reset all singletons', () => {
    const km1 = getPQCKeyManager();
    const ss1 = getPQCSignatureService();
    const mb1 = getSignedMessageBridge();

    resetPQCSingletons();

    expect(getPQCKeyManager()).not.toBe(km1);
    expect(getPQCSignatureService()).not.toBe(ss1);
    expect(getSignedMessageBridge()).not.toBe(mb1);
  });
});

// ============================================================================
// Utility Functions
// ============================================================================

describe('PQC Utility Functions', () => {
  it('getRecommendedAlgorithm returns correct algorithms', () => {
    expect(getRecommendedAlgorithm('standard')).toBe('ML-DSA-65');
    expect(getRecommendedAlgorithm('high')).toBe('ML-DSA-87');
    expect(getRecommendedAlgorithm('maximum')).toBe('SLH-DSA-SHA2-128s');
  });

  it('isAlgorithmSecure returns true for all supported algorithms', () => {
    const algos: PQCAlgorithm[] = ['Dilithium3', 'Dilithium5', 'Falcon512', 'SPHINCS+', 'Hybrid-Ed25519-Dilithium3'];
    for (const algo of algos) {
      expect(isAlgorithmSecure(algo)).toBe(true);
    }
  });

  it('estimateSignatureSize returns FIPS-standard sizes', () => {
    expect(estimateSignatureSize('Dilithium3')).toBe(3309);
    expect(estimateSignatureSize('Dilithium5')).toBe(4627);
    expect(estimateSignatureSize('Falcon512')).toBe(3309);
    expect(estimateSignatureSize('SPHINCS+')).toBe(7856);
    expect(estimateSignatureSize('Hybrid-Ed25519-Dilithium3')).toBe(64 + 3309);
  });
});
