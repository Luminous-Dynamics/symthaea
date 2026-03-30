// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * PQC Signature Integration Tests
 *
 * Tests for Post-Quantum Cryptographic signatures in cross-hApp messaging.
 */

import { describe, it, expect, beforeEach } from 'vitest';
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
  type PQCKeyPair,
  type PQCAlgorithm,
  type SignatureStrength,
} from '../../src/bridge/pqc-signatures.js';
import type { CrossHappMessage, HappId } from '../../src/bridge/cross-happ.js';

// ============================================================================
// Test Fixtures
// ============================================================================

function createTestMessage(overrides: Partial<CrossHappMessage> = {}): CrossHappMessage {
  return {
    id: `msg-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    type: 'reputation_query',
    sourceHapp: 'governance',
    targetHapp: 'identity',
    payload: { subjectDid: 'did:mycelix:alice', contextHapps: ['governance', 'finance'] },
    timestamp: Date.now(),
    ...overrides,
  };
}

// ============================================================================
// PQCKeyManager Tests
// ============================================================================

describe('Integration: PQCKeyManager', () => {
  let keyManager: PQCKeyManager;

  beforeEach(() => {
    keyManager = new PQCKeyManager();
  });

  describe('generateKeyPair', () => {
    it('should generate a Dilithium3 key pair by default', () => {
      const keyPair = keyManager.generateKeyPair();

      expect(keyPair).toBeDefined();
      expect(keyPair.algorithm).toBe('Dilithium3');
      expect(keyPair.publicKey).toBeInstanceOf(Uint8Array);
      expect(keyPair.privateKey).toBeInstanceOf(Uint8Array);
      expect(keyPair.keyId).toMatch(/^pqc-dilithium3-/);
      expect(keyPair.createdAt).toBeLessThanOrEqual(Date.now());
    });

    it('should generate key pairs for all supported algorithms', () => {
      const algorithms: PQCAlgorithm[] = [
        'Dilithium3',
        'Dilithium5',
        'Falcon512',
        'SPHINCS+',
        'Hybrid-Ed25519-Dilithium3',
      ];

      for (const algorithm of algorithms) {
        const keyPair = keyManager.generateKeyPair(algorithm);
        expect(keyPair.algorithm).toBe(algorithm);
        expect(keyPair.publicKey.length).toBeGreaterThan(0);
        expect(keyPair.privateKey.length).toBeGreaterThan(0);
      }
    });

    it('should set owner DID when provided', () => {
      const keyPair = keyManager.generateKeyPair('Dilithium3', 'did:mycelix:alice');
      expect(keyPair.ownerDid).toBe('did:mycelix:alice');
    });

    it('should set expiration when TTL provided', () => {
      const ttlMs = 3600000; // 1 hour
      const keyPair = keyManager.generateKeyPair('Dilithium3', undefined, ttlMs);

      expect(keyPair.expiresAt).toBeDefined();
      expect(keyPair.expiresAt).toBeGreaterThan(Date.now());
      expect(keyPair.expiresAt).toBeLessThanOrEqual(Date.now() + ttlMs + 1000);
    });

    it('should generate unique key IDs', () => {
      const keyPair1 = keyManager.generateKeyPair();
      const keyPair2 = keyManager.generateKeyPair();

      expect(keyPair1.keyId).not.toBe(keyPair2.keyId);
    });
  });

  describe('registerKey', () => {
    it('should register a public key', () => {
      const keyPair = keyManager.generateKeyPair('Dilithium3', 'did:mycelix:alice');

      const registered = keyManager.registerKey(
        keyPair.keyId,
        keyPair.publicKey,
        keyPair.algorithm,
        'did:mycelix:alice',
        'governance'
      );

      expect(registered.keyId).toBe(keyPair.keyId);
      expect(registered.ownerDid).toBe('did:mycelix:alice');
      expect(registered.happId).toBe('governance');
      expect(registered.revoked).toBe(false);
    });

    it('should throw when registering duplicate key ID', () => {
      const keyPair = keyManager.generateKeyPair('Dilithium3', 'did:mycelix:alice');

      keyManager.registerKey(
        keyPair.keyId,
        keyPair.publicKey,
        keyPair.algorithm,
        'did:mycelix:alice',
        'governance'
      );

      expect(() => {
        keyManager.registerKey(
          keyPair.keyId,
          keyPair.publicKey,
          keyPair.algorithm,
          'did:mycelix:alice',
          'governance'
        );
      }).toThrow(/already registered/);
    });
  });

  describe('getKey', () => {
    it('should retrieve a registered key', () => {
      const keyPair = keyManager.generateKeyPair('Dilithium3', 'did:mycelix:alice');
      keyManager.registerKey(
        keyPair.keyId,
        keyPair.publicKey,
        keyPair.algorithm,
        'did:mycelix:alice',
        'governance'
      );

      const retrieved = keyManager.getKey(keyPair.keyId);

      expect(retrieved).toBeDefined();
      expect(retrieved!.keyId).toBe(keyPair.keyId);
    });

    it('should return null for unregistered key', () => {
      const result = keyManager.getKey('non-existent-key');
      expect(result).toBeNull();
    });
  });

  describe('getKeysByDid', () => {
    it('should retrieve all keys for a DID', () => {
      const keyPair1 = keyManager.generateKeyPair('Dilithium3', 'did:mycelix:alice');
      const keyPair2 = keyManager.generateKeyPair('Dilithium5', 'did:mycelix:alice');
      const keyPair3 = keyManager.generateKeyPair('Dilithium3', 'did:mycelix:bob');

      keyManager.registerKey(
        keyPair1.keyId,
        keyPair1.publicKey,
        keyPair1.algorithm,
        'did:mycelix:alice',
        'governance'
      );
      keyManager.registerKey(
        keyPair2.keyId,
        keyPair2.publicKey,
        keyPair2.algorithm,
        'did:mycelix:alice',
        'finance'
      );
      keyManager.registerKey(
        keyPair3.keyId,
        keyPair3.publicKey,
        keyPair3.algorithm,
        'did:mycelix:bob',
        'governance'
      );

      const aliceKeys = keyManager.getKeysByDid('did:mycelix:alice');

      expect(aliceKeys.length).toBe(2);
      expect(aliceKeys.every((k) => k.ownerDid === 'did:mycelix:alice')).toBe(true);
    });

    it('should return empty array for unknown DID', () => {
      const keys = keyManager.getKeysByDid('did:mycelix:unknown');
      expect(keys).toEqual([]);
    });
  });

  describe('getKeysByHapp', () => {
    it('should retrieve all keys for a hApp', () => {
      const keyPair1 = keyManager.generateKeyPair('Dilithium3');
      const keyPair2 = keyManager.generateKeyPair('Dilithium5');

      keyManager.registerKey(
        keyPair1.keyId,
        keyPair1.publicKey,
        keyPair1.algorithm,
        'did:mycelix:alice',
        'governance'
      );
      keyManager.registerKey(
        keyPair2.keyId,
        keyPair2.publicKey,
        keyPair2.algorithm,
        'did:mycelix:bob',
        'governance'
      );

      const govKeys = keyManager.getKeysByHapp('governance');

      expect(govKeys.length).toBe(2);
      expect(govKeys.every((k) => k.happId === 'governance')).toBe(true);
    });
  });

  describe('isKeyValid', () => {
    it('should return true for valid key', () => {
      const keyPair = keyManager.generateKeyPair();
      keyManager.registerKey(
        keyPair.keyId,
        keyPair.publicKey,
        keyPair.algorithm,
        'did:mycelix:alice',
        'governance'
      );

      expect(keyManager.isKeyValid(keyPair.keyId)).toBe(true);
    });

    it('should return false for revoked key', () => {
      const keyPair = keyManager.generateKeyPair();
      keyManager.registerKey(
        keyPair.keyId,
        keyPair.publicKey,
        keyPair.algorithm,
        'did:mycelix:alice',
        'governance'
      );
      keyManager.revokeKey(keyPair.keyId, 'Compromised');

      expect(keyManager.isKeyValid(keyPair.keyId)).toBe(false);
    });

    it('should return false for non-existent key', () => {
      expect(keyManager.isKeyValid('non-existent')).toBe(false);
    });
  });

  describe('revokeKey', () => {
    it('should revoke a registered key', () => {
      const keyPair = keyManager.generateKeyPair();
      keyManager.registerKey(
        keyPair.keyId,
        keyPair.publicKey,
        keyPair.algorithm,
        'did:mycelix:alice',
        'governance'
      );

      const result = keyManager.revokeKey(keyPair.keyId, 'Key compromised');

      expect(result).toBe(true);
      const key = keyManager.getKey(keyPair.keyId);
      expect(key!.revoked).toBe(true);
      expect(key!.revokedAt).toBeDefined();
      expect(key!.revokedReason).toBe('Key compromised');
    });

    it('should return false for non-existent key', () => {
      const result = keyManager.revokeKey('non-existent');
      expect(result).toBe(false);
    });
  });

  describe('cleanup', () => {
    it('should remove expired keys', () => {
      const keyPair = keyManager.generateKeyPair('Dilithium3', undefined, -1000); // Already expired
      keyManager.registerKey(
        keyPair.keyId,
        keyPair.publicKey,
        keyPair.algorithm,
        'did:mycelix:alice',
        'governance',
        Date.now() - 1000 // Expired in past
      );

      const removed = keyManager.cleanup();

      expect(removed).toBe(1);
      expect(keyManager.getKey(keyPair.keyId)).toBeNull();
    });
  });

  describe('clear', () => {
    it('should remove all keys', () => {
      keyManager.generateKeyPair();
      keyManager.generateKeyPair();

      const keyPair = keyManager.generateKeyPair();
      keyManager.registerKey(
        keyPair.keyId,
        keyPair.publicKey,
        keyPair.algorithm,
        'did:mycelix:alice',
        'governance'
      );

      keyManager.clear();

      expect(keyManager.keyCount).toBe(0);
    });
  });
});

// ============================================================================
// PQCSignatureService Tests
// ============================================================================

describe('Integration: PQCSignatureService', () => {
  let keyManager: PQCKeyManager;
  let signatureService: PQCSignatureService;
  let testKeyPair: PQCKeyPair;

  beforeEach(() => {
    keyManager = new PQCKeyManager();
    signatureService = new PQCSignatureService(keyManager);

    // Generate and register a test key
    testKeyPair = keyManager.generateKeyPair('Dilithium3', 'did:mycelix:alice');
    keyManager.registerKey(
      testKeyPair.keyId,
      testKeyPair.publicKey,
      testKeyPair.algorithm,
      'did:mycelix:alice',
      'governance'
    );
  });

  describe('sign', () => {
    it('should create a valid signature', async () => {
      const data = 'Hello, quantum world!';
      const signature = await signatureService.sign(data, testKeyPair, 'did:mycelix:alice');

      expect(signature).toBeDefined();
      expect(signature.signature).toBeInstanceOf(Uint8Array);
      expect(signature.algorithm).toBe('Dilithium3');
      expect(signature.keyId).toBe(testKeyPair.keyId);
      expect(signature.signerDid).toBe('did:mycelix:alice');
      expect(signature.signedAt).toBeLessThanOrEqual(Date.now());
      expect(signature.contentHash).toMatch(/^[a-f0-9]{64}$/);
      expect(signature.nonce).toBeDefined();
    });

    it('should create different signatures for same data (due to nonce)', async () => {
      const data = 'Same data';

      const sig1 = await signatureService.sign(data, testKeyPair, 'did:mycelix:alice');
      const sig2 = await signatureService.sign(data, testKeyPair, 'did:mycelix:alice');

      expect(sig1.nonce).not.toBe(sig2.nonce);
      expect(sig1.contentHash).toBe(sig2.contentHash); // Same content
    });

    it('should sign binary data', async () => {
      const data = new Uint8Array([1, 2, 3, 4, 5]);
      const signature = await signatureService.sign(data, testKeyPair, 'did:mycelix:alice');

      expect(signature).toBeDefined();
      expect(signature.contentHash.length).toBe(64);
    });
  });

  describe('verify', () => {
    it('should verify a valid signature', async () => {
      const data = 'Hello, quantum world!';
      const signature = await signatureService.sign(data, testKeyPair, 'did:mycelix:alice');

      const result = await signatureService.verify(data, signature);

      expect(result.valid).toBe(true);
      expect(result.keyId).toBe(testKeyPair.keyId);
      expect(result.signerDid).toBe('did:mycelix:alice');
      expect(result.algorithm).toBe('Dilithium3');
      expect(result.verifiedAt).toBeLessThanOrEqual(Date.now());
    });

    it('should reject signature from unregistered key', async () => {
      const unregisteredKey = keyManager.generateKeyPair('Dilithium5');
      const data = 'Test data';
      const signature = await signatureService.sign(data, unregisteredKey, 'did:mycelix:bob');

      const result = await signatureService.verify(data, signature);

      expect(result.valid).toBe(false);
      expect(result.failureReason).toBe('Key not registered');
    });

    it('should reject signature from revoked key', async () => {
      const data = 'Test data';
      const signature = await signatureService.sign(data, testKeyPair, 'did:mycelix:alice');

      // Revoke the key
      keyManager.revokeKey(testKeyPair.keyId, 'Compromised');

      const result = await signatureService.verify(data, signature);

      expect(result.valid).toBe(false);
      expect(result.failureReason).toContain('revoked');
    });

    it('should reject signature with wrong DID', async () => {
      const data = 'Test data';
      const signature = await signatureService.sign(data, testKeyPair, 'did:mycelix:bob');

      const result = await signatureService.verify(data, signature);

      expect(result.valid).toBe(false);
      expect(result.failureReason).toContain('DID does not match');
    });

    it('should reject tampered content', async () => {
      const data = 'Original message';
      const signature = await signatureService.sign(data, testKeyPair, 'did:mycelix:alice');

      const result = await signatureService.verify('Tampered message', signature);

      expect(result.valid).toBe(false);
      expect(result.failureReason).toContain('Content hash mismatch');
    });

    it('should reject replayed nonce', async () => {
      const data = 'Test data';
      const signature = await signatureService.sign(data, testKeyPair, 'did:mycelix:alice');

      // First verification should succeed
      const result1 = await signatureService.verify(data, signature);
      expect(result1.valid).toBe(true);

      // Replay should fail
      const result2 = await signatureService.verify(data, signature);
      expect(result2.valid).toBe(false);
      expect(result2.failureReason).toContain('Nonce already used');
    });

    it('should reject old signatures', async () => {
      const data = 'Test data';
      const signature = await signatureService.sign(data, testKeyPair, 'did:mycelix:alice');

      // Modify signature timestamp to be old
      signature.signedAt = Date.now() - 10 * 60 * 1000; // 10 minutes ago

      // Use short window for test
      signatureService.setNonceWindow(60000); // 1 minute

      const result = await signatureService.verify(data, signature);

      expect(result.valid).toBe(false);
      expect(result.failureReason).toContain('too old');
    });
  });
});

// ============================================================================
// SignedMessageBridge Tests
// ============================================================================

describe('Integration: SignedMessageBridge', () => {
  let messageBridge: SignedMessageBridge;

  beforeEach(() => {
    resetPQCSingletons();
    messageBridge = new SignedMessageBridge();
  });

  describe('generateHappKeyPair', () => {
    it('should generate and register a key pair for a hApp', () => {
      const keyPair = messageBridge.generateHappKeyPair(
        'governance',
        'did:mycelix:governance-service'
      );

      expect(keyPair).toBeDefined();
      expect(keyPair.algorithm).toBe('Dilithium3');
      expect(messageBridge.getRegisteredHapps()).toContain('governance');
    });

    it('should use specified algorithm', () => {
      const keyPair = messageBridge.generateHappKeyPair(
        'finance',
        'did:mycelix:finance-service',
        'Dilithium5'
      );

      expect(keyPair.algorithm).toBe('Dilithium5');
    });
  });

  describe('signMessage', () => {
    it('should sign a cross-hApp message', async () => {
      messageBridge.generateHappKeyPair('governance', 'did:mycelix:governance-service');

      const message = createTestMessage({ sourceHapp: 'governance' });
      const signedMessage = await messageBridge.signMessage(message, 'did:mycelix:governance-service');

      expect(signedMessage).toBeDefined();
      expect(signedMessage.signature).toBeDefined();
      expect(signedMessage.signerDid).toBe('did:mycelix:governance-service');
      expect(signedMessage.id).toBe(message.id);
    });

    it('should throw when no key registered for hApp', async () => {
      const message = createTestMessage({ sourceHapp: 'governance' });

      await expect(messageBridge.signMessage(message, 'did:mycelix:user')).rejects.toThrow(
        /No key pair registered/
      );
    });
  });

  describe('verifyMessage', () => {
    it('should verify a signed message', async () => {
      messageBridge.generateHappKeyPair('governance', 'did:mycelix:governance-service');

      const message = createTestMessage({ sourceHapp: 'governance' });
      const signedMessage = await messageBridge.signMessage(message, 'did:mycelix:governance-service');

      const result = await messageBridge.verifyMessage(signedMessage);

      expect(result.valid).toBe(true);
      expect(result.signerDid).toBe('did:mycelix:governance-service');
    });

    it('should reject tampered message', async () => {
      messageBridge.generateHappKeyPair('governance', 'did:mycelix:governance-service');

      const message = createTestMessage({ sourceHapp: 'governance' });
      const signedMessage = await messageBridge.signMessage(message, 'did:mycelix:governance-service');

      // Tamper with payload
      signedMessage.payload = { malicious: true };

      const result = await messageBridge.verifyMessage(signedMessage);

      expect(result.valid).toBe(false);
      expect(result.failureReason).toContain('Content hash mismatch');
    });

    it('should reject message with different timestamp', async () => {
      messageBridge.generateHappKeyPair('governance', 'did:mycelix:governance-service');

      const message = createTestMessage({ sourceHapp: 'governance' });
      const signedMessage = await messageBridge.signMessage(message, 'did:mycelix:governance-service');

      // Tamper with timestamp
      signedMessage.timestamp = signedMessage.timestamp + 1000;

      const result = await messageBridge.verifyMessage(signedMessage);

      expect(result.valid).toBe(false);
    });
  });
});

// ============================================================================
// Utility Function Tests
// ============================================================================

describe('Integration: PQC Utility Functions', () => {
  describe('getRecommendedAlgorithm', () => {
    it('should return ML-DSA-65 for standard strength', () => {
      expect(getRecommendedAlgorithm('standard')).toBe('ML-DSA-65');
    });

    it('should return ML-DSA-87 for high strength', () => {
      expect(getRecommendedAlgorithm('high')).toBe('ML-DSA-87');
    });

    it('should return SLH-DSA-SHA2-128s for maximum strength', () => {
      expect(getRecommendedAlgorithm('maximum')).toBe('SLH-DSA-SHA2-128s');
    });
  });

  describe('isAlgorithmSecure', () => {
    it('should return true for all supported algorithms', () => {
      const algorithms: PQCAlgorithm[] = [
        'Dilithium3',
        'Dilithium5',
        'Falcon512',
        'SPHINCS+',
        'Hybrid-Ed25519-Dilithium3',
      ];

      for (const algorithm of algorithms) {
        expect(isAlgorithmSecure(algorithm)).toBe(true);
      }
    });
  });

  describe('estimateSignatureSize', () => {
    it('should return expected sizes for algorithms', () => {
      expect(estimateSignatureSize('ML-DSA-65')).toBe(3309);
      expect(estimateSignatureSize('ML-DSA-87')).toBe(4627);
      expect(estimateSignatureSize('Falcon512')).toBe(3309);
      expect(estimateSignatureSize('SLH-DSA-SHA2-128s')).toBe(7856);
    });

    it('should return hybrid size for hybrid algorithm', () => {
      const hybridSize = estimateSignatureSize('Hybrid-Ed25519-ML-DSA-65');
      expect(hybridSize).toBe(64 + 3309);
    });
  });
});

// ============================================================================
// Singleton Tests
// ============================================================================

describe('Integration: PQC Singletons', () => {
  beforeEach(() => {
    resetPQCSingletons();
  });

  it('should return singleton key manager', () => {
    const manager1 = getPQCKeyManager();
    const manager2 = getPQCKeyManager();

    expect(manager1).toBe(manager2);
  });

  it('should return singleton signature service', () => {
    const service1 = getPQCSignatureService();
    const service2 = getPQCSignatureService();

    expect(service1).toBe(service2);
  });

  it('should return singleton message bridge', () => {
    const bridge1 = getSignedMessageBridge();
    const bridge2 = getSignedMessageBridge();

    expect(bridge1).toBe(bridge2);
  });

  it('should reset singletons correctly', () => {
    const manager1 = getPQCKeyManager();
    resetPQCSingletons();
    const manager2 = getPQCKeyManager();

    expect(manager1).not.toBe(manager2);
  });
});

// ============================================================================
// Cross-Algorithm Compatibility Tests
// ============================================================================

describe('Integration: Cross-Algorithm Compatibility', () => {
  let keyManager: PQCKeyManager;
  let signatureService: PQCSignatureService;

  beforeEach(() => {
    keyManager = new PQCKeyManager();
    signatureService = new PQCSignatureService(keyManager);
  });

  it('should reject signature with mismatched algorithm claim', async () => {
    // Generate Dilithium3 key
    const keyPair = keyManager.generateKeyPair('Dilithium3', 'did:mycelix:alice');
    keyManager.registerKey(
      keyPair.keyId,
      keyPair.publicKey,
      keyPair.algorithm,
      'did:mycelix:alice',
      'governance'
    );

    // Sign with correct key
    const data = 'Test data';
    const signature = await signatureService.sign(data, keyPair, 'did:mycelix:alice');

    // Tamper with algorithm claim
    signature.algorithm = 'Dilithium5';

    const result = await signatureService.verify(data, signature);

    expect(result.valid).toBe(false);
    expect(result.failureReason).toContain('Algorithm mismatch');
  });

  it('should support hybrid algorithm signing and verification', async () => {
    const keyPair = keyManager.generateKeyPair('Hybrid-Ed25519-Dilithium3', 'did:mycelix:alice');
    keyManager.registerKey(
      keyPair.keyId,
      keyPair.publicKey,
      keyPair.algorithm,
      'did:mycelix:alice',
      'governance'
    );

    const data = 'Hybrid signature test';
    const signature = await signatureService.sign(data, keyPair, 'did:mycelix:alice');

    expect(signature.algorithm).toBe('Hybrid-Ed25519-Dilithium3');
    expect(signature.signature.length).toBeGreaterThan(32);

    const result = await signatureService.verify(data, signature);
    expect(result.valid).toBe(true);
  });
});
