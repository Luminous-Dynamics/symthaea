/**
 * Mobile Module Tests
 *
 * Tests for QR code utilities, credential builders, and mobile helpers.
 */

import { describe, it, expect } from 'vitest';
import {
  // QR Code utilities
  QRPayloadBuilder,
  encodeQRPayload,
  decodeQRPayload,
  isPayloadExpired,
  validateQRPayload,
  createWalletConnectPayload,
  createCredentialRequestPayload,
  createPaymentRequestPayload,
  CredentialRequestBuilder,
  // Biometric utilities
  biometricErrorToMessage,
  SimulatedBiometricProvider,
  // Types
  type QRPayload,
  type BiometricErrorCode,
} from '../src/mobile/index.js';

// =============================================================================
// QRPayloadBuilder Tests
// =============================================================================

describe('QRPayloadBuilder', () => {
  it('should build a basic payload', () => {
    const payload = new QRPayloadBuilder('wallet_connect', { test: 'data' }).build();

    expect(payload.type).toBe('wallet_connect');
    expect(payload.version).toBe(1);
    expect(payload.data).toEqual({ test: 'data' });
  });

  it('should set expiration with expiresIn', () => {
    const before = Date.now();
    const payload = new QRPayloadBuilder('wallet_connect', {})
      .expiresIn(60000) // 1 minute
      .build();

    expect(payload.expiresAt).toBeGreaterThanOrEqual(before + 60000);
  });

  it('should set absolute expiration', () => {
    const expTime = Date.now() + 300000;
    const payload = new QRPayloadBuilder('wallet_connect', {}).expiresAtTime(expTime).build();

    expect(payload.expiresAt).toBe(expTime);
  });
});

// =============================================================================
// QR Payload Encoding/Decoding Tests
// =============================================================================

describe('QR Payload Encoding/Decoding', () => {
  it('should encode and decode a payload', () => {
    const original = new QRPayloadBuilder('wallet_connect', { happId: 'test' }).build();

    const encoded = encodeQRPayload(original);
    expect(typeof encoded).toBe('string');
    expect(encoded.startsWith('mycelix://')).toBe(true);

    const decoded = decodeQRPayload(encoded);
    expect(decoded).not.toBeNull();
    expect(decoded?.type).toBe('wallet_connect');
    expect(decoded?.data).toEqual({ happId: 'test' });
  });

  it('should handle invalid QR data gracefully', () => {
    const result = decodeQRPayload('not-valid-base64-or-json');
    expect(result).toBeNull();
  });

  it('should handle empty string', () => {
    const result = decodeQRPayload('');
    expect(result).toBeNull();
  });
});

// =============================================================================
// Payload Expiration Tests
// =============================================================================

describe('isPayloadExpired', () => {
  it('should return false for non-expired payload', () => {
    const payload: QRPayload = {
      type: 'wallet_connect',
      version: 1,
      data: {},
      expiresAt: Date.now() + 60000, // 1 minute in future
    };

    expect(isPayloadExpired(payload)).toBe(false);
  });

  it('should return true for expired payload', () => {
    const payload: QRPayload = {
      type: 'wallet_connect',
      version: 1,
      data: {},
      expiresAt: Date.now() - 60000, // 1 minute ago
    };

    expect(isPayloadExpired(payload)).toBe(true);
  });

  it('should return false for payload without expiration', () => {
    const payload: QRPayload = {
      type: 'wallet_connect',
      version: 1,
      data: {},
    };

    expect(isPayloadExpired(payload)).toBe(false);
  });
});

// =============================================================================
// Payload Validation Tests
// =============================================================================

describe('validateQRPayload', () => {
  it('should validate a correct payload', () => {
    const payload: QRPayload = {
      type: 'wallet_connect',
      version: 1,
      data: { happId: 'test' },
    };

    const result = validateQRPayload(payload);
    expect(result.valid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  it('should reject payload without type', () => {
    const payload = {
      version: 1,
      data: {},
    } as unknown as QRPayload;

    const result = validateQRPayload(payload);
    expect(result.valid).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
  });

  it('should reject payload without data', () => {
    const payload = {
      type: 'wallet_connect',
      version: 1,
    } as unknown as QRPayload;

    const result = validateQRPayload(payload);
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Missing payload data');
  });
});

// =============================================================================
// Payload Factory Tests
// =============================================================================

describe('Payload Factory Functions', () => {
  describe('createWalletConnectPayload', () => {
    it('should create valid wallet connect payload', () => {
      const payload = createWalletConnectPayload('my-happ', 'My App', ['read_profile']);

      expect(payload.type).toBe('wallet_connect');
      expect(payload.version).toBe(1);
      expect(payload.data.happId).toBe('my-happ');
      expect(payload.data.happName).toBe('My App');
      expect(payload.data.requiredPermissions).toEqual(['read_profile']);
      expect(payload.data.sessionId).toBeDefined();
    });

    it('should include callback URL when provided', () => {
      const payload = createWalletConnectPayload('happ', 'App', [], 'https://callback.url');
      expect(payload.data.callbackUrl).toBe('https://callback.url');
    });

    it('should set expiration', () => {
      const payload = createWalletConnectPayload('happ', 'App', []);
      expect(payload.expiresAt).toBeGreaterThan(Date.now());
    });
  });

  describe('createCredentialRequestPayload', () => {
    it('should create valid credential request payload', () => {
      const payload = createCredentialRequestPayload('agent:verifier', 'Verifier App', [
        { credentialType: 'IdentityCredential' as any, required: true },
      ]);

      expect(payload.type).toBe('credential_request');
      expect(payload.data.verifier).toBe('agent:verifier');
      expect(payload.data.verifierName).toBe('Verifier App');
      expect(payload.data.requestedCredentials).toHaveLength(1);
    });
  });

  describe('createPaymentRequestPayload', () => {
    it('should create valid payment request payload', () => {
      const payload = createPaymentRequestPayload('agent:recipient', 100, 'MYC', 'Test payment');

      expect(payload.type).toBe('payment_request');
      expect(payload.data.recipient).toBe('agent:recipient');
      expect(payload.data.amount).toBe(100);
      expect(payload.data.currency).toBe('MYC');
      expect(payload.data.description).toBe('Test payment');
    });
  });
});

// =============================================================================
// CredentialRequestBuilder Tests
// =============================================================================

describe('CredentialRequestBuilder', () => {
  it('should build credential request with single credential', () => {
    const request = new CredentialRequestBuilder().request('IdentityCredential', true).build();

    expect(request).toHaveLength(1);
    expect(request[0].type).toBe('IdentityCredential');
    expect(request[0].required).toBe(true);
  });

  it('should build credential request with specific claims', () => {
    const request = new CredentialRequestBuilder()
      .requestClaims('EducationCredential', ['degree', 'institution'], true)
      .build();

    expect(request).toHaveLength(1);
    expect(request[0].claimsToReveal).toEqual(['degree', 'institution']);
  });

  it('should build complex multi-credential request', () => {
    const request = new CredentialRequestBuilder()
      .request('IdentityCredential', true)
      .requestClaims('EducationCredential', ['degree'], false)
      .build();

    expect(request).toHaveLength(2);
  });
});

// =============================================================================
// Biometric Error Messages Tests
// =============================================================================

describe('biometricErrorToMessage', () => {
  it('should return message for user_cancelled', () => {
    expect(biometricErrorToMessage('user_cancelled')).toBe('Authentication was cancelled');
  });

  it('should return message for not_available', () => {
    expect(biometricErrorToMessage('not_available')).toBe(
      'Biometric authentication is not available on this device.'
    );
  });

  it('should return message for not_enrolled', () => {
    expect(biometricErrorToMessage('not_enrolled')).toBe(
      'No biometrics enrolled. Please set up biometrics in your device settings.'
    );
  });

  it('should return message for lockout', () => {
    expect(biometricErrorToMessage('lockout')).toBe('Too many failed attempts. Please try again later.');
  });

  it('should return message for hardware_error', () => {
    expect(biometricErrorToMessage('hardware_error')).toBe(
      'There was a problem with the biometric sensor. Please try again.'
    );
  });

  it('should return message for timeout', () => {
    expect(biometricErrorToMessage('timeout')).toBe('Authentication timed out. Please try again.');
  });

  it('should return default message for unknown error', () => {
    expect(biometricErrorToMessage('unknown_error' as BiometricErrorCode)).toBe(
      'Authentication failed. Please try again.'
    );
  });
});

// =============================================================================
// Simulated Biometric Provider Tests
// =============================================================================

describe('SimulatedBiometricProvider', () => {
  it('should create with default options', () => {
    const provider = new SimulatedBiometricProvider();
    expect(provider).toBeDefined();
  });

  it('should check capability', async () => {
    const provider = new SimulatedBiometricProvider();
    const capability = await provider.checkCapability();
    expect(capability.available).toBe(true);
    expect(capability.enrolled).toBe(true);
    expect(capability.types).toContain('fingerprint');
  });

  it('should authenticate successfully by default', async () => {
    const provider = new SimulatedBiometricProvider();
    const result = await provider.authenticate('Test prompt');
    expect(result.success).toBe(true);
    expect(result.biometricType).toBeDefined();
  });

  it('should fail authentication when configured', async () => {
    const provider = new SimulatedBiometricProvider();
    provider.configure({ shouldSucceed: false });
    const result = await provider.authenticate('Test prompt');
    expect(result.success).toBe(false);
    expect(result.errorCode).toBeDefined();
  });

  it('should configure custom types', async () => {
    const provider = new SimulatedBiometricProvider();
    provider.configure({ types: ['face'] });
    const capability = await provider.checkCapability();
    expect(capability.types).toEqual(['face']);
  });

  it('should configure unavailable state', async () => {
    const provider = new SimulatedBiometricProvider();
    provider.configure({ available: false });
    const capability = await provider.checkCapability();
    expect(capability.available).toBe(false);
  });
});
