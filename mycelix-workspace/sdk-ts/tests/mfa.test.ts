// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * MFA (Multi-Factor Authentication) Zome Client Tests
 *
 * Comprehensive unit tests for the MfaClient from src/integrations/identity/zomes/mfa.ts
 * Tests cover all methods with mocked Holochain client, including error handling.
 *
 * @module @mycelix/sdk/tests/mfa
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { AppClient } from '@holochain/client';
import {
  MfaClient,
  type MfaState,
  type EnrolledFactor,
  type FactorType,
  type AssuranceLevel,
  type FactorCategory,
  type VerificationChallenge,
  type MfaVerificationResult,
  type FlEligibilityResult,
  type FactorEnrollment,
  type FactorProof,
} from '../src/integrations/identity/zomes/mfa.js';
import { IdentitySdkError, IdentitySdkErrorCode } from '../src/integrations/identity/types.js';

// =============================================================================
// Test Fixtures
// =============================================================================

const VALID_DID = 'did:mycelix:uhCAk123abc456def789';
const INVALID_DID = 'invalid-did-format';
const FACTOR_ID = 'factor-123-abc';
const PRIMARY_KEY_HASH = 'sha256:abc123def456';

function createMockEnrolledFactor(overrides: Partial<EnrolledFactor> = {}): EnrolledFactor {
  const now = Date.now();
  return {
    id: FACTOR_ID,
    factorType: 'PrimaryKeyPair',
    category: 'possession',
    createdAt: now,
    lastUsedAt: now,
    metadata: {},
    verified: true,
    ...overrides,
  };
}

function createMockMfaState(overrides: Partial<MfaState> = {}): MfaState {
  return {
    did: VALID_DID,
    factors: [createMockEnrolledFactor()],
    assuranceLevel: 'Basic',
    lastVerifiedAt: Date.now(),
    flEligible: false,
    ...overrides,
  };
}

function createMockChallenge(overrides: Partial<VerificationChallenge> = {}): VerificationChallenge {
  return {
    challengeId: 'challenge-123',
    factorId: FACTOR_ID,
    factorType: 'PrimaryKeyPair',
    challenge: 'random-challenge-string-base64',
    expiresAt: Date.now() + 300000, // 5 minutes from now
    ...overrides,
  };
}

function createMockVerificationResult(
  overrides: Partial<MfaVerificationResult> = {}
): MfaVerificationResult {
  return {
    success: true,
    factorId: FACTOR_ID,
    verifiedAt: Date.now(),
    newAssuranceLevel: 'Verified',
    ...overrides,
  };
}

function createMockFlEligibility(
  overrides: Partial<FlEligibilityResult> = {}
): FlEligibilityResult {
  return {
    eligible: true,
    requirements: [
      { name: 'Minimum Assurance', met: true, description: 'Must have Verified or higher' },
      { name: 'Active Factor', met: true, description: 'Must have at least one active factor' },
    ],
    currentScore: 0.75,
    requiredScore: 0.5,
    ...overrides,
  };
}

function createMockEnrollmentHistory(): FactorEnrollment[] {
  const now = Date.now();
  return [
    {
      factorId: 'factor-1',
      factorType: 'PrimaryKeyPair',
      action: 'enrolled',
      timestamp: now - 86400000, // 1 day ago
    },
    {
      factorId: 'factor-2',
      factorType: 'HardwareKey',
      action: 'enrolled',
      timestamp: now - 3600000, // 1 hour ago
    },
    {
      factorId: 'factor-1',
      factorType: 'PrimaryKeyPair',
      action: 'verified',
      timestamp: now,
    },
  ];
}

// =============================================================================
// Mock AppClient Factory
// =============================================================================

interface MockZomeCallParams {
  role_name: string;
  zome_name: string;
  fn_name: string;
  payload: unknown;
}

function createMockAppClient(
  responses: Map<string, unknown> = new Map(),
  shouldFail = false,
  errorMessage = 'Zome call failed'
): AppClient {
  return {
    callZome: vi.fn(async (params: MockZomeCallParams): Promise<unknown> => {
      if (shouldFail) {
        throw new Error(errorMessage);
      }
      const key = params.fn_name;
      if (responses.has(key)) {
        return responses.get(key);
      }
      throw new Error(`No mock response for ${key}`);
    }),
  } as unknown as AppClient;
}

// =============================================================================
// MfaClient Tests
// =============================================================================

describe('MfaClient', () => {
  let client: MfaClient;
  let mockAppClient: AppClient;
  let responses: Map<string, unknown>;

  beforeEach(() => {
    responses = new Map<string, unknown>();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // ===========================================================================
  // getMfaState Tests
  // ===========================================================================

  describe('getMfaState', () => {
    it('should successfully fetch MFA state for a valid DID', async () => {
      const mockState = createMockMfaState();
      responses.set('get_mfa_state', mockState);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.getMfaState(VALID_DID);

      expect(result).toEqual(mockState);
      expect(mockAppClient.callZome).toHaveBeenCalledWith({
        role_name: 'mycelix_identity',
        zome_name: 'mfa',
        fn_name: 'get_mfa_state',
        payload: VALID_DID,
      });
    });

    it('should return null when MFA state is not found', async () => {
      responses.set('get_mfa_state', null);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.getMfaState(VALID_DID);

      expect(result).toBeNull();
    });

    it('should throw IdentitySdkError for invalid DID format', async () => {
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      await expect(client.getMfaState(INVALID_DID)).rejects.toThrow(IdentitySdkError);
      await expect(client.getMfaState(INVALID_DID)).rejects.toMatchObject({
        code: IdentitySdkErrorCode.INVALID_DID_FORMAT,
      });
    });

    it('should throw IdentitySdkError for DID without mycelix prefix', async () => {
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      await expect(client.getMfaState('did:example:123')).rejects.toThrow(IdentitySdkError);
    });
  });

  // ===========================================================================
  // createMfaState Tests
  // ===========================================================================

  describe('createMfaState', () => {
    it('should successfully create MFA state for a valid DID', async () => {
      const mockState = createMockMfaState();
      responses.set('create_mfa_state', mockState);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.createMfaState(VALID_DID, PRIMARY_KEY_HASH);

      expect(result).toEqual(mockState);
      expect(mockAppClient.callZome).toHaveBeenCalledWith({
        role_name: 'mycelix_identity',
        zome_name: 'mfa',
        fn_name: 'create_mfa_state',
        payload: {
          did: VALID_DID,
          primary_key_hash: PRIMARY_KEY_HASH,
        },
      });
    });

    it('should throw IdentitySdkError for invalid DID format', async () => {
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      await expect(client.createMfaState(INVALID_DID, PRIMARY_KEY_HASH)).rejects.toThrow(
        IdentitySdkError
      );
    });

    it('should propagate zome call errors as IdentitySdkError', async () => {
      mockAppClient = createMockAppClient(responses, true, 'State already exists');
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      await expect(client.createMfaState(VALID_DID, PRIMARY_KEY_HASH)).rejects.toMatchObject({
        code: IdentitySdkErrorCode.ZOME_CALL_FAILED,
      });
    });
  });

  // ===========================================================================
  // enrollFactor Tests
  // ===========================================================================

  describe('enrollFactor', () => {
    const factorTypes: FactorType[] = [
      'PrimaryKeyPair',
      'HardwareKey',
      'Biometric',
      'SocialRecovery',
      'ReputationAttestation',
      'GitcoinPassport',
      'VerifiableCredential',
      'RecoveryPhrase',
      'SecurityQuestions',
    ];

    it('should successfully enroll a factor with basic input', async () => {
      const newFactor = createMockEnrolledFactor({ factorType: 'HardwareKey' });
      responses.set('enroll_factor', newFactor);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.enrollFactor(VALID_DID, {
        factorType: 'HardwareKey',
        metadata: { deviceName: 'YubiKey 5' },
      });

      expect(result).toEqual(newFactor);
      expect(mockAppClient.callZome).toHaveBeenCalledWith({
        role_name: 'mycelix_identity',
        zome_name: 'mfa',
        fn_name: 'enroll_factor',
        payload: {
          did: VALID_DID,
          factor_type: 'HardwareKey',
          metadata: { deviceName: 'YubiKey 5' },
          proof: undefined,
        },
      });
    });

    factorTypes.forEach((factorType) => {
      it(`should successfully enroll factor type: ${factorType}`, async () => {
        const newFactor = createMockEnrolledFactor({ factorType });
        responses.set('enroll_factor', newFactor);
        mockAppClient = createMockAppClient(responses);
        client = new MfaClient(mockAppClient, 'mycelix_identity');

        const result = await client.enrollFactor(VALID_DID, { factorType });

        expect(result.factorType).toBe(factorType);
      });
    });

    it('should enroll factor with signature proof', async () => {
      const proof: FactorProof = {
        type: 'signature',
        signature: 'sig-abc123',
        message: 'challenge-message',
      };
      const newFactor = createMockEnrolledFactor({ factorType: 'PrimaryKeyPair' });
      responses.set('enroll_factor', newFactor);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.enrollFactor(VALID_DID, {
        factorType: 'PrimaryKeyPair',
        proof,
      });

      expect(result).toEqual(newFactor);
      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: expect.objectContaining({
            proof,
          }),
        })
      );
    });

    it('should enroll factor with WebAuthn proof', async () => {
      const proof: FactorProof = {
        type: 'webauthn',
        authenticatorData: 'auth-data-base64',
        clientDataHash: 'client-hash-base64',
        signature: 'webauthn-sig-base64',
      };
      const newFactor = createMockEnrolledFactor({ factorType: 'HardwareKey' });
      responses.set('enroll_factor', newFactor);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.enrollFactor(VALID_DID, {
        factorType: 'HardwareKey',
        proof,
      });

      expect(result.factorType).toBe('HardwareKey');
    });

    it('should enroll factor with Gitcoin Passport proof', async () => {
      const proof: FactorProof = {
        type: 'gitcoinPassport',
        score: 42,
        checkedAt: Date.now(),
        stamps: ['google', 'twitter', 'github'],
      };
      const newFactor = createMockEnrolledFactor({ factorType: 'GitcoinPassport' });
      responses.set('enroll_factor', newFactor);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.enrollFactor(VALID_DID, {
        factorType: 'GitcoinPassport',
        proof,
      });

      expect(result.factorType).toBe('GitcoinPassport');
    });

    it('should enroll factor with social recovery proof', async () => {
      const proof: FactorProof = {
        type: 'socialRecovery',
        guardianSignatures: [
          { guardianDid: 'did:mycelix:guardian1', signature: 'sig1', timestamp: Date.now() },
          { guardianDid: 'did:mycelix:guardian2', signature: 'sig2', timestamp: Date.now() },
        ],
        threshold: 2,
      };
      const newFactor = createMockEnrolledFactor({ factorType: 'SocialRecovery' });
      responses.set('enroll_factor', newFactor);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.enrollFactor(VALID_DID, {
        factorType: 'SocialRecovery',
        proof,
      });

      expect(result.factorType).toBe('SocialRecovery');
    });

    it('should throw IdentitySdkError for invalid DID format', async () => {
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      await expect(
        client.enrollFactor(INVALID_DID, { factorType: 'HardwareKey' })
      ).rejects.toMatchObject({
        code: IdentitySdkErrorCode.INVALID_DID_FORMAT,
      });
    });
  });

  // ===========================================================================
  // removeFactor Tests
  // ===========================================================================

  describe('removeFactor', () => {
    it('should successfully remove a factor', async () => {
      const updatedState = createMockMfaState({ factors: [] });
      responses.set('remove_factor', updatedState);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.removeFactor(VALID_DID, FACTOR_ID);

      expect(result.factors).toHaveLength(0);
      expect(mockAppClient.callZome).toHaveBeenCalledWith({
        role_name: 'mycelix_identity',
        zome_name: 'mfa',
        fn_name: 'remove_factor',
        payload: {
          did: VALID_DID,
          factor_id: FACTOR_ID,
        },
      });
    });

    it('should throw IdentitySdkError for invalid DID format', async () => {
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      await expect(client.removeFactor(INVALID_DID, FACTOR_ID)).rejects.toMatchObject({
        code: IdentitySdkErrorCode.INVALID_DID_FORMAT,
      });
    });

    it('should propagate zome call errors for non-existent factor', async () => {
      mockAppClient = createMockAppClient(responses, true, 'Factor not found');
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      await expect(client.removeFactor(VALID_DID, 'non-existent')).rejects.toMatchObject({
        code: IdentitySdkErrorCode.ZOME_CALL_FAILED,
      });
    });
  });

  // ===========================================================================
  // generateChallenge Tests
  // ===========================================================================

  describe('generateChallenge', () => {
    it('should successfully generate a verification challenge', async () => {
      const mockChallenge = createMockChallenge();
      responses.set('generate_verification_challenge', mockChallenge);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.generateChallenge(VALID_DID, FACTOR_ID);

      expect(result).toEqual(mockChallenge);
      expect(result.challengeId).toBe('challenge-123');
      expect(result.expiresAt).toBeGreaterThan(Date.now());
      expect(mockAppClient.callZome).toHaveBeenCalledWith({
        role_name: 'mycelix_identity',
        zome_name: 'mfa',
        fn_name: 'generate_verification_challenge',
        payload: {
          did: VALID_DID,
          factor_id: FACTOR_ID,
        },
      });
    });

    it('should generate challenge for different factor types', async () => {
      const hwKeyChallenge = createMockChallenge({ factorType: 'HardwareKey' });
      responses.set('generate_verification_challenge', hwKeyChallenge);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.generateChallenge(VALID_DID, 'hw-key-factor');

      expect(result.factorType).toBe('HardwareKey');
    });

    it('should throw IdentitySdkError for invalid DID format', async () => {
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      await expect(client.generateChallenge(INVALID_DID, FACTOR_ID)).rejects.toMatchObject({
        code: IdentitySdkErrorCode.INVALID_DID_FORMAT,
      });
    });
  });

  // ===========================================================================
  // verifyFactor Tests
  // ===========================================================================

  describe('verifyFactor', () => {
    it('should successfully verify a factor with signature proof', async () => {
      const mockResult = createMockVerificationResult();
      responses.set('verify_factor_proof', mockResult);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const proof: FactorProof = {
        type: 'signature',
        signature: 'verified-sig',
        message: 'challenge-message',
      };

      const result = await client.verifyFactor(VALID_DID, FACTOR_ID, proof);

      expect(result.success).toBe(true);
      expect(result.factorId).toBe(FACTOR_ID);
      expect(result.newAssuranceLevel).toBe('Verified');
      expect(mockAppClient.callZome).toHaveBeenCalledWith({
        role_name: 'mycelix_identity',
        zome_name: 'mfa',
        fn_name: 'verify_factor_proof',
        payload: {
          did: VALID_DID,
          factor_id: FACTOR_ID,
          proof,
        },
      });
    });

    it('should handle failed verification', async () => {
      const mockResult = createMockVerificationResult({
        success: false,
        newAssuranceLevel: undefined,
      });
      responses.set('verify_factor_proof', mockResult);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const proof: FactorProof = {
        type: 'signature',
        signature: 'invalid-sig',
        message: 'wrong-message',
      };

      const result = await client.verifyFactor(VALID_DID, FACTOR_ID, proof);

      expect(result.success).toBe(false);
      expect(result.newAssuranceLevel).toBeUndefined();
    });

    it('should verify with biometric proof', async () => {
      const mockResult = createMockVerificationResult();
      responses.set('verify_factor_proof', mockResult);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const proof: FactorProof = {
        type: 'biometric',
        templateHash: 'biometric-template-hash',
        response: 'biometric-response-data',
      };

      const result = await client.verifyFactor(VALID_DID, 'biometric-factor', proof);

      expect(result.success).toBe(true);
    });

    it('should throw IdentitySdkError for invalid DID format', async () => {
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const proof: FactorProof = {
        type: 'signature',
        signature: 'sig',
        message: 'msg',
      };

      await expect(client.verifyFactor(INVALID_DID, FACTOR_ID, proof)).rejects.toMatchObject({
        code: IdentitySdkErrorCode.INVALID_DID_FORMAT,
      });
    });
  });

  // ===========================================================================
  // getAssuranceLevel Tests
  // ===========================================================================

  describe('getAssuranceLevel', () => {
    const assuranceLevels: AssuranceLevel[] = [
      'Anonymous',
      'Basic',
      'Verified',
      'HighlyAssured',
      'ConstitutionallyCritical',
    ];

    it('should successfully calculate assurance level', async () => {
      const mockAssurance = { level: 'Verified' as AssuranceLevel, score: 0.5 };
      responses.set('calculate_assurance_level', mockAssurance);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.getAssuranceLevel(VALID_DID);

      expect(result.level).toBe('Verified');
      expect(result.score).toBe(0.5);
      expect(mockAppClient.callZome).toHaveBeenCalledWith({
        role_name: 'mycelix_identity',
        zome_name: 'mfa',
        fn_name: 'calculate_assurance_level',
        payload: VALID_DID,
      });
    });

    assuranceLevels.forEach((level, index) => {
      it(`should return correct score for ${level} assurance level`, async () => {
        const score = index * 0.25;
        const mockAssurance = { level, score };
        responses.set('calculate_assurance_level', mockAssurance);
        mockAppClient = createMockAppClient(responses);
        client = new MfaClient(mockAppClient, 'mycelix_identity');

        const result = await client.getAssuranceLevel(VALID_DID);

        expect(result.level).toBe(level);
        expect(result.score).toBe(score);
      });
    });

    it('should throw IdentitySdkError for invalid DID format', async () => {
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      await expect(client.getAssuranceLevel(INVALID_DID)).rejects.toMatchObject({
        code: IdentitySdkErrorCode.INVALID_DID_FORMAT,
      });
    });
  });

  // ===========================================================================
  // checkFlEligibility Tests
  // ===========================================================================

  describe('checkFlEligibility', () => {
    it('should return eligible status when requirements are met', async () => {
      const mockEligibility = createMockFlEligibility({ eligible: true });
      responses.set('check_fl_eligibility', mockEligibility);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.checkFlEligibility(VALID_DID);

      expect(result.eligible).toBe(true);
      expect(result.currentScore).toBeGreaterThanOrEqual(result.requiredScore);
      expect(result.requirements.every((r) => r.met)).toBe(true);
      expect(mockAppClient.callZome).toHaveBeenCalledWith({
        role_name: 'mycelix_identity',
        zome_name: 'mfa',
        fn_name: 'check_fl_eligibility',
        payload: VALID_DID,
      });
    });

    it('should return ineligible status with unmet requirements', async () => {
      const mockEligibility = createMockFlEligibility({
        eligible: false,
        requirements: [
          { name: 'Minimum Assurance', met: false, description: 'Must have Verified or higher' },
          { name: 'Active Factor', met: true, description: 'Must have at least one active factor' },
        ],
        currentScore: 0.25,
        requiredScore: 0.5,
      });
      responses.set('check_fl_eligibility', mockEligibility);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.checkFlEligibility(VALID_DID);

      expect(result.eligible).toBe(false);
      expect(result.currentScore).toBeLessThan(result.requiredScore);
      expect(result.requirements.some((r) => !r.met)).toBe(true);
    });

    it('should throw IdentitySdkError for invalid DID format', async () => {
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      await expect(client.checkFlEligibility(INVALID_DID)).rejects.toMatchObject({
        code: IdentitySdkErrorCode.INVALID_DID_FORMAT,
      });
    });
  });

  // ===========================================================================
  // getEnrollmentHistory Tests
  // ===========================================================================

  describe('getEnrollmentHistory', () => {
    it('should fetch enrollment history with default limit', async () => {
      const mockHistory = createMockEnrollmentHistory();
      responses.set('get_enrollment_history', mockHistory);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.getEnrollmentHistory(VALID_DID);

      expect(result).toHaveLength(3);
      expect(result[0].action).toBe('enrolled');
      expect(mockAppClient.callZome).toHaveBeenCalledWith({
        role_name: 'mycelix_identity',
        zome_name: 'mfa',
        fn_name: 'get_enrollment_history',
        payload: {
          did: VALID_DID,
          limit: 50,
        },
      });
    });

    it('should fetch enrollment history with custom limit', async () => {
      const mockHistory = createMockEnrollmentHistory().slice(0, 2);
      responses.set('get_enrollment_history', mockHistory);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.getEnrollmentHistory(VALID_DID, 2);

      expect(result).toHaveLength(2);
      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: {
            did: VALID_DID,
            limit: 2,
          },
        })
      );
    });

    it('should return empty array when no history exists', async () => {
      responses.set('get_enrollment_history', []);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.getEnrollmentHistory(VALID_DID);

      expect(result).toEqual([]);
    });

    it('should include all action types in history', async () => {
      const mockHistory: FactorEnrollment[] = [
        { factorId: 'f1', factorType: 'PrimaryKeyPair', action: 'enrolled', timestamp: Date.now() },
        { factorId: 'f1', factorType: 'PrimaryKeyPair', action: 'verified', timestamp: Date.now() },
        { factorId: 'f2', factorType: 'HardwareKey', action: 'enrolled', timestamp: Date.now() },
        { factorId: 'f2', factorType: 'HardwareKey', action: 'removed', timestamp: Date.now() },
        { factorId: 'f3', factorType: 'Biometric', action: 'failed', timestamp: Date.now() },
      ];
      responses.set('get_enrollment_history', mockHistory);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.getEnrollmentHistory(VALID_DID, 100);

      expect(result.map((r) => r.action)).toContain('enrolled');
      expect(result.map((r) => r.action)).toContain('verified');
      expect(result.map((r) => r.action)).toContain('removed');
      expect(result.map((r) => r.action)).toContain('failed');
    });

    it('should throw IdentitySdkError for invalid DID format', async () => {
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      await expect(client.getEnrollmentHistory(INVALID_DID)).rejects.toMatchObject({
        code: IdentitySdkErrorCode.INVALID_DID_FORMAT,
      });
    });
  });

  // ===========================================================================
  // getMfaSummary Tests
  // ===========================================================================

  describe('getMfaSummary', () => {
    it('should return summary with all relevant data', async () => {
      const mockState = createMockMfaState({
        factors: [
          createMockEnrolledFactor({ factorType: 'PrimaryKeyPair', category: 'possession' }),
          createMockEnrolledFactor({
            id: 'factor-2',
            factorType: 'Biometric',
            category: 'inherence',
          }),
          createMockEnrolledFactor({
            id: 'factor-3',
            factorType: 'SocialRecovery',
            category: 'social',
          }),
        ],
        assuranceLevel: 'HighlyAssured',
        flEligible: true,
      });
      responses.set('get_mfa_state', mockState);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.getMfaSummary(VALID_DID);

      expect(result).not.toBeNull();
      expect(result!.factorCount).toBe(3);
      expect(result!.assuranceLevel).toBe('HighlyAssured');
      expect(result!.flEligible).toBe(true);
      expect(result!.categories).toContain('possession');
      expect(result!.categories).toContain('inherence');
      expect(result!.categories).toContain('social');
    });

    it('should return null when MFA state does not exist', async () => {
      responses.set('get_mfa_state', null);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.getMfaSummary(VALID_DID);

      expect(result).toBeNull();
    });

    it('should deduplicate categories in summary', async () => {
      const mockState = createMockMfaState({
        factors: [
          createMockEnrolledFactor({ id: 'f1', factorType: 'PrimaryKeyPair', category: 'possession' }),
          createMockEnrolledFactor({ id: 'f2', factorType: 'HardwareKey', category: 'possession' }),
        ],
      });
      responses.set('get_mfa_state', mockState);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.getMfaSummary(VALID_DID);

      expect(result!.categories).toHaveLength(1);
      expect(result!.categories).toContain('possession');
    });

    it('should handle empty factors array', async () => {
      const mockState = createMockMfaState({ factors: [] });
      responses.set('get_mfa_state', mockState);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.getMfaSummary(VALID_DID);

      expect(result!.factorCount).toBe(0);
      expect(result!.categories).toHaveLength(0);
    });
  });

  // ===========================================================================
  // getFactorsByType Tests
  // ===========================================================================

  describe('getFactorsByType', () => {
    it('should filter factors by type', async () => {
      const mockState = createMockMfaState({
        factors: [
          createMockEnrolledFactor({ id: 'f1', factorType: 'PrimaryKeyPair' }),
          createMockEnrolledFactor({ id: 'f2', factorType: 'HardwareKey' }),
          createMockEnrolledFactor({ id: 'f3', factorType: 'PrimaryKeyPair' }),
        ],
      });
      responses.set('get_mfa_state', mockState);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.getFactorsByType(VALID_DID, 'PrimaryKeyPair');

      expect(result).toHaveLength(2);
      expect(result.every((f) => f.factorType === 'PrimaryKeyPair')).toBe(true);
    });

    it('should return empty array when no factors of type exist', async () => {
      const mockState = createMockMfaState({
        factors: [createMockEnrolledFactor({ factorType: 'PrimaryKeyPair' })],
      });
      responses.set('get_mfa_state', mockState);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.getFactorsByType(VALID_DID, 'Biometric');

      expect(result).toHaveLength(0);
    });

    it('should return empty array when state does not exist', async () => {
      responses.set('get_mfa_state', null);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.getFactorsByType(VALID_DID, 'PrimaryKeyPair');

      expect(result).toHaveLength(0);
    });
  });

  // ===========================================================================
  // Error Handling Tests
  // ===========================================================================

  describe('Error Handling', () => {
    it('should wrap zome call errors in IdentitySdkError with ZOME_CALL_FAILED code', async () => {
      mockAppClient = createMockAppClient(responses, true, 'Network timeout');
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      try {
        await client.getMfaState(VALID_DID);
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error).toBeInstanceOf(IdentitySdkError);
        expect((error as IdentitySdkError).code).toBe(IdentitySdkErrorCode.ZOME_CALL_FAILED);
        expect((error as IdentitySdkError).message).toContain('get_mfa_state');
        expect((error as IdentitySdkError).message).toContain('Network timeout');
      }
    });

    it('should include function name and payload in error details', async () => {
      mockAppClient = createMockAppClient(responses, true, 'Database error');
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      try {
        await client.createMfaState(VALID_DID, PRIMARY_KEY_HASH);
        expect.fail('Should have thrown an error');
      } catch (error) {
        const sdkError = error as IdentitySdkError;
        expect(sdkError.details).toBeDefined();
        expect(sdkError.details?.fnName).toBe('create_mfa_state');
        expect(sdkError.details?.payload).toBeDefined();
      }
    });

    it('should handle non-Error throwables gracefully', async () => {
      mockAppClient = {
        callZome: vi.fn().mockRejectedValue('string error'),
      } as unknown as AppClient;
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      try {
        await client.getMfaState(VALID_DID);
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error).toBeInstanceOf(IdentitySdkError);
        expect((error as IdentitySdkError).message).toContain('string error');
      }
    });

    it('should validate DID before making zome call', async () => {
      const callZomeSpy = vi.fn();
      mockAppClient = { callZome: callZomeSpy } as unknown as AppClient;
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      await expect(client.getMfaState(INVALID_DID)).rejects.toThrow(IdentitySdkError);
      expect(callZomeSpy).not.toHaveBeenCalled();
    });
  });

  // ===========================================================================
  // Role Name Configuration Tests
  // ===========================================================================

  describe('Role Name Configuration', () => {
    it('should use custom role name in zome calls', async () => {
      const customRoleName = 'custom_identity_role';
      responses.set('get_mfa_state', createMockMfaState());
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, customRoleName);

      await client.getMfaState(VALID_DID);

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: customRoleName,
        })
      );
    });

    it('should consistently use the same role name across all methods', async () => {
      const roleName = 'test_role';
      responses.set('get_mfa_state', createMockMfaState());
      responses.set('create_mfa_state', createMockMfaState());
      responses.set('enroll_factor', createMockEnrolledFactor());
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, roleName);

      await client.getMfaState(VALID_DID);
      await client.createMfaState(VALID_DID, PRIMARY_KEY_HASH);
      await client.enrollFactor(VALID_DID, { factorType: 'PrimaryKeyPair' });

      const calls = (mockAppClient.callZome as ReturnType<typeof vi.fn>).mock.calls;
      expect(calls.every((call) => call[0].role_name === roleName)).toBe(true);
    });
  });

  // ===========================================================================
  // Edge Cases
  // ===========================================================================

  describe('Edge Cases', () => {
    it('should handle DID with minimum valid format', async () => {
      const minimalDid = 'did:mycelix:a';
      responses.set('get_mfa_state', createMockMfaState({ did: minimalDid }));
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.getMfaState(minimalDid);

      expect(result?.did).toBe(minimalDid);
    });

    it('should handle empty metadata in factor enrollment', async () => {
      const newFactor = createMockEnrolledFactor({ metadata: {} });
      responses.set('enroll_factor', newFactor);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.enrollFactor(VALID_DID, {
        factorType: 'PrimaryKeyPair',
      });

      expect(result.metadata).toEqual({});
    });

    it('should handle factor with null lastUsedAt', async () => {
      const mockState = createMockMfaState({
        factors: [createMockEnrolledFactor({ lastUsedAt: null })],
      });
      responses.set('get_mfa_state', mockState);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.getMfaState(VALID_DID);

      expect(result?.factors[0].lastUsedAt).toBeNull();
    });

    it('should handle very long factor ID', async () => {
      const longFactorId = 'a'.repeat(256);
      const mockChallenge = createMockChallenge({ factorId: longFactorId });
      responses.set('generate_verification_challenge', mockChallenge);
      mockAppClient = createMockAppClient(responses);
      client = new MfaClient(mockAppClient, 'mycelix_identity');

      const result = await client.generateChallenge(VALID_DID, longFactorId);

      expect(result.factorId).toBe(longFactorId);
    });
  });
});
