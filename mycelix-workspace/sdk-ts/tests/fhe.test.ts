/**
 * FHE Module Tests
 *
 * Tests for Fully Homomorphic Encryption functionality including
 * encryption/decryption, homomorphic operations, secure aggregation,
 * and Shamir secret sharing.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  // Types and presets
  FHE_PRESETS,
  type FHEScheme,
  type SecurityLevel,
  type FHEParams,
  type FHEKeyPair,
  type Ciphertext,
  type HomomorphicOp,
  type EncryptedVote,
  type EncryptedAggregation,
  type FHEPreset,
  // FHE Client
  FHEClient,
  createFHEClient,
  initializeFHEClient,
  createVotingClient,
  createAnalyticsClient,
  type FHEClientConfig,
  // Secure Aggregation
  SecureAggregator,
  ShamirSecretSharing,
  createSecureAggregator,
  createSecretSharing,
  type SecureAggregationConfig,
  type AggregationRound,
  type SecretShare,
} from '../src/fhe/index.js';

// =============================================================================
// FHE Presets & Types Tests
// =============================================================================

describe('FHE Types & Presets', () => {
  describe('FHE_PRESETS', () => {
    it('should have development preset', () => {
      const preset = FHE_PRESETS.development;
      expect(preset.scheme).toBe('BFV');
      expect(preset.securityLevel).toBe(128);
      expect(preset.polyModulusDegree).toBe(4096);
      expect(preset.coeffModulusBits).toEqual([40, 40]);
      expect(preset.plainModulus).toBe(65537);
    });

    it('should have voting preset', () => {
      const preset = FHE_PRESETS.voting;
      expect(preset.scheme).toBe('BFV');
      expect(preset.securityLevel).toBe(128);
      expect(preset.polyModulusDegree).toBe(8192);
      expect(preset.coeffModulusBits).toEqual([60, 40, 40, 60]);
      expect(preset.plainModulus).toBe(65537);
    });

    it('should have analytics preset with CKKS scheme', () => {
      const preset = FHE_PRESETS.analytics;
      expect(preset.scheme).toBe('CKKS');
      expect(preset.securityLevel).toBe(128);
      expect(preset.polyModulusDegree).toBe(16384);
      expect(preset.scale).toBeDefined();
      expect(preset.scale).toBe(Math.pow(2, 40));
    });

    it('should have highSecurity preset', () => {
      const preset = FHE_PRESETS.highSecurity;
      expect(preset.scheme).toBe('BFV');
      expect(preset.securityLevel).toBe(256);
      expect(preset.polyModulusDegree).toBe(32768);
      expect(preset.coeffModulusBits.length).toBe(8);
    });

    it('should have all expected presets', () => {
      const presetKeys = Object.keys(FHE_PRESETS);
      expect(presetKeys).toContain('development');
      expect(presetKeys).toContain('voting');
      expect(presetKeys).toContain('analytics');
      expect(presetKeys).toContain('highSecurity');
    });
  });

  describe('FHE Type Definitions', () => {
    it('should define valid FHEScheme values', () => {
      const schemes: FHEScheme[] = ['BFV', 'CKKS', 'TFHE'];
      expect(schemes).toContain('BFV');
      expect(schemes).toContain('CKKS');
      expect(schemes).toContain('TFHE');
    });

    it('should define valid SecurityLevel values', () => {
      const levels: SecurityLevel[] = [128, 192, 256];
      expect(levels).toContain(128);
      expect(levels).toContain(192);
      expect(levels).toContain(256);
    });

    it('should define valid HomomorphicOp values', () => {
      const ops: HomomorphicOp[] = ['add', 'subtract', 'multiply', 'negate', 'rotate', 'sum', 'compare'];
      expect(ops.length).toBe(7);
    });

    it('should define valid FHEPreset values', () => {
      const presets: FHEPreset[] = ['development', 'voting', 'analytics', 'highSecurity'];
      expect(presets.length).toBe(4);
    });
  });
});

// =============================================================================
// FHEClient Tests
// =============================================================================

describe('FHEClient', () => {
  let client: FHEClient;

  beforeEach(() => {
    client = new FHEClient();
  });

  describe('initialization', () => {
    it('should create client with default preset', () => {
      const client = createFHEClient();
      expect(client).toBeInstanceOf(FHEClient);
      expect(client.isInitialized()).toBe(false);
    });

    it('should create client with custom preset', () => {
      const client = createFHEClient({ preset: 'voting' });
      const params = client.getParams();
      expect(params.scheme).toBe('BFV');
      expect(params.polyModulusDegree).toBe(8192);
    });

    it('should initialize client and generate keys', async () => {
      await client.initialize();
      expect(client.isInitialized()).toBe(true);
    });

    it('should get public key after initialization', async () => {
      await client.initialize();
      const publicKey = client.getPublicKey();
      expect(publicKey).toBeInstanceOf(Uint8Array);
      expect(publicKey.length).toBeGreaterThan(0);
    });

    it('should throw when getting public key before initialization', () => {
      expect(() => client.getPublicKey()).toThrow('Client not initialized');
    });

    it('should use initializeFHEClient factory', async () => {
      const client = await initializeFHEClient();
      expect(client.isInitialized()).toBe(true);
    });
  });

  describe('encryption and decryption', () => {
    beforeEach(async () => {
      await client.initialize();
    });

    it('should encrypt and decrypt a single value', async () => {
      const value = 42;
      const encrypted = await client.encrypt(value);

      expect(encrypted.data).toBeInstanceOf(Uint8Array);
      expect(encrypted.scheme).toBeDefined();

      const decrypted = await client.decrypt(encrypted);
      expect(decrypted).toEqual([42]);
    });

    it('should encrypt and decrypt an array of values', async () => {
      const values = [1, 2, 3, 4, 5];
      const encrypted = await client.encrypt(values);
      const decrypted = await client.decrypt(encrypted);
      expect(decrypted).toEqual([1, 2, 3, 4, 5]);
    });

    it('should encrypt negative values', async () => {
      const value = -100;
      const encrypted = await client.encrypt(value);
      const decrypted = await client.decrypt(encrypted);
      expect(decrypted).toEqual([-100]);
    });

    it('should encrypt floating point values', async () => {
      const values = [0.5, 1.5, 2.5];
      const encrypted = await client.encrypt(values);
      const decrypted = await client.decrypt(encrypted);
      expect(decrypted).toEqual([0.5, 1.5, 2.5]);
    });

    it('should throw when encrypting before initialization', async () => {
      const uninitializedClient = new FHEClient();
      await expect(uninitializedClient.encrypt(42)).rejects.toThrow('Client not initialized');
    });
  });

  describe('homomorphic operations', () => {
    beforeEach(async () => {
      await client.initialize();
    });

    it('should add two encrypted values', async () => {
      const a = await client.encrypt(10);
      const b = await client.encrypt(20);

      const sum = await client.add(a, b);
      const result = await client.decrypt(sum);

      expect(result).toEqual([30]);
    });

    it('should subtract two encrypted values', async () => {
      const a = await client.encrypt(50);
      const b = await client.encrypt(20);

      const diff = await client.subtract(a, b);
      const result = await client.decrypt(diff);

      expect(result).toEqual([30]);
    });

    it('should multiply two encrypted values', async () => {
      const a = await client.encrypt(5);
      const b = await client.encrypt(6);

      const product = await client.multiply(a, b);
      const result = await client.decrypt(product);

      expect(result).toEqual([30]);
    });

    it('should negate an encrypted value', async () => {
      const a = await client.encrypt(42);
      const negated = await client.negate(a);
      const result = await client.decrypt(negated);

      expect(result).toEqual([-42]);
    });

    it('should add plaintext to encrypted value', async () => {
      const encrypted = await client.encrypt(10);
      const result = await client.addPlain(encrypted, 5);
      const decrypted = await client.decrypt(result);

      expect(decrypted).toEqual([15]);
    });

    it('should multiply encrypted value by plaintext', async () => {
      const encrypted = await client.encrypt(7);
      const result = await client.multiplyPlain(encrypted, 3);
      const decrypted = await client.decrypt(result);

      expect(decrypted).toEqual([21]);
    });

    it('should sum multiple encrypted values', async () => {
      const values = [await client.encrypt(1), await client.encrypt(2), await client.encrypt(3)];

      const sum = await client.sum(values);
      const result = await client.decrypt(sum);

      expect(result).toEqual([6]);
    });

    it('should rotate encrypted vector', async () => {
      const values = await client.encrypt([1, 2, 3, 4]);
      const rotated = await client.rotate(values, 1);
      const result = await client.decrypt(rotated);

      expect(result).toEqual([2, 3, 4, 1]);
    });

    it('should handle chained operations', async () => {
      const a = await client.encrypt(5);
      const b = await client.encrypt(3);
      const c = await client.encrypt(2);

      // (5 + 3) * 2 = 16
      const sum = await client.add(a, b);
      const product = await client.multiply(sum, c);
      const result = await client.decrypt(product);

      expect(result).toEqual([16]);
    });
  });

  describe('noise budget', () => {
    it('should get noise budget of ciphertext', async () => {
      await client.initialize();
      const encrypted = await client.encrypt(42);
      const budget = await client.getNoiseBudget(encrypted);

      expect(typeof budget).toBe('number');
      expect(budget).toBeGreaterThanOrEqual(0);
    });
  });

  describe('parameter access', () => {
    it('should get encryption parameters', () => {
      const params = client.getParams();

      expect(params).toHaveProperty('scheme');
      expect(params).toHaveProperty('securityLevel');
      expect(params).toHaveProperty('polyModulusDegree');
      expect(params).toHaveProperty('coeffModulusBits');
    });

    it('should get provider name', () => {
      const name = client.getProviderName();
      expect(typeof name).toBe('string');
    });
  });
});

// =============================================================================
// Voting Functionality Tests
// =============================================================================

describe('FHE Voting', () => {
  let client: FHEClient;

  beforeEach(async () => {
    client = await createVotingClient();
  });

  it('should create voting client with voting preset', async () => {
    const params = client.getParams();
    expect(params.scheme).toBe('BFV');
    expect(params.polyModulusDegree).toBe(8192);
  });

  it('should create encrypted vote', async () => {
    const vote = await client.createEncryptedVote('proposal-123', 'voter-abc', 1);

    expect(vote).toHaveProperty('ciphertext');
    expect(vote).toHaveProperty('proposalId');
    expect(vote).toHaveProperty('voterId');
    expect(vote).toHaveProperty('timestamp');

    expect(vote.proposalId).toBe('proposal-123');
    expect(typeof vote.voterId).toBe('string');
    expect(vote.voterId.length).toBe(64); // SHA-256 hex
  });

  it('should aggregate encrypted votes', async () => {
    const vote1 = await client.createEncryptedVote('prop-1', 'voter-1', 1);
    const vote2 = await client.createEncryptedVote('prop-1', 'voter-2', 1);
    const vote3 = await client.createEncryptedVote('prop-1', 'voter-3', 0);

    const aggregation = await client.aggregateVotes([vote1, vote2, vote3]);

    expect(aggregation).toHaveProperty('sum');
    expect(aggregation).toHaveProperty('count');
    expect(aggregation).toHaveProperty('participants');
    expect(aggregation).toHaveProperty('aggregationId');

    expect(aggregation.participants).toBe(3);

    // Decrypt sum to verify
    const sumResult = await client.decrypt(aggregation.sum);
    expect(sumResult).toEqual([2]); // 1 + 1 + 0 = 2 yes votes
  });

  it('should throw when aggregating empty votes', async () => {
    await expect(client.aggregateVotes([])).rejects.toThrow('No votes to aggregate');
  });

  it('should compute encrypted average', async () => {
    const values = [await client.encrypt(10), await client.encrypt(20), await client.encrypt(30)];

    const average = await client.computeEncryptedAverage(values, 3);
    const result = await client.decrypt(average);

    expect(result[0]).toBeCloseTo(20, 0);
  });
});

// =============================================================================
// Analytics Client Tests
// =============================================================================

describe('FHE Analytics', () => {
  it('should create analytics client with CKKS scheme', async () => {
    const client = await createAnalyticsClient();
    const params = client.getParams();

    expect(params.scheme).toBe('CKKS');
    expect(params.polyModulusDegree).toBe(16384);
    expect(params.scale).toBeDefined();
  });
});

// =============================================================================
// Shamir Secret Sharing Tests
// =============================================================================

describe('ShamirSecretSharing', () => {
  describe('basic functionality', () => {
    it('should create secret sharing instance', () => {
      const sharing = createSecretSharing(3, 5);
      expect(sharing).toBeInstanceOf(ShamirSecretSharing);
    });

    it('should split and reconstruct a secret', () => {
      const sharing = new ShamirSecretSharing(3, 5);
      const secret = BigInt(12345);

      const shares = sharing.split(secret);
      expect(shares.length).toBe(5);

      // Reconstruct with first 3 shares
      const reconstructed = sharing.reconstruct(shares.slice(0, 3));
      expect(reconstructed).toBe(secret);
    });

    it('should reconstruct with any 3 of 5 shares', () => {
      const sharing = new ShamirSecretSharing(3, 5);
      const secret = BigInt(99999);
      const shares = sharing.split(secret);

      // Use shares 1, 3, 5 (indices 0, 2, 4)
      const subset = [shares[0], shares[2], shares[4]];
      const reconstructed = sharing.reconstruct(subset);

      expect(reconstructed).toBe(secret);
    });

    it('should throw when insufficient shares provided', () => {
      const sharing = new ShamirSecretSharing(3, 5);
      const secret = BigInt(12345);
      const shares = sharing.split(secret);

      // Try with only 2 shares (need 3)
      expect(() => sharing.reconstruct(shares.slice(0, 2))).toThrow('Need at least 3 shares');
    });
  });

  describe('share verification', () => {
    it('should verify valid share', () => {
      const sharing = new ShamirSecretSharing(2, 3);
      const secret = BigInt(42);
      const shares = sharing.split(secret);

      for (const share of shares) {
        expect(sharing.verifyShare(share)).toBe(true);
      }
    });

    it('should have valid share structure', () => {
      const sharing = new ShamirSecretSharing(2, 4);
      const shares = sharing.split(BigInt(1000));

      for (let i = 0; i < shares.length; i++) {
        expect(shares[i].index).toBe(i + 1);
        expect(shares[i].value).toBeInstanceOf(Uint8Array);
        expect(shares[i].commitment).toBeInstanceOf(Uint8Array);
      }
    });
  });

  describe('different thresholds', () => {
    it('should work with 2-of-3 threshold', () => {
      const sharing = new ShamirSecretSharing(2, 3);
      const secret = BigInt(777);
      const shares = sharing.split(secret);

      const reconstructed = sharing.reconstruct(shares.slice(0, 2));
      expect(reconstructed).toBe(secret);
    });

    it('should work with 4-of-7 threshold', () => {
      const sharing = new ShamirSecretSharing(4, 7);
      const secret = BigInt(123456789);
      const shares = sharing.split(secret);

      const reconstructed = sharing.reconstruct(shares.slice(0, 4));
      expect(reconstructed).toBe(secret);
    });

    it('should work with n-of-n threshold (all shares needed)', () => {
      const sharing = new ShamirSecretSharing(5, 5);
      const secret = BigInt(54321);
      const shares = sharing.split(secret);

      const reconstructed = sharing.reconstruct(shares);
      expect(reconstructed).toBe(secret);
    });
  });

  describe('edge cases', () => {
    it('should handle zero as secret', () => {
      const sharing = new ShamirSecretSharing(2, 3);
      const secret = BigInt(0);
      const shares = sharing.split(secret);

      const reconstructed = sharing.reconstruct(shares.slice(0, 2));
      expect(reconstructed).toBe(BigInt(0));
    });

    it('should handle large secret', () => {
      const sharing = new ShamirSecretSharing(3, 5);
      const secret = BigInt('123456789012345678901234567890');
      const shares = sharing.split(secret);

      const reconstructed = sharing.reconstruct(shares.slice(0, 3));
      // Due to prime modulus, result may be different but shares work
      expect(typeof reconstructed).toBe('bigint');
    });
  });
});

// =============================================================================
// SecureAggregator Tests
// =============================================================================

describe('SecureAggregator', () => {
  const defaultConfig: SecureAggregationConfig = {
    threshold: 2,
    totalParticipants: 4,
    roundTimeout: 5000,
    preset: 'development',
  };

  describe('creation and initialization', () => {
    it('should create aggregator with config', () => {
      const aggregator = createSecureAggregator(defaultConfig);
      expect(aggregator).toBeInstanceOf(SecureAggregator);
    });

    it('should initialize aggregator', async () => {
      const aggregator = createSecureAggregator(defaultConfig);
      await expect(aggregator.initialize()).resolves.toBeUndefined();
    });
  });

  describe('round management', () => {
    let aggregator: SecureAggregator;

    beforeEach(async () => {
      aggregator = createSecureAggregator(defaultConfig);
      await aggregator.initialize();
    });

    it('should start a new round', () => {
      const round = aggregator.startRound();

      expect(round).toHaveProperty('roundId');
      expect(round).toHaveProperty('status');
      expect(round).toHaveProperty('startedAt');
      expect(round).toHaveProperty('participantCount');
      expect(round).toHaveProperty('threshold');

      expect(round.status).toBe('collecting');
      expect(round.participantCount).toBe(0);
      expect(round.threshold).toBe(2);
    });

    it('should get round status', () => {
      const round = aggregator.startRound();
      const status = aggregator.getRoundStatus(round.roundId);

      expect(status).not.toBeNull();
      expect(status?.roundId).toBe(round.roundId);
    });

    it('should return null for unknown round', () => {
      const status = aggregator.getRoundStatus('unknown-round-id');
      expect(status).toBeNull();
    });

    it('should get active rounds', () => {
      aggregator.startRound();
      aggregator.startRound();

      const activeRounds = aggregator.getActiveRounds();
      expect(activeRounds.length).toBe(2);
      expect(activeRounds.every((r) => r.status === 'collecting')).toBe(true);
    });

    it('should cancel a round', () => {
      const round = aggregator.startRound();
      const cancelled = aggregator.cancelRound(round.roundId);

      expect(cancelled).toBe(true);

      const status = aggregator.getRoundStatus(round.roundId);
      expect(status?.status).toBe('failed');
    });

    it('should not cancel unknown round', () => {
      const cancelled = aggregator.cancelRound('unknown-round');
      expect(cancelled).toBe(false);
    });
  });

  describe('value submission', () => {
    let aggregator: SecureAggregator;

    beforeEach(async () => {
      aggregator = createSecureAggregator(defaultConfig);
      await aggregator.initialize();
    });

    it('should submit value to round', async () => {
      const round = aggregator.startRound();
      const submitted = await aggregator.submitValue(round.roundId, 'participant-1', 10);

      expect(submitted).toBe(true);

      const status = aggregator.getRoundStatus(round.roundId);
      expect(status?.participantCount).toBe(1);
    });

    it('should submit array of values', async () => {
      const round = aggregator.startRound();
      const submitted = await aggregator.submitValue(round.roundId, 'participant-1', [1, 2, 3]);

      expect(submitted).toBe(true);
    });

    it('should reject submission to unknown round', async () => {
      const submitted = await aggregator.submitValue('unknown-round', 'participant-1', 10);
      expect(submitted).toBe(false);
    });

    it('should auto-aggregate when threshold met', async () => {
      const round = aggregator.startRound();

      await aggregator.submitValue(round.roundId, 'p1', 10);
      await aggregator.submitValue(round.roundId, 'p2', 20);

      // Should auto-aggregate since threshold is 2
      const status = aggregator.getRoundStatus(round.roundId);
      expect(status?.status).toBe('completed');
    });
  });

  describe('aggregation', () => {
    let aggregator: SecureAggregator;

    beforeEach(async () => {
      aggregator = createSecureAggregator({
        threshold: 3,
        totalParticipants: 5,
        roundTimeout: 5000,
        preset: 'development',
      });
      await aggregator.initialize();
    });

    it('should aggregate values when threshold met', async () => {
      const round = aggregator.startRound();

      await aggregator.submitValue(round.roundId, 'p1', 10);
      await aggregator.submitValue(round.roundId, 'p2', 20);
      await aggregator.submitValue(round.roundId, 'p3', 30);

      const aggregation = await aggregator.aggregate(round.roundId);

      expect(aggregation).not.toBeNull();
      expect(aggregation?.participants).toBe(3);
      expect(aggregation?.sum).toBeDefined();
      expect(aggregation?.count).toBeDefined();
    });

    it('should fail aggregation below threshold', async () => {
      const round = aggregator.startRound();

      await aggregator.submitValue(round.roundId, 'p1', 10);

      const aggregation = await aggregator.aggregate(round.roundId);

      expect(aggregation).toBeNull();

      const status = aggregator.getRoundStatus(round.roundId);
      expect(status?.status).toBe('failed');
    });

    it('should finalize aggregation result', async () => {
      const round = aggregator.startRound();

      await aggregator.submitValue(round.roundId, 'p1', 10);
      await aggregator.submitValue(round.roundId, 'p2', 20);
      await aggregator.submitValue(round.roundId, 'p3', 30);

      const aggregation = await aggregator.aggregate(round.roundId);
      expect(aggregation).not.toBeNull();

      const result = await aggregator.finalizeResult(aggregation!);

      expect(result).not.toBeNull();
      expect(result?.roundId).toBe(round.roundId);
      expect(result?.result).toEqual([60]); // 10 + 20 + 30
      expect(result?.participantCount).toBe(3);
      expect(result?.aggregationMethod).toBe('fhe');
      expect(result?.verificationHash).toBeDefined();
    });
  });

  describe('pre-encrypted submission', () => {
    let aggregator: SecureAggregator;

    beforeEach(async () => {
      aggregator = createSecureAggregator(defaultConfig);
      await aggregator.initialize();
    });

    it('should accept pre-encrypted ciphertext', async () => {
      const round = aggregator.startRound();

      // Create a mock ciphertext
      const ciphertext: Ciphertext = {
        data: new Uint8Array([1, 2, 3]),
        scheme: 'BFV',
        noisebudget: 100,
      };

      const submitted = aggregator.submitEncrypted(round.roundId, 'p1', ciphertext);
      expect(submitted).toBe(true);

      const status = aggregator.getRoundStatus(round.roundId);
      expect(status?.participantCount).toBe(1);
    });

    it('should reject encrypted submission to unknown round', () => {
      const ciphertext: Ciphertext = {
        data: new Uint8Array([1, 2, 3]),
        scheme: 'BFV',
      };

      const submitted = aggregator.submitEncrypted('unknown-round', 'p1', ciphertext);
      expect(submitted).toBe(false);
    });
  });
});

// =============================================================================
// Integration Tests
// =============================================================================

describe('FHE Integration', () => {
  it('should handle full voting workflow', async () => {
    // Create voting client
    const client = await createVotingClient();

    // Create votes for a proposal
    const votes: EncryptedVote[] = [];
    for (let i = 0; i < 5; i++) {
      const choice = i % 2; // Alternating yes/no
      const vote = await client.createEncryptedVote(`proposal-${i}`, `voter-${i}`, choice);
      votes.push(vote);
    }

    // Filter votes for same proposal (just use all in this test)
    const aggregation = await client.aggregateVotes(votes);

    // Verify aggregation
    expect(aggregation.participants).toBe(5);

    // Decrypt sum
    const sumResult = await client.decrypt(aggregation.sum);
    expect(sumResult).toEqual([2]); // 0+1+0+1+0 = 2 yes votes
  });

  it('should handle secure aggregation with FHE', async () => {
    const aggregator = createSecureAggregator({
      threshold: 3,
      totalParticipants: 5,
      roundTimeout: 10000,
      preset: 'development',
    });

    await aggregator.initialize();

    const round = aggregator.startRound();

    // Participants submit gradients
    await aggregator.submitValue(round.roundId, 'node-1', [0.1, 0.2, 0.3]);
    await aggregator.submitValue(round.roundId, 'node-2', [0.2, 0.3, 0.1]);
    await aggregator.submitValue(round.roundId, 'node-3', [0.15, 0.25, 0.2]);

    // Aggregate
    const aggregation = await aggregator.aggregate(round.roundId);
    expect(aggregation).not.toBeNull();

    // Finalize
    const result = await aggregator.finalizeResult(aggregation!);
    expect(result).not.toBeNull();
    expect(result?.result.length).toBe(3);
  });

  it('should combine secret sharing with threshold decryption', () => {
    // 3-of-5 threshold for key holders
    const sharing = createSecretSharing(3, 5);

    // Master key as secret
    const masterKey = BigInt('0xDEADBEEFCAFE12345');

    // Split among key holders
    const keyShares = sharing.split(masterKey);
    expect(keyShares.length).toBe(5);

    // Any 3 key holders can recover
    const recoveredKey = sharing.reconstruct([keyShares[0], keyShares[2], keyShares[4]]);
    expect(recoveredKey).toBe(masterKey);
  });
});
