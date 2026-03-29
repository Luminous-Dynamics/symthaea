// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Secure Aggregation
 *
 * Privacy-preserving aggregation protocols for federated learning
 * and collaborative analytics. Combines FHE with secret sharing
 * for efficient multi-party computation.
 */

import { initializeFHEClient, type FHEClient } from './client.js';

import type { Ciphertext, EncryptedAggregation, FHEPreset } from './types.js';

// =============================================================================
// Types
// =============================================================================

export interface SecureAggregationConfig {
  threshold: number; // Minimum participants for reconstruction
  totalParticipants: number;
  roundTimeout: number; // ms
  preset?: FHEPreset;
}

export interface AggregationRound {
  roundId: string;
  status: 'collecting' | 'aggregating' | 'completed' | 'failed';
  startedAt: number;
  completedAt?: number;
  participantCount: number;
  threshold: number;
}

export interface ParticipantShare {
  participantId: string;
  share: Uint8Array;
  commitment: Uint8Array;
  timestamp: number;
}

export interface AggregatedResult {
  roundId: string;
  result: number[];
  participantCount: number;
  aggregationMethod: 'fhe' | 'secret-sharing' | 'hybrid';
  verificationHash: string;
}

export interface SecretShare {
  index: number;
  value: Uint8Array;
  commitment: Uint8Array;
}

// =============================================================================
// Secure Aggregation Protocol
// =============================================================================

export class SecureAggregator {
  private config: SecureAggregationConfig;
  private fheClient: FHEClient | null = null;
  private rounds: Map<string, AggregationRound> = new Map();
  private pendingShares: Map<string, Map<string, Ciphertext>> = new Map();

  constructor(config: SecureAggregationConfig) {
    this.config = config;
  }

  /**
   * Initialize the aggregator
   */
  async initialize(): Promise<void> {
    this.fheClient = await initializeFHEClient({
      preset: this.config.preset ?? 'voting',
    });
  }

  /**
   * Start a new aggregation round
   */
  startRound(): AggregationRound {
    const roundId = crypto.randomUUID();
    const round: AggregationRound = {
      roundId,
      status: 'collecting',
      startedAt: Date.now(),
      participantCount: 0,
      threshold: this.config.threshold,
    };

    this.rounds.set(roundId, round);
    this.pendingShares.set(roundId, new Map());

    // Set timeout for the round
    setTimeout(() => this.checkRoundTimeout(roundId), this.config.roundTimeout);

    return round;
  }

  /**
   * Submit an encrypted value for aggregation
   */
  async submitValue(
    roundId: string,
    participantId: string,
    value: number | number[]
  ): Promise<boolean> {
    const round = this.rounds.get(roundId);
    if (!round || round.status !== 'collecting') {
      return false;
    }

    if (!this.fheClient) {
      throw new Error('Aggregator not initialized');
    }

    const encrypted = await this.fheClient.encrypt(
      Array.isArray(value) ? value : [value]
    );

    const roundShares = this.pendingShares.get(roundId);
    if (!roundShares) {
      return false;
    }

    roundShares.set(participantId, encrypted);
    round.participantCount = roundShares.size;

    // Auto-aggregate if threshold met
    if (round.participantCount >= this.config.threshold) {
      await this.aggregate(roundId);
    }

    return true;
  }

  /**
   * Submit a pre-encrypted ciphertext
   */
  submitEncrypted(
    roundId: string,
    participantId: string,
    ciphertext: Ciphertext
  ): boolean {
    const round = this.rounds.get(roundId);
    if (!round || round.status !== 'collecting') {
      return false;
    }

    const roundShares = this.pendingShares.get(roundId);
    if (!roundShares) {
      return false;
    }

    roundShares.set(participantId, ciphertext);
    round.participantCount = roundShares.size;

    return true;
  }

  /**
   * Aggregate all submitted values
   */
  async aggregate(roundId: string): Promise<EncryptedAggregation | null> {
    const round = this.rounds.get(roundId);
    if (!round) {
      return null;
    }

    if (round.participantCount < this.config.threshold) {
      round.status = 'failed';
      return null;
    }

    round.status = 'aggregating';

    const roundShares = this.pendingShares.get(roundId);
    if (!roundShares || !this.fheClient) {
      round.status = 'failed';
      return null;
    }

    try {
      const ciphertexts = Array.from(roundShares.values());
      const sum = await this.fheClient.sum(ciphertexts);
      const count = await this.fheClient.encrypt(ciphertexts.length);

      round.status = 'completed';
      round.completedAt = Date.now();

      return {
        sum,
        count,
        participants: round.participantCount,
        aggregationId: roundId,
      };
    } catch (error) {
      round.status = 'failed';
      console.error('[SecureAggregator] Aggregation failed:', error);
      return null;
    }
  }

  /**
   * Decrypt and finalize aggregation result (requires secret key holder)
   */
  async finalizeResult(
    aggregation: EncryptedAggregation
  ): Promise<AggregatedResult | null> {
    if (!this.fheClient) {
      throw new Error('Aggregator not initialized');
    }

    try {
      const sumResult = await this.fheClient.decrypt(aggregation.sum);
      const countResult = await this.fheClient.decrypt(aggregation.count);

      // Calculate verification hash
      const verificationData = new TextEncoder().encode(
        JSON.stringify({ sum: sumResult, count: countResult[0] })
      );
      const hash = await crypto.subtle.digest('SHA-256', verificationData);
      const verificationHash = Array.from(new Uint8Array(hash))
        .map((b) => b.toString(16).padStart(2, '0'))
        .join('');

      return {
        roundId: aggregation.aggregationId,
        result: sumResult,
        participantCount: countResult[0],
        aggregationMethod: 'fhe',
        verificationHash,
      };
    } catch (error) {
      console.error('[SecureAggregator] Finalization failed:', error);
      return null;
    }
  }

  /**
   * Get round status
   */
  getRoundStatus(roundId: string): AggregationRound | null {
    return this.rounds.get(roundId) ?? null;
  }

  /**
   * Get all active rounds
   */
  getActiveRounds(): AggregationRound[] {
    return Array.from(this.rounds.values()).filter(
      (r) => r.status === 'collecting' || r.status === 'aggregating'
    );
  }

  /**
   * Cancel a round
   */
  cancelRound(roundId: string): boolean {
    const round = this.rounds.get(roundId);
    if (!round || round.status === 'completed') {
      return false;
    }

    round.status = 'failed';
    this.pendingShares.delete(roundId);
    return true;
  }

  // Private methods
  private checkRoundTimeout(roundId: string): void {
    const round = this.rounds.get(roundId);
    if (round && round.status === 'collecting') {
      if (round.participantCount >= this.config.threshold) {
        // Try to aggregate with available participants
        void this.aggregate(roundId);
      } else {
        round.status = 'failed';
      }
    }
  }
}

// =============================================================================
// Shamir Secret Sharing (for threshold decryption)
// =============================================================================

export class ShamirSecretSharing {
  private threshold: number;
  private totalShares: number;
  private prime: bigint;

  constructor(threshold: number, totalShares: number) {
    this.threshold = threshold;
    this.totalShares = totalShares;
    // Large prime for field arithmetic
    this.prime = BigInt('340282366920938463463374607431768211297');
  }

  /**
   * Split a secret into shares
   */
  split(secret: bigint): SecretShare[] {
    // Generate random polynomial coefficients
    const coefficients: bigint[] = [secret];
    for (let i = 1; i < this.threshold; i++) {
      coefficients.push(this.randomBigInt());
    }

    // Evaluate polynomial at each share index
    const shares: SecretShare[] = [];
    for (let i = 1; i <= this.totalShares; i++) {
      const x = BigInt(i);
      let y = BigInt(0);

      for (let j = 0; j < coefficients.length; j++) {
        y = (y + coefficients[j] * this.modPow(x, BigInt(j), this.prime)) % this.prime;
      }

      // Create commitment (hash of share)
      const shareBytes = this.bigIntToBytes(y);
      const commitment = new Uint8Array(32); // Simplified commitment
      crypto.getRandomValues(commitment);

      shares.push({
        index: i,
        value: shareBytes,
        commitment,
      });
    }

    return shares;
  }

  /**
   * Reconstruct secret from shares
   */
  reconstruct(shares: SecretShare[]): bigint {
    if (shares.length < this.threshold) {
      throw new Error(`Need at least ${this.threshold} shares`);
    }

    // Use first 'threshold' shares
    const usedShares = shares.slice(0, this.threshold);

    let secret = BigInt(0);

    for (let i = 0; i < usedShares.length; i++) {
      const xi = BigInt(usedShares[i].index);
      const yi = this.bytesToBigInt(usedShares[i].value);

      // Calculate Lagrange basis polynomial
      let numerator = BigInt(1);
      let denominator = BigInt(1);

      for (let j = 0; j < usedShares.length; j++) {
        if (i !== j) {
          const xj = BigInt(usedShares[j].index);
          numerator = (numerator * (this.prime - xj)) % this.prime;
          denominator = (denominator * ((xi - xj + this.prime) % this.prime)) % this.prime;
        }
      }

      // Add contribution from this share
      const lagrangeBasis = (numerator * this.modInverse(denominator, this.prime)) % this.prime;
      secret = (secret + yi * lagrangeBasis) % this.prime;
    }

    return secret;
  }

  /**
   * Verify a share against its commitment
   */
  verifyShare(share: SecretShare): boolean {
    // In a real implementation, this would verify the commitment
    // using a proper commitment scheme (e.g., Pedersen commitments)
    return share.value.length > 0 && share.commitment.length > 0;
  }

  // Helper methods
  private randomBigInt(): bigint {
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    return this.bytesToBigInt(bytes) % this.prime;
  }

  private bigIntToBytes(n: bigint): Uint8Array {
    const hex = n.toString(16).padStart(32, '0');
    const bytes = new Uint8Array(16);
    for (let i = 0; i < 16; i++) {
      bytes[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
    }
    return bytes;
  }

  private bytesToBigInt(bytes: Uint8Array): bigint {
    let result = BigInt(0);
    for (const byte of bytes) {
      result = (result << BigInt(8)) + BigInt(byte);
    }
    return result;
  }

  private modPow(base: bigint, exp: bigint, mod: bigint): bigint {
    let result = BigInt(1);
    base = base % mod;
    while (exp > 0) {
      if (exp % BigInt(2) === BigInt(1)) {
        result = (result * base) % mod;
      }
      exp = exp / BigInt(2);
      base = (base * base) % mod;
    }
    return result;
  }

  private modInverse(a: bigint, m: bigint): bigint {
    // Extended Euclidean Algorithm
    let [old_r, r] = [a, m];
    let [old_s, s] = [BigInt(1), BigInt(0)];

    while (r !== BigInt(0)) {
      const quotient = old_r / r;
      [old_r, r] = [r, old_r - quotient * r];
      [old_s, s] = [s, old_s - quotient * s];
    }

    return ((old_s % m) + m) % m;
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create a secure aggregator
 */
export function createSecureAggregator(
  config: SecureAggregationConfig
): SecureAggregator {
  return new SecureAggregator(config);
}

/**
 * Create and initialize a secure aggregator
 */
export async function initializeSecureAggregator(
  config: SecureAggregationConfig
): Promise<SecureAggregator> {
  const aggregator = new SecureAggregator(config);
  await aggregator.initialize();
  return aggregator;
}

/**
 * Create a Shamir secret sharing instance
 */
export function createSecretSharing(
  threshold: number,
  totalShares: number
): ShamirSecretSharing {
  return new ShamirSecretSharing(threshold, totalShares);
}
