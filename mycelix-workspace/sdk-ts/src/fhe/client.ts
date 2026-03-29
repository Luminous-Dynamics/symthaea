// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * FHE Client
 *
 * High-level client for Fully Homomorphic Encryption operations.
 * Provides encrypted computation for privacy-preserving analytics,
 * voting, and secure aggregation.
 *
 * Note: This implementation uses a simplified simulation for demonstration.
 * Production use should integrate with TFHE-rs, Microsoft SEAL, or OpenFHE
 * via WebAssembly bindings.
 */

import {
  type FHEParams,
  type FHEKeyPair,
  type Ciphertext,
  type FHEScheme,
  type EncryptedVote,
  type EncryptedAggregation,
  type FHEProvider,
  type FHEPreset,
  FHE_PRESETS,
} from './types.js';

// =============================================================================
// Simulated Provider (for demonstration)
// =============================================================================

/**
 * Simulated FHE provider for development and testing.
 * NOT cryptographically secure - use real FHE library for production.
 */
class SimulatedFHEProvider implements FHEProvider {
  readonly name = 'simulated';
  readonly scheme: FHEScheme;
  readonly supportsWASM = false;

  private encoder = new TextEncoder();
  private decoder = new TextDecoder();

  constructor(scheme: FHEScheme = 'BFV') {
    this.scheme = scheme;
  }

  async generateKeys(params: FHEParams): Promise<FHEKeyPair> {
    // Generate random bytes for keys (simulation only)
    const publicKey = crypto.getRandomValues(new Uint8Array(256));
    const secretKey = crypto.getRandomValues(new Uint8Array(256));
    const relinKeys = crypto.getRandomValues(new Uint8Array(512));
    const galoisKeys = crypto.getRandomValues(new Uint8Array(1024));

    return {
      publicKey,
      secretKey,
      relinKeys,
      galoisKeys,
      params,
      createdAt: Date.now(),
    };
  }

  serializePublicKey(keyPair: FHEKeyPair): Uint8Array {
    return keyPair.publicKey;
  }

  deserializePublicKey(data: Uint8Array): Uint8Array {
    return data;
  }

  async encrypt(
    value: number | number[],
    _publicKey: Uint8Array,
    params: FHEParams
  ): Promise<Ciphertext> {
    const values = Array.isArray(value) ? value : [value];

    // Simple encoding: store values as JSON in the "ciphertext"
    // Real FHE would use polynomial encoding
    const encoded = this.encoder.encode(JSON.stringify(values));
    const data = new Uint8Array(encoded.length + 32);
    data.set(crypto.getRandomValues(new Uint8Array(32)), 0); // Random prefix
    data.set(encoded, 32);

    return {
      data,
      scheme: params.scheme,
      noisebudget: 100, // Simulated noise budget
      slots: values.length,
    };
  }

  async decrypt(
    ciphertext: Ciphertext,
    _secretKey: Uint8Array
  ): Promise<number[]> {
    try {
      const encoded = ciphertext.data.slice(32);
      const json = this.decoder.decode(encoded);
      return JSON.parse(json) as number[];
    } catch {
      return [0];
    }
  }

  async add(a: Ciphertext, b: Ciphertext): Promise<Ciphertext> {
    const valuesA = await this.decryptInternal(a);
    const valuesB = await this.decryptInternal(b);

    const result = valuesA.map((v, i) => v + (valuesB[i] ?? 0));
    return this.encryptInternal(result, a.scheme);
  }

  async subtract(a: Ciphertext, b: Ciphertext): Promise<Ciphertext> {
    const valuesA = await this.decryptInternal(a);
    const valuesB = await this.decryptInternal(b);

    const result = valuesA.map((v, i) => v - (valuesB[i] ?? 0));
    return this.encryptInternal(result, a.scheme);
  }

  async multiply(a: Ciphertext, b: Ciphertext): Promise<Ciphertext> {
    const valuesA = await this.decryptInternal(a);
    const valuesB = await this.decryptInternal(b);

    const result = valuesA.map((v, i) => v * (valuesB[i] ?? 1));
    return this.encryptInternal(result, a.scheme, (a.noisebudget ?? 100) - 20);
  }

  async negate(a: Ciphertext): Promise<Ciphertext> {
    const values = await this.decryptInternal(a);
    const result = values.map((v) => -v);
    return this.encryptInternal(result, a.scheme);
  }

  async addPlain(a: Ciphertext, plainValue: number): Promise<Ciphertext> {
    const values = await this.decryptInternal(a);
    const result = values.map((v) => v + plainValue);
    return this.encryptInternal(result, a.scheme);
  }

  async multiplyPlain(a: Ciphertext, plainValue: number): Promise<Ciphertext> {
    const values = await this.decryptInternal(a);
    const result = values.map((v) => v * plainValue);
    return this.encryptInternal(result, a.scheme, (a.noisebudget ?? 100) - 10);
  }

  async sum(ciphertexts: Ciphertext[]): Promise<Ciphertext> {
    if (ciphertexts.length === 0) {
      return this.encryptInternal([0], 'BFV');
    }

    let result = ciphertexts[0];
    for (let i = 1; i < ciphertexts.length; i++) {
      result = await this.add(result, ciphertexts[i]);
    }
    return result;
  }

  async rotate(
    ciphertext: Ciphertext,
    steps: number
  ): Promise<Ciphertext> {
    const values = await this.decryptInternal(ciphertext);
    const n = values.length;
    const normalizedSteps = ((steps % n) + n) % n;

    const rotated = [
      ...values.slice(normalizedSteps),
      ...values.slice(0, normalizedSteps),
    ];
    return this.encryptInternal(rotated, ciphertext.scheme);
  }

  async getNoiseBudget(ciphertext: Ciphertext): Promise<number> {
    return ciphertext.noisebudget ?? 0;
  }

  async relinearize(ciphertext: Ciphertext): Promise<Ciphertext> {
    // In real FHE, this reduces ciphertext size after multiplication
    return { ...ciphertext };
  }

  // Internal helpers
  private async decryptInternal(ciphertext: Ciphertext): Promise<number[]> {
    try {
      const encoded = ciphertext.data.slice(32);
      const json = this.decoder.decode(encoded);
      return JSON.parse(json) as number[];
    } catch {
      return [0];
    }
  }

  private encryptInternal(
    values: number[],
    scheme: FHEScheme,
    noiseBudget: number = 100
  ): Ciphertext {
    const encoded = this.encoder.encode(JSON.stringify(values));
    const data = new Uint8Array(encoded.length + 32);
    data.set(crypto.getRandomValues(new Uint8Array(32)), 0);
    data.set(encoded, 32);

    return {
      data,
      scheme,
      noisebudget: Math.max(0, noiseBudget),
      slots: values.length,
    };
  }
}

// =============================================================================
// FHE Client
// =============================================================================

export interface FHEClientConfig {
  preset?: FHEPreset;
  params?: FHEParams;
  provider?: FHEProvider;
}

export class FHEClient {
  private provider: FHEProvider;
  private params: FHEParams;
  private keyPair: FHEKeyPair | null = null;

  constructor(config: FHEClientConfig = {}) {
    const preset = config.preset ?? 'development';
    const presetParams = FHE_PRESETS[preset];
    this.params = config.params ?? {
      ...presetParams,
      coeffModulusBits: [...presetParams.coeffModulusBits],
    };
    this.provider = config.provider ?? new SimulatedFHEProvider(this.params.scheme);
  }

  /**
   * Initialize the client by generating keys
   */
  async initialize(): Promise<void> {
    this.keyPair = await this.provider.generateKeys(this.params);
  }

  /**
   * Check if client is initialized
   */
  isInitialized(): boolean {
    return this.keyPair !== null;
  }

  /**
   * Get the public key for sharing
   */
  getPublicKey(): Uint8Array {
    if (!this.keyPair) {
      throw new Error('Client not initialized. Call initialize() first.');
    }
    return this.provider.serializePublicKey(this.keyPair);
  }

  /**
   * Encrypt a value or array of values
   */
  async encrypt(value: number | number[]): Promise<Ciphertext> {
    if (!this.keyPair) {
      throw new Error('Client not initialized');
    }
    return this.provider.encrypt(value, this.keyPair.publicKey, this.params);
  }

  /**
   * Decrypt a ciphertext
   */
  async decrypt(ciphertext: Ciphertext): Promise<number[]> {
    if (!this.keyPair) {
      throw new Error('Client not initialized');
    }
    return this.provider.decrypt(ciphertext, this.keyPair.secretKey, this.params);
  }

  /**
   * Add two encrypted values
   */
  async add(a: Ciphertext, b: Ciphertext): Promise<Ciphertext> {
    return this.provider.add(a, b);
  }

  /**
   * Subtract two encrypted values
   */
  async subtract(a: Ciphertext, b: Ciphertext): Promise<Ciphertext> {
    return this.provider.subtract(a, b);
  }

  /**
   * Multiply two encrypted values
   */
  async multiply(a: Ciphertext, b: Ciphertext): Promise<Ciphertext> {
    const result = await this.provider.multiply(a, b, this.keyPair?.relinKeys);
    // Relinearize to reduce ciphertext size
    if (this.keyPair?.relinKeys) {
      return this.provider.relinearize(result, this.keyPair.relinKeys);
    }
    return result;
  }

  /**
   * Negate an encrypted value
   */
  async negate(a: Ciphertext): Promise<Ciphertext> {
    return this.provider.negate(a);
  }

  /**
   * Add a plaintext value to an encrypted value
   */
  async addPlain(encrypted: Ciphertext, plain: number): Promise<Ciphertext> {
    return this.provider.addPlain(encrypted, plain);
  }

  /**
   * Multiply an encrypted value by a plaintext
   */
  async multiplyPlain(encrypted: Ciphertext, plain: number): Promise<Ciphertext> {
    return this.provider.multiplyPlain(encrypted, plain);
  }

  /**
   * Sum multiple encrypted values
   */
  async sum(ciphertexts: Ciphertext[]): Promise<Ciphertext> {
    return this.provider.sum(ciphertexts);
  }

  /**
   * Rotate encrypted vector
   */
  async rotate(ciphertext: Ciphertext, steps: number): Promise<Ciphertext> {
    if (!this.keyPair?.galoisKeys) {
      throw new Error('Galois keys not available');
    }
    return this.provider.rotate(ciphertext, steps, this.keyPair.galoisKeys);
  }

  /**
   * Get remaining noise budget (operations remaining)
   */
  async getNoiseBudget(ciphertext: Ciphertext): Promise<number> {
    if (!this.keyPair) {
      throw new Error('Client not initialized');
    }
    return this.provider.getNoiseBudget(ciphertext, this.keyPair.secretKey);
  }

  /**
   * Create an encrypted vote
   */
  async createEncryptedVote(
    proposalId: string,
    voterId: string,
    choice: number
  ): Promise<EncryptedVote> {
    const ciphertext = await this.encrypt(choice);
    return {
      ciphertext,
      proposalId,
      voterId: await this.hashVoterId(voterId),
      timestamp: Date.now(),
    };
  }

  /**
   * Aggregate encrypted votes
   */
  async aggregateVotes(votes: EncryptedVote[]): Promise<EncryptedAggregation> {
    if (votes.length === 0) {
      throw new Error('No votes to aggregate');
    }

    const sum = await this.sum(votes.map((v) => v.ciphertext));
    const count = await this.encrypt(votes.length);

    return {
      sum,
      count,
      participants: votes.length,
      aggregationId: crypto.randomUUID(),
    };
  }

  /**
   * Compute encrypted average
   */
  async computeEncryptedAverage(
    values: Ciphertext[],
    count: number
  ): Promise<Ciphertext> {
    const sum = await this.sum(values);
    // For BFV, we can't do true division, so multiply by inverse
    // This is approximate and only works for certain parameter choices
    const inverseCount = 1 / count;
    return this.multiplyPlain(sum, inverseCount);
  }

  /**
   * Get encryption parameters
   */
  getParams(): Readonly<FHEParams> {
    return { ...this.params };
  }

  /**
   * Get provider name
   */
  getProviderName(): string {
    return this.provider.name;
  }

  // Private helpers
  private async hashVoterId(voterId: string): Promise<string> {
    const encoder = new TextEncoder();
    const data = encoder.encode(voterId);
    const hash = await crypto.subtle.digest('SHA-256', data);
    return Array.from(new Uint8Array(hash))
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('');
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create an FHE client with default settings
 */
export function createFHEClient(config?: FHEClientConfig): FHEClient {
  return new FHEClient(config);
}

/**
 * Create and initialize an FHE client
 */
export async function initializeFHEClient(
  config?: FHEClientConfig
): Promise<FHEClient> {
  const client = new FHEClient(config);
  await client.initialize();
  return client;
}

/**
 * Create an FHE client for voting operations
 */
export async function createVotingClient(): Promise<FHEClient> {
  const client = new FHEClient({ preset: 'voting' });
  await client.initialize();
  return client;
}

/**
 * Create an FHE client for analytics operations
 */
export async function createAnalyticsClient(): Promise<FHEClient> {
  const client = new FHEClient({ preset: 'analytics' });
  await client.initialize();
  return client;
}
