// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * FHE Types and Interfaces
 *
 * Type definitions for Fully Homomorphic Encryption operations.
 * Supports encrypted computation for privacy-preserving analytics,
 * voting, and aggregation in the Mycelix ecosystem.
 */

// =============================================================================
// Core Types
// =============================================================================

/**
 * FHE scheme types
 * - BFV: Integer arithmetic, good for voting and counting
 * - CKKS: Approximate arithmetic, good for ML and analytics
 * - TFHE: Boolean/small integer, good for comparisons
 */
export type FHEScheme = 'BFV' | 'CKKS' | 'TFHE';

/**
 * Security levels (bits of security)
 */
export type SecurityLevel = 128 | 192 | 256;

/**
 * Encryption parameters
 */
export interface FHEParams {
  scheme: FHEScheme;
  securityLevel: SecurityLevel;
  polyModulusDegree: number;
  coeffModulusBits: number[];
  plainModulus?: number; // BFV only
  scale?: number; // CKKS only
}

/**
 * Key pair for FHE operations
 */
export interface FHEKeyPair {
  publicKey: Uint8Array;
  secretKey: Uint8Array;
  relinKeys?: Uint8Array; // For multiplication
  galoisKeys?: Uint8Array; // For rotation
  params: FHEParams;
  createdAt: number;
}

/**
 * Encrypted ciphertext
 */
export interface Ciphertext {
  data: Uint8Array;
  scheme: FHEScheme;
  noisebudget?: number;
  scale?: number; // CKKS
  slots?: number;
}

/**
 * Plaintext representation
 */
export interface Plaintext {
  data: Uint8Array;
  scheme: FHEScheme;
  scale?: number;
}

// =============================================================================
// Operation Types
// =============================================================================

/**
 * Supported homomorphic operations
 */
export type HomomorphicOp =
  | 'add'
  | 'subtract'
  | 'multiply'
  | 'negate'
  | 'rotate'
  | 'sum'
  | 'compare';

/**
 * Result of an encrypted operation
 */
export interface EncryptedResult<T = unknown> {
  ciphertext: Ciphertext;
  operation: HomomorphicOp;
  operands: number;
  estimatedNoiseBudget?: number;
  metadata?: T;
}

/**
 * Batch encryption for SIMD-style operations
 */
export interface EncryptedBatch {
  ciphertexts: Ciphertext[];
  batchSize: number;
  slots: number;
}

// =============================================================================
// Application-Specific Types
// =============================================================================

/**
 * Encrypted vote for private voting
 */
export interface EncryptedVote {
  ciphertext: Ciphertext;
  proposalId: string;
  voterId: string; // Anonymized/hashed
  timestamp: number;
  proof?: Uint8Array; // ZK proof of valid vote
}

/**
 * Encrypted aggregation result
 */
export interface EncryptedAggregation {
  sum: Ciphertext;
  count: Ciphertext;
  min?: Ciphertext;
  max?: Ciphertext;
  participants: number;
  aggregationId: string;
}

/**
 * Encrypted analytics query
 */
export interface EncryptedQuery {
  queryId: string;
  operation: 'sum' | 'count' | 'average' | 'histogram';
  encryptedData: Ciphertext[];
  publicParams: Uint8Array;
}

/**
 * Private set intersection result
 */
export interface PSIResult {
  intersectionSize: number;
  encryptedIntersection?: Ciphertext;
  proof?: Uint8Array;
}

// =============================================================================
// Provider Interface
// =============================================================================

/**
 * FHE provider interface
 * Implementations can use TFHE-rs, SEAL, OpenFHE, or custom WASM
 */
export interface FHEProvider {
  readonly name: string;
  readonly scheme: FHEScheme;
  readonly supportsWASM: boolean;

  // Key management
  generateKeys(params: FHEParams): Promise<FHEKeyPair>;
  serializePublicKey(keyPair: FHEKeyPair): Uint8Array;
  deserializePublicKey(data: Uint8Array, params: FHEParams): Uint8Array;

  // Encryption/Decryption
  encrypt(value: number | number[], publicKey: Uint8Array, params: FHEParams): Promise<Ciphertext>;
  decrypt(ciphertext: Ciphertext, secretKey: Uint8Array, params: FHEParams): Promise<number[]>;

  // Homomorphic operations
  add(a: Ciphertext, b: Ciphertext): Promise<Ciphertext>;
  subtract(a: Ciphertext, b: Ciphertext): Promise<Ciphertext>;
  multiply(a: Ciphertext, b: Ciphertext, relinKeys?: Uint8Array): Promise<Ciphertext>;
  negate(a: Ciphertext): Promise<Ciphertext>;
  addPlain(a: Ciphertext, plainValue: number): Promise<Ciphertext>;
  multiplyPlain(a: Ciphertext, plainValue: number): Promise<Ciphertext>;

  // Advanced operations
  sum(ciphertexts: Ciphertext[]): Promise<Ciphertext>;
  rotate(ciphertext: Ciphertext, steps: number, galoisKeys: Uint8Array): Promise<Ciphertext>;

  // Utility
  getNoiseBudget(ciphertext: Ciphertext, secretKey: Uint8Array): Promise<number>;
  relinearize(ciphertext: Ciphertext, relinKeys: Uint8Array): Promise<Ciphertext>;
}

// =============================================================================
// Configuration
// =============================================================================

/**
 * Preset parameter configurations
 */
export const FHE_PRESETS = {
  // Fast, lower security - good for development
  development: {
    scheme: 'BFV' as FHEScheme,
    securityLevel: 128 as SecurityLevel,
    polyModulusDegree: 4096,
    coeffModulusBits: [40, 40],
    plainModulus: 65537,
  },

  // Balanced for voting applications
  voting: {
    scheme: 'BFV' as FHEScheme,
    securityLevel: 128 as SecurityLevel,
    polyModulusDegree: 8192,
    coeffModulusBits: [60, 40, 40, 60],
    plainModulus: 65537,
  },

  // High precision for analytics
  analytics: {
    scheme: 'CKKS' as FHEScheme,
    securityLevel: 128 as SecurityLevel,
    polyModulusDegree: 16384,
    coeffModulusBits: [60, 40, 40, 40, 40, 40, 60],
    scale: Math.pow(2, 40),
  },

  // Maximum security
  highSecurity: {
    scheme: 'BFV' as FHEScheme,
    securityLevel: 256 as SecurityLevel,
    polyModulusDegree: 32768,
    coeffModulusBits: [60, 60, 60, 60, 60, 60, 60, 60],
    plainModulus: 65537,
  },
} as const;

export type FHEPreset = keyof typeof FHE_PRESETS;
