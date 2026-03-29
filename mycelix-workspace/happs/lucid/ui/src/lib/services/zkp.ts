// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Zero-Knowledge Proof Service
 *
 * Client-side integration for LUCID's ZK proof system.
 * Enables anonymous belief attestation and privacy-preserving reputation proofs.
 */

import { invoke } from '@tauri-apps/api/core';

// ============================================================================
// TYPES
// ============================================================================

export interface AnonymousBeliefProofInput {
  /** Hash of the belief content (hex-encoded) */
  belief_hash: string;
  /** The agent's secret key (hex-encoded) */
  agent_secret: string;
}

export interface AnonymousBeliefProofOutput {
  /** The serialized proof (JSON) */
  proof_json: string;
  /** The author commitment (hex-encoded) */
  author_commitment: string;
  /** Belief hash (hex-encoded) */
  belief_hash: string;
  /** Whether this is a simulated proof (NOT cryptographically secure) */
  isSimulated: boolean;
  /** Source of proof generation */
  generatedWith: 'tauri' | 'simulation';
}

export interface ReputationRangeProofInput {
  /** The actual reputation value */
  actual_reputation: number;
  /** The minimum threshold to prove */
  min_threshold: number;
}

export interface ReputationRangeProofOutput {
  /** The serialized proof (JSON) */
  proof_json: string;
  /** The reputation commitment (hex-encoded) */
  reputation_commitment: string;
  /** The threshold that was proven */
  min_threshold: number;
  /** Whether this is a simulated proof (NOT cryptographically secure) */
  isSimulated: boolean;
  /** Source of proof generation */
  generatedWith: 'tauri' | 'simulation';
}

export interface VerifyProofInput {
  /** The proof type ("anonymous_belief" or "reputation_range") */
  proof_type: 'anonymous_belief' | 'reputation_range';
  /** The serialized proof (JSON) */
  proof_json: string;
}

export interface VerifyProofOutput {
  /** Whether the proof is valid */
  valid: boolean;
  /** Error message if invalid */
  error: string | null;
}

export interface CommitmentOutput {
  /** The commitment (hex-encoded) */
  commitment: string;
  /** The nonce used (hex-encoded) */
  nonce: string;
}

// ============================================================================
// ZKP SERVICE
// ============================================================================

let zkpAvailable: boolean | null = null;

/**
 * Check if we're in production mode
 * In production, simulated proofs are NOT allowed
 */
export function isProductionMode(): boolean {
  // Check various environment indicators
  if (typeof import.meta !== 'undefined' && import.meta.env) {
    return import.meta.env.PROD === true || import.meta.env.MODE === 'production';
  }
  return false;
}

/**
 * Check if simulated proofs are allowed
 * Only allowed in development/test mode
 */
export function allowSimulatedProofs(): boolean {
  return !isProductionMode();
}

/**
 * Reset ZKP availability cache (for testing)
 */
export function resetZkpCache(): void {
  zkpAvailable = null;
}

/**
 * Check if ZKP functionality is available (Tauri backend)
 */
export async function isZkpAvailable(): Promise<boolean> {
  if (zkpAvailable !== null) return zkpAvailable;

  try {
    zkpAvailable = await invoke<boolean>('zkp_ready');
    return zkpAvailable;
  } catch {
    zkpAvailable = false;
    return false;
  }
}

/**
 * Generate an anonymous belief proof
 *
 * This proves that you authored a belief without revealing your identity.
 * The commitment can be used to later prove authorship if needed.
 *
 * @param beliefContent - The belief content to prove authorship of
 * @param agentSecret - Your secret key (keep this private!)
 * @returns The proof and commitment
 */
export async function generateAnonymousBeliefProof(
  beliefContent: string,
  agentSecret: string
): Promise<AnonymousBeliefProofOutput | null> {
  if (!(await isZkpAvailable())) {
    // In production, reject simulated proofs
    if (!allowSimulatedProofs()) {
      console.error('ZKP required in production mode but not available');
      throw new Error('ZKP_REQUIRED_IN_PRODUCTION: Cannot generate proof without Tauri ZKP backend');
    }
    console.warn('ZKP not available, falling back to simulated proof (DEVELOPMENT ONLY)');
    return simulateAnonymousBeliefProof(beliefContent, agentSecret);
  }

  try {
    // First hash the belief content
    const beliefHash = await invoke<string>('hash_belief_content', { content: beliefContent });

    // Convert agent secret to hex
    const agentSecretHex = stringToHex(agentSecret);

    // Generate the proof via Tauri
    const proof = await invoke<{
      proof_json: string;
      author_commitment: string;
      belief_hash: string;
    }>('generate_anonymous_belief_proof', {
      input: {
        belief_hash: beliefHash,
        agent_secret: agentSecretHex,
      },
    });

    // Add metadata to indicate this is a real proof
    return {
      ...proof,
      isSimulated: false,
      generatedWith: 'tauri',
    };
  } catch (error) {
    console.error('Failed to generate anonymous belief proof:', error);
    return null;
  }
}

/**
 * Generate a reputation range proof
 *
 * This proves that your reputation exceeds a threshold without revealing
 * your exact reputation value.
 *
 * @param actualReputation - Your actual reputation value
 * @param minThreshold - The minimum threshold to prove
 * @returns The proof and commitment
 */
export async function generateReputationRangeProof(
  actualReputation: number,
  minThreshold: number
): Promise<ReputationRangeProofOutput | null> {
  if (!(await isZkpAvailable())) {
    // In production, reject simulated proofs
    if (!allowSimulatedProofs()) {
      console.error('ZKP required in production mode but not available');
      throw new Error('ZKP_REQUIRED_IN_PRODUCTION: Cannot generate proof without Tauri ZKP backend');
    }
    console.warn('ZKP not available, falling back to simulated proof (DEVELOPMENT ONLY)');
    return simulateReputationRangeProof(actualReputation, minThreshold);
  }

  try {
    const proof = await invoke<{
      proof_json: string;
      reputation_commitment: string;
      min_threshold: number;
    }>('generate_reputation_range_proof', {
      input: {
        actual_reputation: actualReputation,
        min_threshold: minThreshold,
      },
    });

    // Add metadata to indicate this is a real proof
    return {
      ...proof,
      isSimulated: false,
      generatedWith: 'tauri',
    };
  } catch (error) {
    console.error('Failed to generate reputation range proof:', error);
    return null;
  }
}

/**
 * Verify a proof
 *
 * @param proofType - The type of proof to verify
 * @param proofJson - The serialized proof
 * @returns Verification result
 */
export async function verifyProof(
  proofType: 'anonymous_belief' | 'reputation_range',
  proofJson: string
): Promise<VerifyProofOutput> {
  // Check if this is a simulated proof
  try {
    const parsed = JSON.parse(proofJson);
    if (parsed.proof_type?.includes('_simulated')) {
      if (!allowSimulatedProofs()) {
        return { valid: false, error: 'Simulated proofs not allowed in production mode' };
      }
      // In development, accept simulated proofs with a warning
      console.warn('Verifying simulated proof - NOT cryptographically secure');
      return { valid: true, error: null };
    }
  } catch {
    // Not a valid JSON, let verification fail naturally
  }

  if (!(await isZkpAvailable())) {
    if (!allowSimulatedProofs()) {
      return { valid: false, error: 'ZKP verification not available in production mode' };
    }
    // In development, accept unknown proofs
    console.warn('ZKP not available, skipping verification (DEVELOPMENT ONLY)');
    return { valid: true, error: null };
  }

  try {
    const result = await invoke<VerifyProofOutput>('verify_proof', {
      input: {
        proof_type: proofType,
        proof_json: proofJson,
      },
    });

    return result;
  } catch (error) {
    return {
      valid: false,
      error: error instanceof Error ? error.message : 'Verification failed',
    };
  }
}

/**
 * Create a commitment to a value
 *
 * This is useful for creating commitments that can be revealed later.
 *
 * @param value - The value to commit to
 * @returns The commitment and nonce
 */
export async function createCommitment(value: string): Promise<CommitmentOutput | null> {
  if (!(await isZkpAvailable())) {
    if (!allowSimulatedProofs()) {
      console.error('ZKP required in production mode but not available');
      throw new Error('ZKP_REQUIRED_IN_PRODUCTION: Cannot create commitment without Tauri ZKP backend');
    }
    console.warn('ZKP not available, using simulated commitment (DEVELOPMENT ONLY)');
    return simulateCommitment(value);
  }

  try {
    return await invoke<CommitmentOutput>('create_value_commitment', { value });
  } catch (error) {
    console.error('Failed to create commitment:', error);
    return null;
  }
}

/**
 * Hash belief content
 *
 * Creates a deterministic hash of belief content for use in proofs.
 *
 * @param content - The belief content
 * @returns Hex-encoded hash
 */
export async function hashBeliefContent(content: string): Promise<string> {
  if (!(await isZkpAvailable())) {
    return simulateHash(content);
  }

  try {
    return await invoke<string>('hash_belief_content', { content });
  } catch (error) {
    console.error('Failed to hash belief content:', error);
    return simulateHash(content);
  }
}

// ============================================================================
// SIMULATION FALLBACKS (for when Tauri is not available)
// WARNING: These are NOT cryptographically secure! Development only!
// ============================================================================

/**
 * Generate a cryptographically random hex string
 * Better than Math.random() but still only for development use
 */
function generateSecureNonce(length: number = 16): string {
  const bytes = new Uint8Array(length);
  crypto.getRandomValues(bytes);
  return Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
}

function simulateAnonymousBeliefProof(
  beliefContent: string,
  agentSecret: string
): AnonymousBeliefProofOutput {
  const beliefHash = simulateHash(beliefContent);
  const nonce = generateSecureNonce(16);
  const commitment = simulateHash(agentSecret + nonce);

  return {
    proof_json: JSON.stringify({
      proof_type: 'anonymous_belief_simulated',
      belief_hash: beliefHash,
      commitment,
      warning: 'SIMULATED_PROOF_NOT_SECURE',
    }),
    author_commitment: commitment,
    belief_hash: beliefHash,
    isSimulated: true,
    generatedWith: 'simulation',
  };
}

function simulateReputationRangeProof(
  actualReputation: number,
  minThreshold: number
): ReputationRangeProofOutput | null {
  if (actualReputation < minThreshold) {
    return null; // Cannot prove false statement
  }

  const nonce = generateSecureNonce(16);
  const commitment = simulateHash(actualReputation.toString() + nonce);

  return {
    proof_json: JSON.stringify({
      proof_type: 'reputation_range_simulated',
      min_threshold: minThreshold,
      commitment,
      warning: 'SIMULATED_PROOF_NOT_SECURE',
    }),
    reputation_commitment: commitment,
    min_threshold: minThreshold,
    isSimulated: true,
    generatedWith: 'simulation',
  };
}

function simulateCommitment(value: string): CommitmentOutput {
  const nonce = generateSecureNonce(16);
  const commitment = simulateHash(value + nonce);

  return {
    commitment,
    nonce,
  };
}

function simulateHash(input: string): string {
  // Simple hash simulation (NOT cryptographically secure!)
  // This is only for development/testing. Real proofs use Blake3 via Tauri.
  let hash = 0;
  for (let i = 0; i < input.length; i++) {
    const char = input.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(16).padStart(16, '0');
}

function stringToHex(str: string): string {
  let hex = '';
  for (let i = 0; i < str.length; i++) {
    hex += str.charCodeAt(i).toString(16).padStart(2, '0');
  }
  return hex;
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/**
 * Prove anonymous authorship and store the proof for a belief
 *
 * This is a convenience function that:
 * 1. Generates the proof
 * 2. Returns data suitable for storing/sharing
 *
 * @param beliefContent - The belief content
 * @param agentSecret - Your secret key
 * @returns Proof data for storage
 */
export async function proveBeliefAuthorship(
  beliefContent: string,
  agentSecret: string
): Promise<{
  beliefHash: string;
  commitment: string;
  proof: string;
} | null> {
  const result = await generateAnonymousBeliefProof(beliefContent, agentSecret);
  if (!result) return null;

  return {
    beliefHash: result.belief_hash,
    commitment: result.author_commitment,
    proof: result.proof_json,
  };
}

/**
 * Prove reputation meets threshold for participating in collective
 *
 * @param reputation - Your reputation score (0-100 typical)
 * @param requiredThreshold - Minimum required (e.g., 50)
 * @returns Proof data if eligible
 */
export async function proveEligibility(
  reputation: number,
  requiredThreshold: number
): Promise<{
  threshold: number;
  commitment: string;
  proof: string;
} | null> {
  const result = await generateReputationRangeProof(reputation, requiredThreshold);
  if (!result) return null;

  return {
    threshold: result.min_threshold,
    commitment: result.reputation_commitment,
    proof: result.proof_json,
  };
}
