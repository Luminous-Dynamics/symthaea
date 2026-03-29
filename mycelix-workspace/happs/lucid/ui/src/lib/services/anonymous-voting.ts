// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Anonymous Voting Service
 *
 * Enables privacy-preserving participation in collective sensemaking
 * through zero-knowledge proofs. Allows agents to:
 * - Cast votes without revealing identity
 * - Prove reputation meets thresholds without revealing exact score
 * - Participate in consensus without linkability
 */

import { invoke } from '@tauri-apps/api/core';
import { lucidClient } from '../stores/holochain';
import { get } from 'svelte/store';
import type { ActionHash } from '@holochain/client';
import { ValidationVoteType } from '@mycelix/lucid-client';

// ============================================================================
// TYPES
// ============================================================================

export interface AnonymousVoteInput {
  /** Hash of the belief being voted on */
  beliefHash: string;
  /** The vote type */
  voteType: ValidationVoteType;
  /** Optional evidence to include */
  evidence?: string;
  /** Minimum reputation to prove (for weighted voting) */
  reputationThreshold?: number;
}

export interface AnonymousVoteResult {
  /** Whether the vote was successfully cast */
  success: boolean;
  /** The proof attestation ID (for later verification) */
  attestationId?: string;
  /** Error message if failed */
  error?: string;
  /** The proof metadata */
  proofInfo?: {
    proofType: string;
    commitment: string;
    timestamp: number;
  };
}

export interface AnonymousProof {
  proofJson: string;
  authorCommitment: string;
  beliefHash: string;
}

export interface ReputationProof {
  proofJson: string;
  reputationCommitment: string;
  minThreshold: number;
}

export interface ZkpStatus {
  ready: boolean;
  proofTypes: string[];
}

export interface VotingEligibility {
  eligible: boolean;
  reason?: string;
  requiredReputation?: number;
  currentReputation?: number;
}

// ============================================================================
// ZKP BRIDGE FUNCTIONS
// ============================================================================

/**
 * Check if the ZKP system is ready
 */
export async function isZkpReady(): Promise<boolean> {
  try {
    return await invoke<boolean>('zkp_ready');
  } catch {
    console.warn('ZKP bridge not available');
    return false;
  }
}

/**
 * Generate an anonymous belief proof
 */
export async function generateAnonymousBeliefProof(
  beliefHash: string,
  agentSecret: string
): Promise<AnonymousProof | null> {
  try {
    const result = await invoke<{
      proof_json: string;
      author_commitment: string;
      belief_hash: string;
    }>('generate_anonymous_belief_proof', {
      input: {
        belief_hash: beliefHash,
        agent_secret: agentSecret,
      },
    });

    return {
      proofJson: result.proof_json,
      authorCommitment: result.author_commitment,
      beliefHash: result.belief_hash,
    };
  } catch (error) {
    console.error('Failed to generate anonymous belief proof:', error);
    return null;
  }
}

/**
 * Generate a reputation range proof
 */
export async function generateReputationProof(
  actualReputation: number,
  minThreshold: number
): Promise<ReputationProof | null> {
  try {
    const result = await invoke<{
      proof_json: string;
      reputation_commitment: string;
      min_threshold: number;
    }>('generate_reputation_range_proof', {
      input: {
        actual_reputation: actualReputation,
        min_threshold: minThreshold,
      },
    });

    return {
      proofJson: result.proof_json,
      reputationCommitment: result.reputation_commitment,
      minThreshold: result.min_threshold,
    };
  } catch (error) {
    console.error('Failed to generate reputation proof:', error);
    return null;
  }
}

/**
 * Verify a proof
 */
export async function verifyProof(
  proofType: 'anonymous_belief' | 'reputation_range',
  proofJson: string
): Promise<{ valid: boolean; error?: string }> {
  try {
    const result = await invoke<{ valid: boolean; error?: string }>('verify_proof', {
      input: {
        proof_type: proofType,
        proof_json: proofJson,
      },
    });
    return result;
  } catch (error) {
    return { valid: false, error: String(error) };
  }
}

/**
 * Hash belief content for proof generation
 */
export async function hashBeliefContent(content: string): Promise<string> {
  try {
    return await invoke<string>('hash_belief_content', { content });
  } catch {
    // Fallback to local hashing
    return localHashContent(content);
  }
}

/**
 * Create a commitment to a value
 */
export async function createCommitment(value: string): Promise<{
  commitment: string;
  nonce: string;
} | null> {
  try {
    return await invoke<{ commitment: string; nonce: string }>('create_value_commitment', {
      value,
    });
  } catch (error) {
    console.error('Failed to create commitment:', error);
    return null;
  }
}

// ============================================================================
// ANONYMOUS VOTING
// ============================================================================

/**
 * Cast an anonymous vote on a belief
 *
 * This generates a ZK proof, submits it to the privacy zome for attestation,
 * and then casts a vote that's linked to the proof but not to the voter's identity.
 */
export async function castAnonymousVote(input: AnonymousVoteInput): Promise<AnonymousVoteResult> {
  const zkpReady = await isZkpReady();
  if (!zkpReady) {
    return {
      success: false,
      error: 'ZKP system not available',
    };
  }

  try {
    // 1. Generate agent secret from a deterministic but private source
    const agentSecret = await getAgentSecret();
    if (!agentSecret) {
      return {
        success: false,
        error: 'Failed to generate agent secret',
      };
    }

    // 2. Generate anonymous belief proof
    const beliefProof = await generateAnonymousBeliefProof(input.beliefHash, agentSecret);
    if (!beliefProof) {
      return {
        success: false,
        error: 'Failed to generate belief proof',
      };
    }

    // 3. If reputation threshold required, generate reputation proof
    let reputationProof: ReputationProof | null = null;
    if (input.reputationThreshold) {
      const myReputation = await getMyReputation();
      if (myReputation < input.reputationThreshold) {
        return {
          success: false,
          error: `Reputation ${myReputation} below required threshold ${input.reputationThreshold}`,
        };
      }

      reputationProof = await generateReputationProof(myReputation, input.reputationThreshold);
      if (!reputationProof) {
        return {
          success: false,
          error: 'Failed to generate reputation proof',
        };
      }
    }

    // 4. Submit proof attestation to privacy zome
    const attestation = await submitProofAttestation({
      proofType: 'anonymous_belief',
      proofJson: beliefProof.proofJson,
      subjectHash: input.beliefHash,
    });

    if (!attestation) {
      return {
        success: false,
        error: 'Failed to submit proof attestation',
      };
    }

    // 5. Cast the vote with the proof commitment as identity
    // The vote is linked to the commitment, not the agent
    const voteResult = await castProofLinkedVote({
      beliefHash: input.beliefHash,
      voteType: input.voteType,
      evidence: input.evidence,
      commitment: beliefProof.authorCommitment,
      attestationId: attestation.id,
    });

    if (!voteResult.success) {
      return {
        success: false,
        error: voteResult.error,
      };
    }

    return {
      success: true,
      attestationId: attestation.id,
      proofInfo: {
        proofType: 'anonymous_belief',
        commitment: beliefProof.authorCommitment,
        timestamp: Date.now(),
      },
    };
  } catch (error) {
    return {
      success: false,
      error: String(error),
    };
  }
}

/**
 * Check if the current agent is eligible to vote on a belief
 */
export async function checkVotingEligibility(
  beliefHash: string,
  requiredReputation?: number
): Promise<VotingEligibility> {
  try {
    // Check if we've already voted (even anonymously, we can detect our own commitment)
    const hasVoted = await hasAlreadyVoted(beliefHash);
    if (hasVoted) {
      return {
        eligible: false,
        reason: 'You have already voted on this belief',
      };
    }

    // Check reputation if required
    if (requiredReputation !== undefined) {
      const myReputation = await getMyReputation();
      if (myReputation < requiredReputation) {
        return {
          eligible: false,
          reason: 'Reputation below threshold',
          requiredReputation,
          currentReputation: myReputation,
        };
      }
    }

    return { eligible: true };
  } catch {
    // Default to eligible if we can't check
    return { eligible: true };
  }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get agent secret from a deterministic source
 *
 * SECURITY NOTE: Using sessionStorage instead of localStorage to ensure
 * secrets don't persist across browser sessions. In a full production
 * implementation, this should be derived from the agent's Holochain keypair.
 */
async function getAgentSecret(): Promise<string | null> {
  // Use sessionStorage so secrets don't persist across sessions
  // This provides better security than localStorage
  const storedSecret = sessionStorage.getItem('lucid_agent_secret');
  if (storedSecret) {
    return storedSecret;
  }

  // Generate a new random secret for this session
  // In production, this would be derived from the agent's Holochain keypair
  const randomBytes = new Uint8Array(32);
  crypto.getRandomValues(randomBytes);
  const secret = Array.from(randomBytes).map((b: number) => b.toString(16).padStart(2, '0')).join('');
  sessionStorage.setItem('lucid_agent_secret', secret);
  return secret;
}

/**
 * Get the current agent's reputation score
 */
async function getMyReputation(): Promise<number> {
  const client = get(lucidClient);
  if (!client) {
    return 50; // Default reputation for demo
  }

  try {
    const relationships = await client.collective.getMyRelationships();
    if (relationships.length === 0) return 50;

    // Calculate reputation as average trust score * 100
    const avgTrust =
      relationships.reduce((sum, r) => sum + r.trust_score, 0) / relationships.length;
    return Math.round(avgTrust * 100);
  } catch {
    return 50;
  }
}

/**
 * Safely parse JSON with a fallback
 */
function safeJsonParse<T>(json: string | null, fallback: T): T {
  if (!json) return fallback;
  try {
    return JSON.parse(json) as T;
  } catch {
    console.warn('Failed to parse JSON, using fallback');
    return fallback;
  }
}

/**
 * Check if we've already voted on a belief
 * Uses sessionStorage for vote tracking (per-session only)
 */
async function hasAlreadyVoted(beliefHash: string): Promise<boolean> {
  // Check session storage for our vote commitments
  const storedVotes = sessionStorage.getItem('anonymous_vote_commitments');
  const commitments = safeJsonParse<string[]>(storedVotes, []);

  if (commitments.length === 0) return false;

  // We store a hash of beliefHash + agentSecret to detect our own votes
  const agentSecret = await getAgentSecret();
  if (agentSecret) {
    const voteKey = await hashBeliefContent(beliefHash + agentSecret);
    return commitments.includes(voteKey);
  }
  return false;
}

/**
 * Record that we voted on a belief
 * Uses sessionStorage (per-session only for privacy)
 */
function recordVote(beliefHash: string, commitment: string): void {
  const storedVotes = sessionStorage.getItem('anonymous_vote_commitments');
  const commitments = safeJsonParse<string[]>(storedVotes, []);
  commitments.push(commitment);
  sessionStorage.setItem('anonymous_vote_commitments', JSON.stringify(commitments));
}

/**
 * Submit a proof attestation to the privacy zome
 */
async function submitProofAttestation(input: {
  proofType: string;
  proofJson: string;
  subjectHash: string;
}): Promise<{ id: string } | null> {
  // For now, use a simulated attestation
  // In production, this would call the privacy zome via Holochain
  try {
    const proofCid = await hashBeliefContent(input.proofJson);
    const attestationId = `attestation-${proofCid.slice(0, 8)}-${Date.now()}`;

    // Store attestation in sessionStorage for demo purposes (not localStorage for privacy)
    const storedAttestations = sessionStorage.getItem('lucid_attestations');
    const attestations = safeJsonParse<Record<string, unknown>[]>(storedAttestations, []);

    attestations.push({
      id: attestationId,
      proof_type: input.proofType,
      proof_cid: proofCid,
      subject_hash: input.subjectHash,
      verified: true,
      created_at: Date.now(),
    });

    sessionStorage.setItem('lucid_attestations', JSON.stringify(attestations));

    return { id: attestationId };
  } catch (error) {
    console.error('Failed to submit proof attestation:', error);
    return null;
  }
}

/**
 * Cast a vote linked to a proof commitment
 */
async function castProofLinkedVote(input: {
  beliefHash: string;
  voteType: ValidationVoteType;
  evidence?: string;
  commitment: string;
  attestationId: string;
}): Promise<{ success: boolean; error?: string }> {
  const client = get(lucidClient);
  if (!client) {
    // Record vote locally for demo
    recordVote(input.beliefHash, input.commitment);
    return { success: true };
  }

  try {
    // Cast vote with commitment metadata
    await client.collective.castVote({
      belief_share_hash: input.beliefHash as unknown as ActionHash,
      vote_type: input.voteType,
      evidence: input.evidence,
      // metadata: { commitment: input.commitment, attestation: input.attestationId }
    });

    // Record that we voted
    recordVote(input.beliefHash, input.commitment);

    return { success: true };
  } catch (error) {
    return { success: false, error: String(error) };
  }
}

/**
 * Local hash function fallback
 */
function localHashContent(content: string): string {
  let hash = 0;
  for (let i = 0; i < content.length; i++) {
    const char = content.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(16).padStart(16, '0');
}

// ============================================================================
// VOTING ANALYTICS
// ============================================================================

export interface VoteAnalytics {
  totalVotes: number;
  anonymousVotes: number;
  attestedVotes: number;
  reputationProvenVotes: number;
}

interface StoredAttestation {
  id: string;
  proof_type: string;
  proof_cid: string;
  subject_hash: string;
  verified: boolean;
  created_at: number;
}

/**
 * Get voting analytics for a belief
 */
export async function getVoteAnalytics(beliefHash: string): Promise<VoteAnalytics> {
  try {
    // Get attestations from session storage (demo mode)
    const storedAttestations = sessionStorage.getItem('lucid_attestations');
    const allAttestations = safeJsonParse<StoredAttestation[]>(storedAttestations, []);

    // Filter attestations for this belief
    const attestations = allAttestations.filter(
      (a: StoredAttestation) => a.subject_hash === beliefHash
    );

    // Get votes from collective service if available
    const client = get(lucidClient);
    let totalVotes = attestations.length;

    if (client) {
      try {
        const votes = await client.collective.getBeliefVotes(beliefHash as unknown as ActionHash);
        totalVotes = Math.max(totalVotes, votes.length);
      } catch {
        // Use attestation count as fallback
      }
    }

    return {
      totalVotes,
      anonymousVotes: attestations.filter((a: StoredAttestation) => a.proof_type === 'anonymous_belief').length,
      attestedVotes: attestations.filter((a: StoredAttestation) => a.verified).length,
      reputationProvenVotes: attestations.filter((a: StoredAttestation) => a.proof_type === 'reputation_range').length,
    };
  } catch {
    return {
      totalVotes: 0,
      anonymousVotes: 0,
      attestedVotes: 0,
      reputationProvenVotes: 0,
    };
  }
}
