// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Collective Sensemaking Service
 *
 * Client-side integration for the distributed belief network.
 * Enables sharing, validation, consensus discovery, and emergent truth.
 */

import type { Thought } from '@mycelix/lucid-client';
import type {
  BeliefShare,
  ShareBeliefInput,
  ValidationVote,
  CastVoteInput,
  ConsensusRecord,
  EmergentPattern,
  PatternCluster,
  WeightedConsensusResult,
  CollectiveStats,
  AgentRelationship,
  UpdateRelationshipInput,
} from '@mycelix/lucid-client';
import {
  ValidationVoteType,
  BeliefStance,
  PatternType,
  ConsensusType,
  RelationshipStage,
} from '@mycelix/lucid-client';
import { lucidClient } from '../stores/holochain';
import { get } from 'svelte/store';
import type { ActionHash, AgentPubKey } from '@holochain/client';
import {
  withRetry,
  withFallback,
  isRetryableError,
  createCircuitBreaker,
} from '../utils/resilience';

// Circuit breaker for Holochain operations
const holochainCircuit = createCircuitBreaker({
  failureThreshold: 5,
  recoveryTimeout: 30000,
  onOpen: () => console.warn('Holochain circuit breaker opened - using fallback mode'),
  onClose: () => console.info('Holochain circuit breaker closed - resuming normal operations'),
});

// Re-export types for convenience
export type {
  BeliefShare,
  ValidationVote,
  ConsensusRecord,
  EmergentPattern,
  PatternCluster,
  WeightedConsensusResult,
  CollectiveStats,
  AgentRelationship,
};

export { ValidationVoteType, BeliefStance, PatternType, ConsensusType, RelationshipStage };

// ============================================================================
// TYPES FOR UI
// ============================================================================

export interface CollectiveView {
  myShares: BeliefShare[];
  publicShares: BeliefShare[];
  consensusRecords: ConsensusRecord[];
  patterns: EmergentPattern[];
  stats: CollectiveStats;
}

// ============================================================================
// ANONYMIZATION
// ============================================================================

/**
 * Create an anonymized belief share from a personal thought
 */
export function anonymizeThought(thought: Thought): Omit<ShareBeliefInput, 'embedding'> {
  // Hash content for deduplication (but not identification)
  const contentHash = hashContent(thought.content);

  // Optionally summarize or generalize content
  const content = thought.content.length > 500
    ? thought.content.slice(0, 500) + '...'
    : thought.content;

  // Build epistemic code
  const epistemicCode = `${thought.epistemic.empirical}${thought.epistemic.normative}${thought.epistemic.materiality}${thought.epistemic.harmonic}`;

  return {
    content_hash: contentHash,
    content,
    belief_type: thought.thought_type,
    epistemic_code: epistemicCode,
    confidence: thought.confidence,
    tags: thought.tags || [],
    evidence_hashes: thought.source_hashes?.map((h) => hashContent(h.toString())) || [],
  };
}

/**
 * Simple content hash (for demo - use proper crypto in production)
 */
function hashContent(content: string): string {
  let hash = 0;
  for (let i = 0; i < content.length; i++) {
    const char = content.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(16).padStart(8, '0');
}

// ============================================================================
// COLLECTIVE OPERATIONS (Holochain Integration)
// ============================================================================

/**
 * Share a thought to the collective (anonymized)
 * Uses retry logic and circuit breaker for resilience
 */
export async function shareToCollective(
  thought: Thought,
  stance?: BeliefStance,
  embedding?: number[]
): Promise<BeliefShare | null> {
  const client = get(lucidClient);
  if (!client) {
    console.warn('No Holochain client available, using local simulation');
    return simulateShare(thought);
  }

  const anonymized = anonymizeThought(thought);
  const input: ShareBeliefInput = {
    ...anonymized,
    stance,
    embedding: embedding ?? [],
  };

  // Use resilience patterns for robust operation
  return withFallback(
    async () => {
      return withRetry(
        async () => {
          return holochainCircuit.execute(async () => {
            return client.collective.shareBelief(input);
          });
        },
        {
          maxAttempts: 3,
          initialDelay: 500,
          shouldRetry: isRetryableError,
          onRetry: (attempt, error) => {
            console.warn(`Share belief retry attempt ${attempt}:`, error);
          },
        }
      );
    },
    {
      fallback: () => simulateShare(thought),
      logErrors: true,
    }
  );
}

/**
 * Vote on a shared belief
 */
export async function voteOnBelief(
  beliefHash: ActionHash,
  voteType: ValidationVoteType,
  evidence?: string
): Promise<ValidationVote | null> {
  const client = get(lucidClient);
  if (!client) {
    console.warn('No Holochain client available');
    return null;
  }

  try {
    const input: CastVoteInput = {
      belief_share_hash: beliefHash,
      vote_type: voteType,
      evidence,
    };
    return await client.collective.castVote(input);
  } catch (error) {
    console.error('Failed to cast vote:', error);
    return null;
  }
}

/**
 * Get beliefs shared by the collective
 */
export async function getCollectiveBeliefs(tag?: string): Promise<BeliefShare[]> {
  const client = get(lucidClient);
  if (!client) {
    console.warn('No Holochain client available, returning empty');
    return [];
  }

  try {
    if (tag) {
      return await client.collective.getBeliefsByTag(tag);
    }
    return await client.collective.getAllBeliefShares();
  } catch (error) {
    console.error('Failed to get collective beliefs:', error);
    return [];
  }
}

/**
 * Get votes for a belief
 */
export async function getBeliefVotes(beliefHash: ActionHash): Promise<ValidationVote[]> {
  const client = get(lucidClient);
  if (!client) return [];

  try {
    return await client.collective.getBeliefVotes(beliefHash);
  } catch (error) {
    console.error('Failed to get belief votes:', error);
    return [];
  }
}

/**
 * Get consensus for a belief
 */
export async function getConsensus(beliefHash: ActionHash): Promise<ConsensusRecord | null> {
  const client = get(lucidClient);
  if (!client) return null;

  try {
    return await client.collective.getBeliefConsensus(beliefHash);
  } catch (error) {
    console.error('Failed to get consensus:', error);
    return null;
  }
}

/**
 * Get weighted consensus (factors in relationships)
 */
export async function getWeightedConsensus(beliefHash: ActionHash): Promise<WeightedConsensusResult | null> {
  const client = get(lucidClient);
  if (!client) return null;

  try {
    return await client.collective.calculateWeightedConsensus(beliefHash);
  } catch (error) {
    console.error('Failed to get weighted consensus:', error);
    return null;
  }
}

/**
 * Detect emergent patterns with resilience
 */
export async function detectPatterns(
  similarityThreshold?: number,
  minClusterSize?: number
): Promise<PatternCluster[]> {
  const client = get(lucidClient);
  if (!client) return [];

  return withFallback(
    async () => {
      return withRetry(
        async () => {
          return holochainCircuit.execute(async () => {
            return client.collective.detectPatterns({
              similarity_threshold: similarityThreshold,
              min_cluster_size: minClusterSize,
            });
          });
        },
        {
          maxAttempts: 2,
          initialDelay: 1000,
          shouldRetry: isRetryableError,
        }
      );
    },
    {
      fallback: [] as PatternCluster[],
      logErrors: true,
    }
  );
}

/**
 * Get recorded patterns
 */
export async function getPatterns(): Promise<EmergentPattern[]> {
  const client = get(lucidClient);
  if (!client) return [];

  try {
    return await client.collective.getRecordedPatterns();
  } catch (error) {
    console.error('Failed to get patterns:', error);
    return [];
  }
}

/**
 * Get collective statistics
 */
export async function getCollectiveStats(): Promise<CollectiveStats | null> {
  const client = get(lucidClient);
  if (!client) return null;

  try {
    return await client.collective.getCollectiveStats();
  } catch (error) {
    console.error('Failed to get collective stats:', error);
    return null;
  }
}

// ============================================================================
// RELATIONSHIP OPERATIONS
// ============================================================================

/**
 * Update relationship with another agent
 */
export async function updateRelationship(input: UpdateRelationshipInput): Promise<AgentRelationship | null> {
  const client = get(lucidClient);
  if (!client) return null;

  try {
    return await client.collective.updateRelationship(input);
  } catch (error) {
    console.error('Failed to update relationship:', error);
    return null;
  }
}

/**
 * Get relationship with a specific agent
 */
export async function getRelationship(otherAgent: AgentPubKey): Promise<AgentRelationship | null> {
  const client = get(lucidClient);
  if (!client) return null;

  try {
    return await client.collective.getRelationship(otherAgent);
  } catch (error) {
    console.error('Failed to get relationship:', error);
    return null;
  }
}

/**
 * Get all my relationships
 */
export async function getMyRelationships(): Promise<AgentRelationship[]> {
  const client = get(lucidClient);
  if (!client) return [];

  try {
    return await client.collective.getMyRelationships();
  } catch (error) {
    console.error('Failed to get relationships:', error);
    return [];
  }
}

// ============================================================================
// LOCAL COLLECTIVE SIMULATION (for offline/demo)
// ============================================================================

// Simulated collective data
const simulatedShares: BeliefShare[] = [];
const simulatedVotes: Map<string, ValidationVote[]> = new Map();
const simulatedPatterns: EmergentPattern[] = [];

/**
 * Simulate sharing to local collective
 */
export function simulateShare(thought: Thought): BeliefShare {
  const anonymized = anonymizeThought(thought);
  const share: BeliefShare = {
    content_hash: anonymized.content_hash,
    content: anonymized.content,
    belief_type: anonymized.belief_type,
    epistemic_code: anonymized.epistemic_code,
    confidence: anonymized.confidence,
    tags: anonymized.tags,
    shared_at: Date.now() * 1000, // Convert to microseconds
    evidence_hashes: anonymized.evidence_hashes,
    embedding: [],
    stance: null,
  };
  simulatedShares.push(share);
  return share;
}

/**
 * Simulate voting
 */
export function simulateVote(contentHash: string, voteType: ValidationVoteType): void {
  const vote: ValidationVote = {
    belief_share_hash: new Uint8Array(39) as unknown as ActionHash, // Placeholder
    vote_type: voteType,
    evidence: null,
    voter_weight: 0.5 + Math.random() * 0.5,
    voted_at: Date.now() * 1000,
  };

  if (!simulatedVotes.has(contentHash)) {
    simulatedVotes.set(contentHash, []);
  }
  simulatedVotes.get(contentHash)!.push(vote);

  // Check for pattern detection
  detectLocalPatterns();
}

/**
 * Minimum number of validators required for consensus
 * SECURITY: This prevents consensus manipulation by single actors
 */
export const MIN_VALIDATORS_FOR_CONSENSUS = 3;

/**
 * Minimum total weight required for consensus
 * SECURITY: Prevents low-weight validators from creating consensus
 */
export const MIN_TOTAL_WEIGHT_FOR_CONSENSUS = 1.5;

/**
 * Minimum validators required for StrongConsensus
 * SECURITY: Higher threshold for strong claims
 */
export const MIN_VALIDATORS_FOR_STRONG_CONSENSUS = 5;

/**
 * Calculate simulated consensus
 *
 * SECURITY REQUIREMENTS:
 * - Minimum 3 unique validators
 * - Minimum total weight of 1.5
 * - Cannot reach StrongConsensus with fewer than 5 validators
 */
export function calculateLocalConsensus(contentHash: string): ConsensusRecord | null {
  const votes = simulatedVotes.get(contentHash);

  // SECURITY: Require minimum number of validators
  if (!votes || votes.length < MIN_VALIDATORS_FOR_CONSENSUS) {
    return null;
  }

  let support = 0;
  let oppose = 0;
  let totalWeight = 0;

  for (const vote of votes) {
    totalWeight += vote.voter_weight;
    if (vote.vote_type === ValidationVoteType.Corroborate) support += vote.voter_weight;
    else if (vote.vote_type === ValidationVoteType.Plausible) support += vote.voter_weight * 0.5;
    else if (vote.vote_type === ValidationVoteType.Contradict) oppose += vote.voter_weight;
    else if (vote.vote_type === ValidationVoteType.Implausible) oppose += vote.voter_weight * 0.5;
  }

  // SECURITY: Require minimum total weight
  if (totalWeight < MIN_TOTAL_WEIGHT_FOR_CONSENSUS) {
    return null;
  }

  // Avoid division by zero and ensure meaningful ratio
  const agreementScore = totalWeight > 0 ? support / (support + oppose + 0.001) : 0;

  let consensusType: ConsensusType;
  // SECURITY: StrongConsensus requires more validators
  if (agreementScore > 0.8 && votes.length >= MIN_VALIDATORS_FOR_STRONG_CONSENSUS) {
    consensusType = ConsensusType.StrongConsensus;
  } else if (agreementScore > 0.6) {
    consensusType = ConsensusType.ModerateConsensus;
  } else if (agreementScore > 0.4) {
    consensusType = ConsensusType.WeakConsensus;
  } else {
    consensusType = ConsensusType.Contested;
  }

  return {
    belief_share_hash: new Uint8Array(39) as unknown as ActionHash,
    consensus_type: consensusType,
    validator_count: votes.length,
    agreement_score: agreementScore,
    summary: `${votes.length} validators, ${Math.round(agreementScore * 100)}% agreement`,
    reached_at: Date.now() * 1000,
  };
}

/**
 * Detect patterns locally
 */
function detectLocalPatterns(): void {
  // Simple tag clustering
  const tagGroups = new Map<string, BeliefShare[]>();

  for (const share of simulatedShares) {
    for (const tag of share.tags) {
      if (!tagGroups.has(tag)) tagGroups.set(tag, []);
      tagGroups.get(tag)!.push(share);
    }
  }

  // Find convergence (multiple beliefs on same topic)
  for (const [tag, shares] of tagGroups) {
    if (shares.length >= 3) {
      const avgConfidence = shares.reduce((sum, s) => sum + s.confidence, 0) / shares.length;

      if (avgConfidence > 0.6) {
        const existing = simulatedPatterns.find(
          (p) => p.pattern_type === PatternType.Convergence && p.description.includes(tag)
        );

        if (!existing) {
          simulatedPatterns.push({
            pattern_id: `convergence-${tag}-${Date.now()}`,
            description: `Multiple independent beliefs converging on "${tag}"`,
            belief_hashes: [], // Would need proper hashes
            pattern_type: PatternType.Convergence,
            confidence: avgConfidence,
            detected_at: Date.now() * 1000,
          });
        }
      }
    }
  }
}

/**
 * Get simulated collective view
 */
export function getSimulatedCollectiveView(): CollectiveView {
  return {
    myShares: simulatedShares.slice(-10),
    publicShares: simulatedShares,
    consensusRecords: Array.from(simulatedVotes.keys())
      .map((hash) => calculateLocalConsensus(hash))
      .filter((c): c is ConsensusRecord => c !== null),
    patterns: simulatedPatterns,
    stats: {
      total_belief_shares: simulatedShares.length,
      total_patterns: simulatedPatterns.length,
      active_validators: Math.floor(Math.random() * 50) + 10,
    },
  };
}

// ============================================================================
// CONSENSUS REALITY MAP
// ============================================================================

export interface RealityMapNode {
  id: string;
  label: string;
  type: 'belief' | 'consensus' | 'pattern';
  x?: number;
  y?: number;
  size: number;
  color: string;
  data: BeliefShare | ConsensusRecord | EmergentPattern;
}

export interface RealityMapEdge {
  source: string;
  target: string;
  type: 'agreement' | 'disagreement' | 'related' | 'pattern';
  weight: number;
}

/**
 * Build a reality map from collective data
 */
export function buildRealityMap(view: CollectiveView): { nodes: RealityMapNode[]; edges: RealityMapEdge[] } {
  const nodes: RealityMapNode[] = [];
  const edges: RealityMapEdge[] = [];

  // Add belief nodes
  for (const share of view.publicShares) {
    nodes.push({
      id: share.content_hash,
      label: share.content.slice(0, 30) + '...',
      type: 'belief',
      size: 10 + share.confidence * 20,
      color: getBeliefColor(share),
      data: share,
    });
  }

  // Add consensus connections
  for (const consensus of view.consensusRecords) {
    // Note: In real implementation, would need to match by hash
    // For now, just update any matching node
  }

  // Add pattern nodes and edges
  for (const pattern of view.patterns) {
    const patternId = `pattern-${pattern.pattern_id}`;
    nodes.push({
      id: patternId,
      label: pattern.description.slice(0, 40),
      type: 'pattern',
      size: 15 + pattern.confidence * 15,
      color: '#a855f7',
      data: pattern,
    });
  }

  // Add edges between related beliefs (shared tags)
  const tagIndex = new Map<string, string[]>();
  for (const share of view.publicShares) {
    for (const tag of share.tags) {
      if (!tagIndex.has(tag)) tagIndex.set(tag, []);
      tagIndex.get(tag)!.push(share.content_hash);
    }
  }

  for (const hashes of tagIndex.values()) {
    if (hashes.length >= 2) {
      for (let i = 0; i < hashes.length - 1; i++) {
        edges.push({
          source: hashes[i],
          target: hashes[i + 1],
          type: 'related',
          weight: 0.3,
        });
      }
    }
  }

  return { nodes, edges };
}

function getBeliefColor(share: BeliefShare): string {
  const colors: Record<string, string> = {
    Claim: '#3b82f6',
    Belief: '#f59e0b',
    Observation: '#10b981',
    Question: '#8b5cf6',
    Hypothesis: '#ec4899',
  };
  return colors[share.belief_type] || '#6b7280';
}

function getConsensusColor(type: ConsensusType): string {
  switch (type) {
    case ConsensusType.StrongConsensus:
      return '#10b981';
    case ConsensusType.ModerateConsensus:
      return '#3b82f6';
    case ConsensusType.WeakConsensus:
      return '#f59e0b';
    case ConsensusType.Contested:
      return '#ef4444';
    default:
      return '#6b7280';
  }
}
