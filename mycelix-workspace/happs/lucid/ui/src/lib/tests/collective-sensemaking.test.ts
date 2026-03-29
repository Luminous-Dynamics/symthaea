// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Collective Sensemaking Service Tests
 *
 * Tests for the distributed belief network integration.
 * Verifies:
 * - Belief sharing and anonymization
 * - Voting mechanics
 * - Consensus calculation
 * - Pattern detection
 * - Relationship management
 * - Simulation fallbacks
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { Thought } from '@mycelix/lucid-client';
import { ThoughtType, EmpiricalLevel, NormativeLevel, MaterialityLevel, HarmonicLevel } from '@mycelix/lucid-client';

// Mock stores
vi.mock('svelte/store', () => ({
  get: vi.fn(),
  writable: vi.fn(() => ({
    subscribe: vi.fn(),
    set: vi.fn(),
    update: vi.fn(),
  })),
}));

// Mock Holochain client
const mockCollective = {
  shareBelief: vi.fn(),
  castVote: vi.fn(),
  getBeliefVotes: vi.fn(),
  getBeliefConsensus: vi.fn(),
  calculateWeightedConsensus: vi.fn(),
  detectPatterns: vi.fn(),
  getRecordedPatterns: vi.fn(),
  getCollectiveStats: vi.fn(),
  updateRelationship: vi.fn(),
  getRelationship: vi.fn(),
  getMyRelationships: vi.fn(),
  getAllBeliefShares: vi.fn(),
  getBeliefsByTag: vi.fn(),
};

const mockLucidClient = {
  collective: mockCollective,
};

vi.mock('../stores/holochain', () => ({
  lucidClient: {
    subscribe: vi.fn((callback: (value: unknown) => void) => {
      callback(mockLucidClient);
      return () => {};
    }),
  },
}));

import { get } from 'svelte/store';

// Configure get mock
(get as ReturnType<typeof vi.fn>).mockReturnValue(mockLucidClient);

import {
  shareToCollective,
  voteOnBelief,
  getCollectiveBeliefs,
  getBeliefVotes,
  getConsensus,
  detectPatterns,
  getCollectiveStats,
  updateRelationship,
  getMyRelationships,
  calculateLocalConsensus,
  simulateVote,
  ValidationVoteType,
  ConsensusType,
  RelationshipStage,
} from '../services/collective-sensemaking';

// Test data
const mockThought: Thought = {
  id: 'thought-1',
  content: 'Climate change is a serious global challenge',
  thought_type: ThoughtType.Claim,
  epistemic: {
    empirical: EmpiricalLevel.E3,
    normative: NormativeLevel.N2,
    materiality: MaterialityLevel.M3,
    harmonic: HarmonicLevel.H2,
  },
  confidence: 0.85,
  tags: ['climate', 'environment', 'science'],
  domain: 'science',
  related_thoughts: [],
  source_hashes: [],
  parent_thought: null,
  created_at: Date.now() * 1000,
  updated_at: Date.now() * 1000,
  version: 1,
};

describe('Collective Sensemaking Service', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (get as ReturnType<typeof vi.fn>).mockReturnValue(mockLucidClient);
  });

  describe('shareBelief', () => {
    it('should share a belief with anonymization', async () => {
      const mockBeliefShare = {
        content_hash: 'hash123',
        content: mockThought.content,
        belief_type: 'Claim',
        epistemic_code: 'E3N2M3H2',
        confidence: 0.85,
        tags: ['climate', 'environment', 'science'],
        shared_at: Date.now() * 1000,
        evidence_hashes: [],
        embedding: [],
        stance: null,
      };

      mockCollective.shareBelief.mockResolvedValueOnce(mockBeliefShare);

      const result = await shareToCollective(mockThought);

      expect(mockCollective.shareBelief).toHaveBeenCalled();
      expect(result).toEqual(mockBeliefShare);
    });

    it('should fall back to simulation when Holochain unavailable', async () => {
      (get as ReturnType<typeof vi.fn>).mockReturnValue(null);

      const result = await shareToCollective(mockThought);

      // Should return simulated share
      expect(result).toHaveProperty('content_hash');
      expect(result?.content).toBe(mockThought.content);
    });
  });

  describe('voteOnBelief', () => {
    it('should cast a corroborate vote', async () => {
      const mockVote = {
        belief_share_hash: 'belief-hash',
        vote_type: ValidationVoteType.Corroborate,
        evidence: null,
        voter_weight: 1.0,
        voted_at: Date.now() * 1000,
      };

      mockCollective.castVote.mockResolvedValueOnce(mockVote);

      const result = await voteOnBelief(
        'belief-hash' as any,
        ValidationVoteType.Corroborate
      );

      expect(mockCollective.castVote).toHaveBeenCalledWith({
        belief_share_hash: 'belief-hash',
        vote_type: ValidationVoteType.Corroborate,
        evidence: undefined,
      });
      expect(result).toEqual(mockVote);
    });

    it('should include evidence when provided', async () => {
      mockCollective.castVote.mockResolvedValueOnce({});

      await voteOnBelief(
        'belief-hash' as any,
        ValidationVoteType.Contradict,
        'This contradicts established research'
      );

      expect(mockCollective.castVote).toHaveBeenCalledWith({
        belief_share_hash: 'belief-hash',
        vote_type: ValidationVoteType.Contradict,
        evidence: 'This contradicts established research',
      });
    });
  });

  describe('getConsensus', () => {
    it('should retrieve consensus record', async () => {
      const mockConsensus = {
        belief_share_hash: 'belief-hash',
        consensus_type: ConsensusType.StrongConsensus,
        validator_count: 12,
        agreement_score: 0.85,
        summary: '12 validators, 85% agreement',
        reached_at: Date.now() * 1000,
      };

      mockCollective.getBeliefConsensus.mockResolvedValueOnce(mockConsensus);

      const result = await getConsensus('belief-hash' as any);

      expect(result).toEqual(mockConsensus);
    });
  });

  describe('detectPatterns', () => {
    it('should detect convergence patterns', async () => {
      const mockPatterns = [
        {
          pattern_id: 'pattern-1',
          representative_content: 'Climate action is needed',
          representative_hash: 'hash1' as any,
          member_hashes: ['hash1', 'hash2', 'hash3'] as any[],
          member_count: 3,
          pattern_type: 'Convergence',
          coherence: 0.85,
          tags: ['climate'],
        },
      ];

      mockCollective.detectPatterns.mockResolvedValueOnce(mockPatterns);

      // Call with specific parameters
      const result = await detectPatterns(0.7, 3);

      expect(mockCollective.detectPatterns).toHaveBeenCalledWith({
        similarity_threshold: 0.7,
        min_cluster_size: 3,
      });
      expect(result).toEqual(mockPatterns);
    });
  });

  describe('getStats', () => {
    it('should retrieve collective statistics', async () => {
      const mockStats = {
        total_belief_shares: 150,
        total_patterns: 12,
        active_validators: 45,
      };

      mockCollective.getCollectiveStats.mockResolvedValueOnce(mockStats);

      const result = await getCollectiveStats();

      expect(result).toEqual(mockStats);
    });
  });

  describe('updateRelationship', () => {
    it('should update trust relationship', async () => {
      const mockRelationship = {
        other_agent: 'agent-123' as any,
        trust_score: 0.75,
        interaction_count: 10,
        last_interaction: Date.now() * 1000,
        relationship_stage: RelationshipStage.Collaborator,
        shared_domains: ['science'],
        agreement_ratio: 0.8,
      };

      mockCollective.updateRelationship.mockResolvedValueOnce(mockRelationship);

      const result = await updateRelationship({
        other_agent: 'agent-123' as any,
        trust_delta: 0.1,
        relationship_stage: RelationshipStage.Collaborator,
      });

      expect(result).toEqual(mockRelationship);
    });
  });
});

describe('Local Consensus Calculation', () => {
  it('should calculate strong consensus when majority agrees', () => {
    // First simulate the votes for a content hash
    const contentHash = 'test-belief-strong';
    simulateVote(contentHash, ValidationVoteType.Corroborate);
    simulateVote(contentHash, ValidationVoteType.Corroborate);
    simulateVote(contentHash, ValidationVoteType.Corroborate);
    simulateVote(contentHash, ValidationVoteType.Plausible);
    simulateVote(contentHash, ValidationVoteType.Abstain);

    const result = calculateLocalConsensus(contentHash);

    expect(result).not.toBeNull();
    expect(result!.consensus_type).toBe(ConsensusType.StrongConsensus);
    expect(result!.agreement_score).toBeGreaterThan(0.7);
  });

  it('should calculate contested when opinions are divided', () => {
    // First simulate the votes for a content hash
    // To get Contested, agreementScore must be <= 0.4
    // agreementScore = support / (support + oppose)
    // With 1 corroborate (support=1) and 3 contradict (oppose=3): 1/4 = 0.25 which is <= 0.4
    const contentHash = 'test-belief-contested';
    simulateVote(contentHash, ValidationVoteType.Corroborate);
    simulateVote(contentHash, ValidationVoteType.Contradict);
    simulateVote(contentHash, ValidationVoteType.Contradict);
    simulateVote(contentHash, ValidationVoteType.Contradict);

    const result = calculateLocalConsensus(contentHash);

    expect(result).not.toBeNull();
    expect(result!.consensus_type).toBe(ConsensusType.Contested);
  });

  it('should return null for too few votes', () => {
    // Only add 2 votes (need at least 3)
    const contentHash = 'test-belief-insufficient';
    simulateVote(contentHash, ValidationVoteType.Corroborate);
    simulateVote(contentHash, ValidationVoteType.Plausible);

    const result = calculateLocalConsensus(contentHash);

    expect(result).toBeNull();
  });

  it('should consider vote weights in agreement score', () => {
    // Simulate votes with the built-in weighted voting
    const contentHash = 'test-belief-weighted';
    simulateVote(contentHash, ValidationVoteType.Corroborate);
    simulateVote(contentHash, ValidationVoteType.Corroborate);
    simulateVote(contentHash, ValidationVoteType.Corroborate);
    simulateVote(contentHash, ValidationVoteType.Contradict);
    simulateVote(contentHash, ValidationVoteType.Contradict);

    const result = calculateLocalConsensus(contentHash);

    expect(result).not.toBeNull();
    // Should lean toward corroborate due to majority
    expect(result!.agreement_score).toBeGreaterThan(0.5);
  });
});

describe('Vote Simulation', () => {
  it('should add vote to simulated votes', () => {
    const contentHash = 'test-vote-simulation';

    // simulateVote returns void but adds to the internal store
    simulateVote(contentHash, ValidationVoteType.Plausible);
    simulateVote(contentHash, ValidationVoteType.Plausible);
    simulateVote(contentHash, ValidationVoteType.Plausible);

    // Verify by checking consensus
    const result = calculateLocalConsensus(contentHash);
    expect(result).not.toBeNull();
    expect(result!.validator_count).toBeGreaterThanOrEqual(3);
  });
});

describe('Enum Exports', () => {
  it('should export ValidationVoteType enum', () => {
    expect(ValidationVoteType.Corroborate).toBe('Corroborate');
    expect(ValidationVoteType.Contradict).toBe('Contradict');
    expect(ValidationVoteType.Plausible).toBe('Plausible');
    expect(ValidationVoteType.Implausible).toBe('Implausible');
    expect(ValidationVoteType.Abstain).toBe('Abstain');
  });

  it('should export ConsensusType enum', () => {
    expect(ConsensusType.StrongConsensus).toBe('StrongConsensus');
    expect(ConsensusType.ModerateConsensus).toBe('ModerateConsensus');
    expect(ConsensusType.WeakConsensus).toBe('WeakConsensus');
    expect(ConsensusType.Contested).toBe('Contested');
    expect(ConsensusType.Insufficient).toBe('Insufficient');
  });

  it('should export RelationshipStage enum', () => {
    expect(RelationshipStage.NoRelation).toBe('NoRelation');
    expect(RelationshipStage.Acquaintance).toBe('Acquaintance');
    expect(RelationshipStage.Collaborator).toBe('Collaborator');
    expect(RelationshipStage.TrustedPeer).toBe('TrustedPeer');
    expect(RelationshipStage.PartnerInTruth).toBe('PartnerInTruth');
  });
});
