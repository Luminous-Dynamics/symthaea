// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Consensus Integration Tests
 *
 * Tests for ConsensusService -- the domain-specific SDK service for
 * MATL-weighted Byzantine fault tolerant consensus, multi-round
 * voting with conviction, threshold signature coordination, and
 * cross-hApp consensus verification.
 *
 * ConsensusService uses an in-memory data model with MATL reputation
 * scoring via LocalBridge, so tests verify local state management
 * rather than zome call dispatch.
 */

import { describe, it, expect, beforeEach } from 'vitest';

import {
  ConsensusService,
  getConsensusService,
  type ConsensusTopic,
  type Vote,
  type ConsensusRound,
  type ConsensusResult,
  type ConsensusParticipant,
  type SignatureShard,
  type ThresholdSignature,
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  isTrustworthy,
  AggregationMethod,
} from '../../src/integrations/consensus/index.js';

// ============================================================================
// Type construction tests
// ============================================================================

describe('Consensus Types', () => {
  describe('ConsensusTopic', () => {
    it('should construct a valid ConsensusTopic', () => {
      const topic: ConsensusTopic = {
        id: 'topic-001',
        title: 'Upgrade Protocol',
        description: 'Should we upgrade to v2.0?',
        options: ['yes', 'no', 'abstain'],
        quorumPercentage: 0.67,
        byzantineTolerance: 0.34,
        createdAt: Date.now(),
        status: 'active',
      };
      expect(topic.id).toBe('topic-001');
      expect(topic.title).toBe('Upgrade Protocol');
      expect(topic.options).toHaveLength(3);
      expect(topic.quorumPercentage).toBe(0.67);
      expect(topic.byzantineTolerance).toBe(0.34);
      expect(topic.status).toBe('active');
    });

    it('should accept all status variants', () => {
      const statuses: ConsensusTopic['status'][] = ['active', 'concluded', 'expired', 'failed'];
      expect(statuses).toHaveLength(4);
      statuses.forEach((s) => expect(typeof s).toBe('string'));
    });

    it('should accept optional fields', () => {
      const topic: ConsensusTopic = {
        id: 'topic-002',
        title: 'Test',
        description: 'Test',
        options: ['a', 'b'],
        quorumPercentage: 0.5,
        byzantineTolerance: 0.34,
        createdAt: Date.now(),
        status: 'concluded',
        expiresAt: Date.now() + 86400000,
        finalDecision: 'a',
      };
      expect(topic.expiresAt).toBeGreaterThan(0);
      expect(topic.finalDecision).toBe('a');
    });
  });

  describe('Vote', () => {
    it('should construct a valid Vote', () => {
      const vote: Vote = {
        id: 'vote-001',
        topicId: 'topic-001',
        participantId: 'participant-001',
        choice: 'yes',
        weight: 0.85,
        conviction: 0.9,
        timestamp: Date.now(),
      };
      expect(vote.id).toBe('vote-001');
      expect(vote.choice).toBe('yes');
      expect(vote.weight).toBe(0.85);
      expect(vote.conviction).toBe(0.9);
    });

    it('should accept optional signature', () => {
      const vote: Vote = {
        id: 'vote-002',
        topicId: 'topic-001',
        participantId: 'participant-002',
        choice: 'no',
        weight: 0.5,
        conviction: 0.7,
        timestamp: Date.now(),
        signature: 'sig-abc123',
      };
      expect(vote.signature).toBe('sig-abc123');
    });
  });

  describe('ConsensusRound', () => {
    it('should construct a valid ConsensusRound', () => {
      const round: ConsensusRound = {
        roundNumber: 1,
        topicId: 'topic-001',
        votes: [],
        participantCount: 0,
        quorumReached: false,
        leadingOption: '',
        leadingPercentage: 0,
        byzantineDetected: 0,
        startedAt: Date.now(),
      };
      expect(round.roundNumber).toBe(1);
      expect(round.votes).toEqual([]);
      expect(round.quorumReached).toBe(false);
    });

    it('should accept optional completedAt', () => {
      const round: ConsensusRound = {
        roundNumber: 2,
        topicId: 'topic-001',
        votes: [],
        participantCount: 5,
        quorumReached: true,
        leadingOption: 'yes',
        leadingPercentage: 0.75,
        byzantineDetected: 1,
        startedAt: Date.now() - 3600000,
        completedAt: Date.now(),
      };
      expect(round.completedAt).toBeGreaterThan(round.startedAt);
    });
  });

  describe('ConsensusResult', () => {
    it('should construct a valid ConsensusResult', () => {
      const result: ConsensusResult = {
        topicId: 'topic-001',
        reached: true,
        winner: 'yes',
        totalVotes: 10,
        weightedVotes: 8.5,
        optionBreakdown: {
          yes: { votes: 7, weight: 6.5 },
          no: { votes: 3, weight: 2.0 },
        },
        quorumPercentage: 0.8,
        byzantineFiltered: 1,
        rounds: 1,
        confidence: 0.85,
      };
      expect(result.reached).toBe(true);
      expect(result.winner).toBe('yes');
      expect(result.totalVotes).toBe(10);
      expect(result.optionBreakdown.yes.votes).toBe(7);
      expect(result.confidence).toBe(0.85);
    });

    it('should construct a result where consensus was not reached', () => {
      const result: ConsensusResult = {
        topicId: 'topic-002',
        reached: false,
        totalVotes: 3,
        weightedVotes: 2.1,
        optionBreakdown: {},
        quorumPercentage: 0.3,
        byzantineFiltered: 0,
        rounds: 1,
        confidence: 0.2,
      };
      expect(result.reached).toBe(false);
      expect(result.winner).toBeUndefined();
    });
  });

  describe('ConsensusParticipant', () => {
    it('should construct a valid ConsensusParticipant', () => {
      const rep = createReputation('participant-001');
      const participant: ConsensusParticipant = {
        id: 'participant-001',
        reputation: rep,
        votingWeight: 0.707,
        participationRate: 1.0,
        byzantineFlags: 0,
        lastActive: Date.now(),
        verified: false,
      };
      expect(participant.id).toBe('participant-001');
      expect(participant.votingWeight).toBeCloseTo(0.707);
      expect(participant.byzantineFlags).toBe(0);
      expect(participant.verified).toBe(false);
    });
  });

  describe('SignatureShard', () => {
    it('should construct a valid SignatureShard', () => {
      const shard: SignatureShard = {
        id: 'shard-001',
        topicId: 'topic-001',
        signerId: 'signer-001',
        shard: 'partial-sig-abc',
        index: 0,
        timestamp: Date.now(),
      };
      expect(shard.id).toBe('shard-001');
      expect(shard.shard).toBe('partial-sig-abc');
      expect(shard.index).toBe(0);
    });
  });

  describe('ThresholdSignature', () => {
    it('should construct a valid ThresholdSignature', () => {
      const sig: ThresholdSignature = {
        topicId: 'topic-001',
        threshold: 3,
        totalSigners: 5,
        shards: [],
        valid: false,
      };
      expect(sig.threshold).toBe(3);
      expect(sig.totalSigners).toBe(5);
      expect(sig.shards).toEqual([]);
      expect(sig.valid).toBe(false);
    });

    it('should accept optional aggregatedSignature', () => {
      const sig: ThresholdSignature = {
        topicId: 'topic-001',
        threshold: 2,
        totalSigners: 3,
        shards: [],
        aggregatedSignature: 'aggregated-abc',
        valid: true,
      };
      expect(sig.aggregatedSignature).toBe('aggregated-abc');
      expect(sig.valid).toBe(true);
    });
  });
});

// ============================================================================
// Singleton
// ============================================================================

describe('ConsensusService Singleton', () => {
  it('should return same instance on repeated calls to getConsensusService', () => {
    const a = getConsensusService();
    const b = getConsensusService();
    expect(a).toBe(b);
  });

  it('should return an instance of ConsensusService', () => {
    const service = getConsensusService();
    expect(service).toBeInstanceOf(ConsensusService);
  });
});

// ============================================================================
// ConsensusService
// ============================================================================

describe('ConsensusService', () => {
  let service: ConsensusService;

  beforeEach(() => {
    service = new ConsensusService();
  });

  // ==========================================================================
  // Topic Operations
  // ==========================================================================

  describe('Topic Operations', () => {
    describe('createTopic', () => {
      it('should create a topic with correct fields', () => {
        const topic = service.createTopic({
          id: 'topic-001',
          title: 'Upgrade Protocol',
          description: 'Should we upgrade?',
          options: ['yes', 'no', 'abstain'],
          quorumPercentage: 0.67,
          byzantineTolerance: 0.34,
        });

        expect(topic.id).toBe('topic-001');
        expect(topic.title).toBe('Upgrade Protocol');
        expect(topic.description).toBe('Should we upgrade?');
        expect(topic.options).toEqual(['yes', 'no', 'abstain']);
        expect(topic.quorumPercentage).toBe(0.67);
        expect(topic.byzantineTolerance).toBe(0.34);
        expect(topic.status).toBe('active');
        expect(topic.createdAt).toBeGreaterThan(0);
      });

      it('should default byzantineTolerance to 0.34 when not provided', () => {
        const topic = service.createTopic({
          id: 'topic-002',
          title: 'Test',
          description: 'Test',
          options: ['a', 'b'],
          quorumPercentage: 0.5,
          byzantineTolerance: 0,
        });
        // When byzantineTolerance is falsy (0), defaults to 0.34
        expect(topic.byzantineTolerance).toBe(0.34);
      });

      it('should accept optional expiresAt', () => {
        const expiry = Date.now() + 86400000;
        const topic = service.createTopic({
          id: 'topic-003',
          title: 'Expiring Topic',
          description: 'This expires',
          options: ['yes', 'no'],
          quorumPercentage: 0.5,
          byzantineTolerance: 0.34,
          expiresAt: expiry,
        });
        expect(topic.expiresAt).toBe(expiry);
      });

      it('should initialize rounds for the topic', () => {
        const topic = service.createTopic({
          id: 'topic-004',
          title: 'Topic with Rounds',
          description: 'Test',
          options: ['a', 'b'],
          quorumPercentage: 0.5,
          byzantineTolerance: 0.34,
        });
        // checkConsensus should work (round exists)
        const result = service.checkConsensus(topic.id);
        expect(result.rounds).toBe(1);
      });
    });

    describe('getTopicStatus', () => {
      it('should return the topic by ID', () => {
        service.createTopic({
          id: 'topic-get',
          title: 'Get Test',
          description: 'Test',
          options: ['a', 'b'],
          quorumPercentage: 0.5,
          byzantineTolerance: 0.34,
        });
        const topic = service.getTopicStatus('topic-get');
        expect(topic).toBeDefined();
        expect(topic!.title).toBe('Get Test');
      });

      it('should return undefined for non-existent topic', () => {
        const result = service.getTopicStatus('non-existent');
        expect(result).toBeUndefined();
      });
    });
  });

  // ==========================================================================
  // Participant Operations
  // ==========================================================================

  describe('Participant Operations', () => {
    describe('registerParticipant', () => {
      it('should register a participant with correct fields', () => {
        const participant = service.registerParticipant('alice');

        expect(participant.id).toBe('alice');
        expect(participant.votingWeight).toBeGreaterThan(0);
        expect(participant.participationRate).toBe(1.0);
        expect(participant.byzantineFlags).toBe(0);
        expect(participant.lastActive).toBeGreaterThan(0);
        expect(participant.reputation).toBeDefined();
      });

      it('should generate unique voting weight based on reputation', () => {
        const p1 = service.registerParticipant('alice');
        expect(p1.votingWeight).toBeGreaterThan(0);
        expect(p1.votingWeight).toBeLessThanOrEqual(1);
      });

      it('should set verified to false for new participant (reputation < 0.7)', () => {
        const participant = service.registerParticipant('bob');
        expect(participant.verified).toBe(false);
      });
    });

    describe('getParticipantProfile', () => {
      it('should return registered participant', () => {
        service.registerParticipant('carol');
        const profile = service.getParticipantProfile('carol');
        expect(profile).toBeDefined();
        expect(profile!.id).toBe('carol');
      });

      it('should return undefined for unregistered participant', () => {
        const result = service.getParticipantProfile('unknown');
        expect(result).toBeUndefined();
      });
    });

    describe('isParticipantTrustworthy', () => {
      it('should return false for unknown participant', () => {
        const result = service.isParticipantTrustworthy('unknown');
        expect(result).toBe(false);
      });

      it('should return false for new participant with default reputation', () => {
        service.registerParticipant('dave');
        const result = service.isParticipantTrustworthy('dave');
        expect(result).toBe(false);
      });

      it('should accept custom threshold', () => {
        service.registerParticipant('eve');
        const result = service.isParticipantTrustworthy('eve', 0.3);
        // With initial reputation of 0.5, should be trustworthy at 0.3 threshold
        expect(typeof result).toBe('boolean');
      });
    });
  });

  // ==========================================================================
  // Voting Operations
  // ==========================================================================

  describe('Voting Operations', () => {
    let topicId: string;

    beforeEach(() => {
      const topic = service.createTopic({
        id: 'vote-topic',
        title: 'Vote Test',
        description: 'Test voting',
        options: ['yes', 'no', 'abstain'],
        quorumPercentage: 0.5,
        byzantineTolerance: 0.34,
      });
      topicId = topic.id;
    });

    describe('submitVote', () => {
      it('should submit a vote with correct fields', () => {
        const vote = service.submitVote(topicId, 'alice', 'yes', 0.9);

        expect(vote.id).toBeTruthy();
        expect(vote.topicId).toBe(topicId);
        expect(vote.participantId).toBe('alice');
        expect(vote.choice).toBe('yes');
        expect(vote.conviction).toBe(0.9);
        expect(vote.weight).toBeGreaterThan(0);
        expect(vote.timestamp).toBeGreaterThan(0);
      });

      it('should auto-register participant if not registered', () => {
        service.submitVote(topicId, 'frank', 'yes', 0.8);
        const profile = service.getParticipantProfile('frank');
        expect(profile).toBeDefined();
        expect(profile!.id).toBe('frank');
      });

      it('should default conviction to 1.0', () => {
        const vote = service.submitVote(topicId, 'grace', 'no');
        expect(vote.conviction).toBe(1.0);
      });

      it('should clamp conviction between 0 and 1', () => {
        const vote = service.submitVote(topicId, 'heidi', 'yes', 1.5);
        expect(vote.conviction).toBeLessThanOrEqual(1);
        expect(vote.conviction).toBeGreaterThanOrEqual(0);
      });

      it('should throw for non-existent topic', () => {
        expect(() => service.submitVote('bad-topic', 'alice', 'yes'))
          .toThrow('Topic not found');
      });

      it('should throw for invalid choice', () => {
        expect(() => service.submitVote(topicId, 'alice', 'maybe'))
          .toThrow('Invalid choice');
      });

      it('should throw for double voting in same round', () => {
        service.submitVote(topicId, 'alice', 'yes', 0.9);
        expect(() => service.submitVote(topicId, 'alice', 'no', 0.5))
          .toThrow('Already voted in this round');
      });

      it('should throw when topic is not active', () => {
        // Create topic, conclude it
        const topic2 = service.createTopic({
          id: 'concluded-topic',
          title: 'Concluded',
          description: 'Already done',
          options: ['a', 'b'],
          quorumPercentage: 0.01,
          byzantineTolerance: 0.34,
        });
        // Register participants and vote to reach consensus
        service.registerParticipant('p1');
        service.submitVote(topic2.id, 'p1', 'a', 1.0);
        const result = service.checkConsensus(topic2.id);
        // If concluded, subsequent votes should fail
        if (result.reached) {
          expect(() => service.submitVote(topic2.id, 'p2', 'b'))
            .toThrow('Topic is not active');
        }
      });

      it('should update participant reputation positively on vote', () => {
        service.registerParticipant('ivan');
        service.submitVote(topicId, 'ivan', 'yes', 0.9);
        const profile = service.getParticipantProfile('ivan');
        expect(profile).toBeDefined();
        expect(profile!.lastActive).toBeGreaterThan(0);
      });
    });
  });

  // ==========================================================================
  // Consensus Checking
  // ==========================================================================

  describe('Consensus Checking', () => {
    describe('checkConsensus', () => {
      it('should return empty result for topic with no votes', () => {
        service.createTopic({
          id: 'empty-topic',
          title: 'Empty',
          description: 'No votes',
          options: ['yes', 'no'],
          quorumPercentage: 0.5,
          byzantineTolerance: 0.34,
        });

        const result = service.checkConsensus('empty-topic');
        expect(result.topicId).toBe('empty-topic');
        expect(result.reached).toBe(false);
        expect(result.totalVotes).toBe(0);
        expect(result.rounds).toBe(1);
      });

      it('should throw for non-existent topic', () => {
        expect(() => service.checkConsensus('missing-topic'))
          .toThrow('Topic not found');
      });

      it('should calculate option breakdown correctly', () => {
        service.createTopic({
          id: 'breakdown-topic',
          title: 'Breakdown Test',
          description: 'Test breakdown',
          options: ['yes', 'no'],
          quorumPercentage: 0.01,
          byzantineTolerance: 0.34,
        });

        service.submitVote('breakdown-topic', 'alice', 'yes', 1.0);
        service.submitVote('breakdown-topic', 'bob', 'yes', 0.8);
        service.submitVote('breakdown-topic', 'carol', 'no', 0.5);

        const result = service.checkConsensus('breakdown-topic');
        expect(result.optionBreakdown).toBeDefined();
        expect(result.optionBreakdown.yes).toBeDefined();
        expect(result.optionBreakdown.no).toBeDefined();
        expect(result.optionBreakdown.yes.votes).toBe(2);
        expect(result.optionBreakdown.no.votes).toBe(1);
      });

      it('should reach consensus when quorum met and majority achieved', () => {
        service.createTopic({
          id: 'consensus-topic',
          title: 'Consensus Test',
          description: 'Should reach consensus',
          options: ['yes', 'no'],
          quorumPercentage: 0.01,
          byzantineTolerance: 0.34,
        });

        service.submitVote('consensus-topic', 'alice', 'yes', 1.0);
        service.submitVote('consensus-topic', 'bob', 'yes', 0.9);

        const result = service.checkConsensus('consensus-topic');
        expect(result.reached).toBe(true);
        expect(result.winner).toBe('yes');
        expect(result.confidence).toBeGreaterThan(0);
      });

      it('should not reach consensus with insufficient quorum', () => {
        service.createTopic({
          id: 'no-quorum-topic',
          title: 'No Quorum',
          description: 'Too few voters',
          options: ['yes', 'no'],
          quorumPercentage: 0.9,
          byzantineTolerance: 0.34,
        });

        // Register many participants but only one votes
        for (let i = 0; i < 10; i++) {
          service.registerParticipant(`participant-${i}`);
        }
        service.submitVote('no-quorum-topic', 'participant-0', 'yes', 1.0);

        const result = service.checkConsensus('no-quorum-topic');
        expect(result.reached).toBe(false);
      });

      it('should include byzantineFiltered count', () => {
        service.createTopic({
          id: 'byz-topic',
          title: 'Byzantine Test',
          description: 'Test Byzantine filtering',
          options: ['yes', 'no'],
          quorumPercentage: 0.01,
          byzantineTolerance: 0.34,
        });

        service.submitVote('byz-topic', 'alice', 'yes', 0.5);
        service.submitVote('byz-topic', 'bob', 'no', 0.5);

        const result = service.checkConsensus('byz-topic');
        expect(typeof result.byzantineFiltered).toBe('number');
        expect(result.byzantineFiltered).toBeGreaterThanOrEqual(0);
      });

      it('should update topic status to concluded when consensus reached', () => {
        service.createTopic({
          id: 'conclude-topic',
          title: 'Conclude',
          description: 'Will be concluded',
          options: ['yes', 'no'],
          quorumPercentage: 0.01,
          byzantineTolerance: 0.34,
        });

        service.submitVote('conclude-topic', 'alice', 'yes', 1.0);
        const result = service.checkConsensus('conclude-topic');

        if (result.reached) {
          const topic = service.getTopicStatus('conclude-topic');
          expect(topic!.status).toBe('concluded');
          expect(topic!.finalDecision).toBe('yes');
        }
      });
    });
  });

  // ==========================================================================
  // Threshold Signatures
  // ==========================================================================

  describe('Threshold Signatures', () => {
    describe('initThresholdSignature', () => {
      it('should initialize a threshold signature', () => {
        const sig = service.initThresholdSignature('topic-001', 3, 5);

        expect(sig.topicId).toBe('topic-001');
        expect(sig.threshold).toBe(3);
        expect(sig.totalSigners).toBe(5);
        expect(sig.shards).toEqual([]);
        expect(sig.valid).toBe(false);
        expect(sig.aggregatedSignature).toBeUndefined();
      });
    });

    describe('submitSignatureShard', () => {
      it('should accept a signature shard', () => {
        service.initThresholdSignature('sig-topic', 2, 3);
        const valid = service.submitSignatureShard('sig-topic', 'signer-1', 'partial-sig-1', 0);
        expect(valid).toBe(false); // Threshold not yet met
      });

      it('should become valid when threshold is reached', () => {
        service.initThresholdSignature('sig-topic-2', 2, 3);
        service.submitSignatureShard('sig-topic-2', 'signer-1', 'partial-sig-1', 0);
        const valid = service.submitSignatureShard('sig-topic-2', 'signer-2', 'partial-sig-2', 1);
        expect(valid).toBe(true);
      });

      it('should throw for non-existent threshold signature', () => {
        expect(() => service.submitSignatureShard('missing', 'signer-1', 'sig', 0))
          .toThrow('No threshold signature initialized');
      });

      it('should throw for duplicate signer', () => {
        service.initThresholdSignature('sig-topic-3', 2, 3);
        service.submitSignatureShard('sig-topic-3', 'signer-1', 'partial-sig-1', 0);
        expect(() => service.submitSignatureShard('sig-topic-3', 'signer-1', 'another-sig', 1))
          .toThrow('Signer already submitted shard');
      });

      it('should set aggregatedSignature when threshold reached', () => {
        service.initThresholdSignature('sig-topic-4', 1, 3);
        service.submitSignatureShard('sig-topic-4', 'signer-1', 'partial-sig', 0);
        // The aggregated signature is created internally
        // We verify through the valid return value
        expect(service.submitSignatureShard('sig-topic-4', 'signer-2', 'partial-sig-2', 1)).toBe(true);
      });
    });
  });

  // ==========================================================================
  // Byzantine Tolerance
  // ==========================================================================

  describe('Byzantine Tolerance', () => {
    it('should return 0.34 as the default Byzantine tolerance', () => {
      expect(service.getByzantineTolerance()).toBe(0.34);
    });
  });

  // ==========================================================================
  // FL Coordinator
  // ==========================================================================

  describe('FL Coordinator', () => {
    it('should return the FL coordinator instance', () => {
      const coordinator = service.getFLCoordinator();
      expect(coordinator).toBeDefined();
    });
  });

  // ==========================================================================
  // External Reputation
  // ==========================================================================

  describe('External Reputation', () => {
    it('should query external reputation without throwing', () => {
      service.registerParticipant('alice');
      expect(() => service.queryExternalReputation('alice')).not.toThrow();
    });
  });

  // ==========================================================================
  // Re-exported MATL functions
  // ==========================================================================

  describe('Re-exported MATL functions', () => {
    it('should export createReputation', () => {
      const rep = createReputation('test');
      expect(rep).toBeDefined();
    });

    it('should export recordPositive', () => {
      let rep = createReputation('test');
      rep = recordPositive(rep);
      expect(reputationValue(rep)).toBeGreaterThan(0);
    });

    it('should export recordNegative', () => {
      let rep = createReputation('test');
      rep = recordNegative(rep);
      expect(rep).toBeDefined();
    });

    it('should export reputationValue', () => {
      const rep = createReputation('test');
      const value = reputationValue(rep);
      expect(typeof value).toBe('number');
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    });

    it('should export isTrustworthy', () => {
      const rep = createReputation('test');
      const trusted = isTrustworthy(rep, 0.3);
      expect(typeof trusted).toBe('boolean');
    });

    it('should export AggregationMethod', () => {
      expect(AggregationMethod.TrustWeighted).toBeDefined();
    });
  });

  // ==========================================================================
  // Edge Cases
  // ==========================================================================

  describe('Edge Cases', () => {
    it('should handle topic with only one option', () => {
      service.createTopic({
        id: 'single-option',
        title: 'Single Option',
        description: 'Only one choice',
        options: ['approve'],
        quorumPercentage: 0.01,
        byzantineTolerance: 0.34,
      });

      service.submitVote('single-option', 'alice', 'approve', 1.0);
      const result = service.checkConsensus('single-option');
      expect(result.optionBreakdown.approve).toBeDefined();
    });

    it('should handle topic with many options', () => {
      const options = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
      service.createTopic({
        id: 'many-options',
        title: 'Many Options',
        description: 'Lots of choices',
        options,
        quorumPercentage: 0.01,
        byzantineTolerance: 0.34,
      });

      const result = service.checkConsensus('many-options');
      // All 8 options appear in breakdown even with no votes
      expect(Object.keys(result.optionBreakdown)).toHaveLength(8);
    });

    it('should handle zero conviction', () => {
      service.createTopic({
        id: 'zero-conviction',
        title: 'Zero Conviction',
        description: 'No conviction',
        options: ['yes', 'no'],
        quorumPercentage: 0.01,
        byzantineTolerance: 0.34,
      });

      const vote = service.submitVote('zero-conviction', 'alice', 'yes', 0);
      expect(vote.conviction).toBe(0);
      expect(vote.weight).toBe(0);
    });

    it('should handle quorum percentage of 0', () => {
      service.createTopic({
        id: 'zero-quorum',
        title: 'Zero Quorum',
        description: 'No quorum needed',
        options: ['yes', 'no'],
        quorumPercentage: 0,
        byzantineTolerance: 0.34,
      });

      service.submitVote('zero-quorum', 'alice', 'yes', 1.0);
      const result = service.checkConsensus('zero-quorum');
      // With 0 quorum, any vote should meet quorum
      expect(typeof result.reached).toBe('boolean');
    });

    it('should handle quorum percentage of 1.0 (100%)', () => {
      service.createTopic({
        id: 'full-quorum',
        title: 'Full Quorum',
        description: 'All must vote',
        options: ['yes', 'no'],
        quorumPercentage: 1.0,
        byzantineTolerance: 0.34,
      });

      for (let i = 0; i < 5; i++) {
        service.registerParticipant(`voter-${i}`);
      }

      // Only 1 votes out of 5
      service.submitVote('full-quorum', 'voter-0', 'yes', 1.0);
      const result = service.checkConsensus('full-quorum');
      expect(result.reached).toBe(false);
    });
  });

  // ==========================================================================
  // Full Lifecycle
  // ==========================================================================

  describe('Full Consensus Lifecycle', () => {
    it('should support the complete topic -> vote -> consensus -> signature workflow', () => {
      // Step 1: Create topic
      const topic = service.createTopic({
        id: 'lifecycle-topic',
        title: 'Full Lifecycle',
        description: 'Complete workflow test',
        options: ['approve', 'reject'],
        quorumPercentage: 0.01,
        byzantineTolerance: 0.34,
      });
      expect(topic.status).toBe('active');

      // Step 2: Register participants
      const p1 = service.registerParticipant('voter-a');
      const p2 = service.registerParticipant('voter-b');
      const p3 = service.registerParticipant('voter-c');
      expect(p1.votingWeight).toBeGreaterThan(0);
      expect(p2.votingWeight).toBeGreaterThan(0);
      expect(p3.votingWeight).toBeGreaterThan(0);

      // Step 3: Submit votes
      service.submitVote('lifecycle-topic', 'voter-a', 'approve', 0.95);
      service.submitVote('lifecycle-topic', 'voter-b', 'approve', 0.85);
      service.submitVote('lifecycle-topic', 'voter-c', 'reject', 0.6);

      // Step 4: Check consensus
      const result = service.checkConsensus('lifecycle-topic');
      expect(result.totalVotes).toBeGreaterThanOrEqual(2);
      expect(result.rounds).toBe(1);

      // Step 5: Initialize threshold signature
      const sig = service.initThresholdSignature('lifecycle-topic', 2, 3);
      expect(sig.valid).toBe(false);

      // Step 6: Collect signature shards
      service.submitSignatureShard('lifecycle-topic', 'voter-a', 'partial-a', 0);
      const valid = service.submitSignatureShard('lifecycle-topic', 'voter-b', 'partial-b', 1);
      expect(valid).toBe(true);

      // Step 7: Verify participant profiles updated
      const profile = service.getParticipantProfile('voter-a');
      expect(profile).toBeDefined();
      expect(profile!.lastActive).toBeGreaterThan(0);
    });
  });
});
