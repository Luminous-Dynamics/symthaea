/**
 * Bridge Client Tests
 *
 * Verifies event broadcasting, consciousness integration, weighted consensus,
 * cross-hApp queries, and personal cluster dispatch for the BridgeClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { BridgeClient } from '../bridge';
import type { AppClient } from '@holochain/client';

// ============================================================================
// MOCK HELPERS
// ============================================================================

function createMockClient(): AppClient {
  return {
    callZome: vi.fn(),
  } as unknown as AppClient;
}

/** Record with entry in Present variant (used by extractEntry) */
function mockRecord(entry: Record<string, unknown>) {
  return {
    entry: { Present: entry },
    signed_action: { hashed: { hash: new Uint8Array(32) } },
  };
}

const EVENT_ENTRY = {
  id: 'evt-1',
  event_type: 'ProposalPassed',
  proposal_id: 'prop-42',
  subject: 'Proposal passed with 75% approval',
  payload: '{"approval_percentage":75}',
  source_happ: 'governance',
  timestamp: 1708300000,
};

const PARTICIPATION_SCORE = {
  did: 'did:mycelix:alice',
  dao_memberships: 3,
  proposals_created: 5,
  votes_cast: 42,
  participation_rate: 0.87,
  alignment_score: 0.72,
  delegation_trust: 0.65,
  overall_score: 0.78,
};

const GATE_RESULT = {
  passed: true,
  phi: 0.72,
  required_phi: 0.3,
  action_type: 'vote',
  failure_reason: null,
  gate_id: 'gate-123',
};

const GATE_V2_RESULT = {
  passed: true,
  phi: 0.72,
  required_phi: 0.3,
  provenance: 'Attested',
  action_type: 'vote',
  failure_reason: null,
};

const WEIGHT_RESULT = {
  reputation: 0.85,
  reputation_squared: 0.7225,
  phi: 0.72,
  consciousness_multiplier: 0.916,
  harmonic_alignment: 0.6,
  harmonic_bonus: 0.12,
  final_weight: 0.78,
  was_capped: false,
  uncapped_weight: 0.78,
  calculation_breakdown: 'rep²=0.7225 × (0.7+0.3×0.72) × (1+0.2×0.6)',
};

const ROUND_RESULT = {
  proposal_id: 'prop-42',
  round: 1,
  proposal_type: 'Standard',
  total_weight: 5.2,
  weighted_approvals: 3.8,
  weighted_rejections: 1.4,
  vote_count: 7,
  required_threshold: 0.5,
  approval_percentage: 73.08,
  quorum_met: true,
  consensus_reached: true,
  rejected: false,
  result: 'Approved',
};

// ============================================================================
// TESTS
// ============================================================================

describe('BridgeClient', () => {
  let client: BridgeClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new BridgeClient(mockAppClient);
  });

  // --------------------------------------------------------------------------
  // Initialization
  // --------------------------------------------------------------------------

  describe('initialization', () => {
    it('should create client with default config', () => {
      expect(client).toBeInstanceOf(BridgeClient);
    });

    it('should use governance as default role and bridge as zome', () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
      client.getRecentEvents();

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'governance',
          zome_name: 'bridge',
        })
      );
    });

    it('should accept custom role name', () => {
      const custom = new BridgeClient(mockAppClient, { roleName: 'custom' });
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
      custom.getRecentEvents();

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({ role_name: 'custom' })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Event Broadcasting
  // --------------------------------------------------------------------------

  describe('broadcastEvent', () => {
    it('should pass snake_case fields to zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(EVENT_ENTRY)
      );

      await client.broadcastEvent({
        eventType: 'ProposalPassed',
        proposalId: 'prop-42',
        subject: 'Passed',
        payload: '{}',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'broadcast_governance_event',
          payload: {
            event_type: 'ProposalPassed',
            proposal_id: 'prop-42',
            subject: 'Passed',
            payload: '{}',
          },
        })
      );
    });

    it('should map response to GovernanceBridgeEvent', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(EVENT_ENTRY)
      );

      const result = await client.broadcastEvent({
        eventType: 'ProposalPassed',
        proposalId: 'prop-42',
        subject: 'Passed',
        payload: '{}',
      });

      expect(result.id).toBe('evt-1');
      expect(result.eventType).toBe('ProposalPassed');
      expect(result.proposalId).toBe('prop-42');
      expect(result.timestamp).toBe(1708300000);
    });
  });

  describe('broadcastProposalCreated', () => {
    it('should compose event with proposer info', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(EVENT_ENTRY)
      );

      await client.broadcastProposalCreated('prop-42', 'did:mycelix:alice', 'Fund Treasury');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: expect.objectContaining({
            event_type: 'ProposalCreated',
            proposal_id: 'prop-42',
          }),
        })
      );
    });
  });

  describe('broadcastProposalPassed', () => {
    it('should include approval percentage', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(EVENT_ENTRY)
      );

      await client.broadcastProposalPassed('prop-42', 75);

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: expect.objectContaining({
            event_type: 'ProposalPassed',
            payload: JSON.stringify({ approval_percentage: 75 }),
          }),
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Event Queries
  // --------------------------------------------------------------------------

  describe('getRecentEvents', () => {
    it('should pass limit parameter', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
      await client.getRecentEvents(25);

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_recent_events',
          payload: 25,
        })
      );
    });

    it('should default limit to 50', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
      await client.getRecentEvents();

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({ payload: 50 })
      );
    });

    it('should map array of event records', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(EVENT_ENTRY),
        mockRecord({ ...EVENT_ENTRY, id: 'evt-2', event_type: 'VoteReceived' }),
      ]);

      const result = await client.getRecentEvents();
      expect(result).toHaveLength(2);
      expect(result[0].eventType).toBe('ProposalPassed');
      expect(result[1].eventType).toBe('VoteReceived');
    });
  });

  // --------------------------------------------------------------------------
  // Participation & Reputation
  // --------------------------------------------------------------------------

  describe('getParticipationScore', () => {
    it('should map snake_case response to camelCase', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        PARTICIPATION_SCORE
      );

      const result = await client.getParticipationScore('did:mycelix:alice');

      expect(result.did).toBe('did:mycelix:alice');
      expect(result.daoMemberships).toBe(3);
      expect(result.proposalsCreated).toBe(5);
      expect(result.votesCast).toBe(42);
      expect(result.participationRate).toBe(0.87);
      expect(result.overallScore).toBe(0.78);
    });
  });

  describe('reportParticipationScore', () => {
    it('should convert camelCase to snake_case', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      await client.reportParticipationScore({
        did: 'did:mycelix:alice',
        daoMemberships: 3,
        proposalsCreated: 5,
        votesCast: 42,
        participationRate: 0.87,
        alignmentScore: 0.72,
        delegationTrust: 0.65,
        overallScore: 0.78,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'report_participation_score',
          payload: expect.objectContaining({
            dao_memberships: 3,
            proposals_created: 5,
            votes_cast: 42,
          }),
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Consciousness Integration
  // --------------------------------------------------------------------------

  describe('verifyConsciousnessGate', () => {
    it('should map gate result correctly', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(GATE_RESULT);

      const result = await client.verifyConsciousnessGate({
        actionType: 'vote',
      });

      expect(result.passed).toBe(true);
      expect(result.phi).toBe(0.72);
      expect(result.requiredPhi).toBe(0.3);
      expect(result.actionType).toBe('vote');
      expect(result.gateId).toBe('gate-123');
      expect(result.failureReason).toBeUndefined();
    });

    it('should pass action_id as null when not provided', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(GATE_RESULT);

      await client.verifyConsciousnessGate({ actionType: 'vote' });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'verify_consciousness_gate',
          payload: { action_type: 'vote', action_id: null },
        })
      );
    });
  });

  describe('verifyConsciousnessGateV2', () => {
    it('should include provenance and actionType', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(GATE_V2_RESULT);

      const result = await client.verifyConsciousnessGateV2({
        actionType: 'vote',
      });

      expect(result.passed).toBe(true);
      expect(result.provenance).toBe('Attested');
      expect(result.actionType).toBe('vote');
    });
  });

  describe('recordPhiAttestation', () => {
    it('should pass phi, cycleId, capturedAtUs, signature', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({})
      );

      await client.recordPhiAttestation({
        phi: 0.72,
        cycleId: 1000,
        capturedAtUs: 1708300000000,
        signature: new Uint8Array([1, 2, 3]),
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'record_phi_attestation',
          payload: {
            phi: 0.72,
            cycle_id: 1000,
            captured_at_us: 1708300000000,
            signature: new Uint8Array([1, 2, 3]),
          },
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Phi Config
  // --------------------------------------------------------------------------

  describe('getPhiThresholds', () => {
    it('should map threshold tiers', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        basic: 0.1,
        proposal_submission: 0.3,
        voting: 0.2,
        constitutional: 0.5,
      });

      const result = await client.getPhiThresholds();

      expect(result.basic).toBe(0.1);
      expect(result.proposalSubmission).toBe(0.3);
      expect(result.voting).toBe(0.2);
      expect(result.constitutional).toBe(0.5);
    });
  });

  // --------------------------------------------------------------------------
  // Weighted Consensus
  // --------------------------------------------------------------------------

  describe('calculateHolisticVoteWeight', () => {
    it('should map weight breakdown', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(WEIGHT_RESULT);

      const result = await client.calculateHolisticVoteWeight({
        harmonicAlignment: 0.6,
      });

      expect(result.reputation).toBe(0.85);
      expect(result.reputationSquared).toBe(0.7225);
      expect(result.phi).toBe(0.72);
      expect(result.consciousnessMultiplier).toBe(0.916);
      expect(result.finalWeight).toBe(0.78);
      expect(result.wasCapped).toBe(false);
    });
  });

  describe('castWeightedVote', () => {
    it('should pass all fields including optional reason', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        vote_id: 'vote-1',
        weight: 0.78,
        weight_breakdown: 'rep²×consciousness×harmonic',
        decision: 'Approve',
        phi_at_vote: 0.72,
        proposal_type: 'Standard',
        threshold_required: 0.5,
      });

      const result = await client.castWeightedVote({
        proposalId: 'prop-42',
        proposalType: 'Standard',
        round: 1,
        decision: 'Approve',
        harmonicAlignment: 0.6,
        reason: 'Aligns with community values',
      });

      expect(result.voteId).toBe('vote-1');
      expect(result.decision).toBe('Approve');
      expect(result.phiAtVote).toBe(0.72);

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: expect.objectContaining({
            reason: 'Aligns with community values',
            harmonic_alignment: 0.6,
          }),
        })
      );
    });
  });

  describe('calculateRoundResult', () => {
    it('should map round result with approval percentage', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(ROUND_RESULT);

      const result = await client.calculateRoundResult({
        proposalId: 'prop-42',
        round: 1,
        proposalType: 'Standard',
        eligibleVoters: 10,
      });

      expect(result.proposalId).toBe('prop-42');
      expect(result.approvalPercentage).toBe(73.08);
      expect(result.quorumMet).toBe(true);
      expect(result.consensusReached).toBe(true);
      expect(result.result).toBe('Approved');
    });
  });

  // --------------------------------------------------------------------------
  // Personal Cluster Bridge
  // --------------------------------------------------------------------------

  describe('dispatchPersonalCall', () => {
    it('should pass zome_name and fn_name', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce({ ok: true });

      await client.dispatchPersonalCall({
        zomeName: 'identity_vault',
        fnName: 'get_profile',
        payload: null,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'dispatch_personal_call',
          payload: {
            zome_name: 'identity_vault',
            fn_name: 'get_profile',
            payload: null,
          },
        })
      );
    });
  });

  describe('requestPhiCredential', () => {
    it('should call with null payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce({});
      await client.requestPhiCredential();

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'request_phi_credential',
          payload: null,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Cross-hApp Proposals
  // --------------------------------------------------------------------------

  describe('registerCrossHappProposal', () => {
    it('should map camelCase input to snake_case', async () => {
      const entry = {
        id: 'xhp-1',
        original_proposal_hash: 'hash-abc',
        source_happ: 'finance',
        title: 'Budget Q1',
        proposal_type: 'Standard',
        status: 'Active',
        vote_weight_for: 100,
        vote_weight_against: 20,
        vote_weight_abstain: 5,
        voting_ends_at: 1710800000,
        created_at: 1708200000,
      };
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(entry)
      );

      const result = await client.registerCrossHappProposal({
        originalProposalHash: 'hash-abc',
        sourceHapp: 'finance',
        title: 'Budget Q1',
        proposalType: 'Standard',
        status: 'Active',
        voteWeightFor: 100,
        voteWeightAgainst: 20,
        voteWeightAbstain: 5,
        votingEndsAt: 1710800000,
        createdAt: 1708200000,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'register_cross_happ_proposal',
          payload: expect.objectContaining({
            original_proposal_hash: 'hash-abc',
            source_happ: 'finance',
            vote_weight_for: 100,
          }),
        })
      );
      expect(result.sourceHapp).toBe('finance');
      expect(result.voteWeightFor).toBe(100);
    });
  });

  describe('getCrossHappProposals', () => {
    it('should filter by source hApp', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
      await client.getCrossHappProposals('finance');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_cross_happ_proposals',
          payload: { source_happ: 'finance' },
        })
      );
    });

    it('should pass undefined source_happ when no filter', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
      await client.getCrossHappProposals();

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: { source_happ: undefined },
        })
      );
    });
  });

  describe('queryDelegationChain', () => {
    it('should return delegation DID chain', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        'did:mycelix:alice',
        'did:mycelix:bob',
        'did:mycelix:carol',
      ]);

      const result = await client.queryDelegationChain('dao-1', 'did:mycelix:alice', 'governance');

      expect(result).toEqual([
        'did:mycelix:alice',
        'did:mycelix:bob',
        'did:mycelix:carol',
      ]);
    });
  });

  // --------------------------------------------------------------------------
  // hApp Registration
  // --------------------------------------------------------------------------

  describe('registerHapp', () => {
    it('should send snake_case registration payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        registered: true,
        happId: 'finance',
      });

      const result = await client.registerHapp('finance', 'Finance hApp', ['voting', 'treasury']);

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'register_happ',
          payload: {
            happ_id: 'finance',
            happ_name: 'Finance hApp',
            capabilities: ['voting', 'treasury'],
          },
        })
      );
      expect(result.registered).toBe(true);
    });
  });

  describe('getRegisteredHapps', () => {
    it('should map snake_case response to camelCase', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        {
          happ_id: 'finance',
          happ_name: 'Finance hApp',
          capabilities: ['voting'],
          registered_at: 1708200000,
        },
      ]);

      const result = await client.getRegisteredHapps();

      expect(result).toHaveLength(1);
      expect(result[0].happId).toBe('finance');
      expect(result[0].happName).toBe('Finance hApp');
      expect(result[0].registeredAt).toBe(1708200000);
    });
  });

  // --------------------------------------------------------------------------
  // Additional Consciousness Methods
  // --------------------------------------------------------------------------

  describe('recordConsciousnessSnapshot', () => {
    it('should map all consciousness fields to snake_case', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockRecord({}));

      await client.recordConsciousnessSnapshot({
        phi: 0.65,
        metaAwareness: 0.7,
        selfModelAccuracy: 0.8,
        coherence: 0.75,
        affectiveValence: 0.6,
        careActivation: 0.5,
        source: 'symthaea',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'record_consciousness_snapshot',
          payload: {
            phi: 0.65,
            meta_awareness: 0.7,
            self_model_accuracy: 0.8,
            coherence: 0.75,
            affective_valence: 0.6,
            care_activation: 0.5,
            source: 'symthaea',
          },
        })
      );
    });

    it('should default source to null', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockRecord({}));

      await client.recordConsciousnessSnapshot({
        phi: 0.65,
        metaAwareness: 0.7,
        selfModelAccuracy: 0.8,
        coherence: 0.75,
        affectiveValence: 0.6,
        careActivation: 0.5,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: expect.objectContaining({ source: null }),
        })
      );
    });
  });

  describe('getAgentConsciousnessHistory', () => {
    it('should return null for missing agent', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getAgentConsciousnessHistory('did:mycelix:unknown');
      expect(result).toBeNull();
    });
  });

  describe('getAgentSnapshots', () => {
    it('should pass agent_did and limit', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);

      await client.getAgentSnapshots({ agentDid: 'did:mycelix:alice', limit: 10 });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_agent_snapshots',
          payload: { agent_did: 'did:mycelix:alice', limit: 10 },
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Additional Personal Cluster Methods
  // --------------------------------------------------------------------------

  describe('requestKVector', () => {
    it('should call with null payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce({ trust: [0.9] });
      const result = await client.requestKVector();

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'request_k_vector',
          payload: null,
        })
      );
      expect(result).toEqual({ trust: [0.9] });
    });
  });

  describe('requestIdentityProof', () => {
    it('should call with null payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce({ verified: true });
      await client.requestIdentityProof();

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'request_identity_proof',
          payload: null,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Participant Status & Federated Reputation
  // --------------------------------------------------------------------------

  describe('getParticipantStatus', () => {
    it('should map full status response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        agent_did: 'did:mycelix:alice',
        is_active: true,
        base_reputation: 0.85,
        effective_reputation: 0.92,
        streak_count: 5,
        streak_bonus: 0.07,
        in_cooldown: false,
        current_phi: 0.65,
        federated_score: 0.88,
        rounds_participated: 42,
        successful_votes: 38,
        success_rate: 0.905,
        slashing_events: 0,
        can_vote_standard: true,
        can_vote_emergency: true,
        can_vote_constitutional: false,
      });

      const result = await client.getParticipantStatus();

      expect(result.agentDid).toBe('did:mycelix:alice');
      expect(result.isActive).toBe(true);
      expect(result.streakCount).toBe(5);
      expect(result.successRate).toBe(0.905);
      expect(result.canVoteConstitutional).toBe(false);
    });
  });

  describe('getAdaptiveThreshold', () => {
    it('should map threshold response', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        base_threshold: 0.51,
        min_voter_phi: 0.2,
        min_participation: 0.33,
        quorum: 0.25,
        max_extension_secs: 86400,
      });

      const result = await client.getAdaptiveThreshold('Standard');

      expect(result.baseThreshold).toBe(0.51);
      expect(result.minVoterPhi).toBe(0.2);
      expect(result.quorum).toBe(0.25);
      expect(result.maxExtensionSecs).toBe(86400);
    });
  });

  describe('updateFederatedReputation', () => {
    it('should send optional fields as null when omitted', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockRecord({}));

      await client.updateFederatedReputation({
        pogqScore: 0.95,
        flContributions: 12,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'update_federated_reputation',
          payload: expect.objectContaining({
            pogq_score: 0.95,
            fl_contributions: 12,
            identity_verification: null,
            credential_count: null,
            stake_weight: null,
          }),
        })
      );
    });
  });

  describe('getRoundVotes', () => {
    it('should pass proposal_id and round', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);

      await client.getRoundVotes({ proposalId: 'prop-42', round: 1 });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_round_votes',
          payload: { proposal_id: 'prop-42', round: 1 },
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Additional Event Query Methods
  // --------------------------------------------------------------------------

  describe('getEventsByType', () => {
    it('should send event_type filter', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
      await client.getEventsByType('VoteReceived', 25);

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_events_by_type',
          payload: { event_type: 'VoteReceived', limit: 25 },
        })
      );
    });
  });

  describe('getEventsForDAO', () => {
    it('should pass dao_id and limit', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
      await client.getEventsForDAO(new Uint8Array(32), 10);

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_events_for_dao',
          payload: expect.objectContaining({ limit: 10 }),
        })
      );
    });
  });

  describe('broadcastProposalFailed', () => {
    it('should include failure reason', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(EVENT_ENTRY)
      );

      await client.broadcastProposalFailed('prop-42', 'Quorum not met');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: expect.objectContaining({
            event_type: 'ProposalFailed',
            payload: JSON.stringify({ reason: 'Quorum not met' }),
          }),
        })
      );
    });
  });

  describe('broadcastVoteReceived', () => {
    it('should include voter and weight info', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(EVENT_ENTRY)
      );

      await client.broadcastVoteReceived('prop-42', 'did:mycelix:alice', 'Approve', 0.85);

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: expect.objectContaining({
            event_type: 'VoteReceived',
            proposal_id: 'prop-42',
          }),
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Error Handling
  // --------------------------------------------------------------------------

  describe('error handling', () => {
    it('should wrap zome call failures', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Conductor unavailable')
      );

      await expect(client.getRecentEvents()).rejects.toThrow();
    });

    it('should propagate network errors on broadcast', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Network timeout')
      );

      await expect(
        client.broadcastEvent({
          eventType: 'ProposalPassed',
          proposalId: 'prop-42',
          subject: 'Test',
          payload: '{}',
        })
      ).rejects.toThrow();
    });
  });
});
