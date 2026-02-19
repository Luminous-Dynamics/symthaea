/**
 * Voting Client Tests
 *
 * Verifies zome call arguments, response mapping, delegation, Φ-weighted
 * voting, quadratic voting, and utility functions for the VotingClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { VotingClient } from '../voting';
import type { AppClient } from '@holochain/client';

// ============================================================================
// MOCK HELPERS
// ============================================================================

function createMockClient(): AppClient {
  return {
    callZome: vi.fn(),
  } as unknown as AppClient;
}

/** Mock a Holochain record using the extractEntry() structure */
function mockRecord(entry: Record<string, unknown>) {
  return {
    entry: { Present: entry },
    signed_action: { hashed: { hash: new Uint8Array(32) } },
  };
}

const VOTE_ENTRY = {
  proposal_id: 'proposal-1',
  voter_did: 'did:mycelix:alice',
  choice: 'Approve',
  weight: 1.2,
  reason: 'Strong alignment',
  delegated_from: null,
  voted_at: 1708300000,
};

const QUORUM_STATUS = {
  proposal_id: 'proposal-1',
  current_votes: 15,
  required_votes: 10,
  quorum_met: true,
  quorum_percentage: 75.0,
};

const VOTING_STATS = {
  proposal_id: 'proposal-1',
  approve_weight: 8.5,
  reject_weight: 3.2,
  abstain_weight: 1.0,
  total_weight: 12.7,
  voter_count: 15,
  percentages: { approve: 66.9, reject: 25.2, abstain: 7.9 },
  is_active: true,
  time_remaining: 3600,
};

const POWER_BREAKDOWN = {
  base_power: 1.0,
  matl_multiplier: 1.2,
  participation_bonus: 0.1,
  delegated_power: 0.5,
  stake_multiplier: 1.0,
  total_power: 1.8,
};

// ============================================================================
// TESTS
// ============================================================================

describe('VotingClient', () => {
  let client: VotingClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new VotingClient(mockAppClient);
  });

  // --------------------------------------------------------------------------
  // Initialization
  // --------------------------------------------------------------------------

  describe('initialization', () => {
    it('should create client with default config', () => {
      expect(client).toBeInstanceOf(VotingClient);
    });

    it('should use governance role and voting zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(VOTE_ENTRY)
      );

      await client.castVote({
        proposalId: 'p1',
        choice: 'Approve',
        weight: 1.0,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'governance',
          zome_name: 'voting',
        })
      );
    });

    it('should accept custom role name', async () => {
      const custom = new VotingClient(mockAppClient, { roleName: 'custom' });
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(VOTE_ENTRY)
      );

      await custom.castVote({
        proposalId: 'p1',
        choice: 'Approve',
        weight: 1.0,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({ role_name: 'custom' })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Vote Casting
  // --------------------------------------------------------------------------

  describe('castVote', () => {
    it('should pass snake_case payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(VOTE_ENTRY)
      );

      await client.castVote({
        proposalId: 'proposal-1',
        choice: 'Approve',
        weight: 1.2,
        reason: 'Strong alignment',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'cast_vote',
          payload: {
            proposal_id: 'proposal-1',
            choice: 'Approve',
            weight: 1.2,
            reason: 'Strong alignment',
            delegated_from: undefined,
          },
        })
      );
    });

    it('should map response to Vote with camelCase', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(VOTE_ENTRY)
      );

      const result = await client.castVote({
        proposalId: 'proposal-1',
        choice: 'Approve',
        weight: 1.2,
      });

      expect(result.proposalId).toBe('proposal-1');
      expect(result.voterDid).toBe('did:mycelix:alice');
      expect(result.choice).toBe('Approve');
      expect(result.weight).toBe(1.2);
      expect(result.delegatedFrom).toBeNull();
      expect(result.votedAt).toBe(1708300000);
    });
  });

  describe('castVoteAuto', () => {
    it('should pass proposal_id and choice', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(VOTE_ENTRY)
      );

      await client.castVoteAuto('proposal-1', 'Approve', 'Good proposal');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'cast_vote_auto',
          payload: {
            proposal_id: 'proposal-1',
            choice: 'Approve',
            reason: 'Good proposal',
          },
        })
      );
    });
  });

  describe('castDelegatedVote', () => {
    it('should include delegator_did in payload', async () => {
      const delegatedVote = {
        ...VOTE_ENTRY,
        delegated_from: 'did:mycelix:bob',
      };
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(delegatedVote)
      );

      const result = await client.castDelegatedVote(
        'proposal-1',
        'did:mycelix:bob',
        'Approve',
        'On behalf of Bob'
      );

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'cast_delegated_vote',
          payload: {
            proposal_id: 'proposal-1',
            delegator_did: 'did:mycelix:bob',
            choice: 'Approve',
            reason: 'On behalf of Bob',
          },
        })
      );

      expect(result.delegatedFrom).toBe('did:mycelix:bob');
    });
  });

  describe('changeVote', () => {
    it('should pass new_choice to zome', async () => {
      const changedVote = { ...VOTE_ENTRY, choice: 'Reject' };
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(changedVote)
      );

      const result = await client.changeVote('proposal-1', 'Reject', 'Changed mind');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'change_vote',
          payload: {
            proposal_id: 'proposal-1',
            new_choice: 'Reject',
            reason: 'Changed mind',
          },
        })
      );

      expect(result.choice).toBe('Reject');
    });
  });

  // --------------------------------------------------------------------------
  // Vote Queries
  // --------------------------------------------------------------------------

  describe('getVotesForProposal', () => {
    it('should map array of records', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(VOTE_ENTRY),
        mockRecord({ ...VOTE_ENTRY, voter_did: 'did:mycelix:bob', choice: 'Reject' }),
      ]);

      const result = await client.getVotesForProposal('proposal-1');

      expect(result).toHaveLength(2);
      expect(result[0].voterDid).toBe('did:mycelix:alice');
      expect(result[1].voterDid).toBe('did:mycelix:bob');
      expect(result[1].choice).toBe('Reject');
    });
  });

  describe('getVote', () => {
    it('should return null for missing vote', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getVote('proposal-1', 'did:mycelix:unknown');
      expect(result).toBeNull();
    });

    it('should pass voter_did in payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(VOTE_ENTRY)
      );

      await client.getVote('proposal-1', 'did:mycelix:alice');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_vote',
          payload: {
            proposal_id: 'proposal-1',
            voter_did: 'did:mycelix:alice',
          },
        })
      );
    });
  });

  describe('getMyVote', () => {
    it('should return null when not voted', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getMyVote('proposal-1');
      expect(result).toBeNull();
    });
  });

  describe('hasVoted', () => {
    it('should return boolean from zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(true);

      const result = await client.hasVoted('proposal-1', 'did:mycelix:alice');
      expect(result).toBe(true);
    });
  });

  // --------------------------------------------------------------------------
  // Voting Power
  // --------------------------------------------------------------------------

  describe('getVotingPower', () => {
    it('should pass include_delegated flag', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(2.5);

      const result = await client.getVotingPower({
        daoId: 'dao-1',
        voterDid: 'did:mycelix:alice',
        includeDelegated: true,
      });

      expect(result).toBe(2.5);
      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_voting_power',
          payload: {
            dao_id: 'dao-1',
            voter_did: 'did:mycelix:alice',
            include_delegated: true,
          },
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Φ-Weighted Voting
  // --------------------------------------------------------------------------

  describe('castPhiWeightedVote', () => {
    it('should pass tier and voter_did', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({})
      );

      await client.castPhiWeightedVote({
        proposalId: 'proposal-1',
        voterDid: 'did:mycelix:alice',
        tier: 'Major',
        choice: 'For',
        reason: 'Important change',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'cast_phi_weighted_vote',
          payload: {
            proposal_id: 'proposal-1',
            voter_did: 'did:mycelix:alice',
            tier: 'Major',
            choice: 'For',
            reason: 'Important change',
          },
        })
      );
    });
  });

  describe('castDelegatedPhiVote', () => {
    it('should pass delegate_did and tier', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({})
      );

      await client.castDelegatedPhiVote({
        proposalId: 'proposal-1',
        delegateDid: 'did:mycelix:bob',
        tier: 'Constitutional',
        choice: 'Against',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'cast_delegated_phi_vote',
          payload: {
            proposal_id: 'proposal-1',
            delegate_did: 'did:mycelix:bob',
            tier: 'Constitutional',
            choice: 'Against',
            reason: undefined,
          },
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Quadratic Voting
  // --------------------------------------------------------------------------

  describe('castQuadraticVote', () => {
    it('should pass credits_to_spend', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({})
      );

      await client.castQuadraticVote({
        proposalId: 'proposal-1',
        voterDid: 'did:mycelix:alice',
        choice: 'For',
        creditsToSpend: 9,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'cast_quadratic_vote',
          payload: {
            proposal_id: 'proposal-1',
            voter_did: 'did:mycelix:alice',
            choice: 'For',
            credits_to_spend: 9,
            reason: undefined,
          },
        })
      );
    });
  });

  describe('queryVoiceCredits', () => {
    it('should map response to VoiceCredits', async () => {
      const credits = {
        owner: 'did:mycelix:alice',
        allocated: 100,
        spent: 25,
        remaining: 75,
        period_start: 1708000000,
        period_end: 1710000000,
      };
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(credits);

      const result = await client.queryVoiceCredits('did:mycelix:alice');

      expect(result.remaining).toBe(75);
    });
  });

  // --------------------------------------------------------------------------
  // Tallying
  // --------------------------------------------------------------------------

  describe('tallyVotes', () => {
    it('should pass tier and thresholds', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({})
      );

      await client.tallyVotes({
        proposalId: 'proposal-1',
        tier: 'Basic',
        quorumOverride: 0.3,
        approvalOverride: 0.51,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'tally_votes',
          payload: {
            proposal_id: 'proposal-1',
            tier: 'Basic',
            quorum_override: 0.3,
            approval_override: 0.51,
          },
        })
      );
    });
  });

  describe('getProposalTally', () => {
    it('should return null for untallied proposal', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getProposalTally('proposal-1');
      expect(result).toBeNull();
    });
  });

  // --------------------------------------------------------------------------
  // Utility Functions
  // --------------------------------------------------------------------------

  describe('aggregateVotes', () => {
    it('should sum weights by choice', () => {
      const votes = [
        { id: '1', proposalId: 'p1', voterDid: 'a', choice: 'Approve' as const, weight: 1.5, votedAt: 0 },
        { id: '2', proposalId: 'p1', voterDid: 'b', choice: 'Reject' as const, weight: 0.8, votedAt: 0 },
        { id: '3', proposalId: 'p1', voterDid: 'c', choice: 'Approve' as const, weight: 1.2, votedAt: 0 },
        { id: '4', proposalId: 'p1', voterDid: 'd', choice: 'Abstain' as const, weight: 0.5, votedAt: 0 },
      ];

      const agg = client.aggregateVotes(votes);

      expect(agg.approve).toBeCloseTo(2.7);
      expect(agg.reject).toBeCloseTo(0.8);
      expect(agg.abstain).toBeCloseTo(0.5);
      expect(agg.voterCount).toBe(4);
      expect(agg.total).toBeCloseTo(4.0);
    });
  });

  describe('calculateParticipationRate', () => {
    it('should return percentage', () => {
      expect(client.calculateParticipationRate(75, 100)).toBe(0.75);
    });

    it('should handle zero eligible', () => {
      expect(client.calculateParticipationRate(0, 0)).toBe(0);
    });
  });

  // --------------------------------------------------------------------------
  // Error Handling
  // --------------------------------------------------------------------------

  describe('error handling', () => {
    it('should propagate zome errors on cast', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Already voted')
      );

      await expect(
        client.castVote({ proposalId: 'p1', choice: 'Approve', weight: 1.0 })
      ).rejects.toThrow();
    });

    it('should propagate zome errors on query', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Network timeout')
      );

      await expect(client.getVotesForProposal('p1')).rejects.toThrow();
    });
  });
});
