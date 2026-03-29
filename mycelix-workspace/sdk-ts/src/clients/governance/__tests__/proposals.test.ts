// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Proposals Client Tests
 *
 * Verifies zome call arguments, response mapping, lifecycle operations,
 * discussion system, and helper methods for the ProposalsClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ProposalsClient, VOTING_PERIODS } from '../proposals';
import type { AppClient } from '@holochain/client';

// ============================================================================
// MOCK HELPERS
// ============================================================================

function createMockClient(): AppClient {
  return {
    callZome: vi.fn(),
  } as unknown as AppClient;
}

function mockRecord(entry: Record<string, unknown>) {
  return {
    entry: { Present: entry },
    signed_action: { hashed: { hash: new Uint8Array(32) } },
  };
}

const PROPOSAL_ENTRY = {
  dao_id: 'dao-1',
  title: 'Fund Community Garden',
  description: 'Allocate 5000 MYC for garden project',
  proposer_did: 'did:mycelix:alice',
  proposal_type: 'Standard',
  status: 'Draft',
  category: 'Funding',
  voting_starts_at: 1708300000,
  voting_ends_at: 1708904800,
  quorum: 0.33,
  threshold: 0.51,
  approve_weight: 0,
  reject_weight: 0,
  abstain_weight: 0,
  voter_count: 0,
  execution_payload: '{"action":"transfer","amount":5000}',
  execution_target_happ: null,
  discussion_url: null,
  created_at: 1708200000,
  executed_at: null,
};

const FINALIZE_RESULT = {
  proposal_id: 'proposal-1',
  final_status: 'Approved',
  total_votes: 42,
  approve_weight: 30.5,
  reject_weight: 8.0,
  abstain_weight: 3.5,
  quorum_met: true,
  approval_percentage: 79.2,
  participation_rate: 0.85,
  passed: true,
};

const DISCUSSION_READINESS = {
  ready: true,
  reasoning: 'Discussion has sufficient diversity',
  contributor_count: 8,
  contribution_count: 24,
  harmony_diversity: 0.85,
  unaddressed_concerns: 1,
};

// ============================================================================
// TESTS
// ============================================================================

describe('ProposalsClient', () => {
  let client: ProposalsClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new ProposalsClient(mockAppClient);
  });

  // --------------------------------------------------------------------------
  // Initialization
  // --------------------------------------------------------------------------

  describe('initialization', () => {
    it('should create client with default config', () => {
      expect(client).toBeInstanceOf(ProposalsClient);
    });

    it('should use governance role and proposals zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(PROPOSAL_ENTRY)
      );

      await client.createProposal({
        daoId: 'dao-1',
        title: 'Test',
        description: 'Test desc',
        proposalType: 'Standard',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'governance',
          zome_name: 'proposals',
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // CRUD Operations
  // --------------------------------------------------------------------------

  describe('createProposal', () => {
    it('should pass snake_case payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(PROPOSAL_ENTRY)
      );

      await client.createProposal({
        daoId: 'dao-1',
        title: 'Fund Community Garden',
        description: 'Allocate 5000 MYC',
        proposalType: 'Standard',
        category: 'Funding',
        quorum: 0.33,
        threshold: 0.51,
        votingPeriodHours: 168,
        executionPayload: '{"action":"transfer"}',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_proposal',
          payload: {
            dao_id: 'dao-1',
            title: 'Fund Community Garden',
            description: 'Allocate 5000 MYC',
            proposal_type: 'Standard',
            category: 'Funding',
            quorum: 0.33,
            threshold: 0.51,
            voting_period_hours: 168,
            execution_payload: '{"action":"transfer"}',
            execution_target_happ: undefined,
            discussion_url: undefined,
          },
        })
      );
    });

    it('should map response to Proposal with camelCase', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(PROPOSAL_ENTRY)
      );

      const result = await client.createProposal({
        daoId: 'dao-1',
        title: 'Fund Community Garden',
        description: 'Allocate 5000 MYC',
        proposalType: 'Standard',
      });

      expect(result.daoId).toBe('dao-1');
      expect(result.title).toBe('Fund Community Garden');
      expect(result.proposerDid).toBe('did:mycelix:alice');
      expect(result.proposalType).toBe('Standard');
      expect(result.status).toBe('Draft');
      expect(result.quorum).toBe(0.33);
      expect(result.threshold).toBe(0.51);
      expect(result.createdAt).toBe(1708200000);
    });
  });

  describe('getProposal', () => {
    it('should return proposal for valid ID', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(PROPOSAL_ENTRY)
      );

      const result = await client.getProposal('proposal-1');
      expect(result).not.toBeNull();
      expect(result!.title).toBe('Fund Community Garden');
    });

    it('should return null for missing proposal', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getProposal('nonexistent');
      expect(result).toBeNull();
    });
  });

  describe('updateProposal', () => {
    it('should pass proposal_id and snake_case updates', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...PROPOSAL_ENTRY, title: 'Updated Title' })
      );

      const result = await client.updateProposal({
        proposalId: 'proposal-1',
        title: 'Updated Title',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'update_proposal',
          payload: {
            proposal_id: 'proposal-1',
            title: 'Updated Title',
            description: undefined,
            category: undefined,
            execution_payload: undefined,
            discussion_url: undefined,
          },
        })
      );
      expect(result.title).toBe('Updated Title');
    });
  });

  describe('listProposals', () => {
    it('should pass filter with snake_case fields', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(PROPOSAL_ENTRY),
      ]);

      await client.listProposals({
        daoId: 'dao-1',
        status: 'Active',
        proposerDid: 'did:mycelix:alice',
        proposalType: 'Standard',
        category: 'Funding',
        votingOpen: true,
        limit: 10,
        offset: 0,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'list_proposals',
          payload: {
            dao_id: 'dao-1',
            status: 'Active',
            proposer_did: 'did:mycelix:alice',
            proposal_type: 'Standard',
            category: 'Funding',
            voting_open: true,
            limit: 10,
            offset: 0,
          },
        })
      );
    });

    it('should map array of records', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(PROPOSAL_ENTRY),
        mockRecord({ ...PROPOSAL_ENTRY, title: 'Second Proposal' }),
      ]);

      const result = await client.listProposals({});
      expect(result).toHaveLength(2);
      expect(result[1].title).toBe('Second Proposal');
    });
  });

  // --------------------------------------------------------------------------
  // Convenience Query Methods
  // --------------------------------------------------------------------------

  describe('getActiveProposals', () => {
    it('should delegate to listProposals with Active status', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord({ ...PROPOSAL_ENTRY, status: 'Active' }),
      ]);

      const result = await client.getActiveProposals('dao-1', 5);

      expect(result).toHaveLength(1);
      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'list_proposals',
          payload: expect.objectContaining({
            dao_id: 'dao-1',
            status: 'Active',
            limit: 5,
          }),
        })
      );
    });
  });

  describe('getProposalsByProposer', () => {
    it('should delegate to listProposals with proposer filter', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);

      await client.getProposalsByProposer('did:mycelix:alice');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: expect.objectContaining({
            proposer_did: 'did:mycelix:alice',
          }),
        })
      );
    });
  });

  describe('getProposalsByCategory', () => {
    it('should delegate to listProposals with dao and category', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);

      await client.getProposalsByCategory('dao-1', 'Treasury');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: expect.objectContaining({
            dao_id: 'dao-1',
            category: 'Treasury',
          }),
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Lifecycle
  // --------------------------------------------------------------------------

  describe('activateProposal', () => {
    it('should send proposal ID as payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...PROPOSAL_ENTRY, status: 'Active' })
      );

      const result = await client.activateProposal('proposal-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'activate_proposal',
          payload: 'proposal-1',
        })
      );
      expect(result.status).toBe('Active');
    });
  });

  describe('cancelProposal', () => {
    it('should send proposal ID as payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...PROPOSAL_ENTRY, status: 'Cancelled' })
      );

      const result = await client.cancelProposal('proposal-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'cancel_proposal',
          payload: 'proposal-1',
        })
      );
      expect(result.status).toBe('Cancelled');
    });
  });

  describe('vetoProposal', () => {
    it('should pass proposal_id and reason', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...PROPOSAL_ENTRY, status: 'Cancelled' })
      );

      await client.vetoProposal('proposal-1', 'Violates charter');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'veto_proposal',
          payload: {
            proposal_id: 'proposal-1',
            reason: 'Violates charter',
          },
        })
      );
    });
  });

  describe('finalizeProposal', () => {
    it('should map ProposalResult with camelCase', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(FINALIZE_RESULT);

      const result = await client.finalizeProposal('proposal-1');

      expect(result.proposalId).toBe('proposal-1');
      expect(result.finalStatus).toBe('Approved');
      expect(result.totalVotes).toBe(42);
      expect(result.approveWeight).toBe(30.5);
      expect(result.rejectWeight).toBe(8.0);
      expect(result.abstainWeight).toBe(3.5);
      expect(result.quorumMet).toBe(true);
      expect(result.approvalPercentage).toBe(79.2);
      expect(result.participationRate).toBe(0.85);
      expect(result.passed).toBe(true);
    });
  });

  describe('markExecuted', () => {
    it('should send proposal ID as payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...PROPOSAL_ENTRY, status: 'Executed', executed_at: 1709000000 })
      );

      const result = await client.markExecuted('proposal-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'mark_executed',
          payload: 'proposal-1',
        })
      );
      expect(result.status).toBe('Executed');
      expect(result.executedAt).toBe(1709000000);
    });
  });

  describe('getRecentProposals', () => {
    it('should pass limit to zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(PROPOSAL_ENTRY),
      ]);

      const result = await client.getRecentProposals(5);

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_recent_proposals',
          payload: 5,
        })
      );
      expect(result).toHaveLength(1);
    });

    it('should default limit to 10', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);

      await client.getRecentProposals();

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: 10,
        })
      );
    });
  });

  describe('getProposalHistory', () => {
    it('should pass proposal ID to zome', async () => {
      const history = [
        { status: 'Draft', timestamp: 1708200000 },
        { status: 'Active', timestamp: 1708300000 },
      ];
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(history);

      const result = await client.getProposalHistory('proposal-1');

      expect(result).toHaveLength(2);
      expect(result[0].status).toBe('Draft');
    });
  });

  // --------------------------------------------------------------------------
  // Discussion System
  // --------------------------------------------------------------------------

  describe('generateProposalId', () => {
    it('should return string from zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce('MIP-0042');

      const id = await client.generateProposalId();
      expect(id).toBe('MIP-0042');
    });
  });

  describe('addContribution', () => {
    it('should pass snake_case payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({})
      );

      await client.addContribution({
        proposalId: 'proposal-1',
        contributorDid: 'did:mycelix:alice',
        content: 'Great idea!',
        harmonyTags: ['Compassion'],
        stance: 'Support',
        parentId: 'contrib-0',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'add_contribution',
          payload: {
            proposal_id: 'proposal-1',
            contributor_did: 'did:mycelix:alice',
            content: 'Great idea!',
            harmony_tags: ['Compassion'],
            stance: 'Support',
            parent_id: 'contrib-0',
          },
        })
      );
    });
  });

  describe('getDiscussion', () => {
    it('should pass proposal ID to zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord({}),
      ]);

      const result = await client.getDiscussion('proposal-1');
      expect(result).toHaveLength(1);
    });
  });

  describe('getReplies', () => {
    it('should pass contribution ID to zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);

      await client.getReplies('contrib-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_replies',
          payload: 'contrib-1',
        })
      );
    });
  });

  describe('reflectOnDiscussion', () => {
    it('should pass proposal ID to zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({})
      );

      await client.reflectOnDiscussion('proposal-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'reflect_on_discussion',
          payload: 'proposal-1',
        })
      );
    });
  });

  describe('getDiscussionReflections', () => {
    it('should return array of records', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord({}),
        mockRecord({}),
      ]);

      const result = await client.getDiscussionReflections('proposal-1');
      expect(result).toHaveLength(2);
    });
  });

  describe('isDiscussionReady', () => {
    it('should map response to DiscussionReadiness', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        DISCUSSION_READINESS
      );

      const result = await client.isDiscussionReady('proposal-1');

      expect(result.ready).toBe(true);
      expect(result.reasoning).toBe('Discussion has sufficient diversity');
      expect(result.contributorCount).toBe(8);
      expect(result.contributionCount).toBe(24);
      expect(result.harmonyDiversity).toBe(0.85);
      expect(result.unaddressedConcerns).toBe(1);
    });
  });

  // --------------------------------------------------------------------------
  // Helper Methods
  // --------------------------------------------------------------------------

  describe('isVotingOpen', () => {
    it('should return true for active proposal in voting window', () => {
      const now = Date.now() * 1000;
      const proposal = {
        id: 'p-1',
        daoId: 'dao-1',
        title: 'Test',
        description: '',
        proposerDid: 'did:mycelix:alice',
        proposalType: 'Standard' as const,
        status: 'Active' as const,
        votingStartsAt: now - 1000000,
        votingEndsAt: now + 1000000,
        quorum: 0.33,
        threshold: 0.51,
        approveWeight: 0,
        rejectWeight: 0,
        abstainWeight: 0,
        voterCount: 0,
        createdAt: now - 2000000,
      };

      expect(client.isVotingOpen(proposal)).toBe(true);
    });

    it('should return false for non-active proposal', () => {
      const now = Date.now() * 1000;
      const proposal = {
        id: 'p-1',
        daoId: 'dao-1',
        title: 'Test',
        description: '',
        proposerDid: 'did:mycelix:alice',
        proposalType: 'Standard' as const,
        status: 'Draft' as const,
        votingStartsAt: now - 1000000,
        votingEndsAt: now + 1000000,
        quorum: 0.33,
        threshold: 0.51,
        approveWeight: 0,
        rejectWeight: 0,
        abstainWeight: 0,
        voterCount: 0,
        createdAt: now - 2000000,
      };

      expect(client.isVotingOpen(proposal)).toBe(false);
    });
  });

  describe('getApprovalPercentage', () => {
    it('should calculate percentage from weights', () => {
      const proposal = {
        id: 'p-1', daoId: 'dao-1', title: '', description: '',
        proposerDid: '', proposalType: 'Standard' as const, status: 'Active' as const,
        votingStartsAt: 0, votingEndsAt: 0, quorum: 0.33, threshold: 0.51,
        approveWeight: 75, rejectWeight: 25, abstainWeight: 10,
        voterCount: 10, createdAt: 0,
      };

      expect(client.getApprovalPercentage(proposal)).toBe(75);
    });

    it('should return 0 when no votes', () => {
      const proposal = {
        id: 'p-1', daoId: 'dao-1', title: '', description: '',
        proposerDid: '', proposalType: 'Standard' as const, status: 'Active' as const,
        votingStartsAt: 0, votingEndsAt: 0, quorum: 0.33, threshold: 0.51,
        approveWeight: 0, rejectWeight: 0, abstainWeight: 0,
        voterCount: 0, createdAt: 0,
      };

      expect(client.getApprovalPercentage(proposal)).toBe(0);
    });
  });

  describe('hasQuorum', () => {
    it('should return true when quorum met', () => {
      const proposal = {
        id: 'p-1', daoId: 'dao-1', title: '', description: '',
        proposerDid: '', proposalType: 'Standard' as const, status: 'Active' as const,
        votingStartsAt: 0, votingEndsAt: 0, quorum: 0.33, threshold: 0.51,
        approveWeight: 30, rejectWeight: 10, abstainWeight: 5,
        voterCount: 0, createdAt: 0,
      };

      expect(client.hasQuorum(proposal, 100)).toBe(true);
    });

    it('should return false when quorum not met', () => {
      const proposal = {
        id: 'p-1', daoId: 'dao-1', title: '', description: '',
        proposerDid: '', proposalType: 'Standard' as const, status: 'Active' as const,
        votingStartsAt: 0, votingEndsAt: 0, quorum: 0.33, threshold: 0.51,
        approveWeight: 10, rejectWeight: 5, abstainWeight: 0,
        voterCount: 0, createdAt: 0,
      };

      expect(client.hasQuorum(proposal, 100)).toBe(false);
    });
  });

  describe('wouldPass', () => {
    it('should return true when quorum met and threshold exceeded', () => {
      const proposal = {
        id: 'p-1', daoId: 'dao-1', title: '', description: '',
        proposerDid: '', proposalType: 'Standard' as const, status: 'Active' as const,
        votingStartsAt: 0, votingEndsAt: 0, quorum: 0.33, threshold: 0.51,
        approveWeight: 70, rejectWeight: 30, abstainWeight: 10,
        voterCount: 0, createdAt: 0,
      };

      expect(client.wouldPass(proposal, 100)).toBe(true);
    });

    it('should return false when quorum not met', () => {
      const proposal = {
        id: 'p-1', daoId: 'dao-1', title: '', description: '',
        proposerDid: '', proposalType: 'Standard' as const, status: 'Active' as const,
        votingStartsAt: 0, votingEndsAt: 0, quorum: 0.33, threshold: 0.51,
        approveWeight: 5, rejectWeight: 0, abstainWeight: 0,
        voterCount: 0, createdAt: 0,
      };

      expect(client.wouldPass(proposal, 100)).toBe(false);
    });
  });

  describe('getVotingPeriod', () => {
    it('should return correct hours for each type', () => {
      expect(client.getVotingPeriod('Standard')).toBe(168);
      expect(client.getVotingPeriod('Emergency')).toBe(24);
      expect(client.getVotingPeriod('Constitutional')).toBe(720);
      expect(client.getVotingPeriod('Parameter')).toBe(168);
      expect(client.getVotingPeriod('Funding')).toBe(168);
    });
  });

  describe('getStatusDescription', () => {
    it('should return description for each status', () => {
      expect(client.getStatusDescription('Draft')).toContain('prepared');
      expect(client.getStatusDescription('Active')).toContain('voting');
      expect(client.getStatusDescription('Executed')).toContain('executed');
    });
  });

  // --------------------------------------------------------------------------
  // Error Handling
  // --------------------------------------------------------------------------

  describe('error handling', () => {
    it('should propagate zome errors on create', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Not a DAO member')
      );

      await expect(
        client.createProposal({
          daoId: 'dao-1',
          title: 'Test',
          description: 'Test',
          proposalType: 'Standard',
        })
      ).rejects.toThrow();
    });

    it('should propagate zome errors on finalize', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Voting period not ended')
      );

      await expect(client.finalizeProposal('proposal-1')).rejects.toThrow();
    });
  });
});
