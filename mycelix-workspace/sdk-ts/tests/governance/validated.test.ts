// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Tests for governance/validated.ts
 *
 * Tests validation schemas and validated client wrappers for Governance hApp.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  ValidatedProposalsClient,
  ValidatedVotingClient,
  ValidatedDAOClient,
  ValidatedDelegationClient,
  ValidatedExecutionClient,
  createValidatedGovernanceClients,
} from '../../src/governance/validated.js';
import { MycelixError, ErrorCode } from '../../src/errors.js';

// Mock zome callable
const createMockZomeCallable = () => ({
  callZome: vi.fn().mockResolvedValue({ entry: {}, action_hash: 'test-hash' }),
});

describe('Governance Validated Clients', () => {
  let mockZomeClient: ReturnType<typeof createMockZomeCallable>;

  beforeEach(() => {
    mockZomeClient = createMockZomeCallable();
    vi.clearAllMocks();
  });

  describe('ValidatedProposalsClient', () => {
    let client: ValidatedProposalsClient;

    beforeEach(() => {
      client = new ValidatedProposalsClient(mockZomeClient);
    });

    describe('createProposal', () => {
      const validInput = {
        dao_id: 'dao-123',
        title: 'Test Proposal',
        description: 'A test proposal description',
        proposal_type: 'Standard' as const,
        voting_period_hours: 24,
        quorum: 0.5,
      };

      it('should accept valid input', async () => {
        await expect(client.createProposal(validInput)).resolves.toBeDefined();
      });

      it('should reject empty dao_id', async () => {
        await expect(
          client.createProposal({ ...validInput, dao_id: '' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject empty title', async () => {
        await expect(
          client.createProposal({ ...validInput, title: '' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject title over 200 characters', async () => {
        await expect(
          client.createProposal({ ...validInput, title: 'x'.repeat(201) })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject empty description', async () => {
        await expect(
          client.createProposal({ ...validInput, description: '' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject invalid proposal_type', async () => {
        await expect(
          client.createProposal({ ...validInput, proposal_type: 'Invalid' as any })
        ).rejects.toThrow(MycelixError);
      });

      it('should accept Emergency proposal_type', async () => {
        await expect(
          client.createProposal({ ...validInput, proposal_type: 'Emergency' })
        ).resolves.toBeDefined();
      });

      it('should accept Constitutional proposal_type', async () => {
        await expect(
          client.createProposal({ ...validInput, proposal_type: 'Constitutional' })
        ).resolves.toBeDefined();
      });

      it('should reject voting_period_hours less than 1', async () => {
        await expect(
          client.createProposal({ ...validInput, voting_period_hours: 0 })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject voting_period_hours greater than 720', async () => {
        await expect(
          client.createProposal({ ...validInput, voting_period_hours: 721 })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject negative quorum', async () => {
        await expect(
          client.createProposal({ ...validInput, quorum: -0.1 })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject quorum greater than 1', async () => {
        await expect(
          client.createProposal({ ...validInput, quorum: 1.1 })
        ).rejects.toThrow(MycelixError);
      });

      it('should accept optional threshold', async () => {
        await expect(
          client.createProposal({ ...validInput, threshold: 0.6 })
        ).resolves.toBeDefined();
      });

      it('should reject threshold greater than 1', async () => {
        await expect(
          client.createProposal({ ...validInput, threshold: 1.5 })
        ).rejects.toThrow(MycelixError);
      });

      it('should accept optional execution_payload', async () => {
        await expect(
          client.createProposal({ ...validInput, execution_payload: 'payload' })
        ).resolves.toBeDefined();
      });

      it('should accept optional category', async () => {
        await expect(
          client.createProposal({ ...validInput, category: 'governance' })
        ).resolves.toBeDefined();
      });
    });

    describe('getProposal', () => {
      it('should accept valid proposalId', async () => {
        await expect(client.getProposal('proposal-123')).resolves.toBeDefined();
      });

      it('should reject empty proposalId', async () => {
        await expect(client.getProposal('')).rejects.toThrow(MycelixError);
      });
    });

    describe('getProposalsByDAO', () => {
      it('should accept valid daoId', async () => {
        await expect(client.getProposalsByDAO('dao-123')).resolves.toBeDefined();
      });

      it('should reject empty daoId', async () => {
        await expect(client.getProposalsByDAO('')).rejects.toThrow(MycelixError);
      });
    });

    describe('getActiveProposals', () => {
      it('should accept valid daoId', async () => {
        await expect(client.getActiveProposals('dao-123')).resolves.toBeDefined();
      });

      it('should reject empty daoId', async () => {
        await expect(client.getActiveProposals('')).rejects.toThrow(MycelixError);
      });
    });

    describe('getProposalsByProposer', () => {
      it('should accept valid DID', async () => {
        await expect(client.getProposalsByProposer('did:example:123')).resolves.toBeDefined();
      });

      it('should reject non-DID string', async () => {
        await expect(client.getProposalsByProposer('not-a-did')).rejects.toThrow(MycelixError);
      });
    });

    describe('cancelProposal', () => {
      it('should accept valid proposalId', async () => {
        await expect(client.cancelProposal('proposal-123')).resolves.toBeDefined();
      });

      it('should reject empty proposalId', async () => {
        await expect(client.cancelProposal('')).rejects.toThrow(MycelixError);
      });
    });

    describe('finalizeProposal', () => {
      it('should accept valid proposalId', async () => {
        await expect(client.finalizeProposal('proposal-123')).resolves.toBeDefined();
      });

      it('should reject empty proposalId', async () => {
        await expect(client.finalizeProposal('')).rejects.toThrow(MycelixError);
      });
    });
  });

  describe('ValidatedVotingClient', () => {
    let client: ValidatedVotingClient;

    beforeEach(() => {
      client = new ValidatedVotingClient(mockZomeClient);
    });

    describe('castVote', () => {
      const validInput = {
        proposal_id: 'proposal-123',
        choice: 'Approve' as const,
        weight: 1,
      };

      it('should accept valid input', async () => {
        await expect(client.castVote(validInput)).resolves.toBeDefined();
      });

      it('should reject empty proposal_id', async () => {
        await expect(
          client.castVote({ ...validInput, proposal_id: '' })
        ).rejects.toThrow(MycelixError);
      });

      it('should accept Reject choice', async () => {
        await expect(
          client.castVote({ ...validInput, choice: 'Reject' })
        ).resolves.toBeDefined();
      });

      it('should accept Abstain choice', async () => {
        await expect(
          client.castVote({ ...validInput, choice: 'Abstain' })
        ).resolves.toBeDefined();
      });

      it('should reject invalid choice', async () => {
        await expect(
          client.castVote({ ...validInput, choice: 'Invalid' as any })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject negative weight', async () => {
        await expect(
          client.castVote({ ...validInput, weight: -1 })
        ).rejects.toThrow(MycelixError);
      });

      it('should accept zero weight', async () => {
        await expect(
          client.castVote({ ...validInput, weight: 0 })
        ).resolves.toBeDefined();
      });

      it('should accept optional reason', async () => {
        await expect(
          client.castVote({ ...validInput, reason: 'My reasoning' })
        ).resolves.toBeDefined();
      });
    });

    describe('getVotesForProposal', () => {
      it('should accept valid proposalId', async () => {
        await expect(client.getVotesForProposal('proposal-123')).resolves.toBeDefined();
      });

      it('should reject empty proposalId', async () => {
        await expect(client.getVotesForProposal('')).rejects.toThrow(MycelixError);
      });
    });

    describe('getVotesByVoter', () => {
      it('should accept valid DID', async () => {
        await expect(client.getVotesByVoter('did:example:voter')).resolves.toBeDefined();
      });

      it('should reject non-DID string', async () => {
        await expect(client.getVotesByVoter('invalid')).rejects.toThrow(MycelixError);
      });
    });

    describe('hasVoted', () => {
      it('should accept valid inputs', async () => {
        await expect(client.hasVoted('proposal-123', 'did:example:voter')).resolves.toBeDefined();
      });

      it('should reject empty proposalId', async () => {
        await expect(client.hasVoted('', 'did:example:voter')).rejects.toThrow(MycelixError);
      });

      it('should reject invalid voterDid', async () => {
        await expect(client.hasVoted('proposal-123', 'invalid')).rejects.toThrow(MycelixError);
      });
    });

    describe('getVoteTally', () => {
      it('should accept valid proposalId', async () => {
        await expect(client.getVoteTally('proposal-123')).resolves.toBeDefined();
      });

      it('should reject empty proposalId', async () => {
        await expect(client.getVoteTally('')).rejects.toThrow(MycelixError);
      });
    });
  });

  describe('ValidatedDAOClient', () => {
    let client: ValidatedDAOClient;

    beforeEach(() => {
      client = new ValidatedDAOClient(mockZomeClient);
    });

    describe('createDAO', () => {
      const validInput = {
        name: 'Test DAO',
        description: 'A test DAO',
        default_voting_period: 24,
        default_quorum: 0.5,
        default_threshold: 0.6,
      };

      it('should accept valid input', async () => {
        await expect(client.createDAO(validInput)).resolves.toBeDefined();
      });

      it('should reject empty name', async () => {
        await expect(
          client.createDAO({ ...validInput, name: '' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject name over 100 characters', async () => {
        await expect(
          client.createDAO({ ...validInput, name: 'x'.repeat(101) })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject empty description', async () => {
        await expect(
          client.createDAO({ ...validInput, description: '' })
        ).rejects.toThrow(MycelixError);
      });

      it('should accept optional charter_hash', async () => {
        await expect(
          client.createDAO({ ...validInput, charter_hash: 'hash123' })
        ).resolves.toBeDefined();
      });

      it('should reject voting period less than 1', async () => {
        await expect(
          client.createDAO({ ...validInput, default_voting_period: 0 })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject voting period greater than 720', async () => {
        await expect(
          client.createDAO({ ...validInput, default_voting_period: 721 })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject negative default_quorum', async () => {
        await expect(
          client.createDAO({ ...validInput, default_quorum: -0.1 })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject default_quorum greater than 1', async () => {
        await expect(
          client.createDAO({ ...validInput, default_quorum: 1.5 })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject negative default_threshold', async () => {
        await expect(
          client.createDAO({ ...validInput, default_threshold: -0.1 })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject default_threshold greater than 1', async () => {
        await expect(
          client.createDAO({ ...validInput, default_threshold: 1.5 })
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('getDAO', () => {
      it('should accept valid daoId', async () => {
        await expect(client.getDAO('dao-123')).resolves.toBeDefined();
      });

      it('should reject empty daoId', async () => {
        await expect(client.getDAO('')).rejects.toThrow(MycelixError);
      });
    });

    describe('listDAOs', () => {
      it('should not require any validation', async () => {
        await expect(client.listDAOs()).resolves.toBeDefined();
      });
    });

    describe('joinDAO', () => {
      it('should accept valid daoId with default voting power', async () => {
        await expect(client.joinDAO('dao-123')).resolves.toBeDefined();
      });

      it('should accept valid daoId with custom voting power', async () => {
        await expect(client.joinDAO('dao-123', 5)).resolves.toBeDefined();
      });

      it('should reject empty daoId', async () => {
        await expect(client.joinDAO('')).rejects.toThrow(MycelixError);
      });

      it('should reject negative voting power', async () => {
        await expect(client.joinDAO('dao-123', -1)).rejects.toThrow(MycelixError);
      });
    });

    describe('getMembers', () => {
      it('should accept valid daoId', async () => {
        await expect(client.getMembers('dao-123')).resolves.toBeDefined();
      });

      it('should reject empty daoId', async () => {
        await expect(client.getMembers('')).rejects.toThrow(MycelixError);
      });
    });

    describe('getMember', () => {
      it('should accept valid inputs', async () => {
        await expect(client.getMember('dao-123', 'did:example:member')).resolves.toBeDefined();
      });

      it('should reject empty daoId', async () => {
        await expect(client.getMember('', 'did:example:member')).rejects.toThrow(MycelixError);
      });

      it('should reject invalid DID', async () => {
        await expect(client.getMember('dao-123', 'invalid')).rejects.toThrow(MycelixError);
      });
    });

    describe('updateMemberRole', () => {
      it('should accept valid inputs with Member role', async () => {
        await expect(
          client.updateMemberRole('dao-123', 'did:example:member', 'Member')
        ).resolves.toBeDefined();
      });

      it('should accept Delegate role', async () => {
        await expect(
          client.updateMemberRole('dao-123', 'did:example:member', 'Delegate')
        ).resolves.toBeDefined();
      });

      it('should accept Steward role', async () => {
        await expect(
          client.updateMemberRole('dao-123', 'did:example:member', 'Steward')
        ).resolves.toBeDefined();
      });

      it('should accept Admin role', async () => {
        await expect(
          client.updateMemberRole('dao-123', 'did:example:member', 'Admin')
        ).resolves.toBeDefined();
      });

      it('should reject empty daoId', async () => {
        await expect(
          client.updateMemberRole('', 'did:example:member', 'Member')
        ).rejects.toThrow(MycelixError);
      });

      it('should reject invalid DID', async () => {
        await expect(
          client.updateMemberRole('dao-123', 'invalid', 'Member')
        ).rejects.toThrow(MycelixError);
      });

      it('should reject invalid role', async () => {
        await expect(
          client.updateMemberRole('dao-123', 'did:example:member', 'Invalid' as any)
        ).rejects.toThrow(MycelixError);
      });
    });
  });

  describe('ValidatedDelegationClient', () => {
    let client: ValidatedDelegationClient;

    beforeEach(() => {
      client = new ValidatedDelegationClient(mockZomeClient);
    });

    describe('delegate', () => {
      const validInput = {
        delegate: 'did:example:delegate',
        dao_id: 'dao-123',
        power: 1,
      };

      it('should accept valid input', async () => {
        await expect(client.delegate(validInput)).resolves.toBeDefined();
      });

      it('should reject non-DID delegate', async () => {
        await expect(
          client.delegate({ ...validInput, delegate: 'invalid' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject empty dao_id', async () => {
        await expect(
          client.delegate({ ...validInput, dao_id: '' })
        ).rejects.toThrow(MycelixError);
      });

      it('should reject negative power', async () => {
        await expect(
          client.delegate({ ...validInput, power: -1 })
        ).rejects.toThrow(MycelixError);
      });

      it('should accept optional categories', async () => {
        await expect(
          client.delegate({ ...validInput, categories: ['governance', 'treasury'] })
        ).resolves.toBeDefined();
      });

      it('should accept optional expires_at', async () => {
        await expect(
          client.delegate({ ...validInput, expires_at: Date.now() + 86400000 })
        ).resolves.toBeDefined();
      });

      it('should reject non-positive expires_at', async () => {
        await expect(
          client.delegate({ ...validInput, expires_at: 0 })
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('revokeDelegation', () => {
      it('should accept valid delegationId', async () => {
        await expect(client.revokeDelegation('delegation-123')).resolves.toBeUndefined();
      });

      it('should reject empty delegationId', async () => {
        await expect(client.revokeDelegation('')).rejects.toThrow(MycelixError);
      });
    });

    describe('getDelegationsFrom', () => {
      it('should accept valid DID', async () => {
        await expect(client.getDelegationsFrom('did:example:delegator')).resolves.toBeDefined();
      });

      it('should reject non-DID string', async () => {
        await expect(client.getDelegationsFrom('invalid')).rejects.toThrow(MycelixError);
      });
    });

    describe('getDelegationsTo', () => {
      it('should accept valid DID', async () => {
        await expect(client.getDelegationsTo('did:example:delegate')).resolves.toBeDefined();
      });

      it('should reject non-DID string', async () => {
        await expect(client.getDelegationsTo('invalid')).rejects.toThrow(MycelixError);
      });
    });

    describe('getEffectiveVotingPower', () => {
      it('should accept valid inputs', async () => {
        await expect(
          client.getEffectiveVotingPower('did:example:voter', 'dao-123')
        ).resolves.toBeDefined();
      });

      it('should reject invalid DID', async () => {
        await expect(
          client.getEffectiveVotingPower('invalid', 'dao-123')
        ).rejects.toThrow(MycelixError);
      });

      it('should reject empty daoId', async () => {
        await expect(
          client.getEffectiveVotingPower('did:example:voter', '')
        ).rejects.toThrow(MycelixError);
      });
    });
  });

  describe('ValidatedExecutionClient', () => {
    let client: ValidatedExecutionClient;

    beforeEach(() => {
      client = new ValidatedExecutionClient(mockZomeClient);
    });

    describe('requestExecution', () => {
      it('should accept valid inputs', async () => {
        await expect(
          client.requestExecution('proposal-123', 'target-happ')
        ).resolves.toBeDefined();
      });

      it('should reject empty proposalId', async () => {
        await expect(
          client.requestExecution('', 'target-happ')
        ).rejects.toThrow(MycelixError);
      });

      it('should reject empty targetHapp', async () => {
        await expect(
          client.requestExecution('proposal-123', '')
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('getPendingExecutions', () => {
      it('should accept no arguments', async () => {
        await expect(client.getPendingExecutions()).resolves.toBeDefined();
      });

      it('should accept valid targetHapp', async () => {
        await expect(client.getPendingExecutions('target-happ')).resolves.toBeDefined();
      });

      it('should reject empty targetHapp', async () => {
        await expect(client.getPendingExecutions('')).rejects.toThrow(MycelixError);
      });
    });

    describe('acknowledgeExecution', () => {
      it('should accept valid inputs with success', async () => {
        await expect(
          client.acknowledgeExecution('execution-123', true)
        ).resolves.toBeDefined();
      });

      it('should accept valid inputs with failure', async () => {
        await expect(
          client.acknowledgeExecution('execution-123', false)
        ).resolves.toBeDefined();
      });

      it('should accept optional error message', async () => {
        await expect(
          client.acknowledgeExecution('execution-123', false, 'Error message')
        ).resolves.toBeDefined();
      });

      it('should reject empty executionId', async () => {
        await expect(
          client.acknowledgeExecution('', true)
        ).rejects.toThrow(MycelixError);
      });

      it('should reject non-boolean success', async () => {
        await expect(
          client.acknowledgeExecution('execution-123', 'true' as any)
        ).rejects.toThrow(MycelixError);
      });
    });

    describe('getExecutionsForProposal', () => {
      it('should accept valid proposalId', async () => {
        await expect(client.getExecutionsForProposal('proposal-123')).resolves.toBeDefined();
      });

      it('should reject empty proposalId', async () => {
        await expect(client.getExecutionsForProposal('')).rejects.toThrow(MycelixError);
      });
    });
  });

  describe('createValidatedGovernanceClients', () => {
    it('should create all validated clients', () => {
      const clients = createValidatedGovernanceClients(mockZomeClient);

      expect(clients.proposals).toBeInstanceOf(ValidatedProposalsClient);
      expect(clients.voting).toBeInstanceOf(ValidatedVotingClient);
      expect(clients.dao).toBeInstanceOf(ValidatedDAOClient);
      expect(clients.delegation).toBeInstanceOf(ValidatedDelegationClient);
      expect(clients.execution).toBeInstanceOf(ValidatedExecutionClient);
    });
  });

  describe('Error message formatting', () => {
    let client: ValidatedProposalsClient;

    beforeEach(() => {
      client = new ValidatedProposalsClient(mockZomeClient);
    });

    it('should include field path in error message', async () => {
      try {
        await client.createProposal({
          dao_id: '',
          title: 'Test',
          description: 'Test',
          proposal_type: 'Standard',
          voting_period_hours: 24,
          quorum: 0.5,
        });
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(MycelixError);
        expect((error as MycelixError).message).toContain('dao_id');
      }
    });

    it('should include context in error message', async () => {
      try {
        await client.createProposal({
          dao_id: 'dao-123',
          title: '',
          description: 'Test',
          proposal_type: 'Standard',
          voting_period_hours: 24,
          quorum: 0.5,
        });
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(MycelixError);
        expect((error as MycelixError).message).toContain('createProposal');
      }
    });

    it('should have INVALID_ARGUMENT error code', async () => {
      try {
        await client.getProposal('');
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(MycelixError);
        expect((error as MycelixError).code).toBe(ErrorCode.INVALID_ARGUMENT);
      }
    });
  });
});
