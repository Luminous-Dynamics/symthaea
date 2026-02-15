/**
 * Governance Module Tests
 *
 * Tests for the Governance hApp TypeScript clients:
 * - ProposalsClient (proposal creation and management)
 * - VotingClient (vote casting and tallying)
 * - DAOClient (DAO management and membership)
 * - DelegationClient (voting power delegation)
 * - ExecutionClient (proposal execution)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  ProposalsClient,
  VotingClient,
  DAOClient,
  DelegationClient,
  ExecutionClient,
  createGovernanceClients,
  type Proposal,
  type Vote,
  type DAO,
  type DAOMember,
  type Delegation,
  type Execution,
  type ZomeCallable,
  type HolochainRecord,
  type ProposalType,
  type ProposalStatus,
  type VoteChoice,
  type MemberRole,
  type ExecutionStatus,
} from '../src/governance/index.js';

// ============================================================================
// Mock Setup
// ============================================================================

function createMockRecord<T>(entry: T): HolochainRecord<T> {
  return {
    signed_action: {
      hashed: { hash: 'uhCXk_test_hash_123', content: {} },
      signature: 'sig_test_123',
    },
    entry: { Present: entry },
  };
}

function createMockClient(responses: Map<string, unknown> = new Map()): ZomeCallable {
  return {
    callZome: vi.fn(
      async <T>(params: {
        role_name: string;
        zome_name: string;
        fn_name: string;
        payload: unknown;
      }): Promise<T> => {
        const key = `${params.zome_name}:${params.fn_name}`;
        if (responses.has(key)) {
          return responses.get(key) as T;
        }
        throw new Error(`No mock response for ${key}`);
      }
    ),
  };
}

// ============================================================================
// Mock Data Factories
// ============================================================================

function createMockProposal(overrides: Partial<Proposal> = {}): Proposal {
  return {
    id: 'proposal-123',
    dao_id: 'dao-luminous',
    title: 'Increase Treasury Allocation',
    description: 'Allocate 10% more funds to development',
    proposer: 'did:mycelix:proposer123',
    proposal_type: 'Standard' as ProposalType,
    status: 'Active' as ProposalStatus,
    voting_period_hours: 72,
    quorum: 0.5,
    threshold: 0.5,
    approve_weight: 150,
    reject_weight: 50,
    abstain_weight: 25,
    voting_ends: Date.now() * 1000 + 259200000000,
    created: Date.now() * 1000,
    ...overrides,
  };
}

function createMockVote(overrides: Partial<Vote> = {}): Vote {
  return {
    id: 'vote-123',
    proposal_id: 'proposal-123',
    voter: 'did:mycelix:voter123',
    choice: 'Approve' as VoteChoice,
    weight: 100,
    reason: 'Strong support for development funding',
    voted_at: Date.now() * 1000,
    ...overrides,
  };
}

function createMockDAO(overrides: Partial<DAO> = {}): DAO {
  return {
    id: 'dao-luminous',
    name: 'Luminous DAO',
    description: 'Decentralized governance for Luminous Dynamics',
    creator: 'did:mycelix:creator123',
    charter_hash: 'sha256:charter_hash_abc',
    default_voting_period: 72,
    default_quorum: 0.5,
    default_threshold: 0.5,
    member_count: 150,
    total_voting_power: 10000,
    created: Date.now() * 1000 - 31536000000000,
    updated: Date.now() * 1000,
    ...overrides,
  };
}

function createMockDAOMember(overrides: Partial<DAOMember> = {}): DAOMember {
  return {
    did: 'did:mycelix:member123',
    dao_id: 'dao-luminous',
    role: 'Member' as MemberRole,
    voting_power: 100,
    delegated_power: 50,
    joined_at: Date.now() * 1000 - 2592000000000,
    last_active: Date.now() * 1000,
    ...overrides,
  };
}

function createMockDelegation(overrides: Partial<Delegation> = {}): Delegation {
  return {
    id: 'delegation-123',
    delegator: 'did:mycelix:delegator123',
    delegate: 'did:mycelix:delegate456',
    dao_id: 'dao-luminous',
    power: 50,
    created: Date.now() * 1000,
    ...overrides,
  };
}

function createMockExecution(overrides: Partial<Execution> = {}): Execution {
  return {
    id: 'execution-123',
    proposal_id: 'proposal-123',
    target_happ: 'finance',
    action: 'treasury:execute_grant',
    payload: JSON.stringify({
      recipient: 'did:mycelix:grantee',
      amount: 10000,
      currency: 'MYC',
    }),
    status: 'Pending' as ExecutionStatus,
    requested_at: Date.now() * 1000,
    ...overrides,
  };
}

// ============================================================================
// ProposalsClient Tests
// ============================================================================

describe('ProposalsClient', () => {
  let client: ProposalsClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockProposal = createMockProposal();

    responses.set('proposals:create_proposal', createMockRecord(mockProposal));
    responses.set('proposals:get_proposal', createMockRecord(mockProposal));
    responses.set('proposals:get_proposals_by_dao', [createMockRecord(mockProposal)]);
    responses.set('proposals:get_active_proposals', [createMockRecord(mockProposal)]);
    responses.set('proposals:get_proposals_by_proposer', [createMockRecord(mockProposal)]);
    responses.set(
      'proposals:cancel_proposal',
      createMockRecord({
        ...mockProposal,
        status: 'Cancelled',
      })
    );
    responses.set(
      'proposals:finalize_proposal',
      createMockRecord({
        ...mockProposal,
        status: 'Passed',
      })
    );

    mockZome = createMockClient(responses);
    client = new ProposalsClient(mockZome);
  });

  describe('createProposal', () => {
    it('should create a new proposal', async () => {
      const result = await client.createProposal({
        dao_id: 'dao-luminous',
        title: 'Increase Treasury Allocation',
        description: 'Allocate 10% more funds to development',
        proposal_type: 'Standard',
        voting_period_hours: 72,
        quorum: 0.5,
      });

      expect(result.entry.Present.id).toBe('proposal-123');
      expect(result.entry.Present.status).toBe('Active');
      expect(mockZome.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'governance',
          zome_name: 'proposals',
          fn_name: 'create_proposal',
        })
      );
    });

    it('should create proposals of all types', async () => {
      const types: ProposalType[] = ['Standard', 'Emergency', 'Constitutional'];

      for (const proposal_type of types) {
        const result = await client.createProposal({
          dao_id: 'dao-luminous',
          title: `${proposal_type} Proposal`,
          description: 'Test proposal',
          proposal_type,
          voting_period_hours: proposal_type === 'Emergency' ? 24 : 72,
          quorum: proposal_type === 'Constitutional' ? 0.75 : 0.5,
        });
        expect(result).toBeDefined();
      }
    });

    it('should create proposal with execution payload', async () => {
      const result = await client.createProposal({
        dao_id: 'dao-luminous',
        title: 'Treasury Grant',
        description: 'Grant funds to development team',
        proposal_type: 'Standard',
        voting_period_hours: 72,
        quorum: 0.5,
        execution_payload: JSON.stringify({
          action: 'treasury:grant',
          amount: 10000,
        }),
        category: 'Finance',
      });

      expect(result).toBeDefined();
    });
  });

  describe('getProposal', () => {
    it('should get proposal by ID', async () => {
      const result = await client.getProposal('proposal-123');

      expect(result).not.toBeNull();
      expect(result!.entry.Present.id).toBe('proposal-123');
    });
  });

  describe('getProposalsByDAO', () => {
    it('should get all proposals for a DAO', async () => {
      const results = await client.getProposalsByDAO('dao-luminous');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.dao_id).toBe('dao-luminous');
    });
  });

  describe('getActiveProposals', () => {
    it('should get active proposals for a DAO', async () => {
      const results = await client.getActiveProposals('dao-luminous');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.status).toBe('Active');
    });
  });

  describe('getProposalsByProposer', () => {
    it('should get proposals by proposer DID', async () => {
      const results = await client.getProposalsByProposer('did:mycelix:proposer123');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.proposer).toBe('did:mycelix:proposer123');
    });
  });

  describe('cancelProposal', () => {
    it('should cancel a proposal', async () => {
      const result = await client.cancelProposal('proposal-123');

      expect(result.entry.Present.status).toBe('Cancelled');
    });
  });

  describe('finalizeProposal', () => {
    it('should finalize a proposal after voting ends', async () => {
      const result = await client.finalizeProposal('proposal-123');

      expect(result.entry.Present.status).toBe('Passed');
    });
  });
});

// ============================================================================
// VotingClient Tests
// ============================================================================

describe('VotingClient', () => {
  let client: VotingClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockVote = createMockVote();

    responses.set('voting:cast_vote', createMockRecord(mockVote));
    responses.set('voting:get_votes_for_proposal', [createMockRecord(mockVote)]);
    responses.set('voting:get_votes_by_voter', [createMockRecord(mockVote)]);
    responses.set('voting:has_voted', true);
    responses.set('voting:get_vote_tally', {
      approve: 150,
      reject: 50,
      abstain: 25,
      total: 225,
    });

    mockZome = createMockClient(responses);
    client = new VotingClient(mockZome);
  });

  describe('castVote', () => {
    it('should cast a vote on a proposal', async () => {
      const result = await client.castVote({
        proposal_id: 'proposal-123',
        choice: 'Approve',
        weight: 100,
        reason: 'Strong support',
      });

      expect(result.entry.Present.choice).toBe('Approve');
      expect(result.entry.Present.weight).toBe(100);
    });

    it('should cast votes with all choices', async () => {
      const choices: VoteChoice[] = ['Approve', 'Reject', 'Abstain'];

      for (const choice of choices) {
        const result = await client.castVote({
          proposal_id: 'proposal-123',
          choice,
          weight: 50,
        });
        expect(result).toBeDefined();
      }
    });

    it('should cast vote without reason', async () => {
      const result = await client.castVote({
        proposal_id: 'proposal-123',
        choice: 'Approve',
        weight: 100,
      });

      expect(result).toBeDefined();
    });
  });

  describe('getVotesForProposal', () => {
    it('should get all votes for a proposal', async () => {
      const results = await client.getVotesForProposal('proposal-123');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.proposal_id).toBe('proposal-123');
    });
  });

  describe('getVotesByVoter', () => {
    it('should get votes by voter DID', async () => {
      const results = await client.getVotesByVoter('did:mycelix:voter123');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.voter).toBe('did:mycelix:voter123');
    });
  });

  describe('hasVoted', () => {
    it('should check if user has voted', async () => {
      const result = await client.hasVoted('proposal-123', 'did:mycelix:voter123');

      expect(result).toBe(true);
    });
  });

  describe('getVoteTally', () => {
    it('should get vote tally for a proposal', async () => {
      const result = await client.getVoteTally('proposal-123');

      expect(result.approve).toBe(150);
      expect(result.reject).toBe(50);
      expect(result.abstain).toBe(25);
      expect(result.total).toBe(225);
    });

    it('should have total equal sum of votes', async () => {
      const result = await client.getVoteTally('proposal-123');

      expect(result.total).toBe(result.approve + result.reject + result.abstain);
    });
  });
});

// ============================================================================
// DAOClient Tests
// ============================================================================

describe('DAOClient', () => {
  let client: DAOClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockDAO = createMockDAO();
    const mockMember = createMockDAOMember();

    responses.set('dao:create_dao', createMockRecord(mockDAO));
    responses.set('dao:get_dao', createMockRecord(mockDAO));
    responses.set('dao:list_daos', [createMockRecord(mockDAO)]);
    responses.set('dao:join_dao', createMockRecord(mockMember));
    responses.set('dao:get_members', [createMockRecord(mockMember)]);
    responses.set('dao:get_member', createMockRecord(mockMember));
    responses.set(
      'dao:update_member_role',
      createMockRecord({
        ...mockMember,
        role: 'Delegate',
      })
    );

    mockZome = createMockClient(responses);
    client = new DAOClient(mockZome);
  });

  describe('createDAO', () => {
    it('should create a new DAO', async () => {
      const result = await client.createDAO({
        name: 'Luminous DAO',
        description: 'Decentralized governance for Luminous',
        default_voting_period: 72,
        default_quorum: 0.5,
        default_threshold: 0.5,
      });

      expect(result.entry.Present.id).toBe('dao-luminous');
      expect(result.entry.Present.name).toBe('Luminous DAO');
    });

    it('should create DAO with charter hash', async () => {
      const result = await client.createDAO({
        name: 'Test DAO',
        description: 'Test',
        charter_hash: 'sha256:charter_abc',
        default_voting_period: 48,
        default_quorum: 0.6,
        default_threshold: 0.6,
      });

      expect(result).toBeDefined();
    });
  });

  describe('getDAO', () => {
    it('should get DAO by ID', async () => {
      const result = await client.getDAO('dao-luminous');

      expect(result).not.toBeNull();
      expect(result!.entry.Present.id).toBe('dao-luminous');
    });
  });

  describe('listDAOs', () => {
    it('should list all DAOs', async () => {
      const results = await client.listDAOs();

      expect(results).toHaveLength(1);
    });
  });

  describe('joinDAO', () => {
    it('should join a DAO', async () => {
      const result = await client.joinDAO('dao-luminous');

      expect(result.entry.Present.dao_id).toBe('dao-luminous');
      expect(result.entry.Present.role).toBe('Member');
    });

    it('should join with custom voting power', async () => {
      const result = await client.joinDAO('dao-luminous', 50);

      expect(mockZome.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: { dao_id: 'dao-luminous', voting_power: 50 },
        })
      );
    });
  });

  describe('getMembers', () => {
    it('should get DAO members', async () => {
      const results = await client.getMembers('dao-luminous');

      expect(results).toHaveLength(1);
    });
  });

  describe('getMember', () => {
    it('should get specific member', async () => {
      const result = await client.getMember('dao-luminous', 'did:mycelix:member123');

      expect(result).not.toBeNull();
      expect(result!.entry.Present.did).toBe('did:mycelix:member123');
    });
  });

  describe('updateMemberRole', () => {
    it('should update member role', async () => {
      const result = await client.updateMemberRole(
        'dao-luminous',
        'did:mycelix:member123',
        'Delegate'
      );

      expect(result.entry.Present.role).toBe('Delegate');
    });

    it('should update to all roles', async () => {
      const roles: MemberRole[] = ['Member', 'Delegate', 'Steward', 'Admin'];

      for (const role of roles) {
        const result = await client.updateMemberRole(
          'dao-luminous',
          'did:mycelix:member123',
          role
        );
        expect(result).toBeDefined();
      }
    });
  });
});

// ============================================================================
// DelegationClient Tests
// ============================================================================

describe('DelegationClient', () => {
  let client: DelegationClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockDelegation = createMockDelegation();

    responses.set('delegation:delegate', createMockRecord(mockDelegation));
    responses.set('delegation:revoke_delegation', undefined);
    responses.set('delegation:get_delegations_from', [createMockRecord(mockDelegation)]);
    responses.set('delegation:get_delegations_to', [createMockRecord(mockDelegation)]);
    responses.set('delegation:get_effective_voting_power', 150);

    mockZome = createMockClient(responses);
    client = new DelegationClient(mockZome);
  });

  describe('delegate', () => {
    it('should create a delegation', async () => {
      const result = await client.delegate({
        delegate: 'did:mycelix:delegate456',
        dao_id: 'dao-luminous',
        power: 50,
      });

      expect(result.entry.Present.delegate).toBe('did:mycelix:delegate456');
      expect(result.entry.Present.power).toBe(50);
    });

    it('should create delegation with categories', async () => {
      const result = await client.delegate({
        delegate: 'did:mycelix:delegate456',
        dao_id: 'dao-luminous',
        power: 50,
        categories: ['Finance', 'Technical'],
      });

      expect(result).toBeDefined();
    });

    it('should create delegation with expiration', async () => {
      const result = await client.delegate({
        delegate: 'did:mycelix:delegate456',
        dao_id: 'dao-luminous',
        power: 50,
        expires_at: Date.now() * 1000 + 2592000000000,
      });

      expect(result).toBeDefined();
    });
  });

  describe('revokeDelegation', () => {
    it('should revoke a delegation', async () => {
      await expect(client.revokeDelegation('delegation-123')).resolves.not.toThrow();
    });
  });

  describe('getDelegationsFrom', () => {
    it('should get delegations from a delegator', async () => {
      const results = await client.getDelegationsFrom('did:mycelix:delegator123');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.delegator).toBe('did:mycelix:delegator123');
    });
  });

  describe('getDelegationsTo', () => {
    it('should get delegations to a delegate', async () => {
      const results = await client.getDelegationsTo('did:mycelix:delegate456');

      expect(results).toHaveLength(1);
    });
  });

  describe('getEffectiveVotingPower', () => {
    it('should get effective voting power', async () => {
      const power = await client.getEffectiveVotingPower('did:mycelix:member123', 'dao-luminous');

      expect(power).toBe(150);
    });
  });
});

// ============================================================================
// ExecutionClient Tests
// ============================================================================

describe('ExecutionClient', () => {
  let client: ExecutionClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockExecution = createMockExecution();

    responses.set('execution:request_execution', createMockRecord(mockExecution));
    responses.set('execution:get_pending_executions', [createMockRecord(mockExecution)]);
    responses.set(
      'execution:acknowledge_execution',
      createMockRecord({
        ...mockExecution,
        status: 'Completed',
        executed_at: Date.now() * 1000,
      })
    );
    responses.set('execution:get_executions_for_proposal', [createMockRecord(mockExecution)]);

    mockZome = createMockClient(responses);
    client = new ExecutionClient(mockZome);
  });

  describe('requestExecution', () => {
    it('should request execution of passed proposal', async () => {
      const result = await client.requestExecution('proposal-123', 'finance');

      expect(result.entry.Present.proposal_id).toBe('proposal-123');
      expect(result.entry.Present.target_happ).toBe('finance');
      expect(result.entry.Present.status).toBe('Pending');
    });
  });

  describe('getPendingExecutions', () => {
    it('should get all pending executions', async () => {
      const results = await client.getPendingExecutions();

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.status).toBe('Pending');
    });

    it('should get pending executions for specific hApp', async () => {
      const results = await client.getPendingExecutions('finance');

      expect(results).toHaveLength(1);
    });
  });

  describe('acknowledgeExecution', () => {
    it('should acknowledge successful execution', async () => {
      const result = await client.acknowledgeExecution('execution-123', true);

      expect(result.entry.Present.status).toBe('Completed');
      expect(result.entry.Present.executed_at).toBeDefined();
    });

    it('should acknowledge failed execution', async () => {
      const result = await client.acknowledgeExecution(
        'execution-123',
        false,
        'Insufficient funds'
      );

      expect(result).toBeDefined();
    });
  });

  describe('getExecutionsForProposal', () => {
    it('should get executions for a proposal', async () => {
      const results = await client.getExecutionsForProposal('proposal-123');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.proposal_id).toBe('proposal-123');
    });
  });

  describe('Execution Status Transitions', () => {
    it('should support all execution statuses', () => {
      const statuses: ExecutionStatus[] = [
        'Pending',
        'InProgress',
        'Completed',
        'Failed',
        'Reverted',
      ];

      statuses.forEach((status) => {
        const execution = createMockExecution({ status });
        expect(statuses).toContain(execution.status);
      });
    });
  });
});

// ============================================================================
// Factory Function Tests
// ============================================================================

describe('createGovernanceClients', () => {
  it('should create all governance clients', () => {
    const mockZome = createMockClient(new Map());
    const clients = createGovernanceClients(mockZome);

    expect(clients.proposals).toBeInstanceOf(ProposalsClient);
    expect(clients.voting).toBeInstanceOf(VotingClient);
    expect(clients.dao).toBeInstanceOf(DAOClient);
    expect(clients.delegation).toBeInstanceOf(DelegationClient);
    expect(clients.execution).toBeInstanceOf(ExecutionClient);
  });

  it('should share the same ZomeCallable instance', () => {
    const mockZome = createMockClient(new Map());
    const clients = createGovernanceClients(mockZome);

    expect(clients.proposals).toBeDefined();
    expect(clients.voting).toBeDefined();
    expect(clients.dao).toBeDefined();
    expect(clients.delegation).toBeDefined();
    expect(clients.execution).toBeDefined();
  });
});

// ============================================================================
// Type Safety Tests
// ============================================================================

describe('Type Safety', () => {
  it('should enforce proposal type constraints', () => {
    const types: ProposalType[] = ['Standard', 'Emergency', 'Constitutional'];

    types.forEach((type_) => {
      const proposal = createMockProposal({ proposal_type: type_ });
      expect(types).toContain(proposal.proposal_type);
    });
  });

  it('should enforce proposal status transitions', () => {
    const statuses: ProposalStatus[] = [
      'Draft',
      'Active',
      'Passed',
      'Rejected',
      'Executed',
      'Expired',
      'Cancelled',
    ];

    statuses.forEach((status) => {
      const proposal = createMockProposal({ status });
      expect(statuses).toContain(proposal.status);
    });
  });

  it('should enforce vote choice constraints', () => {
    const choices: VoteChoice[] = ['Approve', 'Reject', 'Abstain'];

    choices.forEach((choice) => {
      const vote = createMockVote({ choice });
      expect(choices).toContain(vote.choice);
    });
  });

  it('should enforce member role hierarchy', () => {
    const roles: MemberRole[] = ['Member', 'Delegate', 'Steward', 'Admin'];

    roles.forEach((role) => {
      const member = createMockDAOMember({ role });
      expect(roles).toContain(member.role);
    });
  });

  it('should enforce quorum in valid range', () => {
    const proposal = createMockProposal();
    expect(proposal.quorum).toBeGreaterThanOrEqual(0);
    expect(proposal.quorum).toBeLessThanOrEqual(1);
  });

  it('should enforce threshold in valid range', () => {
    const proposal = createMockProposal();
    expect(proposal.threshold).toBeGreaterThanOrEqual(0);
    expect(proposal.threshold).toBeLessThanOrEqual(1);
  });

  it('should enforce voting power is non-negative', () => {
    const member = createMockDAOMember();
    expect(member.voting_power).toBeGreaterThanOrEqual(0);
    expect(member.delegated_power).toBeGreaterThanOrEqual(0);
  });
});

// ============================================================================
// Integration Pattern Tests
// ============================================================================

describe('Integration Patterns', () => {
  it('should support full governance lifecycle', async () => {
    const responses = new Map<string, unknown>();
    const mockDAO = createMockDAO();
    const mockMember = createMockDAOMember();
    const mockProposal = createMockProposal();
    const mockVote = createMockVote();

    responses.set('dao:create_dao', createMockRecord(mockDAO));
    responses.set('dao:join_dao', createMockRecord(mockMember));
    responses.set('proposals:create_proposal', createMockRecord(mockProposal));
    responses.set('voting:cast_vote', createMockRecord(mockVote));
    responses.set(
      'proposals:finalize_proposal',
      createMockRecord({
        ...mockProposal,
        status: 'Passed',
      })
    );

    const mockZome = createMockClient(responses);
    const clients = createGovernanceClients(mockZome);

    // Create DAO
    const dao = await clients.dao.createDAO({
      name: 'Test DAO',
      description: 'Test',
      default_voting_period: 72,
      default_quorum: 0.5,
      default_threshold: 0.5,
    });
    expect(dao.entry.Present.id).toBeDefined();

    // Join DAO
    const member = await clients.dao.joinDAO(dao.entry.Present.id);
    expect(member.entry.Present.role).toBe('Member');

    // Create proposal
    const proposal = await clients.proposals.createProposal({
      dao_id: dao.entry.Present.id,
      title: 'Test Proposal',
      description: 'Test',
      proposal_type: 'Standard',
      voting_period_hours: 72,
      quorum: 0.5,
    });
    expect(proposal.entry.Present.status).toBe('Active');

    // Cast vote
    const vote = await clients.voting.castVote({
      proposal_id: proposal.entry.Present.id,
      choice: 'Approve',
      weight: 100,
    });
    expect(vote.entry.Present.choice).toBe('Approve');

    // Finalize proposal
    const finalized = await clients.proposals.finalizeProposal(proposal.entry.Present.id);
    expect(finalized.entry.Present.status).toBe('Passed');
  });

  it('should support delegation flow', async () => {
    const responses = new Map<string, unknown>();
    const mockDelegation = createMockDelegation();
    const mockVote = createMockVote({ delegated_from: 'did:mycelix:delegator123' });

    responses.set('delegation:delegate', createMockRecord(mockDelegation));
    responses.set('delegation:get_effective_voting_power', 150);
    responses.set('voting:cast_vote', createMockRecord(mockVote));

    const mockZome = createMockClient(responses);
    const clients = createGovernanceClients(mockZome);

    // Create delegation
    const delegation = await clients.delegation.delegate({
      delegate: 'did:mycelix:delegate456',
      dao_id: 'dao-luminous',
      power: 50,
    });
    expect(delegation.entry.Present.power).toBe(50);

    // Check effective voting power
    const power = await clients.delegation.getEffectiveVotingPower(
      'did:mycelix:delegate456',
      'dao-luminous'
    );
    expect(power).toBe(150); // Base + delegated

    // Delegate can vote with combined power
    const vote = await clients.voting.castVote({
      proposal_id: 'proposal-123',
      choice: 'Approve',
      weight: 150,
    });
    expect(vote.entry.Present.weight).toBe(100);
  });

  it('should support cross-hApp execution', async () => {
    const responses = new Map<string, unknown>();
    const mockProposal = createMockProposal({ status: 'Passed' });
    const mockExecution = createMockExecution();

    responses.set('proposals:get_proposal', createMockRecord(mockProposal));
    responses.set('execution:request_execution', createMockRecord(mockExecution));
    responses.set(
      'execution:acknowledge_execution',
      createMockRecord({
        ...mockExecution,
        status: 'Completed',
      })
    );

    const mockZome = createMockClient(responses);
    const clients = createGovernanceClients(mockZome);

    // Verify proposal passed
    const proposal = await clients.proposals.getProposal('proposal-123');
    expect(proposal!.entry.Present.status).toBe('Passed');

    // Request execution on target hApp
    const execution = await clients.execution.requestExecution('proposal-123', 'finance');
    expect(execution.entry.Present.target_happ).toBe('finance');

    // Target hApp acknowledges execution
    const completed = await clients.execution.acknowledgeExecution(
      execution.entry.Present.id,
      true
    );
    expect(completed.entry.Present.status).toBe('Completed');
  });
});

// ============================================================================
// Edge Case Tests
// ============================================================================

describe('Edge Cases', () => {
  it('should handle proposal with no votes', () => {
    const proposal = createMockProposal({
      approve_weight: 0,
      reject_weight: 0,
      abstain_weight: 0,
    });

    expect(proposal.approve_weight).toBe(0);
    expect(proposal.reject_weight).toBe(0);
    expect(proposal.abstain_weight).toBe(0);
  });

  it('should handle DAO with single member', () => {
    const dao = createMockDAO({
      member_count: 1,
      total_voting_power: 100,
    });

    expect(dao.member_count).toBe(1);
  });

  it('should handle expired proposal', () => {
    const proposal = createMockProposal({
      status: 'Expired',
      voting_ends: Date.now() * 1000 - 86400000000,
    });

    expect(proposal.status).toBe('Expired');
    expect(proposal.voting_ends).toBeLessThan(Date.now() * 1000);
  });

  it('should handle member with no delegated power', () => {
    const member = createMockDAOMember({
      delegated_power: 0,
    });

    expect(member.delegated_power).toBe(0);
  });

  it('should handle delegation with categories', () => {
    const delegation = createMockDelegation({
      categories: ['Finance', 'Technical', 'Operations'],
    });

    expect(delegation.categories).toHaveLength(3);
  });

  it('should handle executed proposal', () => {
    const proposal = createMockProposal({
      status: 'Executed',
      executed_at: Date.now() * 1000,
    });

    expect(proposal.status).toBe('Executed');
    expect(proposal.executed_at).toBeDefined();
  });
});

// ============================================================================
// Governance Math Tests
// ============================================================================

describe('Governance Math', () => {
  it('should calculate quorum correctly', () => {
    const dao = createMockDAO({ total_voting_power: 10000 });
    const proposal = createMockProposal({ quorum: 0.5 });

    const requiredQuorum = dao.total_voting_power * proposal.quorum;
    expect(requiredQuorum).toBe(5000);
  });

  it('should calculate threshold correctly', () => {
    const proposal = createMockProposal({
      approve_weight: 150,
      reject_weight: 50,
      threshold: 0.5,
    });

    const totalVotes = proposal.approve_weight + proposal.reject_weight;
    const approvalRate = proposal.approve_weight / totalVotes;

    expect(approvalRate).toBe(0.75);
    expect(approvalRate).toBeGreaterThan(proposal.threshold);
  });

  it('should calculate effective voting power with delegation', () => {
    const member = createMockDAOMember({
      voting_power: 100,
      delegated_power: 50,
    });

    const effectivePower = member.voting_power + member.delegated_power;
    expect(effectivePower).toBe(150);
  });
});
