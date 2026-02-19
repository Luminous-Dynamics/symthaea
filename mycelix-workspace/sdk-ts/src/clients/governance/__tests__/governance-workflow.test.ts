/**
 * Governance Multi-Client Workflow Integration Tests
 *
 * Tests realistic end-to-end governance flows using multiple clients together:
 * 1. Create DAO → Create Proposal → Vote → Finalize → Execute → Release Funds
 * 2. Delegation-aware voting flow
 * 3. Guardian veto during timelock
 * 4. Treasury allocation workflow
 *
 * Uses mock callZome that tracks state across calls to simulate
 * realistic cross-client interactions.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { DAOClient } from '../dao';
import { ProposalsClient } from '../proposals';
import { VotingClient } from '../voting';
import { ExecutionClient } from '../execution';
import { TreasuryClient } from '../treasury';
import { DelegationClient } from '../delegation';
import type { AppClient } from '@holochain/client';

// ============================================================================
// STATEFUL MOCK — tracks IDs across client calls
// ============================================================================

interface MockState {
  daoCreated: boolean;
  memberJoined: boolean;
  proposalStatus: string;
  timelockStatus: string;
  fundStatus: string;
  votesCast: number;
  delegationActive: boolean;
  treasuryCreated: boolean;
  allocationStatus: string;
}

function createStatefulMock() {
  const state: MockState = {
    daoCreated: false,
    memberJoined: false,
    proposalStatus: 'Draft',
    timelockStatus: 'Pending',
    fundStatus: 'Locked',
    votesCast: 0,
    delegationActive: false,
    treasuryCreated: false,
    allocationStatus: 'Proposed',
  };

  const mockCallZome = vi.fn().mockImplementation((args: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }) => {
    const { zome_name, fn_name, payload } = args;

    // DAO operations
    if (zome_name === 'dao') {
      if (fn_name === 'create_dao') {
        state.daoCreated = true;
        return Promise.resolve(mockRecord({
          name: (payload as any).name,
          description: (payload as any).description,
          charter: (payload as any).charter,
          founder_did: 'did:mycelix:alice',
          default_voting_period_hours: (payload as any).default_voting_period_hours,
          default_quorum: (payload as any).default_quorum,
          default_threshold: (payload as any).default_threshold,
          member_count: 1,
          total_voting_power: 1.0,
          active: true,
          created_at: 1708200000,
          updated_at: 1708200000,
        }));
      }
      if (fn_name === 'join_dao') {
        state.memberJoined = true;
        return Promise.resolve(mockRecord({
          dao_id: (payload as any).dao_id,
          member_did: 'did:mycelix:bob',
          role: 'Member',
          voting_power: 1.0,
          reputation_score: 0.8,
          active: true,
          joined_at: 1708200000,
        }));
      }
      if (fn_name === 'get_dao_stats') {
        return Promise.resolve({
          member_count: state.memberJoined ? 3 : 1,
          active_proposals: state.proposalStatus === 'Active' ? 1 : 0,
          total_voting_power: state.memberJoined ? 3.0 : 1.0,
          total_proposals: 1,
          passed_proposals: state.proposalStatus === 'Passed' ? 1 : 0,
        });
      }
    }

    // Proposal operations
    if (zome_name === 'proposals') {
      if (fn_name === 'create_proposal') {
        return Promise.resolve(mockRecord({
          id: 'proposal-001',
          dao_id: (payload as any).dao_id,
          title: (payload as any).title,
          description: (payload as any).description,
          proposal_type: (payload as any).proposal_type,
          proposer_did: 'did:mycelix:alice',
          status: 'Draft',
          quorum: (payload as any).quorum,
          threshold: (payload as any).threshold,
          voting_period_hours: (payload as any).voting_period_hours,
          execution_payload: (payload as any).execution_payload,
          created_at: 1708200000,
        }));
      }
      if (fn_name === 'activate_proposal') {
        state.proposalStatus = 'Active';
        return Promise.resolve(mockRecord({
          id: 'proposal-001',
          status: 'Active',
          voting_starts: 1708200000,
          voting_ends: 1708804800,
        }));
      }
      if (fn_name === 'finalize_proposal') {
        state.proposalStatus = state.votesCast >= 2 ? 'Passed' : 'Rejected';
        return Promise.resolve({
          proposal_id: 'proposal-001',
          final_status: state.proposalStatus,
          total_votes: state.votesCast,
          approve_weight: state.votesCast * 1.0,
          reject_weight: 0,
          abstain_weight: 0,
          quorum_met: state.votesCast >= 2,
          approval_percentage: 100.0,
          participation_rate: state.votesCast / 3.0,
          passed: state.votesCast >= 2,
        });
      }
      if (fn_name === 'mark_executed') {
        state.proposalStatus = 'Executed';
        return Promise.resolve(mockRecord({
          id: 'proposal-001',
          status: 'Executed',
        }));
      }
    }

    // Voting operations
    if (zome_name === 'voting') {
      if (fn_name === 'cast_vote') {
        state.votesCast++;
        return Promise.resolve(mockRecord({
          id: `vote-${state.votesCast}`,
          proposal_id: (payload as any).proposal_id,
          voter_did: `did:mycelix:voter${state.votesCast}`,
          choice: (payload as any).choice,
          weight: (payload as any).weight,
          reason: (payload as any).reason,
          delegated_from: (payload as any).delegated_from,
          cast_at: 1708200000,
        }));
      }
      if (fn_name === 'get_proposal_votes') {
        const votes = [];
        for (let i = 1; i <= state.votesCast; i++) {
          votes.push(mockRecord({
            id: `vote-${i}`,
            proposal_id: payload,
            choice: 'Approve',
            weight: 1.0,
          }));
        }
        return Promise.resolve(votes);
      }
      if (fn_name === 'get_voting_stats') {
        return Promise.resolve({
          total_votes: state.votesCast,
          approve_count: state.votesCast,
          reject_count: 0,
          abstain_count: 0,
          total_weight: state.votesCast * 1.0,
          approve_weight: state.votesCast * 1.0,
          reject_weight: 0,
          abstain_weight: 0,
          participation_rate: state.votesCast / 3.0,
          quorum_met: state.votesCast >= 2,
          approve_percentage: 100.0,
        });
      }
      if (fn_name === 'cast_delegated_vote') {
        state.votesCast++;
        return Promise.resolve(mockRecord({
          id: `vote-${state.votesCast}`,
          proposal_id: (payload as any).proposal_id,
          voter_did: 'did:mycelix:bob',
          choice: (payload as any).choice,
          weight: 1.0,
          delegated_from: (payload as any).delegator_did,
          cast_at: 1708200000,
        }));
      }
    }

    // Delegation operations
    if (zome_name === 'delegation') {
      if (fn_name === 'create_delegation') {
        state.delegationActive = true;
        return Promise.resolve(mockRecord({
          id: 'del-001',
          delegator_did: 'did:mycelix:carol',
          delegate_did: (payload as any).delegate_did,
          dao_id: (payload as any).dao_id,
          scope: (payload as any).scope,
          power_percentage: (payload as any).power_percentage ?? 1.0,
          active: true,
          created_at: 1708200000,
        }));
      }
      if (fn_name === 'list_delegations') {
        if (state.delegationActive) {
          return Promise.resolve([mockRecord({
            id: 'del-001',
            delegator_did: 'did:mycelix:carol',
            delegate_did: (payload as any).delegate_did ?? 'did:mycelix:bob',
            dao_id: (payload as any).dao_id,
            scope: 'All',
            power_percentage: 1.0,
            active: true,
            created_at: 1708200000,
          })]);
        }
        return Promise.resolve([]);
      }
    }

    // Execution operations (uses nested record: { entry: { Present: { entry } } })
    if (zome_name === 'execution') {
      if (fn_name === 'create_timelock') {
        return Promise.resolve(mockNestedRecord({
          id: 'timelock-001',
          proposal_id: (payload as any).proposal_id,
          actions: (payload as any).actions,
          started: 1708200000,
          expires: 1708373600,
          status: 'Pending',
        }));
      }
      if (fn_name === 'mark_timelock_ready') {
        state.timelockStatus = 'Ready';
        return Promise.resolve(mockNestedRecord({
          id: 'timelock-001',
          status: 'Ready',
          started: 1708200000,
          expires: 1708373600,
        }));
      }
      if (fn_name === 'execute_timelock') {
        state.timelockStatus = 'Executed';
        return Promise.resolve(mockNestedRecord({
          id: 'timelock-001',
          status: 'Executed',
        }));
      }
      if (fn_name === 'veto_timelock') {
        state.timelockStatus = 'Cancelled';
        return Promise.resolve(mockNestedRecord({
          id: 'timelock-001',
          status: 'Cancelled',
          cancellation_reason: (payload as any).reason,
        }));
      }
      if (fn_name === 'lock_proposal_funds') {
        state.fundStatus = 'Locked';
        return Promise.resolve(mockNestedRecord({
          id: 'fund-001',
          proposal_id: (payload as any).proposal_id,
          timelock_id: (payload as any).timelock_id,
          source_account: (payload as any).source_account,
          amount: (payload as any).amount,
          currency: (payload as any).currency,
          locked_at: 1708200000,
          status: 'Locked',
        }));
      }
      if (fn_name === 'release_locked_funds') {
        state.fundStatus = 'Released';
        return Promise.resolve(mockNestedRecord({
          id: 'fund-001',
          proposal_id: (payload as any).proposal_id,
          status: 'Released',
        }));
      }
      if (fn_name === 'refund_locked_funds') {
        state.fundStatus = 'Refunded';
        return Promise.resolve(mockNestedRecord({
          id: 'fund-001',
          proposal_id: (payload as any).proposal_id,
          status: 'Refunded',
        }));
      }
    }

    // Treasury operations
    if (zome_name === 'treasury') {
      if (fn_name === 'create_treasury') {
        state.treasuryCreated = true;
        return Promise.resolve(mockRecord({
          id: 'treasury-001',
          dao_id: (payload as any).dao_id,
          name: (payload as any).name,
          governance_threshold: (payload as any).governance_threshold,
          discretionary_limit: (payload as any).discretionary_limit,
          multi_sig_threshold: (payload as any).multi_sig_threshold,
          multi_sig_signers: (payload as any).multi_sig_signers,
          total_balance: 0,
          created_at: 1708200000,
        }));
      }
      if (fn_name === 'propose_allocation') {
        return Promise.resolve(mockRecord({
          id: 'alloc-001',
          treasury_id: (payload as any).treasury_id,
          purpose: (payload as any).purpose,
          amount: (payload as any).amount,
          currency: (payload as any).currency,
          recipient_did: (payload as any).recipient_did,
          status: 'Proposed',
          proposed_at: 1708200000,
        }));
      }
      if (fn_name === 'approve_allocation') {
        state.allocationStatus = 'Approved';
        return Promise.resolve(mockRecord({
          id: 'alloc-001',
          status: 'Approved',
        }));
      }
      if (fn_name === 'execute_allocation') {
        state.allocationStatus = 'Executed';
        return Promise.resolve(mockRecord({
          id: 'alloc-001',
          status: 'Executed',
        }));
      }
      if (fn_name === 'get_balance') {
        return Promise.resolve({
          treasury_id: (payload as any).treasury_id ?? payload,
          currency: 'MYC',
          available: state.fundStatus === 'Released' ? 95000 : 100000,
          locked: state.fundStatus === 'Locked' ? 5000 : 0,
          total: 100000,
        });
      }
    }

    // Default fallback
    return Promise.resolve(null);
  });

  const client = { callZome: mockCallZome } as unknown as AppClient;
  return { client, state, mockCallZome };
}

/** Flat mock: dao, proposals, voting, delegation, treasury, bridge */
function mockRecord(entry: Record<string, unknown>) {
  return {
    entry: { Present: entry },
    signed_action: { hashed: { hash: new Uint8Array(32) } },
  };
}

/** Nested mock: execution, councils, constitution, threshold-signing */
function mockNestedRecord(entry: Record<string, unknown>) {
  return {
    entry: { Present: { entry } },
    signed_action: { hashed: { hash: new Uint8Array(32) } },
  };
}

// ============================================================================
// WORKFLOW 1: Complete Governance Loop
// Create DAO → Create Proposal → Vote → Finalize → Timelock → Execute → Release
// ============================================================================

describe('Complete Governance Workflow', () => {
  let dao: DAOClient;
  let proposals: ProposalsClient;
  let voting: VotingClient;
  let execution: ExecutionClient;
  let treasury: TreasuryClient;
  let state: MockState;
  let mockCallZome: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    const mock = createStatefulMock();
    dao = new DAOClient(mock.client);
    proposals = new ProposalsClient(mock.client);
    voting = new VotingClient(mock.client);
    execution = new ExecutionClient(mock.client);
    treasury = new TreasuryClient(mock.client);
    state = mock.state;
    mockCallZome = mock.mockCallZome;
  });

  it('end-to-end: create DAO → proposal → vote → finalize → execute → release funds', async () => {
    // Step 1: Create DAO
    const createdDao = await dao.createDAO({
      name: 'Richardson Community DAO',
      description: 'Governance for local projects',
      charter: 'Decisions require 2/3 majority with 33% quorum',
      defaultVotingPeriodHours: 168,
      defaultQuorum: 0.33,
      defaultThreshold: 0.66,
    });
    expect(createdDao.name).toBe('Richardson Community DAO');
    expect(createdDao.active).toBe(true);
    expect(state.daoCreated).toBe(true);

    // Step 2: Members join
    const membership = await dao.joinDAO({ daoId: 'dao-001' });
    expect(membership.role).toBe('Member');
    expect(state.memberJoined).toBe(true);

    // Step 3: Create treasury
    const createdTreasury = await treasury.createTreasury({
      daoId: 'dao-001',
      name: 'Community Fund',
      description: 'Main treasury',
      governanceThreshold: 0.66,
      discretionaryLimit: 1000,
      multiSigThreshold: 2,
      multiSigSigners: ['did:mycelix:alice', 'did:mycelix:bob', 'did:mycelix:carol'],
    });
    expect(state.treasuryCreated).toBe(true);

    // Step 4: Create proposal
    const proposal = await proposals.createProposal({
      daoId: 'dao-001',
      title: 'Fund Community Garden',
      description: 'Allocate 5000 MYC for garden infrastructure',
      proposalType: 'Funding',
      quorum: 0.33,
      threshold: 0.66,
      votingPeriodHours: 168,
      executionPayload: JSON.stringify({ action: 'transfer', amount: 5000, currency: 'MYC' }),
    });
    expect(proposal.status).toBe('Draft');

    // Step 5: Activate proposal (start voting)
    const activated = await proposals.activateProposal('proposal-001');
    expect(activated.status).toBe('Active');
    expect(state.proposalStatus).toBe('Active');

    // Step 6: Cast votes (3 members approve)
    const vote1 = await voting.castVote({
      proposalId: 'proposal-001',
      choice: 'Approve',
      weight: 1.0,
      reason: 'Strong community benefit',
    });
    expect(vote1.choice).toBe('Approve');

    const vote2 = await voting.castVote({
      proposalId: 'proposal-001',
      choice: 'Approve',
      weight: 1.0,
      reason: 'Aligns with charter',
    });

    const vote3 = await voting.castVote({
      proposalId: 'proposal-001',
      choice: 'Approve',
      weight: 1.0,
    });
    expect(state.votesCast).toBe(3);

    // Step 7: Finalize proposal
    const result = await proposals.finalizeProposal('proposal-001');
    expect(result.passed).toBe(true);
    expect(result.quorumMet).toBe(true);
    expect(result.totalVotes).toBe(3);
    expect(result.approvalPercentage).toBe(100.0);

    // Step 8: Create timelock
    const timelock = await execution.createTimelock({
      proposalId: 'proposal-001',
      actions: JSON.stringify({ action: 'transfer', amount: 5000 }),
      delayHours: 48,
    });
    expect(timelock.status).toBe('Pending');

    // Step 9: Lock funds
    const funds = await execution.lockFunds(
      'proposal-001',
      'timelock-001',
      'treasury-001',
      5000,
      'MYC'
    );
    expect(funds.status).toBe('Locked');
    expect(funds.amount).toBe(5000);

    // Step 10: Mark timelock ready (delay expired)
    const ready = await execution.markTimelockReady('timelock-001');
    expect(ready.status).toBe('Ready');
    expect(state.timelockStatus).toBe('Ready');

    // Step 11: Execute timelock
    await execution.executeTimelock({ timelockId: 'timelock-001' });
    expect(state.timelockStatus).toBe('Executed');

    // Step 12: Release locked funds
    const released = await execution.releaseFunds('proposal-001');
    expect(released.status).toBe('Released');
    expect(state.fundStatus).toBe('Released');

    // Step 13: Mark proposal executed
    const executed = await proposals.markExecuted('proposal-001');
    expect(executed.status).toBe('Executed');

    // Verify full pipeline: 13 zome calls in correct order
    const callOrder = mockCallZome.mock.calls.map(
      (c: any[]) => `${c[0].zome_name}.${c[0].fn_name}`
    );
    expect(callOrder).toEqual([
      'dao.create_dao',
      'dao.join_dao',
      'treasury.create_treasury',
      'proposals.create_proposal',
      'proposals.activate_proposal',
      'voting.cast_vote',
      'voting.cast_vote',
      'voting.cast_vote',
      'proposals.finalize_proposal',
      'execution.create_timelock',
      'execution.lock_proposal_funds',
      'execution.mark_timelock_ready',
      'execution.execute_timelock',
      'execution.release_locked_funds',
      'proposals.mark_executed',
    ]);
  });
});

// ============================================================================
// WORKFLOW 2: Delegation-Aware Voting
// Delegate → Vote on behalf → Finalize with delegated weight
// ============================================================================

describe('Delegation-Aware Voting Workflow', () => {
  let dao: DAOClient;
  let proposals: ProposalsClient;
  let voting: VotingClient;
  let delegation: DelegationClient;
  let state: MockState;

  beforeEach(() => {
    const mock = createStatefulMock();
    dao = new DAOClient(mock.client);
    proposals = new ProposalsClient(mock.client);
    voting = new VotingClient(mock.client);
    delegation = new DelegationClient(mock.client);
    state = mock.state;
  });

  it('delegate votes → cast on behalf → finalize counts delegated weight', async () => {
    // Carol delegates to Bob
    const del = await delegation.createDelegation({
      daoId: 'dao-001',
      delegateDid: 'did:mycelix:bob',
      scope: 'All',
    });
    expect(del.active).toBe(true);
    expect(state.delegationActive).toBe(true);

    // Check Bob's delegations
    const delegationsTo = await delegation.getDelegationsTo('dao-001', 'did:mycelix:bob');
    expect(delegationsTo).toHaveLength(1);
    expect(delegationsTo[0].delegatorDid).toBe('did:mycelix:carol');

    // Alice votes directly
    await voting.castVote({
      proposalId: 'proposal-001',
      choice: 'Approve',
      weight: 1.0,
      reason: 'Direct vote',
    });

    // Bob votes on behalf of Carol (delegated)
    const delegatedVote = await voting.castDelegatedVote(
      'proposal-001',
      'did:mycelix:carol',
      'Approve',
      'Voting as delegate'
    );
    expect(delegatedVote.delegatedFrom).toBe('did:mycelix:carol');
    expect(state.votesCast).toBe(2);

    // Finalize — should pass with 2 votes meeting quorum
    const result = await proposals.finalizeProposal('proposal-001');
    expect(result.passed).toBe(true);
    expect(result.totalVotes).toBe(2);
  });
});

// ============================================================================
// WORKFLOW 3: Guardian Veto During Timelock
// Pass proposal → Create timelock → Guardian veto → Refund funds
// ============================================================================

describe('Guardian Veto Workflow', () => {
  let proposals: ProposalsClient;
  let voting: VotingClient;
  let execution: ExecutionClient;
  let state: MockState;

  beforeEach(() => {
    const mock = createStatefulMock();
    proposals = new ProposalsClient(mock.client);
    voting = new VotingClient(mock.client);
    execution = new ExecutionClient(mock.client);
    state = mock.state;
  });

  it('pass proposal → lock funds → guardian veto → refund', async () => {
    // Votes pass the proposal
    await voting.castVote({ proposalId: 'proposal-001', choice: 'Approve', weight: 1.0 });
    await voting.castVote({ proposalId: 'proposal-001', choice: 'Approve', weight: 1.0 });
    const result = await proposals.finalizeProposal('proposal-001');
    expect(result.passed).toBe(true);

    // Create timelock and lock funds
    const timelock = await execution.createTimelock({
      proposalId: 'proposal-001',
      actions: JSON.stringify({ action: 'transfer', amount: 5000 }),
      delayHours: 48,
    });
    const funds = await execution.lockFunds(
      'proposal-001', 'timelock-001', 'treasury-001', 5000, 'MYC'
    );
    expect(funds.status).toBe('Locked');

    // Guardian vetoes during delay period
    await execution.vetoTimelock({
      timelockId: 'timelock-001',
      reason: 'Security concern identified in execution payload',
    });
    expect(state.timelockStatus).toBe('Cancelled');

    // Refund locked funds
    const refunded = await execution.refundFunds('proposal-001');
    expect(refunded.status).toBe('Refunded');
    expect(state.fundStatus).toBe('Refunded');
  });
});

// ============================================================================
// WORKFLOW 4: Treasury Allocation Lifecycle
// Create treasury → Propose allocation → Approve → Execute
// ============================================================================

describe('Treasury Allocation Workflow', () => {
  let treasury: TreasuryClient;
  let state: MockState;

  beforeEach(() => {
    const mock = createStatefulMock();
    treasury = new TreasuryClient(mock.client);
    state = mock.state;
  });

  it('create treasury → propose allocation → approve → execute', async () => {
    // Create treasury
    const t = await treasury.createTreasury({
      daoId: 'dao-001',
      name: 'Development Fund',
      description: 'Fund for development',
      governanceThreshold: 0.66,
      discretionaryLimit: 500,
      multiSigThreshold: 2,
      multiSigSigners: ['did:mycelix:alice', 'did:mycelix:bob'],
    });
    expect(state.treasuryCreated).toBe(true);

    // Propose allocation (above discretionary limit → requires governance)
    const allocation = await treasury.proposeAllocation({
      treasuryId: 'treasury-001',
      purpose: 'Equipment for community workshop',
      amount: 3000,
      currency: 'MYC',
      recipientDid: 'did:mycelix:workshop',
    });
    expect(allocation.status).toBe('Proposed');
    expect(allocation.amount).toBe(3000);

    // Multi-sig approvals
    const approved = await treasury.approveAllocation('alloc-001');
    expect(approved.status).toBe('Approved');
    expect(state.allocationStatus).toBe('Approved');

    // Execute allocation
    const executed = await treasury.executeAllocation('alloc-001');
    expect(executed.status).toBe('Executed');
    expect(state.allocationStatus).toBe('Executed');
  });
});

// ============================================================================
// WORKFLOW 5: Rejected Proposal (Quorum Not Met)
// ============================================================================

describe('Rejected Proposal Workflow', () => {
  let proposals: ProposalsClient;
  let voting: VotingClient;
  let state: MockState;

  beforeEach(() => {
    const mock = createStatefulMock();
    proposals = new ProposalsClient(mock.client);
    voting = new VotingClient(mock.client);
    state = mock.state;
  });

  it('single vote → finalize → rejected (quorum not met)', async () => {
    // Only 1 out of 3 members votes
    await voting.castVote({
      proposalId: 'proposal-001',
      choice: 'Approve',
      weight: 1.0,
    });
    expect(state.votesCast).toBe(1);

    // Finalize — quorum not met, should be rejected
    const result = await proposals.finalizeProposal('proposal-001');
    expect(result.passed).toBe(false);
    expect(result.quorumMet).toBe(false);
    expect(result.finalStatus).toBe('Rejected');
  });
});
