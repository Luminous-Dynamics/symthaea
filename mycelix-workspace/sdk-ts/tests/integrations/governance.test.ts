/**
 * Governance Integration Tests
 *
 * Tests for GovernanceService - proposals, voting, and delegation
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  GovernanceService,
  getGovernanceService,
  resetGovernanceService,
  type Proposal,
  type Vote,
  type DAOMember,
  type Delegation,
} from '../../src/integrations/governance/index.js';

describe('Governance Integration', () => {
  let service: GovernanceService;

  beforeEach(() => {
    resetGovernanceService();
    service = new GovernanceService();
  });

  describe('GovernanceService', () => {
    describe('createProposal', () => {
      it('should create a new proposal', () => {
        // Register proposer first (required for security)
        service.registerMember({
          did: 'did:mycelix:alice',
          daoId: 'luminous-core',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        const proposal = service.createProposal({
          title: 'Increase Treasury Allocation',
          description: 'Proposal to increase development fund by 10%',
          proposerId: 'did:mycelix:alice',
          daoId: 'luminous-core',
          votingPeriodHours: 72,
          quorumPercentage: 0.6,
        });

        expect(proposal).toBeDefined();
        expect(proposal.id).toMatch(/^proposal-/);
        expect(proposal.title).toBe('Increase Treasury Allocation');
        expect(proposal.status).toBe('active');
        expect(proposal.approvesWeight).toBe(0);
        expect(proposal.rejectsWeight).toBe(0);
      });

      it('should set voting end time based on period', () => {
        // Register proposer first
        service.registerMember({
          did: 'did:mycelix:proposer',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        const proposal = service.createProposal({
          title: 'Test Proposal',
          description: 'Description',
          proposerId: 'did:mycelix:proposer',
          daoId: 'test-dao',
          votingPeriodHours: 24,
          quorumPercentage: 0.5,
        });

        const expectedEnd = Date.now() + 24 * 60 * 60 * 1000;
        expect(proposal.votingEnds).toBeGreaterThan(Date.now());
        expect(proposal.votingEnds).toBeLessThanOrEqual(expectedEnd + 1000);
      });

      it('should support optional execution payload', () => {
        // Register proposer first
        service.registerMember({
          did: 'did:mycelix:proposer',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        const proposal = service.createProposal({
          title: 'Executable Proposal',
          description: 'This will execute something',
          proposerId: 'did:mycelix:proposer',
          daoId: 'test-dao',
          votingPeriodHours: 48,
          quorumPercentage: 0.5,
          executionPayload: JSON.stringify({ action: 'transfer', amount: 1000 }),
        });

        expect(proposal.executionPayload).toBeDefined();
        const payload = JSON.parse(proposal.executionPayload!);
        expect(payload.action).toBe('transfer');
      });

      it('should throw if proposer is not a registered member', () => {
        expect(() => {
          service.createProposal({
            title: 'Test Proposal',
            description: 'Description',
            proposerId: 'did:mycelix:unregistered',
            daoId: 'test-dao',
            votingPeriodHours: 24,
            quorumPercentage: 0.5,
          });
        }).toThrow('Proposer must be a registered member of the DAO');
      });
    });

    describe('castVote', () => {
      it('should cast an approve vote', () => {
        // Register proposer and voter
        service.registerMember({
          did: 'did:mycelix:proposer',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });
        service.registerMember({
          did: 'did:mycelix:voter1',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        const proposal = service.createProposal({
          title: 'Vote Test',
          description: 'Test',
          proposerId: 'did:mycelix:proposer',
          daoId: 'test-dao',
          votingPeriodHours: 72,
          quorumPercentage: 0.5,
        });

        const vote = service.castVote({
          proposalId: proposal.id,
          voterId: 'did:mycelix:voter1',
          choice: 'approve',
          weight: 100,
        });

        expect(vote).toBeDefined();
        expect(vote.choice).toBe('approve');
        expect(vote.weight).toBe(100);

        const updated = service.getProposal(proposal.id);
        expect(updated!.approvesWeight).toBe(100);
      });

      it('should cast a reject vote', () => {
        // Register proposer and voter
        service.registerMember({
          did: 'did:mycelix:proposer',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });
        service.registerMember({
          did: 'did:mycelix:voter1',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        const proposal = service.createProposal({
          title: 'Reject Test',
          description: 'Test',
          proposerId: 'did:mycelix:proposer',
          daoId: 'test-dao',
          votingPeriodHours: 72,
          quorumPercentage: 0.5,
        });

        service.castVote({
          proposalId: proposal.id,
          voterId: 'did:mycelix:voter1',
          choice: 'reject',
          weight: 50,
        });

        const updated = service.getProposal(proposal.id);
        expect(updated!.rejectsWeight).toBe(50);
      });

      it('should cast an abstain vote', () => {
        // Register proposer and voter
        service.registerMember({
          did: 'did:mycelix:proposer',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });
        service.registerMember({
          did: 'did:mycelix:voter1',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        const proposal = service.createProposal({
          title: 'Abstain Test',
          description: 'Test',
          proposerId: 'did:mycelix:proposer',
          daoId: 'test-dao',
          votingPeriodHours: 72,
          quorumPercentage: 0.5,
        });

        service.castVote({
          proposalId: proposal.id,
          voterId: 'did:mycelix:voter1',
          choice: 'abstain',
          weight: 25,
        });

        const updated = service.getProposal(proposal.id);
        expect(updated!.abstainWeight).toBe(25);
      });

      it('should throw for non-existent proposal', () => {
        expect(() => {
          service.castVote({
            proposalId: 'proposal-fake',
            voterId: 'did:mycelix:voter',
            choice: 'approve',
            weight: 100,
          });
        }).toThrow('Proposal proposal-fake not found');
      });

      it('should throw for duplicate vote', () => {
        // Register proposer and voter
        service.registerMember({
          did: 'did:mycelix:proposer',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });
        service.registerMember({
          did: 'did:mycelix:voter1',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        const proposal = service.createProposal({
          title: 'Duplicate Test',
          description: 'Test',
          proposerId: 'did:mycelix:proposer',
          daoId: 'test-dao',
          votingPeriodHours: 72,
          quorumPercentage: 0.5,
        });

        service.castVote({
          proposalId: proposal.id,
          voterId: 'did:mycelix:voter1',
          choice: 'approve',
          weight: 100,
        });

        expect(() => {
          service.castVote({
            proposalId: proposal.id,
            voterId: 'did:mycelix:voter1',
            choice: 'reject',
            weight: 100,
          });
        }).toThrow('Already voted on this proposal');
      });

      it('should throw if voter is not a registered member', () => {
        service.registerMember({
          did: 'did:mycelix:proposer',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        const proposal = service.createProposal({
          title: 'Vote Test',
          description: 'Test',
          proposerId: 'did:mycelix:proposer',
          daoId: 'test-dao',
          votingPeriodHours: 72,
          quorumPercentage: 0.5,
        });

        expect(() => {
          service.castVote({
            proposalId: proposal.id,
            voterId: 'did:mycelix:unregistered',
            choice: 'approve',
            weight: 100,
          });
        }).toThrow('Voter must be a registered member of the DAO');
      });
    });

    describe('registerMember', () => {
      it('should register a new DAO member', () => {
        const member = service.registerMember({
          did: 'did:mycelix:member1',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        expect(member).toBeDefined();
        expect(member.did).toBe('did:mycelix:member1');
        expect(member.role).toBe('member');
        expect(member.votingPower).toBe(100);
        expect(member.reputation).toBeDefined();
      });

      it('should support different roles', () => {
        const steward = service.registerMember({
          did: 'did:mycelix:steward',
          daoId: 'test-dao',
          role: 'steward',
          votingPower: 500,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        expect(steward.role).toBe('steward');
      });
    });

    describe('delegate', () => {
      it('should delegate voting power', () => {
        service.registerMember({
          did: 'did:mycelix:delegator',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        service.registerMember({
          did: 'did:mycelix:delegate',
          daoId: 'test-dao',
          role: 'delegate',
          votingPower: 200,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        const delegation = service.delegate({
          delegatorId: 'did:mycelix:delegator',
          delegateId: 'did:mycelix:delegate',
          daoId: 'test-dao',
          power: 100,
        });

        expect(delegation).toBeDefined();
        expect(delegation.power).toBe(100);
      });

      it('should support category-specific delegation', () => {
        service.registerMember({
          did: 'did:mycelix:delegator',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        const delegation = service.delegate({
          delegatorId: 'did:mycelix:delegator',
          delegateId: 'did:mycelix:delegate',
          daoId: 'test-dao',
          power: 50,
          categories: ['treasury', 'technical'],
        });

        expect(delegation.categories).toContain('treasury');
        expect(delegation.categories).toContain('technical');
      });
    });

    describe('finalizeProposal', () => {
      it('should pass proposal with majority approval and quorum', () => {
        // Register proposer and voters
        service.registerMember({
          did: 'did:mycelix:proposer',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });
        service.registerMember({
          did: 'did:mycelix:voter1',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });
        service.registerMember({
          did: 'did:mycelix:voter2',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        const proposal = service.createProposal({
          title: 'Finalize Test',
          description: 'Test',
          proposerId: 'did:mycelix:proposer',
          daoId: 'test-dao',
          votingPeriodHours: 1, // Minimum 1 hour
          quorumPercentage: 0.5,
        });

        service.castVote({ proposalId: proposal.id, voterId: 'did:mycelix:voter1', choice: 'approve', weight: 100 });
        service.castVote({ proposalId: proposal.id, voterId: 'did:mycelix:voter2', choice: 'approve', weight: 100 });

        const result = service.finalizeProposal(proposal.id);

        expect(result.passed).toBe(true);
        expect(result.quorumMet).toBe(true);
        expect(result.approvalPercentage).toBe(1);
      });

      it('should reject proposal without majority', () => {
        // Register proposer and voters
        service.registerMember({
          did: 'did:mycelix:proposer',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });
        service.registerMember({
          did: 'did:mycelix:voter1',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });
        service.registerMember({
          did: 'did:mycelix:voter2',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        const proposal = service.createProposal({
          title: 'Reject Test',
          description: 'Test',
          proposerId: 'did:mycelix:proposer',
          daoId: 'test-dao',
          votingPeriodHours: 1,
          quorumPercentage: 0.5,
        });

        service.castVote({ proposalId: proposal.id, voterId: 'did:mycelix:voter1', choice: 'approve', weight: 40 });
        service.castVote({ proposalId: proposal.id, voterId: 'did:mycelix:voter2', choice: 'reject', weight: 60 });

        const result = service.finalizeProposal(proposal.id);

        expect(result.passed).toBe(false);
        expect(result.approvalPercentage).toBe(0.4);
      });

      it('should throw for non-existent proposal', () => {
        expect(() => {
          service.finalizeProposal('proposal-fake');
        }).toThrow('Proposal proposal-fake not found');
      });
    });

    describe('error paths and edge cases', () => {
      it('should throw for quorum percentage above 1.0', () => {
        service.registerMember({
          did: 'did:mycelix:proposer',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        expect(() => {
          service.createProposal({
            title: 'Invalid Quorum',
            description: 'Test',
            proposerId: 'did:mycelix:proposer',
            daoId: 'test-dao',
            votingPeriodHours: 24,
            quorumPercentage: 1.5,
          });
        }).toThrow('Quorum percentage must be <= 1');
      });

      it('should throw for negative quorum percentage', () => {
        service.registerMember({
          did: 'did:mycelix:proposer',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        expect(() => {
          service.createProposal({
            title: 'Negative Quorum',
            description: 'Test',
            proposerId: 'did:mycelix:proposer',
            daoId: 'test-dao',
            votingPeriodHours: 24,
            quorumPercentage: -0.5,
          });
        }).toThrow('Quorum percentage must be >= 0');
      });

      it('should throw for negative voting period', () => {
        service.registerMember({
          did: 'did:mycelix:proposer',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        expect(() => {
          service.createProposal({
            title: 'Negative Period',
            description: 'Test',
            proposerId: 'did:mycelix:proposer',
            daoId: 'test-dao',
            votingPeriodHours: -24,
            quorumPercentage: 0.5,
          });
        }).toThrow('Voting period must be positive');
      });

      it('should throw for zero voting period', () => {
        service.registerMember({
          did: 'did:mycelix:proposer',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        expect(() => {
          service.createProposal({
            title: 'Zero Period',
            description: 'Test',
            proposerId: 'did:mycelix:proposer',
            daoId: 'test-dao',
            votingPeriodHours: 0,
            quorumPercentage: 0.5,
          });
        }).toThrow('Voting period must be positive');
      });

      it('should throw for voting period exceeding maximum (90 days)', () => {
        service.registerMember({
          did: 'did:mycelix:proposer',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        expect(() => {
          service.createProposal({
            title: 'Excessive Period',
            description: 'Test',
            proposerId: 'did:mycelix:proposer',
            daoId: 'test-dao',
            votingPeriodHours: 3000, // > 2160 (90 days)
            quorumPercentage: 0.5,
          });
        }).toThrow('Voting period must be <= 2160');
      });

      it('should throw when voting on a finalized (non-active) proposal', () => {
        service.registerMember({
          did: 'did:mycelix:proposer',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });
        service.registerMember({
          did: 'did:mycelix:voter1',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });
        service.registerMember({
          did: 'did:mycelix:voter2',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        const proposal = service.createProposal({
          title: 'Finalize Test',
          description: 'Test',
          proposerId: 'did:mycelix:proposer',
          daoId: 'test-dao',
          votingPeriodHours: 1,
          quorumPercentage: 0.5,
        });

        service.castVote({ proposalId: proposal.id, voterId: 'did:mycelix:voter1', choice: 'approve', weight: 100 });
        service.finalizeProposal(proposal.id);

        // Proposal is now 'passed' - voting should fail
        expect(() => {
          service.castVote({
            proposalId: proposal.id,
            voterId: 'did:mycelix:voter2',
            choice: 'approve',
            weight: 100,
          });
        }).toThrow('Proposal is not active');
      });

      it('should return 0 participation rate when no members exist in DAO', () => {
        service.registerMember({
          did: 'did:mycelix:proposer',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        const proposal = service.createProposal({
          title: 'No Votes',
          description: 'Test',
          proposerId: 'did:mycelix:proposer',
          daoId: 'test-dao',
          votingPeriodHours: 1,
          quorumPercentage: 0.5,
        });

        // Finalize without any votes
        const result = service.finalizeProposal(proposal.id);

        expect(result.passed).toBe(false);
        expect(result.totalVotes).toBe(0);
        expect(result.approvalPercentage).toBe(0);
      });

      it('should reject proposal when quorum is not met', () => {
        // Register 5 members but only 1 votes
        for (let i = 0; i < 5; i++) {
          service.registerMember({
            did: `did:mycelix:member${i}`,
            daoId: 'test-dao',
            role: 'member',
            votingPower: 100,
            delegatedPower: 0,
            joinedAt: Date.now(),
          });
        }

        const proposal = service.createProposal({
          title: 'Low Participation',
          description: 'Test',
          proposerId: 'did:mycelix:member0',
          daoId: 'test-dao',
          votingPeriodHours: 1,
          quorumPercentage: 0.6, // 60% quorum
        });

        // Only 1 out of 5 members votes (20% participation < 60% quorum)
        service.castVote({ proposalId: proposal.id, voterId: 'did:mycelix:member1', choice: 'approve', weight: 100 });

        const result = service.finalizeProposal(proposal.id);

        expect(result.quorumMet).toBe(false);
        expect(result.passed).toBe(false);
      });
    });

    describe('getActiveProposals', () => {
      it('should return only active proposals for a DAO', () => {
        // Register proposers
        service.registerMember({
          did: 'did:mycelix:proposer',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });
        service.registerMember({
          did: 'did:mycelix:proposer',
          daoId: 'other-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        service.createProposal({
          title: 'Active 1',
          description: 'Test',
          proposerId: 'did:mycelix:proposer',
          daoId: 'test-dao',
          votingPeriodHours: 72,
          quorumPercentage: 0.5,
        });

        service.createProposal({
          title: 'Active 2',
          description: 'Test',
          proposerId: 'did:mycelix:proposer',
          daoId: 'test-dao',
          votingPeriodHours: 72,
          quorumPercentage: 0.5,
        });

        service.createProposal({
          title: 'Different DAO',
          description: 'Test',
          proposerId: 'did:mycelix:proposer',
          daoId: 'other-dao',
          votingPeriodHours: 72,
          quorumPercentage: 0.5,
        });

        const activeProposals = service.getActiveProposals('test-dao');

        expect(activeProposals.length).toBe(2);
        expect(activeProposals.every(p => p.daoId === 'test-dao')).toBe(true);
      });
    });

    describe('getVotes', () => {
      it('should return all votes for a proposal', () => {
        // Register proposer and voters
        service.registerMember({
          did: 'did:mycelix:proposer',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });
        service.registerMember({
          did: 'did:mycelix:voter1',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });
        service.registerMember({
          did: 'did:mycelix:voter2',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });
        service.registerMember({
          did: 'did:mycelix:voter3',
          daoId: 'test-dao',
          role: 'member',
          votingPower: 100,
          delegatedPower: 0,
          joinedAt: Date.now(),
        });

        const proposal = service.createProposal({
          title: 'Votes Test',
          description: 'Test',
          proposerId: 'did:mycelix:proposer',
          daoId: 'test-dao',
          votingPeriodHours: 72,
          quorumPercentage: 0.5,
        });

        service.castVote({ proposalId: proposal.id, voterId: 'did:mycelix:voter1', choice: 'approve', weight: 100 });
        service.castVote({ proposalId: proposal.id, voterId: 'did:mycelix:voter2', choice: 'reject', weight: 50 });
        service.castVote({ proposalId: proposal.id, voterId: 'did:mycelix:voter3', choice: 'abstain', weight: 25 });

        const votes = service.getVotes(proposal.id);

        expect(votes.length).toBe(3);
      });
    });

    describe('getEffectiveVotingPower', () => {
      it('should return combined voting and delegated power', () => {
        service.registerMember({
          did: 'did:mycelix:poweruser',
          daoId: 'test-dao',
          role: 'delegate',
          votingPower: 100,
          delegatedPower: 50,
          joinedAt: Date.now(),
        });

        const power = service.getEffectiveVotingPower('did:mycelix:poweruser', 'test-dao');

        expect(power).toBe(150);
      });

      it('should return 0 for unknown member', () => {
        const power = service.getEffectiveVotingPower('did:mycelix:unknown', 'test-dao');

        expect(power).toBe(0);
      });
    });
  });

  describe('getGovernanceService', () => {
    it('should return singleton instance', () => {
      const service1 = getGovernanceService();
      const service2 = getGovernanceService();

      expect(service1).toBe(service2);
    });
  });
});
