// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Governance hApp Conductor Integration Tests
 *
 * These tests verify the Governance clients work correctly with a real
 * Holochain conductor. They require the conductor harness to be available.
 *
 * Run with: npm run test:conductor
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import {
  ProposalsClient,
  VotingClient,
  DAOClient,
  DelegationClient,
  ExecutionClient,
  createGovernanceClients,
  type ZomeCallable,
  type DAO,
  type Proposal,
  type Vote,
} from '../../src/governance/index.js';
import {
  createValidatedGovernanceClients,
  ValidatedProposalsClient,
  ValidatedVotingClient,
  ValidatedDAOClient,
} from '../../src/governance/validated.js';
import { MycelixError, ErrorCode } from '../../src/errors.js';

// Conductor harness (dynamically imported if available)
let conductorHarness: {
  start: () => Promise<void>;
  stop: () => Promise<void>;
  createClient: () => ZomeCallable;
  getAgentPubKey: () => string;
} | null = null;

let client: ZomeCallable;
let agentPubKey: string;

// Skip all tests if conductor is not available
const describeConductor = process.env.HOLOCHAIN_CONDUCTOR_AVAILABLE ? describe : describe.skip;

describeConductor('Governance Conductor Integration Tests', () => {
  beforeAll(async () => {
    try {
      // Try to import the conductor harness
      const harness = await import('./conductor-harness.js');
      conductorHarness = harness.default || harness;
      await conductorHarness.start();
      client = conductorHarness.createClient();
      agentPubKey = conductorHarness.getAgentPubKey();
    } catch (error) {
      console.log('Conductor not available, skipping integration tests');
    }
  }, 60000);

  afterAll(async () => {
    if (conductorHarness) {
      await conductorHarness.stop();
    }
  }, 30000);

  describe('DAOClient', () => {
    let daoClient: DAOClient;
    let testDaoId: string;

    beforeAll(() => {
      daoClient = new DAOClient(client);
    });

    it('should create a DAO', async () => {
      const result = await daoClient.createDAO({
        name: 'Test DAO',
        description: 'A test DAO for integration testing',
        default_voting_period: 72,
        default_quorum: 0.5,
        default_threshold: 0.5,
      });

      expect(result).toBeDefined();
      expect(result.signed_action).toBeDefined();
      expect(result.entry).toBeDefined();

      const dao = result.entry.Present as DAO;
      expect(dao.name).toBe('Test DAO');
      expect(dao.default_voting_period).toBe(72);
      expect(dao.default_quorum).toBe(0.5);
      testDaoId = dao.id;
    });

    it('should list DAOs', async () => {
      const result = await daoClient.listDAOs();

      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);
    });

    it('should get a DAO by ID', async () => {
      const result = await daoClient.getDAO(testDaoId);

      expect(result).toBeDefined();
      if (result) {
        const dao = result.entry.Present as DAO;
        expect(dao.id).toBe(testDaoId);
        expect(dao.name).toBe('Test DAO');
      }
    });

    it('should join a DAO', async () => {
      const result = await daoClient.joinDAO(testDaoId, 100);

      expect(result).toBeDefined();
      const member = result.entry.Present;
      expect(member.dao_id).toBe(testDaoId);
      expect(member.voting_power).toBe(100);
    });

    it('should get DAO members', async () => {
      const result = await daoClient.getMembers(testDaoId);

      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);
    });
  });

  describe('ProposalsClient', () => {
    let proposalsClient: ProposalsClient;
    let daoClient: DAOClient;
    let testDaoId: string;
    let testProposalId: string;

    beforeAll(async () => {
      daoClient = new DAOClient(client);
      proposalsClient = new ProposalsClient(client);

      // Create a DAO for testing
      const dao = await daoClient.createDAO({
        name: 'Proposal Test DAO',
        description: 'DAO for proposal testing',
        default_voting_period: 24,
        default_quorum: 0.5,
        default_threshold: 0.5,
      });
      testDaoId = (dao.entry.Present as DAO).id;

      // Join the DAO
      await daoClient.joinDAO(testDaoId, 100);
    });

    it('should create a proposal', async () => {
      const result = await proposalsClient.createProposal({
        dao_id: testDaoId,
        title: 'Test Proposal',
        description: 'This is a test proposal for integration testing',
        proposal_type: 'Standard',
        voting_period_hours: 24,
        quorum: 0.5,
      });

      expect(result).toBeDefined();
      expect(result.signed_action).toBeDefined();

      const proposal = result.entry.Present as Proposal;
      expect(proposal.title).toBe('Test Proposal');
      expect(proposal.dao_id).toBe(testDaoId);
      expect(proposal.status).toBe('Active');
      testProposalId = proposal.id;
    });

    it('should get proposal by ID', async () => {
      const result = await proposalsClient.getProposal(testProposalId);

      expect(result).toBeDefined();
      if (result) {
        const proposal = result.entry.Present as Proposal;
        expect(proposal.id).toBe(testProposalId);
        expect(proposal.title).toBe('Test Proposal');
      }
    });

    it('should get proposals by DAO', async () => {
      const result = await proposalsClient.getProposalsByDAO(testDaoId);

      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);
    });

    it('should get active proposals', async () => {
      const result = await proposalsClient.getActiveProposals(testDaoId);

      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);
    });

    it('should get proposals by proposer', async () => {
      const did = `did:mycelix:${agentPubKey}`;
      const result = await proposalsClient.getProposalsByProposer(did);

      expect(Array.isArray(result)).toBe(true);
    });
  });

  describe('VotingClient', () => {
    let votingClient: VotingClient;
    let proposalsClient: ProposalsClient;
    let daoClient: DAOClient;
    let testDaoId: string;
    let testProposalId: string;

    beforeAll(async () => {
      daoClient = new DAOClient(client);
      proposalsClient = new ProposalsClient(client);
      votingClient = new VotingClient(client);

      // Create a DAO and proposal for testing
      const dao = await daoClient.createDAO({
        name: 'Voting Test DAO',
        description: 'DAO for voting testing',
        default_voting_period: 24,
        default_quorum: 0.5,
        default_threshold: 0.5,
      });
      testDaoId = (dao.entry.Present as DAO).id;

      await daoClient.joinDAO(testDaoId, 100);

      const proposal = await proposalsClient.createProposal({
        dao_id: testDaoId,
        title: 'Voting Test Proposal',
        description: 'Proposal for voting testing',
        proposal_type: 'Standard',
        voting_period_hours: 24,
        quorum: 0.5,
      });
      testProposalId = (proposal.entry.Present as Proposal).id;
    });

    it('should cast a vote', async () => {
      const result = await votingClient.castVote({
        proposal_id: testProposalId,
        choice: 'Approve',
        weight: 100,
        reason: 'Test vote',
      });

      expect(result).toBeDefined();
      const vote = result.entry.Present as Vote;
      expect(vote.proposal_id).toBe(testProposalId);
      expect(vote.choice).toBe('Approve');
      expect(vote.weight).toBe(100);
    });

    it('should check if user has voted', async () => {
      const did = `did:mycelix:${agentPubKey}`;
      const result = await votingClient.hasVoted(testProposalId, did);

      expect(typeof result).toBe('boolean');
      expect(result).toBe(true);
    });

    it('should get votes for proposal', async () => {
      const result = await votingClient.getVotesForProposal(testProposalId);

      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);
    });

    it('should get vote tally', async () => {
      const result = await votingClient.getVoteTally(testProposalId);

      expect(result).toBeDefined();
      expect(typeof result.approve).toBe('number');
      expect(typeof result.reject).toBe('number');
      expect(typeof result.abstain).toBe('number');
      expect(typeof result.total).toBe('number');
      expect(result.approve).toBe(100);
    });
  });

  describe('DelegationClient', () => {
    let delegationClient: DelegationClient;
    let daoClient: DAOClient;
    let testDaoId: string;

    beforeAll(async () => {
      daoClient = new DAOClient(client);
      delegationClient = new DelegationClient(client);

      const dao = await daoClient.createDAO({
        name: 'Delegation Test DAO',
        description: 'DAO for delegation testing',
        default_voting_period: 24,
        default_quorum: 0.5,
        default_threshold: 0.5,
      });
      testDaoId = (dao.entry.Present as DAO).id;

      await daoClient.joinDAO(testDaoId, 100);
    });

    it('should create a delegation', async () => {
      const result = await delegationClient.delegate({
        delegate: 'did:mycelix:delegatePubKey123456789012345678901234',
        dao_id: testDaoId,
        power: 50,
      });

      expect(result).toBeDefined();
      const delegation = result.entry.Present;
      expect(delegation.dao_id).toBe(testDaoId);
      expect(delegation.power).toBe(50);
    });

    it('should get delegations from a delegator', async () => {
      const did = `did:mycelix:${agentPubKey}`;
      const result = await delegationClient.getDelegationsFrom(did);

      expect(Array.isArray(result)).toBe(true);
    });

    it('should get effective voting power', async () => {
      const did = `did:mycelix:${agentPubKey}`;
      const result = await delegationClient.getEffectiveVotingPower(did, testDaoId);

      expect(typeof result).toBe('number');
    });
  });

  describe('ExecutionClient', () => {
    let executionClient: ExecutionClient;

    beforeAll(() => {
      executionClient = new ExecutionClient(client);
    });

    it('should get pending executions', async () => {
      const result = await executionClient.getPendingExecutions();

      expect(Array.isArray(result)).toBe(true);
    });

    it('should get pending executions for specific hApp', async () => {
      const result = await executionClient.getPendingExecutions('finance');

      expect(Array.isArray(result)).toBe(true);
    });
  });

  describe('Validated Clients', () => {
    it('should reject invalid DAO name', async () => {
      const clients = createValidatedGovernanceClients(client);

      await expect(
        clients.dao.createDAO({
          name: '',
          description: 'Test',
          default_voting_period: 24,
          default_quorum: 0.5,
          default_threshold: 0.5,
        })
      ).rejects.toThrow(MycelixError);
    });

    it('should reject invalid quorum', async () => {
      const clients = createValidatedGovernanceClients(client);

      await expect(
        clients.dao.createDAO({
          name: 'Test',
          description: 'Test',
          default_voting_period: 24,
          default_quorum: 1.5, // Invalid: > 1
          default_threshold: 0.5,
        })
      ).rejects.toThrow(MycelixError);
    });

    it('should reject invalid proposal voting period', async () => {
      const clients = createValidatedGovernanceClients(client);

      await expect(
        clients.proposals.createProposal({
          dao_id: 'test-dao',
          title: 'Test',
          description: 'Test',
          proposal_type: 'Standard',
          voting_period_hours: 1000, // Invalid: > 720 (30 days)
          quorum: 0.5,
        })
      ).rejects.toThrow(MycelixError);
    });

    it('should reject invalid vote choice', async () => {
      const clients = createValidatedGovernanceClients(client);

      await expect(
        clients.voting.castVote({
          proposal_id: 'test-proposal',
          choice: 'Maybe' as any, // Invalid choice
          weight: 100,
        })
      ).rejects.toThrow(MycelixError);
    });

    it('should reject invalid DID for delegation', async () => {
      const clients = createValidatedGovernanceClients(client);

      await expect(
        clients.delegation.delegate({
          delegate: 'invalid-did', // Must start with 'did:'
          dao_id: 'test-dao',
          power: 50,
        })
      ).rejects.toThrow(MycelixError);
    });

    it('should accept valid inputs', async () => {
      const clients = createValidatedGovernanceClients(client);

      // This should not throw validation errors
      const result = await clients.dao.listDAOs();
      expect(Array.isArray(result)).toBe(true);
    });
  });

  describe('createGovernanceClients factory', () => {
    it('should create all clients', () => {
      const clients = createGovernanceClients(client);

      expect(clients.proposals).toBeInstanceOf(ProposalsClient);
      expect(clients.voting).toBeInstanceOf(VotingClient);
      expect(clients.dao).toBeInstanceOf(DAOClient);
      expect(clients.delegation).toBeInstanceOf(DelegationClient);
      expect(clients.execution).toBeInstanceOf(ExecutionClient);
    });
  });
});
