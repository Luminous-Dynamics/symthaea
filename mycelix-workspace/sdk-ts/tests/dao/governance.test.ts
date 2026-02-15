/**
 * DAO Governance Module Tests
 *
 * Tests for GovernanceClient - proposal creation and management
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { GovernanceClient } from '../../src/dao/governance';
import type { DAOClient } from '../../src/dao/client';
import type { Proposal, ProposalType, ProposalStatus } from '../../src/dao/types';

// Mock DAOClient
const createMockClient = () => {
  const callZome = vi.fn();
  const getZomeName = vi.fn().mockReturnValue('governance');

  return {
    callZome,
    getZomeName,
  } as unknown as DAOClient;
};

// Mock proposal data
const mockProposal: Proposal = {
  proposal_type: 'TechnicalMIP',
  title: 'Test Proposal',
  description: 'A test proposal for governance',
  author: new Uint8Array(32),
  classification: { e: 'E2', n: 'N2', m: 'M3', override_from_default: false },
  status: 'Draft',
  created_at: Date.now() * 1000,
  updated_at: Date.now() * 1000,
  tags: ['test', 'governance'],
};

const mockRecord = {
  signed_action: {
    hashed: {
      hash: new Uint8Array(32),
    },
  },
  entry: {
    Present: {
      entry: mockProposal,
    },
  },
};

describe('GovernanceClient', () => {
  let client: DAOClient;
  let governance: GovernanceClient;

  beforeEach(() => {
    client = createMockClient();
    governance = new GovernanceClient(client);
  });

  describe('createProposal', () => {
    it('should create a proposal successfully', async () => {
      (client.callZome as any).mockResolvedValue(mockRecord);

      const result = await governance.createProposal({
        proposal_type: 'TechnicalMIP',
        title: 'Test Proposal',
        description: 'A test proposal',
        classification: { e: 'E2', n: 'N2', m: 'M3', override_from_default: false },
        tags: ['test'],
      });

      expect(result.proposal).toBeDefined();
      expect(result.proposal.title).toBe('Test Proposal');
      expect(result.actionHash).toBeDefined();
      expect(client.callZome).toHaveBeenCalledWith(
        'governance',
        'create_proposal',
        expect.any(Object)
      );
    });

    it('should handle different proposal types', async () => {
      const proposalTypes: ProposalType[] = [
        'TechnicalMIP',
        'OperationalMIP',
        'ConstitutionalMIP',
        'EmergencyMIP',
        'SubDAO',
        'Grant',
        'Signaling',
      ];

      for (const type of proposalTypes) {
        (client.callZome as any).mockResolvedValue({
          ...mockRecord,
          entry: { Present: { entry: { ...mockProposal, proposal_type: type } } },
        });

        const result = await governance.createProposal({
          proposal_type: type,
          title: `${type} Proposal`,
          description: 'Testing proposal type',
          classification: { e: 'E2', n: 'N2', m: 'M3', override_from_default: false },
          tags: [],
        });

        expect(result.proposal.proposal_type).toBe(type);
      }
    });
  });

  describe('getProposal', () => {
    it('should retrieve a proposal by hash', async () => {
      (client.callZome as any).mockResolvedValue(mockRecord);

      const result = await governance.getProposal(new Uint8Array(32));

      expect(result).toBeDefined();
      expect(result?.title).toBe('Test Proposal');
    });

    it('should return null for non-existent proposal', async () => {
      (client.callZome as any).mockResolvedValue(null);

      const result = await governance.getProposal(new Uint8Array(32));

      expect(result).toBeNull();
    });
  });

  describe('getAllProposals', () => {
    it('should retrieve all proposals', async () => {
      (client.callZome as any).mockResolvedValue([mockRecord, mockRecord]);

      const result = await governance.getAllProposals();

      expect(result).toHaveLength(2);
      expect(result[0].title).toBe('Test Proposal');
    });

    it('should return empty array when no proposals exist', async () => {
      (client.callZome as any).mockResolvedValue([]);

      const result = await governance.getAllProposals();

      expect(result).toEqual([]);
    });
  });

  describe('getActiveProposals', () => {
    it('should retrieve only active proposals', async () => {
      const activeProposal = { ...mockRecord };
      activeProposal.entry.Present.entry = { ...mockProposal, status: 'Voting' };

      (client.callZome as any).mockResolvedValue([activeProposal]);

      const result = await governance.getActiveProposals();

      expect(result).toHaveLength(1);
      expect(result[0].status).toBe('Voting');
    });
  });

  describe('getProposalsByType', () => {
    it('should filter proposals by type', async () => {
      (client.callZome as any).mockResolvedValue([mockRecord]);

      const result = await governance.getProposalsByType('TechnicalMIP');

      expect(client.callZome).toHaveBeenCalledWith(
        'governance',
        'get_proposals_by_type',
        'TechnicalMIP'
      );
      expect(result).toHaveLength(1);
    });
  });

  describe('getProposalsByAuthor', () => {
    it('should filter proposals by author', async () => {
      const authorKey = new Uint8Array(32);
      (client.callZome as any).mockResolvedValue([mockRecord]);

      const result = await governance.getProposalsByAuthor(authorKey);

      expect(client.callZome).toHaveBeenCalledWith(
        'governance',
        'get_proposals_by_author',
        authorKey
      );
      expect(result).toHaveLength(1);
    });
  });

  describe('updateProposalStatus', () => {
    it('should update proposal status', async () => {
      const updatedRecord = { ...mockRecord };
      updatedRecord.entry.Present.entry = { ...mockProposal, status: 'Voting' };

      (client.callZome as any).mockResolvedValue(updatedRecord);

      const result = await governance.updateProposalStatus(new Uint8Array(32), 'Voting');

      expect(result.status).toBe('Voting');
      expect(client.callZome).toHaveBeenCalledWith(
        'governance',
        'update_proposal_status',
        expect.objectContaining({ new_status: 'Voting' })
      );
    });

    it('should handle all status transitions', async () => {
      const statuses: ProposalStatus[] = [
        'Draft',
        'Review',
        'Voting',
        'Passed',
        'Rejected',
        'Executed',
        'Cancelled',
      ];

      for (const status of statuses) {
        const statusRecord = { ...mockRecord };
        statusRecord.entry.Present.entry = { ...mockProposal, status };

        (client.callZome as any).mockResolvedValue(statusRecord);

        const result = await governance.updateProposalStatus(new Uint8Array(32), status);

        expect(result.status).toBe(status);
      }
    });
  });

  describe('getConfig', () => {
    it('should retrieve governance configuration', async () => {
      const mockConfig = {
        quorum_percentage: 10,
        voting_period_days: 7,
        review_period_days: 3,
      };

      (client.callZome as any).mockResolvedValue(mockConfig);

      const result = await governance.getConfig();

      expect(result).toEqual(mockConfig);
    });
  });

  describe('getSuggestedMechanism', () => {
    it('should return suggested voting mechanism for proposal type', async () => {
      (client.callZome as any).mockResolvedValue('QuadraticVoting');

      const result = await governance.getSuggestedMechanism('TechnicalMIP');

      expect(result).toBe('QuadraticVoting');
      expect(client.callZome).toHaveBeenCalledWith(
        'governance',
        'get_suggested_mechanism',
        'TechnicalMIP'
      );
    });
  });

  describe('error handling', () => {
    it('should propagate zome call errors', async () => {
      (client.callZome as any).mockRejectedValue(new Error('Zome call failed'));

      await expect(governance.getAllProposals()).rejects.toThrow('Zome call failed');
    });
  });
});
