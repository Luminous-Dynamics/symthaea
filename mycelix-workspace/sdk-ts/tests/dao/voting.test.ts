/**
 * DAO Voting Module Tests
 *
 * Tests for VotingClient - vote casting and tallying
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { VotingClient } from '../../src/dao/voting';
import type { DAOClient } from '../../src/dao/client';
import type { Vote, QuadraticVote, VoteTally } from '../../src/dao/types';

// Mock DAOClient
const createMockClient = () => {
  const callZome = vi.fn();
  const getZomeName = vi.fn().mockReturnValue('governance');
  const getMyPubKey = vi.fn().mockResolvedValue(new Uint8Array(32));

  return {
    callZome,
    getZomeName,
    getMyPubKey,
  } as unknown as DAOClient;
};

// Mock vote data
const mockVoterKey = new Uint8Array(32);
const mockProposalHash = new Uint8Array(32);

const mockVote: Vote = {
  proposal: mockProposalHash,
  voter: mockVoterKey,
  choice: 'Yes',
  weight: 100,
  justification: 'I support this proposal',
  cast_at: Date.now() * 1000,
};

const mockQuadraticVote: QuadraticVote = {
  proposal: mockProposalHash,
  voter: mockVoterKey,
  credits: 25,
  effective_votes: 5,
  round_id: 'round-001',
  cast_at: Date.now() * 1000,
};

const mockRecord = {
  signed_action: { hashed: { hash: new Uint8Array(32) } },
  entry: { Present: { entry: mockVote } },
};

const mockQuadraticRecord = {
  signed_action: { hashed: { hash: new Uint8Array(32) } },
  entry: { Present: { entry: mockQuadraticVote } },
};

describe('VotingClient', () => {
  let client: DAOClient;
  let voting: VotingClient;

  beforeEach(() => {
    client = createMockClient();
    voting = new VotingClient(client);
  });

  describe('castVote', () => {
    it('should cast a Yes vote', async () => {
      (client.callZome as any).mockResolvedValue(mockRecord);

      const result = await voting.castVote({
        proposal: mockProposalHash,
        choice: 'Yes',
        justification: 'I agree',
      });

      expect(result.choice).toBe('Yes');
      expect(client.callZome).toHaveBeenCalledWith(
        'governance',
        'cast_vote',
        expect.objectContaining({ choice: 'Yes' })
      );
    });

    it('should cast a No vote', async () => {
      const noVoteRecord = {
        ...mockRecord,
        entry: { Present: { entry: { ...mockVote, choice: 'No' } } },
      };
      (client.callZome as any).mockResolvedValue(noVoteRecord);

      const result = await voting.castVote({
        proposal: mockProposalHash,
        choice: 'No',
        justification: 'I disagree',
      });

      expect(result.choice).toBe('No');
    });

    it('should cast an Abstain vote', async () => {
      const abstainRecord = {
        ...mockRecord,
        entry: { Present: { entry: { ...mockVote, choice: 'Abstain' } } },
      };
      (client.callZome as any).mockResolvedValue(abstainRecord);

      const result = await voting.castVote({
        proposal: mockProposalHash,
        choice: 'Abstain',
      });

      expect(result.choice).toBe('Abstain');
    });
  });

  describe('castQuadraticVote', () => {
    it('should cast a quadratic vote with correct effective votes', async () => {
      (client.callZome as any).mockResolvedValue(mockQuadraticRecord);

      const result = await voting.castQuadraticVote({
        proposal: mockProposalHash,
        credits: 25,
        round_id: 'round-001',
      });

      expect(result.credits).toBe(25);
      expect(result.effective_votes).toBe(5); // sqrt(25) = 5
    });

    it('should handle various credit amounts', async () => {
      const testCases = [
        { credits: 1, effectiveVotes: 1 },
        { credits: 4, effectiveVotes: 2 },
        { credits: 9, effectiveVotes: 3 },
        { credits: 16, effectiveVotes: 4 },
        { credits: 100, effectiveVotes: 10 },
      ];

      for (const { credits, effectiveVotes } of testCases) {
        const record = {
          ...mockQuadraticRecord,
          entry: {
            Present: {
              entry: { ...mockQuadraticVote, credits, effective_votes: effectiveVotes },
            },
          },
        };
        (client.callZome as any).mockResolvedValue(record);

        const result = await voting.castQuadraticVote({
          proposal: mockProposalHash,
          credits,
          round_id: 'round-001',
        });

        expect(result.effective_votes).toBe(effectiveVotes);
      }
    });
  });

  describe('getVotesForProposal', () => {
    it('should retrieve all votes for a proposal', async () => {
      (client.callZome as any).mockResolvedValue([mockRecord, mockRecord, mockRecord]);

      const result = await voting.getVotesForProposal(mockProposalHash);

      expect(result).toHaveLength(3);
      expect(result[0].choice).toBe('Yes');
    });

    it('should return empty array for proposals with no votes', async () => {
      (client.callZome as any).mockResolvedValue([]);

      const result = await voting.getVotesForProposal(mockProposalHash);

      expect(result).toEqual([]);
    });
  });

  describe('tallyVotes', () => {
    it('should return correct vote tally', async () => {
      const mockTally: VoteTally = {
        proposal: mockProposalHash,
        yes_votes: 150,
        no_votes: 50,
        abstain_votes: 25,
        total_weight: 225,
        quorum_reached: true,
        result: 'Passed',
        finalized_at: Date.now() * 1000,
      };

      (client.callZome as any).mockResolvedValue(mockTally);

      const result = await voting.tallyVotes(mockProposalHash);

      expect(result.yes_votes).toBe(150);
      expect(result.no_votes).toBe(50);
      expect(result.quorum_reached).toBe(true);
      expect(result.result).toBe('Passed');
    });

    it('should handle rejected proposals', async () => {
      const mockTally: VoteTally = {
        proposal: mockProposalHash,
        yes_votes: 30,
        no_votes: 70,
        abstain_votes: 10,
        total_weight: 110,
        quorum_reached: true,
        result: 'Rejected',
        finalized_at: Date.now() * 1000,
      };

      (client.callZome as any).mockResolvedValue(mockTally);

      const result = await voting.tallyVotes(mockProposalHash);

      expect(result.result).toBe('Rejected');
    });

    it('should handle quorum not reached', async () => {
      const mockTally: VoteTally = {
        proposal: mockProposalHash,
        yes_votes: 5,
        no_votes: 2,
        abstain_votes: 1,
        total_weight: 8,
        quorum_reached: false,
        result: 'NoQuorum',
        finalized_at: undefined,
      };

      (client.callZome as any).mockResolvedValue(mockTally);

      const result = await voting.tallyVotes(mockProposalHash);

      expect(result.quorum_reached).toBe(false);
      expect(result.result).toBe('NoQuorum');
    });
  });

  describe('hasVoted', () => {
    it('should return true if user has voted', async () => {
      (client.callZome as any).mockResolvedValue([mockRecord]);
      (client.getMyPubKey as any).mockResolvedValue(mockVoterKey);

      const result = await voting.hasVoted(mockProposalHash);

      expect(result).toBe(true);
    });

    it('should return false if user has not voted', async () => {
      (client.callZome as any).mockResolvedValue([]);

      const result = await voting.hasVoted(mockProposalHash);

      expect(result).toBe(false);
    });
  });

  describe('getMyVote', () => {
    it('should return user vote if exists', async () => {
      (client.callZome as any).mockResolvedValue([mockRecord]);
      (client.getMyPubKey as any).mockResolvedValue(mockVoterKey);

      const result = await voting.getMyVote(mockProposalHash);

      expect(result).toBeDefined();
      expect(result?.choice).toBe('Yes');
    });

    it('should return null if user has not voted', async () => {
      (client.callZome as any).mockResolvedValue([]);

      const result = await voting.getMyVote(mockProposalHash);

      expect(result).toBeNull();
    });
  });

  describe('quadratic vote calculations', () => {
    it('should calculate effective votes correctly', () => {
      expect(voting.calculateEffectiveVotes(1)).toBe(1);
      expect(voting.calculateEffectiveVotes(4)).toBe(2);
      expect(voting.calculateEffectiveVotes(9)).toBe(3);
      expect(voting.calculateEffectiveVotes(16)).toBe(4);
      expect(voting.calculateEffectiveVotes(25)).toBe(5);
      expect(voting.calculateEffectiveVotes(100)).toBe(10);
    });

    it('should calculate credits needed correctly', () => {
      expect(voting.calculateCreditsNeeded(1)).toBe(1);
      expect(voting.calculateCreditsNeeded(2)).toBe(4);
      expect(voting.calculateCreditsNeeded(3)).toBe(9);
      expect(voting.calculateCreditsNeeded(5)).toBe(25);
      expect(voting.calculateCreditsNeeded(10)).toBe(100);
    });

    it('should handle non-perfect squares', () => {
      expect(voting.calculateEffectiveVotes(2)).toBeCloseTo(1.414, 2);
      expect(voting.calculateEffectiveVotes(5)).toBeCloseTo(2.236, 2);
      expect(voting.calculateEffectiveVotes(10)).toBeCloseTo(3.162, 2);
    });
  });

  describe('error handling', () => {
    it('should propagate zome call errors', async () => {
      (client.callZome as any).mockRejectedValue(new Error('Network error'));

      await expect(
        voting.castVote({
          proposal: mockProposalHash,
          choice: 'Yes',
        })
      ).rejects.toThrow('Network error');
    });
  });
});
