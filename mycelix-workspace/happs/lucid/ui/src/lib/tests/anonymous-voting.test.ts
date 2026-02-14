/**
 * Anonymous Voting Service Tests
 *
 * Tests for ZK-based anonymous voting functionality.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ValidationVoteType } from '@mycelix/lucid-client';

// Use vi.hoisted to ensure mockInvoke is available when vi.mock runs
const { mockInvoke } = vi.hoisted(() => {
  return { mockInvoke: vi.fn() };
});

vi.mock('@tauri-apps/api/core', () => ({
  invoke: mockInvoke,
}));

// Mock sessionStorage (anonymous-voting uses sessionStorage for security)
const sessionStorageMock = {
  store: {} as Record<string, string>,
  getItem: vi.fn((key: string) => sessionStorageMock.store[key] || null),
  setItem: vi.fn((key: string, value: string) => {
    sessionStorageMock.store[key] = value;
  }),
  removeItem: vi.fn((key: string) => {
    delete sessionStorageMock.store[key];
  }),
  clear: vi.fn(() => {
    sessionStorageMock.store = {};
  }),
};

Object.defineProperty(global, 'sessionStorage', {
  value: sessionStorageMock,
});

// Mock crypto.getRandomValues
Object.defineProperty(global, 'crypto', {
  value: {
    getRandomValues: (arr: Uint8Array) => {
      for (let i = 0; i < arr.length; i++) {
        arr[i] = Math.floor(Math.random() * 256);
      }
      return arr;
    },
  },
});

// Import after mocking
import {
  isZkpReady,
  generateAnonymousBeliefProof,
  generateReputationProof,
  verifyProof,
  hashBeliefContent,
  castAnonymousVote,
  checkVotingEligibility,
  getVoteAnalytics,
} from '../services/anonymous-voting';

describe('Anonymous Voting Service', () => {
  beforeEach(() => {
    mockInvoke.mockReset();
    sessionStorageMock.clear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('isZkpReady', () => {
    it('should return true when ZKP system is ready', async () => {
      mockInvoke.mockResolvedValueOnce(true);

      const result = await isZkpReady();

      expect(result).toBe(true);
      expect(mockInvoke).toHaveBeenCalledWith('zkp_ready');
    });

    it('should return false when ZKP system is not available', async () => {
      mockInvoke.mockRejectedValueOnce(new Error('ZKP not available'));

      const result = await isZkpReady();

      expect(result).toBe(false);
    });
  });

  describe('generateAnonymousBeliefProof', () => {
    it('should generate proof via Tauri', async () => {
      const mockProof = {
        proof_json: '{"proof":"data"}',
        author_commitment: 'commitment123',
        belief_hash: 'hash456',
      };
      mockInvoke.mockResolvedValueOnce(mockProof);

      const result = await generateAnonymousBeliefProof('belief-hash', 'agent-secret');

      expect(result).toEqual({
        proofJson: '{"proof":"data"}',
        authorCommitment: 'commitment123',
        beliefHash: 'hash456',
      });
    });

    it('should return null on error', async () => {
      mockInvoke.mockRejectedValueOnce(new Error('Proof generation failed'));

      const result = await generateAnonymousBeliefProof('belief-hash', 'agent-secret');

      expect(result).toBeNull();
    });
  });

  describe('generateReputationProof', () => {
    it('should generate reputation proof', async () => {
      const mockProof = {
        proof_json: '{"proof":"reputation"}',
        reputation_commitment: 'rep-commit',
        min_threshold: 50,
      };
      mockInvoke.mockResolvedValueOnce(mockProof);

      const result = await generateReputationProof(100, 50);

      expect(result).toEqual({
        proofJson: '{"proof":"reputation"}',
        reputationCommitment: 'rep-commit',
        minThreshold: 50,
      });
    });

    it('should return null on error', async () => {
      mockInvoke.mockRejectedValueOnce(new Error('Reputation proof failed'));

      const result = await generateReputationProof(100, 50);

      expect(result).toBeNull();
    });
  });

  describe('verifyProof', () => {
    it('should verify valid proof', async () => {
      mockInvoke.mockResolvedValueOnce({ valid: true, error: null });

      const result = await verifyProof('anonymous_belief', '{"proof":"data"}');

      expect(result.valid).toBe(true);
      expect(result.error).toBeFalsy(); // error can be null or undefined
    });

    it('should return invalid for failed verification', async () => {
      mockInvoke.mockResolvedValueOnce({ valid: false, error: 'Invalid proof' });

      const result = await verifyProof('anonymous_belief', '{"bad":"proof"}');

      expect(result.valid).toBe(false);
      expect(result.error).toBe('Invalid proof');
    });

    it('should handle verification errors', async () => {
      mockInvoke.mockRejectedValueOnce(new Error('Verification error'));

      const result = await verifyProof('anonymous_belief', '{"proof":"data"}');

      expect(result.valid).toBe(false);
      expect(result.error).toBeDefined();
    });
  });

  describe('hashBeliefContent', () => {
    it('should hash content via Tauri', async () => {
      mockInvoke.mockResolvedValueOnce('abc123hash');

      const result = await hashBeliefContent('test content');

      expect(result).toBe('abc123hash');
      expect(mockInvoke).toHaveBeenCalledWith('hash_belief_content', { content: 'test content' });
    });

    it('should fallback to local hashing on error', async () => {
      mockInvoke.mockRejectedValueOnce(new Error('Hash failed'));

      const result = await hashBeliefContent('test content');

      expect(result).toBeDefined();
      expect(typeof result).toBe('string');
    });
  });

  describe('castAnonymousVote', () => {
    it('should return error when ZKP not available', async () => {
      mockInvoke.mockRejectedValueOnce(new Error('ZKP not available'));

      const result = await castAnonymousVote({
        beliefHash: 'belief123',
        voteType: ValidationVoteType.Corroborate,
      });

      expect(result.success).toBe(false);
      expect(result.error).toBe('ZKP system not available');
    });

    it('should cast anonymous vote successfully', async () => {
      // Mock the full flow - includes hashBeliefContent call in submitProofAttestation
      mockInvoke
        .mockResolvedValueOnce(true) // isZkpReady
        .mockResolvedValueOnce({
          proof_json: '{"proof":"data"}',
          author_commitment: 'commitment',
          belief_hash: 'belief123',
        }) // generateAnonymousBeliefProof
        .mockResolvedValueOnce('hash123'); // hashBeliefContent in submitProofAttestation

      const result = await castAnonymousVote({
        beliefHash: 'belief123',
        voteType: ValidationVoteType.Corroborate,
      });

      expect(result.success).toBe(true);
      expect(result.attestationId).toBeDefined();
    });
  });

  describe('checkVotingEligibility', () => {
    it('should return eligible by default', async () => {
      const result = await checkVotingEligibility('belief123');

      expect(result.eligible).toBe(true);
    });

    it('should detect already voted', async () => {
      // Store a vote commitment
      const agentSecret = 'test-secret';
      sessionStorageMock.store['lucid_agent_secret'] = agentSecret;

      // The hasAlreadyVoted function calls hashBeliefContent which will fall back to local hash
      // if the Tauri invoke fails. We need to store the hash that localHashContent would return.
      // Compute localHashContent('belief123test-secret') - simpler: mock the invoke to return the expected hash
      const expectedVoteKey = 'mocked-vote-hash';
      mockInvoke.mockResolvedValue(expectedVoteKey); // Use mockResolvedValue for all calls
      sessionStorageMock.store['anonymous_vote_commitments'] = JSON.stringify([expectedVoteKey]);

      const result = await checkVotingEligibility('belief123');

      expect(result.eligible).toBe(false);
      expect(result.reason).toContain('already voted');
    });
  });

  describe('getVoteAnalytics', () => {
    it('should return empty analytics when no data', async () => {
      const result = await getVoteAnalytics('belief123');

      expect(result.totalVotes).toBe(0);
      expect(result.anonymousVotes).toBe(0);
      expect(result.attestedVotes).toBe(0);
      expect(result.reputationProvenVotes).toBe(0);
    });

    it('should count attestations correctly', async () => {
      sessionStorageMock.store['lucid_attestations'] = JSON.stringify([
        {
          id: 'att1',
          proof_type: 'anonymous_belief',
          subject_hash: 'belief123',
          verified: true,
        },
        {
          id: 'att2',
          proof_type: 'reputation_range',
          subject_hash: 'belief123',
          verified: true,
        },
        {
          id: 'att3',
          proof_type: 'anonymous_belief',
          subject_hash: 'other-belief',
          verified: true,
        },
      ]);

      const result = await getVoteAnalytics('belief123');

      expect(result.totalVotes).toBe(2);
      expect(result.anonymousVotes).toBe(1);
      expect(result.reputationProvenVotes).toBe(1);
      expect(result.attestedVotes).toBe(2);
    });
  });
});

describe('Vote Type Handling', () => {
  beforeEach(() => {
    mockInvoke.mockReset();
    sessionStorageMock.clear();
  });

  it('should handle all vote types', async () => {
    const voteTypes = [
      ValidationVoteType.Corroborate,
      ValidationVoteType.Plausible,
      ValidationVoteType.Abstain,
      ValidationVoteType.Implausible,
      ValidationVoteType.Contradict,
    ];

    for (const voteType of voteTypes) {
      // Reset mocks for each iteration
      mockInvoke.mockReset();

      // Setup mocks for full flow: isZkpReady, generateAnonymousBeliefProof, hashBeliefContent
      mockInvoke
        .mockResolvedValueOnce(true) // isZkpReady
        .mockResolvedValueOnce({
          proof_json: '{"proof":"data"}',
          author_commitment: `commitment-${voteType}`,
          belief_hash: 'belief123',
        }) // generateAnonymousBeliefProof
        .mockResolvedValueOnce(`hash-${voteType}`); // hashBeliefContent in submitProofAttestation

      const result = await castAnonymousVote({
        beliefHash: 'belief123',
        voteType,
      });

      expect(result.success).toBe(true);
    }
  });
});
