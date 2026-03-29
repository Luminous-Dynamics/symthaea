// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Councils Client Tests
 *
 * Verifies zome call arguments, response mapping, membership, reflection,
 * decision tracking, and holonic queries for the CouncilsClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { CouncilsClient } from '../councils';
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
    entry: { Present: { entry } },
    signed_action: { hashed: { hash: new Uint8Array(32) } },
  };
}

const COUNCIL_ENTRY = {
  id: 'council-1',
  name: 'Water Commons Council',
  purpose: 'Steward local water resources',
  council_type: { type: 'Domain', domain: 'water' },
  parent_council_id: null,
  phi_threshold: 0.3,
  quorum: 0.5,
  supermajority: 0.67,
  can_spawn_children: true,
  max_delegation_depth: 3,
  signing_committee_id: null,
  status: 'Active',
  created_at: 1708200000,
  last_activity: 1708400000,
};

const MEMBERSHIP_ENTRY = {
  id: 'mem-1',
  council_id: 'council-1',
  member_did: 'did:mycelix:alice',
  role: 'Facilitator',
  phi_score: 0.75,
  voting_weight: 1.5,
  can_delegate: true,
  status: 'Active',
  joined_at: 1708200000,
  last_participation: 1708400000,
};

const REFLECTION_ENTRY = {
  id: 'ref-1',
  council_id: 'council-1',
  reflector_did: 'did:mycelix:alice',
  harmony_scores: { Compassion: 0.8, Justice: 0.9, Wisdom: 0.7 },
  narrative: 'The council showed strong collaborative spirit',
  insights: ['Good participation', 'Needs more diversity'],
  timestamp: 1708400000,
};

const DECISION_ENTRY = {
  id: 'dec-1',
  council_id: 'council-1',
  proposal_id: 'proposal-1',
  title: 'Approve water testing budget',
  content: 'Allocate 2000 MYC for quarterly water testing',
  decision_type: 'Resource',
  votes_for: 8,
  votes_against: 2,
  abstentions: 1,
  phi_weighted_result: 0.82,
  passed: true,
  status: 'Approved',
  created_at: 1708300000,
  executed_at: null,
};

// ============================================================================
// TESTS
// ============================================================================

describe('CouncilsClient', () => {
  let client: CouncilsClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new CouncilsClient(mockAppClient);
  });

  // --------------------------------------------------------------------------
  // Initialization
  // --------------------------------------------------------------------------

  describe('initialization', () => {
    it('should create client with default config', () => {
      expect(client).toBeInstanceOf(CouncilsClient);
    });

    it('should use governance role and councils zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(COUNCIL_ENTRY)
      );

      await client.createCouncil({
        name: 'Test',
        purpose: 'Testing',
        councilType: { type: 'Advisory' },
        phiThreshold: 0.3,
        quorum: 0.5,
        supermajority: 0.67,
        canSpawnChildren: false,
        maxDelegationDepth: 1,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'governance',
          zome_name: 'councils',
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Council CRUD
  // --------------------------------------------------------------------------

  describe('createCouncil', () => {
    it('should pass snake_case payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(COUNCIL_ENTRY)
      );

      await client.createCouncil({
        name: 'Water Commons Council',
        purpose: 'Steward local water resources',
        councilType: { type: 'Domain', domain: 'water' },
        phiThreshold: 0.3,
        quorum: 0.5,
        supermajority: 0.67,
        canSpawnChildren: true,
        maxDelegationDepth: 3,
        signingCommitteeId: 'committee-1',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_council',
          payload: {
            name: 'Water Commons Council',
            purpose: 'Steward local water resources',
            council_type: { type: 'Domain', domain: 'water' },
            parent_council_id: undefined,
            phi_threshold: 0.3,
            quorum: 0.5,
            supermajority: 0.67,
            can_spawn_children: true,
            max_delegation_depth: 3,
            signing_committee_id: 'committee-1',
          },
        })
      );
    });

    it('should map response to Council with camelCase', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(COUNCIL_ENTRY)
      );

      const result = await client.createCouncil({
        name: 'Water Commons Council',
        purpose: 'Steward local water resources',
        councilType: { type: 'Domain', domain: 'water' },
        phiThreshold: 0.3,
        quorum: 0.5,
        supermajority: 0.67,
        canSpawnChildren: true,
        maxDelegationDepth: 3,
      });

      expect(result.name).toBe('Water Commons Council');
      expect(result.councilType).toEqual({ type: 'Domain', domain: 'water' });
      expect(result.phiThreshold).toBe(0.3);
      expect(result.quorum).toBe(0.5);
      expect(result.supermajority).toBe(0.67);
      expect(result.canSpawnChildren).toBe(true);
      expect(result.maxDelegationDepth).toBe(3);
      expect(result.status).toBe('Active');
    });
  });

  describe('getCouncil', () => {
    it('should return council for valid ID', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(COUNCIL_ENTRY)
      );

      const result = await client.getCouncil('council-1');
      expect(result).not.toBeNull();
      expect(result!.name).toBe('Water Commons Council');
    });

    it('should return null for missing council', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getCouncil('nonexistent');
      expect(result).toBeNull();
    });
  });

  describe('getAllCouncils', () => {
    it('should return array of councils', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(COUNCIL_ENTRY),
        mockRecord({ ...COUNCIL_ENTRY, name: 'Food Council' }),
      ]);

      const result = await client.getAllCouncils();
      expect(result).toHaveLength(2);
      expect(result[1].name).toBe('Food Council');
    });
  });

  describe('getChildCouncils', () => {
    it('should pass council ID to zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord({ ...COUNCIL_ENTRY, parent_council_id: 'council-1', name: 'Sub-Council' }),
      ]);

      const result = await client.getChildCouncils('council-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_child_councils',
          payload: 'council-1',
        })
      );
      expect(result).toHaveLength(1);
    });
  });

  // --------------------------------------------------------------------------
  // Membership
  // --------------------------------------------------------------------------

  describe('joinCouncil', () => {
    it('should pass snake_case payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(MEMBERSHIP_ENTRY)
      );

      const result = await client.joinCouncil({
        councilId: 'council-1',
        memberDid: 'did:mycelix:alice',
        role: 'Facilitator',
        phiScore: 0.75,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'join_council',
          payload: {
            council_id: 'council-1',
            member_did: 'did:mycelix:alice',
            role: 'Facilitator',
            phi_score: 0.75,
          },
        })
      );
      expect(result.memberDid).toBe('did:mycelix:alice');
      expect(result.phiScore).toBe(0.75);
      expect(result.votingWeight).toBe(1.5);
      expect(result.canDelegate).toBe(true);
    });
  });

  describe('getCouncilMembers', () => {
    it('should return array of memberships', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(MEMBERSHIP_ENTRY),
        mockRecord({ ...MEMBERSHIP_ENTRY, member_did: 'did:mycelix:bob', role: 'Member' }),
      ]);

      const result = await client.getCouncilMembers('council-1');
      expect(result).toHaveLength(2);
    });
  });

  describe('getMemberCouncils', () => {
    it('should pass member DID to zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);

      await client.getMemberCouncils('did:mycelix:alice');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_member_councils',
          payload: 'did:mycelix:alice',
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Reflection
  // --------------------------------------------------------------------------

  describe('reflectOnCouncil', () => {
    it('should pass snake_case payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(REFLECTION_ENTRY)
      );

      const result = await client.reflectOnCouncil({
        councilId: 'council-1',
        reflectorDid: 'did:mycelix:alice',
        harmonyScores: { Compassion: 0.8, Justice: 0.9 },
        narrative: 'Good session',
        insights: ['Productive'],
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'reflect_on_council',
          payload: {
            council_id: 'council-1',
            reflector_did: 'did:mycelix:alice',
            harmony_scores: { Compassion: 0.8, Justice: 0.9 },
            narrative: 'Good session',
            insights: ['Productive'],
          },
        })
      );
      expect(result.reflectorDid).toBe('did:mycelix:alice');
      expect(result.harmonyScores).toHaveProperty('Compassion');
    });
  });

  describe('getCouncilReflections', () => {
    it('should return array of reflections', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(REFLECTION_ENTRY),
      ]);

      const result = await client.getCouncilReflections('council-1');
      expect(result).toHaveLength(1);
      expect(result[0].narrative).toBe('The council showed strong collaborative spirit');
    });
  });

  describe('getLatestReflection', () => {
    it('should return reflection for valid council', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(REFLECTION_ENTRY)
      );

      const result = await client.getLatestReflection('council-1');
      expect(result).not.toBeNull();
      expect(result!.insights).toHaveLength(2);
    });

    it('should return null when no reflections', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getLatestReflection('council-1');
      expect(result).toBeNull();
    });
  });

  describe('getHolonicTree', () => {
    it('should call zome with null payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);

      await client.getHolonicTree();

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_holonic_tree',
          payload: null,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Decisions
  // --------------------------------------------------------------------------

  describe('recordDecision', () => {
    it('should pass snake_case payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(DECISION_ENTRY)
      );

      const result = await client.recordDecision({
        councilId: 'council-1',
        proposalId: 'proposal-1',
        title: 'Approve water testing budget',
        content: 'Allocate 2000 MYC',
        decisionType: 'Resource',
        votesFor: 8,
        votesAgainst: 2,
        abstentions: 1,
        phiWeightedResult: 0.82,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'record_decision',
          payload: {
            council_id: 'council-1',
            proposal_id: 'proposal-1',
            title: 'Approve water testing budget',
            content: 'Allocate 2000 MYC',
            decision_type: 'Resource',
            votes_for: 8,
            votes_against: 2,
            abstentions: 1,
            phi_weighted_result: 0.82,
          },
        })
      );
      expect(result.title).toBe('Approve water testing budget');
      expect(result.decisionType).toBe('Resource');
      expect(result.phiWeightedResult).toBe(0.82);
      expect(result.passed).toBe(true);
    });
  });

  describe('getCouncilDecisions', () => {
    it('should return array of decisions', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(DECISION_ENTRY),
        mockRecord({ ...DECISION_ENTRY, title: 'Second decision', passed: false }),
      ]);

      const result = await client.getCouncilDecisions('council-1');
      expect(result).toHaveLength(2);
      expect(result[0].passed).toBe(true);
      expect(result[1].passed).toBe(false);
    });
  });

  // --------------------------------------------------------------------------
  // Error Handling
  // --------------------------------------------------------------------------

  describe('error handling', () => {
    it('should propagate zome errors on create', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Unauthorized')
      );

      await expect(
        client.createCouncil({
          name: 'Test',
          purpose: 'Test',
          councilType: { type: 'Advisory' },
          phiThreshold: 0.3,
          quorum: 0.5,
          supermajority: 0.67,
          canSpawnChildren: false,
          maxDelegationDepth: 1,
        })
      ).rejects.toThrow();
    });

    it('should propagate zome errors on join', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Phi threshold not met')
      );

      await expect(
        client.joinCouncil({
          councilId: 'council-1',
          memberDid: 'did:mycelix:bob',
          role: 'Member',
          phiScore: 0.1,
        })
      ).rejects.toThrow();
    });
  });
});
