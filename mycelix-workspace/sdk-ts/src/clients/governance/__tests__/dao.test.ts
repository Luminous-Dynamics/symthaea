// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * DAO Client Tests
 *
 * Verifies zome call arguments, response mapping, membership operations,
 * permission checks, and statistics for the DAOClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { DAOClient } from '../dao';
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

const DAO_ENTRY = {
  name: 'Community Governance',
  description: 'Decentralized decision-making',
  charter: 'All decisions require 2/3 majority',
  charter_hash: 'hash-abc',
  founder_did: 'did:mycelix:alice',
  treasury_id: 'treasury-1',
  default_voting_period_hours: 168,
  default_quorum: 0.33,
  default_threshold: 0.51,
  member_count: 42,
  total_voting_power: 100.0,
  active: true,
  created_at: 1708200000,
  updated_at: 1708300000,
};

const MEMBERSHIP_ENTRY = {
  dao_id: 'dao-1',
  member_did: 'did:mycelix:alice',
  member_pub_key: 'uhCAk123',
  role: 'Founder',
  voting_power: 1.0,
  reputation_score: 0.95,
  active: true,
  joined_at: 1708200000,
  last_active_at: 1708400000,
};

// ============================================================================
// TESTS
// ============================================================================

describe('DAOClient', () => {
  let client: DAOClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new DAOClient(mockAppClient);
  });

  // --------------------------------------------------------------------------
  // Initialization
  // --------------------------------------------------------------------------

  describe('initialization', () => {
    it('should create client with default config', () => {
      expect(client).toBeInstanceOf(DAOClient);
    });

    it('should use governance role and dao zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(DAO_ENTRY)
      );

      await client.createDAO({
        name: 'Test',
        description: 'Test DAO',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'governance',
          zome_name: 'dao',
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // DAO CRUD
  // --------------------------------------------------------------------------

  describe('createDAO', () => {
    it('should pass snake_case payload with defaults', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(DAO_ENTRY)
      );

      await client.createDAO({
        name: 'Community Governance',
        description: 'Decentralized decision-making',
        charter: 'All decisions require 2/3 majority',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_dao',
          payload: {
            name: 'Community Governance',
            description: 'Decentralized decision-making',
            charter: 'All decisions require 2/3 majority',
            charter_hash: undefined,
            default_voting_period_hours: 168,
            default_quorum: 0.33,
            default_threshold: 0.51,
          },
        })
      );
    });

    it('should map response to DAO with camelCase', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(DAO_ENTRY)
      );

      const result = await client.createDAO({
        name: 'Community Governance',
        description: 'Decentralized decision-making',
      });

      expect(result.name).toBe('Community Governance');
      expect(result.founderDid).toBe('did:mycelix:alice');
      expect(result.treasuryId).toBe('treasury-1');
      expect(result.defaultVotingPeriodHours).toBe(168);
      expect(result.defaultQuorum).toBe(0.33);
      expect(result.defaultThreshold).toBe(0.51);
      expect(result.memberCount).toBe(42);
      expect(result.active).toBe(true);
    });

    it('should pass custom governance parameters', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(DAO_ENTRY)
      );

      await client.createDAO({
        name: 'Test',
        description: 'Test',
        defaultVotingPeriodHours: 72,
        defaultQuorum: 0.5,
        defaultThreshold: 0.67,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: expect.objectContaining({
            default_voting_period_hours: 72,
            default_quorum: 0.5,
            default_threshold: 0.67,
          }),
        })
      );
    });
  });

  describe('getDAO', () => {
    it('should return DAO for valid ID', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(DAO_ENTRY)
      );

      const result = await client.getDAO('dao-1');
      expect(result).not.toBeNull();
      expect(result!.name).toBe('Community Governance');
    });

    it('should return null for missing DAO', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getDAO('nonexistent');
      expect(result).toBeNull();
    });
  });

  describe('updateDAO', () => {
    it('should pass dao_id and snake_case updates', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...DAO_ENTRY, name: 'Updated DAO' })
      );

      const result = await client.updateDAO({
        daoId: 'dao-1',
        name: 'Updated DAO',
        defaultQuorum: 0.5,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'update_dao',
          payload: {
            dao_id: 'dao-1',
            name: 'Updated DAO',
            description: undefined,
            charter: undefined,
            treasury_id: undefined,
            default_voting_period_hours: undefined,
            default_quorum: 0.5,
            default_threshold: undefined,
          },
        })
      );
      expect(result.name).toBe('Updated DAO');
    });
  });

  describe('listDAOs', () => {
    it('should pass filter with snake_case fields', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(DAO_ENTRY),
      ]);

      await client.listDAOs({
        founderDid: 'did:mycelix:alice',
        active: true,
        nameContains: 'Community',
        limit: 10,
        offset: 0,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'list_daos',
          payload: {
            founder_did: 'did:mycelix:alice',
            active: true,
            name_contains: 'Community',
            limit: 10,
            offset: 0,
          },
        })
      );
    });

    it('should map array of records', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(DAO_ENTRY),
        mockRecord({ ...DAO_ENTRY, name: 'Second DAO' }),
      ]);

      const result = await client.listDAOs();
      expect(result).toHaveLength(2);
      expect(result[1].name).toBe('Second DAO');
    });
  });

  describe('getDAOsByMember', () => {
    it('should pass member DID to zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(DAO_ENTRY),
      ]);

      const result = await client.getDAOsByMember('did:mycelix:alice');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_daos_by_member',
          payload: 'did:mycelix:alice',
        })
      );
      expect(result).toHaveLength(1);
    });
  });

  describe('archiveDAO', () => {
    it('should send dao ID as payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      await client.archiveDAO('dao-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'archive_dao',
          payload: 'dao-1',
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Membership Operations
  // --------------------------------------------------------------------------

  describe('joinDAO', () => {
    it('should pass dao_id with default voting power', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(MEMBERSHIP_ENTRY)
      );

      await client.joinDAO({ daoId: 'dao-1' });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'join_dao',
          payload: {
            dao_id: 'dao-1',
            initial_voting_power: 0,
          },
        })
      );
    });

    it('should map response to DAOMembership', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(MEMBERSHIP_ENTRY)
      );

      const result = await client.joinDAO({ daoId: 'dao-1' });

      expect(result.daoId).toBe('dao-1');
      expect(result.memberDid).toBe('did:mycelix:alice');
      expect(result.role).toBe('Founder');
      expect(result.votingPower).toBe(1.0);
      expect(result.reputationScore).toBe(0.95);
      expect(result.active).toBe(true);
    });

    it('should pass custom initial voting power', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(MEMBERSHIP_ENTRY)
      );

      await client.joinDAO({ daoId: 'dao-1', initialVotingPower: 5.0 });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: expect.objectContaining({
            initial_voting_power: 5.0,
          }),
        })
      );
    });
  });

  describe('leaveDAO', () => {
    it('should send dao ID as payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      await client.leaveDAO('dao-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'leave_dao',
          payload: 'dao-1',
        })
      );
    });
  });

  describe('getMembers', () => {
    it('should return array of memberships', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(MEMBERSHIP_ENTRY),
        mockRecord({ ...MEMBERSHIP_ENTRY, member_did: 'did:mycelix:bob', role: 'Member' }),
      ]);

      const result = await client.getMembers('dao-1');
      expect(result).toHaveLength(2);
      expect(result[0].memberDid).toBe('did:mycelix:alice');
      expect(result[1].role).toBe('Member');
    });
  });

  describe('getMembership', () => {
    it('should pass dao_id and member_did', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(MEMBERSHIP_ENTRY)
      );

      const result = await client.getMembership('dao-1', 'did:mycelix:alice');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_membership',
          payload: {
            dao_id: 'dao-1',
            member_did: 'did:mycelix:alice',
          },
        })
      );
      expect(result).not.toBeNull();
      expect(result!.role).toBe('Founder');
    });

    it('should return null for non-member', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getMembership('dao-1', 'did:mycelix:nobody');
      expect(result).toBeNull();
    });
  });

  describe('getMyMembership', () => {
    it('should pass dao ID to zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(MEMBERSHIP_ENTRY)
      );

      const result = await client.getMyMembership('dao-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_my_membership',
          payload: 'dao-1',
        })
      );
      expect(result).not.toBeNull();
    });

    it('should return null when not a member', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getMyMembership('dao-1');
      expect(result).toBeNull();
    });
  });

  describe('isMember', () => {
    it('should return boolean from zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(true);

      const result = await client.isMember('dao-1', 'did:mycelix:alice');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'is_member',
          payload: {
            dao_id: 'dao-1',
            member_did: 'did:mycelix:alice',
          },
        })
      );
      expect(result).toBe(true);
    });
  });

  describe('updateMemberRole', () => {
    it('should pass dao_id, member_did, and new_role', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...MEMBERSHIP_ENTRY, role: 'Admin' })
      );

      const result = await client.updateMemberRole('dao-1', 'did:mycelix:bob', 'Admin');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'update_member_role',
          payload: {
            dao_id: 'dao-1',
            member_did: 'did:mycelix:bob',
            new_role: 'Admin',
          },
        })
      );
      expect(result.role).toBe('Admin');
    });
  });

  describe('updateMemberVotingPower', () => {
    it('should pass dao_id, member_did, and voting_power', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...MEMBERSHIP_ENTRY, voting_power: 2.5 })
      );

      const result = await client.updateMemberVotingPower('dao-1', 'did:mycelix:bob', 2.5);

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'update_member_voting_power',
          payload: {
            dao_id: 'dao-1',
            member_did: 'did:mycelix:bob',
            voting_power: 2.5,
          },
        })
      );
      expect(result.votingPower).toBe(2.5);
    });
  });

  describe('removeMember', () => {
    it('should pass dao_id and member_did', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      await client.removeMember('dao-1', 'did:mycelix:bob');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'remove_member',
          payload: {
            dao_id: 'dao-1',
            member_did: 'did:mycelix:bob',
          },
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Permission Checks
  // --------------------------------------------------------------------------

  describe('isAdmin', () => {
    it('should return true for Admin role', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...MEMBERSHIP_ENTRY, role: 'Admin' })
      );

      const result = await client.isAdmin('dao-1', 'did:mycelix:bob');
      expect(result).toBe(true);
    });

    it('should return true for Founder role', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...MEMBERSHIP_ENTRY, role: 'Founder' })
      );

      const result = await client.isAdmin('dao-1', 'did:mycelix:alice');
      expect(result).toBe(true);
    });

    it('should return false for Member role', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...MEMBERSHIP_ENTRY, role: 'Member' })
      );

      const result = await client.isAdmin('dao-1', 'did:mycelix:bob');
      expect(result).toBe(false);
    });

    it('should return false for non-member', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.isAdmin('dao-1', 'did:mycelix:nobody');
      expect(result).toBe(false);
    });
  });

  describe('isFounder', () => {
    it('should return true for Founder role', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...MEMBERSHIP_ENTRY, role: 'Founder' })
      );

      const result = await client.isFounder('dao-1', 'did:mycelix:alice');
      expect(result).toBe(true);
    });

    it('should return false for Admin role', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...MEMBERSHIP_ENTRY, role: 'Admin' })
      );

      const result = await client.isFounder('dao-1', 'did:mycelix:bob');
      expect(result).toBe(false);
    });
  });

  // --------------------------------------------------------------------------
  // Statistics
  // --------------------------------------------------------------------------

  describe('getDAOStats', () => {
    it('should return stats from zome', async () => {
      const stats = {
        memberCount: 42,
        totalVotingPower: 100,
        activeProposals: 3,
        totalProposals: 15,
        treasuryBalance: 50000,
      };
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(stats);

      const result = await client.getDAOStats('dao-1');

      expect(result.memberCount).toBe(42);
      expect(result.activeProposals).toBe(3);
    });
  });

  describe('getMemberActivity', () => {
    it('should pass dao_id and member_did', async () => {
      const activity = {
        proposalsCreated: 5,
        votesCast: 12,
        participationRate: 0.85,
        lastActiveAt: 1708400000,
      };
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(activity);

      const result = await client.getMemberActivity('dao-1', 'did:mycelix:alice');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_member_activity',
          payload: {
            dao_id: 'dao-1',
            member_did: 'did:mycelix:alice',
          },
        })
      );
      expect(result.proposalsCreated).toBe(5);
      expect(result.participationRate).toBe(0.85);
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
        client.createDAO({ name: 'Test', description: 'Test' })
      ).rejects.toThrow();
    });

    it('should propagate zome errors on join', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('DAO not found')
      );

      await expect(
        client.joinDAO({ daoId: 'nonexistent' })
      ).rejects.toThrow();
    });
  });
});
