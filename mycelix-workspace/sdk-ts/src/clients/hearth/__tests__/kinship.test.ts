/**
 * Kinship Client Tests
 *
 * Verifies zome call arguments and payload passthrough for the KinshipClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { KinshipClient } from '../kinship';

// ============================================================================
// MOCK HELPERS
// ============================================================================

function createMockClient() {
  return { callZome: vi.fn() } as any;
}

const HASH = new Uint8Array(32);
const AGENT = new Uint8Array(39);
const MOCK_RECORD = { entry: { Present: {} }, signed_action: { hashed: { hash: HASH } } };

// ============================================================================
// TESTS
// ============================================================================

describe('KinshipClient', () => {
  let client: KinshipClient;
  let mockClient: ReturnType<typeof createMockClient>;

  beforeEach(() => {
    mockClient = createMockClient();
    client = new KinshipClient(mockClient, { roleName: 'hearth', timeout: 30000 });
  });

  // --------------------------------------------------------------------------
  // Hearth Management
  // --------------------------------------------------------------------------

  describe('hearth management', () => {
    it('createHearth calls hearth_kinship/create_hearth with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { name: 'Our Hearth', description: 'A family space' };

      await client.createHearth(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'hearth',
          zome_name: 'hearth_kinship',
          fn_name: 'create_hearth',
          payload: input,
        })
      );
    });

    it('getMyHearths calls hearth_kinship/get_my_hearths with null payload', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getMyHearths();

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'hearth',
          zome_name: 'hearth_kinship',
          fn_name: 'get_my_hearths',
          payload: null,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Membership
  // --------------------------------------------------------------------------

  describe('membership', () => {
    it('inviteMember calls invite_member with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, invitee: AGENT, role: 'Member' };

      await client.inviteMember(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'hearth_kinship',
          fn_name: 'invite_member',
          payload: input,
        })
      );
    });

    it('acceptInvitation calls accept_invitation with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);

      await client.acceptInvitation(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'accept_invitation',
          payload: HASH,
        })
      );
    });

    it('declineInvitation calls decline_invitation with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);

      await client.declineInvitation(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'decline_invitation',
          payload: HASH,
        })
      );
    });

    it('leaveHearth calls leave_hearth with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);

      await client.leaveHearth(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'leave_hearth',
          payload: HASH,
        })
      );
    });

    it('updateMemberRole calls update_member_role with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, member: AGENT, new_role: 'Guardian' };

      await client.updateMemberRole(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'update_member_role',
          payload: input,
        })
      );
    });

    it('getHearthMembers calls get_hearth_members with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getHearthMembers(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_hearth_members',
          payload: HASH,
        })
      );
    });

    it('isGuardian calls is_guardian with hash and returns boolean', async () => {
      mockClient.callZome.mockResolvedValueOnce(true);

      const result = await client.isGuardian(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'is_guardian',
          payload: HASH,
        })
      );
      expect(result).toBe(true);
    });

    it('getCallerVoteWeight calls get_caller_vote_weight with hash and returns number', async () => {
      mockClient.callZome.mockResolvedValueOnce(3);

      const result = await client.getCallerVoteWeight(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_caller_vote_weight',
          payload: HASH,
        })
      );
      expect(result).toBe(3);
    });

    it('getCallerRole calls get_caller_role with hash and returns role or null', async () => {
      mockClient.callZome.mockResolvedValueOnce('Guardian');

      const result = await client.getCallerRole(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_caller_role',
          payload: HASH,
        })
      );
      expect(result).toBe('Guardian');
    });

    it('getActiveMemberCount calls get_active_member_count with hash and returns number', async () => {
      mockClient.callZome.mockResolvedValueOnce(5);

      const result = await client.getActiveMemberCount(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_active_member_count',
          payload: HASH,
        })
      );
      expect(result).toBe(5);
    });
  });

  // --------------------------------------------------------------------------
  // Kinship Bonds
  // --------------------------------------------------------------------------

  describe('kinship bonds', () => {
    it('createBond calls create_kinship_bond with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, members: [AGENT], bond_type: 'Sibling' };

      await client.createBond(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_kinship_bond',
          payload: input,
        })
      );
    });

    it('tendBond calls tend_bond with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { bond_hash: HASH, action: 'nurture', note: 'Shared a meal' };

      await client.tendBond(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'tend_bond',
          payload: input,
        })
      );
    });

    it('getBondHealth calls get_bond_health with input and returns number', async () => {
      mockClient.callZome.mockResolvedValueOnce(0.85);
      const input = { bond_hash: HASH };

      const result = await client.getBondHealth(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_bond_health',
          payload: input,
        })
      );
      expect(result).toBe(0.85);
    });

    it('getKinshipGraph calls get_kinship_graph with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getKinshipGraph(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_kinship_graph',
          payload: HASH,
        })
      );
    });

    it('getNeglectedBonds calls get_neglected_bonds with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getNeglectedBonds(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_neglected_bonds',
          payload: HASH,
        })
      );
    });

    it('getBondSnapshots calls get_bond_snapshots with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([]);

      await client.getBondSnapshots(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_bond_snapshots',
          payload: HASH,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Weekly Digests
  // --------------------------------------------------------------------------

  describe('weekly digests', () => {
    it('createWeeklyDigest calls create_weekly_digest with digest', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const digest = { hearth_hash: HASH, epoch_start: 1708200000, epoch_end: 1708804800 };

      await client.createWeeklyDigest(digest as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_weekly_digest',
          payload: digest,
        })
      );
    });

    it('getWeeklyDigests calls get_weekly_digests with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getWeeklyDigests(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_weekly_digests',
          payload: HASH,
        })
      );
    });
  });
});
