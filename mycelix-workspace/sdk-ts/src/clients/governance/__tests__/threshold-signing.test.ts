/**
 * Threshold-Signing Client Tests
 *
 * Verifies zome call arguments, response mapping, and DKG ceremony flow
 * for the ThresholdSigningClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ThresholdSigningClient } from '../threshold-signing';
import type { AppClient } from '@holochain/client';

// ============================================================================
// MOCK HELPERS
// ============================================================================

/** Create a mock AppClient with a vi.fn() callZome */
function createMockClient(): AppClient {
  return {
    callZome: vi.fn(),
  } as unknown as AppClient;
}

/** Wrap entry data in a HolochainRecord-like structure */
function mockRecord(entry: Record<string, unknown>) {
  return {
    entry: { Present: { entry } },
    signed_action: { hashed: { hash: new Uint8Array(32) } },
  };
}

const COMMITTEE_ENTRY = {
  id: 'committee-1',
  name: 'Treasury Signers',
  threshold: 2,
  member_count: 3,
  phase: 'Registration',
  public_key: null,
  commitments: [],
  scope: 'Treasury',
  created_at: 1708300000,
  active: true,
  epoch: 1,
  min_phi: 0.3,
};

const MEMBER_ENTRY = {
  committee_id: 'committee-1',
  participant_id: 1,
  agent: 'uhCAkSomeAgent',
  member_did: 'did:mycelix:alice',
  trust_score: 0.9,
  public_share: null,
  vss_commitment: null,
  deal_submitted: false,
  qualified: false,
  registered_at: 1708300100,
};

const SIGNATURE_ENTRY = {
  id: 'sig-1',
  committee_id: 'committee-1',
  signed_content_hash: [1, 2, 3, 4],
  signed_content_description: 'Treasury allocation #42',
  signature: [5, 6, 7, 8],
  signer_count: 2,
  signers: [1, 3],
  verified: true,
  signed_at: 1708300200,
};

// ============================================================================
// TESTS
// ============================================================================

describe('ThresholdSigningClient', () => {
  let client: ThresholdSigningClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new ThresholdSigningClient(mockAppClient);
  });

  // --------------------------------------------------------------------------
  // Initialization
  // --------------------------------------------------------------------------

  describe('initialization', () => {
    it('should create client with default config', () => {
      expect(client).toBeInstanceOf(ThresholdSigningClient);
    });

    it('should use governance as default role', () => {
      // Verify by making a call and checking the role_name
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(COMMITTEE_ENTRY)
      );

      client.createCommittee({
        name: 'Test',
        threshold: 2,
        memberCount: 3,
        scope: 'All',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'governance',
          zome_name: 'threshold_signing',
        })
      );
    });

    it('should accept custom role name', async () => {
      const custom = new ThresholdSigningClient(mockAppClient, { roleName: 'custom_role' });
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(COMMITTEE_ENTRY),
      ]);

      await custom.getAllCommittees();

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({ role_name: 'custom_role' })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Committee Management
  // --------------------------------------------------------------------------

  describe('createCommittee', () => {
    it('should pass correct payload with snake_case fields', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(COMMITTEE_ENTRY)
      );

      await client.createCommittee({
        name: 'Treasury Signers',
        threshold: 2,
        memberCount: 3,
        scope: 'Treasury',
        minPhi: 0.3,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_committee',
          payload: {
            name: 'Treasury Signers',
            threshold: 2,
            member_count: 3,
            scope: 'Treasury',
            min_phi: 0.3,
          },
        })
      );
    });

    it('should default minPhi to null', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(COMMITTEE_ENTRY)
      );

      await client.createCommittee({
        name: 'Test',
        threshold: 2,
        memberCount: 3,
        scope: 'All',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: expect.objectContaining({ min_phi: null }),
        })
      );
    });

    it('should map response to SigningCommittee', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(COMMITTEE_ENTRY)
      );

      const result = await client.createCommittee({
        name: 'Treasury Signers',
        threshold: 2,
        memberCount: 3,
        scope: 'Treasury',
      });

      expect(result.id).toBe('committee-1');
      expect(result.name).toBe('Treasury Signers');
      expect(result.threshold).toBe(2);
      expect(result.memberCount).toBe(3);
      expect(result.phase).toBe('Registration');
      expect(result.publicKey).toBeUndefined();
      expect(result.commitments).toEqual([]);
      expect(result.scope).toBe('Treasury');
      expect(result.active).toBe(true);
      expect(result.epoch).toBe(1);
      expect(result.minPhi).toBe(0.3);
    });
  });

  describe('getCommittee', () => {
    it('should pass committee ID as payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(COMMITTEE_ENTRY)
      );

      await client.getCommittee('committee-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_committee',
          payload: 'committee-1',
        })
      );
    });

    it('should return null for missing committee', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getCommittee('nonexistent');
      expect(result).toBeNull();
    });
  });

  describe('getAllCommittees', () => {
    it('should map array of records', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(COMMITTEE_ENTRY),
        mockRecord({ ...COMMITTEE_ENTRY, id: 'committee-2', name: 'Protocol Signers' }),
      ]);

      const result = await client.getAllCommittees();

      expect(result).toHaveLength(2);
      expect(result[0].id).toBe('committee-1');
      expect(result[1].id).toBe('committee-2');
    });
  });

  // --------------------------------------------------------------------------
  // DKG Ceremony
  // --------------------------------------------------------------------------

  describe('registerMember', () => {
    it('should pass snake_case fields', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(MEMBER_ENTRY)
      );

      await client.registerMember({
        committeeId: 'committee-1',
        participantId: 1,
        memberDid: 'did:mycelix:alice',
        trustScore: 0.9,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'register_member',
          payload: {
            committee_id: 'committee-1',
            participant_id: 1,
            member_did: 'did:mycelix:alice',
            trust_score: 0.9,
          },
        })
      );
    });

    it('should map response to CommitteeMember', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(MEMBER_ENTRY)
      );

      const result = await client.registerMember({
        committeeId: 'committee-1',
        participantId: 1,
        memberDid: 'did:mycelix:alice',
        trustScore: 0.9,
      });

      expect(result.committeeId).toBe('committee-1');
      expect(result.participantId).toBe(1);
      expect(result.memberDid).toBe('did:mycelix:alice');
      expect(result.trustScore).toBe(0.9);
      expect(result.dealSubmitted).toBe(false);
      expect(result.qualified).toBe(false);
    });
  });

  describe('submitDkgDeal', () => {
    it('should convert Uint8Array to number array for vss_commitment', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...MEMBER_ENTRY, deal_submitted: true })
      );

      const vss = new Uint8Array([10, 20, 30, 40]);
      await client.submitDkgDeal({
        committeeId: 'committee-1',
        vssCommitment: vss,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'submit_dkg_deal',
          payload: {
            committee_id: 'committee-1',
            vss_commitment: [10, 20, 30, 40],
          },
        })
      );
    });
  });

  describe('finalizeDkg', () => {
    it('should convert all byte arrays to number arrays', async () => {
      const completedCommittee = {
        ...COMMITTEE_ENTRY,
        phase: 'Complete',
        public_key: [1, 2, 3],
      };
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(completedCommittee)
      );

      const pk = new Uint8Array([1, 2, 3]);
      const commitments = [new Uint8Array([10, 20]), new Uint8Array([30, 40])];

      await client.finalizeDkg({
        committeeId: 'committee-1',
        combinedPublicKey: pk,
        publicCommitments: commitments,
        qualifiedMembers: [1, 2],
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'finalize_dkg',
          payload: {
            committee_id: 'committee-1',
            combined_public_key: [1, 2, 3],
            public_commitments: [[10, 20], [30, 40]],
            qualified_members: [1, 2],
          },
        })
      );
    });

    it('should map completed committee with publicKey', async () => {
      const completedCommittee = {
        ...COMMITTEE_ENTRY,
        phase: 'Complete',
        public_key: [1, 2, 3],
        commitments: [[10, 20], [30, 40]],
      };
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(completedCommittee)
      );

      const result = await client.finalizeDkg({
        committeeId: 'committee-1',
        combinedPublicKey: new Uint8Array([1, 2, 3]),
        publicCommitments: [],
        qualifiedMembers: [1, 2],
      });

      expect(result.phase).toBe('Complete');
      expect(result.publicKey).toBeInstanceOf(Uint8Array);
      expect(result.publicKey).toEqual(new Uint8Array([1, 2, 3]));
      expect(result.commitments).toHaveLength(2);
      expect(result.commitments[0]).toBeInstanceOf(Uint8Array);
    });
  });

  // --------------------------------------------------------------------------
  // Signature Operations
  // --------------------------------------------------------------------------

  describe('submitSignatureShare', () => {
    it('should include committeeId in payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({})
      );

      await client.submitSignatureShare({
        committeeId: 'committee-1',
        signatureId: 'sig-1',
        participantId: 1,
        share: new Uint8Array([99, 100]),
        contentHash: new Uint8Array([1, 2, 3]),
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'submit_signature_share',
          payload: {
            committee_id: 'committee-1',
            signature_id: 'sig-1',
            participant_id: 1,
            share: [99, 100],
            content_hash: [1, 2, 3],
          },
        })
      );
    });
  });

  describe('combineSignatures', () => {
    it('should map response to ThresholdSignature', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(SIGNATURE_ENTRY)
      );

      const result = await client.combineSignatures({
        committeeId: 'committee-1',
        contentHash: new Uint8Array([1, 2, 3, 4]),
        contentDescription: 'Treasury allocation #42',
        combinedSignature: new Uint8Array([5, 6, 7, 8]),
        signers: [1, 3],
      });

      expect(result.id).toBe('sig-1');
      expect(result.committeeId).toBe('committee-1');
      expect(result.signedContentHash).toEqual(new Uint8Array([1, 2, 3, 4]));
      expect(result.signature).toEqual(new Uint8Array([5, 6, 7, 8]));
      expect(result.signerCount).toBe(2);
      expect(result.signers).toEqual([1, 3]);
      expect(result.verified).toBe(true);
    });
  });

  describe('getProposalSignature', () => {
    it('should return null for missing signature', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getProposalSignature('nonexistent');
      expect(result).toBeNull();
    });
  });

  describe('getCommitteeMembers', () => {
    it('should map array of member records', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(MEMBER_ENTRY),
        mockRecord({ ...MEMBER_ENTRY, participant_id: 2, member_did: 'did:mycelix:bob' }),
      ]);

      const result = await client.getCommitteeMembers('committee-1');

      expect(result).toHaveLength(2);
      expect(result[0].participantId).toBe(1);
      expect(result[1].participantId).toBe(2);
      expect(result[1].memberDid).toBe('did:mycelix:bob');
    });
  });

  // --------------------------------------------------------------------------
  // Key Rotation
  // --------------------------------------------------------------------------

  describe('rotateCommitteeKeys', () => {
    it('should send bare committeeId string (not a struct)', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...COMMITTEE_ENTRY, epoch: 2 })
      );

      await client.rotateCommitteeKeys({
        committeeId: 'committee-1',
        reason: 'periodic rotation',
      });

      // Critical: Rust extern takes a bare String, not {committee_id, reason}
      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'rotate_committee_keys',
          payload: 'committee-1',
        })
      );
    });

    it('should return new epoch committee', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...COMMITTEE_ENTRY, epoch: 2, phase: 'Registration' })
      );

      const result = await client.rotateCommitteeKeys({
        committeeId: 'committee-1',
        reason: 'key compromise',
      });

      expect(result.epoch).toBe(2);
      expect(result.phase).toBe('Registration');
    });
  });

  // --------------------------------------------------------------------------
  // Committee History & Signature Shares
  // --------------------------------------------------------------------------

  describe('getCommitteeHistory', () => {
    it('should pass committeeId and map array of records', async () => {
      const epoch1 = { ...COMMITTEE_ENTRY, epoch: 1, phase: 'Complete' };
      const epoch2 = { ...COMMITTEE_ENTRY, epoch: 2, phase: 'Registration' };
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(epoch1),
        mockRecord(epoch2),
      ]);

      const result = await client.getCommitteeHistory('committee-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_committee_history',
          payload: 'committee-1',
        })
      );
      expect(result).toHaveLength(2);
      expect(result[0].epoch).toBe(1);
      expect(result[0].phase).toBe('Complete');
      expect(result[1].epoch).toBe(2);
      expect(result[1].phase).toBe('Registration');
    });
  });

  describe('getSignatureShares', () => {
    it('should pass signatureId and return raw records', async () => {
      const shareRecord = mockRecord({
        signature_id: 'sig-1',
        participant_id: 1,
        signer: 'uhCAkAgent1',
        share: [99, 100],
        content_hash: [1, 2, 3],
        submitted_at: 1708300250,
      });
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        shareRecord,
      ]);

      const result = await client.getSignatureShares('sig-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_signature_shares',
          payload: 'sig-1',
        })
      );
      expect(result).toHaveLength(1);
    });
  });

  // --------------------------------------------------------------------------
  // Error Handling
  // --------------------------------------------------------------------------

  describe('error handling', () => {
    it('should wrap zome errors', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Committee not found')
      );

      await expect(client.getCommittee('bad-id')).rejects.toThrow();
    });

    it('should wrap network errors', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Network timeout')
      );

      await expect(
        client.createCommittee({
          name: 'Test',
          threshold: 2,
          memberCount: 3,
          scope: 'All',
        })
      ).rejects.toThrow();
    });
  });
});
