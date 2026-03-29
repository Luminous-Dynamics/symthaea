// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Constitution Client Tests
 *
 * Verifies zome call arguments, response mapping, charter operations,
 * amendment lifecycle, and parameter management for ConstitutionClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ConstitutionClient } from '../constitution';
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

const CHARTER_ENTRY = {
  id: 'charter-1',
  version: 1,
  preamble: 'We the members of this DAO...',
  articles: 'Article 1: Governance principles...',
  rights: ['Data sovereignty', 'Exit right', 'Voice'],
  amendment_process: 'Constitutional amendments require supermajority...',
  adopted: 1708200000,
  last_amended: null,
};

const AMENDMENT_ENTRY = {
  id: 'amend-1',
  charter_version: 1,
  amendment_type: 'AddRight',
  article: null,
  original_text: null,
  new_text: 'Right to data portability',
  rationale: 'Members should be able to export their data',
  proposer: 'did:mycelix:alice',
  proposal_id: 'proposal-1',
  status: 'Deliberation',
  created: 1708300000,
  ratified: null,
};

const PARAMETER_ENTRY = {
  name: 'quorum_threshold',
  value: '0.33',
  value_type: 'Float',
  description: 'Minimum quorum for standard proposals',
  min_value: '0.1',
  max_value: '1.0',
  updated: 1708200000,
  changed_by_proposal: null,
};

// ============================================================================
// TESTS
// ============================================================================

describe('ConstitutionClient', () => {
  let client: ConstitutionClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new ConstitutionClient(mockAppClient);
  });

  // --------------------------------------------------------------------------
  // Initialization
  // --------------------------------------------------------------------------

  describe('initialization', () => {
    it('should create client with default config', () => {
      expect(client).toBeInstanceOf(ConstitutionClient);
    });

    it('should use governance role and constitution zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(CHARTER_ENTRY)
      );

      await client.createCharter({
        id: 'charter-1',
        version: 1,
        preamble: 'We...',
        articles: 'Art 1...',
        rights: [],
        amendmentProcess: 'Process...',
        adopted: 1708200000,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'governance',
          zome_name: 'constitution',
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Charter Operations
  // --------------------------------------------------------------------------

  describe('createCharter', () => {
    it('should pass snake_case payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(CHARTER_ENTRY)
      );

      await client.createCharter({
        id: 'charter-1',
        version: 1,
        preamble: 'We the members...',
        articles: 'Article 1...',
        rights: ['Data sovereignty', 'Exit right'],
        amendmentProcess: 'Supermajority required',
        adopted: 1708200000,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_charter',
          payload: {
            id: 'charter-1',
            version: 1,
            preamble: 'We the members...',
            articles: 'Article 1...',
            rights: ['Data sovereignty', 'Exit right'],
            amendment_process: 'Supermajority required',
            adopted: 1708200000,
            last_amended: undefined,
          },
        })
      );
    });

    it('should map response to Charter with camelCase', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(CHARTER_ENTRY)
      );

      const result = await client.createCharter({
        id: 'charter-1',
        version: 1,
        preamble: 'We...',
        articles: 'Art...',
        rights: [],
        amendmentProcess: 'Process...',
        adopted: 1708200000,
      });

      expect(result.id).toBe('charter-1');
      expect(result.version).toBe(1);
      expect(result.preamble).toBe('We the members of this DAO...');
      expect(result.rights).toHaveLength(3);
      expect(result.amendmentProcess).toContain('supermajority');
      expect(result.adopted).toBe(1708200000);
      expect(result.lastAmended).toBeNull();
    });
  });

  describe('getCurrentCharter', () => {
    it('should return charter when exists', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(CHARTER_ENTRY)
      );

      const result = await client.getCurrentCharter();

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_current_charter',
          payload: null,
        })
      );
      expect(result).not.toBeNull();
      expect(result!.version).toBe(1);
    });

    it('should return null when no charter', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getCurrentCharter();
      expect(result).toBeNull();
    });
  });

  // --------------------------------------------------------------------------
  // Amendment Operations
  // --------------------------------------------------------------------------

  describe('proposeAmendment', () => {
    it('should pass snake_case payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(AMENDMENT_ENTRY)
      );

      await client.proposeAmendment({
        amendmentType: 'AddRight',
        newText: 'Right to data portability',
        rationale: 'Members should export data',
        proposerDid: 'did:mycelix:alice',
        proposalId: 'proposal-1',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'propose_amendment',
          payload: {
            amendment_type: 'AddRight',
            article: undefined,
            original_text: undefined,
            new_text: 'Right to data portability',
            rationale: 'Members should export data',
            proposer_did: 'did:mycelix:alice',
            proposal_id: 'proposal-1',
          },
        })
      );
    });

    it('should map response to Amendment with camelCase', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(AMENDMENT_ENTRY)
      );

      const result = await client.proposeAmendment({
        amendmentType: 'AddRight',
        newText: 'Right to data portability',
        rationale: 'Data sovereignty',
        proposerDid: 'did:mycelix:alice',
        proposalId: 'proposal-1',
      });

      expect(result.charterVersion).toBe(1);
      expect(result.amendmentType).toBe('AddRight');
      expect(result.newText).toBe('Right to data portability');
      expect(result.proposer).toBe('did:mycelix:alice');
      expect(result.status).toBe('Deliberation');
      expect(result.ratified).toBeNull();
    });

    it('should pass article and original_text for modify amendments', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...AMENDMENT_ENTRY, amendment_type: 'ModifyArticle' })
      );

      await client.proposeAmendment({
        amendmentType: 'ModifyArticle',
        article: 'Article 3',
        originalText: 'Old text here',
        newText: 'New text here',
        rationale: 'Clarification needed',
        proposerDid: 'did:mycelix:alice',
        proposalId: 'proposal-2',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: expect.objectContaining({
            article: 'Article 3',
            original_text: 'Old text here',
          }),
        })
      );
    });
  });

  describe('ratifyAmendment', () => {
    it('should pass amendment ID as payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...AMENDMENT_ENTRY, status: 'Ratified', ratified: 1709000000 })
      );

      const result = await client.ratifyAmendment('amend-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'ratify_amendment',
          payload: 'amend-1',
        })
      );
      expect(result.status).toBe('Ratified');
      expect(result.ratified).toBe(1709000000);
    });
  });

  // --------------------------------------------------------------------------
  // Parameter Operations
  // --------------------------------------------------------------------------

  describe('setParameter', () => {
    it('should pass snake_case payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(PARAMETER_ENTRY)
      );

      await client.setParameter({
        name: 'quorum_threshold',
        value: '0.33',
        valueType: 'Float',
        description: 'Minimum quorum',
        minValue: '0.1',
        maxValue: '1.0',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'set_parameter',
          payload: {
            name: 'quorum_threshold',
            value: '0.33',
            value_type: 'Float',
            description: 'Minimum quorum',
            min_value: '0.1',
            max_value: '1.0',
            proposal_id: undefined,
          },
        })
      );
    });

    it('should map response to GovernanceParameter', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(PARAMETER_ENTRY)
      );

      const result = await client.setParameter({
        name: 'quorum_threshold',
        value: '0.33',
        valueType: 'Float',
        description: 'Minimum quorum',
      });

      expect(result.name).toBe('quorum_threshold');
      expect(result.value).toBe('0.33');
      expect(result.valueType).toBe('Float');
      expect(result.minValue).toBe('0.1');
      expect(result.maxValue).toBe('1.0');
    });
  });

  describe('updateParameter', () => {
    it('should pass parameter name and new value', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...PARAMETER_ENTRY, value: '0.5' })
      );

      const result = await client.updateParameter({
        parameter: 'quorum_threshold',
        value: '0.5',
        proposalId: 'proposal-1',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'update_parameter',
          payload: {
            parameter: 'quorum_threshold',
            value: '0.5',
            proposal_id: 'proposal-1',
          },
        })
      );
      expect(result.value).toBe('0.5');
    });
  });

  describe('getParameter', () => {
    it('should return parameter for valid name', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(PARAMETER_ENTRY)
      );

      const result = await client.getParameter('quorum_threshold');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_parameter',
          payload: 'quorum_threshold',
        })
      );
      expect(result).not.toBeNull();
      expect(result!.name).toBe('quorum_threshold');
    });

    it('should return null for missing parameter', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getParameter('nonexistent');
      expect(result).toBeNull();
    });
  });

  describe('listParameters', () => {
    it('should return array of parameters', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(PARAMETER_ENTRY),
        mockRecord({ ...PARAMETER_ENTRY, name: 'approval_threshold', value: '0.51' }),
      ]);

      const result = await client.listParameters();

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'list_parameters',
          payload: null,
        })
      );
      expect(result).toHaveLength(2);
      expect(result[0].name).toBe('quorum_threshold');
      expect(result[1].name).toBe('approval_threshold');
    });
  });

  // --------------------------------------------------------------------------
  // Error Handling
  // --------------------------------------------------------------------------

  describe('error handling', () => {
    it('should propagate zome errors on create charter', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Charter already exists')
      );

      await expect(
        client.createCharter({
          id: 'charter-1',
          version: 1,
          preamble: 'We...',
          articles: 'Art...',
          rights: [],
          amendmentProcess: 'Process...',
          adopted: 0,
        })
      ).rejects.toThrow();
    });

    it('should propagate zome errors on propose amendment', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('No charter exists')
      );

      await expect(
        client.proposeAmendment({
          amendmentType: 'AddRight',
          newText: 'Right',
          rationale: 'Reason',
          proposerDid: 'did:mycelix:alice',
          proposalId: 'p-1',
        })
      ).rejects.toThrow();
    });
  });
});
