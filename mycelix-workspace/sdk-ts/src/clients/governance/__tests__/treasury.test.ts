// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Treasury Client Tests
 *
 * Verifies zome call arguments, response mapping, multi-sig operations,
 * balance queries, allocation lifecycle, and helper methods for TreasuryClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { TreasuryClient } from '../treasury';
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

const TREASURY_ENTRY = {
  dao_id: 'dao-1',
  name: 'Community Fund',
  description: 'DAO treasury for community projects',
  governance_threshold: 0.66,
  discretionary_limit: 1000,
  discretionary_period_hours: 168,
  multi_sig_threshold: 3,
  multi_sig_signers: ['did:mycelix:alice', 'did:mycelix:bob', 'did:mycelix:carol'],
  active: true,
  created_at: 1708200000,
};

const ALLOCATION_ENTRY = {
  treasury_id: 'treasury-1',
  proposal_id: 'proposal-1',
  purpose: 'Equipment purchase',
  amount: 5000,
  currency: 'MYC',
  recipient_did: 'did:mycelix:cooperative',
  recipient_address: null,
  status: 'Proposed',
  approved_by: [],
  execution_tx_hash: null,
  requested_at: 1708300000,
  executed_at: null,
};

const BALANCE_RESULT = {
  treasury_id: 'treasury-1',
  currency: 'MYC',
  available: 45000,
  locked: 5000,
  total: 50000,
  updated_at: 1708400000,
};

// ============================================================================
// TESTS
// ============================================================================

describe('TreasuryClient', () => {
  let client: TreasuryClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new TreasuryClient(mockAppClient);
  });

  // --------------------------------------------------------------------------
  // Initialization
  // --------------------------------------------------------------------------

  describe('initialization', () => {
    it('should create client with default config', () => {
      expect(client).toBeInstanceOf(TreasuryClient);
    });

    it('should use governance role and treasury zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(TREASURY_ENTRY)
      );

      await client.createTreasury({
        daoId: 'dao-1',
        name: 'Test',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'governance',
          zome_name: 'treasury',
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Treasury CRUD
  // --------------------------------------------------------------------------

  describe('createTreasury', () => {
    it('should pass snake_case payload with defaults', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(TREASURY_ENTRY)
      );

      await client.createTreasury({
        daoId: 'dao-1',
        name: 'Community Fund',
        description: 'DAO treasury',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_treasury',
          payload: {
            dao_id: 'dao-1',
            name: 'Community Fund',
            description: 'DAO treasury',
            governance_threshold: 0.51,
            discretionary_limit: 0,
            discretionary_period_hours: 168,
            multi_sig_threshold: 1,
            multi_sig_signers: [],
          },
        })
      );
    });

    it('should map response to Treasury with camelCase', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(TREASURY_ENTRY)
      );

      const result = await client.createTreasury({
        daoId: 'dao-1',
        name: 'Community Fund',
      });

      expect(result.daoId).toBe('dao-1');
      expect(result.name).toBe('Community Fund');
      expect(result.governanceThreshold).toBe(0.66);
      expect(result.discretionaryLimit).toBe(1000);
      expect(result.multiSigThreshold).toBe(3);
      expect(result.multiSigSigners).toHaveLength(3);
      expect(result.active).toBe(true);
    });

    it('should pass custom thresholds', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(TREASURY_ENTRY)
      );

      await client.createTreasury({
        daoId: 'dao-1',
        name: 'Fund',
        governanceThreshold: 0.75,
        discretionaryLimit: 500,
        multiSigThreshold: 2,
        multiSigSigners: ['did:mycelix:alice', 'did:mycelix:bob'],
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: expect.objectContaining({
            governance_threshold: 0.75,
            discretionary_limit: 500,
            multi_sig_threshold: 2,
            multi_sig_signers: ['did:mycelix:alice', 'did:mycelix:bob'],
          }),
        })
      );
    });
  });

  describe('getTreasury', () => {
    it('should return treasury for valid ID', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(TREASURY_ENTRY)
      );

      const result = await client.getTreasury('treasury-1');
      expect(result).not.toBeNull();
      expect(result!.name).toBe('Community Fund');
    });

    it('should return null for missing treasury', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getTreasury('nonexistent');
      expect(result).toBeNull();
    });
  });

  describe('getTreasuryByDAO', () => {
    it('should pass dao ID to zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(TREASURY_ENTRY)
      );

      const result = await client.getTreasuryByDAO('dao-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_treasury_by_dao',
          payload: 'dao-1',
        })
      );
      expect(result).not.toBeNull();
    });
  });

  describe('updateTreasury', () => {
    it('should pass treasury_id and snake_case updates', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...TREASURY_ENTRY, name: 'Updated Fund' })
      );

      const result = await client.updateTreasury('treasury-1', {
        name: 'Updated Fund',
        governanceThreshold: 0.75,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'update_treasury',
          payload: {
            treasury_id: 'treasury-1',
            name: 'Updated Fund',
            description: undefined,
            governance_threshold: 0.75,
            discretionary_limit: undefined,
            discretionary_period_hours: undefined,
          },
        })
      );
      expect(result.name).toBe('Updated Fund');
    });
  });

  describe('listTreasuries', () => {
    it('should return array of treasuries', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(TREASURY_ENTRY),
        mockRecord({ ...TREASURY_ENTRY, name: 'Emergency Fund' }),
      ]);

      const result = await client.listTreasuries();
      expect(result).toHaveLength(2);
      expect(result[1].name).toBe('Emergency Fund');
    });
  });

  // --------------------------------------------------------------------------
  // Multi-Sig Management
  // --------------------------------------------------------------------------

  describe('addSigner', () => {
    it('should pass treasury_id and signer_did', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(TREASURY_ENTRY)
      );

      await client.addSigner('treasury-1', 'did:mycelix:dave');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'add_multi_sig_signer',
          payload: {
            treasury_id: 'treasury-1',
            signer_did: 'did:mycelix:dave',
          },
        })
      );
    });
  });

  describe('removeSigner', () => {
    it('should pass treasury_id and signer_did', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(TREASURY_ENTRY)
      );

      await client.removeSigner('treasury-1', 'did:mycelix:carol');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'remove_multi_sig_signer',
          payload: {
            treasury_id: 'treasury-1',
            signer_did: 'did:mycelix:carol',
          },
        })
      );
    });
  });

  describe('updateMultiSigThreshold', () => {
    it('should pass treasury_id and threshold', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...TREASURY_ENTRY, multi_sig_threshold: 2 })
      );

      const result = await client.updateMultiSigThreshold('treasury-1', 2);

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'update_multi_sig_threshold',
          payload: {
            treasury_id: 'treasury-1',
            threshold: 2,
          },
        })
      );
      expect(result.multiSigThreshold).toBe(2);
    });
  });

  // --------------------------------------------------------------------------
  // Balance Operations
  // --------------------------------------------------------------------------

  describe('getBalance', () => {
    it('should map response to TreasuryBalance', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(BALANCE_RESULT);

      const result = await client.getBalance('treasury-1', 'MYC');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_treasury_balance',
          payload: {
            treasury_id: 'treasury-1',
            currency: 'MYC',
          },
        })
      );
      expect(result.treasuryId).toBe('treasury-1');
      expect(result.currency).toBe('MYC');
      expect(result.available).toBe(45000);
      expect(result.locked).toBe(5000);
      expect(result.total).toBe(50000);
    });
  });

  describe('getAllBalances', () => {
    it('should map array of balances', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        BALANCE_RESULT,
        { ...BALANCE_RESULT, currency: 'ETH', available: 10, locked: 0, total: 10 },
      ]);

      const result = await client.getAllBalances('treasury-1');
      expect(result).toHaveLength(2);
      expect(result[0].currency).toBe('MYC');
      expect(result[1].currency).toBe('ETH');
    });
  });

  describe('deposit', () => {
    it('should pass treasury_id, amount, and currency', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ...BALANCE_RESULT,
        available: 55000,
        total: 60000,
      });

      const result = await client.deposit('treasury-1', 10000, 'MYC');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'deposit_to_treasury',
          payload: {
            treasury_id: 'treasury-1',
            amount: 10000,
            currency: 'MYC',
          },
        })
      );
      expect(result.available).toBe(55000);
    });
  });

  // --------------------------------------------------------------------------
  // Allocation Operations
  // --------------------------------------------------------------------------

  describe('proposeAllocation', () => {
    it('should pass snake_case payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(ALLOCATION_ENTRY)
      );

      await client.proposeAllocation({
        treasuryId: 'treasury-1',
        purpose: 'Equipment purchase',
        amount: 5000,
        currency: 'MYC',
        recipientDid: 'did:mycelix:cooperative',
        proposalId: 'proposal-1',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'propose_allocation',
          payload: {
            treasury_id: 'treasury-1',
            purpose: 'Equipment purchase',
            amount: 5000,
            currency: 'MYC',
            recipient_did: 'did:mycelix:cooperative',
            recipient_address: undefined,
            proposal_id: 'proposal-1',
          },
        })
      );
    });

    it('should map response to TreasuryAllocation', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(ALLOCATION_ENTRY)
      );

      const result = await client.proposeAllocation({
        treasuryId: 'treasury-1',
        purpose: 'Equipment purchase',
        amount: 5000,
        currency: 'MYC',
        recipientDid: 'did:mycelix:cooperative',
      });

      expect(result.treasuryId).toBe('treasury-1');
      expect(result.amount).toBe(5000);
      expect(result.currency).toBe('MYC');
      expect(result.status).toBe('Proposed');
      expect(result.approvedBy).toEqual([]);
    });
  });

  describe('getAllocation', () => {
    it('should return allocation for valid ID', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(ALLOCATION_ENTRY)
      );

      const result = await client.getAllocation('alloc-1');
      expect(result).not.toBeNull();
      expect(result!.purpose).toBe('Equipment purchase');
    });

    it('should return null for missing allocation', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getAllocation('nonexistent');
      expect(result).toBeNull();
    });
  });

  describe('getAllocations', () => {
    it('should pass treasury_id and optional status filter', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(ALLOCATION_ENTRY),
      ]);

      await client.getAllocations('treasury-1', 'Proposed');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_allocations_for_treasury',
          payload: {
            treasury_id: 'treasury-1',
            status_filter: 'Proposed',
          },
        })
      );
    });
  });

  describe('getPendingAllocations', () => {
    it('should pass treasury ID to zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(ALLOCATION_ENTRY),
      ]);

      const result = await client.getPendingAllocations('treasury-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_pending_allocations',
          payload: 'treasury-1',
        })
      );
      expect(result).toHaveLength(1);
    });
  });

  describe('approveAllocation', () => {
    it('should send allocation ID as payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...ALLOCATION_ENTRY, status: 'Approved', approved_by: ['did:mycelix:alice'] })
      );

      const result = await client.approveAllocation('alloc-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'approve_allocation',
          payload: 'alloc-1',
        })
      );
      expect(result.status).toBe('Approved');
    });
  });

  describe('executeAllocation', () => {
    it('should send allocation ID as payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...ALLOCATION_ENTRY, status: 'Executed', executed_at: 1709000000 })
      );

      const result = await client.executeAllocation('alloc-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'execute_allocation',
          payload: 'alloc-1',
        })
      );
      expect(result.status).toBe('Executed');
      expect(result.executedAt).toBe(1709000000);
    });
  });

  describe('cancelAllocation', () => {
    it('should pass allocation_id and reason', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...ALLOCATION_ENTRY, status: 'Cancelled' })
      );

      await client.cancelAllocation('alloc-1', 'Budget revised');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'cancel_allocation',
          payload: {
            allocation_id: 'alloc-1',
            reason: 'Budget revised',
          },
        })
      );
    });
  });

  describe('rejectAllocation', () => {
    it('should pass allocation_id and reason', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...ALLOCATION_ENTRY, status: 'Rejected' })
      );

      await client.rejectAllocation('alloc-1', 'Insufficient justification');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'reject_allocation',
          payload: {
            allocation_id: 'alloc-1',
            reason: 'Insufficient justification',
          },
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Convenience Methods
  // --------------------------------------------------------------------------

  describe('quickAllocation', () => {
    it('should delegate to proposeAllocation', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(ALLOCATION_ENTRY)
      );

      await client.quickAllocation('treasury-1', 'Coffee', 50, 'MYC', 'did:mycelix:bob');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'propose_allocation',
          payload: expect.objectContaining({
            treasury_id: 'treasury-1',
            purpose: 'Coffee',
            amount: 50,
            currency: 'MYC',
            recipient_did: 'did:mycelix:bob',
          }),
        })
      );
    });
  });

  describe('getAvailableBalance', () => {
    it('should return available field from balance', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(BALANCE_RESULT);

      const available = await client.getAvailableBalance('treasury-1', 'MYC');
      expect(available).toBe(45000);
    });
  });

  describe('canExecute', () => {
    it('should return true when approved and enough signers', () => {
      const allocation = {
        id: 'a-1', treasuryId: 'treasury-1', purpose: '', amount: 100,
        currency: 'MYC', recipientDid: '', status: 'Approved' as const,
        approvedBy: ['did:mycelix:alice', 'did:mycelix:bob', 'did:mycelix:carol'],
        requestedAt: 0,
      };
      const treasury = {
        id: 't-1', daoId: 'dao-1', name: '', governanceThreshold: 0.51,
        discretionaryLimit: 1000, discretionaryPeriodHours: 168,
        multiSigThreshold: 3, multiSigSigners: [], active: true, createdAt: 0,
      };

      expect(client.canExecute(allocation, treasury)).toBe(true);
    });

    it('should return false when not approved', () => {
      const allocation = {
        id: 'a-1', treasuryId: 'treasury-1', purpose: '', amount: 100,
        currency: 'MYC', recipientDid: '', status: 'Proposed' as const,
        approvedBy: ['did:mycelix:alice', 'did:mycelix:bob', 'did:mycelix:carol'],
        requestedAt: 0,
      };
      const treasury = {
        id: 't-1', daoId: 'dao-1', name: '', governanceThreshold: 0.51,
        discretionaryLimit: 1000, discretionaryPeriodHours: 168,
        multiSigThreshold: 3, multiSigSigners: [], active: true, createdAt: 0,
      };

      expect(client.canExecute(allocation, treasury)).toBe(false);
    });
  });

  describe('requiresGovernance', () => {
    it('should return true when amount exceeds discretionary limit', () => {
      const treasury = {
        id: 't-1', daoId: 'dao-1', name: '', governanceThreshold: 0.51,
        discretionaryLimit: 1000, discretionaryPeriodHours: 168,
        multiSigThreshold: 1, multiSigSigners: [], active: true, createdAt: 0,
      };

      expect(client.requiresGovernance(5000, treasury)).toBe(true);
      expect(client.requiresGovernance(500, treasury)).toBe(false);
    });
  });

  describe('calculateTotalAllocated', () => {
    it('should sum pending allocations for currency', () => {
      const allocations = [
        { id: '1', treasuryId: 't', purpose: '', amount: 1000, currency: 'MYC', recipientDid: '', status: 'Proposed' as const, approvedBy: [], requestedAt: 0 },
        { id: '2', treasuryId: 't', purpose: '', amount: 2000, currency: 'MYC', recipientDid: '', status: 'Approved' as const, approvedBy: [], requestedAt: 0 },
        { id: '3', treasuryId: 't', purpose: '', amount: 3000, currency: 'MYC', recipientDid: '', status: 'Executed' as const, approvedBy: [], requestedAt: 0 },
        { id: '4', treasuryId: 't', purpose: '', amount: 500, currency: 'ETH', recipientDid: '', status: 'Proposed' as const, approvedBy: [], requestedAt: 0 },
      ];

      expect(client.calculateTotalAllocated(allocations, 'MYC')).toBe(3000);
    });
  });

  describe('getStatusDescription', () => {
    it('should return description for each status', () => {
      expect(client.getStatusDescription('Proposed')).toContain('proposed');
      expect(client.getStatusDescription('Executed')).toContain('executed');
      expect(client.getStatusDescription('Rejected')).toContain('rejected');
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
        client.createTreasury({ daoId: 'dao-1', name: 'Test' })
      ).rejects.toThrow();
    });

    it('should propagate zome errors on allocation', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Insufficient funds')
      );

      await expect(
        client.proposeAllocation({
          treasuryId: 'treasury-1',
          purpose: 'Test',
          amount: 999999,
          currency: 'MYC',
          recipientDid: 'did:mycelix:bob',
        })
      ).rejects.toThrow();
    });
  });
});
