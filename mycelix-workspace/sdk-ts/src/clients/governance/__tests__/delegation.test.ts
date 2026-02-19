/**
 * Delegation Client Tests
 *
 * Verifies zome call arguments, response mapping, chain resolution,
 * cycle detection, and convenience methods for the DelegationClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { DelegationClient } from '../delegation';
import type { AppClient } from '@holochain/client';

// ============================================================================
// MOCK HELPERS
// ============================================================================

function createMockClient(): AppClient {
  return {
    callZome: vi.fn(),
  } as unknown as AppClient;
}

/** Mock a Holochain record using the extractEntry() structure */
function mockRecord(entry: Record<string, unknown>) {
  return {
    entry: { Present: entry },
    signed_action: { hashed: { hash: new Uint8Array(32) } },
  };
}

const DELEGATION_ENTRY = {
  delegator_did: 'did:mycelix:alice',
  delegate_did: 'did:mycelix:bob',
  dao_id: 'dao-1',
  scope: 'All',
  category: null,
  power_percentage: 1.0,
  expires_at: null,
  active: true,
  created_at: 1708300000,
};

const CHAIN_RESULT = {
  start_did: 'did:mycelix:alice',
  chain: [
    { from: 'did:mycelix:alice', to: 'did:mycelix:bob', power: 1.0 },
    { from: 'did:mycelix:bob', to: 'did:mycelix:carol', power: 0.5 },
  ],
  has_cycle: false,
};

// ============================================================================
// TESTS
// ============================================================================

describe('DelegationClient', () => {
  let client: DelegationClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new DelegationClient(mockAppClient);
  });

  // --------------------------------------------------------------------------
  // Initialization
  // --------------------------------------------------------------------------

  describe('initialization', () => {
    it('should create client with default config', () => {
      expect(client).toBeInstanceOf(DelegationClient);
    });

    it('should use governance role and delegation zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(DELEGATION_ENTRY)
      );

      await client.createDelegation({
        delegateDid: 'did:mycelix:bob',
        daoId: 'dao-1',
        scope: 'All',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'governance',
          zome_name: 'delegation',
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // CRUD Operations
  // --------------------------------------------------------------------------

  describe('createDelegation', () => {
    it('should pass snake_case payload with default power_percentage', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(DELEGATION_ENTRY)
      );

      await client.createDelegation({
        delegateDid: 'did:mycelix:bob',
        daoId: 'dao-1',
        scope: 'All',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_delegation',
          payload: {
            delegate_did: 'did:mycelix:bob',
            dao_id: 'dao-1',
            scope: 'All',
            category: undefined,
            power_percentage: 1.0,
            expires_at: undefined,
          },
        })
      );
    });

    it('should map response to Delegation with camelCase', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(DELEGATION_ENTRY)
      );

      const result = await client.createDelegation({
        delegateDid: 'did:mycelix:bob',
        daoId: 'dao-1',
        scope: 'All',
      });

      expect(result.delegatorDid).toBe('did:mycelix:alice');
      expect(result.delegateDid).toBe('did:mycelix:bob');
      expect(result.daoId).toBe('dao-1');
      expect(result.scope).toBe('All');
      expect(result.powerPercentage).toBe(1.0);
      expect(result.active).toBe(true);
      expect(result.createdAt).toBe(1708300000);
    });

    it('should pass custom power_percentage and expires_at', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...DELEGATION_ENTRY, power_percentage: 0.5, expires_at: 1710000000 })
      );

      await client.createDelegation({
        delegateDid: 'did:mycelix:bob',
        daoId: 'dao-1',
        scope: 'Category',
        category: 'Treasury',
        powerPercentage: 0.5,
        expiresAt: 1710000000,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: expect.objectContaining({
            power_percentage: 0.5,
            expires_at: 1710000000,
            category: 'Treasury',
          }),
        })
      );
    });
  });

  describe('revokeDelegation', () => {
    it('should send delegation ID as payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      await client.revokeDelegation('delegation-hash-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'revoke_delegation',
          payload: 'delegation-hash-1',
        })
      );
    });
  });

  describe('updateDelegation', () => {
    it('should pass delegation_id and snake_case updates', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...DELEGATION_ENTRY, power_percentage: 0.7 })
      );

      const result = await client.updateDelegation('delegation-hash-1', {
        powerPercentage: 0.7,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'update_delegation',
          payload: {
            delegation_id: 'delegation-hash-1',
            power_percentage: 0.7,
            expires_at: undefined,
            category: undefined,
          },
        })
      );

      expect(result.powerPercentage).toBe(0.7);
    });
  });

  describe('getDelegation', () => {
    it('should return delegation for valid ID', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(DELEGATION_ENTRY)
      );

      const result = await client.getDelegation('delegation-hash-1');
      expect(result).not.toBeNull();
      expect(result!.delegatorDid).toBe('did:mycelix:alice');
    });

    it('should return null for missing delegation', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getDelegation('nonexistent');
      expect(result).toBeNull();
    });
  });

  // --------------------------------------------------------------------------
  // List & Filter
  // --------------------------------------------------------------------------

  describe('listDelegations', () => {
    it('should pass filter with snake_case fields', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(DELEGATION_ENTRY),
      ]);

      await client.listDelegations({
        daoId: 'dao-1',
        delegatorDid: 'did:mycelix:alice',
        activeOnly: true,
        limit: 10,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'list_delegations',
          payload: expect.objectContaining({
            dao_id: 'dao-1',
            delegator_did: 'did:mycelix:alice',
            active_only: true,
            limit: 10,
          }),
        })
      );
    });

    it('should default active_only to true', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);

      await client.listDelegations({});

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: expect.objectContaining({ active_only: true }),
        })
      );
    });
  });

  describe('getDelegationsFrom', () => {
    it('should delegate to listDelegations with delegator filter', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(DELEGATION_ENTRY),
      ]);

      const result = await client.getDelegationsFrom('dao-1', 'did:mycelix:alice');

      expect(result).toHaveLength(1);
      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'list_delegations',
          payload: expect.objectContaining({
            dao_id: 'dao-1',
            delegator_did: 'did:mycelix:alice',
            active_only: true,
          }),
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Chain Resolution
  // --------------------------------------------------------------------------

  describe('getDelegationChain', () => {
    it('should pass start_did and dao_id', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(CHAIN_RESULT);

      const result = await client.getDelegationChain('dao-1', 'did:mycelix:alice');

      expect(result.startDid).toBe('did:mycelix:alice');
      expect(result.chain).toHaveLength(2);
      expect(result.hasCycle).toBe(false);
    });
  });

  describe('wouldCreateCycle', () => {
    it('should return boolean from zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(true);

      const result = await client.wouldCreateCycle(
        'dao-1',
        'did:mycelix:bob',
        'did:mycelix:alice'
      );

      expect(result).toBe(true);
    });
  });

  // --------------------------------------------------------------------------
  // Validation
  // --------------------------------------------------------------------------

  describe('isDelegationValid', () => {
    it('should return true for active non-expired delegation', () => {
      const delegation = {
        id: 'del-1',
        delegatorDid: 'did:mycelix:alice',
        delegateDid: 'did:mycelix:bob',
        daoId: 'dao-1',
        scope: 'All' as const,
        powerPercentage: 1.0,
        active: true,
        createdAt: 1708300000,
      };

      expect(client.isDelegationValid(delegation)).toBe(true);
    });

    it('should return false for inactive delegation', () => {
      const delegation = {
        id: 'del-1',
        delegatorDid: 'did:mycelix:alice',
        delegateDid: 'did:mycelix:bob',
        daoId: 'dao-1',
        scope: 'All' as const,
        powerPercentage: 1.0,
        active: false,
        createdAt: 1708300000,
      };

      expect(client.isDelegationValid(delegation)).toBe(false);
    });
  });

  // --------------------------------------------------------------------------
  // Convenience Methods
  // --------------------------------------------------------------------------

  describe('delegateAll', () => {
    it('should create full-power delegation with scope All', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(DELEGATION_ENTRY)
      );

      await client.delegateAll('dao-1', 'did:mycelix:bob');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_delegation',
          payload: expect.objectContaining({
            delegate_did: 'did:mycelix:bob',
            dao_id: 'dao-1',
            scope: 'All',
            power_percentage: 1.0,
          }),
        })
      );
    });
  });

  describe('delegateForCategory', () => {
    it('should create category-scoped delegation', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...DELEGATION_ENTRY, scope: 'Category', category: 'Treasury' })
      );

      await client.delegateForCategory('dao-1', 'did:mycelix:bob', 'Treasury', 0.5);

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          payload: expect.objectContaining({
            scope: 'Category',
            category: 'Treasury',
            power_percentage: 0.5,
          }),
        })
      );
    });
  });

  describe('getDelegatedPowerReceived', () => {
    it('should return total delegated power', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(2.5);

      const power = await client.getDelegatedPowerReceived('dao-1', 'did:mycelix:bob');
      expect(power).toBe(2.5);
    });
  });

  describe('getDelegatorsCount', () => {
    it('should return count of delegations received', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(DELEGATION_ENTRY),
        mockRecord({ ...DELEGATION_ENTRY, delegator_did: 'did:mycelix:carol' }),
      ]);

      const count = await client.getDelegatorsCount('dao-1', 'did:mycelix:bob');
      expect(count).toBe(2);
    });
  });

  // --------------------------------------------------------------------------
  // Error Handling
  // --------------------------------------------------------------------------

  describe('error handling', () => {
    it('should propagate cycle detection errors', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Circular delegation detected')
      );

      await expect(
        client.createDelegation({
          delegateDid: 'did:mycelix:alice',
          daoId: 'dao-1',
          scope: 'All',
        })
      ).rejects.toThrow();
    });

    it('should propagate revocation errors', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Delegation not found')
      );

      await expect(client.revokeDelegation('nonexistent')).rejects.toThrow();
    });
  });
});
