/**
 * Execution Client Tests
 *
 * Verifies zome call arguments, response mapping, timelock lifecycle,
 * guardian vetoes, fund allocation, and helper methods for ExecutionClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ExecutionClient } from '../execution';
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

const TIMELOCK_ENTRY = {
  id: 'timelock-1',
  proposal_id: 'proposal-1',
  actions: '{"action":"transfer","amount":5000}',
  started: 1708300000,
  expires: 1708472800,
  status: 'Pending',
  cancellation_reason: null,
};

const FUND_ALLOCATION_ENTRY = {
  id: 'fund-1',
  proposal_id: 'proposal-1',
  timelock_id: 'timelock-1',
  source_account: 'treasury-main',
  amount: 5000,
  currency: 'MYC',
  locked_at: 1708300000,
  status: 'Locked',
  status_reason: null,
};

// ============================================================================
// TESTS
// ============================================================================

describe('ExecutionClient', () => {
  let client: ExecutionClient;
  let mockAppClient: AppClient;

  beforeEach(() => {
    mockAppClient = createMockClient();
    client = new ExecutionClient(mockAppClient);
  });

  // --------------------------------------------------------------------------
  // Initialization
  // --------------------------------------------------------------------------

  describe('initialization', () => {
    it('should create client with default config', () => {
      expect(client).toBeInstanceOf(ExecutionClient);
    });

    it('should use governance role and execution zome', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(TIMELOCK_ENTRY)
      );

      await client.createTimelock({
        proposalId: 'proposal-1',
        actions: '{}',
        delayHours: 48,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'governance',
          zome_name: 'execution',
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Timelock Operations
  // --------------------------------------------------------------------------

  describe('createTimelock', () => {
    it('should pass snake_case payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(TIMELOCK_ENTRY)
      );

      await client.createTimelock({
        proposalId: 'proposal-1',
        actions: '{"action":"transfer","amount":5000}',
        delayHours: 48,
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_timelock',
          payload: {
            proposal_id: 'proposal-1',
            actions: '{"action":"transfer","amount":5000}',
            delay_hours: 48,
          },
        })
      );
    });

    it('should map response to Timelock with camelCase', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(TIMELOCK_ENTRY)
      );

      const result = await client.createTimelock({
        proposalId: 'proposal-1',
        actions: '{}',
        delayHours: 48,
      });

      expect(result.id).toBe('timelock-1');
      expect(result.proposalId).toBe('proposal-1');
      expect(result.status).toBe('Pending');
      expect(result.started).toBe(1708300000);
      expect(result.expires).toBe(1708472800);
      expect(result.cancellationReason).toBeNull();
    });
  });

  describe('getProposalTimelock', () => {
    it('should return timelock for valid proposal', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(TIMELOCK_ENTRY)
      );

      const result = await client.getProposalTimelock('proposal-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_proposal_timelock',
          payload: 'proposal-1',
        })
      );
      expect(result).not.toBeNull();
      expect(result!.proposalId).toBe('proposal-1');
    });

    it('should return null for missing timelock', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getProposalTimelock('nonexistent');
      expect(result).toBeNull();
    });
  });

  describe('markTimelockReady', () => {
    it('should pass timelock_id in payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...TIMELOCK_ENTRY, status: 'Ready' })
      );

      const result = await client.markTimelockReady('timelock-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'mark_timelock_ready',
          payload: {
            timelock_id: 'timelock-1',
          },
        })
      );
      expect(result.status).toBe('Ready');
    });
  });

  describe('executeTimelock', () => {
    it('should pass timelock_id in payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...TIMELOCK_ENTRY, status: 'Executed' })
      );

      await client.executeTimelock({ timelockId: 'timelock-1' });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'execute_timelock',
          payload: {
            timelock_id: 'timelock-1',
          },
        })
      );
    });
  });

  describe('getPendingTimelocks', () => {
    it('should return array of pending timelocks', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
        mockRecord(TIMELOCK_ENTRY),
        mockRecord({ ...TIMELOCK_ENTRY, id: 'timelock-2' }),
      ]);

      const result = await client.getPendingTimelocks();

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_pending_timelocks',
          payload: null,
        })
      );
      expect(result).toHaveLength(2);
    });
  });

  // --------------------------------------------------------------------------
  // Guardian Veto
  // --------------------------------------------------------------------------

  describe('vetoTimelock', () => {
    it('should pass timelock_id and reason', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...TIMELOCK_ENTRY, status: 'Cancelled', cancellation_reason: 'Security concern' })
      );

      await client.vetoTimelock({
        timelockId: 'timelock-1',
        reason: 'Security concern',
      });

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'veto_timelock',
          payload: {
            timelock_id: 'timelock-1',
            reason: 'Security concern',
          },
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Fund Allocation
  // --------------------------------------------------------------------------

  describe('lockFunds', () => {
    it('should pass all parameters in snake_case', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(FUND_ALLOCATION_ENTRY)
      );

      const result = await client.lockFunds(
        'proposal-1',
        'timelock-1',
        'treasury-main',
        5000,
        'MYC'
      );

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'lock_proposal_funds',
          payload: {
            proposal_id: 'proposal-1',
            timelock_id: 'timelock-1',
            source_account: 'treasury-main',
            amount: 5000,
            currency: 'MYC',
          },
        })
      );
      expect(result.proposalId).toBe('proposal-1');
      expect(result.timelockId).toBe('timelock-1');
      expect(result.sourceAccount).toBe('treasury-main');
      expect(result.amount).toBe(5000);
      expect(result.currency).toBe('MYC');
      expect(result.status).toBe('Locked');
    });
  });

  describe('releaseFunds', () => {
    it('should pass proposal_id in payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...FUND_ALLOCATION_ENTRY, status: 'Released' })
      );

      const result = await client.releaseFunds('proposal-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'release_locked_funds',
          payload: {
            proposal_id: 'proposal-1',
          },
        })
      );
      expect(result.status).toBe('Released');
    });
  });

  describe('refundFunds', () => {
    it('should pass proposal_id in payload', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord({ ...FUND_ALLOCATION_ENTRY, status: 'Refunded' })
      );

      const result = await client.refundFunds('proposal-1');

      expect(mockAppClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'refund_locked_funds',
          payload: {
            proposal_id: 'proposal-1',
          },
        })
      );
      expect(result.status).toBe('Refunded');
    });
  });

  describe('getFundAllocation', () => {
    it('should return fund allocation for valid proposal', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockRecord(FUND_ALLOCATION_ENTRY)
      );

      const result = await client.getFundAllocation('proposal-1');
      expect(result).not.toBeNull();
      expect(result!.amount).toBe(5000);
    });

    it('should return null for missing allocation', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

      const result = await client.getFundAllocation('nonexistent');
      expect(result).toBeNull();
    });
  });

  // --------------------------------------------------------------------------
  // Convenience Methods
  // --------------------------------------------------------------------------

  describe('getStatusDescription', () => {
    it('should return description for each status', () => {
      expect(client.getStatusDescription('Pending')).toContain('delay');
      expect(client.getStatusDescription('Ready')).toContain('ready');
      expect(client.getStatusDescription('Executed')).toContain('executed');
      expect(client.getStatusDescription('Cancelled')).toContain('cancelled');
      expect(client.getStatusDescription('Failed')).toContain('failed');
    });
  });

  // --------------------------------------------------------------------------
  // Error Handling
  // --------------------------------------------------------------------------

  describe('error handling', () => {
    it('should propagate zome errors on create', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Proposal not passed')
      );

      await expect(
        client.createTimelock({ proposalId: 'p-1', actions: '{}', delayHours: 48 })
      ).rejects.toThrow();
    });

    it('should propagate zome errors on execute', async () => {
      (mockAppClient.callZome as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error('Timelock not ready')
      );

      await expect(
        client.executeTimelock({ timelockId: 'timelock-1' })
      ).rejects.toThrow();
    });
  });
});
