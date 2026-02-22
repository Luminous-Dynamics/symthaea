/**
 * Resources Client Tests
 *
 * Verifies zome call arguments and payload passthrough for the ResourcesClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ResourcesClient } from '../resources';

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

describe('ResourcesClient', () => {
  let client: ResourcesClient;
  let mockClient: ReturnType<typeof createMockClient>;

  beforeEach(() => {
    mockClient = createMockClient();
    client = new ResourcesClient(mockClient, { roleName: 'hearth', timeout: 30000 });
  });

  // --------------------------------------------------------------------------
  // Resource Inventory
  // --------------------------------------------------------------------------

  describe('resource inventory', () => {
    it('registerResource calls hearth_resources/register_resource with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, name: 'Family Car', category: 'Vehicle', quantity: 1 };

      await client.registerResource(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'hearth',
          zome_name: 'hearth_resources',
          fn_name: 'register_resource',
          payload: input,
        })
      );
    });

    it('getHearthInventory calls get_hearth_inventory with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getHearthInventory(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_hearth_inventory',
          payload: HASH,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Loans
  // --------------------------------------------------------------------------

  describe('loans', () => {
    it('lendResource calls lend_resource with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { resource_hash: HASH, borrower: AGENT, due_date: 1709000000 };

      await client.lendResource(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'lend_resource',
          payload: input,
        })
      );
    });

    it('returnResource calls return_resource with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);

      await client.returnResource(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'return_resource',
          payload: HASH,
        })
      );
    });

    it('getResourceLoans calls get_resource_loans with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getResourceLoans(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_resource_loans',
          payload: HASH,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Budgets
  // --------------------------------------------------------------------------

  describe('budgets', () => {
    it('createBudgetCategory calls create_budget_category with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, name: 'Groceries', monthly_limit: 800 };

      await client.createBudgetCategory(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_budget_category',
          payload: input,
        })
      );
    });

    it('logExpense calls log_expense with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { category_hash: HASH, amount: 45.50, description: 'Weekly shop' };

      await client.logExpense(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'log_expense',
          payload: input,
        })
      );
    });

    it('getBudgetSummary calls get_budget_summary with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getBudgetSummary(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_budget_summary',
          payload: HASH,
        })
      );
    });
  });
});
