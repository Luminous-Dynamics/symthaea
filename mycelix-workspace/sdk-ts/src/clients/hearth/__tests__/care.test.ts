/**
 * Hearth Care Client Tests
 *
 * Verifies zome call arguments and payload passthrough for the HearthCareClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { HearthCareClient } from '../care';

// ============================================================================
// MOCK HELPERS
// ============================================================================

function createMockClient() {
  return { callZome: vi.fn() } as any;
}

const HASH = new Uint8Array(32);
const MOCK_RECORD = { entry: { Present: {} }, signed_action: { hashed: { hash: HASH } } };

// ============================================================================
// TESTS
// ============================================================================

describe('HearthCareClient', () => {
  let client: HearthCareClient;
  let mockClient: ReturnType<typeof createMockClient>;

  beforeEach(() => {
    mockClient = createMockClient();
    client = new HearthCareClient(mockClient, { roleName: 'hearth', timeout: 30000 });
  });

  // --------------------------------------------------------------------------
  // Care Schedules
  // --------------------------------------------------------------------------

  describe('care schedules', () => {
    it('createCareSchedule calls hearth_care/create_care_schedule with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, title: 'Morning routine', assignees: [] };

      await client.createCareSchedule(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'hearth',
          zome_name: 'hearth_care',
          fn_name: 'create_care_schedule',
          payload: input,
        })
      );
    });

    it('completeTask calls complete_task with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);

      await client.completeTask(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'complete_task',
          payload: HASH,
        })
      );
    });

    it('getMyCareDuties calls get_my_care_duties with null payload', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getMyCareDuties();

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'hearth_care',
          fn_name: 'get_my_care_duties',
          payload: null,
        })
      );
    });

    it('getHearthSchedule calls get_hearth_schedule with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getHearthSchedule(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_hearth_schedule',
          payload: HASH,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Care Swaps
  // --------------------------------------------------------------------------

  describe('care swaps', () => {
    it('proposeSwap calls propose_swap with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { schedule_hash: HASH, proposed_to: new Uint8Array(39), reason: 'Unavailable' };

      await client.proposeSwap(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'propose_swap',
          payload: input,
        })
      );
    });

    it('acceptSwap calls accept_swap with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);

      await client.acceptSwap(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'accept_swap',
          payload: HASH,
        })
      );
    });

    it('declineSwap calls decline_swap with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);

      await client.declineSwap(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'decline_swap',
          payload: HASH,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Meal Plans
  // --------------------------------------------------------------------------

  describe('meal plans', () => {
    it('createMealPlan calls create_meal_plan with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, week_start: 1708200000, meals: [] };

      await client.createMealPlan(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_meal_plan',
          payload: input,
        })
      );
    });

    it('getHearthMealPlans calls get_hearth_meal_plans with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getHearthMealPlans(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_hearth_meal_plans',
          payload: HASH,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Digests
  // --------------------------------------------------------------------------

  describe('digests', () => {
    it('createCareDigest calls create_care_digest with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, epoch_start: 1708200000, epoch_end: 1708804800 };

      await client.createCareDigest(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_care_digest',
          payload: input,
        })
      );
    });
  });
});
