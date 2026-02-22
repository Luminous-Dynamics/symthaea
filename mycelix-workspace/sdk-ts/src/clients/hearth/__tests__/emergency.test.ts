/**
 * Emergency Client Tests
 *
 * Verifies zome call arguments and payload passthrough for the EmergencyClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { EmergencyClient } from '../emergency';

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

describe('EmergencyClient', () => {
  let client: EmergencyClient;
  let mockClient: ReturnType<typeof createMockClient>;

  beforeEach(() => {
    mockClient = createMockClient();
    client = new EmergencyClient(mockClient, { roleName: 'hearth', timeout: 30000 });
  });

  // --------------------------------------------------------------------------
  // Emergency Plans
  // --------------------------------------------------------------------------

  describe('emergency plans', () => {
    it('createEmergencyPlan calls hearth_emergency/create_emergency_plan with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, plan_type: 'Fire', contacts: [], procedures: [] };

      await client.createEmergencyPlan(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'hearth',
          zome_name: 'hearth_emergency',
          fn_name: 'create_emergency_plan',
          payload: input,
        })
      );
    });

    it('updateEmergencyPlan calls update_emergency_plan with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { plan_hash: HASH, contacts: [AGENT] };

      await client.updateEmergencyPlan(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'update_emergency_plan',
          payload: input,
        })
      );
    });

    it('getEmergencyPlan calls get_emergency_plan with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);

      await client.getEmergencyPlan(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_emergency_plan',
          payload: HASH,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Alerts
  // --------------------------------------------------------------------------

  describe('alerts', () => {
    it('raiseAlert calls raise_alert with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, alert_type: 'Medical', description: 'Fall injury' };

      await client.raiseAlert(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'raise_alert',
          payload: input,
        })
      );
    });

    it('resolveAlert calls resolve_alert with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);

      await client.resolveAlert(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'resolve_alert',
          payload: HASH,
        })
      );
    });

    it('getActiveAlerts calls get_active_alerts with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getActiveAlerts(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_active_alerts',
          payload: HASH,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Safety Check-Ins
  // --------------------------------------------------------------------------

  describe('safety check-ins', () => {
    it('checkIn calls check_in with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { alert_hash: HASH, status: 'Safe', notes: 'Everyone accounted for' };

      await client.checkIn(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'check_in',
          payload: input,
        })
      );
    });

    it('getAlertCheckins calls get_alert_checkins with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getAlertCheckins(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_alert_checkins',
          payload: HASH,
        })
      );
    });
  });
});
