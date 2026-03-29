// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Autonomy Client Tests
 *
 * Verifies zome call arguments and payload passthrough for the AutonomyClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { AutonomyClient } from '../autonomy';

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

describe('AutonomyClient', () => {
  let client: AutonomyClient;
  let mockClient: ReturnType<typeof createMockClient>;

  beforeEach(() => {
    mockClient = createMockClient();
    client = new AutonomyClient(mockClient, { roleName: 'hearth', timeout: 30000 });
  });

  // --------------------------------------------------------------------------
  // Profile Management
  // --------------------------------------------------------------------------

  describe('profile management', () => {
    it('createProfile calls hearth_autonomy/create_autonomy_profile with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, member: AGENT, initial_tier: 'Dependent' };

      await client.createProfile(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'hearth',
          zome_name: 'hearth_autonomy',
          fn_name: 'create_autonomy_profile',
          payload: input,
        })
      );
    });

    it('getAutonomyProfile calls get_autonomy_profile with agent', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);

      await client.getAutonomyProfile(AGENT);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'hearth_autonomy',
          fn_name: 'get_autonomy_profile',
          payload: AGENT,
        })
      );
    });

    it('checkCapability calls check_capability with input and returns boolean', async () => {
      mockClient.callZome.mockResolvedValueOnce(true);
      const input = { member: AGENT, capability: 'manage_budget' };

      const result = await client.checkCapability(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'check_capability',
          payload: input,
        })
      );
      expect(result).toBe(true);
    });

    it('advanceTier calls advance_tier with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { member: AGENT, new_tier: 'Independent', evidence: 'Completed all tasks' };

      await client.advanceTier(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'advance_tier',
          payload: input,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Capability Requests
  // --------------------------------------------------------------------------

  describe('capability requests', () => {
    it('requestCapability calls request_capability with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, capability: 'cook_dinner', reason: 'I want to help' };

      await client.requestCapability(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'request_capability',
          payload: input,
        })
      );
    });

    it('approveCapability calls approve_capability with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { request_hash: HASH, notes: 'Well deserved' };

      await client.approveCapability(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'approve_capability',
          payload: input,
        })
      );
    });

    it('denyCapability calls deny_capability with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { request_hash: HASH, notes: 'Not ready yet' };

      await client.denyCapability(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'deny_capability',
          payload: input,
        })
      );
    });

    it('getPendingRequests calls get_pending_requests with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getPendingRequests(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_pending_requests',
          payload: HASH,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Tier Transitions
  // --------------------------------------------------------------------------

  describe('tier transitions', () => {
    it('progressTransition calls progress_transition with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);

      await client.progressTransition(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'progress_transition',
          payload: HASH,
        })
      );
    });

    it('getActiveTransitions calls get_active_transitions with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getActiveTransitions(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_active_transitions',
          payload: HASH,
        })
      );
    });
  });
});
