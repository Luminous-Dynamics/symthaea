// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Gratitude Client Tests
 *
 * Verifies zome call arguments and payload passthrough for the GratitudeClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { GratitudeClient } from '../gratitude';

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

describe('GratitudeClient', () => {
  let client: GratitudeClient;
  let mockClient: ReturnType<typeof createMockClient>;

  beforeEach(() => {
    mockClient = createMockClient();
    client = new GratitudeClient(mockClient, { roleName: 'hearth', timeout: 30000 });
  });

  // --------------------------------------------------------------------------
  // Expressions
  // --------------------------------------------------------------------------

  describe('expressions', () => {
    it('expressGratitude calls hearth_gratitude/express_gratitude with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, recipient: AGENT, message: 'Thank you for cooking' };

      await client.expressGratitude(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'hearth',
          zome_name: 'hearth_gratitude',
          fn_name: 'express_gratitude',
          payload: input,
        })
      );
    });

    it('getGratitudeStream calls get_gratitude_stream with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getGratitudeStream(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'hearth_gratitude',
          fn_name: 'get_gratitude_stream',
          payload: HASH,
        })
      );
    });

    it('getGratitudeBalance calls get_gratitude_balance with hash', async () => {
      const balanceAnchor = { total_given: 12, total_received: 8 };
      mockClient.callZome.mockResolvedValueOnce(balanceAnchor);

      const result = await client.getGratitudeBalance(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_gratitude_balance',
          payload: HASH,
        })
      );
      expect(result).toEqual(balanceAnchor);
    });
  });

  // --------------------------------------------------------------------------
  // Appreciation Circles
  // --------------------------------------------------------------------------

  describe('appreciation circles', () => {
    it('startCircle calls start_appreciation_circle with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, theme: 'Weekly appreciation' };

      await client.startCircle(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'start_appreciation_circle',
          payload: input,
        })
      );
    });

    it('joinCircle calls join_circle with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);

      await client.joinCircle(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'join_circle',
          payload: HASH,
        })
      );
    });

    it('completeCircle calls complete_circle with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);

      await client.completeCircle(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'complete_circle',
          payload: HASH,
        })
      );
    });

    it('getHearthCircles calls get_hearth_circles with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getHearthCircles(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_hearth_circles',
          payload: HASH,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Digests
  // --------------------------------------------------------------------------

  describe('digests', () => {
    it('createGratitudeDigest calls create_gratitude_digest with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, epoch_start: 1708200000, epoch_end: 1708804800 };

      await client.createGratitudeDigest(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_gratitude_digest',
          payload: input,
        })
      );
    });
  });
});
