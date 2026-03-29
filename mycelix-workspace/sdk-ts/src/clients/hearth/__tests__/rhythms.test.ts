// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Rhythms Client Tests
 *
 * Verifies zome call arguments and payload passthrough for the RhythmsClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { RhythmsClient } from '../rhythms';

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

describe('RhythmsClient', () => {
  let client: RhythmsClient;
  let mockClient: ReturnType<typeof createMockClient>;

  beforeEach(() => {
    mockClient = createMockClient();
    client = new RhythmsClient(mockClient, { roleName: 'hearth', timeout: 30000 });
  });

  // --------------------------------------------------------------------------
  // Rhythm Management
  // --------------------------------------------------------------------------

  describe('rhythm management', () => {
    it('createRhythm calls hearth_rhythms/create_rhythm with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, name: 'Morning walk', frequency: 'Daily' };

      await client.createRhythm(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'hearth',
          zome_name: 'hearth_rhythms',
          fn_name: 'create_rhythm',
          payload: input,
        })
      );
    });

    it('getHearthRhythms calls get_hearth_rhythms with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getHearthRhythms(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_hearth_rhythms',
          payload: HASH,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Occurrences
  // --------------------------------------------------------------------------

  describe('occurrences', () => {
    it('logOccurrence calls log_occurrence with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { rhythm_hash: HASH, notes: 'Beautiful sunrise', participants: [] };

      await client.logOccurrence(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'log_occurrence',
          payload: input,
        })
      );
    });

    it('getRhythmOccurrences calls get_rhythm_occurrences with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getRhythmOccurrences(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_rhythm_occurrences',
          payload: HASH,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Presence
  // --------------------------------------------------------------------------

  describe('presence', () => {
    it('setPresence calls set_presence with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, status: 'Home', note: 'Working from home' };

      await client.setPresence(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'set_presence',
          payload: input,
        })
      );
    });

    it('getHearthPresence calls get_hearth_presence with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getHearthPresence(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_hearth_presence',
          payload: HASH,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Digests
  // --------------------------------------------------------------------------

  describe('digests', () => {
    it('createRhythmDigest calls create_rhythm_digest with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, epoch_start: 1708200000, epoch_end: 1708804800 };

      await client.createRhythmDigest(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_rhythm_digest',
          payload: input,
        })
      );
    });
  });
});
