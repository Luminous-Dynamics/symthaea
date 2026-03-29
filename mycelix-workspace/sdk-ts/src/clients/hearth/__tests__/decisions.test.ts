// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Decisions Client Tests
 *
 * Verifies zome call arguments and payload passthrough for the DecisionsClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { DecisionsClient } from '../decisions';

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

describe('DecisionsClient', () => {
  let client: DecisionsClient;
  let mockClient: ReturnType<typeof createMockClient>;

  beforeEach(() => {
    mockClient = createMockClient();
    client = new DecisionsClient(mockClient, { roleName: 'hearth', timeout: 30000 });
  });

  // --------------------------------------------------------------------------
  // Decision Management
  // --------------------------------------------------------------------------

  describe('decision management', () => {
    it('createDecision calls hearth_decisions/create_decision with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, title: 'Movie night', decision_type: 'Consent' };

      await client.createDecision(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'hearth',
          zome_name: 'hearth_decisions',
          fn_name: 'create_decision',
          payload: input,
        })
      );
    });

    it('getDecision calls get_decision with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);

      await client.getDecision(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          zome_name: 'hearth_decisions',
          fn_name: 'get_decision',
          payload: HASH,
        })
      );
    });

    it('getHearthDecisions calls get_hearth_decisions with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getHearthDecisions(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_hearth_decisions',
          payload: HASH,
        })
      );
    });

    it('finalizeDecision calls finalize_decision with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { decision_hash: HASH, outcome: 'Approved' };

      await client.finalizeDecision(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'finalize_decision',
          payload: input,
        })
      );
    });

    it('closeDecision calls close_decision with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { decision_hash: HASH, reason: 'No longer relevant' };

      await client.closeDecision(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'close_decision',
          payload: input,
        })
      );
    });

    it('getDecisionOutcome calls get_decision_outcome with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);

      await client.getDecisionOutcome(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_decision_outcome',
          payload: HASH,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Voting
  // --------------------------------------------------------------------------

  describe('voting', () => {
    it('castVote calls cast_vote with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { decision_hash: HASH, choice: 0, weight: 1, rationale: 'I agree' };

      await client.castVote(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'cast_vote',
          payload: input,
        })
      );
    });

    it('amendVote calls amend_vote with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { vote_hash: HASH, new_choice: 1, rationale: 'Changed my mind' };

      await client.amendVote(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'amend_vote',
          payload: input,
        })
      );
    });

    it('tallyVotes calls tally_votes with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([[0, 3], [1, 1]]);

      const result = await client.tallyVotes(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'tally_votes',
          payload: HASH,
        })
      );
      expect(result).toEqual([[0, 3], [1, 1]]);
    });

    it('getDecisionVotes calls get_decision_votes with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getDecisionVotes(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_decision_votes',
          payload: HASH,
        })
      );
    });

    it('getVoteHistory calls get_vote_history with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getVoteHistory(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_vote_history',
          payload: HASH,
        })
      );
    });

    it('getMyPendingVotes calls get_my_pending_votes with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getMyPendingVotes(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_my_pending_votes',
          payload: HASH,
        })
      );
    });
  });
});
