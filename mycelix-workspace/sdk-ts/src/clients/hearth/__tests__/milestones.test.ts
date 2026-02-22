/**
 * Milestones Client Tests
 *
 * Verifies zome call arguments and payload passthrough for the MilestonesClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MilestonesClient } from '../milestones';

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

describe('MilestonesClient', () => {
  let client: MilestonesClient;
  let mockClient: ReturnType<typeof createMockClient>;

  beforeEach(() => {
    mockClient = createMockClient();
    client = new MilestonesClient(mockClient, { roleName: 'hearth', timeout: 30000 });
  });

  // --------------------------------------------------------------------------
  // Milestones
  // --------------------------------------------------------------------------

  describe('milestones', () => {
    it('recordMilestone calls hearth_milestones/record_milestone with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, title: 'First steps', milestone_type: 'Development' };

      await client.recordMilestone(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'hearth',
          zome_name: 'hearth_milestones',
          fn_name: 'record_milestone',
          payload: input,
        })
      );
    });

    it('getFamilyTimeline calls get_family_timeline with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getFamilyTimeline(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_family_timeline',
          payload: HASH,
        })
      );
    });

    it('getMemberMilestones calls get_member_milestones with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getMemberMilestones(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_member_milestones',
          payload: HASH,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Life Transitions
  // --------------------------------------------------------------------------

  describe('life transitions', () => {
    it('beginTransition calls begin_transition with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, member: HASH, transition_type: 'SchoolStart' };

      await client.beginTransition(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'begin_transition',
          payload: input,
        })
      );
    });

    it('advanceTransition calls advance_transition with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { transition_hash: HASH, notes: 'Progressing well' };

      await client.advanceTransition(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'advance_transition',
          payload: input,
        })
      );
    });

    it('completeTransition calls complete_transition with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);

      await client.completeTransition(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'complete_transition',
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
