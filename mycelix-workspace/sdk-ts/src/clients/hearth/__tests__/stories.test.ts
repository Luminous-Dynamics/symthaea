/**
 * Stories Client Tests
 *
 * Verifies zome call arguments and payload passthrough for the StoriesClient.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { StoriesClient } from '../stories';

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

describe('StoriesClient', () => {
  let client: StoriesClient;
  let mockClient: ReturnType<typeof createMockClient>;

  beforeEach(() => {
    mockClient = createMockClient();
    client = new StoriesClient(mockClient, { roleName: 'hearth', timeout: 30000 });
  });

  // --------------------------------------------------------------------------
  // Stories
  // --------------------------------------------------------------------------

  describe('stories', () => {
    it('createStory calls hearth_stories/create_story with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, title: 'Grandma\'s Recipe', content: 'Once upon a time...' };

      await client.createStory(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'hearth',
          zome_name: 'hearth_stories',
          fn_name: 'create_story',
          payload: input,
        })
      );
    });

    it('updateStory calls update_story with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { story_hash: HASH, title: 'Updated Title', content: 'New content' };

      await client.updateStory(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'update_story',
          payload: input,
        })
      );
    });

    it('addMediaToStory calls add_media_to_story with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { story_hash: HASH, media_type: 'Image', data: new Uint8Array(64) };

      await client.addMediaToStory(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'add_media_to_story',
          payload: input,
        })
      );
    });

    it('getHearthStories calls get_hearth_stories with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getHearthStories(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_hearth_stories',
          payload: HASH,
        })
      );
    });

    it('searchStoriesByTag calls search_stories_by_tag with compound payload', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.searchStoriesByTag(HASH, 'holiday');

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'search_stories_by_tag',
          payload: { hearth_hash: HASH, tag: 'holiday' },
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Collections
  // --------------------------------------------------------------------------

  describe('collections', () => {
    it('createCollection calls create_collection with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, name: 'Holiday Photos', description: 'Winter 2025' };

      await client.createCollection(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_collection',
          payload: input,
        })
      );
    });

    it('addToCollection calls add_to_collection with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { collection_hash: HASH, story_hash: HASH };

      await client.addToCollection(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'add_to_collection',
          payload: input,
        })
      );
    });

    it('getHearthCollections calls get_hearth_collections with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getHearthCollections(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_hearth_collections',
          payload: HASH,
        })
      );
    });
  });

  // --------------------------------------------------------------------------
  // Traditions
  // --------------------------------------------------------------------------

  describe('traditions', () => {
    it('createTradition calls create_tradition with input', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);
      const input = { hearth_hash: HASH, name: 'Sunday Brunch', frequency: 'Weekly' };

      await client.createTradition(input as any);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'create_tradition',
          payload: input,
        })
      );
    });

    it('observeTradition calls observe_tradition with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce(MOCK_RECORD);

      await client.observeTradition(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'observe_tradition',
          payload: HASH,
        })
      );
    });

    it('getHearthTraditions calls get_hearth_traditions with hash', async () => {
      mockClient.callZome.mockResolvedValueOnce([MOCK_RECORD]);

      await client.getHearthTraditions(HASH);

      expect(mockClient.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          fn_name: 'get_hearth_traditions',
          payload: HASH,
        })
      );
    });
  });
});
