/**
 * Hearth Stories SDK client.
 * Wraps zome calls to the hearth-stories coordinator.
 */

import type { AppClient, Record as HolochainRecord, ActionHash } from '@holochain/client';
import type {
  CreateStoryInput,
  UpdateStoryInput,
  AddMediaInput,
  CreateCollectionInput,
  AddToCollectionInput,
  CreateTraditionInput,
} from './types';
import { HearthError, classifyError } from './errors';

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_stories';

export class StoriesClient {
  constructor(private readonly client: AppClient, private readonly roleName = ROLE_NAME) {}

  // ============================================================================
  // Zome Calls
  // ============================================================================

  private async callZome<T>(fnName: string, payload: unknown): Promise<T> {
    try {
      return await this.client.callZome({
        role_name: this.roleName,
        zome_name: ZOME_NAME,
        fn_name: fnName,
        payload,
      });
    } catch (err) {
      throw new HearthError({
        code: classifyError(err),
        message: `${ZOME_NAME}.${fnName} failed: ${err}`,
        zome: ZOME_NAME,
        fnName,
        cause: err,
      });
    }
  }

  /** Create a new family story. */
  async createStory(input: CreateStoryInput): Promise<HolochainRecord> {
    return this.callZome('create_story', input);
  }

  /** Update an existing family story. */
  async updateStory(input: UpdateStoryInput): Promise<HolochainRecord> {
    return this.callZome('update_story', input);
  }

  /** Add media (photo, video, audio) to a story. */
  async addMediaToStory(input: AddMediaInput): Promise<void> {
    return this.callZome('add_media_to_story', input);
  }

  /** Create a new story collection. */
  async createCollection(input: CreateCollectionInput): Promise<HolochainRecord> {
    return this.callZome('create_collection', input);
  }

  /** Add a story to an existing collection. */
  async addToCollection(input: AddToCollectionInput): Promise<void> {
    return this.callZome('add_to_collection', input);
  }

  /** Create a new family tradition. */
  async createTradition(input: CreateTraditionInput): Promise<HolochainRecord> {
    return this.callZome('create_tradition', input);
  }

  /** Record an observation of a family tradition. */
  async observeTradition(traditionHash: ActionHash): Promise<HolochainRecord> {
    return this.callZome('observe_tradition', traditionHash);
  }

  /** Get all stories for a hearth. */
  async getHearthStories(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_hearth_stories', hearthHash);
  }

  /** Get all traditions for a hearth. */
  async getHearthTraditions(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_hearth_traditions', hearthHash);
  }

  /** Get all story collections for a hearth. */
  async getHearthCollections(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_hearth_collections', hearthHash);
  }

  /** Search stories by tag. */
  async searchStoriesByTag(tag: string): Promise<HolochainRecord[]> {
    return this.callZome('search_stories_by_tag', tag);
  }
}
