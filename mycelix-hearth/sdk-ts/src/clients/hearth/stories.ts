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

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_stories';

export class StoriesClient {
  constructor(private readonly client: AppClient, private readonly roleName = ROLE_NAME) {}

  async createStory(input: CreateStoryInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'create_story',
      payload: input,
    });
  }

  async updateStory(input: UpdateStoryInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'update_story',
      payload: input,
    });
  }

  async addMediaToStory(input: AddMediaInput): Promise<void> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'add_media_to_story',
      payload: input,
    });
  }

  async createCollection(input: CreateCollectionInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'create_collection',
      payload: input,
    });
  }

  async addToCollection(input: AddToCollectionInput): Promise<void> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'add_to_collection',
      payload: input,
    });
  }

  async createTradition(input: CreateTraditionInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'create_tradition',
      payload: input,
    });
  }

  async observeTradition(traditionHash: ActionHash): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'observe_tradition',
      payload: traditionHash,
    });
  }

  async getHearthStories(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_hearth_stories',
      payload: hearthHash,
    });
  }

  async getHearthTraditions(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_hearth_traditions',
      payload: hearthHash,
    });
  }

  async getHearthCollections(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_hearth_collections',
      payload: hearthHash,
    });
  }

  async searchStoriesByTag(tag: string): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'search_stories_by_tag',
      payload: tag,
    });
  }
}
