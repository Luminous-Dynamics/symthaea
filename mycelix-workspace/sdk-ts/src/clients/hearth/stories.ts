/**
 * Stories Zome Client
 *
 * Family stories, traditions, collections, and timelines for Hearth clusters.
 *
 * @module @mycelix/sdk/clients/hearth/stories
 */

import type {
  CreateStoryInput,
  CreateCollectionInput,
  CreateTraditionInput,
  FamilyStory,
  StoryCollection,
  FamilyTradition,
} from './types';
import type { ActionHash } from '../../generated/common';

export interface StoriesClientConfig {
  roleName?: string;
  timeout?: number;
}

interface ZomeCallable {
  callZome<T>(params: { role_name: string; zome_name: string; fn_name: string; payload: unknown }): Promise<T>;
}

export class StoriesClient {
  private readonly zomeName = 'hearth_stories';

  constructor(
    private readonly client: ZomeCallable,
    private readonly config: Required<Pick<StoriesClientConfig, 'roleName' | 'timeout'>>,
  ) {}

  // ============================================================================
  // Stories
  // ============================================================================

  async createStory(input: CreateStoryInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_story', payload: input });
  }

  async updateStory(storyHash: ActionHash, input: Partial<CreateStoryInput>) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'update_story', payload: { story_hash: storyHash, ...input } });
  }

  async addMediaToStory(storyHash: ActionHash, mediaHash: ActionHash) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'add_media_to_story', payload: { story_hash: storyHash, media_hash: mediaHash } });
  }

  async getHearthStories(hearthHash: ActionHash) {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_hearth_stories', payload: hearthHash });
  }

  async searchStoriesByTag(hearthHash: ActionHash, tag: string) {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'search_stories_by_tag', payload: { hearth_hash: hearthHash, tag } });
  }

  async getFamilyTimeline(hearthHash: ActionHash) {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_family_timeline', payload: hearthHash });
  }

  // ============================================================================
  // Collections
  // ============================================================================

  async createCollection(input: CreateCollectionInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_collection', payload: input });
  }

  async addToCollection(collectionHash: ActionHash, storyHash: ActionHash) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'add_to_collection', payload: { collection_hash: collectionHash, story_hash: storyHash } });
  }

  // ============================================================================
  // Traditions
  // ============================================================================

  async createTradition(input: CreateTraditionInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_tradition', payload: input });
  }

  async observeTradition(traditionHash: ActionHash) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'observe_tradition', payload: traditionHash });
  }

  async getUpcomingTraditions(hearthHash: ActionHash) {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_upcoming_traditions', payload: hearthHash });
  }
}
