// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Stories Zome Client
 *
 * Family stories, traditions, collections, and timelines for Hearth clusters.
 *
 * @module @mycelix/sdk/clients/hearth/stories
 */

import type {
  CreateStoryInput,
  UpdateStoryInput,
  AddMediaInput,
  CreateCollectionInput,
  AddToCollectionInput,
  CreateTraditionInput,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { Record } from '@holochain/client';

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

  async createStory(input: CreateStoryInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_story', payload: input });
  }

  async updateStory(input: UpdateStoryInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'update_story', payload: input });
  }

  async addMediaToStory(input: AddMediaInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'add_media_to_story', payload: input });
  }

  async getHearthStories(hearthHash: ActionHash): Promise<Record[]> {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_hearth_stories', payload: hearthHash });
  }

  async searchStoriesByTag(hearthHash: ActionHash, tag: string): Promise<Record[]> {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'search_stories_by_tag', payload: { hearth_hash: hearthHash, tag } });
  }

  // ============================================================================
  // Collections
  // ============================================================================

  async createCollection(input: CreateCollectionInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_collection', payload: input });
  }

  async addToCollection(input: AddToCollectionInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'add_to_collection', payload: input });
  }

  async getHearthCollections(hearthHash: ActionHash): Promise<Record[]> {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_hearth_collections', payload: hearthHash });
  }

  // ============================================================================
  // Traditions
  // ============================================================================

  async createTradition(input: CreateTraditionInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_tradition', payload: input });
  }

  async observeTradition(traditionHash: ActionHash): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'observe_tradition', payload: traditionHash });
  }

  async getHearthTraditions(hearthHash: ActionHash): Promise<Record[]> {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_hearth_traditions', payload: hearthHash });
  }
}
