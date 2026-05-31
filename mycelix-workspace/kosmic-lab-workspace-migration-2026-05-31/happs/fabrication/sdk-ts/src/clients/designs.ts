// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Designs Client
 *
 * Handles HDC-enhanced parametric design operations including:
 * - CRUD operations for designs
 * - Versioning and forking
 * - Parametric variant generation
 * - Discovery and search
 */

import type { AppClient, ActionHash, AgentPubKey, Record } from '@holochain/client';
import type {
  Design,
  DesignCategory,
  CreateDesignInput,
  UpdateDesignInput,
  DesignSearchQuery,
  DesignFile,
  ParametricSchema,
} from '../types';

export class DesignsClient {
  constructor(
    private client: AppClient,
    private roleName: string,
    private zomeName: string = 'designs'
  ) {}

  /**
   * Create a new design
   */
  async create(input: CreateDesignInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'create_design',
      payload: input,
    });
  }

  /**
   * Get a design by hash
   */
  async get(hash: ActionHash): Promise<Record | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_design',
      payload: hash,
    });
  }

  /**
   * Update an existing design
   */
  async update(input: UpdateDesignInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'update_design',
      payload: input,
    });
  }

  /**
   * Delete a design (soft delete)
   */
  async delete(hash: ActionHash): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'delete_design',
      payload: hash,
    });
  }

  /**
   * Fork a design to create a derivative
   */
  async fork(
    parentHash: ActionHash,
    modifications: { title?: string; description?: string; files?: DesignFile[] }
  ): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'fork_design',
      payload: { parent_hash: parentHash, modifications },
    });
  }

  /**
   * Get version history of a design
   */
  async getHistory(hash: ActionHash): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_design_history',
      payload: hash,
    });
  }

  /**
   * Get all forks of a design
   */
  async getForks(hash: ActionHash): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_design_forks',
      payload: hash,
    });
  }

  /**
   * Get all designs by a specific author
   */
  async getByAuthor(author: AgentPubKey): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_designs_by_author',
      payload: author,
    });
  }

  /**
   * Get my designs
   */
  async getMine(): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_my_designs',
      payload: null,
    });
  }

  /**
   * Get designs by category
   */
  async getByCategory(category: DesignCategory): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_designs_by_category',
      payload: category,
    });
  }

  /**
   * Search designs with flexible query
   */
  async search(query: DesignSearchQuery): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'search_designs',
      payload: query,
    });
  }

  /**
   * Get featured/popular designs
   */
  async getFeatured(limit: number = 10): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_featured_designs',
      payload: limit,
    });
  }

  /**
   * Get parametric schema for a design
   */
  async getParameters(hash: ActionHash): Promise<ParametricSchema | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_parameters',
      payload: hash,
    });
  }

  /**
   * Generate a parametric variant with custom parameters
   */
  async generateVariant(
    hash: ActionHash,
    params: Record<string, string | number | boolean>
  ): Promise<DesignFile> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'generate_variant',
      payload: { design_hash: hash, parameters: params },
    });
  }
}
