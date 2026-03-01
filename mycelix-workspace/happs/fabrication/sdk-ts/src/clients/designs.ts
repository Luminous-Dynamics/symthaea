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
  DesignCategory,
  CreateDesignInput,
  UpdateDesignInput,
  DesignSearchQuery,
  ParametricSchema,
  GeneratedVariant,
  ParameterValue,
} from '../types';

export interface PaginationInput {
  offset: number;
  limit: number;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  offset: number;
  limit: number;
}

export interface ForkDesignInput {
  parent_hash: ActionHash;
  modification_notes: string;
  title?: string;
  description?: string;
  intent_modifications?: Array<{
    concept: string;
    role: string;
    weight: number;
  }>;
}

export interface GenerateVariantInput {
  design_hash: ActionHash;
  parameters: { [key: string]: ParameterValue };
}

export class DesignsClient {
  constructor(
    private client: AppClient,
    private roleName: string,
    private zomeName: string = 'designs_coordinator'
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
   * Add a file to a design
   */
  async addFile(input: { design_hash: ActionHash; file: { filename: string; file_type: string; ipfs_cid: string; size_bytes: number; checksum_sha256: string } }): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'add_design_file',
      payload: input,
    });
  }

  /**
   * Fork a design to create a derivative
   */
  async fork(input: ForkDesignInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'fork_design',
      payload: input,
    });
  }

  /**
   * Get version history of a design
   */
  async getHistory(
    hash: ActionHash,
    pagination?: PaginationInput
  ): Promise<PaginatedResponse<Record>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_design_history',
      payload: { hash, pagination: pagination ?? null },
    });
  }

  /**
   * Get all forks of a design
   */
  async getForks(
    hash: ActionHash,
    pagination?: PaginationInput
  ): Promise<PaginatedResponse<Record>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_design_forks',
      payload: { hash, pagination: pagination ?? null },
    });
  }

  /**
   * Get design files
   */
  async getFiles(
    hash: ActionHash,
    pagination?: PaginationInput
  ): Promise<PaginatedResponse<Record>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_design_files',
      payload: { hash, pagination: pagination ?? null },
    });
  }

  /**
   * Get all designs by a specific author
   */
  async getByAuthor(
    author: AgentPubKey,
    pagination?: PaginationInput
  ): Promise<PaginatedResponse<Record>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_designs_by_author',
      payload: { author, pagination: pagination ?? null },
    });
  }

  /**
   * Get designs by category
   */
  async getByCategory(
    category: DesignCategory,
    pagination?: PaginationInput
  ): Promise<PaginatedResponse<Record>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_designs_by_category',
      payload: { category, pagination: pagination ?? null },
    });
  }

  /**
   * Search designs with flexible query
   */
  async search(query: DesignSearchQuery): Promise<PaginatedResponse<Record>> {
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
  async getFeatured(pagination?: PaginationInput): Promise<PaginatedResponse<Record>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_featured_designs',
      payload: { pagination: pagination ?? null },
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
  async generateVariant(input: GenerateVariantInput): Promise<GeneratedVariant> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'generate_variant',
      payload: input,
    });
  }
}
