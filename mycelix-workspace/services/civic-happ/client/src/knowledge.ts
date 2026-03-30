// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Civic Knowledge Client
 *
 * Provides methods to interact with the civic_knowledge zome.
 */

import type { AppClient, ActionHash, RoleNameCallZomeRequest } from '@holochain/client';
import type {
  CivicDomain,
  CivicKnowledge,
  CreateKnowledgeInput,
  KnowledgeSearchResult,
  KnowledgeUpdate,
  KnowledgeValidation,
  SearchInput,
  UpdateKnowledgeInput,
  ValidateKnowledgeInput,
} from './types.js';

export class CivicKnowledgeClient {
  private client: AppClient;
  private roleName: string;
  private zomeName: string;

  constructor(client: AppClient, roleName = 'civic', zomeName = 'civic_knowledge') {
    this.client = client;
    this.roleName = roleName;
    this.zomeName = zomeName;
  }

  private async callZome<T>(fnName: string, payload: unknown): Promise<T> {
    const result = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: fnName,
      payload,
    } as RoleNameCallZomeRequest);
    return result as T;
  }

  /**
   * Create a new civic knowledge entry
   */
  async createKnowledge(input: CreateKnowledgeInput): Promise<ActionHash> {
    return this.callZome('create_knowledge', input);
  }

  /**
   * Get a specific knowledge entry by hash
   */
  async getKnowledge(actionHash: ActionHash): Promise<CivicKnowledge | null> {
    return this.callZome('get_knowledge', actionHash);
  }

  /**
   * Search by civic domain
   */
  async searchByDomain(domain: CivicDomain): Promise<KnowledgeSearchResult[]> {
    return this.callZome('search_by_domain', domain);
  }

  /**
   * Search by geographic scope
   */
  async searchByLocation(geoScope: string): Promise<KnowledgeSearchResult[]> {
    return this.callZome('search_by_location', geoScope);
  }

  /**
   * Search by keyword
   */
  async searchByKeyword(keyword: string): Promise<KnowledgeSearchResult[]> {
    return this.callZome('search_by_keyword', keyword);
  }

  /**
   * Combined search across multiple criteria
   */
  async searchKnowledge(input: SearchInput): Promise<KnowledgeSearchResult[]> {
    return this.callZome('search_knowledge', input);
  }

  /**
   * Submit a validation for a knowledge entry
   */
  async validateKnowledge(input: ValidateKnowledgeInput): Promise<ActionHash> {
    return this.callZome('validate_knowledge', input);
  }

  /**
   * Get all validations for a knowledge entry
   */
  async getKnowledgeValidations(knowledgeHash: ActionHash): Promise<KnowledgeValidation[]> {
    return this.callZome('get_knowledge_validations', knowledgeHash);
  }

  /**
   * Submit an update to existing knowledge
   */
  async updateKnowledge(input: UpdateKnowledgeInput): Promise<ActionHash> {
    return this.callZome('update_knowledge', input);
  }

  /**
   * Get all updates for a knowledge entry
   */
  async getKnowledgeUpdates(knowledgeHash: ActionHash): Promise<KnowledgeUpdate[]> {
    return this.callZome('get_knowledge_updates', knowledgeHash);
  }

  /**
   * Get all knowledge authored by the current agent
   */
  async getMyKnowledge(): Promise<KnowledgeSearchResult[]> {
    return this.callZome('get_my_knowledge', null);
  }

  // ========================================================================
  // Convenience Methods
  // ========================================================================

  /**
   * Search for knowledge relevant to a citizen's question
   * Combines domain detection with keyword search
   */
  async findRelevantKnowledge(
    question: string,
    domain?: CivicDomain,
    location?: string,
  ): Promise<KnowledgeSearchResult[]> {
    // Extract keywords from the question (simple approach)
    const stopWords = new Set([
      'i', 'me', 'my', 'the', 'a', 'an', 'is', 'are', 'was', 'were',
      'do', 'does', 'did', 'have', 'has', 'had', 'can', 'could', 'would',
      'should', 'will', 'what', 'where', 'when', 'how', 'who', 'which',
      'for', 'to', 'of', 'in', 'on', 'at', 'by', 'with', 'about',
    ]);

    const keywords = question
      .toLowerCase()
      .replace(/[^\w\s]/g, '')
      .split(/\s+/)
      .filter(word => word.length > 2 && !stopWords.has(word));

    return this.searchKnowledge({
      domain,
      geo_scope: location,
      keywords,
      limit: 10,
    });
  }

  /**
   * Get the best matching knowledge for a domain
   * Returns entries with the most validations
   */
  async getBestKnowledge(domain: CivicDomain, limit = 5): Promise<KnowledgeSearchResult[]> {
    const results = await this.searchByDomain(domain);

    // Sort by validation count (would need to fetch validations for each)
    // For now, just return the first N results
    return results.slice(0, limit);
  }
}

/**
 * Create a civic knowledge client from an AppClient
 */
export function createKnowledgeClient(
  client: AppClient,
  roleName?: string,
): CivicKnowledgeClient {
  return new CivicKnowledgeClient(client, roleName);
}
