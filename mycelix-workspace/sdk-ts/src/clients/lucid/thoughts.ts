// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * LUCID Thoughts Zome Client
 *
 * Core knowledge graph operations: thoughts, tags, domains, relationships,
 * semantic search, and garden exploration.
 *
 * @module @mycelix/sdk/clients/lucid/thoughts
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  CreateThoughtInput,
  UpdateThoughtInput,
  SearchThoughtsInput,
  CreateTagInput,
  CreateDomainInput,
  CreateRelationshipInput,
  KnowledgeGraphStats,
  UpdateEmbeddingInput,
  UpdateCoherenceInput,
  SemanticSearchInput,
  SemanticSearchResult,
  ExploreGardenInput,
  ThoughtCluster,
  SuggestConnectionsInput,
  ConnectionSuggestion,
  KnowledgeGap,
  DiscoveredPattern,
  CoherenceRangeInput,
  ThoughtType,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface ThoughtsClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: ThoughtsClientConfig = {
  roleName: 'lucid',
};

/**
 * Client for the LUCID core thought operations
 */
export class ThoughtsClient extends ZomeClient {
  protected readonly zomeName = 'lucid';

  constructor(client: AppClient, config: ThoughtsClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Thoughts
  // ============================================================================

  async createThought(input: CreateThoughtInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('create_thought', input);
  }

  async getThought(thoughtId: string): Promise<HolochainRecord | null> {
    return this.callZomeOrNull<HolochainRecord>('get_thought', thoughtId);
  }

  async getThoughtByHash(actionHash: ActionHash): Promise<HolochainRecord | null> {
    return this.callZomeOrNull<HolochainRecord>('get_thought_by_hash', actionHash);
  }

  async updateThought(input: UpdateThoughtInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('update_thought', input);
  }

  async deleteThought(thoughtId: string): Promise<Uint8Array> {
    return this.callZomeOnce<Uint8Array>('delete_thought', thoughtId);
  }

  async getMyThoughts(): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_my_thoughts', null);
  }

  async getThoughtsByTag(tag: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_thoughts_by_tag', tag);
  }

  async getThoughtsByDomain(domain: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_thoughts_by_domain', domain);
  }

  async getThoughtsByType(thoughtType: ThoughtType): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_thoughts_by_type', thoughtType);
  }

  async getChildThoughts(parentThoughtId: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_child_thoughts', parentThoughtId);
  }

  async searchThoughts(input: SearchThoughtsInput): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('search_thoughts', input);
  }

  async getThoughtsByCoherence(input: CoherenceRangeInput): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_thoughts_by_coherence', input);
  }

  // ============================================================================
  // Tags
  // ============================================================================

  async createTag(input: CreateTagInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('create_tag', input);
  }

  async getMyTags(): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_my_tags', null);
  }

  // ============================================================================
  // Domains
  // ============================================================================

  async createDomain(input: CreateDomainInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('create_domain', input);
  }

  async getMyDomains(): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_my_domains', null);
  }

  // ============================================================================
  // Relationships
  // ============================================================================

  async createRelationship(input: CreateRelationshipInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('create_relationship', input);
  }

  async getThoughtRelationships(thoughtId: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_thought_relationships', thoughtId);
  }

  async getRelatedThoughts(thoughtId: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_related_thoughts', thoughtId);
  }

  // ============================================================================
  // Stats & Analytics
  // ============================================================================

  async getStats(): Promise<KnowledgeGraphStats> {
    return this.callZome<KnowledgeGraphStats>('get_stats', null);
  }

  // ============================================================================
  // Embeddings & Coherence (Symthaea bridge)
  // ============================================================================

  async updateThoughtEmbedding(input: UpdateEmbeddingInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('update_thought_embedding', input);
  }

  async updateThoughtCoherence(input: UpdateCoherenceInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('update_thought_coherence', input);
  }

  async semanticSearch(input: SemanticSearchInput): Promise<SemanticSearchResult[]> {
    return this.callZome<SemanticSearchResult[]>('semantic_search', input);
  }

  async getThoughtsNeedingEmbeddings(): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_thoughts_needing_embeddings', null);
  }

  async getLowCoherenceThoughts(threshold: number): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_low_coherence_thoughts', threshold);
  }

  // ============================================================================
  // Epistemic Garden
  // ============================================================================

  async exploreGarden(input: ExploreGardenInput): Promise<ThoughtCluster[]> {
    return this.callZome<ThoughtCluster[]>('explore_garden', input);
  }

  async suggestConnections(input: SuggestConnectionsInput): Promise<ConnectionSuggestion[]> {
    return this.callZome<ConnectionSuggestion[]>('suggest_connections', input);
  }

  async findKnowledgeGaps(): Promise<KnowledgeGap[]> {
    return this.callZome<KnowledgeGap[]>('find_knowledge_gaps', null);
  }

  async discoverPatterns(): Promise<DiscoveredPattern[]> {
    return this.callZome<DiscoveredPattern[]>('discover_patterns', null);
  }
}
