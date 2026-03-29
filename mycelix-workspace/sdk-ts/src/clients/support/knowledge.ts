// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Knowledge Zome Client
 *
 * Handles knowledge articles, resolutions, reputation, and community curation.
 *
 * @module @mycelix/sdk/clients/support/knowledge
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  KnowledgeArticle,
  KnowledgeArticleInput,
  UpdateArticleInput,
  DeprecateInput,
  SupportCategory,
  ResolutionInput,
  ArticleFlagInput,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface KnowledgeClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: KnowledgeClientConfig = {
  roleName: 'commons',
};

/**
 * Client for Support Knowledge operations
 */
export class KnowledgeClient extends ZomeClient {
  protected readonly zomeName = 'support_knowledge';

  constructor(client: AppClient, config: KnowledgeClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Articles
  // ============================================================================

  async createArticle(input: KnowledgeArticleInput): Promise<KnowledgeArticle> {
    const record = await this.callZomeOnce<HolochainRecord>('create_article', {
      title: input.title,
      content: input.content,
      category: input.category,
      tags: input.tags,
      author: input.author,
      source: input.source,
      difficulty_level: input.difficultyLevel,
      upvotes: input.upvotes,
      verified: input.verified,
      deprecated: input.deprecated,
      deprecation_reason: input.deprecationReason,
      version: input.version,
    });
    return this.mapArticle(record);
  }

  async updateArticle(input: UpdateArticleInput): Promise<KnowledgeArticle> {
    const record = await this.callZomeOnce<HolochainRecord>('update_article', {
      original_hash: input.originalHash,
      updated: {
        title: input.updated.title,
        content: input.updated.content,
        category: input.updated.category,
        tags: input.updated.tags,
        author: input.updated.author,
        source: input.updated.source,
        difficulty_level: input.updated.difficultyLevel,
        upvotes: input.updated.upvotes,
        verified: input.updated.verified,
        deprecated: input.updated.deprecated,
        deprecation_reason: input.updated.deprecationReason,
        version: input.updated.version,
      },
    });
    return this.mapArticle(record);
  }

  async deprecateArticle(input: DeprecateInput): Promise<KnowledgeArticle> {
    const record = await this.callZomeOnce<HolochainRecord>('deprecate_article', {
      article_hash: input.articleHash,
      reason: input.reason,
    });
    return this.mapArticle(record);
  }

  async getArticle(articleHash: ActionHash): Promise<KnowledgeArticle | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_article', articleHash);
    if (!record) return null;
    return this.mapArticle(record);
  }

  async searchByCategory(category: SupportCategory): Promise<KnowledgeArticle[]> {
    const records = await this.callZome<HolochainRecord[]>('search_by_category', category);
    return records.map(r => this.mapArticle(r));
  }

  async searchByTag(tag: string): Promise<KnowledgeArticle[]> {
    const records = await this.callZome<HolochainRecord[]>('search_by_tag', tag);
    return records.map(r => this.mapArticle(r));
  }

  async listRecentArticles(): Promise<KnowledgeArticle[]> {
    const records = await this.callZome<HolochainRecord[]>('list_recent_articles', null);
    return records.map(r => this.mapArticle(r));
  }

  // ============================================================================
  // Curation
  // ============================================================================

  async upvoteArticle(articleHash: ActionHash): Promise<void> {
    await this.callZomeOnce<void>('upvote_article', articleHash);
  }

  async verifyArticle(articleHash: ActionHash): Promise<void> {
    await this.callZomeOnce<void>('verify_article', articleHash);
  }

  // ============================================================================
  // Resolutions
  // ============================================================================

  async createResolution(input: ResolutionInput): Promise<Uint8Array> {
    const record = await this.callZomeOnce<HolochainRecord>('create_resolution', {
      ticket_hash: input.ticketHash,
      steps: input.steps,
      root_cause: input.rootCause,
      time_to_resolve_mins: input.timeToResolveMins,
      effectiveness_rating: input.effectivenessRating,
      helper: input.helper,
      requester: input.requester,
      anonymized: input.anonymized,
      helper_signature: input.helperSignature,
      requester_signature: input.requesterSignature,
    });
    return this.getActionHash(record);
  }

  // ============================================================================
  // Flags & Reputation
  // ============================================================================

  async flagArticle(input: ArticleFlagInput): Promise<Uint8Array> {
    const record = await this.callZomeOnce<HolochainRecord>('flag_article', {
      article_hash: input.articleHash,
      flagger: input.flagger,
      reason: input.reason,
      description: input.description,
      created_at: input.createdAt,
    });
    return this.getActionHash(record);
  }

  async getFlags(articleHash: ActionHash): Promise<any[]> {
    return this.callZome<any[]>('get_flags', articleHash);
  }

  async getAgentReputation(agentPubKey: Uint8Array): Promise<any> {
    return this.callZome<any>('get_agent_reputation', agentPubKey);
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private mapArticle(record: HolochainRecord): KnowledgeArticle {
    const entry = this.extractEntry<any>(record);
    return {
      actionHash: record.signed_action.hashed.hash,
      title: entry.title,
      content: entry.content,
      category: entry.category,
      tags: entry.tags,
      author: entry.author,
      source: entry.source,
      difficultyLevel: entry.difficulty_level,
      upvotes: entry.upvotes,
      verified: entry.verified,
      deprecated: entry.deprecated,
      deprecationReason: entry.deprecation_reason,
      version: entry.version,
    };
  }
}
