// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * LUCID Sources Zome Client
 *
 * Source management and citation tracking for epistemic provenance.
 *
 * @module @mycelix/sdk/clients/lucid/sources
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  CreateSourceInput,
  CreateCitationInput,
  SourceType,
} from './types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface SourcesClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: SourcesClientConfig = {
  roleName: 'lucid',
};

/**
 * Client for the LUCID Sources zome
 *
 * Manages information sources and citations, linking thoughts
 * to their epistemic provenance.
 */
export class SourcesClient extends ZomeClient {
  protected readonly zomeName = 'lucid_sources';

  constructor(client: AppClient, config: SourcesClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Sources
  // ============================================================================

  /** Create a new information source */
  async createSource(input: CreateSourceInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('create_source', input);
  }

  /** Get a source by its ID */
  async getSource(sourceId: string): Promise<HolochainRecord | null> {
    return this.callZomeOrNull<HolochainRecord>('get_source', sourceId);
  }

  /** Get a source by URL */
  async getSourceByUrl(url: string): Promise<HolochainRecord | null> {
    return this.callZomeOrNull<HolochainRecord>('get_source_by_url', url);
  }

  /** Get all sources created by the current agent */
  async getMySources(): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_my_sources', null);
  }

  /** Get sources by type (Article, Book, Paper, etc.) */
  async getSourcesByType(sourceType: SourceType): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_sources_by_type', sourceType);
  }

  // ============================================================================
  // Citations
  // ============================================================================

  /** Create a citation linking a thought to a source */
  async createCitation(input: CreateCitationInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('create_citation', input);
  }

  /** Get all citations for a thought */
  async getThoughtCitations(thoughtId: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_thought_citations', thoughtId);
  }

  /** Get all citations of a source */
  async getSourceCitations(sourceId: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_source_citations', sourceId);
  }

  /** Get all sources for a thought (via citations) */
  async getThoughtSources(thoughtId: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_thought_sources', thoughtId);
  }
}
