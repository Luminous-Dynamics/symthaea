/**
 * Sources Zome Client
 *
 * Source tracking and citation operations.
 */

import type { AppClient, Record as HolochainRecord } from '@holochain/client';
import type {
  Source,
  CreateSourceInput,
  Citation,
  CreateCitationInput,
  SourceType,
} from '../types';
import { decodeRecord, decodeRecords } from '../utils';

export class SourcesZomeClient {
  constructor(
    private client: AppClient,
    private roleName: string = 'lucid',
    private zomeName: string = 'sources'
  ) {}

  private async callZome<T>(fnName: string, payload?: unknown): Promise<T> {
    const result = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: fnName,
      payload: payload ?? null,
    });
    return result as T;
  }

  // ============================================================================
  // SOURCE OPERATIONS
  // ============================================================================

  /** Create a new source */
  async createSource(input: CreateSourceInput): Promise<Source> {
    const record = await this.callZome<HolochainRecord>('create_source', input);
    return decodeRecord<Source>(record);
  }

  /** Get a source by ID */
  async getSource(sourceId: string): Promise<Source | null> {
    const record = await this.callZome<HolochainRecord | null>('get_source', sourceId);
    return record ? decodeRecord<Source>(record) : null;
  }

  /** Get a source by URL */
  async getSourceByUrl(url: string): Promise<Source | null> {
    const record = await this.callZome<HolochainRecord | null>('get_source_by_url', url);
    return record ? decodeRecord<Source>(record) : null;
  }

  /** Get all sources for the current user */
  async getMySources(): Promise<Source[]> {
    const records = await this.callZome<HolochainRecord[]>('get_my_sources', null);
    return decodeRecords<Source>(records);
  }

  /** Get sources by type */
  async getSourcesByType(sourceType: SourceType): Promise<Source[]> {
    const records = await this.callZome<HolochainRecord[]>('get_sources_by_type', sourceType);
    return decodeRecords<Source>(records);
  }

  // ============================================================================
  // CITATION OPERATIONS
  // ============================================================================

  /** Create a citation linking a thought to a source */
  async createCitation(input: CreateCitationInput): Promise<Citation> {
    const record = await this.callZome<HolochainRecord>('create_citation', input);
    return decodeRecord<Citation>(record);
  }

  /** Get citations for a thought */
  async getThoughtCitations(thoughtId: string): Promise<Citation[]> {
    const records = await this.callZome<HolochainRecord[]>('get_thought_citations', thoughtId);
    return decodeRecords<Citation>(records);
  }

  /** Get citations for a source */
  async getSourceCitations(sourceId: string): Promise<Citation[]> {
    const records = await this.callZome<HolochainRecord[]>('get_source_citations', sourceId);
    return decodeRecords<Citation>(records);
  }

  /** Get all sources for a thought (via citations) */
  async getThoughtSources(thoughtId: string): Promise<Source[]> {
    const records = await this.callZome<HolochainRecord[]>('get_thought_sources', thoughtId);
    return decodeRecords<Source>(records);
  }
}
