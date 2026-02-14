/**
 * LUCID Zome Client
 *
 * Core operations for thoughts, tags, domains, and relationships.
 */

import type { AppClient, Record as HolochainRecord, ActionHash } from '@holochain/client';
import type {
  Thought,
  CreateThoughtInput,
  UpdateThoughtInput,
  SearchThoughtsInput,
  Tag,
  CreateTagInput,
  Domain,
  CreateDomainInput,
  Relationship,
  CreateRelationshipInput,
  KnowledgeGraphStats,
  ThoughtType,
} from '../types';
import { decodeRecord, decodeRecords } from '../utils';

export class LucidZomeClient {
  constructor(
    private client: AppClient,
    private roleName: string = 'lucid',
    private zomeName: string = 'lucid'
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
  // THOUGHT OPERATIONS
  // ============================================================================

  /** Create a new thought */
  async createThought(input: CreateThoughtInput): Promise<Thought> {
    const record = await this.callZome<HolochainRecord>('create_thought', input);
    return decodeRecord<Thought>(record);
  }

  /** Get a thought by ID */
  async getThought(thoughtId: string): Promise<Thought | null> {
    const record = await this.callZome<HolochainRecord | null>('get_thought', thoughtId);
    return record ? decodeRecord<Thought>(record) : null;
  }

  /** Get a thought by action hash */
  async getThoughtByHash(actionHash: ActionHash): Promise<Thought | null> {
    const record = await this.callZome<HolochainRecord | null>('get_thought_by_hash', actionHash);
    return record ? decodeRecord<Thought>(record) : null;
  }

  /** Update an existing thought */
  async updateThought(input: UpdateThoughtInput): Promise<Thought> {
    const record = await this.callZome<HolochainRecord>('update_thought', input);
    return decodeRecord<Thought>(record);
  }

  /** Delete a thought */
  async deleteThought(thoughtId: string): Promise<ActionHash> {
    return this.callZome<ActionHash>('delete_thought', thoughtId);
  }

  /** Get all thoughts for the current user */
  async getMyThoughts(): Promise<Thought[]> {
    const records = await this.callZome<HolochainRecord[]>('get_my_thoughts', null);
    return decodeRecords<Thought>(records);
  }

  /** Get thoughts by tag */
  async getThoughtsByTag(tag: string): Promise<Thought[]> {
    const records = await this.callZome<HolochainRecord[]>('get_thoughts_by_tag', tag);
    return decodeRecords<Thought>(records);
  }

  /** Get thoughts by domain */
  async getThoughtsByDomain(domain: string): Promise<Thought[]> {
    const records = await this.callZome<HolochainRecord[]>('get_thoughts_by_domain', domain);
    return decodeRecords<Thought>(records);
  }

  /** Get thoughts by type */
  async getThoughtsByType(thoughtType: ThoughtType): Promise<Thought[]> {
    const records = await this.callZome<HolochainRecord[]>('get_thoughts_by_type', thoughtType);
    return decodeRecords<Thought>(records);
  }

  /** Get child thoughts of a parent */
  async getChildThoughts(parentThoughtId: string): Promise<Thought[]> {
    const records = await this.callZome<HolochainRecord[]>('get_child_thoughts', parentThoughtId);
    return decodeRecords<Thought>(records);
  }

  /** Search thoughts with filters */
  async searchThoughts(input: SearchThoughtsInput): Promise<Thought[]> {
    const records = await this.callZome<HolochainRecord[]>('search_thoughts', input);
    return decodeRecords<Thought>(records);
  }

  // ============================================================================
  // TAG OPERATIONS
  // ============================================================================

  /** Create a new tag */
  async createTag(input: CreateTagInput): Promise<Tag> {
    const record = await this.callZome<HolochainRecord>('create_tag', input);
    return decodeRecord<Tag>(record);
  }

  /** Get all tags for the current user */
  async getMyTags(): Promise<Tag[]> {
    const records = await this.callZome<HolochainRecord[]>('get_my_tags', null);
    return decodeRecords<Tag>(records);
  }

  // ============================================================================
  // DOMAIN OPERATIONS
  // ============================================================================

  /** Create a new domain */
  async createDomain(input: CreateDomainInput): Promise<Domain> {
    const record = await this.callZome<HolochainRecord>('create_domain', input);
    return decodeRecord<Domain>(record);
  }

  /** Get all domains for the current user */
  async getMyDomains(): Promise<Domain[]> {
    const records = await this.callZome<HolochainRecord[]>('get_my_domains', null);
    return decodeRecords<Domain>(records);
  }

  // ============================================================================
  // RELATIONSHIP OPERATIONS
  // ============================================================================

  /** Create a relationship between thoughts */
  async createRelationship(input: CreateRelationshipInput): Promise<Relationship> {
    const record = await this.callZome<HolochainRecord>('create_relationship', input);
    return decodeRecord<Relationship>(record);
  }

  /** Get relationships from a thought */
  async getThoughtRelationships(thoughtId: string): Promise<Relationship[]> {
    const records = await this.callZome<HolochainRecord[]>('get_thought_relationships', thoughtId);
    return decodeRecords<Relationship>(records);
  }

  /** Get thoughts related to a given thought */
  async getRelatedThoughts(thoughtId: string): Promise<Thought[]> {
    const records = await this.callZome<HolochainRecord[]>('get_related_thoughts', thoughtId);
    return decodeRecords<Thought>(records);
  }

  // ============================================================================
  // STATISTICS
  // ============================================================================

  /** Get statistics about the knowledge graph */
  async getStats(): Promise<KnowledgeGraphStats> {
    return this.callZome<KnowledgeGraphStats>('get_stats', null);
  }
}
