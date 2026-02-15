/**
 * Mycelix Media Module
 *
 * TypeScript client for the mycelix-media hApp.
 * Provides content publication, attribution, fact-checking, and curation.
 *
 * @module @mycelix/sdk/media
 */

export interface HolochainRecord<T = unknown> {
  signed_action: { hashed: { hash: string; content: unknown }; signature: string };
  entry: { Present: T };
}

export interface ZomeCallable {
  callZome(args: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }): Promise<unknown>;
}

// ============================================================================
// Media Types
// ============================================================================

/** Matches Rust ContentType in media-publication integrity */
export type ContentType = 'Article' | 'Opinion' | 'Investigation' | 'Review' | 'Analysis' | 'Interview' | 'Report' | 'Editorial' | 'Other';
/** Matches Rust LicenseType in media-publication integrity */
export type LicenseType =
  | 'CC0'
  | 'CCBY'
  | 'CCBYSA'
  | 'CCBYNC'
  | 'CCBYNCSA'
  | 'AllRightsReserved'
  | 'Custom';
export type VerificationStatus = 'Unverified' | 'Pending' | 'Verified' | 'Disputed' | 'Debunked';

export interface Content {
  id: string;
  author: string;
  type_: ContentType;
  title: string;
  description?: string;
  content_hash: string;
  storage_uri: string;
  license: LicenseType;
  custom_license?: string;
  tags: string[];
  verification_status: VerificationStatus;
  views: number;
  endorsements: number;
  published_at: number;
  updated_at: number;
}

export interface PublishContentInput {
  type_: ContentType;
  title: string;
  description?: string;
  content_hash: string;
  storage_uri: string;
  license: LicenseType;
  custom_license?: string;
  tags?: string[];
}

export interface Attribution {
  id: string;
  content_id: string;
  contributor: string;
  role: 'Author' | 'Editor' | 'Contributor' | 'Source' | 'Translator';
  description?: string;
  royalty_percentage?: number;
  created_at: number;
}

export interface AddAttributionInput {
  content_id: string;
  contributor: string;
  role: Attribution['role'];
  description?: string;
  royalty_percentage?: number;
}

export interface FactCheck {
  id: string;
  content_id: string;
  checker: string;
  verdict: VerificationStatus;
  reasoning: string;
  sources: string[];
  epistemic_score: { empirical: number; normative: number; metaphorical: number };
  created_at: number;
}

export interface SubmitFactCheckInput {
  content_id: string;
  verdict: VerificationStatus;
  reasoning: string;
  sources: string[];
  epistemic_score: { empirical: number; normative: number; metaphorical: number };
}

export interface CurationList {
  id: string;
  curator: string;
  name: string;
  description: string;
  content_ids: string[];
  public_: boolean;
  created_at: number;
  updated_at: number;
}

export interface CreateCurationListInput {
  name: string;
  description: string;
  content_ids?: string[];
  public_: boolean;
}

export interface Endorsement {
  id: string;
  content_id: string;
  endorser: string;
  weight: number;
  comment?: string;
  created_at: number;
}

// ============================================================================
// Clients
// ============================================================================

const MEDIA_ROLE = 'civic';

export class PublicationClient {
  constructor(private readonly client: ZomeCallable) {}

  async publish(input: PublishContentInput): Promise<HolochainRecord<Content>> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: 'media_publication',
      fn_name: 'publish',
      payload: input,
    }) as Promise<HolochainRecord<Content>>;
  }

  async getContent(contentId: string): Promise<HolochainRecord<Content> | null> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: 'media_publication',
      fn_name: 'get_content',
      payload: contentId,
    }) as Promise<HolochainRecord<Content> | null>;
  }

  async getContentByAuthor(authorDid: string): Promise<HolochainRecord<Content>[]> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: 'media_publication',
      fn_name: 'get_by_author',
      payload: authorDid,
    }) as Promise<HolochainRecord<Content>[]>;
  }

  async getContentByTag(tag: string): Promise<HolochainRecord<Content>[]> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: 'media_publication',
      fn_name: 'get_by_tag',
      payload: tag,
    }) as Promise<HolochainRecord<Content>[]>;
  }

  async getContentByType(type_: ContentType): Promise<HolochainRecord<Content>[]> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: 'media_publication',
      fn_name: 'get_by_type',
      payload: type_,
    }) as Promise<HolochainRecord<Content>[]>;
  }

  async updateContent(
    contentId: string,
    updates: Partial<PublishContentInput>
  ): Promise<HolochainRecord<Content>> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: 'media_publication',
      fn_name: 'update_content',
      payload: { content_id: contentId, ...updates },
    }) as Promise<HolochainRecord<Content>>;
  }

  async recordView(contentId: string): Promise<void> {
    await this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: 'media_publication',
      fn_name: 'record_view',
      payload: contentId,
    });
  }
}

export class AttributionClient {
  constructor(private readonly client: ZomeCallable) {}

  async addAttribution(input: AddAttributionInput): Promise<HolochainRecord<Attribution>> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: 'media_attribution',
      fn_name: 'add_attribution',
      payload: input,
    }) as Promise<HolochainRecord<Attribution>>;
  }

  async getAttributions(contentId: string): Promise<HolochainRecord<Attribution>[]> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: 'media_attribution',
      fn_name: 'get_attributions',
      payload: contentId,
    }) as Promise<HolochainRecord<Attribution>[]>;
  }

  async getContributorWorks(contributorDid: string): Promise<HolochainRecord<Attribution>[]> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: 'media_attribution',
      fn_name: 'get_contributor_works',
      payload: contributorDid,
    }) as Promise<HolochainRecord<Attribution>[]>;
  }
}

export class FactCheckClient {
  constructor(private readonly client: ZomeCallable) {}

  async submitFactCheck(input: SubmitFactCheckInput): Promise<HolochainRecord<FactCheck>> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: 'media_factcheck',
      fn_name: 'submit_fact_check',
      payload: input,
    }) as Promise<HolochainRecord<FactCheck>>;
  }

  async getFactChecks(contentId: string): Promise<HolochainRecord<FactCheck>[]> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: 'media_factcheck',
      fn_name: 'get_fact_checks',
      payload: contentId,
    }) as Promise<HolochainRecord<FactCheck>[]>;
  }

  async getFactChecksByChecker(checkerDid: string): Promise<HolochainRecord<FactCheck>[]> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: 'media_factcheck',
      fn_name: 'get_by_checker',
      payload: checkerDid,
    }) as Promise<HolochainRecord<FactCheck>[]>;
  }

  async getContentVerificationStatus(contentId: string): Promise<VerificationStatus> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: 'media_factcheck',
      fn_name: 'get_verification_status',
      payload: contentId,
    }) as Promise<VerificationStatus>;
  }
}

export class CurationClient {
  constructor(private readonly client: ZomeCallable) {}

  async createList(input: CreateCurationListInput): Promise<HolochainRecord<CurationList>> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: 'media_curation',
      fn_name: 'create_list',
      payload: input,
    }) as Promise<HolochainRecord<CurationList>>;
  }

  async getList(listId: string): Promise<HolochainRecord<CurationList> | null> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: 'media_curation',
      fn_name: 'get_list',
      payload: listId,
    }) as Promise<HolochainRecord<CurationList> | null>;
  }

  async getListsByCurator(curatorDid: string): Promise<HolochainRecord<CurationList>[]> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: 'media_curation',
      fn_name: 'get_by_curator',
      payload: curatorDid,
    }) as Promise<HolochainRecord<CurationList>[]>;
  }

  async addToList(listId: string, contentId: string): Promise<HolochainRecord<CurationList>> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: 'media_curation',
      fn_name: 'add_to_list',
      payload: { list_id: listId, content_id: contentId },
    }) as Promise<HolochainRecord<CurationList>>;
  }

  async removeFromList(listId: string, contentId: string): Promise<HolochainRecord<CurationList>> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: 'media_curation',
      fn_name: 'remove_from_list',
      payload: { list_id: listId, content_id: contentId },
    }) as Promise<HolochainRecord<CurationList>>;
  }

  async endorse(
    contentId: string,
    weight: number,
    comment?: string
  ): Promise<HolochainRecord<Endorsement>> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: 'media_curation',
      fn_name: 'endorse',
      payload: { content_id: contentId, weight, comment },
    }) as Promise<HolochainRecord<Endorsement>>;
  }

  async getEndorsements(contentId: string): Promise<HolochainRecord<Endorsement>[]> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: 'media_curation',
      fn_name: 'get_endorsements',
      payload: contentId,
    }) as Promise<HolochainRecord<Endorsement>[]>;
  }
}

export function createMediaClients(client: ZomeCallable) {
  return {
    publication: new PublicationClient(client),
    attribution: new AttributionClient(client),
    factcheck: new FactCheckClient(client),
    curation: new CurationClient(client),
  };
}

// Unified client
export { MycelixMediaClient, MediaSdkError } from './client';
export type { MediaClientConfig, MediaConnectionOptions } from './client';

export default {
  PublicationClient,
  AttributionClient,
  FactCheckClient,
  CurationClient,
  createMediaClients,
};
