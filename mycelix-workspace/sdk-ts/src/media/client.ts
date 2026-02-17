/**
 * Mycelix Media Client
 *
 * Unified client for the Media hApp SDK, providing access to
 * content publication, attribution, fact-checking, and curation.
 *
 * @module @mycelix/sdk/media
 */

import {
  type AppClient,
  AppWebsocket,
} from '@holochain/client';

import { RetryPolicy, RetryPolicies, type RetryOptions } from '../common/retry';

import {
  PublicationClient,
  AttributionClient,
  FactCheckClient,
  CurationClient,
  type Content,
  type Attribution,
  type FactCheck,
  type Endorsement,
  type ContentType,
  type VerificationStatus,
} from './index';

/**
 * Media SDK Error
 */
export class MediaSdkError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly cause?: unknown
  ) {
    super(message);
    this.name = 'MediaSdkError';
  }
}

/**
 * Configuration for the Media client
 */
export interface MediaClientConfig {
  /** Role ID for the media DNA */
  roleId: string;
  /** Retry configuration for zome calls */
  retry?: RetryOptions | RetryPolicy;
}

const DEFAULT_CONFIG: MediaClientConfig = {
  roleId: 'media',
};

/**
 * Connection options for creating a new client
 */
export interface MediaConnectionOptions {
  /** WebSocket URL to connect to */
  url: string;
  /** Optional timeout in milliseconds */
  timeout?: number;
  /** Retry configuration for connection and zome calls */
  retry?: RetryOptions | RetryPolicy;
}

/**
 * Unified Mycelix Media Client
 *
 * Provides access to all media functionality through a single interface.
 *
 * @example
 * ```typescript
 * import { MycelixMediaClient } from '@mycelix/sdk/media';
 *
 * // Connect to Holochain
 * const media = await MycelixMediaClient.connect({
 *   url: 'ws://localhost:8888',
 * });
 *
 * // Publish content
 * const content = await media.publication.publish({
 *   type_: 'Article',
 *   title: 'My First Article',
 *   content_hash: 'abc123...',
 *   storage_uri: 'ipfs://...',
 *   license: 'CCBY',
 *   tags: ['tech', 'tutorial'],
 * });
 *
 * // Add attribution
 * await media.attribution.addAttribution({
 *   content_id: content.entry.Present.id,
 *   contributor: 'did:mycelix:editor',
 *   role: 'Editor',
 *   royalty_percentage: 10,
 * });
 *
 * // Submit fact-check
 * await media.factCheck.submitFactCheck({
 *   content_id: content.entry.Present.id,
 *   verdict: 'Verified',
 *   reasoning: 'Sources verified against primary documents',
 *   sources: ['https://source1.com', 'https://source2.com'],
 *   epistemic_score: { empirical: 0.9, normative: 0.1, metaphorical: 0 },
 * });
 * ```
 */
export class MycelixMediaClient {
  /** Content publication operations */
  public readonly publication: PublicationClient;

  /** Attribution management */
  public readonly attribution: AttributionClient;

  /** Fact-checking operations */
  public readonly factCheck: FactCheckClient;

  /** Curation and endorsement operations */
  public readonly curation: CurationClient;

  private readonly config: MediaClientConfig;

  /** Retry policy for zome calls */
  private readonly retryPolicy: RetryPolicy;

  /**
   * Create a media client from an existing Holochain client
   *
   * @param client - Existing AppClient instance
   * @param config - Optional configuration overrides
   */
  constructor(
    private readonly client: AppClient,
    config: Partial<MediaClientConfig> = {}
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    // Set up retry policy
    if (config.retry instanceof RetryPolicy) {
      this.retryPolicy = config.retry;
    } else if (config.retry) {
      this.retryPolicy = new RetryPolicy(config.retry);
    } else {
      this.retryPolicy = RetryPolicies.standard;
    }

    // Initialize sub-clients
    this.publication = new PublicationClient(client);
    this.attribution = new AttributionClient(client);
    this.factCheck = new FactCheckClient(client);
    this.curation = new CurationClient(client);
  }

  /**
   * Connect to Holochain and create a media client
   *
   * @param options - Connection options
   * @returns Connected media client
   */
  static async connect(
    options: MediaConnectionOptions
  ): Promise<MycelixMediaClient> {
    // Set up retry for connection
    const retryPolicy = options.retry instanceof RetryPolicy
      ? options.retry
      : options.retry
        ? new RetryPolicy(options.retry)
        : RetryPolicies.network;

    const connectFn = async () => {
      try {
        const client = await AppWebsocket.connect({
          url: new URL(options.url),
          wsClientOptions: { origin: 'mycelix-media-sdk' },
        });

        return new MycelixMediaClient(client, { retry: retryPolicy });
      } catch (error) {
        throw new MediaSdkError(
          'CONNECTION_ERROR',
          `Failed to connect to Holochain: ${error instanceof Error ? error.message : String(error)}`,
          error
        );
      }
    };

    return retryPolicy.execute(connectFn);
  }

  /**
   * Create a media client from an existing AppClient
   *
   * @param client - Existing AppClient instance
   * @param config - Optional configuration overrides
   * @returns Media client
   */
  static fromClient(
    client: AppClient,
    config: Partial<MediaClientConfig> = {}
  ): MycelixMediaClient {
    return new MycelixMediaClient(client, config);
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Publish content with automatic author attribution
   *
   * @param authorDid - Author's DID
   * @param input - Content publication input
   * @returns Published content with attribution
   */
  async publishWithAttribution(
    authorDid: string,
    input: Parameters<PublicationClient['publish']>[0]
  ): Promise<{ content: Content; attribution: Attribution }> {
    const record = await this.publication.publish(input);
    const content = record.entry.Present;

    const attrRecord = await this.attribution.addAttribution({
      content_id: content.id,
      contributor: authorDid,
      role: 'Author',
      royalty_percentage: 100,
    });
    const attribution = attrRecord.entry.Present;

    return { content, attribution };
  }

  /**
   * Get content with all related data
   *
   * @param contentId - Content identifier
   * @returns Content with attributions, fact-checks, and endorsements
   */
  async getContentComplete(contentId: string): Promise<{
    content: Content | null;
    attributions: Attribution[];
    factChecks: FactCheck[];
    endorsements: Endorsement[];
    verificationStatus: VerificationStatus;
  } | null> {
    const record = await this.publication.getContent(contentId);
    if (!record) {
      return null;
    }

    const content = record.entry.Present;

    const [attrRecords, fcRecords, endRecords, status] = await Promise.all([
      this.attribution.getAttributions(contentId),
      this.factCheck.getFactChecks(contentId),
      this.curation.getEndorsements(contentId),
      this.factCheck.getContentVerificationStatus(contentId),
    ]);

    return {
      content,
      attributions: attrRecords.map(r => r.entry.Present),
      factChecks: fcRecords.map(r => r.entry.Present),
      endorsements: endRecords.map(r => r.entry.Present),
      verificationStatus: status,
    };
  }

  /**
   * Search content with filters
   *
   * @param query - Search options
   * @returns Matching content
   */
  async searchContent(query: {
    tag?: string;
    type?: ContentType;
    authorDid?: string;
    verified?: boolean;
    limit?: number;
  }): Promise<Content[]> {
    let results: Content[] = [];

    if (query.tag) {
      const records = await this.publication.getContentByTag(query.tag);
      results = records.map(r => r.entry.Present);
    } else if (query.type) {
      const records = await this.publication.getContentByType(query.type);
      results = records.map(r => r.entry.Present);
    } else if (query.authorDid) {
      const records = await this.publication.getContentByAuthor(query.authorDid);
      results = records.map(r => r.entry.Present);
    }

    // Filter by verification status
    if (query.verified !== undefined) {
      results = results.filter(c =>
        query.verified
          ? c.verification_status === 'Verified'
          : c.verification_status !== 'Verified'
      );
    }

    // Apply limit
    if (query.limit) {
      results = results.slice(0, query.limit);
    }

    return results;
  }

  /**
   * Get creator statistics
   *
   * @param creatorDid - Creator's DID
   * @returns Creator statistics
   */
  async getCreatorStats(creatorDid: string): Promise<{
    contentCount: number;
    totalViews: number;
    totalEndorsements: number;
    verifiedCount: number;
    contentByType: Record<ContentType, number>;
  }> {
    const records = await this.publication.getContentByAuthor(creatorDid);
    const contents = records.map(r => r.entry.Present);

    const contentByType: Record<string, number> = {};
    let totalViews = 0;
    let totalEndorsements = 0;
    let verifiedCount = 0;

    for (const content of contents) {
      totalViews += content.views;
      totalEndorsements += content.endorsements;
      if (content.verification_status === 'Verified') {
        verifiedCount++;
      }
      contentByType[content.type_] = (contentByType[content.type_] || 0) + 1;
    }

    return {
      contentCount: contents.length,
      totalViews,
      totalEndorsements,
      verifiedCount,
      contentByType: contentByType as Record<ContentType, number>,
    };
  }

  /**
   * Get the underlying Holochain client
   */
  getClient(): AppClient {
    return this.client;
  }

  /**
   * Get the current retry policy
   */
  getRetryPolicy(): RetryPolicy {
    return this.retryPolicy;
  }

  /**
   * Create a new client with a different retry policy
   *
   * @param retry - New retry configuration
   * @returns New client instance with updated retry policy
   */
  withRetry(retry: RetryOptions | RetryPolicy): MycelixMediaClient {
    return new MycelixMediaClient(this.client, { ...this.config, retry });
  }
}

export default MycelixMediaClient;
