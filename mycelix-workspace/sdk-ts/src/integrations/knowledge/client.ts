/**
 * Mycelix Knowledge Client
 *
 * Unified client for the Knowledge hApp SDK, providing access to
 * epistemic claims, knowledge graph, fact-checking, and semantic search.
 *
 * @module @mycelix/sdk/integrations/knowledge
 */

import {
  type AppClient,
  AppWebsocket,
} from '@holochain/client';

import { KnowledgeSdkError } from './types';
import { ClaimsClient } from './zomes/claims';
import { FactCheckClient } from './zomes/factcheck';
import { GraphClient } from './zomes/graph';
import { RetryPolicy, RetryPolicies, type RetryOptions } from '../../common/retry';

import type {
  Claim,
  ClaimCluster,
  ConsistencyResult,
  SearchResult,
  SearchFilters,
  Citation,
} from './types';


/**
 * Configuration for the Knowledge client
 */
export interface KnowledgeClientConfig {
  /** Role ID for the knowledge DNA */
  roleId: string;
  /** Zome name */
  zomeName: string;
  /** Retry configuration for zome calls */
  retry?: RetryOptions | RetryPolicy;
}

const DEFAULT_CONFIG: KnowledgeClientConfig = {
  roleId: 'knowledge',
  zomeName: 'knowledge',
};

/**
 * Connection options for creating a new client
 */
export interface KnowledgeConnectionOptions {
  /** WebSocket URL to connect to */
  url: string;
  /** Optional timeout in milliseconds */
  timeout?: number;
  /** Optional app client constructor args */
  appClientArgs?: Record<string, unknown>;
  /** Retry configuration for connection and zome calls */
  retry?: RetryOptions | RetryPolicy;
}

/**
 * Unified Mycelix Knowledge Client
 *
 * Provides access to all knowledge graph functionality through a single interface.
 *
 * @example
 * ```typescript
 * import { MycelixKnowledgeClient } from '@mycelix/sdk/integrations/knowledge';
 *
 * // Connect to Holochain
 * const knowledge = await MycelixKnowledgeClient.connect({
 *   url: 'ws://localhost:8888',
 * });
 *
 * // Submit an epistemic claim
 * const claim = await knowledge.claims.submitClaim({
 *   id: 'claim-001',
 *   author_did: 'did:mycelix:alice',
 *   title: 'Climate Change Evidence',
 *   content: 'Global temperatures have risen 1.1°C since pre-industrial times...',
 *   empirical: 0.95,
 *   normative: 0.05,
 *   metaphysical: 0.0,
 *   tags: ['climate', 'science', 'temperature'],
 * });
 *
 * // Create a supporting relation
 * await knowledge.graph.createSupportRelation(
 *   'claim-002', // source (provides support)
 *   'claim-001', // target (is supported)
 *   'did:mycelix:bob'
 * );
 *
 * // Semantic search
 * const results = await knowledge.graph.semanticSearch({
 *   query: 'climate temperature',
 *   limit: 10,
 *   offset: 0,
 * });
 *
 * // Fact-check a claim from Media hApp
 * await knowledge.factCheck.submitFactCheckRequest({
 *   id: 'fc-001',
 *   media_content_id: 'media-xyz',
 *   claim_text: 'Vaccines cause autism',
 *   requested_by: 'did:mycelix:factchecker',
 * });
 * ```
 */
export class MycelixKnowledgeClient {
  /** Claim operations (submit, update, endorse) */
  public readonly claims: ClaimsClient;

  /** Graph operations (relations, traversal, search) */
  public readonly graph: GraphClient;

  /** Fact-checking and cross-hApp integration */
  public readonly factCheck: FactCheckClient;

  private readonly config: KnowledgeClientConfig;

  /** Retry policy for zome calls */
  private readonly retryPolicy: RetryPolicy;

  /**
   * Create a knowledge client from an existing Holochain client
   *
   * @param client - Existing AppClient instance
   * @param config - Optional configuration overrides
   */
  constructor(
    private readonly client: AppClient,
    config: Partial<KnowledgeClientConfig> = {}
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
    this.claims = new ClaimsClient(client, this.config);
    this.graph = new GraphClient(client, this.config);
    this.factCheck = new FactCheckClient(client, this.config);
  }

  /**
   * Connect to Holochain and create a knowledge client
   *
   * @param options - Connection options
   * @returns Connected knowledge client
   *
   * @example
   * ```typescript
   * const knowledge = await MycelixKnowledgeClient.connect({
   *   url: 'ws://localhost:8888',
   *   timeout: 30000,
   * });
   * ```
   */
  static async connect(
    options: KnowledgeConnectionOptions
  ): Promise<MycelixKnowledgeClient> {
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
          ...options.appClientArgs,
        });

        return new MycelixKnowledgeClient(client, { retry: retryPolicy });
      } catch (error) {
        throw new KnowledgeSdkError(
          'CONNECTION_ERROR',
          `Failed to connect to Holochain: ${error instanceof Error ? error.message : String(error)}`,
          error
        );
      }
    };

    return retryPolicy.execute(connectFn);
  }

  /**
   * Create a knowledge client from an existing AppClient
   *
   * Use this when you already have a Holochain connection.
   *
   * @param client - Existing AppClient instance
   * @param config - Optional configuration overrides
   * @returns Knowledge client
   */
  static fromClient(
    client: AppClient,
    config: Partial<KnowledgeClientConfig> = {}
  ): MycelixKnowledgeClient {
    return new MycelixKnowledgeClient(client, config);
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Get comprehensive claim status
   *
   * Returns claim details, relations, cluster info, and endorsements.
   *
   * @param claimId - Claim identifier
   * @returns Claim status summary
   */
  async getClaimStatus(claimId: string): Promise<{
    claim: Claim | null;
    supportingClaims: Claim[];
    contradictingClaims: Claim[];
    cluster: ClaimCluster;
  }> {
    const [claim, supportingClaims, contradictingClaims, cluster] = await Promise.all([
      this.claims.getClaim(claimId),
      this.graph.getSupportingClaims(claimId),
      this.graph.getContradictingClaims(claimId),
      this.graph.getClaimCluster(claimId, 50),
    ]);

    return {
      claim,
      supportingClaims,
      contradictingClaims,
      cluster,
    };
  }

  /**
   * Quick search with default parameters
   *
   * @param query - Search query
   * @param limit - Maximum results (default 20)
   * @returns Search results
   */
  async search(query: string, limit: number = 20): Promise<SearchResult[]> {
    return this.graph.semanticSearch({
      query,
      limit,
      offset: 0,
    });
  }

  /**
   * Verify consistency of multiple claims
   *
   * @param claimIds - Array of claim identifiers
   * @returns Consistency result
   */
  async checkConsistency(claimIds: string[]): Promise<ConsistencyResult> {
    return this.graph.verifyConsistency(claimIds);
  }

  /**
   * Get claims by author with full details
   *
   * @param authorDid - Author's DID
   * @returns Array of claims with endorsement counts
   */
  async getAuthorClaims(authorDid: string): Promise<Claim[]> {
    return this.claims.getClaimsByAuthor(authorDid);
  }

  /**
   * Explore related claims from a starting point
   *
   * @param claimId - Starting claim identifier
   * @param depth - How deep to explore
   * @returns Array of related claim IDs
   */
  async exploreRelated(claimId: string, depth: number = 2): Promise<string[]> {
    const result = await this.graph.traverseGraph({
      start_claim_id: claimId,
      direction: 'Both',
      max_depth: depth,
      limit: 100,
    });
    return result.visited_claims.filter(id => id !== claimId);
  }

  /**
   * Get a claim with all its supporting evidence
   *
   * Returns the claim along with citations, sources, and supporting claims
   * for comprehensive evidence review.
   *
   * @param claimId - Claim identifier
   * @returns Claim with all evidence or null if not found
   *
   * @example
   * ```typescript
   * const evidence = await knowledge.getClaimWithEvidence('claim-001');
   * if (evidence) {
   *   console.log(`Claim: ${evidence.claim.title}`);
   *   console.log(`Citations: ${evidence.citations.length}`);
   *   console.log(`Supporting claims: ${evidence.supportingClaims.length}`);
   * }
   * ```
   */
  async getClaimWithEvidence(claimId: string): Promise<{
    claim: Claim;
    citations: Citation[];
    supportingClaims: Claim[];
    contradictingClaims: Claim[];
    cluster: ClaimCluster;
  } | null> {
    const claim = await this.claims.getClaim(claimId);
    if (!claim) return null;

    const [citations, supportingClaims, contradictingClaims, cluster] = await Promise.all([
      this.claims.getClaimCitations(claimId),
      this.graph.getSupportingClaims(claimId),
      this.graph.getContradictingClaims(claimId),
      this.graph.getClaimCluster(claimId, 50),
    ]);

    return {
      claim,
      citations,
      supportingClaims,
      contradictingClaims,
      cluster,
    };
  }

  /**
   * Query the knowledge graph with filtering and search options
   *
   * Combines semantic search with graph traversal for comprehensive results.
   *
   * @param options - Query options
   * @returns Query results with claims and graph context
   *
   * @example
   * ```typescript
   * const results = await knowledge.queryKnowledgeGraph({
   *   query: 'climate change evidence',
   *   filters: { min_empirical: 0.7 },
   *   includeRelated: true,
   *   relatedDepth: 2,
   * });
   * ```
   */
  async queryKnowledgeGraph(options: {
    /** Search query string */
    query: string;
    /** Search filters */
    filters?: SearchFilters;
    /** Maximum results */
    limit?: number;
    /** Include related claims for each result */
    includeRelated?: boolean;
    /** Depth for related claim traversal */
    relatedDepth?: number;
  }): Promise<{
    results: SearchResult[];
    relatedClaims?: Map<string, string[]>;
  }> {
    const { query, filters, limit = 20, includeRelated = false, relatedDepth = 1 } = options;

    const results = await this.graph.semanticSearch({
      query,
      filters,
      limit,
      offset: 0,
    });

    if (!includeRelated) {
      return { results };
    }

    // Fetch related claims for each result
    const relatedClaims = new Map<string, string[]>();
    await Promise.all(
      results.map(async (result) => {
        const related = await this.graph.getReachableClaims(result.claim_id, relatedDepth);
        relatedClaims.set(result.claim_id, related.filter(id => id !== result.claim_id));
      })
    );

    return { results, relatedClaims };
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
  withRetry(retry: RetryOptions | RetryPolicy): MycelixKnowledgeClient {
    return new MycelixKnowledgeClient(this.client, { ...this.config, retry });
  }
}
