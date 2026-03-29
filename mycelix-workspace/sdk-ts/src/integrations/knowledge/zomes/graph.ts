// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Graph Zome Client
 *
 * Handles knowledge graph operations including relations, traversal, and search.
 *
 * @module @mycelix/sdk/integrations/knowledge/zomes/graph
 */

import { KnowledgeSdkError } from '../types';

import type {
  ClaimRelation,

  TraversalInput,
  TraversalResult,
  PathInput,
  ClaimPath,
  ClaimCluster,
  GraphQuery,
  SemanticSearchInput,
  SearchResult,
  ConsistencyResult,
  ReasoningInput,
  ReasoningResult,
  ContradictionReport,
  InferenceRequest,

  Claim,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Default configuration for the Graph client
 */
export interface GraphClientConfig {
  /** Role ID for the knowledge DNA */
  roleId: string;
  /** Zome name */
  zomeName: string;
}

const DEFAULT_CONFIG: GraphClientConfig = {
  roleId: 'knowledge',
  zomeName: 'knowledge',
};

/**
 * Client for knowledge graph operations
 *
 * @example
 * ```typescript
 * const graph = new GraphClient(holochainClient);
 *
 * // Create a relation between claims
 * await graph.createRelation({
 *   id: 'rel-001',
 *   source_claim_id: 'claim-001',
 *   target_claim_id: 'claim-002',
 *   relation_type: 'Supports',
 *   strength: 0.9,
 *   created_by: 'did:mycelix:abc123',
 *   created_at: Date.now() * 1000,
 * });
 *
 * // Traverse the graph
 * const result = await graph.traverseGraph({
 *   start_claim_id: 'claim-001',
 *   direction: 'Both',
 *   max_depth: 3,
 *   limit: 100,
 * });
 *
 * // Semantic search
 * const results = await graph.semanticSearch({
 *   query: 'climate change evidence',
 *   limit: 10,
 *   offset: 0,
 * });
 * ```
 */
export class GraphClient {
  private readonly config: GraphClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<GraphClientConfig> = {}
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Call a zome function with error handling
   */
  private async call<T>(fnName: string, payload: unknown): Promise<T> {
    try {
      const result = await this.client.callZome({
        role_name: this.config.roleId,
        zome_name: this.config.zomeName,
        fn_name: fnName,
        payload,
      });
      return result as T;
    } catch (error) {
      throw new KnowledgeSdkError(
        'ZOME_ERROR',
        `Failed to call ${fnName}: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Extract entry from a Holochain record
   */
  private extractEntry<T>(record: HolochainRecord): T {
    if (!record.entry || !('Present' in record.entry)) {
      throw new KnowledgeSdkError(
        'INVALID_INPUT',
        'Record does not contain an entry'
      );
    }
    return (record.entry as unknown as { Present: { entry: T } }).Present.entry;
  }

  // ============================================================================
  // Relation Operations
  // ============================================================================

  /**
   * Create a relation between claims
   *
   * @param relation - Relation parameters
   * @returns The created relation
   */
  async createRelation(relation: ClaimRelation): Promise<ClaimRelation> {
    const record = await this.call<HolochainRecord>('create_relation', relation);
    return this.extractEntry<ClaimRelation>(record);
  }

  /**
   * Get relations for a claim
   *
   * @param claimId - Claim identifier
   * @returns Array of relations
   */
  async getClaimRelations(claimId: string): Promise<ClaimRelation[]> {
    const records = await this.call<HolochainRecord[]>('get_claim_relations', claimId);
    return records.map(r => this.extractEntry<ClaimRelation>(r));
  }

  /**
   * Get claims that support a given claim
   *
   * @param claimId - Claim identifier
   * @returns Array of supporting claims
   */
  async getSupportingClaims(claimId: string): Promise<Claim[]> {
    const records = await this.call<HolochainRecord[]>('get_supporting_claims', claimId);
    return records.map(r => this.extractEntry<Claim>(r));
  }

  /**
   * Get claims that contradict a given claim
   *
   * @param claimId - Claim identifier
   * @returns Array of contradicting claims
   */
  async getContradictingClaims(claimId: string): Promise<Claim[]> {
    const records = await this.call<HolochainRecord[]>('get_contradicting_claims', claimId);
    return records.map(r => this.extractEntry<Claim>(r));
  }

  // ============================================================================
  // Graph Traversal
  // ============================================================================

  /**
   * Traverse the claim graph from a starting point
   *
   * Uses BFS to explore the graph, following relations up to the specified
   * depth and collecting all reachable claims.
   *
   * @param input - Traversal parameters
   * @returns Traversal result with visited claims and paths
   */
  async traverseGraph(input: TraversalInput): Promise<TraversalResult> {
    return this.call<TraversalResult>('traverse_graph', input);
  }

  /**
   * Find the shortest path between two claims
   *
   * @param input - Path finding parameters
   * @returns The path or null if no path exists
   */
  async findPath(input: PathInput): Promise<ClaimPath | null> {
    return this.call<ClaimPath | null>('find_path', input);
  }

  /**
   * Get the connected component (cluster) containing a claim
   *
   * @param claimId - Center claim identifier
   * @param maxSize - Maximum cluster size
   * @returns Cluster information
   */
  async getClaimCluster(claimId: string, maxSize: number = 100): Promise<ClaimCluster> {
    return this.call<ClaimCluster>('get_claim_cluster', { claim_id: claimId, max_size: maxSize });
  }

  // ============================================================================
  // Search Operations
  // ============================================================================

  /**
   * Query the knowledge graph
   *
   * @param query - Query parameters
   * @returns Array of matching claim data
   */
  async queryGraph(query: GraphQuery): Promise<Record<string, unknown>[]> {
    return this.call<Record<string, unknown>[]>('query_graph', query);
  }

  /**
   * Semantic search for claims
   *
   * Searches claims by content similarity with optional filters.
   *
   * @param input - Search parameters
   * @returns Array of search results
   */
  async semanticSearch(input: SemanticSearchInput): Promise<SearchResult[]> {
    return this.call<SearchResult[]>('semantic_search', input);
  }

  // ============================================================================
  // Inference and Consistency
  // ============================================================================

  /**
   * Request inference on claims
   *
   * @param input - Inference request parameters
   * @returns The inference request
   */
  async requestInference(input: InferenceRequest): Promise<InferenceRequest> {
    const record = await this.call<HolochainRecord>('request_inference', input);
    return this.extractEntry<InferenceRequest>(record);
  }

  /**
   * Verify consistency among a set of claims
   *
   * Checks for logical consistency by:
   * 1. Detecting explicit contradiction relations
   * 2. Analyzing E/N/M dimensional conflicts
   * 3. Checking for semantic contradictions
   *
   * @param claimIds - Array of claim identifiers
   * @returns Consistency result
   */
  async verifyConsistency(claimIds: string[]): Promise<ConsistencyResult> {
    return this.call<ConsistencyResult>('verify_consistency', claimIds);
  }

  /**
   * Perform logical reasoning on claims
   *
   * @param input - Reasoning parameters
   * @returns Reasoning result
   */
  async reasonOverClaims(input: ReasoningInput): Promise<ReasoningResult> {
    return this.call<ReasoningResult>('reason_over_claims', input);
  }

  /**
   * Detect potential contradictions among claims
   *
   * @param claimIds - Array of claim identifiers
   * @returns Contradiction report
   */
  async detectContradictions(claimIds: string[]): Promise<ContradictionReport> {
    return this.call<ContradictionReport>('detect_contradictions', claimIds);
  }

  // ============================================================================
  // Helpers
  // ============================================================================

  /**
   * Create a supporting relation
   *
   * @param sourceClaimId - Source claim (the one providing support)
   * @param targetClaimId - Target claim (the one being supported)
   * @param createdBy - Creator's DID
   * @param strength - Relation strength (0-1)
   * @returns The created relation
   */
  async createSupportRelation(
    sourceClaimId: string,
    targetClaimId: string,
    createdBy: string,
    strength: number = 1.0
  ): Promise<ClaimRelation> {
    return this.createRelation({
      id: `rel-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
      source_claim_id: sourceClaimId,
      target_claim_id: targetClaimId,
      relation_type: 'Supports',
      strength,
      created_by: createdBy,
      created_at: Date.now() * 1000,
    });
  }

  /**
   * Create a contradiction relation
   *
   * @param sourceClaimId - Source claim
   * @param targetClaimId - Target claim
   * @param createdBy - Creator's DID
   * @param strength - Relation strength (0-1)
   * @returns The created relation
   */
  async createContradictionRelation(
    sourceClaimId: string,
    targetClaimId: string,
    createdBy: string,
    strength: number = 1.0
  ): Promise<ClaimRelation> {
    return this.createRelation({
      id: `rel-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
      source_claim_id: sourceClaimId,
      target_claim_id: targetClaimId,
      relation_type: 'Contradicts',
      strength,
      created_by: createdBy,
      created_at: Date.now() * 1000,
    });
  }

  /**
   * Get all claims reachable from a starting point
   *
   * Convenience method for simple traversal.
   *
   * @param claimId - Starting claim identifier
   * @param maxDepth - Maximum depth to traverse
   * @returns Array of reachable claim IDs
   */
  async getReachableClaims(claimId: string, maxDepth: number = 3): Promise<string[]> {
    const result = await this.traverseGraph({
      start_claim_id: claimId,
      direction: 'Both',
      max_depth: maxDepth,
      limit: 1000,
    });
    return result.visited_claims;
  }

  /**
   * Check if two claims are connected
   *
   * @param claimA - First claim identifier
   * @param claimB - Second claim identifier
   * @param maxLength - Maximum path length to search
   * @returns True if connected
   */
  async areClaimsConnected(
    claimA: string,
    claimB: string,
    maxLength: number = 5
  ): Promise<boolean> {
    const path = await this.findPath({
      from_claim: claimA,
      to_claim: claimB,
      max_length: maxLength,
    });
    return path !== null;
  }
}
