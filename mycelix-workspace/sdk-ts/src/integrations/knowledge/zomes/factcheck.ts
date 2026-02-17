/**
 * Fact-Check Zome Client
 *
 * Handles fact-checking operations and cross-hApp integration.
 *
 * @module @mycelix/sdk/integrations/knowledge/zomes/factcheck
 */

import { KnowledgeSdkError } from '../types';

import type {
  FactCheckRequest,
  SubmitFactCheckInput,
  CompleteFactCheckInput,
  FactCheckResult,
  FactCheckVerdict,
  CrossHappLink,
  LinkExternalInput,
  GetLinkedClaimsInput,
  KnowledgeAuthority,
  Claim,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Default configuration for the FactCheck client
 */
export interface FactCheckClientConfig {
  /** Role ID for the knowledge DNA */
  roleId: string;
  /** Zome name */
  zomeName: string;
}

const DEFAULT_CONFIG: FactCheckClientConfig = {
  roleId: 'knowledge',
  zomeName: 'knowledge',
};

/**
 * Client for fact-checking and cross-hApp operations
 *
 * @example
 * ```typescript
 * const factCheck = new FactCheckClient(holochainClient);
 *
 * // Submit a fact-check request from Media hApp
 * const request = await factCheck.submitFactCheckRequest({
 *   id: 'fc-001',
 *   media_content_id: 'media-abc',
 *   claim_text: 'The Earth is flat',
 *   requested_by: 'did:mycelix:user123',
 * });
 *
 * // Later, complete the fact-check
 * const result = await factCheck.completeFactCheck({
 *   request_id: 'fc-001',
 *   verdict: 'False',
 *   confidence: 0.99,
 *   supporting_claims: [],
 *   contradicting_claims: ['claim-spherical-earth'],
 *   explanation: 'Overwhelming scientific evidence...',
 *   sources: ['source-nasa', 'source-physics'],
 *   completed_by: 'did:mycelix:factchecker',
 * });
 * ```
 */
export class FactCheckClient {
  private readonly config: FactCheckClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<FactCheckClientConfig> = {}
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
  // Fact-Check Operations
  // ============================================================================

  /**
   * Submit a fact-check request
   *
   * Used by Media hApp to request verification of claims.
   *
   * @param input - Fact-check request parameters
   * @returns The created request
   */
  async submitFactCheckRequest(input: SubmitFactCheckInput): Promise<FactCheckRequest> {
    const record = await this.call<HolochainRecord>('submit_factcheck_request', input);
    return this.extractEntry<FactCheckRequest>(record);
  }

  /**
   * Complete a fact-check
   *
   * Called by fact-checkers to submit their verdict.
   *
   * @param input - Fact-check completion parameters
   * @returns The result
   */
  async completeFactCheck(input: CompleteFactCheckInput): Promise<Record<string, unknown>> {
    return this.call<Record<string, unknown>>('complete_factcheck', input);
  }

  /**
   * Verify a claim for external fact-checking
   *
   * @param claimId - Claim identifier
   * @returns Verification data
   */
  async verifyClaimForFactCheck(claimId: string): Promise<Record<string, unknown>> {
    return this.call<Record<string, unknown>>('verify_claim_for_factcheck', claimId);
  }

  // ============================================================================
  // Cross-hApp Integration
  // ============================================================================

  /**
   * Link a claim to an external hApp entry
   *
   * @param input - Link parameters
   * @returns The created link
   */
  async linkToExternal(input: LinkExternalInput): Promise<CrossHappLink> {
    const record = await this.call<HolochainRecord>('link_to_external', input);
    return this.extractEntry<CrossHappLink>(record);
  }

  /**
   * Get claims linked to an external entry
   *
   * @param input - Query parameters
   * @returns Array of linked claims
   */
  async getLinkedClaims(input: GetLinkedClaimsInput): Promise<Claim[]> {
    const records = await this.call<HolochainRecord[]>('get_linked_claims', input);
    return records.map(r => this.extractEntry<Claim>(r));
  }

  /**
   * Query knowledge data for cross-hApp requests
   *
   * @param did - DID to query
   * @returns Knowledge summary
   */
  async queryKnowledge(did: string): Promise<Record<string, unknown>> {
    return this.call<Record<string, unknown>>('query_knowledge', did);
  }

  // ============================================================================
  // Authority Operations
  // ============================================================================

  /**
   * Calculate knowledge authority for a DID
   *
   * @param did - DID to analyze
   * @returns Authority score and details
   */
  async calculateKnowledgeAuthority(did: string): Promise<KnowledgeAuthority> {
    return this.call<KnowledgeAuthority>('calculate_knowledge_authority', did);
  }

  // ============================================================================
  // Helpers
  // ============================================================================

  /**
   * Quick fact-check: submit and wait for automated processing
   *
   * @param claimText - Claim to verify
   * @param requestedBy - Requester's DID
   * @param mediaContentId - Associated media content
   * @returns Fact-check request
   */
  async quickFactCheck(
    claimText: string,
    requestedBy: string,
    mediaContentId: string = 'direct-request'
  ): Promise<FactCheckRequest> {
    const id = `fc-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    return this.submitFactCheckRequest({
      id,
      media_content_id: mediaContentId,
      claim_text: claimText,
      requested_by: requestedBy,
    });
  }

  /**
   * Link claim to media content
   *
   * Convenience method for linking claims to Media hApp entries.
   *
   * @param claimId - Claim identifier
   * @param mediaId - Media content identifier
   * @param createdBy - Creator's DID
   * @returns The created link
   */
  async linkToMedia(
    claimId: string,
    mediaId: string,
    createdBy: string
  ): Promise<CrossHappLink> {
    return this.linkToExternal({
      id: `link-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
      claim_id: claimId,
      target_happ: 'media',
      target_entry_type: 'MediaContent',
      target_entry_id: mediaId,
      link_type: 'Describes',
      created_by: createdBy,
    });
  }

  /**
   * Link claim to governance proposal
   *
   * @param claimId - Claim identifier
   * @param proposalId - Proposal identifier
   * @param createdBy - Creator's DID
   * @returns The created link
   */
  async linkToProposal(
    claimId: string,
    proposalId: string,
    createdBy: string
  ): Promise<CrossHappLink> {
    return this.linkToExternal({
      id: `link-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
      claim_id: claimId,
      target_happ: 'governance',
      target_entry_type: 'Proposal',
      target_entry_id: proposalId,
      link_type: 'Substantiates',
      created_by: createdBy,
    });
  }

  /**
   * Get verdict confidence level description
   *
   * @param result - Fact-check result
   * @returns Human-readable confidence description
   */
  getConfidenceDescription(result: FactCheckResult): string {
    const { confidence } = result;
    if (confidence >= 0.95) return 'Very High Confidence';
    if (confidence >= 0.8) return 'High Confidence';
    if (confidence >= 0.6) return 'Moderate Confidence';
    if (confidence >= 0.4) return 'Low Confidence';
    return 'Very Low Confidence';
  }

  /**
   * Get verdict severity
   *
   * @param verdict - Fact-check verdict
   * @returns Severity level (1-5, where 5 is most severe)
   */
  getVerdictSeverity(verdict: FactCheckVerdict): number {
    const severityMap: Record<FactCheckVerdict, number> = {
      True: 1,
      MostlyTrue: 2,
      Mixed: 3,
      OutOfContext: 3,
      Satire: 2,
      MostlyFalse: 4,
      False: 5,
      Unverifiable: 3,
    };
    return severityMap[verdict] ?? 3;
  }
}
