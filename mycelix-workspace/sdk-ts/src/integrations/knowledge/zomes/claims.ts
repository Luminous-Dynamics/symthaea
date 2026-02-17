/**
 * Claims Zome Client
 *
 * Handles claim creation, updates, endorsements, and queries.
 *
 * @module @mycelix/sdk/integrations/knowledge/zomes/claims
 */

import { KnowledgeSdkError } from '../types';

import type {
  Claim,
  SubmitClaimInput,
  ClaimUpdate,
  ClaimEndorsement,
  Ontology,
  Source,
  Citation,
  AddSourceInput,
  CiteSourceInput,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Default configuration for the Claims client
 */
export interface ClaimsClientConfig {
  /** Role ID for the knowledge DNA */
  roleId: string;
  /** Zome name */
  zomeName: string;
}

const DEFAULT_CONFIG: ClaimsClientConfig = {
  roleId: 'knowledge',
  zomeName: 'knowledge',
};

/**
 * Client for claim operations
 *
 * @example
 * ```typescript
 * const claims = new ClaimsClient(holochainClient);
 *
 * // Submit a new claim
 * const claim = await claims.submitClaim({
 *   id: 'claim-001',
 *   author_did: 'did:mycelix:abc123',
 *   title: 'Climate Change Evidence',
 *   content: 'Global temperatures have risen...',
 *   empirical: 0.9,
 *   normative: 0.1,
 *   metaphysical: 0.0,
 *   tags: ['climate', 'science', 'environment'],
 * });
 *
 * // Endorse a claim
 * await claims.endorseClaim({
 *   id: 'endorsement-001',
 *   claim_id: 'claim-001',
 *   endorser_did: 'did:mycelix:def456',
 *   weight: 0.8,
 *   created_at: Date.now() * 1000,
 * });
 * ```
 */
export class ClaimsClient {
  private readonly config: ClaimsClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<ClaimsClientConfig> = {}
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
      const message = error instanceof Error ? error.message : String(error);

      if (message.includes('must be 0-1')) {
        throw new KnowledgeSdkError('INVALID_ENM_SCORES', message, error);
      }

      throw new KnowledgeSdkError(
        'ZOME_ERROR',
        `Failed to call ${fnName}: ${message}`,
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
  // Claim Operations
  // ============================================================================

  /**
   * Submit a new claim
   *
   * @param input - Claim parameters
   * @returns The created claim
   * @throws {KnowledgeSdkError} If E/N/M scores are not in 0-1 range
   */
  async submitClaim(input: SubmitClaimInput): Promise<Claim> {
    // Validate E/N/M scores
    if (input.empirical < 0 || input.empirical > 1) {
      throw new KnowledgeSdkError('INVALID_ENM_SCORES', 'Empirical score must be 0-1');
    }
    if (input.normative < 0 || input.normative > 1) {
      throw new KnowledgeSdkError('INVALID_ENM_SCORES', 'Normative score must be 0-1');
    }
    if (input.metaphysical < 0 || input.metaphysical > 1) {
      throw new KnowledgeSdkError('INVALID_ENM_SCORES', 'Metaphysical score must be 0-1');
    }

    const now = Date.now() * 1000;
    const claimInput: Claim = {
      ...input,
      created_at: now,
      updated_at: now,
      endorsement_count: 0,
    };

    const record = await this.call<HolochainRecord>('submit_claim', claimInput);
    return this.extractEntry<Claim>(record);
  }

  /**
   * Get a claim by ID
   *
   * @param claimId - Claim identifier
   * @returns The claim or null if not found
   */
  async getClaim(claimId: string): Promise<Claim | null> {
    const record = await this.call<HolochainRecord | null>('get_claim', claimId);
    if (!record) return null;
    return this.extractEntry<Claim>(record);
  }

  /**
   * Get claims by author
   *
   * @param authorDid - Author's DID
   * @returns Array of claims
   */
  async getClaimsByAuthor(authorDid: string): Promise<Claim[]> {
    const records = await this.call<HolochainRecord[]>('get_claims_by_author', authorDid);
    return records.map(r => this.extractEntry<Claim>(r));
  }

  /**
   * Get claims by tag
   *
   * @param tag - Tag to search for
   * @returns Array of claims
   */
  async getClaimsByTag(tag: string): Promise<Claim[]> {
    const records = await this.call<HolochainRecord[]>('get_claims_by_tag', tag);
    return records.map(r => this.extractEntry<Claim>(r));
  }

  /**
   * Update a claim
   *
   * @param claimId - Claim identifier
   * @param updates - Fields to update
   * @returns The updated claim
   */
  async updateClaim(claimId: string, updates: ClaimUpdate): Promise<Claim> {
    const record = await this.call<HolochainRecord>('update_claim', {
      claim_id: claimId,
      ...updates,
    });
    return this.extractEntry<Claim>(record);
  }

  // ============================================================================
  // Endorsement Operations
  // ============================================================================

  /**
   * Endorse a claim
   *
   * @param endorsement - Endorsement parameters
   * @returns The created endorsement
   */
  async endorseClaim(endorsement: ClaimEndorsement): Promise<ClaimEndorsement> {
    const record = await this.call<HolochainRecord>('endorse_claim', endorsement);
    return this.extractEntry<ClaimEndorsement>(record);
  }

  /**
   * Remove an endorsement
   *
   * @param endorsementId - Endorsement identifier
   */
  async removeEndorsement(endorsementId: string): Promise<void> {
    await this.call<void>('remove_endorsement', endorsementId);
  }

  // ============================================================================
  // Ontology Operations
  // ============================================================================

  /**
   * Register a new ontology
   *
   * @param ontology - Ontology parameters
   * @returns The created ontology
   */
  async registerOntology(ontology: Ontology): Promise<Ontology> {
    const record = await this.call<HolochainRecord>('register_ontology', ontology);
    return this.extractEntry<Ontology>(record);
  }

  /**
   * Get an ontology by ID
   *
   * @param ontologyId - Ontology identifier
   * @returns The ontology or null if not found
   */
  async getOntology(ontologyId: string): Promise<Ontology | null> {
    const record = await this.call<HolochainRecord | null>('get_ontology', ontologyId);
    if (!record) return null;
    return this.extractEntry<Ontology>(record);
  }

  // ============================================================================
  // Source and Citation Operations
  // ============================================================================

  /**
   * Add a source
   *
   * @param input - Source parameters
   * @returns The created source
   */
  async addSource(input: AddSourceInput): Promise<Source> {
    const record = await this.call<HolochainRecord>('add_source', input);
    return this.extractEntry<Source>(record);
  }

  /**
   * Cite a source in a claim
   *
   * @param input - Citation parameters
   * @returns The created citation
   */
  async citeSource(input: CiteSourceInput): Promise<Citation> {
    const record = await this.call<HolochainRecord>('cite_source', input);
    return this.extractEntry<Citation>(record);
  }

  /**
   * Get citations for a claim
   *
   * @param claimId - Claim identifier
   * @returns Array of citations
   */
  async getClaimCitations(claimId: string): Promise<Citation[]> {
    const records = await this.call<HolochainRecord[]>('get_claim_citations', claimId);
    return records.map(r => this.extractEntry<Citation>(r));
  }

  // ============================================================================
  // Helpers
  // ============================================================================

  /**
   * Classify a claim by its dominant E/N/M dimension
   *
   * @param claim - Claim to classify
   * @returns Dominant dimension
   */
  classifyDimension(claim: Claim): 'Empirical' | 'Normative' | 'Metaphysical' {
    const { empirical, normative, metaphysical } = claim;

    if (empirical >= normative && empirical >= metaphysical) {
      return 'Empirical';
    }
    if (normative >= empirical && normative >= metaphysical) {
      return 'Normative';
    }
    return 'Metaphysical';
  }

  /**
   * Calculate E/N/M vector magnitude
   *
   * @param claim - Claim to analyze
   * @returns Magnitude (0-√3)
   */
  getENMMagnitude(claim: Claim): number {
    const { empirical, normative, metaphysical } = claim;
    return Math.sqrt(empirical ** 2 + normative ** 2 + metaphysical ** 2);
  }
}
