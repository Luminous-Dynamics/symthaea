/**
 * Verification Client
 *
 * Handles safety verification with Knowledge hApp integration:
 * - Design verification submission
 * - Safety claims with epistemic classification
 * - Knowledge hApp bridging (via coordinator cross-zome calls)
 */

import type { AppClient, ActionHash, Record } from '@holochain/client';
import type {
  VerificationSummary,
  SubmitVerificationInput,
  SubmitClaimInput,
  EpistemicScore,
} from '../types';

export interface PaginationInput {
  offset: number;
  limit: number;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  offset: number;
  limit: number;
}

export class VerificationClient {
  constructor(
    private client: AppClient,
    private roleName: string,
    private zomeName: string = 'verification_coordinator'
  ) {}

  // =========================================================================
  // VERIFICATION
  // =========================================================================

  /**
   * Submit a design verification
   */
  async submitVerification(input: SubmitVerificationInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'submit_verification',
      payload: input,
    });
  }

  /**
   * Get all verifications for a design
   */
  async getDesignVerifications(
    designHash: ActionHash,
    pagination?: PaginationInput
  ): Promise<PaginatedResponse<Record>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_design_verifications',
      payload: { hash: designHash, pagination: pagination ?? null },
    });
  }

  /**
   * Get verification summary for a design
   */
  async getVerificationSummary(designHash: ActionHash): Promise<VerificationSummary> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_verification_summary',
      payload: designHash,
    });
  }

  // =========================================================================
  // SAFETY CLAIMS (Knowledge Bridge)
  // =========================================================================

  /**
   * Submit a safety claim
   *
   * Claims are classified using epistemic dimensions:
   * - Empirical (E): Measurable, testable facts
   * - Normative (N): Standards, requirements, best practices
   * - Mythic (M): Theoretical, intuitive, experiential knowledge
   */
  async submitSafetyClaim(input: SubmitClaimInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'submit_safety_claim',
      payload: input,
    });
  }

  /**
   * Get all claims for a design
   */
  async getDesignClaims(
    designHash: ActionHash,
    pagination?: PaginationInput
  ): Promise<PaginatedResponse<Record>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_design_claims',
      payload: { hash: designHash, pagination: pagination ?? null },
    });
  }

  /**
   * Get epistemic score for a design
   *
   * Returns aggregated E/N/M scores from all claims.
   * Scores come from Knowledge hApp cross-zome call (with fallback defaults).
   */
  async getEpistemicScore(designHash: ActionHash): Promise<EpistemicScore> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_epistemic_score',
      payload: designHash,
    });
  }
}
