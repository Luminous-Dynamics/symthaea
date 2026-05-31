// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Verification Client
 *
 * Handles safety verification with Knowledge hApp integration:
 * - Design verification submission
 * - Safety claims with epistemic classification
 * - Knowledge hApp bridging
 */

import type { AppClient, ActionHash, Record } from '@holochain/client';
import type {
  DesignVerification,
  VerificationType,
  VerificationResult,
  SafetyClaim,
  SafetyClaimType,
  ClaimEpistemic,
  EpistemicScore,
  VerificationSummary,
  SubmitVerificationInput,
  SubmitClaimInput,
  SafetyClass,
} from '../types';

export interface VerificationRequestInput {
  designHash: ActionHash;
  targetSafetyClass: SafetyClass;
  bounty?: number;
  deadline?: number;
}

export interface KnowledgeQueryResult {
  claims: Array<{
    hash: ActionHash;
    text: string;
    epistemic: ClaimEpistemic;
    confidence: number;
  }>;
  relevanceScore: number;
}

export class VerificationClient {
  constructor(
    private client: AppClient,
    private roleName: string,
    private zomeName: string = 'verification'
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
  async getDesignVerifications(designHash: ActionHash): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_design_verifications',
      payload: designHash,
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
   * Verify a safety claim
   */
  async verifySafetyClaim(claimHash: ActionHash): Promise<VerificationResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'verify_safety_claim',
      payload: claimHash,
    });
  }

  /**
   * Get all claims for a design
   */
  async getDesignClaims(designHash: ActionHash): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_design_claims',
      payload: designHash,
    });
  }

  /**
   * Dispute a safety claim
   */
  async disputeClaim(claimHash: ActionHash, reason: string): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'dispute_claim',
      payload: { claim: claimHash, reason },
    });
  }

  // =========================================================================
  // KNOWLEDGE hApp BRIDGE
  // =========================================================================

  /**
   * Bridge a local claim to Knowledge hApp
   */
  async bridgeToKnowledge(localClaimHash: ActionHash): Promise<ActionHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'bridge_to_knowledge_claim',
      payload: localClaimHash,
    });
  }

  /**
   * Query Knowledge hApp for relevant claims
   */
  async queryKnowledge(claimText: string): Promise<KnowledgeQueryResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'query_knowledge_for_claim',
      payload: claimText,
    });
  }

  /**
   * Get epistemic score for a design
   *
   * Returns aggregated E/N/M scores from all claims
   */
  async getEpistemicScore(designHash: ActionHash): Promise<EpistemicScore> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_epistemic_score',
      payload: designHash,
    });
  }

  // =========================================================================
  // VERIFICATION REQUESTS
  // =========================================================================

  /**
   * Create a verification request (with optional bounty)
   */
  async createRequest(input: VerificationRequestInput): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'create_verification_request',
      payload: input,
    });
  }

  /**
   * Accept a verification request
   */
  async acceptRequest(requestHash: ActionHash): Promise<Record> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'accept_verification_request',
      payload: requestHash,
    });
  }

  /**
   * Get open verification requests
   */
  async getOpenRequests(): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_open_verification_requests',
      payload: null,
    });
  }

  /**
   * Get my verification requests
   */
  async getMyRequests(): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_my_verification_requests',
      payload: null,
    });
  }

  /**
   * Get my verifications (as verifier)
   */
  async getMyVerifications(): Promise<Record[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: 'get_my_verifications',
      payload: null,
    });
  }
}
