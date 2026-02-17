/**
 * Epistemic Client
 *
 * Classification, DKG claims, and dispute resolution.
 */

import { type AgentPubKey, type Record as HolochainRecord } from '@holochain/client';

import type { DAOClient } from './client';
import type {
  EpistemicClassification,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  DKGClaim,
  VerificationMethod,
  VerificationState,
  DisputeTarget,
  DisputedAxis,
  ResolutionBody,
  StoreDKGClaimInput,
  FileDisputeInput,
} from './types';

/**
 * Dispute entry type
 */
export interface Dispute {
  id: string;
  target: DisputeTarget;
  disputed_axis: DisputedAxis;
  filer: AgentPubKey;
  justification: string;
  resolution_body: ResolutionBody;
  status: 'Filed' | 'UnderReview' | 'Resolved' | 'Dismissed';
  resolution?: string;
  created_at: number;
  resolved_at?: number;
}

/**
 * Default classifications by proposal type
 */
const DEFAULT_CLASSIFICATIONS: Record<string, EpistemicClassification> = {
  TechnicalMIP: { e: 'E3', n: 'N1', m: 'M2', override_from_default: false },
  EconomicMIP: { e: 'E2', n: 'N2', m: 'M3', override_from_default: false },
  GovernanceMIP: { e: 'E1', n: 'N3', m: 'M2', override_from_default: false },
  SocialMIP: { e: 'E1', n: 'N2', m: 'M1', override_from_default: false },
  CulturalMIP: { e: 'E0', n: 'N2', m: 'M1', override_from_default: false },
  ConstitutionalAmendment: { e: 'E1', n: 'N3', m: 'M3', override_from_default: false },
  TreasuryAllocation: { e: 'E3', n: 'N2', m: 'M3', override_from_default: false },
  StreamingGrant: { e: 'E2', n: 'N1', m: 'M2', override_from_default: false },
  EmergencyAction: { e: 'E2', n: 'N3', m: 'M3', override_from_default: false },
};

/**
 * Client for epistemic operations
 */
export class EpistemicClient {
  constructor(private client: DAOClient) {}

  private get zome() {
    return this.client.getZomeName('epistemic');
  }

  // =========================================================================
  // CLASSIFICATION
  // =========================================================================

  /**
   * Get suggested classification for a proposal type
   *
   * @example
   * ```typescript
   * const classification = await epistemic.suggestClassification('TechnicalMIP');
   * console.log(`Suggested: E${classification.e} N${classification.n} M${classification.m}`);
   * ```
   */
  async suggestClassification(proposalType: string): Promise<EpistemicClassification> {
    // First try to get from zome (which may have learned preferences)
    try {
      return await this.client.callZome<EpistemicClassification>(
        this.zome,
        'suggest_classification',
        proposalType
      );
    } catch {
      // Fall back to defaults
      return DEFAULT_CLASSIFICATIONS[proposalType] || {
        e: 'E1',
        n: 'N1',
        m: 'M1',
        override_from_default: false,
      };
    }
  }

  /**
   * Validate a classification against proposal type constraints
   */
  async validateClassification(
    proposalType: string,
    classification: EpistemicClassification
  ): Promise<{ valid: boolean; warnings: string[] }> {
    return this.client.callZome(
      this.zome,
      'validate_classification',
      { proposal_type: proposalType, classification }
    );
  }

  /**
   * Calculate epistemic score from classification
   *
   * Score = (E-level × 25) + (N-level × 33) + (M-level × 33)
   * Range: 0-100
   */
  calculateEpistemicScore(classification: EpistemicClassification): number {
    const eScore = parseInt(classification.e[1]) * 25; // E0-E4 → 0-100
    const nScore = parseInt(classification.n[1]) * 33; // N0-N3 → 0-99
    const mScore = parseInt(classification.m[1]) * 33; // M0-M3 → 0-99

    return Math.min(100, Math.round((eScore + nScore + mScore) / 3));
  }

  // =========================================================================
  // DKG (DECENTRALIZED KNOWLEDGE GRAPH)
  // =========================================================================

  /**
   * Store a new DKG claim
   *
   * @example
   * ```typescript
   * const claim = await epistemic.storeClaim({
   *   content: 'Solar panel efficiency increased by 15% with new coating',
   *   classification: { e: 'E4', n: 'N0', m: 'M2', override_from_default: false },
   *   verification_method: 'PublicReproduction',
   *   proof_cid: 'bafybeig...',
   * });
   * ```
   */
  async storeClaim(input: StoreDKGClaimInput): Promise<DKGClaim> {
    const record = await this.client.callZome<HolochainRecord>(
      this.zome,
      'store_dkg_claim',
      input
    );

    return this.decodeClaim(record);
  }

  /**
   * Get a claim by ID
   */
  async getClaim(claimId: string): Promise<DKGClaim | null> {
    const record = await this.client.callZome<HolochainRecord | null>(
      this.zome,
      'get_dkg_claim',
      claimId
    );

    if (!record) return null;
    return this.decodeClaim(record);
  }

  /**
   * Get all claims by author
   */
  async getClaimsByAuthor(author: AgentPubKey): Promise<DKGClaim[]> {
    const records = await this.client.callZome<HolochainRecord[]>(
      this.zome,
      'get_claims_by_author',
      author
    );

    return records.map(r => this.decodeClaim(r));
  }

  /**
   * Get related claims
   */
  async getRelatedClaims(claimId: string): Promise<DKGClaim[]> {
    const records = await this.client.callZome<HolochainRecord[]>(
      this.zome,
      'get_related_claims',
      claimId
    );

    return records.map(r => this.decodeClaim(r));
  }

  /**
   * Link claims with a relationship
   */
  async linkClaims(
    sourceClaimId: string,
    targetClaimId: string,
    relationType: 'Supports' | 'Refutes' | 'Clarifies' | 'Supercedes' | 'Implements' | 'Disputes'
  ): Promise<void> {
    await this.client.callZome(
      this.zome,
      'link_claims',
      { source_claim_id: sourceClaimId, target_claim_id: targetClaimId, relation_type: relationType }
    );
  }

  /**
   * Update claim verification status
   */
  async updateVerification(
    claimId: string,
    method: VerificationMethod,
    status: VerificationState,
    proofCid?: string
  ): Promise<DKGClaim> {
    const record = await this.client.callZome<HolochainRecord>(
      this.zome,
      'update_claim_verification',
      { claim_id: claimId, method, status, proof_cid: proofCid }
    );

    return this.decodeClaim(record);
  }

  // =========================================================================
  // DISPUTES
  // =========================================================================

  /**
   * File a dispute against a proposal, claim, or action
   *
   * @example
   * ```typescript
   * await epistemic.fileDispute({
   *   target: { Proposal: proposalHash },
   *   disputed_axis: 'Empirical',
   *   justification: 'The claimed data is not reproducible...',
   * });
   * ```
   */
  async fileDispute(input: FileDisputeInput): Promise<Dispute> {
    const record = await this.client.callZome<HolochainRecord>(
      this.zome,
      'file_dispute',
      input
    );

    return this.decodeDispute(record);
  }

  /**
   * Get a dispute by ID
   */
  async getDispute(disputeId: string): Promise<Dispute | null> {
    const record = await this.client.callZome<HolochainRecord | null>(
      this.zome,
      'get_dispute',
      disputeId
    );

    if (!record) return null;
    return this.decodeDispute(record);
  }

  /**
   * Get all disputes for a target
   */
  async getDisputesForTarget(target: DisputeTarget): Promise<Dispute[]> {
    const records = await this.client.callZome<HolochainRecord[]>(
      this.zome,
      'get_disputes_for_target',
      target
    );

    return records.map(r => this.decodeDispute(r));
  }

  /**
   * Submit evidence for a dispute
   */
  async submitEvidence(
    disputeId: string,
    evidenceCid: string,
    description: string
  ): Promise<void> {
    await this.client.callZome(
      this.zome,
      'submit_dispute_evidence',
      { dispute_id: disputeId, evidence_cid: evidenceCid, description }
    );
  }

  /**
   * Get disputes filed by the current user
   */
  async getMyDisputes(): Promise<Dispute[]> {
    const myPubKey = await this.client.getMyPubKey();
    const records = await this.client.callZome<HolochainRecord[]>(
      this.zome,
      'get_disputes_by_filer',
      myPubKey
    );

    return records.map(r => this.decodeDispute(r));
  }

  // =========================================================================
  // HELPERS
  // =========================================================================

  /**
   * Parse an epistemic classification string (e.g., "E2N1M3")
   */
  parseClassificationString(str: string): EpistemicClassification | null {
    const match = str.match(/E([0-4])N([0-3])M([0-3])/i);
    if (!match) return null;

    return {
      e: `E${match[1]}` as EmpiricalLevel,
      n: `N${match[2]}` as NormativeLevel,
      m: `M${match[3]}` as MaterialityLevel,
      override_from_default: false,
    };
  }

  /**
   * Format classification as string (e.g., "E2N1M3")
   */
  formatClassification(classification: EpistemicClassification): string {
    return `${classification.e}${classification.n}${classification.m}`;
  }

  /**
   * Get human-readable description of a classification
   */
  describeClassification(classification: EpistemicClassification): {
    empirical: string;
    normative: string;
    materiality: string;
  } {
    const empiricalDescriptions: Record<EmpiricalLevel, string> = {
      E0: 'Purely normative/subjective',
      E1: 'Observable but not measurable',
      E2: 'Measurable with estimation',
      E3: 'Precisely measurable',
      E4: 'Cryptographically verifiable',
    };

    const normativeDescriptions: Record<NormativeLevel, string> = {
      N0: 'No value judgment',
      N1: 'Operational preference',
      N2: 'Community policy',
      N3: 'Constitutional principle',
    };

    const materialityDescriptions: Record<MaterialityLevel, string> = {
      M0: 'Negligible impact',
      M1: 'Minor operational impact',
      M2: 'Significant resource impact',
      M3: 'Critical/irreversible impact',
    };

    return {
      empirical: empiricalDescriptions[classification.e],
      normative: normativeDescriptions[classification.n],
      materiality: materialityDescriptions[classification.m],
    };
  }

  private decodeClaim(record: HolochainRecord): DKGClaim {
    return (record as any).entry.Present.entry as DKGClaim;
  }

  private decodeDispute(record: HolochainRecord): Dispute {
    return (record as any).entry.Present.entry as Dispute;
  }
}
