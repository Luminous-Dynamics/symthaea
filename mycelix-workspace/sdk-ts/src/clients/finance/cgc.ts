/**
 * CGC (Civic Gifting Credits) Zome Client
 *
 * Recognition-based social support system implementing Commons Charter Article I.
 *
 * Key Principles:
 * - CGCs are for RECOGNITION, not accumulation
 * - Monthly allocation of 10 CGC per verified member (non-cumulative)
 * - You cannot "spend" recognition you've received
 * - High-activity flagging for audit support
 *
 * @module @mycelix/sdk/clients/finance/cgc
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client.js';

import type {
  CgcAllocationStatus,
  GiftCgcInput,
  GiftRecord,
  RecognitionSummary,
  CulturalAlias,
  RegisterAliasInput,
} from './types.js';
import type { AppClient } from '@holochain/client';

const FINANCE_ROLE = 'finance';
const ZOME_NAME = 'cgc';

/**
 * CGC Client for Civic Gifting Credits (recognition system)
 *
 * CGC represents recognition, not value transfer. Every verified member
 * receives 10 CGC monthly to gift others for their contributions.
 *
 * @example
 * ```typescript
 * const cgc = new CgcClient(client);
 *
 * // Get your current allocation
 * const allocation = await cgc.getOrCreateAllocation('did:mycelix:alice');
 * console.log(`You have ${allocation.remaining} CGC to give this month`);
 *
 * // Gift CGC to recognize someone's contribution
 * await cgc.giftCgc({
 *   receiver_did: 'did:mycelix:bob',
 *   amount: 3,
 *   gratitude_message: 'Thank you for your help with documentation!',
 *   contribution_type: 'Documentation',
 * });
 * ```
 */
export class CgcClient extends ZomeClient {
  protected readonly zomeName = ZOME_NAME;

  constructor(client: AppClient, config: Partial<ZomeClientConfig> = {}) {
    super(client, { roleName: FINANCE_ROLE, ...config });
  }

  // ===========================================================================
  // Allocations
  // ===========================================================================

  /**
   * Get or create CGC allocation for current cycle
   *
   * Every verified member receives 10 CGC per month (non-cumulative).
   * This function ensures an allocation exists for the current cycle.
   */
  async getOrCreateAllocation(memberDid: string): Promise<CgcAllocationStatus> {
    return this.callZome<CgcAllocationStatus>('get_or_create_allocation', memberDid);
  }

  // ===========================================================================
  // Gifting
  // ===========================================================================

  /**
   * Gift CGC to another member
   *
   * This is the core action: giving recognition to someone for their contribution.
   * The gift comes from your monthly allocation, reducing your "remaining" balance.
   * The receiver gains RECOGNITION (tracked separately), not spendable credits.
   *
   * @throws If giver has insufficient remaining CGC
   * @throws If attempting to gift to yourself
   */
  async giftCgc(input: GiftCgcInput): Promise<GiftRecord> {
    return this.callZomeOnce<GiftRecord>('gift_cgc', input);
  }

  /**
   * Get gifts given by a member in current cycle
   */
  async getGiftsGiven(memberDid: string): Promise<GiftRecord[]> {
    return this.callZome<GiftRecord[]>('get_gifts_given', memberDid);
  }

  /**
   * Get gifts received by a member in current cycle
   */
  async getGiftsReceived(memberDid: string): Promise<GiftRecord[]> {
    return this.callZome<GiftRecord[]>('get_gifts_received', memberDid);
  }

  // ===========================================================================
  // Recognition
  // ===========================================================================

  /**
   * Get recognition summary for a member
   *
   * Shows total received, unique givers, breakdown by contribution type,
   * and whether the member is flagged for high activity.
   */
  async getRecognitionSummary(memberDid: string): Promise<RecognitionSummary> {
    return this.callZome<RecognitionSummary>('get_recognition_summary', memberDid);
  }

  /**
   * Get high-activity report for a cycle (for Audit Guild)
   *
   * Returns members exceeding thresholds per Article I, Section 2.b.
   */
  async getHighActivityReport(cycleId: string): Promise<RecognitionSummary[]> {
    return this.callZome<RecognitionSummary[]>('get_high_activity_report', cycleId);
  }

  /**
   * Get CGC recognition input for reputation calculation
   *
   * Returns normalized value (0-1) with max 10% weight per Article I.4.
   */
  async getCgcReputationInput(memberDid: string): Promise<number> {
    return this.callZome<number>('get_cgc_reputation_input', memberDid);
  }

  // ===========================================================================
  // Cultural Naming
  // ===========================================================================

  /**
   * Register a cultural alias for a DAO (Article I, Section 5)
   *
   * DAOs can register culturally meaningful names for their CGC system.
   */
  async registerCulturalAlias(input: RegisterAliasInput): Promise<CulturalAlias> {
    return this.callZomeOnce<CulturalAlias>('register_cultural_alias', input);
  }

  /**
   * Get all cultural aliases for a DAO
   */
  async getDaoAliases(daoDid: string): Promise<CulturalAlias[]> {
    return this.callZome<CulturalAlias[]>('get_dao_aliases', daoDid);
  }
}
