/**
 * Credit Scoring Zome Client
 *
 * Credit profile and collateral management for the Finance hApp.
 *
 * DEPRECATION NOTICE: This module is deprecated and scheduled for removal in v6.0.
 * The credit scoring approach conflicts with the Mycelix Constitution's vision.
 * Please use CGC, TEND, or HEARTH modules instead.
 *
 * @module @mycelix/sdk/clients/finance/credit-scoring
 * @deprecated Use CGC, TEND, or HEARTH modules for new development
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client.js';

import type {
  CreditProfile,
  CreateProfileInput,
  PaymentRecord,
  RecordPaymentInput,
  CollateralRecord,
  RegisterCollateralInput,
  LockCollateralInput,
  UpdateMatlInput,
  ScoreRangeInput,
} from './types.js';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

const FINANCE_ROLE = 'finance';
const ZOME_NAME = 'credit_scoring';

/**
 * Credit Scoring Client for managing credit profiles and collateral
 *
 * @deprecated This module is deprecated. Use CGC for recognition,
 * TEND for time exchange, or HEARTH for community resource pooling.
 *
 * @example
 * ```typescript
 * const creditScoring = new CreditScoringClient(client);
 *
 * // Create a credit profile (deprecated)
 * const profile = await creditScoring.createCreditProfile({
 *   did: 'did:mycelix:...',
 *   matl_score: 0.75,
 *   collateral_ratio: 0.5,
 * });
 * ```
 */
export class CreditScoringClient extends ZomeClient {
  protected readonly zomeName = ZOME_NAME;

  constructor(client: AppClient, config: Partial<ZomeClientConfig> = {}) {
    super(client, { roleName: FINANCE_ROLE, ...config });
  }

  // ===========================================================================
  // Credit Profiles
  // ===========================================================================

  /**
   * Create a new credit profile
   * @deprecated Use CGC for recognition-based systems
   */
  async createCreditProfile(input: CreateProfileInput): Promise<CreditProfile> {
    const record = await this.callZomeOnce<HolochainRecord>('create_credit_profile', input);
    return this.extractEntry<CreditProfile>(record);
  }

  /**
   * Get credit profile for a DID
   */
  async getCreditProfile(did: string): Promise<CreditProfile | null> {
    return this.callZomeOrNull<CreditProfile>('get_credit_profile', did);
  }

  /**
   * Update credit score based on payment history
   */
  async updateCreditScore(did: string): Promise<CreditProfile | null> {
    return this.callZomeOrNull<CreditProfile>('update_credit_score', did);
  }

  /**
   * Update MATL score (called from bridge when trust changes)
   */
  async updateMatlScore(input: UpdateMatlInput): Promise<CreditProfile | null> {
    return this.callZomeOrNull<CreditProfile>('update_matl_score', input);
  }

  /**
   * Get profiles within a score range (for offer matching)
   */
  async getProfilesByScoreRange(input: ScoreRangeInput): Promise<CreditProfile[]> {
    const records = await this.callZome<HolochainRecord[]>('get_profiles_by_score_range', input);
    return records.map((r) => this.extractEntry<CreditProfile>(r));
  }

  // ===========================================================================
  // Payment Records
  // ===========================================================================

  /**
   * Record a payment for credit history
   */
  async recordPayment(input: RecordPaymentInput): Promise<PaymentRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('record_payment', input);
    return this.extractEntry<PaymentRecord>(record);
  }

  /**
   * Get all payment records for a profile
   */
  async getPaymentRecords(did: string): Promise<PaymentRecord[]> {
    const records = await this.callZome<HolochainRecord[]>('get_payment_records', did);
    return records.map((r) => this.extractEntry<PaymentRecord>(r));
  }

  // ===========================================================================
  // Collateral Management
  // ===========================================================================

  /**
   * Register collateral
   */
  async registerCollateral(input: RegisterCollateralInput): Promise<CollateralRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('register_collateral', input);
    return this.extractEntry<CollateralRecord>(record);
  }

  /**
   * Get all collateral for a profile
   */
  async getCollateral(did: string): Promise<CollateralRecord[]> {
    const records = await this.callZome<HolochainRecord[]>('get_collateral', did);
    return records.map((r) => this.extractEntry<CollateralRecord>(r));
  }

  /**
   * Lock collateral for a loan
   */
  async lockCollateral(input: LockCollateralInput): Promise<CollateralRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('lock_collateral', input);
    return this.extractEntry<CollateralRecord>(record);
  }

  /**
   * Release collateral after loan is repaid
   */
  async releaseCollateral(collateralId: string): Promise<CollateralRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('release_collateral', collateralId);
    return this.extractEntry<CollateralRecord>(record);
  }
}
