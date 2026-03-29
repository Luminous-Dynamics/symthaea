// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Lending Zome Client
 *
 * P2P lending functionality for the Finance hApp.
 *
 * DEPRECATION NOTICE: This module is deprecated and scheduled for removal in v6.0.
 * Traditional lending with interest-bearing loans conflicts with the Mycelix
 * Constitution's vision. Please use TEND, HEARTH, or CGC modules instead.
 *
 * @module @mycelix/sdk/clients/finance/lending
 * @deprecated Use TEND, HEARTH, or CGC modules for new development
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client.js';

import type {
  Loan,
  LoanOffer,
  LoanStatus,
  RequestLoanInput,
  CreateOfferInput,
  FundLoanInput,
  PaymentSchedule,
  CreateScheduleInput,
  MatchOffersInput,
  CreditAssessmentResult,
  PrivacyCreditProof,
  ProofType,
} from './types.js';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

const FINANCE_ROLE = 'finance';
const ZOME_NAME = 'lending';

/**
 * Lending Client for P2P loan management
 *
 * @deprecated This module is deprecated. Use TEND for time exchange,
 * HEARTH for commons pooling, or CGC for civic gifting credits.
 *
 * @example
 * ```typescript
 * const lending = new LendingClient(client);
 *
 * // Request a loan (deprecated)
 * const loan = await lending.requestLoan({
 *   borrower_did: 'did:mycelix:...',
 *   amount: 1000,
 *   currency: 'MYC',
 *   term_days: 30,
 *   collateral_ids: [],
 * });
 * ```
 */
export class LendingClient extends ZomeClient {
  protected readonly zomeName = ZOME_NAME;

  constructor(client: AppClient, config: Partial<ZomeClientConfig> = {}) {
    super(client, { roleName: FINANCE_ROLE, ...config });
  }

  // ===========================================================================
  // Loan Management
  // ===========================================================================

  /**
   * Request a new loan
   * @deprecated Use TEND for interest-free time exchange
   */
  async requestLoan(input: RequestLoanInput): Promise<Loan> {
    const record = await this.callZomeOnce<HolochainRecord>('request_loan', input);
    return this.extractEntry<Loan>(record);
  }

  /**
   * Get a loan by ID
   */
  async getLoan(loanId: string): Promise<Loan | null> {
    return this.callZomeOrNull<Loan>('get_loan', loanId);
  }

  /**
   * Get all loans for a borrower
   */
  async getBorrowerLoans(borrowerDid: string): Promise<Loan[]> {
    const records = await this.callZome<HolochainRecord[]>('get_borrower_loans', borrowerDid);
    return records.map((r) => this.extractEntry<Loan>(r));
  }

  /**
   * Get all loans for a lender
   */
  async getLenderLoans(lenderDid: string): Promise<Loan[]> {
    const records = await this.callZome<HolochainRecord[]>('get_lender_loans', lenderDid);
    return records.map((r) => this.extractEntry<Loan>(r));
  }

  /**
   * Get loans by status
   */
  async getLoansByStatus(status: LoanStatus): Promise<Loan[]> {
    const records = await this.callZome<HolochainRecord[]>('get_loans_by_status', status);
    return records.map((r) => this.extractEntry<Loan>(r));
  }

  /**
   * Fund a loan request
   */
  async fundLoan(input: FundLoanInput): Promise<Loan> {
    const record = await this.callZomeOnce<HolochainRecord>('fund_loan', input);
    return this.extractEntry<Loan>(record);
  }

  /**
   * Repay a loan in full
   */
  async repayLoan(loanId: string): Promise<Loan> {
    const record = await this.callZomeOnce<HolochainRecord>('repay_loan', loanId);
    return this.extractEntry<Loan>(record);
  }

  /**
   * Mark a loan as defaulted (lender only)
   */
  async defaultLoan(loanId: string): Promise<Loan> {
    const record = await this.callZomeOnce<HolochainRecord>('default_loan', loanId);
    return this.extractEntry<Loan>(record);
  }

  /**
   * Cancel a loan request (borrower only, while still in Requested status)
   */
  async cancelLoan(loanId: string): Promise<Loan> {
    const record = await this.callZomeOnce<HolochainRecord>('cancel_loan', loanId);
    return this.extractEntry<Loan>(record);
  }

  // ===========================================================================
  // Loan Offers
  // ===========================================================================

  /**
   * Create a loan offer
   * @deprecated Use HEARTH for community resource pooling
   */
  async createLoanOffer(input: CreateOfferInput): Promise<LoanOffer> {
    const record = await this.callZomeOnce<HolochainRecord>('create_loan_offer', input);
    return this.extractEntry<LoanOffer>(record);
  }

  /**
   * Get all active loan offers
   */
  async getActiveOffers(): Promise<LoanOffer[]> {
    const records = await this.callZome<HolochainRecord[]>('get_active_offers', null);
    return records.map((r) => this.extractEntry<LoanOffer>(r));
  }

  /**
   * Deactivate a loan offer
   */
  async deactivateOffer(offerId: string): Promise<LoanOffer> {
    const record = await this.callZomeOnce<HolochainRecord>('deactivate_offer', offerId);
    return this.extractEntry<LoanOffer>(record);
  }

  /**
   * Match offers for a borrower based on their profile
   */
  async matchOffersForBorrower(input: MatchOffersInput): Promise<LoanOffer[]> {
    const records = await this.callZome<HolochainRecord[]>('match_offers_for_borrower', input);
    return records.map((r) => this.extractEntry<LoanOffer>(r));
  }

  // ===========================================================================
  // Payment Schedules
  // ===========================================================================

  /**
   * Create a payment schedule for a funded loan
   */
  async createPaymentSchedule(input: CreateScheduleInput): Promise<PaymentSchedule> {
    const record = await this.callZomeOnce<HolochainRecord>('create_payment_schedule', input);
    return this.extractEntry<PaymentSchedule>(record);
  }

  /**
   * Get payment schedule for a loan
   */
  async getPaymentSchedule(loanId: string): Promise<PaymentSchedule | null> {
    return this.callZomeOrNull<PaymentSchedule>('get_payment_schedule', loanId);
  }

  // ===========================================================================
  // Ethical Credit Assessment
  // ===========================================================================

  /**
   * Calculate ethical credit assessment for a borrower
   *
   * This assessment uses ONLY lending-domain data and includes:
   * - Temporal decay (older events matter less)
   * - Floor guarantee (no one is completely excluded)
   * - Full transparency report
   */
  async calculateCreditAssessment(
    borrowerDid: string,
    optInAdditionalData?: boolean
  ): Promise<CreditAssessmentResult> {
    return this.callZome<CreditAssessmentResult>('calculate_credit_assessment', {
      borrower_did: borrowerDid,
      opt_in_additional_data: optInAdditionalData,
    });
  }

  /**
   * Get adjusted interest rate based on credit assessment
   */
  async getAdjustedInterestRate(
    borrowerDid: string,
    baseRate: number
  ): Promise<{
    borrower_did: string;
    base_rate: number;
    credit_score: number;
    rate_adjustment: number;
    adjusted_rate: number;
    floor_applied: boolean;
  }> {
    return this.callZome('get_adjusted_interest_rate', {
      borrower_did: borrowerDid,
      base_rate: baseRate,
    });
  }

  // ===========================================================================
  // Privacy-Preserving Credit Proofs
  // ===========================================================================

  /**
   * Generate a privacy-preserving threshold proof
   *
   * Allows proving creditworthiness without revealing exact score.
   */
  async generateThresholdProof(
    borrowerDid: string,
    threshold: number,
    proofType: ProofType = { MeetsMinimum: null }
  ): Promise<PrivacyCreditProof> {
    return this.callZome<PrivacyCreditProof>('generate_threshold_proof', {
      borrower_did: borrowerDid,
      threshold,
      proof_type: proofType,
    });
  }

  /**
   * Verify a threshold proof (for lenders)
   */
  async verifyThresholdProof(proof: PrivacyCreditProof): Promise<{
    proof_id: string;
    is_valid: boolean;
    reason: string;
    verified_at: number;
  }> {
    return this.callZome('verify_threshold_proof', proof);
  }

  /**
   * Request a loan using only a privacy proof
   */
  async requestLoanWithProof(input: {
    borrower_did: string;
    amount: number;
    currency: string;
    term_days: number;
    collateral_ids: string[];
    credit_proof: PrivacyCreditProof;
  }): Promise<Loan> {
    const record = await this.callZomeOnce<HolochainRecord>('request_loan_with_proof', input);
    return this.extractEntry<Loan>(record);
  }

  // ===========================================================================
  // Appeals
  // ===========================================================================

  /**
   * File a credit appeal
   */
  async fileCreditAppeal(input: {
    borrower_did: string;
    disputed_component: 'payment_history' | 'collateral_ratio' | 'account_age';
    reason: string;
    evidence?: string;
  }): Promise<{
    id: string;
    borrower_did: string;
    disputed_component: string;
    reason: string;
    status: string;
    filed_at: number;
  }> {
    return this.callZomeOnce('file_credit_appeal', input);
  }

  /**
   * Get appeals for a borrower
   */
  async getMyAppeals(borrowerDid: string): Promise<
    Array<{
      id: string;
      borrower_did: string;
      disputed_component: string;
      reason: string;
      status: string;
    }>
  > {
    return this.callZome('get_my_appeals', borrowerDid);
  }

  /**
   * Request a fresh start (credit history reset)
   *
   * Note: Has a 90-day waiting period and erases ALL history.
   */
  async requestFreshStart(borrowerDid: string): Promise<{
    borrower_did: string;
    requested_at: number;
    effective_at: number;
    acknowledged_consequences: boolean;
  }> {
    return this.callZomeOnce('request_fresh_start', borrowerDid);
  }
}
