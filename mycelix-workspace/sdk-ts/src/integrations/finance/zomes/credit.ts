// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Credit Zome Client
 *
 * Handles MATL-integrated credit scoring and calculations.
 *
 * @module @mycelix/sdk/integrations/finance/zomes/credit
 */

import { FinanceSdkError } from '../types';

import type {
  CreditScore,
  CreditScoreResult,
  CreditCalculationInput,
  CreditTier,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Default configuration for the Credit client
 */
export interface CreditClientConfig {
  /** Role ID for the finance DNA */
  roleId: string;
  /** Zome name */
  zomeName: string;
}

const DEFAULT_CONFIG: CreditClientConfig = {
  roleId: 'finance',
  zomeName: 'finance',
};

/**
 * Client for credit scoring operations
 *
 * Credit scores are MATL-integrated, combining traditional credit factors
 * with trust scores from the MATL system.
 *
 * @example
 * ```typescript
 * const credit = new CreditClient(holochainClient);
 *
 * // Calculate credit score
 * const score = await credit.calculateCreditScore('did:mycelix:alice');
 *
 * // Get comprehensive credit analysis
 * const result = await credit.calculateComprehensiveCredit({
 *   did: 'did:mycelix:alice',
 *   on_time_payments: 24,
 *   late_payments: 1,
 *   missed_payments: 0,
 *   current_debt: 5000,
 *   total_credit_limit: 20000,
 *   credit_history_months: 36,
 *   loan_count: 2,
 *   credit_line_count: 1,
 *   recent_inquiries: 2,
 *   matl_score: 0.75,
 *   collateral_value: 10000,
 *   governance_participation: 0.6,
 *   annual_income: 60000,
 * });
 *
 * console.log(`Score: ${result.score}, Tier: ${result.tier}`);
 * console.log(`Max loan: ${result.max_loan_amount}`);
 * console.log(`Recommended rate: ${result.recommended_rate}%`);
 * ```
 */
export class CreditClient {
  private readonly config: CreditClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<CreditClientConfig> = {}
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
      throw new FinanceSdkError(
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
      throw new FinanceSdkError(
        'INVALID_INPUT',
        'Record does not contain an entry'
      );
    }
    return (record.entry as unknown as { Present: { entry: T } }).Present.entry;
  }

  // ============================================================================
  // Credit Score Operations
  // ============================================================================

  /**
   * Calculate credit score for a DID
   *
   * Queries MATL trust score via Bridge and combines with payment history,
   * utilization, credit age, and collateral factors.
   *
   * @param did - DID to calculate score for
   * @returns The calculated credit score
   */
  async calculateCreditScore(did: string): Promise<CreditScore> {
    return this.call<CreditScore>('calculate_credit_score', did);
  }

  /**
   * Get existing credit score
   *
   * @param did - DID to query
   * @returns Credit score or null if not found
   */
  async getCreditScore(did: string): Promise<CreditScore | null> {
    const record = await this.call<HolochainRecord | null>('get_credit_score', did);
    if (!record) return null;
    return this.extractEntry<CreditScore>(record);
  }

  /**
   * Calculate comprehensive credit score with full factor breakdown
   *
   * @param input - All credit factors
   * @returns Comprehensive credit result with tier and recommendations
   */
  async calculateComprehensiveCredit(
    input: CreditCalculationInput
  ): Promise<CreditScoreResult> {
    return this.call<CreditScoreResult>('calculate_comprehensive_credit', input);
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Get credit tier from score
   *
   * @param score - Numeric credit score (0-1000)
   * @returns Credit tier
   */
  getCreditTier(score: number): CreditTier {
    if (score >= 900) return 'Excellent';
    if (score >= 750) return 'Good';
    if (score >= 650) return 'Fair';
    if (score >= 550) return 'Poor';
    return 'VeryPoor';
  }

  /**
   * Get recommended interest rate based on score
   *
   * @param score - Numeric credit score (0-1000)
   * @returns Recommended annual interest rate percentage
   */
  getRecommendedRate(score: number): number {
    if (score >= 900) return 3.5;
    if (score >= 800) return 5.0;
    if (score >= 700) return 7.5;
    if (score >= 600) return 12.0;
    if (score >= 500) return 18.0;
    return 25.0;
  }

  /**
   * Calculate maximum loan amount based on score and income
   *
   * @param score - Numeric credit score (0-1000)
   * @param annualIncome - Annual income
   * @returns Maximum loan amount
   */
  calculateMaxLoan(score: number, annualIncome: number): number {
    let multiplier: number;
    if (score >= 900) multiplier = 5.0;
    else if (score >= 800) multiplier = 4.0;
    else if (score >= 700) multiplier = 3.0;
    else if (score >= 600) multiplier = 2.0;
    else if (score >= 500) multiplier = 1.0;
    else multiplier = 0.5;

    return Math.floor(annualIncome * multiplier);
  }

  /**
   * Get human-readable tier description
   *
   * @param tier - Credit tier
   * @returns Description of the tier
   */
  getTierDescription(tier: CreditTier): string {
    const descriptions: Record<CreditTier, string> = {
      Excellent: 'Exceptional credit. Qualifies for best rates and highest limits.',
      Good: 'Strong credit. Access to competitive rates and good limits.',
      Fair: 'Adequate credit. May face higher rates or require collateral.',
      Poor: 'Limited credit options. Likely needs collateral or co-signer.',
      VeryPoor: 'Credit building needed. Limited to secured products.',
    };
    return descriptions[tier];
  }

  /**
   * Calculate debt-to-income ratio
   *
   * @param monthlyDebt - Total monthly debt payments
   * @param monthlyIncome - Monthly income
   * @returns DTI ratio as percentage
   */
  calculateDTI(monthlyDebt: number, monthlyIncome: number): number {
    if (monthlyIncome === 0) return 100;
    return Math.round((monthlyDebt / monthlyIncome) * 100);
  }

  /**
   * Check if DTI is acceptable for lending
   *
   * @param dti - Debt-to-income ratio
   * @returns Whether DTI is acceptable
   */
  isAcceptableDTI(dti: number): boolean {
    return dti <= 43; // Standard mortgage guideline
  }

  /**
   * Estimate credit score impact of actions
   *
   * @param currentScore - Current credit score
   * @param action - Action type
   * @returns Estimated new score
   */
  estimateScoreImpact(
    currentScore: number,
    action: 'on_time_payment' | 'missed_payment' | 'new_credit_inquiry' | 'paid_off_loan'
  ): number {
    const impacts: Record<string, number> = {
      on_time_payment: 5,
      missed_payment: -30,
      new_credit_inquiry: -5,
      paid_off_loan: 15,
    };

    const impact = impacts[action] || 0;
    return Math.max(300, Math.min(1000, currentScore + impact));
  }
}
