// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Investments Zome Client
 *
 * Handles energy project investments and dividends.
 *
 * @module @mycelix/sdk/integrations/energy/zomes/investments
 */

import { EnergySdkError } from '../types';

import type {
  Investment,
  InvestInput,
  Dividend,
  EnergyProject,
} from '../types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Configuration for the Investments client
 */
export interface InvestmentsClientConfig {
  roleId: string;
  zomeName: string;
}

const DEFAULT_CONFIG: InvestmentsClientConfig = {
  roleId: 'energy',
  zomeName: 'investments',
};

/**
 * Client for project investment operations
 *
 * @example
 * ```typescript
 * const investments = new InvestmentsClient(holochainClient);
 *
 * // Invest in a project
 * const investment = await investments.invest({
 *   project_id: 'solar-project-001',
 *   amount: 5000,
 *   currency: 'USD',
 * });
 *
 * // Get dividends
 * const dividends = await investments.getDividends(investment.id);
 * ```
 */
export class InvestmentsClient {
  private readonly config: InvestmentsClientConfig;

  constructor(
    private readonly client: AppClient,
    config: Partial<InvestmentsClientConfig> = {}
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
      throw new EnergySdkError(
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
      throw new EnergySdkError(
        'INVALID_INPUT',
        'Record does not contain an entry'
      );
    }
    return (record.entry as unknown as { Present: { entry: T } }).Present.entry;
  }

  // ============================================================================
  // Investment Operations
  // ============================================================================

  /**
   * Invest in a project
   *
   * @param input - Investment parameters
   * @returns The created investment
   */
  async invest(input: InvestInput): Promise<Investment> {
    const record = await this.call<HolochainRecord>('invest', {
      ...input,
      invested_at: Date.now() * 1000,
      status: 'Active',
    });
    return this.extractEntry<Investment>(record);
  }

  /**
   * Get an investment by ID
   *
   * @param investmentId - Investment identifier
   * @returns The investment or null if not found
   */
  async getInvestment(investmentId: string): Promise<Investment | null> {
    const record = await this.call<HolochainRecord | null>('get_investment', investmentId);
    if (!record) return null;
    return this.extractEntry<Investment>(record);
  }

  /**
   * Get investments by project
   *
   * @param projectId - Project identifier
   * @returns Array of investments
   */
  async getInvestmentsByProject(projectId: string): Promise<Investment[]> {
    const records = await this.call<HolochainRecord[]>('get_by_project', projectId);
    return records.map(r => this.extractEntry<Investment>(r));
  }

  /**
   * Get investments by investor
   *
   * @param investorDid - Investor's DID
   * @returns Array of investments
   */
  async getInvestmentsByInvestor(investorDid: string): Promise<Investment[]> {
    const records = await this.call<HolochainRecord[]>('get_by_investor', investorDid);
    return records.map(r => this.extractEntry<Investment>(r));
  }

  /**
   * Get active investments by investor
   *
   * @param investorDid - Investor's DID
   * @returns Array of active investments
   */
  async getActiveInvestments(investorDid: string): Promise<Investment[]> {
    const investments = await this.getInvestmentsByInvestor(investorDid);
    return investments.filter(i => i.status === 'Active');
  }

  /**
   * Exit an investment (sell or withdraw)
   *
   * @param investmentId - Investment identifier
   * @param reason - Exit reason
   * @returns The updated investment
   */
  async exitInvestment(investmentId: string, reason: string): Promise<Investment> {
    const record = await this.call<HolochainRecord>('exit_investment', {
      investment_id: investmentId,
      reason,
    });
    return this.extractEntry<Investment>(record);
  }

  // ============================================================================
  // Dividend Operations
  // ============================================================================

  /**
   * Get dividends for an investment
   *
   * @param investmentId - Investment identifier
   * @returns Array of dividends
   */
  async getDividends(investmentId: string): Promise<Dividend[]> {
    const records = await this.call<HolochainRecord[]>('get_dividends', investmentId);
    return records.map(r => this.extractEntry<Dividend>(r));
  }

  /**
   * Get all dividends for an investor
   *
   * @param investorDid - Investor's DID
   * @returns Array of dividends
   */
  async getDividendsByInvestor(investorDid: string): Promise<Dividend[]> {
    const records = await this.call<HolochainRecord[]>('get_dividends_by_investor', investorDid);
    return records.map(r => this.extractEntry<Dividend>(r));
  }

  /**
   * Get dividends for a project
   *
   * @param projectId - Project identifier
   * @returns Array of dividends
   */
  async getDividendsByProject(projectId: string): Promise<Dividend[]> {
    const records = await this.call<HolochainRecord[]>('get_dividends_by_project', projectId);
    return records.map(r => this.extractEntry<Dividend>(r));
  }

  /**
   * Distribute dividends for a project
   *
   * @param projectId - Project identifier
   * @param totalAmount - Total amount to distribute
   * @param currency - Currency code
   * @param periodStart - Period start timestamp
   * @param periodEnd - Period end timestamp
   * @param energyGeneratedKwh - Energy generated during period
   * @returns Array of distributed dividends
   */
  async distributeDividends(
    projectId: string,
    totalAmount: number,
    currency: string,
    periodStart: number,
    periodEnd: number,
    energyGeneratedKwh: number
  ): Promise<Dividend[]> {
    const records = await this.call<HolochainRecord[]>('distribute_dividends', {
      project_id: projectId,
      total_amount: totalAmount,
      currency,
      period_start: periodStart,
      period_end: periodEnd,
      energy_generated_kwh: energyGeneratedKwh,
    });
    return records.map(r => this.extractEntry<Dividend>(r));
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Quick investment with default currency
   *
   * @param projectId - Project identifier
   * @param amount - Investment amount
   * @param currency - Currency code (default USD)
   * @returns The created investment
   */
  async quickInvest(
    projectId: string,
    amount: number,
    currency: string = 'USD'
  ): Promise<Investment> {
    return this.invest({
      project_id: projectId,
      amount,
      currency,
    });
  }

  /**
   * Calculate ownership percentage for an investment amount
   *
   * @param amount - Investment amount
   * @param project - The project
   * @returns Ownership percentage
   */
  calculateOwnershipPercentage(amount: number, project: EnergyProject): number {
    if (!project.investment_goal || project.investment_goal <= 0) {
      return 0;
    }
    return (amount / project.investment_goal) * 100;
  }

  /**
   * Calculate expected annual dividend
   *
   * @param investment - The investment
   * @param annualRevenue - Project's annual revenue
   * @param distributionRate - Percentage of revenue distributed (0-1)
   * @returns Expected annual dividend
   */
  calculateExpectedDividend(
    investment: Investment,
    annualRevenue: number,
    distributionRate: number = 0.8
  ): number {
    return (investment.ownership_percentage / 100) * annualRevenue * distributionRate;
  }

  /**
   * Get total invested amount for an investor
   *
   * @param investorDid - Investor's DID
   * @returns Total invested amount
   */
  async getTotalInvested(investorDid: string): Promise<number> {
    const investments = await this.getActiveInvestments(investorDid);
    return investments.reduce((sum, i) => sum + i.amount, 0);
  }

  /**
   * Get total dividends received by an investor
   *
   * @param investorDid - Investor's DID
   * @returns Total dividends received
   */
  async getTotalDividendsReceived(investorDid: string): Promise<number> {
    const dividends = await this.getDividendsByInvestor(investorDid);
    return dividends.reduce((sum, d) => sum + d.amount, 0);
  }

  /**
   * Calculate return on investment
   *
   * @param investment - The investment
   * @param dividends - Dividends received
   * @returns ROI percentage
   */
  calculateROI(investment: Investment, dividends: Dividend[]): number {
    const totalDividends = dividends.reduce((sum, d) => sum + d.amount, 0);
    return (totalDividends / investment.amount) * 100;
  }

  /**
   * Get investment summary for an investor
   *
   * @param investorDid - Investor's DID
   * @returns Investment summary
   */
  async getInvestmentSummary(investorDid: string): Promise<{
    totalInvested: number;
    activeInvestments: number;
    totalDividendsReceived: number;
    projectsInvested: number;
    averageOwnership: number;
  }> {
    const investments = await this.getInvestmentsByInvestor(investorDid);
    const dividends = await this.getDividendsByInvestor(investorDid);

    const activeInvestments = investments.filter(i => i.status === 'Active');
    const totalInvested = activeInvestments.reduce((sum, i) => sum + i.amount, 0);
    const totalDividendsReceived = dividends.reduce((sum, d) => sum + d.amount, 0);
    const projectsInvested = new Set(activeInvestments.map(i => i.project_id)).size;
    const averageOwnership = activeInvestments.length > 0
      ? activeInvestments.reduce((sum, i) => sum + i.ownership_percentage, 0) / activeInvestments.length
      : 0;

    return {
      totalInvested,
      activeInvestments: activeInvestments.length,
      totalDividendsReceived,
      projectsInvested,
      averageOwnership,
    };
  }
}
