// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Finances Zome Client
 *
 * Handles monthly charges, payments, and financial summaries.
 *
 * @module @mycelix/sdk/clients/housing/finances
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  MonthlyCharge,
  GenerateMonthlyChargesInput,
  PaymentRecord,
  RecordPaymentInput,
  FinancialSummary,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface FinancesClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: FinancesClientConfig = {
  roleName: 'commons',
};

/**
 * Client for Housing Finance operations
 */
export class FinancesClient extends ZomeClient {
  protected readonly zomeName = 'housing_finances';

  constructor(client: AppClient, config: FinancesClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  async generateMonthlyCharges(input: GenerateMonthlyChargesInput): Promise<MonthlyCharge[]> {
    const records = await this.callZomeOnce<HolochainRecord[]>('generate_monthly_charges', {
      building_id: input.buildingId,
      period: input.period,
      special_assessment: input.specialAssessment,
      special_assessment_reason: input.specialAssessmentReason,
    });
    return records.map(r => this.mapCharge(r));
  }

  async getChargesForMember(memberDid: string, period?: string): Promise<MonthlyCharge[]> {
    const records = await this.callZome<HolochainRecord[]>('get_charges_for_member', {
      member_did: memberDid,
      period: period ?? null,
    });
    return records.map(r => this.mapCharge(r));
  }

  async getChargesForBuilding(buildingId: ActionHash, period: string): Promise<MonthlyCharge[]> {
    const records = await this.callZome<HolochainRecord[]>('get_charges_for_building', {
      building_id: buildingId,
      period,
    });
    return records.map(r => this.mapCharge(r));
  }

  async recordPayment(input: RecordPaymentInput): Promise<PaymentRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('record_payment', {
      charge_id: input.chargeId,
      amount: input.amount,
      method: input.method,
      reference: input.reference,
    });
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      chargeId: entry.charge_id,
      memberDid: entry.member_did,
      amount: entry.amount,
      method: entry.method,
      reference: entry.reference,
      paidAt: entry.paid_at,
    };
  }

  async getFinancialSummary(buildingId: ActionHash, period: string): Promise<FinancialSummary> {
    const result = await this.callZome<any>('get_financial_summary', {
      building_id: buildingId,
      period,
    });
    return {
      buildingId: result.building_id,
      period: result.period,
      totalRevenue: result.total_revenue,
      totalExpenses: result.total_expenses,
      netIncome: result.net_income,
      outstandingCharges: result.outstanding_charges,
      reserveFund: result.reserve_fund,
      occupancyRate: result.occupancy_rate,
    };
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private mapCharge(record: HolochainRecord): MonthlyCharge {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      buildingId: entry.building_id,
      unitId: entry.unit_id,
      memberDid: entry.member_did,
      period: entry.period,
      baseRent: entry.base_rent,
      maintenanceFee: entry.maintenance_fee,
      utilities: entry.utilities,
      specialAssessment: entry.special_assessment,
      totalDue: entry.total_due,
      totalPaid: entry.total_paid,
      status: entry.status,
      dueDate: entry.due_date,
      createdAt: entry.created_at,
    };
  }
}
