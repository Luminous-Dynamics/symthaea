// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Membership Zome Client
 *
 * Handles member applications, approvals, rent-to-own, and payments.
 *
 * @module @mycelix/sdk/clients/housing/membership
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  Member,
  SubmitApplicationInput,
  ApproveMemberInput,
  RentToOwnAgreement,
  CreateRentToOwnInput,
  RecordRentPaymentInput,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface MembershipClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: MembershipClientConfig = {
  roleName: 'commons',
};

/**
 * Client for Housing Membership operations
 */
export class MembershipClient extends ZomeClient {
  protected readonly zomeName = 'housing_membership';

  constructor(client: AppClient, config: MembershipClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  async submitApplication(input: SubmitApplicationInput): Promise<Member> {
    const record = await this.callZomeOnce<HolochainRecord>('submit_application', {
      building_id: input.buildingId,
      name: input.name,
      membership_type: input.membershipType ?? 'Provisional',
      statement: input.statement,
    });
    return this.mapMember(record);
  }

  async approveMember(input: ApproveMemberInput): Promise<Member> {
    const record = await this.callZomeOnce<HolochainRecord>('approve_member', {
      application_id: input.applicationId,
      unit_id: input.unitId,
      equity_share: input.equityShare ?? 0,
      voting_weight: input.votingWeight ?? 1,
    });
    return this.mapMember(record);
  }

  async getMember(memberId: ActionHash): Promise<Member | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_member', memberId);
    if (!record) return null;
    return this.mapMember(record);
  }

  async getMembersForBuilding(buildingId: ActionHash): Promise<Member[]> {
    const records = await this.callZome<HolochainRecord[]>('get_members_for_building', buildingId);
    return records.map(r => this.mapMember(r));
  }

  async suspendMember(memberId: ActionHash, reason: string): Promise<Member> {
    const record = await this.callZomeOnce<HolochainRecord>('suspend_member', {
      member_id: memberId,
      reason,
    });
    return this.mapMember(record);
  }

  // ============================================================================
  // Rent-to-Own
  // ============================================================================

  async createRentToOwn(input: CreateRentToOwnInput): Promise<RentToOwnAgreement> {
    const record = await this.callZomeOnce<HolochainRecord>('create_rent_to_own', {
      member_id: input.memberId,
      unit_id: input.unitId,
      total_equity_target: input.totalEquityTarget,
      monthly_equity_contribution: input.monthlyEquityContribution,
      term_months: input.termMonths,
    });
    return this.mapRentToOwn(record);
  }

  async getRentToOwn(agreementId: ActionHash): Promise<RentToOwnAgreement | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_rent_to_own', agreementId);
    if (!record) return null;
    return this.mapRentToOwn(record);
  }

  async recordRentPayment(input: RecordRentPaymentInput): Promise<void> {
    await this.callZomeOnce<void>('record_rent_payment', {
      member_id: input.memberId,
      amount: input.amount,
      period: input.period,
      notes: input.notes,
    });
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private mapMember(record: HolochainRecord): Member {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      buildingId: entry.building_id,
      memberDid: entry.member_did,
      memberPubKey: entry.member_pub_key,
      name: entry.name,
      unitId: entry.unit_id,
      membershipType: entry.membership_type,
      equityShare: entry.equity_share,
      equityAccumulated: entry.equity_accumulated,
      votingWeight: entry.voting_weight,
      status: entry.status,
      joinedAt: entry.joined_at,
      updatedAt: entry.updated_at,
    };
  }

  private mapRentToOwn(record: HolochainRecord): RentToOwnAgreement {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      memberId: entry.member_id,
      unitId: entry.unit_id,
      totalEquityTarget: entry.total_equity_target,
      monthlyEquityContribution: entry.monthly_equity_contribution,
      equityAccumulated: entry.equity_accumulated,
      termMonths: entry.term_months,
      startDate: entry.start_date,
      projectedCompletionDate: entry.projected_completion_date,
      status: entry.status,
      createdAt: entry.created_at,
    };
  }
}
