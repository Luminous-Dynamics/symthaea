// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Kinship Zome Client
 *
 * Core membership, relationships, and bond management for Hearth clusters.
 *
 * @module @mycelix/sdk/clients/hearth/kinship
 */

import type {
  CreateHearthInput,
  CreateBondInput,
  TendBondInput,
  InviteMemberInput,
  UpdateMemberRoleInput,
  GetBondHealthInput,
  WeeklyDigest,
  BondUpdate,
  MemberRole,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { Record } from '@holochain/client';

export interface KinshipClientConfig {
  roleName?: string;
  timeout?: number;
}

interface ZomeCallable {
  callZome<T>(params: { role_name: string; zome_name: string; fn_name: string; payload: unknown }): Promise<T>;
}

export class KinshipClient {
  private readonly zomeName = 'hearth_kinship';

  constructor(
    private readonly client: ZomeCallable,
    private readonly config: Required<Pick<KinshipClientConfig, 'roleName' | 'timeout'>>,
  ) {}

  // ============================================================================
  // Hearth Management
  // ============================================================================

  async createHearth(input: CreateHearthInput) {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_hearth', payload: input });
  }

  async getMyHearths() {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_my_hearths', payload: null });
  }

  // ============================================================================
  // Membership
  // ============================================================================

  async inviteMember(input: InviteMemberInput) {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'invite_member', payload: input });
  }

  async acceptInvitation(invitationHash: ActionHash) {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'accept_invitation', payload: invitationHash });
  }

  async declineInvitation(invitationHash: ActionHash) {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'decline_invitation', payload: invitationHash });
  }

  async leaveHearth(hearthHash: ActionHash) {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'leave_hearth', payload: hearthHash });
  }

  async updateMemberRole(input: UpdateMemberRoleInput) {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'update_member_role', payload: input });
  }

  async getHearthMembers(hearthHash: ActionHash) {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_hearth_members', payload: hearthHash });
  }

  async isGuardian(hearthHash: ActionHash): Promise<boolean> {
    return this.client.callZome<boolean>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'is_guardian', payload: hearthHash });
  }

  async getCallerVoteWeight(hearthHash: ActionHash): Promise<number> {
    return this.client.callZome<number>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_caller_vote_weight', payload: hearthHash });
  }

  async getCallerRole(hearthHash: ActionHash): Promise<MemberRole | null> {
    return this.client.callZome<MemberRole | null>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_caller_role', payload: hearthHash });
  }

  async getActiveMemberCount(hearthHash: ActionHash): Promise<number> {
    return this.client.callZome<number>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_active_member_count', payload: hearthHash });
  }

  // ============================================================================
  // Kinship Bonds
  // ============================================================================

  async createBond(input: CreateBondInput) {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_kinship_bond', payload: input });
  }

  async tendBond(input: TendBondInput) {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'tend_bond', payload: input });
  }

  async getBondHealth(input: GetBondHealthInput): Promise<number> {
    return this.client.callZome<number>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_bond_health', payload: input });
  }

  async getKinshipGraph(hearthHash: ActionHash) {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_kinship_graph', payload: hearthHash });
  }

  async getNeglectedBonds(hearthHash: ActionHash) {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_neglected_bonds', payload: hearthHash });
  }

  async getBondSnapshots(hearthHash: ActionHash) {
    return this.client.callZome<BondUpdate[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_bond_snapshots', payload: hearthHash });
  }

  // ============================================================================
  // Weekly Digests
  // ============================================================================

  async createWeeklyDigest(digest: WeeklyDigest) {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_weekly_digest', payload: digest });
  }

  async getWeeklyDigests(hearthHash: ActionHash) {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_weekly_digests', payload: hearthHash });
  }
}
