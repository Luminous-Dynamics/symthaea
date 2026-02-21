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
  BondHealthResult,
  Hearth,
  HearthMembership,
  KinshipBond,
  HearthInvitation,
} from './types';
import type { ActionHash, AgentPubKey } from '../../generated/common';

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
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_hearth', payload: input });
  }

  async getMyHearths() {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_my_hearths', payload: null });
  }

  // ============================================================================
  // Membership
  // ============================================================================

  async inviteMember(input: InviteMemberInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'invite_member', payload: input });
  }

  async acceptInvitation(invitationHash: ActionHash) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'accept_invitation', payload: invitationHash });
  }

  async declineInvitation(invitationHash: ActionHash) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'decline_invitation', payload: invitationHash });
  }

  async leaveHearth(hearthHash: ActionHash) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'leave_hearth', payload: hearthHash });
  }

  async updateMemberRole(input: UpdateMemberRoleInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'update_member_role', payload: input });
  }

  async getHearthMembers(hearthHash: ActionHash) {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_hearth_members', payload: hearthHash });
  }

  // ============================================================================
  // Kinship Bonds
  // ============================================================================

  async createBond(input: CreateBondInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_kinship_bond', payload: input });
  }

  async tendBond(input: TendBondInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'tend_bond', payload: input });
  }

  async getBondHealth(bondHash: ActionHash): Promise<BondHealthResult> {
    return this.client.callZome<BondHealthResult>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_bond_health', payload: bondHash });
  }

  async getKinshipGraph(hearthHash: ActionHash) {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_kinship_graph', payload: hearthHash });
  }

  async getNeglectedBonds(hearthHash: ActionHash) {
    return this.client.callZome<BondHealthResult[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_neglected_bonds', payload: hearthHash });
  }
}
