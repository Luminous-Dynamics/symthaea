/**
 * Autonomy Zome Client
 *
 * Graduated autonomy profiles, capability requests, and guardian approvals
 * for Hearth clusters.
 *
 * @module @mycelix/sdk/clients/hearth/autonomy
 */

import type {
  CreateAutonomyProfileInput,
  RequestCapabilityInput,
  ApproveCapabilityInput,
  CheckCapabilityInput,
  AdvanceTierInput,
} from './types';
import type { ActionHash, AgentPubKey } from '../../generated/common';
import type { Record } from '@holochain/client';

export interface AutonomyClientConfig {
  roleName?: string;
  timeout?: number;
}

interface ZomeCallable {
  callZome<T>(params: { role_name: string; zome_name: string; fn_name: string; payload: unknown }): Promise<T>;
}

export class AutonomyClient {
  private readonly zomeName = 'hearth_autonomy';

  constructor(
    private readonly client: ZomeCallable,
    private readonly config: Required<Pick<AutonomyClientConfig, 'roleName' | 'timeout'>>,
  ) {}

  // ============================================================================
  // Profile Management
  // ============================================================================

  async createProfile(input: CreateAutonomyProfileInput) {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_autonomy_profile', payload: input });
  }

  async getAutonomyProfile(member: AgentPubKey): Promise<Record | null> {
    return this.client.callZome<Record | null>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_autonomy_profile', payload: member });
  }

  async checkCapability(input: CheckCapabilityInput): Promise<boolean> {
    return this.client.callZome<boolean>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'check_capability', payload: input });
  }

  async advanceTier(input: AdvanceTierInput) {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'advance_tier', payload: input });
  }

  // ============================================================================
  // Capability Requests
  // ============================================================================

  async requestCapability(input: RequestCapabilityInput) {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'request_capability', payload: input });
  }

  async approveCapability(input: ApproveCapabilityInput) {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'approve_capability', payload: input });
  }

  async denyCapability(input: ApproveCapabilityInput) {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'deny_capability', payload: input });
  }

  async getPendingRequests(hearthHash: ActionHash) {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_pending_requests', payload: hearthHash });
  }

  // ============================================================================
  // Tier Transitions
  // ============================================================================

  async progressTransition(transitionHash: ActionHash) {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'progress_transition', payload: transitionHash });
  }

  async getActiveTransitions(hearthHash: ActionHash) {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_active_transitions', payload: hearthHash });
  }
}
