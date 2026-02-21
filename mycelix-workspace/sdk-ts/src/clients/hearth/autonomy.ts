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
  AutonomyProfile,
  AutonomyRequest,
  GuardianApproval,
} from './types';
import type { ActionHash, AgentPubKey } from '../../generated/common';

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
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_profile', payload: input });
  }

  async getCurrentCapabilities(memberHash: AgentPubKey) {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_current_capabilities', payload: memberHash });
  }

  async checkCapability(input: CheckCapabilityInput): Promise<boolean> {
    return this.client.callZome<boolean>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'check_capability', payload: input });
  }

  async advanceTier(input: AdvanceTierInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'advance_tier', payload: input });
  }

  // ============================================================================
  // Capability Requests
  // ============================================================================

  async requestCapability(input: RequestCapabilityInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'request_capability', payload: input });
  }

  async approveCapability(input: ApproveCapabilityInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'approve_capability', payload: input });
  }

  async denyCapability(input: ApproveCapabilityInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'deny_capability', payload: input });
  }

  async getPendingRequests(hearthHash: ActionHash) {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_pending_requests', payload: hearthHash });
  }
}
