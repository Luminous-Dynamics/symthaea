/**
 * Hearth Autonomy SDK client.
 * Wraps zome calls to the hearth-autonomy coordinator.
 */

import type { AppClient, Record as HolochainRecord, ActionHash, AgentPubKey } from '@holochain/client';
import type {
  CreateAutonomyProfileInput,
  RequestCapabilityInput,
  ApproveCapabilityInput,
  AdvanceTierInput,
  CheckCapabilityInput,
} from './types';

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_autonomy';

export class AutonomyClient {
  constructor(private readonly client: AppClient, private readonly roleName = ROLE_NAME) {}

  async createAutonomyProfile(input: CreateAutonomyProfileInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'create_autonomy_profile',
      payload: input,
    });
  }

  async requestCapability(input: RequestCapabilityInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'request_capability',
      payload: input,
    });
  }

  async approveCapability(input: ApproveCapabilityInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'approve_capability',
      payload: input,
    });
  }

  async denyCapability(input: ApproveCapabilityInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'deny_capability',
      payload: input,
    });
  }

  async advanceTier(input: AdvanceTierInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'advance_tier',
      payload: input,
    });
  }

  async progressTransition(transitionHash: ActionHash): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'progress_transition',
      payload: transitionHash,
    });
  }

  async getAutonomyProfile(member: AgentPubKey): Promise<HolochainRecord | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_autonomy_profile',
      payload: member,
    });
  }

  async checkCapability(input: CheckCapabilityInput): Promise<boolean> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'check_capability',
      payload: input,
    });
  }

  async getPendingRequests(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_pending_requests',
      payload: hearthHash,
    });
  }

  async getActiveTransitions(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_active_transitions',
      payload: hearthHash,
    });
  }
}
