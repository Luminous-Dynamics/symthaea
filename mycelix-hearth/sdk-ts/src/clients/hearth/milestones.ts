/**
 * Hearth Milestones SDK client.
 * Wraps zome calls to the hearth-milestones coordinator.
 */

import type { AppClient, Record as HolochainRecord, ActionHash, AgentPubKey } from '@holochain/client';
import type {
  RecordMilestoneInput,
  BeginTransitionInput,
} from './types';

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_milestones';

export class MilestonesClient {
  constructor(private readonly client: AppClient, private readonly roleName = ROLE_NAME) {}

  // ============================================================================
  // Zome Calls
  // ============================================================================

  async recordMilestone(input: RecordMilestoneInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'record_milestone',
      payload: input,
    });
  }

  async beginTransition(input: BeginTransitionInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'begin_transition',
      payload: input,
    });
  }

  async advanceTransition(transitionHash: ActionHash): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'advance_transition',
      payload: transitionHash,
    });
  }

  async completeTransition(transitionHash: ActionHash): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'complete_transition',
      payload: transitionHash,
    });
  }

  async getFamilyTimeline(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_family_timeline',
      payload: hearthHash,
    });
  }

  async getMemberMilestones(member: AgentPubKey): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_member_milestones',
      payload: member,
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
