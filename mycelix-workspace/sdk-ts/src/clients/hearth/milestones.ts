/**
 * Milestones Zome Client
 *
 * Life milestones, transitions, family timelines, and birthday tracking
 * for Hearth clusters.
 *
 * @module @mycelix/sdk/clients/hearth/milestones
 */

import type {
  RecordMilestoneInput,
  BeginTransitionInput,
  AdvanceTransitionInput,
} from './types';
import type { ActionHash } from '../../generated/common';

export interface MilestonesClientConfig {
  roleName?: string;
  timeout?: number;
}

interface ZomeCallable {
  callZome<T>(params: { role_name: string; zome_name: string; fn_name: string; payload: unknown }): Promise<T>;
}

export class MilestonesClient {
  private readonly zomeName = 'hearth_milestones';

  constructor(
    private readonly client: ZomeCallable,
    private readonly config: Required<Pick<MilestonesClientConfig, 'roleName' | 'timeout'>>,
  ) {}

  // ============================================================================
  // Milestones
  // ============================================================================

  async recordMilestone(input: RecordMilestoneInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'record_milestone', payload: input });
  }

  async getFamilyTimeline(hearthHash: ActionHash) {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_family_timeline', payload: hearthHash });
  }

  async getUpcomingBirthdays(hearthHash: ActionHash) {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_upcoming_birthdays', payload: hearthHash });
  }

  // ============================================================================
  // Life Transitions
  // ============================================================================

  async beginTransition(input: BeginTransitionInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'begin_transition', payload: input });
  }

  async advanceTransition(input: AdvanceTransitionInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'advance_transition', payload: input });
  }

  async completeTransition(transitionHash: ActionHash) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'complete_transition', payload: transitionHash });
  }

  async getActiveTransitions(hearthHash: ActionHash) {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_active_transitions', payload: hearthHash });
  }
}
