// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Milestones Zome Client
 *
 * Life milestones, liminal transitions, and family timelines
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
import type { Record } from '@holochain/client';

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

  async recordMilestone(input: RecordMilestoneInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'record_milestone', payload: input });
  }

  async getFamilyTimeline(hearthHash: ActionHash): Promise<Record[]> {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_family_timeline', payload: hearthHash });
  }

  async getMemberMilestones(memberHash: ActionHash): Promise<Record[]> {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_member_milestones', payload: memberHash });
  }

  // ============================================================================
  // Life Transitions
  // ============================================================================

  async beginTransition(input: BeginTransitionInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'begin_transition', payload: input });
  }

  async advanceTransition(input: AdvanceTransitionInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'advance_transition', payload: input });
  }

  async completeTransition(transitionHash: ActionHash): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'complete_transition', payload: transitionHash });
  }

  async getActiveTransitions(hearthHash: ActionHash): Promise<Record[]> {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_active_transitions', payload: hearthHash });
  }
}
