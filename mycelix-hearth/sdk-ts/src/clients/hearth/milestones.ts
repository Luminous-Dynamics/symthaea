/**
 * Hearth Milestones SDK client.
 * Wraps zome calls to the hearth-milestones coordinator.
 */

import type { AppClient, Record as HolochainRecord, ActionHash, AgentPubKey } from '@holochain/client';
import type {
  RecordMilestoneInput,
  BeginTransitionInput,
} from './types';
import { HearthError, classifyError } from './errors';

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_milestones';

export class MilestonesClient {
  constructor(private readonly client: AppClient, private readonly roleName = ROLE_NAME) {}

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private async callZome<T>(fnName: string, payload: unknown): Promise<T> {
    try {
      return await this.client.callZome({
        role_name: this.roleName,
        zome_name: ZOME_NAME,
        fn_name: fnName,
        payload,
      });
    } catch (err) {
      throw new HearthError({
        code: classifyError(err),
        message: `${ZOME_NAME}.${fnName} failed: ${err}`,
        zome: ZOME_NAME,
        fnName,
        cause: err,
      });
    }
  }

  // ============================================================================
  // Zome Calls
  // ============================================================================

  /** Record a life milestone for a hearth member. */
  async recordMilestone(input: RecordMilestoneInput): Promise<HolochainRecord> {
    return this.callZome('record_milestone', input);
  }

  /** Begin a life transition (e.g., new school, moving). */
  async beginTransition(input: BeginTransitionInput): Promise<HolochainRecord> {
    return this.callZome('begin_transition', input);
  }

  /** Record progress on an active life transition. */
  async advanceTransition(transitionHash: ActionHash): Promise<HolochainRecord> {
    return this.callZome('advance_transition', transitionHash);
  }

  /** Mark a life transition as complete. */
  async completeTransition(transitionHash: ActionHash): Promise<HolochainRecord> {
    return this.callZome('complete_transition', transitionHash);
  }

  /** Get the full milestone timeline for a hearth. */
  async getFamilyTimeline(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_family_timeline', hearthHash);
  }

  /** Get all milestones for a specific member. */
  async getMemberMilestones(member: AgentPubKey): Promise<HolochainRecord[]> {
    return this.callZome('get_member_milestones', member);
  }

  /** Get all active life transitions for a hearth. */
  async getActiveTransitions(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_active_transitions', hearthHash);
  }
}
