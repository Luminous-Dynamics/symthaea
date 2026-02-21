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
import { HearthError, classifyError } from './errors';

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_autonomy';

export class AutonomyClient {
  constructor(private readonly client: AppClient, private readonly roleName = ROLE_NAME) {}

  // ============================================================================
  // Private helper
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

  /** Create an autonomy profile for a youth member. */
  async createAutonomyProfile(input: CreateAutonomyProfileInput): Promise<HolochainRecord> {
    return this.callZome('create_autonomy_profile', input);
  }

  /** Request a new capability (permission) for a youth. */
  async requestCapability(input: RequestCapabilityInput): Promise<HolochainRecord> {
    return this.callZome('request_capability', input);
  }

  /** Approve a pending capability request (guardian action). */
  async approveCapability(input: ApproveCapabilityInput): Promise<HolochainRecord> {
    return this.callZome('approve_capability', input);
  }

  /** Deny a pending capability request (guardian action). */
  async denyCapability(input: ApproveCapabilityInput): Promise<HolochainRecord> {
    return this.callZome('deny_capability', input);
  }

  /** Advance a youth to the next autonomy tier. */
  async advanceTier(input: AdvanceTierInput): Promise<HolochainRecord> {
    return this.callZome('advance_tier', input);
  }

  /** Record progress on an active tier transition. */
  async progressTransition(transitionHash: ActionHash): Promise<HolochainRecord> {
    return this.callZome('progress_transition', transitionHash);
  }

  /** Get the autonomy profile for an agent. Returns null if none exists. */
  async getAutonomyProfile(member: AgentPubKey): Promise<HolochainRecord | null> {
    return this.callZome('get_autonomy_profile', member);
  }

  /** Check if a youth has a specific capability. */
  async checkCapability(input: CheckCapabilityInput): Promise<boolean> {
    return this.callZome('check_capability', input);
  }

  /** Get all pending capability requests for a hearth. */
  async getPendingRequests(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_pending_requests', hearthHash);
  }

  /** Get all active tier transitions for a hearth. */
  async getActiveTransitions(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_active_transitions', hearthHash);
  }
}
