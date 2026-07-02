// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Hearth Kinship SDK client.
 * Wraps zome calls to the hearth-kinship coordinator.
 */

import type { AppClient, Record as HolochainRecord, ActionHash } from '@holochain/client';
import type {
  CreateHearthInput,
  InviteMemberInput,
  AcceptInvitationInput,
  UpdateMemberRoleInput,
  CreateBondInput,
  TendBondInput,
  GetBondHealthInput,
  WeeklyDigest,
  BondUpdate,
  MemberRole,
  KinshipSignal,
  KinshipSignalType,
} from './types';
import { HearthError, classifyError } from './errors';
import { withGateRetry } from './consciousness-gate';

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_kinship';

const KINSHIP_SIGNAL_TYPES: ReadonlySet<string> = new Set([
  'MemberJoined',
  'MemberDeparted',
  'BondTended',
]);

export type KinshipSignalHandler = (signal: KinshipSignal) => void;

export class KinshipClient {
  private signalHandlers: Map<string, Set<KinshipSignalHandler>> = new Map();
  private listening = false;
  private refreshFn?: () => Promise<void>;

  constructor(
    private readonly client: AppClient,
    private readonly roleName = ROLE_NAME,
    refreshFn?: () => Promise<void>,
  ) {
    this.refreshFn = refreshFn;
  }

  // ============================================================================
  // Zome Calls
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

  /** Create a new hearth (family/household unit). */
  async createHearth(input: CreateHearthInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('create_hearth', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Invite an agent to join the hearth. */
  async inviteMember(input: InviteMemberInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('invite_member', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Accept a pending hearth invitation. */
  async acceptInvitation(input: AcceptInvitationInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('accept_invitation', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Decline a pending hearth invitation. */
  async declineInvitation(invitationHash: ActionHash): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('decline_invitation', invitationHash);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Leave a hearth voluntarily. */
  async leaveHearth(membershipHash: ActionHash): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('leave_hearth', membershipHash);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Update a member's role within the hearth. */
  async updateMemberRole(input: UpdateMemberRoleInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('update_member_role', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Create a kinship bond between the caller and another member. */
  async createKinshipBond(input: CreateBondInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('create_kinship_bond', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Tend (strengthen) an existing kinship bond. */
  async tendBond(input: TendBondInput): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('tend_bond', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Get the health status of a kinship bond. */
  async getBondHealth(input: GetBondHealthInput): Promise<number> {
    return this.callZome('get_bond_health', input);
  }

  /** Create a weekly digest entry for a hearth epoch. */
  async createWeeklyDigest(input: WeeklyDigest): Promise<HolochainRecord> {
    const call = () => this.callZome<HolochainRecord>('create_weekly_digest', input);
    return this.refreshFn ? withGateRetry(call, this.refreshFn) : call();
  }

  /** Get all weekly digests for a hearth. */
  async getWeeklyDigests(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_weekly_digests', hearthHash);
  }

  /** Check if the caller is a guardian of the given hearth. */
  async isGuardian(hearthHash: ActionHash): Promise<boolean> {
    return this.callZome('is_guardian', hearthHash);
  }

  /** Get the caller's vote weight in basis points for a hearth. */
  async getCallerVoteWeight(hearthHash: ActionHash): Promise<number> {
    return this.callZome('get_caller_vote_weight', hearthHash);
  }

  /** Get the caller's role in a hearth. */
  async getCallerRole(hearthHash: ActionHash): Promise<MemberRole | null> {
    return this.callZome('get_caller_role', hearthHash);
  }

  /** Get the count of active members in a hearth. */
  async getActiveMemberCount(hearthHash: ActionHash): Promise<number> {
    return this.callZome('get_active_member_count', hearthHash);
  }

  /** Get all membership records for a hearth. */
  async getHearthMembers(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_hearth_members', hearthHash);
  }

  /** Get all hearths the caller is a member of. */
  async getMyHearths(): Promise<HolochainRecord[]> {
    return this.callZome('get_my_hearths', null);
  }

  /** Get the full kinship bond graph for a hearth. */
  async getKinshipGraph(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_kinship_graph', hearthHash);
  }

  /** Get bonds that haven't been tended recently. */
  async getNeglectedBonds(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_neglected_bonds', hearthHash);
  }

  /** Get historical snapshots of bond strengths. */
  async getBondSnapshots(hearthHash: ActionHash): Promise<BondUpdate[]> {
    return this.callZome('get_bond_snapshots', hearthHash);
  }

  // ============================================================================
  // Signal Handling
  // ============================================================================

  /**
   * Subscribe to kinship signals. Returns an unsubscribe function.
   *
   * @param handler - Callback invoked for each matching signal
   * @param signalType - Optional filter: only receive signals of this type.
   *                     Pass '*' or omit to receive all kinship signals.
   *
   * @example
   * ```ts
   * const unsub = client.onSignal((signal) => {
   *   if (signal.type === 'MemberJoined') console.log('Welcome!', signal.agent);
   * });
   * // Later:
   * unsub();
   * ```
   */
  onSignal(
    handler: KinshipSignalHandler,
    signalType: KinshipSignalType | '*' = '*',
  ): () => void {
    this.ensureListening();

    const key = signalType;
    if (!this.signalHandlers.has(key)) {
      this.signalHandlers.set(key, new Set());
    }
    this.signalHandlers.get(key)!.add(handler);

    return () => {
      const handlers = this.signalHandlers.get(key);
      if (handlers) {
        handlers.delete(handler);
        if (handlers.size === 0) {
          this.signalHandlers.delete(key);
        }
      }
    };
  }

  private ensureListening(): void {
    if (this.listening) return;
    this.listening = true;

    this.client.on('signal', (signal) => {
      try {
        if (signal.type !== 'app') return;
        const parsed = signal.value.payload as Record<string, unknown>;
        if (!parsed || typeof parsed !== 'object') return;

        // Rust enums serialize as { "VariantName": { fields... } }
        const variantName = Object.keys(parsed)[0];
        if (!variantName || !KINSHIP_SIGNAL_TYPES.has(variantName)) return;

        const fields = parsed[variantName] as Record<string, unknown>;
        const typedSignal = { type: variantName, ...fields } as KinshipSignal;

        // Notify type-specific handlers
        const typeHandlers = this.signalHandlers.get(variantName);
        if (typeHandlers) {
          typeHandlers.forEach((h) => h(typedSignal));
        }

        // Notify wildcard handlers
        const wildcardHandlers = this.signalHandlers.get('*');
        if (wildcardHandlers) {
          wildcardHandlers.forEach((h) => h(typedSignal));
        }
      } catch {
        // Ignore non-kinship signals
      }
    });
  }
}
