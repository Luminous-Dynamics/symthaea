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

  constructor(private readonly client: AppClient, private readonly roleName = ROLE_NAME) {}

  // ============================================================================
  // Zome Calls
  // ============================================================================

  async createHearth(input: CreateHearthInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'create_hearth',
      payload: input,
    });
  }

  async inviteMember(input: InviteMemberInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'invite_member',
      payload: input,
    });
  }

  async acceptInvitation(input: AcceptInvitationInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'accept_invitation',
      payload: input,
    });
  }

  async declineInvitation(invitationHash: ActionHash): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'decline_invitation',
      payload: invitationHash,
    });
  }

  async leaveHearth(membershipHash: ActionHash): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'leave_hearth',
      payload: membershipHash,
    });
  }

  async updateMemberRole(input: UpdateMemberRoleInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'update_member_role',
      payload: input,
    });
  }

  async createKinshipBond(input: CreateBondInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'create_kinship_bond',
      payload: input,
    });
  }

  async tendBond(input: TendBondInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'tend_bond',
      payload: input,
    });
  }

  async getBondHealth(input: GetBondHealthInput): Promise<number> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_bond_health',
      payload: input,
    });
  }

  async createWeeklyDigest(input: WeeklyDigest): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'create_weekly_digest',
      payload: input,
    });
  }

  async getWeeklyDigests(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_weekly_digests',
      payload: hearthHash,
    });
  }

  async isGuardian(hearthHash: ActionHash): Promise<boolean> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'is_guardian',
      payload: hearthHash,
    });
  }

  async getCallerVoteWeight(hearthHash: ActionHash): Promise<number> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_caller_vote_weight',
      payload: hearthHash,
    });
  }

  async getCallerRole(hearthHash: ActionHash): Promise<MemberRole | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_caller_role',
      payload: hearthHash,
    });
  }

  async getActiveMemberCount(hearthHash: ActionHash): Promise<number> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_active_member_count',
      payload: hearthHash,
    });
  }

  async getHearthMembers(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_hearth_members',
      payload: hearthHash,
    });
  }

  async getMyHearths(): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_my_hearths',
      payload: null,
    });
  }

  async getKinshipGraph(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_kinship_graph',
      payload: hearthHash,
    });
  }

  async getNeglectedBonds(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_neglected_bonds',
      payload: hearthHash,
    });
  }

  async getBondSnapshots(hearthHash: ActionHash): Promise<BondUpdate[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_bond_snapshots',
      payload: hearthHash,
    });
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
        const parsed = signal.payload as Record<string, unknown>;
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
