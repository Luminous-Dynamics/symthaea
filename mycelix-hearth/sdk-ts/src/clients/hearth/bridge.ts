/**
 * Hearth Bridge SDK client.
 * Wraps zome calls to the hearth-bridge coordinator.
 */

import type { AppClient, Record as HolochainRecord, ActionHash, AgentPubKey } from '@holochain/client';
import type {
  DispatchInput,
  DispatchResult,
  CrossClusterDispatchInput,
  EventTypeQuery,
  ResolveQueryInput,
  SeveranceInput,
  HearthSyncInput,
  BridgeHealth,
  CrossZomeCallFailedSignal,
} from './types';

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_bridge';

const BRIDGE_SIGNAL_TYPES: ReadonlySet<string> = new Set(['CrossZomeCallFailed']);

export type BridgeSignalHandler = (signal: CrossZomeCallFailedSignal) => void;

export class BridgeClient {
  private signalHandlers: Map<string, Set<BridgeSignalHandler>> = new Map();
  private listening = false;

  constructor(private readonly client: AppClient, private readonly roleName = ROLE_NAME) {}

  // ============================================================================
  // Zome Calls — Intra-Cluster Dispatch
  // ============================================================================

  async dispatchCall(input: DispatchInput): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'dispatch_call',
      payload: input,
    });
  }

  async queryHearth(input: DispatchInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'query_hearth',
      payload: input,
    });
  }

  async resolveQuery(input: ResolveQueryInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'resolve_query',
      payload: input,
    });
  }

  async broadcastEvent(input: DispatchInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'broadcast_event',
      payload: input,
    });
  }

  // ============================================================================
  // Zome Calls — Event Queries
  // ============================================================================

  async getDomainEvents(domain: string): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_domain_events',
      payload: domain,
    });
  }

  async getAllEvents(): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_all_events',
      payload: null,
    });
  }

  async getEventsByType(query: EventTypeQuery): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_events_by_type',
      payload: query,
    });
  }

  async getMyQueries(): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_my_queries',
      payload: null,
    });
  }

  // ============================================================================
  // Zome Calls — Cross-Cluster Dispatch
  // ============================================================================

  async dispatchPersonalCall(input: CrossClusterDispatchInput): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'dispatch_personal_call',
      payload: input,
    });
  }

  async dispatchIdentityCall(input: CrossClusterDispatchInput): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'dispatch_identity_call',
      payload: input,
    });
  }

  async dispatchCommonsCall(input: CrossClusterDispatchInput): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'dispatch_commons_call',
      payload: input,
    });
  }

  async dispatchCivicCall(input: CrossClusterDispatchInput): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'dispatch_civic_call',
      payload: input,
    });
  }

  // ============================================================================
  // Zome Calls — Cross-Cluster Convenience
  // ============================================================================

  async verifyMemberIdentity(agent: AgentPubKey): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'verify_member_identity',
      payload: agent,
    });
  }

  async escalateEmergency(alertData: string): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'escalate_emergency',
      payload: alertData,
    });
  }

  async queryTimebankBalance(agent: AgentPubKey): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'query_timebank_balance',
      payload: agent,
    });
  }

  // ============================================================================
  // Zome Calls — Lifecycle & Sync
  // ============================================================================

  async initiateSeverance(input: SeveranceInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'initiate_severance',
      payload: input,
    });
  }

  async hearthSync(input: HearthSyncInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'hearth_sync',
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

  async healthCheck(): Promise<BridgeHealth> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'health_check',
      payload: null,
    });
  }

  // ============================================================================
  // Signal Handling
  // ============================================================================

  /**
   * Subscribe to bridge signals. Returns an unsubscribe function.
   *
   * @param handler - Callback invoked for each matching signal
   * @param signalType - Optional filter: only receive signals of this type.
   *                     Pass '*' or omit to receive all bridge signals.
   *
   * @example
   * ```ts
   * const unsub = client.onSignal((signal) => {
   *   console.log('Bridge call failed:', signal.zome, signal.function, signal.error);
   * });
   * // Later:
   * unsub();
   * ```
   */
  onSignal(
    handler: BridgeSignalHandler,
    signalType: 'CrossZomeCallFailed' | '*' = '*',
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
        if (!variantName || !BRIDGE_SIGNAL_TYPES.has(variantName)) return;

        const fields = parsed[variantName] as Record<string, unknown>;
        const typedSignal = { type: variantName, ...fields } as CrossZomeCallFailedSignal;

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
        // Ignore non-bridge signals
      }
    });
  }
}
