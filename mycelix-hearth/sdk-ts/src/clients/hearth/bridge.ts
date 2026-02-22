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
  ConsciousnessCredential,
  GateAuditInput,
} from './types';
import { HearthError, classifyError } from './errors';

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_bridge';

const BRIDGE_SIGNAL_TYPES: ReadonlySet<string> = new Set(['CrossZomeCallFailed']);

export type BridgeSignalHandler = (signal: CrossZomeCallFailedSignal) => void;

export class BridgeClient {
  private signalHandlers: Map<string, Set<BridgeSignalHandler>> = new Map();
  private listening = false;

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
  // Zome Calls — Intra-Cluster Dispatch
  // ============================================================================

  /** Dispatch a synchronous RPC call to a hearth domain zome. */
  async dispatchCall(input: DispatchInput): Promise<DispatchResult> {
    return this.callZome('dispatch_call', input);
  }

  /** Submit an audited async query to the hearth bridge. */
  async queryHearth(input: DispatchInput): Promise<HolochainRecord> {
    return this.callZome('query_hearth', input);
  }

  /** Resolve a pending query with a result. */
  async resolveQuery(input: ResolveQueryInput): Promise<HolochainRecord> {
    return this.callZome('resolve_query', input);
  }

  /** Broadcast an event to connected clients. */
  async broadcastEvent(input: DispatchInput): Promise<HolochainRecord> {
    return this.callZome('broadcast_event', input);
  }

  // ============================================================================
  // Zome Calls — Event Queries
  // ============================================================================

  /** Get events for a specific domain. */
  async getDomainEvents(domain: string): Promise<HolochainRecord[]> {
    return this.callZome('get_domain_events', domain);
  }

  /** Get all events across all domains. */
  async getAllEvents(): Promise<HolochainRecord[]> {
    return this.callZome('get_all_events', null);
  }

  /** Get events filtered by type. */
  async getEventsByType(query: EventTypeQuery): Promise<HolochainRecord[]> {
    return this.callZome('get_events_by_type', query);
  }

  /** Get the caller's pending queries. */
  async getMyQueries(): Promise<HolochainRecord[]> {
    return this.callZome('get_my_queries', null);
  }

  // ============================================================================
  // Zome Calls — Cross-Cluster Dispatch
  // ============================================================================

  /** Dispatch a cross-cluster call to the Personal cluster. */
  async dispatchPersonalCall(input: CrossClusterDispatchInput): Promise<DispatchResult> {
    return this.callZome('dispatch_personal_call', input);
  }

  /** Dispatch a cross-cluster call to the Identity cluster. */
  async dispatchIdentityCall(input: CrossClusterDispatchInput): Promise<DispatchResult> {
    return this.callZome('dispatch_identity_call', input);
  }

  /** Dispatch a cross-cluster call to the Commons cluster. */
  async dispatchCommonsCall(input: CrossClusterDispatchInput): Promise<DispatchResult> {
    return this.callZome('dispatch_commons_call', input);
  }

  /** Dispatch a cross-cluster call to the Civic cluster. */
  async dispatchCivicCall(input: CrossClusterDispatchInput): Promise<DispatchResult> {
    return this.callZome('dispatch_civic_call', input);
  }

  // ============================================================================
  // Zome Calls — Cross-Cluster Convenience
  // ============================================================================

  /** Verify a member's identity via the Identity cluster. */
  async verifyMemberIdentity(agent: AgentPubKey): Promise<DispatchResult> {
    return this.callZome('verify_member_identity', agent);
  }

  /** Escalate an emergency alert to the Civic cluster. */
  async escalateEmergency(alertData: string): Promise<DispatchResult> {
    return this.callZome('escalate_emergency', alertData);
  }

  /** Query a member's timebank balance from Commons. */
  async queryTimebankBalance(agent: AgentPubKey): Promise<DispatchResult> {
    return this.callZome('query_timebank_balance', agent);
  }

  // ============================================================================
  // Zome Calls — Lifecycle & Sync
  // ============================================================================

  /** Initiate a severance (member departure) process. */
  async initiateSeverance(input: SeveranceInput): Promise<HolochainRecord> {
    return this.callZome('initiate_severance', input);
  }

  /** Trigger a hearth sync to assemble a weekly digest. */
  async hearthSync(input: HearthSyncInput): Promise<HolochainRecord> {
    return this.callZome('hearth_sync', input);
  }

  /** Get weekly digests for a hearth. */
  async getWeeklyDigests(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.callZome('get_weekly_digests', hearthHash);
  }

  /** Check the health of the bridge zome. */
  async healthCheck(): Promise<BridgeHealth> {
    return this.callZome('health_check', null);
  }

  // ============================================================================
  // Zome Calls — Consciousness Credential
  // ============================================================================

  /** Get a consciousness credential for the specified DID. */
  async getConsciousnessCredential(did: string): Promise<ConsciousnessCredential> {
    return this.callZome('get_consciousness_credential', did);
  }

  /** Log a governance gate audit event. */
  async logGovernanceGate(input: GateAuditInput): Promise<void> {
    return this.callZome('log_governance_gate', input);
  }

  /**
   * Force-refresh the caller's consciousness credential.
   *
   * Used by `withGateRetry` as the retry callback: when a domain zome
   * rejects a call because the cached credential has expired, this
   * method asks the bridge to re-fetch from identity and cache a fresh one.
   */
  async refreshCredential(): Promise<void> {
    await this.callZome('get_consciousness_credential', null);
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
        if (signal.type !== 'app') return;
        const parsed = signal.value.payload as Record<string, unknown>;
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
