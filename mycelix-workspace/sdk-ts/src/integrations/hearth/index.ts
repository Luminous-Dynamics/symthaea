/**
 * @mycelix/sdk Hearth Cluster Integration
 *
 * Cluster-level client for the mycelix-hearth DNA which unifies
 * kinship, gratitude, stories, care, autonomy, emergency, decisions,
 * resources, milestones, and rhythms into a single Holochain DNA
 * with cross-domain dispatch.
 *
 * ## Architecture
 *
 * All 10 domains share one DNA role (`hearth`) and communicate via
 * `hearth_bridge` — a coordinator zome that dispatches calls between
 * domain zomes using `call(CallTargetCell::Local, ...)`.
 *
 * ## Naming Note
 *
 * "HEARTH" also exists in the finance hApp as a commons pool governance
 * concept. The finance HEARTH operates against the `finance` DNA role;
 * the family HEARTH is its own cluster DNA at `hearth` role.
 *
 * ## Usage
 *
 * ```typescript
 * import { HearthBridgeClient, createHearthBridgeClient } from '@mycelix/sdk/integrations/hearth';
 *
 * const bridge = createHearthBridgeClient(appClient);
 *
 * // Cross-domain dispatch: call any zome function by name
 * const result = await bridge.dispatch('hearth_kinship', 'get_bonds', payload);
 *
 * // Audited query with auto-dispatch
 * const query = await bridge.query({
 *   domain: 'kinship',
 *   query_type: 'get_family_bonds',
 *   params: JSON.stringify({ hearth_id: '...' }),
 * });
 *
 * // Event broadcasting
 * await bridge.broadcastEvent({
 *   domain: 'gratitude',
 *   event_type: 'gratitude_expressed',
 *   payload: JSON.stringify({ ... }),
 * });
 *
 * // Health check across all 10 domains
 * const health = await bridge.healthCheck();
 * ```
 *
 * @packageDocumentation
 * @module integrations/hearth
 */

// ============================================================================
// Types
// ============================================================================

/** Input for cross-domain dispatch via the bridge */
export interface DispatchInput {
  /** Target zome name (e.g., "hearth_kinship", "hearth_care") */
  zome: string;
  /** Target function name */
  fn_name: string;
  /** MessagePack-encoded payload (use @msgpack/msgpack to encode) */
  payload: Uint8Array;
}

/** Result of a dispatched cross-domain call */
export interface DispatchResult {
  success: boolean;
  response?: Uint8Array;
  error?: string;
}

/** Input for an audited cross-domain query */
export interface HearthQueryInput {
  domain: 'kinship' | 'gratitude' | 'stories' | 'care' | 'autonomy' | 'emergency' | 'decisions' | 'resources' | 'milestones' | 'rhythms';
  query_type: string;
  params: string;
}

/** Input for broadcasting a cross-domain event */
export interface HearthEventInput {
  domain: 'kinship' | 'gratitude' | 'stories' | 'care' | 'autonomy' | 'emergency' | 'decisions' | 'resources' | 'milestones' | 'rhythms';
  event_type: string;
  payload: string;
  related_hashes?: string[];
}

/** Bridge health status */
export interface BridgeHealth {
  healthy: boolean;
  agent: string;
  total_events: number;
  total_queries: number;
  domains: string[];
}

/** Events by type query */
export interface EventTypeQuery {
  domain: string;
  event_type: string;
}

/** Input for cross-cluster dispatch to another DNA */
export interface CrossClusterDispatchInput {
  /** hApp role name of the target DNA */
  role: string;
  /** Target zome in the remote DNA */
  zome: string;
  /** Target function name */
  fn_name: string;
  /** MessagePack-encoded payload */
  payload: Uint8Array;
}

/** Input for verifying a member's identity via the identity DNA */
export interface VerifyMemberIdentityInput {
  member_did: string;
  hearth_id: string;
}

/** Result of a member identity verification */
export interface VerifyMemberIdentityResult {
  verified: boolean;
  identity_found: boolean;
  recommendation: string;
  error?: string;
}

/** Input for querying care availability via the commons care domain */
export interface QueryCareAvailabilityInput {
  skill_needed: string;
  hearth_id: string;
  location?: string;
}

/** Result of a care availability query */
export interface CareAvailabilityResult {
  available_count: number;
  recommendation: string;
  error?: string;
}

/** Input for escalating a hearth emergency to the civic emergency system */
export interface EscalateEmergencyInput {
  hearth_id: string;
  emergency_type: string;
  severity: string;
  description: string;
  location?: string;
}

/** Result of an emergency escalation */
export interface EscalateEmergencyResult {
  escalated: boolean;
  incident_id?: string;
  recommendation: string;
  error?: string;
}

/** Input for syncing hearth state across clusters */
export interface HearthSyncInput {
  hearth_id: string;
  sync_domains: string[];
}

/** Result of a hearth sync operation */
export interface HearthSyncResult {
  synced: boolean;
  domains_synced: string[];
  error?: string;
}

/** Input for the H3 severance protocol */
export interface SeveranceInput {
  hearth_hash: Uint8Array;
  member_hash: Uint8Array;
  export_milestones: boolean;
  export_care_history: boolean;
  export_bond_snapshot: boolean;
  new_role: string;
}

/** Result of a severance operation */
export interface SeveranceResult {
  success: boolean;
  exported_data?: Uint8Array;
  new_role_assigned: boolean;
  error?: string;
}

/** Input for querying the bridge audit trail */
export interface AuditTrailQuery {
  /** Start of time range (inclusive), microseconds since epoch */
  from_us: number;
  /** End of time range (inclusive), microseconds since epoch */
  to_us: number;
  /** Optional domain filter */
  domain?: string;
  /** Optional event type filter */
  event_type?: string;
}

/** A single audit trail entry */
export interface AuditTrailEntry {
  domain: string;
  event_type: string;
  source_agent: string;
  payload_preview: string;
  created_at_us: number;
  action_hash: Uint8Array;
}

/** Result of an audit trail query */
export interface AuditTrailResult {
  entries: AuditTrailEntry[];
  total_matched: number;
  query_from_us: number;
  query_to_us: number;
}

/** Holochain ZomeCallable interface (minimal) */
interface ZomeCallable {
  callZome<T>(params: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }): Promise<T>;
}

// ============================================================================
// Constants
// ============================================================================

/** All domain names available in the hearth cluster */
export const HEARTH_DOMAINS = [
  'kinship', 'gratitude', 'stories', 'care', 'autonomy',
  'emergency', 'decisions', 'resources', 'milestones', 'rhythms',
] as const;

/** All domain zomes in the hearth DNA */
export const HEARTH_ZOMES = [
  'hearth_kinship', 'hearth_gratitude', 'hearth_stories', 'hearth_care', 'hearth_autonomy',
  'hearth_emergency', 'hearth_decisions', 'hearth_resources', 'hearth_milestones', 'hearth_rhythms',
] as const;

/** hApp role for the hearth DNA */
const HEARTH_ROLE = 'hearth';

/** Bridge coordinator zome name */
const BRIDGE_ZOME = 'hearth_bridge';

// ============================================================================
// Hearth Bridge Client
// ============================================================================

/**
 * Client for the hearth-bridge coordinator zome.
 *
 * Provides cross-domain dispatch, audited queries, event broadcasting,
 * and health monitoring across all 10 hearth domains.
 *
 * ## Single-DNA Architecture
 *
 * Unlike commons (which splits into two sub-cluster DNAs), hearth uses
 * a single DNA role. All calls route to `hearth` -- no sub-cluster
 * routing logic is needed.
 */
export class HearthBridgeClient {
  constructor(private readonly client: ZomeCallable) {}

  // --- Cross-Domain Dispatch ---

  /**
   * Dispatch a synchronous call to any domain zome in the hearth cluster.
   * All zomes live on the single `hearth` DNA role.
   */
  async dispatch(zome: string, fn_name: string, payload: Uint8Array): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: HEARTH_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'dispatch_call',
      payload: { zome, fn_name, payload: Array.from(payload) },
    });
  }

  // --- Audited Queries ---

  /** Submit an audited cross-domain query with optional auto-dispatch */
  async query(input: HearthQueryInput): Promise<unknown> {
    return this.client.callZome({
      role_name: HEARTH_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'query_hearth',
      payload: {
        domain: input.domain,
        query_type: input.query_type,
        requester: null, // filled by zome from agent_info()
        params: input.params,
        result: null,
        created_at: null, // filled by zome from sys_time()
        resolved_at: null,
        success: null,
      },
    });
  }

  /** Resolve a pending query with a result */
  async resolveQuery(queryHash: Uint8Array, result: string, success: boolean): Promise<unknown> {
    return this.client.callZome({
      role_name: HEARTH_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'resolve_query',
      payload: { query_hash: queryHash, result, success },
    });
  }

  /** Get all queries for a specific domain */
  async getDomainQueries(domain: string): Promise<unknown[]> {
    return this.client.callZome({
      role_name: HEARTH_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_domain_queries',
      payload: domain,
    });
  }

  /** Get my queries */
  async getMyQueries(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: HEARTH_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_my_queries',
      payload: null,
    });
  }

  // --- Event Broadcasting ---

  /** Broadcast a cross-domain event */
  async broadcastEvent(input: HearthEventInput): Promise<unknown> {
    return this.client.callZome({
      role_name: HEARTH_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'broadcast_event',
      payload: {
        domain: input.domain,
        event_type: input.event_type,
        source_agent: null, // filled by zome
        payload: input.payload,
        created_at: null,
        related_hashes: input.related_hashes ?? [],
      },
    });
  }

  /** Get events for a specific domain */
  async getDomainEvents(domain: string): Promise<unknown[]> {
    return this.client.callZome({
      role_name: HEARTH_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_domain_events',
      payload: domain,
    });
  }

  /** Get events by type within a domain */
  async getEventsByType(query: EventTypeQuery): Promise<unknown[]> {
    return this.client.callZome({
      role_name: HEARTH_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_events_by_type',
      payload: query,
    });
  }

  /** Get all events across all domains */
  async getAllEvents(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: HEARTH_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_all_events',
      payload: null,
    });
  }

  /** Get my events */
  async getMyEvents(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: HEARTH_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_my_events',
      payload: null,
    });
  }

  // --- Cross-Cluster (Hearth → Personal) ---

  /** Dispatch a call to any zome in the personal DNA (cross-cluster) */
  async dispatchPersonalCall(zome: string, fn_name: string, payload: Uint8Array): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: HEARTH_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'dispatch_personal_call',
      payload: { role: 'personal', zome, fn_name, payload: Array.from(payload) },
    });
  }

  // --- Cross-Cluster (Hearth → Identity) ---

  /** Dispatch a call to any zome in the identity DNA (cross-cluster) */
  async dispatchIdentityCall(zome: string, fn_name: string, payload: Uint8Array): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: HEARTH_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'dispatch_identity_call',
      payload: { role: 'identity', zome, fn_name, payload: Array.from(payload) },
    });
  }

  // --- Cross-Cluster (Hearth → Commons) ---

  /** Dispatch a call to any zome in the commons DNA (cross-cluster) */
  async dispatchCommonsCall(zome: string, fn_name: string, payload: Uint8Array): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: HEARTH_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'dispatch_commons_call',
      payload: { role: 'commons', zome, fn_name, payload: Array.from(payload) },
    });
  }

  // --- Cross-Cluster (Hearth → Civic) ---

  /** Dispatch a call to any zome in the civic DNA (cross-cluster) */
  async dispatchCivicCall(zome: string, fn_name: string, payload: Uint8Array): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: HEARTH_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'dispatch_civic_call',
      payload: { role: 'civic', zome, fn_name, payload: Array.from(payload) },
    });
  }

  // --- Typed Convenience Functions (cross-cluster) ---

  /** Verify a hearth member's identity via the identity DNA */
  async verifyMemberIdentity(input: VerifyMemberIdentityInput): Promise<VerifyMemberIdentityResult> {
    return this.client.callZome({
      role_name: HEARTH_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'verify_member_identity',
      payload: input,
    });
  }

  /** Query care availability from the commons care domain */
  async queryCareAvailability(input: QueryCareAvailabilityInput): Promise<CareAvailabilityResult> {
    return this.client.callZome({
      role_name: HEARTH_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'query_care_availability',
      payload: input,
    });
  }

  /** Escalate a hearth emergency to the civic emergency system */
  async escalateEmergency(input: EscalateEmergencyInput): Promise<EscalateEmergencyResult> {
    return this.client.callZome({
      role_name: HEARTH_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'escalate_emergency',
      payload: input,
    });
  }

  /** Sync hearth state across clusters */
  async hearthSync(input: HearthSyncInput): Promise<HearthSyncResult> {
    return this.client.callZome({
      role_name: HEARTH_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'hearth_sync',
      payload: input,
    });
  }

  // --- H3 Severance Protocol ---

  /**
   * Initiate severance of a member from a hearth (H3 protocol).
   *
   * Exports requested data (milestones, care history, bond snapshot)
   * and assigns the member a new role. This is an irreversible operation
   * that preserves data dignity while enabling clean separation.
   */
  async initiateSeverance(input: SeveranceInput): Promise<SeveranceResult> {
    return this.client.callZome({
      role_name: HEARTH_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'initiate_severance',
      payload: {
        hearth_hash: Array.from(input.hearth_hash),
        member_hash: Array.from(input.member_hash),
        export_milestones: input.export_milestones,
        export_care_history: input.export_care_history,
        export_bond_snapshot: input.export_bond_snapshot,
        new_role: input.new_role,
      },
    });
  }

  // --- Audit Trail ---

  /** Query the bridge audit trail with time range and optional domain/type filters */
  async queryAuditTrail(query: AuditTrailQuery): Promise<AuditTrailResult> {
    return this.client.callZome({
      role_name: HEARTH_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'query_audit_trail',
      payload: {
        from_us: query.from_us,
        to_us: query.to_us,
        domain: query.domain ?? null,
        event_type: query.event_type ?? null,
      },
    });
  }

  // --- Health ---

  /** Health check across all 10 hearth domains */
  async healthCheck(): Promise<BridgeHealth> {
    return this.client.callZome({
      role_name: HEARTH_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'health_check',
      payload: null,
    });
  }
}

// ============================================================================
// Bridge Event Signals
// ============================================================================

/** Signal payload emitted by the hearth bridge when a cross-domain event is created */
export interface HearthBridgeEventSignal {
  signal_type: 'hearth_bridge_event';
  domain: string;
  event_type: string;
  payload: string;
  action_hash: Uint8Array;
}

/** Type guard for hearth bridge event signals */
export function isHearthBridgeSignal(signal: unknown): signal is HearthBridgeEventSignal {
  return (
    typeof signal === 'object' &&
    signal !== null &&
    'signal_type' in signal &&
    (signal as HearthBridgeEventSignal).signal_type === 'hearth_bridge_event'
  );
}

/** Callback type for signal subscriptions */
export type HearthBridgeSignalHandler = (signal: HearthBridgeEventSignal) => void;

// ============================================================================
// Factory
// ============================================================================

/** Create a HearthBridgeClient from an AppWebsocket or compatible client */
export function createHearthBridgeClient(client: ZomeCallable): HearthBridgeClient {
  return new HearthBridgeClient(client);
}
