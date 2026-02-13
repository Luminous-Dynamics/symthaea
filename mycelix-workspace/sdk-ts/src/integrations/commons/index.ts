/**
 * @mycelix/sdk Commons Cluster Integration
 *
 * Cluster-level client for the mycelix-commons DNA which unifies
 * property, housing, care, mutual-aid, water, food, and transport
 * domains into a single Holochain DNA with cross-domain dispatch.
 *
 * ## Architecture
 *
 * All 5 domains share one DNA role (`commons`) and communicate via
 * `commons_bridge` — a coordinator zome that dispatches calls between
 * domain zomes using `call(CallTargetCell::Local, ...)`.
 *
 * ## Usage
 *
 * ```typescript
 * import { CommonsBridgeClient, createCommonsBridgeClient } from '@mycelix/sdk/integrations/commons';
 *
 * const bridge = createCommonsBridgeClient(appClient);
 *
 * // Cross-domain dispatch: call any zome function by name
 * const result = await bridge.dispatch('property_registry', 'get_asset', payload);
 *
 * // Audited query with auto-dispatch
 * const query = await bridge.query({
 *   domain: 'housing',
 *   query_type: 'get_clt_lease',
 *   params: JSON.stringify({ unit_id: '...' }),
 * });
 *
 * // Event broadcasting
 * await bridge.broadcastEvent({
 *   domain: 'care',
 *   event_type: 'match_completed',
 *   payload: JSON.stringify({ ... }),
 * });
 *
 * // Health check across all 5 domains
 * const health = await bridge.healthCheck();
 * ```
 *
 * @packageDocumentation
 * @module integrations/commons
 */

// ============================================================================
// Types
// ============================================================================

/** Input for cross-domain dispatch via the bridge */
export interface DispatchInput {
  /** Target zome name (e.g., "property_registry", "housing_clt") */
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
export interface CommonsQueryInput {
  domain: 'property' | 'housing' | 'care' | 'mutualaid' | 'water' | 'food' | 'transport';
  query_type: string;
  params: string;
}

/** Input for broadcasting a cross-domain event */
export interface CommonsEventInput {
  domain: 'property' | 'housing' | 'care' | 'mutualaid' | 'water' | 'food' | 'transport';
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

/** Input for cross-cluster dispatch to the civic DNA */
export interface CrossClusterDispatchInput {
  /** hApp role name of the target DNA (always "civic" for commons→civic) */
  role: string;
  /** Target zome in the civic DNA */
  zome: string;
  /** Target function name */
  fn_name: string;
  /** MessagePack-encoded payload */
  payload: Uint8Array;
}

/** Input for checking active emergencies near a location */
export interface CheckEmergencyForAreaInput {
  lat: number;
  lon: number;
}

/** Result of an emergency area check */
export interface EmergencyAreaCheckResult {
  has_active_emergencies: boolean;
  active_count: number;
  recommendation: string;
  error?: string;
}

/** Input for checking justice disputes affecting a property */
export interface CheckJusticeDisputesInput {
  resource_id: string;
}

/** Result of a justice dispute check */
export interface JusticeDisputeCheckResult {
  has_pending_cases: boolean;
  recommendation: string;
  error?: string;
}

/** Input for verifying property ownership */
export interface PropertyOwnershipQuery {
  property_id: string;
  requester_did: string;
}

/** Result of a property ownership verification */
export interface PropertyOwnershipResult {
  is_owner: boolean;
  owner_did?: string;
  error?: string;
}

/** Input for checking care provider availability */
export interface CareAvailabilityQuery {
  skill_needed: string;
  location?: string;
}

/** Result of a care availability query */
export interface CareAvailabilityResult {
  available_count: number;
  recommendation: string;
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

const COMMONS_ROLE = 'commons';
const BRIDGE_ZOME = 'commons_bridge';

/** All domain zomes available in the commons cluster */
export const COMMONS_DOMAINS = ['property', 'housing', 'care', 'mutualaid', 'water', 'food', 'transport'] as const;

export const COMMONS_ZOMES = [
  'property_registry', 'property_transfer', 'property_disputes', 'property_commons',
  'housing_units', 'housing_membership', 'housing_finances', 'housing_maintenance', 'housing_clt', 'housing_governance',
  'care_timebank', 'care_circles', 'care_matching', 'care_plans', 'care_credentials',
  'mutualaid_needs', 'mutualaid_circles', 'mutualaid_governance', 'mutualaid_pools', 'mutualaid_requests', 'mutualaid_resources', 'mutualaid_timebank',
  'water_flow', 'water_purity', 'water_capture', 'water_steward', 'water_wisdom',
  'food_production', 'food_distribution', 'food_preservation', 'food_knowledge',
  'transport_routes', 'transport_sharing', 'transport_impact',
] as const;

// ============================================================================
// Commons Bridge Client
// ============================================================================

/**
 * Client for the commons-bridge coordinator zome.
 *
 * Provides cross-domain dispatch, audited queries, event broadcasting,
 * and health monitoring across all 5 commons domains.
 */
export class CommonsBridgeClient {
  constructor(private readonly client: ZomeCallable) {}

  // --- Cross-Domain Dispatch ---

  /** Dispatch a synchronous call to any domain zome in the commons cluster */
  async dispatch(zome: string, fn_name: string, payload: Uint8Array): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'dispatch_call',
      payload: { zome, fn_name, payload: Array.from(payload) },
    });
  }

  // --- Audited Queries ---

  /** Submit an audited cross-domain query with optional auto-dispatch */
  async query(input: CommonsQueryInput): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'query_commons',
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
      role_name: COMMONS_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'resolve_query',
      payload: { query_hash: queryHash, result, success },
    });
  }

  /** Get all queries for a specific domain */
  async getDomainQueries(domain: string): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_domain_queries',
      payload: domain,
    });
  }

  /** Get my queries */
  async getMyQueries(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_my_queries',
      payload: null,
    });
  }

  // --- Event Broadcasting ---

  /** Broadcast a cross-domain event */
  async broadcastEvent(input: CommonsEventInput): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
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
      role_name: COMMONS_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_domain_events',
      payload: domain,
    });
  }

  /** Get events by type within a domain */
  async getEventsByType(query: EventTypeQuery): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_events_by_type',
      payload: query,
    });
  }

  /** Get all events across all domains */
  async getAllEvents(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_all_events',
      payload: null,
    });
  }

  /** Get my events */
  async getMyEvents(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_my_events',
      payload: null,
    });
  }

  // --- Cross-Cluster (Commons → Civic) ---

  /** Dispatch a call to any zome in the civic DNA (cross-cluster) */
  async dispatchCivicCall(zome: string, fn_name: string, payload: Uint8Array): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'dispatch_civic_call',
      payload: { role: 'civic', zome, fn_name, payload: Array.from(payload) },
    });
  }

  /** Check if there are active emergencies near a lat/lon (queries civic emergency_incidents) */
  async checkEmergencyForArea(input: CheckEmergencyForAreaInput): Promise<EmergencyAreaCheckResult> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'check_emergency_for_area',
      payload: input,
    });
  }

  /** Check if there are pending justice disputes affecting a property (queries civic justice_cases) */
  async checkJusticeDisputesForProperty(input: CheckJusticeDisputesInput): Promise<JusticeDisputeCheckResult> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'check_justice_disputes_for_property',
      payload: input,
    });
  }

  // --- Typed Convenience Functions (intra-cluster) ---

  /** Verify property ownership — typed wrapper for property_registry.verify_ownership */
  async verifyPropertyOwnership(input: PropertyOwnershipQuery): Promise<PropertyOwnershipResult> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'verify_property_ownership',
      payload: input,
    });
  }

  /** Check care provider availability — typed wrapper for care_matching.check_availability */
  async checkCareAvailability(input: CareAvailabilityQuery): Promise<CareAvailabilityResult> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'check_care_availability',
      payload: input,
    });
  }

  // --- Audit Trail ---

  /** Query the bridge audit trail with time range and optional domain/type filters */
  async queryAuditTrail(query: AuditTrailQuery): Promise<AuditTrailResult> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
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

  /** Health check across all 7 commons domains */
  async healthCheck(): Promise<BridgeHealth> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'health_check',
      payload: null,
    });
  }
}

// ============================================================================
// Bridge Event Signals
// ============================================================================

/** Signal payload emitted by the commons bridge when a cross-domain event is created */
export interface CommonsBridgeEventSignal {
  signal_type: 'commons_bridge_event';
  domain: string;
  event_type: string;
  payload: string;
  action_hash: Uint8Array;
}

/** Type guard for commons bridge event signals */
export function isCommonsBridgeSignal(signal: unknown): signal is CommonsBridgeEventSignal {
  return (
    typeof signal === 'object' &&
    signal !== null &&
    'signal_type' in signal &&
    (signal as CommonsBridgeEventSignal).signal_type === 'commons_bridge_event'
  );
}

/** Callback type for signal subscriptions */
export type BridgeSignalHandler = (signal: CommonsBridgeEventSignal) => void;

// ============================================================================
// Factory
// ============================================================================

/** Create a CommonsBridgeClient from an AppWebsocket or compatible client */
export function createCommonsBridgeClient(client: ZomeCallable): CommonsBridgeClient {
  return new CommonsBridgeClient(client);
}

// ============================================================================
// Re-exports from domain integrations
// ============================================================================

export { PropertyBridgeClient, getPropertyBridgeClient } from '../property/index.js';
export { MutualAidService, getMutualAidService } from '../mutualaid/index.js';
export { FoodClient, createFoodClient } from '../food/index.js';
export { TransportClient, createTransportClient } from '../transport/index.js';
