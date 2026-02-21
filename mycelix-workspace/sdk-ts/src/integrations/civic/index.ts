/**
 * @mycelix/sdk Civic Cluster Integration
 *
 * Cluster-level client for the mycelix-civic DNA which unifies
 * justice, emergency, and media domains into a single Holochain DNA
 * with cross-domain dispatch.
 *
 * ## Architecture
 *
 * All 3 domains share one DNA role (`civic`) and communicate via
 * `civic_bridge` — a coordinator zome that dispatches calls between
 * domain zomes using `call(CallTargetCell::Local, ...)`.
 *
 * ## Usage
 *
 * ```typescript
 * import { CivicBridgeClient, createCivicBridgeClient } from '@mycelix/sdk/integrations/civic';
 *
 * const bridge = createCivicBridgeClient(appClient);
 *
 * // Cross-domain dispatch
 * const result = await bridge.dispatch('justice_cases', 'file_case', payload);
 *
 * // Audited query
 * const query = await bridge.query({
 *   domain: 'emergency',
 *   query_type: 'get_active_incidents',
 *   params: '{}',
 * });
 *
 * // Event broadcasting
 * await bridge.broadcastEvent({
 *   domain: 'media',
 *   event_type: 'factcheck_completed',
 *   payload: JSON.stringify({ article_id: '...', verdict: 'verified' }),
 * });
 *
 * // Health check across all 3 civic domains
 * const health = await bridge.healthCheck();
 * ```
 *
 * @packageDocumentation
 * @module integrations/civic
 */

// ============================================================================
// Re-exports from shared bridge-routing types
// ============================================================================

export type { CivicZome, CivicDomain, ConsciousnessCredential, GovernanceEligibility } from '../bridge-routing.js';

// ============================================================================
// Types
// ============================================================================

/** Input for cross-domain dispatch via the bridge */
export interface DispatchInput {
  /** Target zome name (e.g., "justice_cases", "emergency_incidents") */
  zome: import('../bridge-routing.js').CivicZome | string;
  /** Target function name */
  fn_name: string;
  /** MessagePack-encoded payload */
  payload: Uint8Array;
}

/** Result of a dispatched cross-domain call */
export interface DispatchResult {
  success: boolean;
  response?: Uint8Array;
  error?: string;
}

/** Input for an audited cross-domain query */
export interface CivicQueryInput {
  domain: import('../bridge-routing.js').CivicDomain | string;
  query_type: string;
  params: string;
}

/** Input for broadcasting a cross-domain event */
export interface CivicEventInput {
  domain: import('../bridge-routing.js').CivicDomain | string;
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

/** Input for cross-cluster dispatch to the commons DNA */
export interface CrossClusterDispatchInput {
  /** hApp role name of the target DNA (always "commons" for civic→commons) */
  role: string;
  /** Target zome in the commons DNA */
  zome: string;
  /** Target function name */
  fn_name: string;
  /** MessagePack-encoded payload */
  payload: Uint8Array;
}

/** Input for querying property registry before enforcement */
export interface QueryPropertyForEnforcementInput {
  property_id: string;
  case_id: string;
}

/** Result of a property enforcement query */
export interface PropertyEnforcementResult {
  property_found: boolean;
  enforcement_advisory: string;
  error?: string;
}

/** Input for checking housing capacity for emergency sheltering */
export interface CheckHousingCapacityInput {
  disaster_id: string;
  area: string;
}

/** Result of a housing capacity check */
export interface HousingCapacityResult {
  commons_reachable: boolean;
  recommendation: string;
  error?: string;
}

/** Input for verifying care credentials for evidence */
export interface VerifyCareCredentialsInput {
  provider_did: string;
  case_id: string;
}

/** Result of a care credential verification */
export interface CareCredentialVerifyResult {
  commons_reachable: boolean;
  recommendation: string;
  error?: string;
}

/** Input for checking active justice cases in an area */
export interface JusticeAreaQuery {
  area: string;
  case_type?: string;
}

/** Result of an area justice case query */
export interface JusticeAreaResult {
  active_cases: number;
  recommendation: string;
  error?: string;
}

/** Input for checking factcheck status of a claim */
export interface FactcheckStatusQuery {
  claim_id: string;
}

/** Result of a factcheck status query */
export interface FactcheckStatusResult {
  has_factcheck: boolean;
  verdict?: string;
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

const CIVIC_ROLE = 'civic';
const BRIDGE_ZOME = 'civic_bridge';

/** All domain zomes available in the civic cluster */
export const CIVIC_DOMAINS = ['justice', 'emergency', 'media'] as const;

export const CIVIC_ZOMES = [
  'justice_cases', 'justice_evidence', 'justice_arbitration', 'justice_restorative', 'justice_enforcement',
  'emergency_incidents', 'emergency_triage', 'emergency_resources', 'emergency_coordination', 'emergency_shelters', 'emergency_comms',
  'media_publication', 'media_attribution', 'media_factcheck', 'media_curation',
] as const;

// ============================================================================
// Civic Bridge Client
// ============================================================================

/**
 * Client for the civic-bridge coordinator zome.
 *
 * Provides cross-domain dispatch, audited queries, event broadcasting,
 * and health monitoring across all 3 civic domains.
 */
export class CivicBridgeClient {
  constructor(private readonly client: ZomeCallable) {}

  // --- Cross-Domain Dispatch ---

  /** Dispatch a synchronous call to any domain zome in the civic cluster */
  async dispatch(zome: string, fn_name: string, payload: Uint8Array): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: CIVIC_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'dispatch_call',
      payload: { zome, fn_name, payload: Array.from(payload) },
    });
  }

  // --- Audited Queries ---

  /** Submit an audited cross-domain query with optional auto-dispatch */
  async query(input: CivicQueryInput): Promise<unknown> {
    return this.client.callZome({
      role_name: CIVIC_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'query_civic',
      payload: {
        domain: input.domain,
        query_type: input.query_type,
        requester: null,
        params: input.params,
        result: null,
        created_at: null,
        resolved_at: null,
        success: null,
      },
    });
  }

  /** Resolve a pending query with a result */
  async resolveQuery(queryHash: Uint8Array, result: string, success: boolean): Promise<unknown> {
    return this.client.callZome({
      role_name: CIVIC_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'resolve_query',
      payload: { query_hash: queryHash, result, success },
    });
  }

  /** Get my queries */
  async getMyQueries(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: CIVIC_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_my_queries',
      payload: null,
    });
  }

  // --- Event Broadcasting ---

  /** Broadcast a cross-domain event */
  async broadcastEvent(input: CivicEventInput): Promise<unknown> {
    return this.client.callZome({
      role_name: CIVIC_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'broadcast_event',
      payload: {
        domain: input.domain,
        event_type: input.event_type,
        source_agent: null,
        payload: input.payload,
        created_at: null,
        related_hashes: input.related_hashes ?? [],
      },
    });
  }

  /** Get events for a specific domain */
  async getDomainEvents(domain: string): Promise<unknown[]> {
    return this.client.callZome({
      role_name: CIVIC_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_domain_events',
      payload: domain,
    });
  }

  /** Get events by type within a domain */
  async getEventsByType(query: EventTypeQuery): Promise<unknown[]> {
    return this.client.callZome({
      role_name: CIVIC_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_events_by_type',
      payload: query,
    });
  }

  /** Get all events across all domains */
  async getAllEvents(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: CIVIC_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_all_events',
      payload: null,
    });
  }

  /** Get my events */
  async getMyEvents(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: CIVIC_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_my_events',
      payload: null,
    });
  }

  // --- Cross-Cluster (Civic → Commons) ---

  /** Dispatch a call to any zome in the commons DNA (cross-cluster) */
  async dispatchCommonsCall(zome: string, fn_name: string, payload: Uint8Array): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: CIVIC_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'dispatch_commons_call',
      payload: { role: 'commons', zome, fn_name, payload: Array.from(payload) },
    });
  }

  /** Query property registry before enforcement action */
  async queryPropertyForEnforcement(input: QueryPropertyForEnforcementInput): Promise<PropertyEnforcementResult> {
    return this.client.callZome({
      role_name: CIVIC_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'query_property_for_enforcement',
      payload: input,
    });
  }

  /** Check housing capacity for emergency sheltering */
  async checkHousingCapacityForSheltering(input: CheckHousingCapacityInput): Promise<HousingCapacityResult> {
    return this.client.callZome({
      role_name: CIVIC_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'check_housing_capacity_for_sheltering',
      payload: input,
    });
  }

  /** Verify care provider credentials for evidence in justice cases */
  async verifyCareCredentialsForEvidence(input: VerifyCareCredentialsInput): Promise<CareCredentialVerifyResult> {
    return this.client.callZome({
      role_name: CIVIC_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'verify_care_credentials_for_evidence',
      payload: input,
    });
  }

  // --- Typed Convenience Functions (intra-cluster) ---

  /** Get active cases for an area — typed wrapper for justice_cases.get_cases_for_area */
  async getActiveCasesForArea(input: JusticeAreaQuery): Promise<JusticeAreaResult> {
    return this.client.callZome({
      role_name: CIVIC_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_active_cases_for_area',
      payload: input,
    });
  }

  /** Check factcheck status of a claim — typed wrapper for media_factcheck.check_status */
  async checkFactcheckStatus(input: FactcheckStatusQuery): Promise<FactcheckStatusResult> {
    return this.client.callZome({
      role_name: CIVIC_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'check_factcheck_status',
      payload: input,
    });
  }

  // --- Audit Trail ---

  /** Query the bridge audit trail with time range and optional domain/type filters */
  async queryAuditTrail(query: AuditTrailQuery): Promise<AuditTrailResult> {
    return this.client.callZome({
      role_name: CIVIC_ROLE,
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

  /** Health check across all 3 civic domains */
  async healthCheck(): Promise<BridgeHealth> {
    return this.client.callZome({
      role_name: CIVIC_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'health_check',
      payload: null,
    });
  }
}

// ============================================================================
// Bridge Event Signals
// ============================================================================

/** Signal payload emitted by the civic bridge when a cross-domain event is created */
export interface CivicBridgeEventSignal {
  signal_type: 'civic_bridge_event';
  domain: string;
  event_type: string;
  payload: string;
  action_hash: Uint8Array;
}

/** Type guard for civic bridge event signals */
export function isCivicBridgeSignal(signal: unknown): signal is CivicBridgeEventSignal {
  return (
    typeof signal === 'object' &&
    signal !== null &&
    'signal_type' in signal &&
    (signal as CivicBridgeEventSignal).signal_type === 'civic_bridge_event'
  );
}

/** Callback type for signal subscriptions */
export type CivicBridgeSignalHandler = (signal: CivicBridgeEventSignal) => void;

// ============================================================================
// Factory
// ============================================================================

/** Create a CivicBridgeClient from an AppWebsocket or compatible client */
export function createCivicBridgeClient(client: ZomeCallable): CivicBridgeClient {
  return new CivicBridgeClient(client);
}

// ============================================================================
// Re-exports from domain integrations
// ============================================================================

export { JusticeBridgeClient, getJusticeBridgeClient } from '../justice/index.js';
export { MediaBridgeClient, getMediaBridgeClient } from '../media/index.js';
