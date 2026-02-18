/**
 * @mycelix/sdk Personal (Sovereign) Cluster Integration
 *
 * Cluster-level client for the mycelix-personal DNA which provides
 * the Sovereign tier: private identity vault, health vault, credential
 * wallet, and cross-cluster bridge for selective disclosure.
 *
 * ## Architecture
 *
 * All 4 zomes share one DNA role (`personal`) and communicate via
 * `personal_bridge` — a coordinator zome that dispatches calls between
 * domain zomes using `call(CallTargetCell::Local, ...)` and to other
 * clusters (identity, commons, civic, governance) via
 * `call(CallTargetCell::OtherRole(...), ...)`.
 *
 * ## Usage
 *
 * ```typescript
 * import { PersonalBridgeClient, createPersonalBridgeClient } from '@mycelix/sdk/integrations/personal';
 *
 * const bridge = createPersonalBridgeClient(appClient);
 *
 * // Dispatch to any personal zome
 * const result = await bridge.dispatch('identity_vault', 'get_profile', payload);
 *
 * // Audited query with auto-dispatch
 * const query = await bridge.query({
 *   domain: 'identity',
 *   query_type: 'get_profile',
 *   params: '{}',
 * });
 *
 * // Credential presentation for governance voting
 * const phi = await bridge.presentPhiCredential();
 *
 * // DID resolution via identity cluster
 * const did = await bridge.resolveDid('did:mycelix:uhCAk...');
 *
 * // Health check
 * const health = await bridge.healthCheck();
 * ```
 *
 * @packageDocumentation
 * @module integrations/personal
 */

// ============================================================================
// Types
// ============================================================================

/** Input for cross-domain dispatch via the bridge */
export interface DispatchInput {
  /** Target zome name (e.g., "identity_vault", "health_vault", "credential_wallet") */
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

/** Input for an audited personal query */
export interface PersonalQueryInput {
  domain: 'identity' | 'health' | 'credential';
  query_type: string;
  params: string;
}

/** Input for broadcasting a personal event */
export interface PersonalEventInput {
  domain: 'identity' | 'health' | 'credential';
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

/** Input for cross-cluster dispatch */
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

/** Credential types supported by the personal cluster */
export type CredentialType =
  | 'FederatedLearning'
  | 'Governance'
  | 'Identity'
  | 'HealthRecord'
  | 'Education'
  | 'Custom';

/** Disclosure scope for credential presentations */
export type DisclosureScope =
  | { Full: null }
  | { ExistenceOnly: null }
  | { SelectedFields: string[] };

/** Credential presentation result */
export interface CredentialPresentation {
  credential_type: CredentialType;
  disclosed_data: string;
  scope: DisclosureScope;
  presented_at: number;
}

/** Input for Phi attestation submission */
export interface SubmitPhiAttestationInput {
  /** Phi value from Symthaea cognitive cycle, in [0.0, 1.0] */
  phi: number;
  /** Symthaea cognitive cycle number (must be > 0) */
  cycle_id: number;
}

/** Input for querying identity verification from the identity bridge */
export interface QueryIdentityInput {
  did: string;
  source_happ: string;
  requested_fields: string[];
}

/** Input for creating a verifiable presentation */
export interface CreatePresentationInput {
  credential_hashes: Uint8Array[];
  challenge?: string;
  domain?: string;
  verifier_did?: string;
}

// ============================================================================
// Constants
// ============================================================================

const PERSONAL_ROLE = 'personal';
const BRIDGE_ZOME = 'personal_bridge';

/** All domain zomes available in the personal cluster */
export const PERSONAL_DOMAINS = ['identity', 'health', 'credential'] as const;

export const PERSONAL_ZOMES = [
  'identity_vault',
  'health_vault',
  'credential_wallet',
  'personal_bridge',
] as const;

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
// Personal Bridge Client
// ============================================================================

/**
 * Client for the personal-bridge coordinator zome.
 *
 * Provides intra-cluster dispatch, credential presentations,
 * cross-cluster dispatch (to identity, commons, civic, governance),
 * and identity/credential proxy functions.
 */
export class PersonalBridgeClient {
  constructor(private readonly client: ZomeCallable) {}

  // --- Intra-Cluster Dispatch ---

  /** Dispatch a synchronous call to any domain zome in the personal cluster */
  async dispatch(zome: string, fn_name: string, payload: Uint8Array): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'dispatch_call',
      payload: { zome, fn_name, payload: Array.from(payload) },
    });
  }

  // --- Audited Queries ---

  /** Submit an audited personal query with optional auto-dispatch */
  async query(input: PersonalQueryInput): Promise<unknown> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'query_personal',
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
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'resolve_query',
      payload: { query_hash: queryHash, result, success },
    });
  }

  /** Get my queries */
  async getMyQueries(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_my_queries',
      payload: null,
    });
  }

  // --- Event Broadcasting ---

  /** Broadcast a personal event */
  async broadcastEvent(input: PersonalEventInput): Promise<unknown> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
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
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_domain_events',
      payload: domain,
    });
  }

  /** Get events by type within a domain */
  async getEventsByType(query: EventTypeQuery): Promise<unknown[]> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_events_by_type',
      payload: query,
    });
  }

  /** Get all events across all domains */
  async getAllEvents(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_all_events',
      payload: null,
    });
  }

  // --- Credential Presentations ---

  /** Present a Phi credential for governance voting */
  async presentPhiCredential(): Promise<CredentialPresentation> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'present_phi_credential',
      payload: null,
    });
  }

  /** Present an identity proof with selective disclosure of specified fields */
  async presentIdentityProof(fields: string[]): Promise<CredentialPresentation> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'present_identity_proof',
      payload: fields,
    });
  }

  /** Present a K-vector credential for governance trust scoring */
  async presentKVector(): Promise<CredentialPresentation> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'present_k_vector',
      payload: null,
    });
  }

  // --- Phi Attestation ---

  /** Submit a signed Phi attestation to the governance cluster */
  async submitPhiAttestation(input: SubmitPhiAttestationInput): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'submit_phi_attestation',
      payload: input,
    });
  }

  // --- Cross-Cluster Dispatch ---

  /** Dispatch a call to any zome in the commons DNA */
  async dispatchCommonsCall(zome: string, fn_name: string, payload: Uint8Array): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'dispatch_commons_call',
      payload: { role: 'commons', zome, fn_name, payload: Array.from(payload) },
    });
  }

  /** Dispatch a call to any zome in the civic DNA */
  async dispatchCivicCall(zome: string, fn_name: string, payload: Uint8Array): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'dispatch_civic_call',
      payload: { role: 'civic', zome, fn_name, payload: Array.from(payload) },
    });
  }

  /** Dispatch a call to any zome in the governance DNA */
  async dispatchGovernanceCall(zome: string, fn_name: string, payload: Uint8Array): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'dispatch_governance_call',
      payload: { role: 'governance', zome, fn_name, payload: Array.from(payload) },
    });
  }

  /** Dispatch a call to any zome in the identity DNA */
  async dispatchIdentityCall(zome: string, fn_name: string, payload: Uint8Array): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'dispatch_identity_call',
      payload: { role: 'identity', zome, fn_name, payload: Array.from(payload) },
    });
  }

  // --- Identity Cluster Proxies (DID Registry) ---

  /** Resolve a DID document from the identity cluster */
  async resolveDid(did: string): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'resolve_did',
      payload: did,
    });
  }

  /** Check whether a DID is currently active */
  async isDidActive(did: string): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'is_did_active',
      payload: did,
    });
  }

  /** Query identity verification including MATL score and credential count */
  async queryIdentity(input: QueryIdentityInput): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'query_identity',
      payload: input,
    });
  }

  /** Get MATL trust score for a DID */
  async getMatlScore(did: string): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_matl_score',
      payload: did,
    });
  }

  // --- Identity Cluster Proxies (Verifiable Credentials) ---

  /** Verify a credential by ID via the identity cluster */
  async verifyCredential(credentialId: string): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'verify_credential',
      payload: credentialId,
    });
  }

  /** Get a credential record by ID */
  async getCredential(credentialId: string): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_credential',
      payload: credentialId,
    });
  }

  /** Check whether a credential has been revoked */
  async isCredentialRevoked(credentialId: string): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'is_credential_revoked',
      payload: credentialId,
    });
  }

  /** Get all credentials held by the calling agent from the identity cluster */
  async getMyCredentialsFromIdentity(): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_my_credentials_from_identity',
      payload: null,
    });
  }

  /** Create a verifiable presentation from credentials in the identity cluster */
  async createPresentation(input: CreatePresentationInput): Promise<DispatchResult> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'create_presentation',
      payload: {
        credential_hashes: input.credential_hashes,
        challenge: input.challenge ?? null,
        domain: input.domain ?? null,
        verifier_did: input.verifier_did ?? null,
      },
    });
  }

  // --- Health ---

  /** Health check across all personal domains */
  async healthCheck(): Promise<BridgeHealth> {
    return this.client.callZome({
      role_name: PERSONAL_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'health_check',
      payload: null,
    });
  }
}

// ============================================================================
// Bridge Event Signals
// ============================================================================

/** Signal payload emitted by the personal bridge when a cross-domain event is created */
export interface PersonalBridgeEventSignal {
  signal_type: 'personal_bridge_event';
  domain: string;
  event_type: string;
  payload: string;
  action_hash: Uint8Array;
}

/** Type guard for personal bridge event signals */
export function isPersonalBridgeSignal(signal: unknown): signal is PersonalBridgeEventSignal {
  return (
    typeof signal === 'object' &&
    signal !== null &&
    'signal_type' in signal &&
    (signal as PersonalBridgeEventSignal).signal_type === 'personal_bridge_event'
  );
}

/** Callback type for signal subscriptions */
export type PersonalBridgeSignalHandler = (signal: PersonalBridgeEventSignal) => void;

// ============================================================================
// Factory
// ============================================================================

/** Create a PersonalBridgeClient from an AppWebsocket or compatible client */
export function createPersonalBridgeClient(client: ZomeCallable): PersonalBridgeClient {
  return new PersonalBridgeClient(client);
}
