/**
 * Identity Bridge Client
 *
 * Client for cross-hApp identity operations in Mycelix-Identity.
 * Enables identity verification, reputation aggregation, and event broadcasting.
 *
 * @module @mycelix/sdk/clients/identity/bridge
 */

import { DEFAULT_IDENTITY_CLIENT_CONFIG } from './types.js';
import { ZomeClient } from '../../core/zome-client.js';

import type {
  HappRegistration,
  IdentityVerificationResult,
  AggregatedReputation,
  BridgeEvent,
  RegisterHappInput,
  QueryIdentityInput,
  ReportReputationInput,
  BroadcastEventInput,
  GetEventsInput,
  TrustCheckInput,
  IdentityClientConfig,
} from './types.js';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';


/**
 * hApp registration record wrapper
 */
export interface HappRegistrationRecord {
  hash: Uint8Array;
  registration: HappRegistration;
}

/**
 * Bridge event record wrapper
 */
export interface BridgeEventRecord {
  hash: Uint8Array;
  event: BridgeEvent;
}

/**
 * Identity Bridge Client
 *
 * Provides cross-hApp identity operations including DID verification,
 * reputation aggregation across hApps, and identity event broadcasting.
 *
 * @example
 * ```typescript
 * const bridgeClient = new BridgeClient(appClient);
 *
 * // Register your hApp with the identity bridge
 * await bridgeClient.registerHapp({
 *   happ_id: 'mycelix-marketplace',
 *   happ_name: 'Mycelix Marketplace',
 *   capabilities: ['trade', 'reputation', 'escrow'],
 * });
 *
 * // Query identity from another hApp
 * const verification = await bridgeClient.queryIdentity({
 *   did: 'did:mycelix:abc123',
 *   source_happ: 'mycelix-marketplace',
 *   requested_fields: ['matl_score', 'credential_count'],
 * });
 *
 * // Report reputation from your hApp
 * await bridgeClient.reportReputation({
 *   did: 'did:mycelix:abc123',
 *   source_happ: 'mycelix-marketplace',
 *   score: 0.85,
 *   interactions: 42,
 * });
 *
 * // Get aggregated reputation
 * const reputation = await bridgeClient.getReputation('did:mycelix:abc123');
 * console.log(`Aggregate score: ${reputation.aggregate_score}`);
 * ```
 */
export class BridgeClient extends ZomeClient {
  protected readonly zomeName = 'identity_bridge';

  constructor(client: AppClient, config: Partial<IdentityClientConfig> = {}) {
    const mergedConfig = { ...DEFAULT_IDENTITY_CLIENT_CONFIG, ...config };
    super(client, {
      roleName: mergedConfig.roleName,
      timeout: mergedConfig.timeout,
    });
  }

  // ==========================================================================
  // HAPP REGISTRATION
  // ==========================================================================

  /**
   * Register a hApp with the identity bridge
   *
   * Registered hApps can query identities and report reputation.
   *
   * @param input - Registration parameters
   * @returns Created registration record
   */
  async registerHapp(input: RegisterHappInput): Promise<HappRegistrationRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('register_happ', input);
    return this.recordToRegistrationRecord(record);
  }

  /**
   * Get all registered hApps
   *
   * @returns Array of registration records
   */
  async getRegisteredHapps(): Promise<HappRegistrationRecord[]> {
    const records = await this.callZome<HolochainRecord[]>('get_registered_happs', null);
    return records.map((r) => this.recordToRegistrationRecord(r));
  }

  // ==========================================================================
  // IDENTITY QUERIES
  // ==========================================================================

  /**
   * Query identity verification for a DID
   *
   * Performs comprehensive identity verification including:
   * - DID existence check
   * - Deactivation status
   * - MATL score aggregation
   * - Credential count
   *
   * @param input - Query parameters
   * @returns Identity verification result
   */
  async queryIdentity(input: QueryIdentityInput): Promise<IdentityVerificationResult> {
    return this.callZome<IdentityVerificationResult>('query_identity', input);
  }

  /**
   * Verify if a DID exists
   *
   * Quick check without full verification.
   *
   * @param did - DID to verify
   * @returns true if DID exists, false otherwise
   */
  async verifyDid(did: string): Promise<boolean> {
    return this.callZome<boolean>('verify_did', did);
  }

  /**
   * Check if a DID meets a trust threshold
   *
   * @param input - Trust check parameters
   * @returns true if DID meets threshold, false otherwise
   */
  async isTrustworthy(input: TrustCheckInput): Promise<boolean> {
    return this.callZome<boolean>('is_trustworthy', input);
  }

  // ==========================================================================
  // REPUTATION
  // ==========================================================================

  /**
   * Report reputation for a DID from your hApp
   *
   * Contributes to the aggregated reputation score for a DID.
   *
   * @param input - Reputation report parameters
   * @returns Created reputation record
   */
  async reportReputation(input: ReportReputationInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('report_reputation', input);
  }

  /**
   * Get aggregated reputation for a DID
   *
   * Returns reputation scores from all hApps weighted by interactions.
   *
   * @param did - DID to get reputation for
   * @returns Aggregated reputation across all hApps
   */
  async getReputation(did: string): Promise<AggregatedReputation> {
    return this.callZome<AggregatedReputation>('get_reputation', did);
  }

  /**
   * Get MATL score for a DID
   *
   * Returns the computed MATL (Mycelix Adaptive Trust Layer) score.
   *
   * @param did - DID to get MATL score for
   * @returns MATL score (0.0 - 1.0)
   */
  async getMatlScore(did: string): Promise<number> {
    return this.callZome<number>('get_matl_score', did);
  }

  // ==========================================================================
  // EVENT BROADCASTING
  // ==========================================================================

  /**
   * Broadcast an identity event
   *
   * Events are published for other hApps to receive.
   *
   * @param input - Event broadcast parameters
   * @returns Created event record
   */
  async broadcastEvent(input: BroadcastEventInput): Promise<BridgeEventRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('broadcast_event', input);
    return this.recordToEventRecord(record);
  }

  /**
   * Get recent identity events
   *
   * Retrieve events with optional filtering by type and time.
   *
   * @param input - Query parameters
   * @returns Array of event records
   */
  async getRecentEvents(input?: GetEventsInput): Promise<BridgeEventRecord[]> {
    const records = await this.callZome<HolochainRecord[]>('get_recent_events', input ?? {});
    return records.map((r) => this.recordToEventRecord(r));
  }

  // ==========================================================================
  // PRIVATE HELPERS
  // ==========================================================================

  private recordToRegistrationRecord(record: HolochainRecord): HappRegistrationRecord {
    const registration = this.extractEntry<HappRegistration>(record);
    return {
      hash: this.getActionHash(record),
      registration,
    };
  }

  private recordToEventRecord(record: HolochainRecord): BridgeEventRecord {
    const event = this.extractEntry<BridgeEvent>(record);
    return {
      hash: this.getActionHash(record),
      event,
    };
  }
}
