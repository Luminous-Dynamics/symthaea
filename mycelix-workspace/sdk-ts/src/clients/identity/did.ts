// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * DID Registry Client
 *
 * Client for DID document management in Mycelix-Identity.
 * Implements W3C DID Core Specification (did:mycelix method).
 *
 * @module @mycelix/sdk/clients/identity/did
 */

import { IdentitySdkError, IdentitySdkErrorCode, DEFAULT_IDENTITY_CLIENT_CONFIG } from './types.js';
import { ZomeClient } from '../../core/zome-client.js';

import type {
  DidDocument,
  DidDeactivation,
  DidRecord,
  UpdateDidInput,
  VerificationMethod,
  ServiceEndpoint,
  IdentityClientConfig,
} from './types.js';
import type { AppClient, AgentPubKey, Record as HolochainRecord } from '@holochain/client';


/**
 * DID Registry Client
 *
 * Provides comprehensive DID document management including creation,
 * resolution, updates, and deactivation.
 *
 * @example
 * ```typescript
 * const didClient = new DidClient(appClient, { roleName: 'mycelix_identity' });
 *
 * // Create a DID for the current agent
 * const myDid = await didClient.createDid();
 * console.log(`Created DID: ${myDid.document.id}`);
 *
 * // Resolve a DID
 * const resolved = await didClient.resolveDid('did:mycelix:abc123...');
 *
 * // Add a service endpoint
 * await didClient.addServiceEndpoint({
 *   id: 'did:mycelix:abc123#messaging',
 *   type: 'MessagingService',
 *   serviceEndpoint: 'https://example.com/messaging',
 * });
 * ```
 */
export class DidClient extends ZomeClient {
  protected readonly zomeName = 'did_registry';

  constructor(client: AppClient, config: Partial<IdentityClientConfig> = {}) {
    const mergedConfig = { ...DEFAULT_IDENTITY_CLIENT_CONFIG, ...config };
    super(client, {
      roleName: mergedConfig.roleName,
      timeout: mergedConfig.timeout,
    });
  }

  // ==========================================================================
  // DID CREATION & RETRIEVAL
  // ==========================================================================

  /**
   * Create a new DID for the current agent
   *
   * Creates a DID document linked to the agent's public key.
   * The DID format is: did:mycelix:<agent_pub_key>
   *
   * @returns Created DID record with document
   * @throws IdentitySdkError if creation fails
   */
  async createDid(): Promise<DidRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('create_did', null);
    return this.recordToDidRecord(record);
  }

  /**
   * Get the current agent's DID document
   *
   * @returns DID record or null if none exists
   */
  async getMyDid(): Promise<DidRecord | null> {
    const record = await this.callZome<HolochainRecord | null>('get_my_did', null);
    return record ? this.recordToDidRecord(record) : null;
  }

  /**
   * Get DID document for a specific agent
   *
   * @param agentPubKey - Public key of the agent
   * @returns DID record or null if not found
   */
  async getDidDocument(agentPubKey: AgentPubKey): Promise<DidRecord | null> {
    const record = await this.callZome<HolochainRecord | null>('get_did_document', agentPubKey);
    return record ? this.recordToDidRecord(record) : null;
  }

  /**
   * Resolve a DID string to its document
   *
   * @param did - DID string (e.g., "did:mycelix:...")
   * @returns DID record or null if not found
   * @throws IdentitySdkError if DID format is invalid
   */
  async resolveDid(did: string): Promise<DidRecord | null> {
    this.validateDidFormat(did);
    const record = await this.callZome<HolochainRecord | null>('resolve_did', did);
    return record ? this.recordToDidRecord(record) : null;
  }

  // ==========================================================================
  // DID STATUS
  // ==========================================================================

  /**
   * Check if a DID is active (not deactivated)
   *
   * @param did - DID string to check
   * @returns true if active, false if deactivated
   */
  async isDidActive(did: string): Promise<boolean> {
    this.validateDidFormat(did);
    return this.callZome<boolean>('is_did_active', did);
  }

  /**
   * Get deactivation record for a DID
   *
   * @param did - DID string to check
   * @returns Deactivation record or null if not deactivated
   */
  async getDidDeactivation(did: string): Promise<DidDeactivation | null> {
    this.validateDidFormat(did);
    return this.callZome<DidDeactivation | null>('get_did_deactivation', did);
  }

  // ==========================================================================
  // DID UPDATES
  // ==========================================================================

  /**
   * Update a DID document
   *
   * @param input - Update input with fields to update
   * @returns Updated DID record
   * @throws IdentitySdkError if update fails
   */
  async updateDidDocument(input: UpdateDidInput): Promise<DidRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('update_did_document', input);
    return this.recordToDidRecord(record);
  }

  /**
   * Deactivate a DID (permanent - cannot be undone)
   *
   * @param reason - Reason for deactivation
   * @returns Deactivation record
   * @throws IdentitySdkError if deactivation fails
   */
  async deactivateDid(reason: string): Promise<DidDeactivation> {
    const record = await this.callZomeOnce<HolochainRecord>('deactivate_did', reason);
    return this.extractEntry<DidDeactivation>(record);
  }

  // ==========================================================================
  // SERVICE ENDPOINTS
  // ==========================================================================

  /**
   * Add a service endpoint to the DID document
   *
   * @param service - Service endpoint to add
   * @returns Updated DID record
   */
  async addServiceEndpoint(service: ServiceEndpoint): Promise<DidRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('add_service_endpoint', service);
    return this.recordToDidRecord(record);
  }

  /**
   * Remove a service endpoint from the DID document
   *
   * @param serviceId - ID of the service to remove
   * @returns Updated DID record
   */
  async removeServiceEndpoint(serviceId: string): Promise<DidRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('remove_service_endpoint', serviceId);
    return this.recordToDidRecord(record);
  }

  // ==========================================================================
  // VERIFICATION METHODS
  // ==========================================================================

  /**
   * Add a verification method to the DID document
   *
   * @param method - Verification method to add
   * @returns Updated DID record
   */
  async addVerificationMethod(method: VerificationMethod): Promise<DidRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('add_verification_method', method);
    return this.recordToDidRecord(record);
  }

  // ==========================================================================
  // PRIVATE HELPERS
  // ==========================================================================

  private validateDidFormat(did: string): void {
    if (!did.startsWith('did:mycelix:')) {
      throw new IdentitySdkError(
        IdentitySdkErrorCode.INVALID_DID_FORMAT,
        'DID must start with "did:mycelix:"',
        { did }
      );
    }
  }

  private recordToDidRecord(record: HolochainRecord): DidRecord {
    const document = this.extractEntry<DidDocument>(record);
    return {
      hash: this.getActionHash(record),
      document,
    };
  }
}
