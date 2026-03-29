// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * DID Registry Zome Client
 *
 * Client for DID document management in Mycelix-Identity.
 * Implements W3C DID Core Specification (did:mycelix method).
 */

import { IdentitySdkError, IdentitySdkErrorCode } from '../types.js';

import type {
  DidDocument,
  DidDeactivation,
  DidRecord,
  VerificationMethod,
  ServiceEndpoint,
  UpdateDidInput,
} from '../types.js';
import type { AppClient, AgentPubKey, Record as HolochainRecord } from '@holochain/client';

/**
 * DID Registry Zome Client
 *
 * @example
 * ```typescript
 * const didClient = new DidRegistryClient(appClient, 'mycelix_identity');
 *
 * // Create a DID for the current agent
 * const myDid = await didClient.createDid();
 * console.log(`Created DID: ${myDid.document.id}`);
 *
 * // Resolve a DID
 * const resolved = await didClient.resolveDid('did:mycelix:abc123...');
 * ```
 */
export class DidRegistryClient {
  private readonly roleName: string;
  private readonly zomeName = 'did_registry';

  constructor(
    private readonly client: AppClient,
    roleName: string
  ) {
    this.roleName = roleName;
  }

  /**
   * Create a new DID for the current agent
   *
   * Creates a DID document linked to the agent's public key.
   * The DID format is: did:mycelix:<agent_pub_key_base64>
   *
   * @returns Created DID record with document
   */
  async createDid(): Promise<DidRecord> {
    const record = await this.call<HolochainRecord>('create_did', null);
    return this.recordToDidRecord(record);
  }

  /**
   * Get the current agent's DID document
   *
   * @returns DID record or null if none exists
   */
  async getMyDid(): Promise<DidRecord | null> {
    const record = await this.call<HolochainRecord | null>('get_my_did', null);
    return record ? this.recordToDidRecord(record) : null;
  }

  /**
   * Get DID document for a specific agent
   *
   * @param agentPubKey - Public key of the agent
   * @returns DID record or null if not found
   */
  async getDidDocument(agentPubKey: AgentPubKey): Promise<DidRecord | null> {
    const record = await this.call<HolochainRecord | null>('get_did_document', agentPubKey);
    return record ? this.recordToDidRecord(record) : null;
  }

  /**
   * Resolve a DID string to its document
   *
   * @param did - DID string (e.g., "did:mycelix:...")
   * @returns DID record or null if not found
   */
  async resolveDid(did: string): Promise<DidRecord | null> {
    if (!did.startsWith('did:mycelix:')) {
      throw new IdentitySdkError(
        IdentitySdkErrorCode.INVALID_DID_FORMAT,
        'DID must start with "did:mycelix:"',
        { did }
      );
    }
    const record = await this.call<HolochainRecord | null>('resolve_did', did);
    return record ? this.recordToDidRecord(record) : null;
  }

  /**
   * Check if a DID is active (not deactivated)
   *
   * @param did - DID string to check
   * @returns true if active, false if deactivated
   */
  async isDidActive(did: string): Promise<boolean> {
    return this.call<boolean>('is_did_active', did);
  }

  /**
   * Update a DID document
   *
   * @param input - Update input with original hash and updates
   * @returns Updated DID record
   */
  async updateDidDocument(input: UpdateDidInput): Promise<DidRecord> {
    const record = await this.call<HolochainRecord>('update_did_document', input);
    return this.recordToDidRecord(record);
  }

  /**
   * Deactivate a DID (permanent - cannot be undone)
   *
   * @param reason - Reason for deactivation
   * @returns Deactivation record
   */
  async deactivateDid(reason: string): Promise<DidDeactivation> {
    const record = await this.call<HolochainRecord>('deactivate_did', reason);
    return this.extractEntry<DidDeactivation>(record);
  }

  /**
   * Get deactivation record for a DID
   *
   * @param did - DID string to check
   * @returns Deactivation record or null if not deactivated
   */
  async getDidDeactivation(did: string): Promise<DidDeactivation | null> {
    return this.call<DidDeactivation | null>('get_did_deactivation', did);
  }

  /**
   * Add a service endpoint to the DID document
   *
   * @param service - Service endpoint to add
   * @returns Updated DID record
   */
  async addServiceEndpoint(service: ServiceEndpoint): Promise<DidRecord> {
    const record = await this.call<HolochainRecord>('add_service_endpoint', service);
    return this.recordToDidRecord(record);
  }

  /**
   * Remove a service endpoint from the DID document
   *
   * @param serviceId - ID of the service to remove
   * @returns Updated DID record
   */
  async removeServiceEndpoint(serviceId: string): Promise<DidRecord> {
    const record = await this.call<HolochainRecord>('remove_service_endpoint', serviceId);
    return this.recordToDidRecord(record);
  }

  /**
   * Add a verification method to the DID document
   *
   * @param method - Verification method to add
   * @returns Updated DID record
   */
  async addVerificationMethod(method: VerificationMethod): Promise<DidRecord> {
    const record = await this.call<HolochainRecord>('add_verification_method', method);
    return this.recordToDidRecord(record);
  }

  // ============================================================================
  // PRIVATE HELPERS
  // ============================================================================

  private async call<T>(fnName: string, payload: unknown): Promise<T> {
    try {
      const result = await this.client.callZome({
        role_name: this.roleName,
        zome_name: this.zomeName,
        fn_name: fnName,
        payload,
      });
      return result as T;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new IdentitySdkError(
        IdentitySdkErrorCode.ZOME_CALL_FAILED,
        `DID Registry zome call failed: ${message}`,
        { fnName, payload }
      );
    }
  }

  private recordToDidRecord(record: HolochainRecord): DidRecord {
    const document = this.extractEntry<DidDocument>(record);
    return {
      hash: record.signed_action.hashed.hash,
      document,
    };
  }

  private extractEntry<T>(record: HolochainRecord): T {
    if (!record.entry || !('Present' in record.entry)) {
      throw new IdentitySdkError(
        IdentitySdkErrorCode.INVALID_INPUT,
        'Record does not contain app entry'
      );
    }
    // Holochain returns entry as { Present: entry_data }
    // The entry is already decoded by the client
    return (record.entry as { Present: T }).Present;
  }
}
