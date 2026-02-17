/**
 * Revocation Client
 *
 * Client for credential revocation management in Mycelix-Identity.
 * Enables credential revocation, suspension, and status checking.
 *
 * @module @mycelix/sdk/clients/identity/revocation
 */

import { DEFAULT_IDENTITY_CLIENT_CONFIG } from './types.js';
import { ZomeClient } from '../../core/zome-client.js';

import type {
  RevocationEntry,
  RevocationCheckResult,
  RevokeCredentialInput,
  SuspendCredentialInput,
  ReinstateCredentialInput,
  IdentityClientConfig,
} from './types.js';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';


/**
 * Revocation record wrapper
 */
export interface RevocationRecord {
  hash: Uint8Array;
  entry: RevocationEntry;
}

/**
 * Revocation Client
 *
 * Provides credential revocation management including permanent revocation,
 * temporary suspension, and status checking.
 *
 * @example
 * ```typescript
 * const revocationClient = new RevocationClient(appClient);
 *
 * // Revoke a credential permanently
 * await revocationClient.revokeCredential({
 *   credential_id: 'urn:uuid:credential123',
 *   issuer_did: 'did:mycelix:issuer456',
 *   reason: 'Fraudulent claims detected',
 * });
 *
 * // Suspend a credential temporarily
 * const suspensionEnd = Date.now() + (30 * 24 * 60 * 60 * 1000); // 30 days
 * await revocationClient.suspendCredential({
 *   credential_id: 'urn:uuid:credential789',
 *   issuer_did: 'did:mycelix:issuer456',
 *   reason: 'Under investigation',
 *   suspension_end: suspensionEnd,
 * });
 *
 * // Check revocation status
 * const status = await revocationClient.checkRevocationStatus('urn:uuid:credential123');
 * console.log(`Status: ${status.status}`);
 *
 * // Batch check multiple credentials
 * const statuses = await revocationClient.batchCheckRevocation([
 *   'urn:uuid:credential123',
 *   'urn:uuid:credential456',
 *   'urn:uuid:credential789',
 * ]);
 * ```
 */
export class RevocationClient extends ZomeClient {
  protected readonly zomeName = 'revocation';

  constructor(client: AppClient, config: Partial<IdentityClientConfig> = {}) {
    const mergedConfig = { ...DEFAULT_IDENTITY_CLIENT_CONFIG, ...config };
    super(client, {
      roleName: mergedConfig.roleName,
      timeout: mergedConfig.timeout,
    });
  }

  // ==========================================================================
  // REVOCATION MANAGEMENT
  // ==========================================================================

  /**
   * Revoke a credential permanently
   *
   * Once revoked, a credential cannot be reinstated.
   *
   * @param input - Revocation parameters
   * @returns Created revocation record
   */
  async revokeCredential(input: RevokeCredentialInput): Promise<RevocationRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('revoke_credential', input);
    return this.recordToRevocationRecord(record);
  }

  /**
   * Suspend a credential temporarily
   *
   * Suspended credentials can be reinstated before the suspension end time.
   *
   * @param input - Suspension parameters
   * @returns Created suspension record
   */
  async suspendCredential(input: SuspendCredentialInput): Promise<RevocationRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('suspend_credential', input);
    return this.recordToRevocationRecord(record);
  }

  /**
   * Reinstate a suspended credential
   *
   * Only suspended credentials can be reinstated.
   *
   * @param input - Reinstatement parameters
   * @returns Updated revocation record
   */
  async reinstateCredential(input: ReinstateCredentialInput): Promise<RevocationRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('reinstate_credential', input);
    return this.recordToRevocationRecord(record);
  }

  // ==========================================================================
  // STATUS CHECKING
  // ==========================================================================

  /**
   * Check revocation status of a credential
   *
   * Returns the current status including whether the credential is
   * active, revoked, or suspended.
   *
   * @param credentialId - ID of the credential to check
   * @returns Revocation check result
   */
  async checkRevocationStatus(credentialId: string): Promise<RevocationCheckResult> {
    return this.callZome<RevocationCheckResult>('check_revocation_status', credentialId);
  }

  /**
   * Batch check revocation status of multiple credentials
   *
   * Efficiently checks multiple credentials in a single call.
   *
   * @param credentialIds - Array of credential IDs to check
   * @returns Array of revocation check results
   */
  async batchCheckRevocation(credentialIds: string[]): Promise<RevocationCheckResult[]> {
    return this.callZome<RevocationCheckResult[]>('batch_check_revocation', credentialIds);
  }

  // ==========================================================================
  // QUERIES
  // ==========================================================================

  /**
   * Get all revocations by issuer
   *
   * @param issuerDid - DID of the issuer
   * @returns Array of revocation records
   */
  async getRevocationsByIssuer(issuerDid: string): Promise<RevocationRecord[]> {
    const records = await this.callZome<HolochainRecord[]>('get_revocations_by_issuer', issuerDid);
    return records.map((r) => this.recordToRevocationRecord(r));
  }

  // ==========================================================================
  // PRIVATE HELPERS
  // ==========================================================================

  private recordToRevocationRecord(record: HolochainRecord): RevocationRecord {
    const entry = this.extractEntry<RevocationEntry>(record);
    return {
      hash: this.getActionHash(record),
      entry,
    };
  }
}
