// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Verifiable Credential Client
 *
 * Client for W3C Verifiable Credentials in Mycelix-Identity.
 * Implements VC Data Model 2.0 specification with selective disclosure.
 *
 * @module @mycelix/sdk/clients/identity/credentials
 */

import { DEFAULT_IDENTITY_CLIENT_CONFIG } from './types.js';
import { ZomeClient } from '../../core/zome-client.js';

import type {
  VerifiableCredential,
  VerifiablePresentation,
  CredentialRequest,
  CredentialRecord,
  PresentationRecord,
  RequestRecord,
  VerificationResult,
  CredentialStatusResponse,
  IssueCredentialInput,
  CreatePresentationInput,
  CreateDerivedInput,
  RequestCredentialInput,
  UpdateRequestStatusInput,
  IdentityClientConfig,
} from './types.js';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';


/**
 * Verifiable Credential Client
 *
 * Provides comprehensive credential management including issuance,
 * verification, presentations, and selective disclosure.
 *
 * @example
 * ```typescript
 * const credentialsClient = new CredentialsClient(appClient);
 *
 * // Issue a credential
 * const credential = await credentialsClient.issueCredential({
 *   subject_did: 'did:mycelix:bob123',
 *   schema_id: 'mycelix:schema:employment',
 *   claims: {
 *     employer: 'Acme Corp',
 *     role: 'Engineer',
 *     startDate: '2024-01-15',
 *   },
 *   expiration_days: 365,
 * });
 *
 * // Verify a credential
 * const result = await credentialsClient.verifyCredential(credential.credential.id);
 * if (result.valid) {
 *   console.log('Credential is valid!');
 * }
 *
 * // Create selective disclosure
 * const derived = await credentialsClient.createDerivedCredential({
 *   credential_id: credential.credential.id,
 *   selected_claims: ['role'], // Only reveal role
 * });
 * ```
 */
export class CredentialsClient extends ZomeClient {
  protected readonly zomeName = 'verifiable_credential';

  constructor(client: AppClient, config: Partial<IdentityClientConfig> = {}) {
    const mergedConfig = { ...DEFAULT_IDENTITY_CLIENT_CONFIG, ...config };
    super(client, {
      roleName: mergedConfig.roleName,
      timeout: mergedConfig.timeout,
    });
  }

  // ==========================================================================
  // CREDENTIAL ISSUANCE
  // ==========================================================================

  /**
   * Issue a new verifiable credential
   *
   * Creates a W3C-compliant verifiable credential with cryptographic proof.
   *
   * @param input - Credential issuance parameters
   * @returns Created credential record
   */
  async issueCredential(input: IssueCredentialInput): Promise<CredentialRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('issue_credential', input);
    return this.recordToCredentialRecord(record);
  }

  /**
   * Get a credential by its ID
   *
   * @param credentialId - Unique credential identifier
   * @returns Credential record or null if not found
   */
  async getCredential(credentialId: string): Promise<CredentialRecord | null> {
    const record = await this.callZome<HolochainRecord | null>('get_credential', credentialId);
    return record ? this.recordToCredentialRecord(record) : null;
  }

  /**
   * Get all credentials issued by a specific DID
   *
   * @param issuerDid - DID of the issuer
   * @returns Array of credential records
   */
  async getCredentialsIssuedBy(issuerDid: string): Promise<CredentialRecord[]> {
    const records = await this.callZome<HolochainRecord[]>('get_credentials_issued_by', issuerDid);
    return records.map((r) => this.recordToCredentialRecord(r));
  }

  /**
   * Get all credentials for a specific subject
   *
   * @param subjectDid - DID of the credential subject
   * @returns Array of credential records
   */
  async getCredentialsForSubject(subjectDid: string): Promise<CredentialRecord[]> {
    const records = await this.callZome<HolochainRecord[]>('get_credentials_for_subject', subjectDid);
    return records.map((r) => this.recordToCredentialRecord(r));
  }

  /**
   * Get credentials issued by the current agent
   *
   * @returns Array of credential records
   */
  async getMyIssuedCredentials(): Promise<CredentialRecord[]> {
    const records = await this.callZome<HolochainRecord[]>('get_my_issued_credentials', null);
    return records.map((r) => this.recordToCredentialRecord(r));
  }

  /**
   * Get credentials held by the current agent
   *
   * @returns Array of credential records
   */
  async getMyCredentials(): Promise<CredentialRecord[]> {
    const records = await this.callZome<HolochainRecord[]>('get_my_credentials', null);
    return records.map((r) => this.recordToCredentialRecord(r));
  }

  // ==========================================================================
  // CREDENTIAL VERIFICATION
  // ==========================================================================

  /**
   * Verify a credential's validity
   *
   * Performs comprehensive verification including:
   * - Issuer signature verification
   * - Expiration status check
   * - Revocation status check
   * - Schema compliance verification
   *
   * @param credentialId - ID of credential to verify
   * @returns Verification result with detailed status
   */
  async verifyCredential(credentialId: string): Promise<VerificationResult> {
    return this.callZome<VerificationResult>('verify_credential', credentialId);
  }

  /**
   * Check if a credential is revoked
   *
   * @param credentialId - ID of credential to check
   * @returns true if revoked, false otherwise
   */
  async isCredentialRevoked(credentialId: string): Promise<boolean> {
    return this.callZome<boolean>('is_credential_revoked', credentialId);
  }

  /**
   * Get detailed credential status
   *
   * @param credentialId - ID of credential
   * @returns Status including revocation and expiration details
   */
  async getCredentialStatus(credentialId: string): Promise<CredentialStatusResponse> {
    return this.callZome<CredentialStatusResponse>('get_credential_status', credentialId);
  }

  // ==========================================================================
  // PRESENTATIONS
  // ==========================================================================

  /**
   * Create a verifiable presentation
   *
   * Bundles one or more credentials with a holder proof for
   * presenting to a verifier.
   *
   * @param input - Presentation creation parameters
   * @returns Created presentation record
   */
  async createPresentation(input: CreatePresentationInput): Promise<PresentationRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('create_presentation', input);
    return this.recordToPresentationRecord(record);
  }

  // ==========================================================================
  // SELECTIVE DISCLOSURE (DERIVED CREDENTIALS)
  // ==========================================================================

  /**
   * Create a derived credential with selective disclosure
   *
   * Creates a new credential revealing only selected claims from
   * an original credential, with cryptographic proof of derivation.
   *
   * @param input - Derivation parameters
   * @returns Created derived credential record
   */
  async createDerivedCredential(input: CreateDerivedInput): Promise<CredentialRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('create_derived_credential', input);
    return this.recordToCredentialRecord(record);
  }

  // ==========================================================================
  // CREDENTIAL REQUESTS
  // ==========================================================================

  /**
   * Request a credential from an issuer
   *
   * Creates a request that the issuer can review and approve/reject.
   *
   * @param input - Request parameters
   * @returns Created request record
   */
  async requestCredential(input: RequestCredentialInput): Promise<RequestRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('request_credential', input);
    return this.recordToRequestRecord(record);
  }

  /**
   * Get pending credential requests for an issuer
   *
   * @param issuerDid - DID of the issuer
   * @returns Array of pending request records
   */
  async getPendingRequests(issuerDid: string): Promise<RequestRecord[]> {
    const records = await this.callZome<HolochainRecord[]>('get_pending_requests', issuerDid);
    return records.map((r) => this.recordToRequestRecord(r));
  }

  /**
   * Update the status of a credential request
   *
   * Used by issuers to approve, reject, or mark requests as under review.
   *
   * @param input - Status update parameters
   * @returns Updated request record
   */
  async updateRequestStatus(input: UpdateRequestStatusInput): Promise<RequestRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('update_request_status', input);
    return this.recordToRequestRecord(record);
  }

  // ==========================================================================
  // PRIVATE HELPERS
  // ==========================================================================

  private recordToCredentialRecord(record: HolochainRecord): CredentialRecord {
    const credential = this.extractEntry<VerifiableCredential>(record);
    return {
      hash: this.getActionHash(record),
      credential,
    };
  }

  private recordToPresentationRecord(record: HolochainRecord): PresentationRecord {
    const presentation = this.extractEntry<VerifiablePresentation>(record);
    return {
      hash: this.getActionHash(record),
      presentation,
    };
  }

  private recordToRequestRecord(record: HolochainRecord): RequestRecord {
    const request = this.extractEntry<CredentialRequest>(record);
    return {
      hash: this.getActionHash(record),
      request,
    };
  }
}
