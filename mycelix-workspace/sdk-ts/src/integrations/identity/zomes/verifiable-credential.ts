/**
 * Verifiable Credential Zome Client
 *
 * Client for W3C Verifiable Credentials in Mycelix-Identity.
 * Implements VC Data Model 2.0 specification.
 */

import { IdentitySdkError, IdentitySdkErrorCode } from '../types.js';

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
} from '../types.js';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

/**
 * Verifiable Credential Zome Client
 *
 * @example
 * ```typescript
 * const vcClient = new VerifiableCredentialClient(appClient, 'mycelix_identity');
 *
 * // Issue a credential
 * const credential = await vcClient.issueCredential({
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
 * const result = await vcClient.verifyCredential(credential.credential.id);
 * if (result.is_valid) {
 *   console.log('Credential is valid!');
 * }
 * ```
 */
export class VerifiableCredentialClient {
  private readonly roleName: string;
  private readonly zomeName = 'verifiable_credential';

  constructor(
    private readonly client: AppClient,
    roleName: string
  ) {
    this.roleName = roleName;
  }

  // ============================================================================
  // CREDENTIAL ISSUANCE
  // ============================================================================

  /**
   * Issue a new verifiable credential
   *
   * @param input - Credential issuance parameters
   * @returns Created credential record
   */
  async issueCredential(input: IssueCredentialInput): Promise<CredentialRecord> {
    const record = await this.call<HolochainRecord>('issue_credential', input);
    return this.recordToCredentialRecord(record);
  }

  /**
   * Get a credential by its ID
   *
   * @param credentialId - Unique credential identifier
   * @returns Credential record or null if not found
   */
  async getCredential(credentialId: string): Promise<CredentialRecord | null> {
    const record = await this.call<HolochainRecord | null>('get_credential', credentialId);
    return record ? this.recordToCredentialRecord(record) : null;
  }

  /**
   * Get all credentials issued by a specific DID
   *
   * @param issuerDid - DID of the issuer
   * @returns Array of credential records
   */
  async getCredentialsIssuedBy(issuerDid: string): Promise<CredentialRecord[]> {
    const records = await this.call<HolochainRecord[]>('get_credentials_issued_by', issuerDid);
    return records.map((r) => this.recordToCredentialRecord(r));
  }

  /**
   * Get all credentials for a specific subject
   *
   * @param subjectDid - DID of the credential subject
   * @returns Array of credential records
   */
  async getCredentialsForSubject(subjectDid: string): Promise<CredentialRecord[]> {
    const records = await this.call<HolochainRecord[]>('get_credentials_for_subject', subjectDid);
    return records.map((r) => this.recordToCredentialRecord(r));
  }

  /**
   * Get credentials issued by the current agent
   *
   * @returns Array of credential records
   */
  async getMyIssuedCredentials(): Promise<CredentialRecord[]> {
    const records = await this.call<HolochainRecord[]>('get_my_issued_credentials', null);
    return records.map((r) => this.recordToCredentialRecord(r));
  }

  /**
   * Get credentials held by the current agent
   *
   * @returns Array of credential records
   */
  async getMyCredentials(): Promise<CredentialRecord[]> {
    const records = await this.call<HolochainRecord[]>('get_my_credentials', null);
    return records.map((r) => this.recordToCredentialRecord(r));
  }

  // ============================================================================
  // CREDENTIAL VERIFICATION
  // ============================================================================

  /**
   * Verify a credential's validity
   *
   * Checks:
   * - Issuer signature
   * - Expiration status
   * - Revocation status
   * - Schema compliance
   *
   * @param credentialId - ID of credential to verify
   * @returns Verification result with detailed status
   */
  async verifyCredential(credentialId: string): Promise<VerificationResult> {
    return this.call<VerificationResult>('verify_credential', credentialId);
  }

  /**
   * Check if a credential is revoked
   *
   * @param credentialId - ID of credential to check
   * @returns true if revoked, false otherwise
   */
  async isCredentialRevoked(credentialId: string): Promise<boolean> {
    return this.call<boolean>('is_credential_revoked', credentialId);
  }

  /**
   * Get detailed credential status
   *
   * @param credentialId - ID of credential
   * @returns Status including revocation and expiration details
   */
  async getCredentialStatus(credentialId: string): Promise<CredentialStatusResponse> {
    return this.call<CredentialStatusResponse>('get_credential_status', credentialId);
  }

  // ============================================================================
  // PRESENTATIONS
  // ============================================================================

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
    const record = await this.call<HolochainRecord>('create_presentation', input);
    return this.recordToPresentationRecord(record);
  }

  // ============================================================================
  // SELECTIVE DISCLOSURE (DERIVED CREDENTIALS)
  // ============================================================================

  /**
   * Create a derived credential with selective disclosure
   *
   * Creates a new credential revealing only selected claims from
   * an original credential, with proof of derivation.
   *
   * @param input - Derivation parameters
   * @returns Created derived credential record
   */
  async createDerivedCredential(input: CreateDerivedInput): Promise<CredentialRecord> {
    const record = await this.call<HolochainRecord>('create_derived_credential', input);
    return this.recordToCredentialRecord(record);
  }

  // ============================================================================
  // CREDENTIAL REQUESTS
  // ============================================================================

  /**
   * Request a credential from an issuer
   *
   * Creates a request that the issuer can approve or reject.
   *
   * @param input - Request parameters
   * @returns Created request record
   */
  async requestCredential(input: RequestCredentialInput): Promise<RequestRecord> {
    const record = await this.call<HolochainRecord>('request_credential', input);
    return this.recordToRequestRecord(record);
  }

  /**
   * Get pending credential requests for an issuer
   *
   * @param issuerDid - DID of the issuer
   * @returns Array of pending request records
   */
  async getPendingRequests(issuerDid: string): Promise<RequestRecord[]> {
    const records = await this.call<HolochainRecord[]>('get_pending_requests', issuerDid);
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
    const record = await this.call<HolochainRecord>('update_request_status', input);
    return this.recordToRequestRecord(record);
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
        `Verifiable Credential zome call failed: ${message}`,
        { fnName, payload }
      );
    }
  }

  private recordToCredentialRecord(record: HolochainRecord): CredentialRecord {
    const credential = this.extractEntry<VerifiableCredential>(record);
    return {
      hash: record.signed_action.hashed.hash,
      credential,
    };
  }

  private recordToPresentationRecord(record: HolochainRecord): PresentationRecord {
    const presentation = this.extractEntry<VerifiablePresentation>(record);
    return {
      hash: record.signed_action.hashed.hash,
      presentation,
    };
  }

  private recordToRequestRecord(record: HolochainRecord): RequestRecord {
    const request = this.extractEntry<CredentialRequest>(record);
    return {
      hash: record.signed_action.hashed.hash,
      request,
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
