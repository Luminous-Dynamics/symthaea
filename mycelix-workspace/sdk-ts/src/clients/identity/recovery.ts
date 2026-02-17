/**
 * Recovery Client
 *
 * Client for social recovery in Mycelix-Identity.
 * Enables DID recovery through trusted trustees with threshold voting.
 *
 * @module @mycelix/sdk/clients/identity/recovery
 */

import { DEFAULT_IDENTITY_CLIENT_CONFIG } from './types.js';
import { ZomeClient } from '../../core/zome-client.js';

import type {
  RecoveryConfig,
  RecoveryRequest,
  RecoveryVote,
  SetupRecoveryInput,
  InitiateRecoveryInput,
  VoteOnRecoveryInput,
  IdentityClientConfig,
} from './types.js';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';


/**
 * Recovery record wrapper
 */
export interface RecoveryConfigRecord {
  hash: Uint8Array;
  config: RecoveryConfig;
}

/**
 * Recovery request record wrapper
 */
export interface RecoveryRequestRecord {
  hash: Uint8Array;
  request: RecoveryRequest;
}

/**
 * Recovery vote record wrapper
 */
export interface RecoveryVoteRecord {
  hash: Uint8Array;
  vote: RecoveryVote;
}

/**
 * Recovery Client
 *
 * Provides social recovery functionality for DID recovery through
 * trusted trustees with threshold-based voting.
 *
 * @example
 * ```typescript
 * const recoveryClient = new RecoveryClient(appClient);
 *
 * // Set up recovery with 3 trustees, requiring 2 approvals
 * await recoveryClient.setupRecovery({
 *   did: 'did:mycelix:alice123',
 *   trustees: [
 *     'did:mycelix:bob456',
 *     'did:mycelix:carol789',
 *     'did:mycelix:dave012',
 *   ],
 *   threshold: 2,
 *   time_lock: 604800, // 7 days
 * });
 *
 * // Initiate recovery (by a trustee)
 * const request = await recoveryClient.initiateRecovery({
 *   did: 'did:mycelix:alice123',
 *   initiator_did: 'did:mycelix:bob456',
 *   new_agent: newAgentPubKey,
 *   reason: 'Lost device',
 * });
 *
 * // Vote on recovery (by another trustee)
 * await recoveryClient.voteOnRecovery({
 *   request_id: request.request.id,
 *   trustee_did: 'did:mycelix:carol789',
 *   vote: 'Approve',
 *   comment: 'Verified via video call',
 * });
 * ```
 */
export class RecoveryClient extends ZomeClient {
  protected readonly zomeName = 'recovery';

  constructor(client: AppClient, config: Partial<IdentityClientConfig> = {}) {
    const mergedConfig = { ...DEFAULT_IDENTITY_CLIENT_CONFIG, ...config };
    super(client, {
      roleName: mergedConfig.roleName,
      timeout: mergedConfig.timeout,
    });
  }

  // ==========================================================================
  // RECOVERY CONFIGURATION
  // ==========================================================================

  /**
   * Set up recovery for a DID
   *
   * Configures social recovery with designated trustees and threshold.
   *
   * @param input - Recovery setup parameters
   * @returns Created recovery config record
   */
  async setupRecovery(input: SetupRecoveryInput): Promise<RecoveryConfigRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('setup_recovery', input);
    return this.recordToConfigRecord(record);
  }

  /**
   * Get recovery configuration for a DID
   *
   * @param did - DID to get recovery config for
   * @returns Recovery config record or null if not configured
   */
  async getRecoveryConfig(did: string): Promise<RecoveryConfigRecord | null> {
    const record = await this.callZome<HolochainRecord | null>('get_recovery_config', did);
    return record ? this.recordToConfigRecord(record) : null;
  }

  // ==========================================================================
  // RECOVERY REQUESTS
  // ==========================================================================

  /**
   * Initiate a recovery request
   *
   * Only trustees can initiate recovery. The initiator's vote is
   * automatically counted as an approval.
   *
   * @param input - Recovery initiation parameters
   * @returns Created recovery request record
   */
  async initiateRecovery(input: InitiateRecoveryInput): Promise<RecoveryRequestRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('initiate_recovery', input);
    return this.recordToRequestRecord(record);
  }

  /**
   * Vote on a recovery request
   *
   * Trustees vote to approve, reject, or abstain on recovery requests.
   *
   * @param input - Vote parameters
   * @returns Created vote record
   */
  async voteOnRecovery(input: VoteOnRecoveryInput): Promise<RecoveryVoteRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('vote_on_recovery', input);
    return this.recordToVoteRecord(record);
  }

  /**
   * Get all votes for a recovery request
   *
   * @param requestId - ID of the recovery request
   * @returns Array of vote records
   */
  async getRecoveryVotes(requestId: string): Promise<RecoveryVoteRecord[]> {
    const records = await this.callZome<HolochainRecord[]>('get_recovery_votes', requestId);
    return records.map((r) => this.recordToVoteRecord(r));
  }

  // ==========================================================================
  // RECOVERY EXECUTION
  // ==========================================================================

  /**
   * Execute an approved recovery
   *
   * Can only be executed after the time lock expires and threshold is met.
   *
   * @param requestId - ID of the approved recovery request
   * @returns Updated recovery request record
   */
  async executeRecovery(requestId: string): Promise<RecoveryRequestRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('execute_recovery', requestId);
    return this.recordToRequestRecord(record);
  }

  /**
   * Cancel a recovery request
   *
   * Only the DID owner can cancel a recovery request before execution.
   *
   * @param requestId - ID of the recovery request to cancel
   * @returns Updated recovery request record
   */
  async cancelRecovery(requestId: string): Promise<RecoveryRequestRecord> {
    const record = await this.callZomeOnce<HolochainRecord>('cancel_recovery', requestId);
    return this.recordToRequestRecord(record);
  }

  // ==========================================================================
  // TRUSTEE QUERIES
  // ==========================================================================

  /**
   * Get all DIDs for which this trustee has responsibilities
   *
   * Returns recovery configs where the given DID is a trustee.
   *
   * @param trusteeDid - DID of the trustee
   * @returns Array of recovery config records
   */
  async getTrusteeResponsibilities(trusteeDid: string): Promise<RecoveryConfigRecord[]> {
    const records = await this.callZome<HolochainRecord[]>('get_trustee_responsibilities', trusteeDid);
    return records.map((r) => this.recordToConfigRecord(r));
  }

  // ==========================================================================
  // PRIVATE HELPERS
  // ==========================================================================

  private recordToConfigRecord(record: HolochainRecord): RecoveryConfigRecord {
    const config = this.extractEntry<RecoveryConfig>(record);
    return {
      hash: this.getActionHash(record),
      config,
    };
  }

  private recordToRequestRecord(record: HolochainRecord): RecoveryRequestRecord {
    const request = this.extractEntry<RecoveryRequest>(record);
    return {
      hash: this.getActionHash(record),
      request,
    };
  }

  private recordToVoteRecord(record: HolochainRecord): RecoveryVoteRecord {
    const vote = this.extractEntry<RecoveryVote>(record);
    return {
      hash: this.getActionHash(record),
      vote,
    };
  }
}
