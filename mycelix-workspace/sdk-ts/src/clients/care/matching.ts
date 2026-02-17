/**
 * Matching Zome Client
 *
 * Handles care matching between service requests and offers.
 *
 * @module @mycelix/sdk/clients/care/matching
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type { CareMatch } from './types';
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface MatchingClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: MatchingClientConfig = {
  roleName: 'commons',
};

/**
 * Client for Care Matching operations
 */
export class MatchingClient extends ZomeClient {
  protected readonly zomeName = 'care_matching';

  constructor(client: AppClient, config: MatchingClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  async findMatchesForRequest(requestId: ActionHash): Promise<CareMatch[]> {
    const records = await this.callZome<HolochainRecord[]>('find_matches_for_request', requestId);
    return records.map(r => this.mapMatch(r));
  }

  async suggestMatch(requestId: ActionHash, offerId: ActionHash, reason: string): Promise<CareMatch> {
    const record = await this.callZomeOnce<HolochainRecord>('suggest_match', {
      request_id: requestId,
      offer_id: offerId,
      reason,
    });
    return this.mapMatch(record);
  }

  async acceptMatch(matchId: ActionHash): Promise<CareMatch> {
    const record = await this.callZomeOnce<HolochainRecord>('accept_match', matchId);
    return this.mapMatch(record);
  }

  async declineMatch(matchId: ActionHash): Promise<CareMatch> {
    const record = await this.callZomeOnce<HolochainRecord>('decline_match', matchId);
    return this.mapMatch(record);
  }

  async getMatchesForOffer(offerId: ActionHash): Promise<CareMatch[]> {
    const records = await this.callZome<HolochainRecord[]>('get_matches_for_offer', offerId);
    return records.map(r => this.mapMatch(r));
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private mapMatch(record: HolochainRecord): CareMatch {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      requestId: entry.request_id,
      offerId: entry.offer_id,
      requesterDid: entry.requester_did,
      providerDid: entry.provider_did,
      confidence: entry.confidence,
      reason: entry.reason,
      status: entry.status,
      createdAt: entry.created_at,
    };
  }
}
