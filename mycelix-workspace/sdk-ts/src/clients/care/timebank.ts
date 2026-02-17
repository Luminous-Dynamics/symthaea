/**
 * Timebank Zome Client
 *
 * Handles service offers, requests, time exchanges, and balance tracking.
 *
 * @module @mycelix/sdk/clients/care/timebank
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  ServiceOffer,
  CreateServiceOfferInput,
  ServiceRequest,
  CreateServiceRequestInput,
  TimeExchange,
  CompleteExchangeInput,
  RateExchangeInput,
  TimeBalance,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface TimebankClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: TimebankClientConfig = {
  roleName: 'commons',
};

/**
 * Client for Timebank operations
 *
 * Manages service offers, requests, time exchanges, and balance tracking
 * within the care economy.
 */
export class TimebankClient extends ZomeClient {
  protected readonly zomeName = 'care_timebank';

  constructor(client: AppClient, config: TimebankClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Service Offers
  // ============================================================================

  async createServiceOffer(input: CreateServiceOfferInput): Promise<ServiceOffer> {
    const record = await this.callZomeOnce<HolochainRecord>('create_service_offer', {
      title: input.title,
      description: input.description,
      category: input.category,
      estimated_minutes: input.estimatedMinutes,
      tags: input.tags ?? [],
      location: input.location,
    });
    return this.mapServiceOffer(record);
  }

  async getOffersByCategory(category: string): Promise<ServiceOffer[]> {
    const records = await this.callZome<HolochainRecord[]>('get_offers_by_category', category);
    return records.map(r => this.mapServiceOffer(r));
  }

  async getMyOffers(): Promise<ServiceOffer[]> {
    const records = await this.callZome<HolochainRecord[]>('get_my_offers', null);
    return records.map(r => this.mapServiceOffer(r));
  }

  async updateOfferAvailability(offerId: ActionHash, available: boolean): Promise<ServiceOffer> {
    const record = await this.callZomeOnce<HolochainRecord>('update_offer_availability', {
      offer_id: offerId,
      available,
    });
    return this.mapServiceOffer(record);
  }

  // ============================================================================
  // Service Requests
  // ============================================================================

  async createServiceRequest(input: CreateServiceRequestInput): Promise<ServiceRequest> {
    const record = await this.callZomeOnce<HolochainRecord>('create_service_request', {
      title: input.title,
      description: input.description,
      category: input.category,
      urgency: input.urgency ?? 'Medium',
      preferred_time_start: input.preferredTimeStart,
      preferred_time_end: input.preferredTimeEnd,
      location: input.location,
    });
    return this.mapServiceRequest(record);
  }

  async getOpenRequests(category?: string): Promise<ServiceRequest[]> {
    const records = await this.callZome<HolochainRecord[]>('get_open_requests', category ?? null);
    return records.map(r => this.mapServiceRequest(r));
  }

  // ============================================================================
  // Time Exchanges
  // ============================================================================

  async completeExchange(input: CompleteExchangeInput): Promise<TimeExchange> {
    const record = await this.callZomeOnce<HolochainRecord>('complete_exchange', {
      offer_id: input.offerId,
      request_id: input.requestId,
      receiver_did: input.receiverDid,
      minutes: input.minutes,
      description: input.description,
    });
    return this.mapTimeExchange(record);
  }

  async rateExchange(input: RateExchangeInput): Promise<TimeExchange> {
    const record = await this.callZomeOnce<HolochainRecord>('rate_exchange', {
      exchange_id: input.exchangeId,
      rating: input.rating,
      comment: input.comment,
    });
    return this.mapTimeExchange(record);
  }

  // ============================================================================
  // Balance
  // ============================================================================

  async getMyBalance(): Promise<TimeBalance> {
    const result = await this.callZome<any>('get_my_balance', null);
    return {
      did: result.did,
      earned: result.earned,
      spent: result.spent,
      balance: result.balance,
      totalExchanges: result.total_exchanges,
      averageProviderRating: result.average_provider_rating,
      averageReceiverRating: result.average_receiver_rating,
    };
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private mapServiceOffer(record: HolochainRecord): ServiceOffer {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      providerDid: entry.provider_did,
      providerPubKey: entry.provider_pub_key,
      title: entry.title,
      description: entry.description,
      category: entry.category,
      estimatedMinutes: entry.estimated_minutes,
      tags: entry.tags,
      location: entry.location,
      available: entry.available,
      averageRating: entry.average_rating,
      completedCount: entry.completed_count,
      createdAt: entry.created_at,
      updatedAt: entry.updated_at,
    };
  }

  private mapServiceRequest(record: HolochainRecord): ServiceRequest {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      requesterDid: entry.requester_did,
      requesterPubKey: entry.requester_pub_key,
      title: entry.title,
      description: entry.description,
      category: entry.category,
      urgency: entry.urgency,
      preferredTimeStart: entry.preferred_time_start,
      preferredTimeEnd: entry.preferred_time_end,
      location: entry.location,
      status: entry.status,
      createdAt: entry.created_at,
      updatedAt: entry.updated_at,
    };
  }

  private mapTimeExchange(record: HolochainRecord): TimeExchange {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      offerId: entry.offer_id,
      requestId: entry.request_id,
      providerDid: entry.provider_did,
      receiverDid: entry.receiver_did,
      minutes: entry.minutes,
      description: entry.description,
      providerRating: entry.provider_rating,
      receiverRating: entry.receiver_rating,
      status: entry.status,
      completedAt: entry.completed_at,
      createdAt: entry.created_at,
    };
  }
}
