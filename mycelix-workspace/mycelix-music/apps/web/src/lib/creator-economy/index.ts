// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Creator Economy
 *
 * Monetization and analytics:
 * - Subscription tiers
 * - Royalty tracking
 * - Revenue analytics
 * - Payout management
 * - NFT integration
 * - Merchandise
 */

// ==================== Types ====================

export interface SubscriptionTier {
  id: string;
  name: string;
  price: number;
  currency: string;
  interval: 'month' | 'year';
  benefits: string[];
  maxSubscribers?: number;
  isActive: boolean;
}

export interface Subscription {
  id: string;
  tier: SubscriptionTier;
  subscriber: { id: string; name: string; avatar: string };
  status: 'active' | 'canceled' | 'past_due' | 'paused';
  currentPeriodStart: Date;
  currentPeriodEnd: Date;
  cancelAtPeriodEnd: boolean;
}

export interface RoyaltyPayment {
  id: string;
  trackId: string;
  trackTitle: string;
  platform: 'streaming' | 'download' | 'sync' | 'performance';
  streams: number;
  amount: number;
  currency: string;
  period: { start: Date; end: Date };
  status: 'pending' | 'paid' | 'disputed';
  paidAt?: Date;
}

export interface RoyaltySplit {
  userId: string;
  userName: string;
  role: 'artist' | 'producer' | 'writer' | 'label' | 'publisher';
  percentage: number;
}

export interface Payout {
  id: string;
  amount: number;
  currency: string;
  method: 'bank_transfer' | 'paypal' | 'stripe' | 'crypto';
  status: 'pending' | 'processing' | 'completed' | 'failed';
  createdAt: Date;
  processedAt?: Date;
  breakdown: {
    streaming: number;
    subscriptions: number;
    tips: number;
    merchandise: number;
    sync: number;
  };
}

export interface AnalyticsPeriod {
  start: Date;
  end: Date;
  granularity: 'hour' | 'day' | 'week' | 'month';
}

export interface StreamingAnalytics {
  totalStreams: number;
  uniqueListeners: number;
  totalMinutes: number;
  revenue: number;
  topTracks: Array<{ trackId: string; title: string; streams: number; revenue: number }>;
  topCountries: Array<{ country: string; streams: number; percentage: number }>;
  topPlatforms: Array<{ platform: string; streams: number; percentage: number }>;
  demographics: {
    ageGroups: Array<{ range: string; percentage: number }>;
    gender: Array<{ type: string; percentage: number }>;
  };
  timeline: Array<{ date: string; streams: number; revenue: number }>;
}

export interface AudienceInsights {
  totalFollowers: number;
  newFollowers: number;
  unfollowers: number;
  engagementRate: number;
  topListeners: Array<{ userId: string; name: string; streams: number; playlists: number }>;
  listenerRetention: Array<{ week: string; percentage: number }>;
  playlistAdditions: number;
  shares: number;
  saves: number;
}

export interface Tip {
  id: string;
  from: { id: string; name: string; avatar: string };
  amount: number;
  currency: string;
  message?: string;
  trackId?: string;
  createdAt: Date;
}

export interface MerchandiseItem {
  id: string;
  name: string;
  description: string;
  images: string[];
  price: number;
  currency: string;
  variants: Array<{ id: string; name: string; stock: number }>;
  category: 'clothing' | 'accessories' | 'vinyl' | 'digital' | 'other';
  isActive: boolean;
  sales: number;
}

export interface NFTRelease {
  id: string;
  trackId: string;
  title: string;
  description: string;
  artwork: string;
  supply: number;
  minted: number;
  price: number;
  currency: 'ETH' | 'SOL' | 'MATIC';
  blockchain: 'ethereum' | 'solana' | 'polygon';
  contractAddress: string;
  royaltyPercentage: number;
  benefits: string[];
  status: 'draft' | 'minting' | 'live' | 'sold_out';
}

// ==================== Subscription Service ====================

export class SubscriptionService {
  private baseUrl: string;

  constructor(baseUrl = '/api') {
    this.baseUrl = baseUrl;
  }

  // Creator-side
  async getTiers(): Promise<SubscriptionTier[]> {
    const response = await fetch(`${this.baseUrl}/creator/subscription-tiers`);
    return response.json();
  }

  async createTier(tier: Omit<SubscriptionTier, 'id' | 'isActive'>): Promise<SubscriptionTier> {
    const response = await fetch(`${this.baseUrl}/creator/subscription-tiers`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(tier),
    });
    return response.json();
  }

  async updateTier(tierId: string, updates: Partial<SubscriptionTier>): Promise<SubscriptionTier> {
    const response = await fetch(`${this.baseUrl}/creator/subscription-tiers/${tierId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    return response.json();
  }

  async getSubscribers(tierId?: string): Promise<Subscription[]> {
    const params = tierId ? `?tier=${tierId}` : '';
    const response = await fetch(`${this.baseUrl}/creator/subscribers${params}`);
    return response.json();
  }

  async getSubscriberStats(): Promise<{
    total: number;
    active: number;
    mrr: number;
    churnRate: number;
    growth: number;
  }> {
    const response = await fetch(`${this.baseUrl}/creator/subscribers/stats`);
    return response.json();
  }

  // Fan-side
  async subscribeToCreator(creatorId: string, tierId: string): Promise<Subscription> {
    const response = await fetch(`${this.baseUrl}/subscriptions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ creatorId, tierId }),
    });
    return response.json();
  }

  async cancelSubscription(subscriptionId: string): Promise<void> {
    await fetch(`${this.baseUrl}/subscriptions/${subscriptionId}`, { method: 'DELETE' });
  }

  async getMySubscriptions(): Promise<Subscription[]> {
    const response = await fetch(`${this.baseUrl}/subscriptions/mine`);
    return response.json();
  }
}

// ==================== Royalty Service ====================

export class RoyaltyService {
  private baseUrl: string;

  constructor(baseUrl = '/api') {
    this.baseUrl = baseUrl;
  }

  async getRoyalties(period?: AnalyticsPeriod): Promise<RoyaltyPayment[]> {
    const params = period ? `?start=${period.start.toISOString()}&end=${period.end.toISOString()}` : '';
    const response = await fetch(`${this.baseUrl}/creator/royalties${params}`);
    return response.json();
  }

  async getRoyaltySummary(): Promise<{
    pending: number;
    thisMonth: number;
    lastMonth: number;
    allTime: number;
    currency: string;
  }> {
    const response = await fetch(`${this.baseUrl}/creator/royalties/summary`);
    return response.json();
  }

  async getTrackRoyalties(trackId: string): Promise<RoyaltyPayment[]> {
    const response = await fetch(`${this.baseUrl}/creator/tracks/${trackId}/royalties`);
    return response.json();
  }

  async setSplits(trackId: string, splits: RoyaltySplit[]): Promise<void> {
    await fetch(`${this.baseUrl}/creator/tracks/${trackId}/splits`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ splits }),
    });
  }

  async getSplits(trackId: string): Promise<RoyaltySplit[]> {
    const response = await fetch(`${this.baseUrl}/creator/tracks/${trackId}/splits`);
    return response.json();
  }
}

// ==================== Analytics Service ====================

export class AnalyticsService {
  private baseUrl: string;

  constructor(baseUrl = '/api') {
    this.baseUrl = baseUrl;
  }

  async getStreamingAnalytics(period: AnalyticsPeriod): Promise<StreamingAnalytics> {
    const params = new URLSearchParams({
      start: period.start.toISOString(),
      end: period.end.toISOString(),
      granularity: period.granularity,
    });
    const response = await fetch(`${this.baseUrl}/creator/analytics/streaming?${params}`);
    return response.json();
  }

  async getAudienceInsights(period: AnalyticsPeriod): Promise<AudienceInsights> {
    const params = new URLSearchParams({
      start: period.start.toISOString(),
      end: period.end.toISOString(),
    });
    const response = await fetch(`${this.baseUrl}/creator/analytics/audience?${params}`);
    return response.json();
  }

  async getTrackAnalytics(trackId: string, period: AnalyticsPeriod): Promise<{
    streams: number;
    uniqueListeners: number;
    saves: number;
    playlistAdds: number;
    skipRate: number;
    completionRate: number;
    peakListeners: { time: string; count: number };
    timeline: Array<{ date: string; streams: number }>;
  }> {
    const params = new URLSearchParams({
      start: period.start.toISOString(),
      end: period.end.toISOString(),
    });
    const response = await fetch(`${this.baseUrl}/creator/tracks/${trackId}/analytics?${params}`);
    return response.json();
  }

  async getRevenueBreakdown(period: AnalyticsPeriod): Promise<{
    total: number;
    streaming: number;
    subscriptions: number;
    tips: number;
    merchandise: number;
    sync: number;
    nft: number;
    timeline: Array<{ date: string; amount: number; source: string }>;
  }> {
    const params = new URLSearchParams({
      start: period.start.toISOString(),
      end: period.end.toISOString(),
    });
    const response = await fetch(`${this.baseUrl}/creator/analytics/revenue?${params}`);
    return response.json();
  }

  async getRealTimeListeners(): Promise<{
    current: number;
    peak: number;
    byTrack: Array<{ trackId: string; title: string; listeners: number }>;
    byCountry: Array<{ country: string; listeners: number }>;
  }> {
    const response = await fetch(`${this.baseUrl}/creator/analytics/realtime`);
    return response.json();
  }
}

// ==================== Payout Service ====================

export class PayoutService {
  private baseUrl: string;

  constructor(baseUrl = '/api') {
    this.baseUrl = baseUrl;
  }

  async getPayouts(): Promise<Payout[]> {
    const response = await fetch(`${this.baseUrl}/creator/payouts`);
    return response.json();
  }

  async getPayoutSettings(): Promise<{
    method: string;
    threshold: number;
    schedule: 'weekly' | 'biweekly' | 'monthly';
    accountDetails: Record<string, string>;
  }> {
    const response = await fetch(`${this.baseUrl}/creator/payouts/settings`);
    return response.json();
  }

  async updatePayoutSettings(settings: {
    method?: string;
    threshold?: number;
    schedule?: string;
    accountDetails?: Record<string, string>;
  }): Promise<void> {
    await fetch(`${this.baseUrl}/creator/payouts/settings`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    });
  }

  async requestPayout(): Promise<Payout> {
    const response = await fetch(`${this.baseUrl}/creator/payouts/request`, { method: 'POST' });
    return response.json();
  }

  async getBalance(): Promise<{
    available: number;
    pending: number;
    currency: string;
  }> {
    const response = await fetch(`${this.baseUrl}/creator/balance`);
    return response.json();
  }
}

// ==================== Tips Service ====================

export class TipsService {
  private baseUrl: string;

  constructor(baseUrl = '/api') {
    this.baseUrl = baseUrl;
  }

  async getTips(): Promise<Tip[]> {
    const response = await fetch(`${this.baseUrl}/creator/tips`);
    return response.json();
  }

  async getTipSettings(): Promise<{
    enabled: boolean;
    suggestedAmounts: number[];
    thankYouMessage: string;
  }> {
    const response = await fetch(`${this.baseUrl}/creator/tips/settings`);
    return response.json();
  }

  async updateTipSettings(settings: {
    enabled?: boolean;
    suggestedAmounts?: number[];
    thankYouMessage?: string;
  }): Promise<void> {
    await fetch(`${this.baseUrl}/creator/tips/settings`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    });
  }

  // Fan-side
  async sendTip(creatorId: string, amount: number, message?: string, trackId?: string): Promise<Tip> {
    const response = await fetch(`${this.baseUrl}/tips`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ creatorId, amount, message, trackId }),
    });
    return response.json();
  }
}

// ==================== Merchandise Service ====================

export class MerchandiseService {
  private baseUrl: string;

  constructor(baseUrl = '/api') {
    this.baseUrl = baseUrl;
  }

  async getItems(): Promise<MerchandiseItem[]> {
    const response = await fetch(`${this.baseUrl}/creator/merchandise`);
    return response.json();
  }

  async createItem(item: Omit<MerchandiseItem, 'id' | 'sales'>): Promise<MerchandiseItem> {
    const response = await fetch(`${this.baseUrl}/creator/merchandise`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(item),
    });
    return response.json();
  }

  async updateItem(itemId: string, updates: Partial<MerchandiseItem>): Promise<MerchandiseItem> {
    const response = await fetch(`${this.baseUrl}/creator/merchandise/${itemId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    return response.json();
  }

  async deleteItem(itemId: string): Promise<void> {
    await fetch(`${this.baseUrl}/creator/merchandise/${itemId}`, { method: 'DELETE' });
  }

  async getOrders(): Promise<Array<{
    id: string;
    item: MerchandiseItem;
    variant: string;
    quantity: number;
    total: number;
    status: string;
    createdAt: Date;
    customer: { name: string; email: string; address: string };
  }>> {
    const response = await fetch(`${this.baseUrl}/creator/merchandise/orders`);
    return response.json();
  }
}

// ==================== NFT Service ====================

export class NFTService {
  private baseUrl: string;

  constructor(baseUrl = '/api') {
    this.baseUrl = baseUrl;
  }

  async getReleases(): Promise<NFTRelease[]> {
    const response = await fetch(`${this.baseUrl}/creator/nfts`);
    return response.json();
  }

  async createRelease(release: Omit<NFTRelease, 'id' | 'minted' | 'contractAddress' | 'status'>): Promise<NFTRelease> {
    const response = await fetch(`${this.baseUrl}/creator/nfts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(release),
    });
    return response.json();
  }

  async launchRelease(releaseId: string): Promise<NFTRelease> {
    const response = await fetch(`${this.baseUrl}/creator/nfts/${releaseId}/launch`, { method: 'POST' });
    return response.json();
  }

  async getHolders(releaseId: string): Promise<Array<{
    wallet: string;
    tokenId: number;
    acquiredAt: Date;
  }>> {
    const response = await fetch(`${this.baseUrl}/creator/nfts/${releaseId}/holders`);
    return response.json();
  }

  // Fan-side
  async mint(releaseId: string): Promise<{ transactionHash: string; tokenId: number }> {
    const response = await fetch(`${this.baseUrl}/nfts/${releaseId}/mint`, { method: 'POST' });
    return response.json();
  }

  async getMyNFTs(): Promise<Array<NFTRelease & { tokenId: number }>> {
    const response = await fetch(`${this.baseUrl}/nfts/mine`);
    return response.json();
  }
}

// ==================== Creator Economy Manager ====================

export class CreatorEconomyManager {
  public readonly subscriptions: SubscriptionService;
  public readonly royalties: RoyaltyService;
  public readonly analytics: AnalyticsService;
  public readonly payouts: PayoutService;
  public readonly tips: TipsService;
  public readonly merchandise: MerchandiseService;
  public readonly nfts: NFTService;

  constructor(baseUrl = '/api') {
    this.subscriptions = new SubscriptionService(baseUrl);
    this.royalties = new RoyaltyService(baseUrl);
    this.analytics = new AnalyticsService(baseUrl);
    this.payouts = new PayoutService(baseUrl);
    this.tips = new TipsService(baseUrl);
    this.merchandise = new MerchandiseService(baseUrl);
    this.nfts = new NFTService(baseUrl);
  }
}

// ==================== Singleton ====================

let creatorEconomyManager: CreatorEconomyManager | null = null;

export function getCreatorEconomyManager(baseUrl?: string): CreatorEconomyManager {
  if (!creatorEconomyManager) {
    creatorEconomyManager = new CreatorEconomyManager(baseUrl);
  }
  return creatorEconomyManager;
}

export default {
  CreatorEconomyManager,
  getCreatorEconomyManager,
};
