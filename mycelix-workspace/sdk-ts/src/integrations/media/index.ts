// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Media Integration
 *
 * hApp-specific adapter for Mycelix-Media providing:
 * - Decentralized content publishing with attribution
 * - Content moderation with community governance
 * - Royalty distribution for collaborative works
 * - Cross-hApp content licensing via Bridge
 * - Reputation-based content curation
 *
 * @packageDocumentation
 * @module integrations/media
 */

import { LocalBridge } from '../../bridge/index.js';
import {
  createReputation,
  recordPositive,
  recordNegative,
  type ReputationScore,
} from '../../matl/index.js';
import { MediaValidators } from '../../utils/validation.js';

// ============================================================================
// Media-Specific Types
// ============================================================================

/** Content type categories (matches Rust ContentType in media-publication integrity) */
export type ContentType =
  | 'Article'
  | 'Opinion'
  | 'Investigation'
  | 'Review'
  | 'Analysis'
  | 'Interview'
  | 'Report'
  | 'Editorial'
  | 'Other';

/** License type (matches Rust LicenseType in media-publication integrity) */
export type LicenseType =
  | 'CC0'
  | 'CCBY'
  | 'CCBYSA'
  | 'CCBYNC'
  | 'CCBYNCSA'
  | 'AllRightsReserved'
  | 'Custom';

/** License record (matches Rust License struct in media-publication integrity) */
export interface LicenseRecord {
  license_type: LicenseType;
  attribution_required: boolean;
  commercial_use: boolean;
  derivative_works: boolean;
}

/** Content status */
export type ContentStatus = 'draft' | 'published' | 'flagged' | 'hidden' | 'archived';

/** Moderation action */
export type ModerationAction = 'approve' | 'flag' | 'hide' | 'escalate';

/** Content record */
export interface Content {
  id: string;
  type: ContentType;
  title: string;
  description: string;
  creatorId: string;
  contributors: Contributor[];
  contentHash: string;
  storageRef: string;
  license: LicenseType;
  status: ContentStatus;
  views: number;
  likes: number;
  shares: number;
  tags: string[];
  publishedAt?: number;
  createdAt: number;
  updatedAt: number;
}

/** Contributor with royalty share */
export interface Contributor {
  did: string;
  role: string;
  royaltyShare: number; // 0-100
  addedAt: number;
}

/** Content creator profile */
export interface CreatorProfile {
  did: string;
  displayName: string;
  bio?: string;
  reputation: ReputationScore;
  contentCount: number;
  totalViews: number;
  totalEarnings: number;
  followers: number;
  following: number;
  joinedAt: number;
}

/** Moderation report */
export interface ModerationReport {
  id: string;
  contentId: string;
  reporterId: string;
  reason: string;
  category: 'spam' | 'harmful' | 'copyright' | 'misinformation' | 'other';
  status: 'pending' | 'reviewed' | 'resolved';
  action?: ModerationAction;
  reviewerId?: string;
  createdAt: number;
  resolvedAt?: number;
}

/** Royalty payment */
export interface RoyaltyPayment {
  id: string;
  contentId: string;
  recipientId: string;
  amount: number;
  currency: string;
  reason: 'view' | 'license' | 'tip' | 'subscription';
  timestamp: number;
}

/** Content license grant */
export interface LicenseGrant {
  id: string;
  contentId: string;
  granteeId: string;
  license: LicenseType;
  customTerms?: string;
  fee?: number;
  expiresAt?: number;
  grantedAt: number;
}

// ============================================================================
// Media Service
// ============================================================================

/**
 * Media service for content management and distribution
 */
export class MediaService {
  private content = new Map<string, Content>();
  private creators = new Map<string, CreatorProfile>();
  private reports = new Map<string, ModerationReport>();
  private payments: RoyaltyPayment[] = [];
  private licenses = new Map<string, LicenseGrant[]>();
  private bridge: LocalBridge;

  constructor() {
    this.bridge = new LocalBridge();
    this.bridge.registerHapp('media');
  }

  /**
   * Publish new content
   */
  publishContent(
    creatorId: string,
    type: ContentType,
    title: string,
    description: string,
    contentHash: string,
    storageRef: string,
    license: LicenseType = 'CCBY',
    tags: string[] = []
  ): Content {
    // Validate title and description are not empty
    MediaValidators.content(title, description);

    const id = `content-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    const content: Content = {
      id,
      type,
      title,
      description,
      creatorId,
      contributors: [
        {
          did: creatorId,
          role: 'creator',
          royaltyShare: 100,
          addedAt: Date.now(),
        },
      ],
      contentHash,
      storageRef,
      license,
      status: 'published',
      views: 0,
      likes: 0,
      shares: 0,
      tags,
      publishedAt: Date.now(),
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    this.content.set(id, content);
    this.licenses.set(id, []);

    // Update creator profile
    this.ensureCreator(creatorId);
    const creator = this.creators.get(creatorId)!;
    creator.contentCount++;
    creator.reputation = recordPositive(creator.reputation);

    return content;
  }

  /**
   * Add a contributor to content
   */
  addContributor(contentId: string, contributorDid: string, role: string, royaltyShare: number): Content {
    const content = this.content.get(contentId);
    if (!content) throw new Error('Content not found');

    // Validate royalty shares sum to 100
    const currentTotal = content.contributors.reduce((sum, c) => sum + c.royaltyShare, 0);
    if (currentTotal - content.contributors[0].royaltyShare + royaltyShare > 100) {
      throw new Error('Royalty shares exceed 100%');
    }

    // Adjust primary creator's share
    content.contributors[0].royaltyShare -= royaltyShare;

    content.contributors.push({
      did: contributorDid,
      role,
      royaltyShare,
      addedAt: Date.now(),
    });

    content.updatedAt = Date.now();
    return content;
  }

  /**
   * Record a content view
   */
  recordView(contentId: string, _viewerId: string): void {
    const content = this.content.get(contentId);
    if (!content) throw new Error('Content not found');

    content.views++;

    // Update creator stats
    const creator = this.creators.get(content.creatorId);
    if (creator) {
      creator.totalViews++;
    }

    // In production, would trigger micropayment here
  }

  /**
   * Like content
   */
  likeContent(contentId: string, _likerId: string): void {
    const content = this.content.get(contentId);
    if (!content) throw new Error('Content not found');

    content.likes++;
    content.updatedAt = Date.now();

    // Boost creator reputation
    const creator = this.creators.get(content.creatorId);
    if (creator) {
      creator.reputation = recordPositive(creator.reputation);
    }
  }

  /**
   * Report content for moderation
   */
  reportContent(
    contentId: string,
    reporterId: string,
    reason: string,
    category: ModerationReport['category']
  ): ModerationReport {
    const content = this.content.get(contentId);
    if (!content) throw new Error('Content not found');

    const report: ModerationReport = {
      id: `report-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      contentId,
      reporterId,
      reason,
      category,
      status: 'pending',
      createdAt: Date.now(),
    };

    this.reports.set(report.id, report);

    // Auto-flag if multiple reports
    const contentReports = Array.from(this.reports.values()).filter((r) => r.contentId === contentId);
    if (contentReports.length >= 3) {
      content.status = 'flagged';
    }

    return report;
  }

  /**
   * Moderate content
   */
  moderateContent(reportId: string, reviewerId: string, action: ModerationAction): ModerationReport {
    const report = this.reports.get(reportId);
    if (!report) throw new Error('Report not found');

    const content = this.content.get(report.contentId);
    if (!content) throw new Error('Content not found');

    report.status = 'resolved';
    report.action = action;
    report.reviewerId = reviewerId;
    report.resolvedAt = Date.now();

    switch (action) {
      case 'approve':
        content.status = 'published';
        break;
      case 'hide':
        content.status = 'hidden';
        // Penalize creator
        const creator = this.creators.get(content.creatorId);
        if (creator) {
          creator.reputation = recordNegative(creator.reputation);
        }
        break;
      case 'flag':
        content.status = 'flagged';
        break;
    }

    return report;
  }

  /**
   * Grant a license
   * @param grantorId - The ID of the person granting the license (must be content owner)
   * @throws {Error} If grantor is not the content owner
   */
  grantLicense(
    contentId: string,
    granteeId: string,
    license: LicenseType,
    grantorId: string,
    fee?: number,
    customTerms?: string,
    expiresAt?: number
  ): LicenseGrant {
    const content = this.content.get(contentId);
    if (!content) throw new Error('Content not found');

    // Verify grantor owns the content
    if (content.creatorId !== grantorId) {
      throw new Error('Only the content owner can grant licenses');
    }

    const grant: LicenseGrant = {
      id: `license-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      contentId,
      granteeId,
      license,
      customTerms,
      fee,
      expiresAt,
      grantedAt: Date.now(),
    };

    const contentLicenses = this.licenses.get(contentId) || [];
    contentLicenses.push(grant);
    this.licenses.set(contentId, contentLicenses);

    // Record payment if fee exists
    if (fee) {
      this.distributeRoyalties(contentId, fee, 'license');
    }

    return grant;
  }

  /**
   * Distribute royalties to contributors
   */
  distributeRoyalties(contentId: string, amount: number, reason: RoyaltyPayment['reason']): void {
    const content = this.content.get(contentId);
    if (!content) return;

    for (const contributor of content.contributors) {
      const payment = (amount * contributor.royaltyShare) / 100;
      if (payment > 0) {
        this.payments.push({
          id: `payment-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          contentId,
          recipientId: contributor.did,
          amount: payment,
          currency: 'MCX',
          reason,
          timestamp: Date.now(),
        });

        // Update creator earnings
        const creator = this.creators.get(contributor.did);
        if (creator) {
          creator.totalEarnings += payment;
        }
      }
    }
  }

  /**
   * Get content by ID
   */
  getContent(contentId: string): Content | undefined {
    return this.content.get(contentId);
  }

  /**
   * Get creator profile
   */
  getCreator(did: string): CreatorProfile | undefined {
    return this.creators.get(did);
  }

  /**
   * Search content
   */
  searchContent(query: string, tags?: string[], type?: ContentType): Content[] {
    const queryLower = query.toLowerCase();
    return Array.from(this.content.values()).filter((c) => {
      if (c.status !== 'published') return false;
      const matchesQuery = c.title.toLowerCase().includes(queryLower) || c.description.toLowerCase().includes(queryLower);
      const matchesTags = !tags || tags.some((t) => c.tags.includes(t));
      const matchesType = !type || c.type === type;
      return matchesQuery && matchesTags && matchesType;
    });
  }

  private ensureCreator(did: string): void {
    if (!this.creators.has(did)) {
      this.creators.set(did, {
        did,
        displayName: did.slice(0, 16) + '...',
        reputation: createReputation(did),
        contentCount: 0,
        totalViews: 0,
        totalEarnings: 0,
        followers: 0,
        following: 0,
        joinedAt: Date.now(),
      });
    }
  }
}

// Singleton
let instance: MediaService | null = null;

export function getMediaService(): MediaService {
  if (!instance) instance = new MediaService();
  return instance;
}

export function resetMediaService(): void {
  instance = null;
}

// ============================================================================
// Bridge Zome Types (matching Rust media_bridge zome)
// ============================================================================

import { type MycelixClient } from '../../client/index.js';

/** Content reference for cross-hApp */
export interface ContentReference {
  id: string;
  content_hash: string;
  source_happ: string;
  title: string;
  content_type: ContentType;
  creator_did: string;
  license: LicenseType;
  views: number;
  likes: number;
  created_at: number;
}

/** License verification result */
export interface LicenseVerification {
  id: string;
  content_id: string;
  licensee_did: string;
  license_type: LicenseType;
  is_valid: boolean;
  expires_at?: number;
  custom_terms?: string;
  verified_at: number;
}

/** Royalty distribution record */
export interface RoyaltyDistribution {
  id: string;
  content_id: string;
  source_happ: string;
  total_amount: number;
  currency: string;
  recipients: { did: string; share: number; amount: number }[];
  distributed_at: number;
}

/** Media bridge event types */
export type MediaBridgeEventType =
  | 'ContentPublished'
  | 'ContentLicensed'
  | 'RoyaltyDistributed'
  | 'ContentFlagged'
  | 'ContentRemoved'
  | 'CreatorVerified';

/** Media bridge event */
export interface MediaBridgeEvent {
  id: string;
  event_type: MediaBridgeEventType;
  content_id?: string;
  creator_did?: string;
  payload: string;
  source_happ: string;
  timestamp: number;
}

/** Query content input */
export interface QueryContentInput {
  source_happ: string;
  creator_did?: string;
  content_type?: ContentType;
  tags?: string[];
  license_types?: LicenseType[];
  limit?: number;
}

/** Request license input */
export interface RequestLicenseInput {
  content_id: string;
  licensee_did: string;
  license_type: LicenseType;
  purpose: string;
  duration_days?: number;
}

/** Distribute royalties input */
export interface DistributeRoyaltiesInput {
  content_id: string;
  total_amount: number;
  currency: string;
  reason: 'view' | 'license' | 'tip' | 'subscription';
}

/** Verify license input */
export interface VerifyLicenseInput {
  content_id: string;
  licensee_did: string;
  source_happ: string;
}

// ============================================================================
// Media Bridge Client (Holochain Zome Calls)
// ============================================================================

const MEDIA_ROLE = 'civic';
const BRIDGE_ZOME = 'civic_bridge';

/**
 * Media Bridge Client - Direct Holochain zome calls for cross-hApp media
 *
 * @example
 * ```typescript
 * import { MediaBridgeClient, createClient } from '@mycelix/sdk';
 *
 * const client = createClient({ installedAppId: 'mycelix-media' });
 * await client.connect();
 *
 * const mediaClient = new MediaBridgeClient(client);
 *
 * // Query CC-licensed content
 * const content = await mediaClient.queryContent({
 *   source_happ: 'my-blog',
 *   license_types: ['CCBY', 'CCBYSA'],
 *   content_type: 'Article',
 * });
 *
 * // Request license for use in another hApp
 * const license = await mediaClient.requestLicense({
 *   content_id: content[0].id,
 *   licensee_did: 'did:mycelix:alice',
 *   license_type: 'CCBY',
 *   purpose: 'Educational use in course materials',
 * });
 * ```
 */
export class MediaBridgeClient {
  constructor(private client: MycelixClient) {}

  /**
   * Query content from the media network
   */
  async queryContent(input: QueryContentInput): Promise<ContentReference[]> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'query_content',
      payload: input,
    });
  }

  /**
   * Get content by ID
   */
  async getContent(contentId: string): Promise<ContentReference | null> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_content',
      payload: contentId,
    });
  }

  /**
   * Request a license for content
   */
  async requestLicense(input: RequestLicenseInput): Promise<LicenseVerification> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'request_license',
      payload: input,
    });
  }

  /**
   * Verify a license is valid
   */
  async verifyLicense(input: VerifyLicenseInput): Promise<LicenseVerification> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'verify_license',
      payload: input,
    });
  }

  /**
   * Distribute royalties to content creators
   */
  async distributeRoyalties(input: DistributeRoyaltiesInput): Promise<RoyaltyDistribution> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'distribute_royalties',
      payload: input,
    });
  }

  /**
   * Get royalty history for content
   */
  async getRoyaltyHistory(contentId: string): Promise<RoyaltyDistribution[]> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_royalty_history',
      payload: contentId,
    });
  }

  /**
   * Get creator earnings
   */
  async getCreatorEarnings(creatorDid: string): Promise<{
    total_earnings: number;
    currency: string;
    content_count: number;
    license_grants: number;
  }> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_creator_earnings',
      payload: creatorDid,
    });
  }

  /**
   * Broadcast a media event
   */
  async broadcastMediaEvent(
    eventType: MediaBridgeEventType,
    contentId?: string,
    creatorDid?: string,
    payload: string = '{}'
  ): Promise<MediaBridgeEvent> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'broadcast_media_event',
      payload: { event_type: eventType, content_id: contentId, creator_did: creatorDid, payload },
    });
  }

  /**
   * Get recent media events
   */
  async getRecentEvents(limit?: number): Promise<MediaBridgeEvent[]> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_recent_events',
      payload: limit ?? 50,
    });
  }

  /**
   * Get events by creator
   */
  async getEventsByCreator(creatorDid: string): Promise<MediaBridgeEvent[]> {
    return this.client.callZome({
      role_name: MEDIA_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_events_by_creator',
      payload: creatorDid,
    });
  }
}

// Bridge client singleton
let bridgeInstance: MediaBridgeClient | null = null;

export function getMediaBridgeClient(client: MycelixClient): MediaBridgeClient {
  if (!bridgeInstance) bridgeInstance = new MediaBridgeClient(client);
  return bridgeInstance;
}

export function resetMediaBridgeClient(): void {
  bridgeInstance = null;
}

// ============================================================================
// Unified Client (re-exported from src/media)
// ============================================================================

/**
 * Unified Mycelix Media Client
 *
 * Use this client for most use cases. Re-exported from the main media module.
 *
 * @example
 * ```typescript
 * import { MycelixMediaClient } from '@mycelix/sdk/integrations/media';
 *
 * const media = await MycelixMediaClient.connect({
 *   url: 'ws://localhost:8888',
 * });
 *
 * // Publish content with attribution
 * const content = await media.publishWithAttribution(
 *   { type_: 'Article', title: 'My Article', ... },
 *   [{ contributor: 'did:mycelix:alice', role: 'Author', royalty_percentage: 80 }]
 * );
 * ```
 */
export {
  MycelixMediaClient,
  MediaSdkError,
  PublicationClient,
  AttributionClient,
  FactCheckClient,
  CurationClient,
  createMediaClients,
} from '../../media/index';
export type {
  MediaClientConfig,
  MediaConnectionOptions,
  HolochainRecord,
  ZomeCallable,
  Content as MediaContent,
  ContentType as MediaContentType,
  LicenseType as MediaLicenseType,
  VerificationStatus as MediaVerificationStatus,
  Attribution as MediaAttribution,
  FactCheck as MediaFactCheck,
  CurationList as MediaCurationList,
  Endorsement as MediaEndorsement,
  PublishContentInput,
  AddAttributionInput,
  SubmitFactCheckInput,
  CreateCurationListInput,
} from '../../media/index';
