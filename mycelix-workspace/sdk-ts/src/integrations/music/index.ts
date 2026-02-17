/**
 * @mycelix/sdk Music Integration
 *
 * hApp-specific adapter for Mycelix-Music providing:
 * - Royalty tracking and fair distribution for artists
 * - Streaming play count verification
 * - Collaborative work attribution and splits
 * - Cross-hApp reputation for verified artists
 * - Federated Learning for recommendation systems
 *
 * @packageDocumentation
 * @module integrations/music
 * @see {@link MusicRoyaltyService} - Main service class
 * @see {@link getMusicService} - Singleton accessor
 *
 * @example Basic royalty distribution
 * ```typescript
 * import { getMusicService } from '@mycelix/sdk/integrations/music';
 *
 * const music = getMusicService();
 *
 * // Register a track with collaborators
 * const track = music.registerTrack({
 *   id: 'track-001',
 *   title: 'Collaborative Song',
 *   artists: [
 *     { artistId: 'artist-1', name: 'Primary Artist', share: 0.6 },
 *     { artistId: 'artist-2', name: 'Featured Artist', share: 0.25 },
 *     { artistId: 'artist-3', name: 'Producer', share: 0.15 },
 *   ],
 *   isrc: 'US-XXX-00-00001',
 *   releaseDate: Date.now(),
 * });
 *
 * // Record streaming plays
 * music.recordPlay(track.id, 'listener-123', 'spotify');
 *
 * // Distribute royalties
 * const distribution = music.distributeRoyalties(track.id, 1000.00, 'USD');
 * distribution.payments.forEach(p => {
 *   console.log(`${p.artistName}: $${p.amount.toFixed(2)}`);
 * });
 * ```
 */

import { LocalBridge, createReputationQuery } from '../../bridge/index.js';
import { FLCoordinator, AggregationMethod } from '../../fl/index.js';
import {
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  isTrustworthy,
  type ReputationScore,
} from '../../matl/index.js';

// ============================================================================
// Music-Specific Types
// ============================================================================

/**
 * Artist collaborator on a track
 */
export interface TrackArtist {
  artistId: string;
  name: string;
  role?: 'primary' | 'featured' | 'producer' | 'writer' | 'performer';
  share: number; // 0-1, percentage of royalties
}

/**
 * Complete track registration
 */
export interface Track {
  id: string;
  title: string;
  artists: TrackArtist[];
  isrc?: string; // International Standard Recording Code
  releaseDate: number;
  duration?: number; // seconds
  genre?: string;
  album?: string;
  verified: boolean;
}

/**
 * Streaming play record
 */
export interface PlayRecord {
  id: string;
  trackId: string;
  listenerId: string;
  platform: string;
  timestamp: number;
  durationPlayed: number; // seconds
  completed: boolean; // played >= 30 seconds or full track
  verified: boolean;
}

/**
 * Royalty distribution result
 */
export interface RoyaltyDistribution {
  trackId: string;
  totalAmount: number;
  currency: string;
  periodStart: number;
  periodEnd: number;
  totalPlays: number;
  payments: RoyaltyPayment[];
}

/**
 * Individual royalty payment
 */
export interface RoyaltyPayment {
  artistId: string;
  artistName: string;
  share: number;
  amount: number;
  plays: number;
}

/**
 * Artist profile with reputation
 */
export interface ArtistProfile {
  artistId: string;
  name: string;
  reputation: ReputationScore;
  trustScore: number;
  totalTracks: number;
  totalPlays: number;
  totalEarnings: number;
  verified: boolean;
  genres: string[];
  collaboratorCount: number;
}

/**
 * Collaboration agreement between artists
 */
export interface CollaborationAgreement {
  id: string;
  trackId: string;
  artists: TrackArtist[];
  createdAt: number;
  signedBy: string[];
  status: 'pending' | 'active' | 'disputed' | 'completed';
}

// ============================================================================
// Music Royalty Service
// ============================================================================

/**
 * MusicRoyaltyService - Fair royalty tracking and distribution for artists
 *
 * @remarks
 * This service provides:
 * - Transparent royalty splits between collaborators
 * - Verified play count tracking
 * - Cross-platform streaming aggregation
 * - MATL-based artist reputation
 * - FL-powered recommendation models
 *
 * @example
 * ```typescript
 * const music = new MusicRoyaltyService();
 *
 * // Register track
 * const track = music.registerTrack({
 *   id: 'track-001',
 *   title: 'My Song',
 *   artists: [{ artistId: 'me', name: 'Artist', share: 1.0 }],
 *   releaseDate: Date.now(),
 * });
 *
 * // After streaming period, distribute royalties
 * const dist = music.distributeRoyalties('track-001', 500, 'USD');
 * ```
 */
export class MusicRoyaltyService {
  private tracks: Map<string, Track> = new Map();
  private plays: Map<string, PlayRecord[]> = new Map();
  private artistReputations: Map<string, ReputationScore> = new Map();
  private artistProfiles: Map<string, Partial<ArtistProfile>> = new Map();
  private collaborations: Map<string, CollaborationAgreement> = new Map();
  private bridge: LocalBridge;
  private flCoordinator: FLCoordinator;

  constructor() {
    this.bridge = new LocalBridge();
    this.bridge.registerHapp('music');

    this.flCoordinator = new FLCoordinator({
      minParticipants: 10,
      aggregationMethod: AggregationMethod.TrustWeighted,
      byzantineTolerance: 0.34,
    });
  }

  /**
   * Register a new track with collaborator splits
   *
   * @param input - Track registration data
   * @returns Registered track with verification status
   *
   * @remarks
   * - Artist shares must sum to 1.0 (100%)
   * - Each artist gets reputation initialized
   * - Track is verified if all artists are verified
   *
   * @example
   * ```typescript
   * const track = service.registerTrack({
   *   id: 'track-001',
   *   title: 'Collaboration',
   *   artists: [
   *     { artistId: 'artist-1', name: 'Lead', share: 0.5 },
   *     { artistId: 'artist-2', name: 'Feature', share: 0.3 },
   *     { artistId: 'artist-3', name: 'Producer', share: 0.2 },
   *   ],
   *   releaseDate: Date.now(),
   * });
   * ```
   */
  registerTrack(input: Omit<Track, 'verified'>): Track {
    // Validate shares sum to 1.0
    const totalShare = input.artists.reduce((sum, a) => sum + a.share, 0);
    if (Math.abs(totalShare - 1.0) > 0.001) {
      throw new Error(`Artist shares must sum to 1.0, got ${totalShare}`);
    }

    // Initialize artist reputations if needed
    for (const artist of input.artists) {
      if (!this.artistReputations.has(artist.artistId)) {
        this.artistReputations.set(artist.artistId, createReputation(artist.artistId));
      }

      // Update artist profile
      const profile = this.artistProfiles.get(artist.artistId) || {};
      profile.totalTracks = (profile.totalTracks || 0) + 1;
      profile.genres = profile.genres || [];
      if (input.genre && !profile.genres.includes(input.genre)) {
        profile.genres.push(input.genre);
      }
      this.artistProfiles.set(artist.artistId, profile);
    }

    // Check if all artists are verified
    const allVerified = input.artists.every((a) => {
      const rep = this.artistReputations.get(a.artistId);
      return rep && reputationValue(rep) >= 0.8;
    });

    const track: Track = {
      ...input,
      verified: allVerified,
    };

    this.tracks.set(track.id, track);
    this.plays.set(track.id, []);

    // Create collaboration agreement
    const agreement: CollaborationAgreement = {
      id: `collab-${track.id}`,
      trackId: track.id,
      artists: input.artists,
      createdAt: Date.now(),
      signedBy: input.artists.map((a) => a.artistId),
      status: 'active',
    };
    this.collaborations.set(agreement.id, agreement);

    return track;
  }

  /**
   * Record a streaming play
   *
   * @param trackId - Track that was played
   * @param listenerId - Listener identifier
   * @param platform - Streaming platform (spotify, apple, etc.)
   * @param durationPlayed - Seconds played (default: 30)
   * @returns Play record
   *
   * @remarks
   * A play is counted as "completed" if:
   * - Duration >= 30 seconds, OR
   * - Full track was played
   *
   * Only completed plays count toward royalties.
   */
  recordPlay(
    trackId: string,
    listenerId: string,
    platform: string,
    durationPlayed = 30
  ): PlayRecord {
    const track = this.tracks.get(trackId);
    if (!track) {
      throw new Error(`Track not found: ${trackId}`);
    }

    const trackDuration = track.duration || 180; // Default 3 min
    const completed = durationPlayed >= 30 || durationPlayed >= trackDuration;

    const play: PlayRecord = {
      id: `play-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      trackId,
      listenerId,
      platform,
      timestamp: Date.now(),
      durationPlayed,
      completed,
      verified: track.verified,
    };

    const trackPlays = this.plays.get(trackId) || [];
    trackPlays.push(play);
    this.plays.set(trackId, trackPlays);

    // Update artist play counts
    if (completed) {
      for (const artist of track.artists) {
        const profile = this.artistProfiles.get(artist.artistId) || {};
        profile.totalPlays = (profile.totalPlays || 0) + 1;
        this.artistProfiles.set(artist.artistId, profile);
      }
    }

    return play;
  }

  /**
   * Distribute royalties for a track
   *
   * @param trackId - Track to distribute royalties for
   * @param totalAmount - Total royalty pool
   * @param currency - Currency code (USD, EUR, etc.)
   * @param periodStart - Start of royalty period (default: 30 days ago)
   * @param periodEnd - End of royalty period (default: now)
   * @returns Distribution with individual payments
   *
   * @example
   * ```typescript
   * const dist = service.distributeRoyalties('track-001', 1000, 'USD');
   * console.log(`Distributing $${dist.totalAmount} for ${dist.totalPlays} plays`);
   * dist.payments.forEach(p => {
   *   console.log(`  ${p.artistName}: $${p.amount.toFixed(2)} (${p.plays} plays)`);
   * });
   * ```
   */
  distributeRoyalties(
    trackId: string,
    totalAmount: number,
    currency: string,
    periodStart = Date.now() - 30 * 24 * 60 * 60 * 1000,
    periodEnd = Date.now()
  ): RoyaltyDistribution {
    const track = this.tracks.get(trackId);
    if (!track) {
      throw new Error(`Track not found: ${trackId}`);
    }

    const trackPlays = this.plays.get(trackId) || [];
    const periodPlays = trackPlays.filter(
      (p) => p.timestamp >= periodStart && p.timestamp <= periodEnd && p.completed
    );

    const payments: RoyaltyPayment[] = track.artists.map((artist) => {
      const amount = totalAmount * artist.share;
      const plays = Math.round(periodPlays.length * artist.share);

      // Update artist earnings
      const profile = this.artistProfiles.get(artist.artistId) || {};
      profile.totalEarnings = (profile.totalEarnings || 0) + amount;
      this.artistProfiles.set(artist.artistId, profile);

      // Record positive reputation for successful distribution
      let rep = this.artistReputations.get(artist.artistId) || createReputation(artist.artistId);
      rep = recordPositive(rep);
      this.artistReputations.set(artist.artistId, rep);

      return {
        artistId: artist.artistId,
        artistName: artist.name,
        share: artist.share,
        amount,
        plays,
      };
    });

    return {
      trackId,
      totalAmount,
      currency,
      periodStart,
      periodEnd,
      totalPlays: periodPlays.length,
      payments,
    };
  }

  /**
   * Get comprehensive artist profile
   */
  getArtistProfile(artistId: string): ArtistProfile {
    const reputation = this.artistReputations.get(artistId) || createReputation(artistId);
    const profile = this.artistProfiles.get(artistId) || {};

    // Count unique collaborators
    const collaboratorSet = new Set<string>();
    for (const track of this.tracks.values()) {
      if (track.artists.some((a) => a.artistId === artistId)) {
        track.artists.forEach((a) => {
          if (a.artistId !== artistId) collaboratorSet.add(a.artistId);
        });
      }
    }

    return {
      artistId,
      name: profile.name || artistId,
      reputation,
      trustScore: reputationValue(reputation),
      totalTracks: profile.totalTracks || 0,
      totalPlays: profile.totalPlays || 0,
      totalEarnings: profile.totalEarnings || 0,
      verified: reputationValue(reputation) >= 0.8 && (profile.totalTracks || 0) >= 5,
      genres: profile.genres || [],
      collaboratorCount: collaboratorSet.size,
    };
  }

  /**
   * Get track statistics
   */
  getTrackStats(trackId: string): {
    track: Track;
    totalPlays: number;
    completedPlays: number;
    platformBreakdown: Record<string, number>;
    uniqueListeners: number;
  } {
    const track = this.tracks.get(trackId);
    if (!track) {
      throw new Error(`Track not found: ${trackId}`);
    }

    const trackPlays = this.plays.get(trackId) || [];
    const platformBreakdown: Record<string, number> = {};
    const listeners = new Set<string>();

    for (const play of trackPlays) {
      platformBreakdown[play.platform] = (platformBreakdown[play.platform] || 0) + 1;
      listeners.add(play.listenerId);
    }

    return {
      track,
      totalPlays: trackPlays.length,
      completedPlays: trackPlays.filter((p) => p.completed).length,
      platformBreakdown,
      uniqueListeners: listeners.size,
    };
  }

  /**
   * Check if artist is trustworthy
   */
  isArtistTrustworthy(artistId: string, threshold = 0.7): boolean {
    const reputation = this.artistReputations.get(artistId);
    if (!reputation) return false;
    return isTrustworthy(reputation, threshold);
  }

  /**
   * Query artist reputation from other hApps
   */
  queryExternalReputation(artistId: string): void {
    const query = createReputationQuery('music', artistId);
    this.bridge.send('identity', query);
    this.bridge.send('marketplace', query);
  }

  /**
   * Get collaboration agreement for a track
   */
  getCollaboration(trackId: string): CollaborationAgreement | undefined {
    return this.collaborations.get(`collab-${trackId}`);
  }

  /**
   * Dispute a collaboration agreement
   */
  disputeCollaboration(trackId: string, disputantId: string, _reason: string): void {
    const collab = this.collaborations.get(`collab-${trackId}`);
    if (!collab) {
      throw new Error(`Collaboration not found for track: ${trackId}`);
    }

    collab.status = 'disputed';
    this.collaborations.set(`collab-${trackId}`, collab);

    // Record negative reputation for dispute
    let rep = this.artistReputations.get(disputantId) || createReputation(disputantId);
    rep = recordNegative(rep);
    this.artistReputations.set(disputantId, rep);
  }

  /**
   * Get FL coordinator for recommendation models
   */
  getFLCoordinator(): FLCoordinator {
    return this.flCoordinator;
  }
}

// ============================================================================
// Exports
// ============================================================================

export {
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  isTrustworthy,
  AggregationMethod,
};

// Default service instance
let defaultService: MusicRoyaltyService | null = null;

/**
 * Get the default music royalty service instance
 */
export function getMusicService(): MusicRoyaltyService {
  if (!defaultService) {
    defaultService = new MusicRoyaltyService();
  }
  return defaultService;
}
