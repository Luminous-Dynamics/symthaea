/**
 * Music Integration Tests
 *
 * Tests for MusicRoyaltyService - track registration, streaming play
 * recording, royalty distribution, artist profiles, collaboration
 * agreements, and reputation tracking.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  MusicRoyaltyService,
  getMusicService,
  type Track,
  type PlayRecord,
  type RoyaltyDistribution,
  type ArtistProfile,
  type CollaborationAgreement,
} from '../../src/integrations/music/index.js';

describe('Music Integration', () => {
  let service: MusicRoyaltyService;

  beforeEach(() => {
    service = new MusicRoyaltyService();
  });

  describe('registerTrack', () => {
    it('should register a track with valid artist shares', () => {
      const track = service.registerTrack({
        id: 'track-001',
        title: 'Test Song',
        artists: [
          { artistId: 'a-1', name: 'Primary', share: 0.6 },
          { artistId: 'a-2', name: 'Featured', share: 0.4 },
        ],
        releaseDate: Date.now(),
      });

      expect(track).toBeDefined();
      expect(track.id).toBe('track-001');
      expect(track.title).toBe('Test Song');
      expect(track.artists).toHaveLength(2);
    });

    it('should throw when artist shares do not sum to 1.0', () => {
      expect(() =>
        service.registerTrack({
          id: 'track-bad',
          title: 'Bad Shares',
          artists: [
            { artistId: 'a-1', name: 'A', share: 0.5 },
            { artistId: 'a-2', name: 'B', share: 0.3 },
          ],
          releaseDate: Date.now(),
        }),
      ).toThrow('Artist shares must sum to 1.0');
    });

    it('should initialize artist reputations', () => {
      service.registerTrack({
        id: 'track-rep',
        title: 'Rep Track',
        artists: [{ artistId: 'new-artist', name: 'New', share: 1.0 }],
        releaseDate: Date.now(),
      });

      const profile = service.getArtistProfile('new-artist');
      expect(profile.trustScore).toBeGreaterThan(0);
    });

    it('should create a collaboration agreement for the track', () => {
      service.registerTrack({
        id: 'track-collab',
        title: 'Collab',
        artists: [
          { artistId: 'c-1', name: 'A', share: 0.5 },
          { artistId: 'c-2', name: 'B', share: 0.5 },
        ],
        releaseDate: Date.now(),
      });

      const collab = service.getCollaboration('track-collab');
      expect(collab).toBeDefined();
      expect(collab!.status).toBe('active');
      expect(collab!.signedBy).toContain('c-1');
      expect(collab!.signedBy).toContain('c-2');
    });

    it('should track genre in artist profile', () => {
      service.registerTrack({
        id: 'track-genre',
        title: 'Genre Track',
        artists: [{ artistId: 'genre-artist', name: 'Artist', share: 1.0 }],
        releaseDate: Date.now(),
        genre: 'jazz',
      });

      const profile = service.getArtistProfile('genre-artist');
      expect(profile.genres).toContain('jazz');
    });

    it('should increment totalTracks for each artist', () => {
      service.registerTrack({
        id: 'track-ct-1',
        title: 'Track 1',
        artists: [{ artistId: 'ct-artist', name: 'Counter', share: 1.0 }],
        releaseDate: Date.now(),
      });
      service.registerTrack({
        id: 'track-ct-2',
        title: 'Track 2',
        artists: [{ artistId: 'ct-artist', name: 'Counter', share: 1.0 }],
        releaseDate: Date.now(),
      });

      const profile = service.getArtistProfile('ct-artist');
      expect(profile.totalTracks).toBe(2);
    });
  });

  describe('recordPlay', () => {
    it('should record a completed play', () => {
      service.registerTrack({
        id: 'track-play',
        title: 'Playable',
        artists: [{ artistId: 'play-artist', name: 'Artist', share: 1.0 }],
        releaseDate: Date.now(),
      });

      const play = service.recordPlay('track-play', 'listener-1', 'spotify', 45);

      expect(play).toBeDefined();
      expect(play.id).toMatch(/^play-/);
      expect(play.trackId).toBe('track-play');
      expect(play.completed).toBe(true);
      expect(play.durationPlayed).toBe(45);
    });

    it('should mark play as incomplete if duration < 30 seconds', () => {
      service.registerTrack({
        id: 'track-short',
        title: 'Short',
        artists: [{ artistId: 'short-artist', name: 'Artist', share: 1.0 }],
        releaseDate: Date.now(),
        duration: 180,
      });

      const play = service.recordPlay('track-short', 'listener-1', 'spotify', 15);
      expect(play.completed).toBe(false);
    });

    it('should mark play as completed if full track played even under 30s', () => {
      service.registerTrack({
        id: 'track-tiny',
        title: 'Tiny',
        artists: [{ artistId: 'tiny-artist', name: 'Artist', share: 1.0 }],
        releaseDate: Date.now(),
        duration: 20,
      });

      const play = service.recordPlay('track-tiny', 'listener-1', 'spotify', 20);
      expect(play.completed).toBe(true);
    });

    it('should throw for non-existent track', () => {
      expect(() =>
        service.recordPlay('track-nonexistent', 'listener', 'spotify'),
      ).toThrow('Track not found');
    });

    it('should update artist play counts for completed plays', () => {
      service.registerTrack({
        id: 'track-count',
        title: 'Count',
        artists: [{ artistId: 'count-artist', name: 'Artist', share: 1.0 }],
        releaseDate: Date.now(),
      });

      service.recordPlay('track-count', 'l-1', 'spotify', 30);
      service.recordPlay('track-count', 'l-2', 'apple', 30);

      const profile = service.getArtistProfile('count-artist');
      expect(profile.totalPlays).toBe(2);
    });
  });

  describe('distributeRoyalties', () => {
    it('should distribute royalties proportionally', () => {
      service.registerTrack({
        id: 'track-dist',
        title: 'Distribute',
        artists: [
          { artistId: 'd-1', name: 'Lead', share: 0.6 },
          { artistId: 'd-2', name: 'Feature', share: 0.4 },
        ],
        releaseDate: Date.now(),
      });

      service.recordPlay('track-dist', 'listener', 'spotify', 30);

      const dist = service.distributeRoyalties('track-dist', 1000, 'USD');

      expect(dist.totalAmount).toBe(1000);
      expect(dist.currency).toBe('USD');
      expect(dist.payments).toHaveLength(2);

      const leadPayment = dist.payments.find((p) => p.artistId === 'd-1');
      const featurePayment = dist.payments.find((p) => p.artistId === 'd-2');

      expect(leadPayment!.amount).toBe(600);
      expect(featurePayment!.amount).toBe(400);
    });

    it('should throw for non-existent track', () => {
      expect(() => service.distributeRoyalties('fake-track', 100, 'USD')).toThrow(
        'Track not found',
      );
    });

    it('should only count completed plays in period', () => {
      service.registerTrack({
        id: 'track-period',
        title: 'Period',
        artists: [{ artistId: 'p-artist', name: 'Artist', share: 1.0 }],
        releaseDate: Date.now(),
        duration: 180,
      });

      service.recordPlay('track-period', 'l-1', 'spotify', 30); // completed
      service.recordPlay('track-period', 'l-2', 'spotify', 10); // incomplete

      const dist = service.distributeRoyalties('track-period', 500, 'USD');
      expect(dist.totalPlays).toBe(1);
    });

    it('should update artist earnings', () => {
      service.registerTrack({
        id: 'track-earn',
        title: 'Earnings',
        artists: [{ artistId: 'earn-artist', name: 'Artist', share: 1.0 }],
        releaseDate: Date.now(),
      });

      service.recordPlay('track-earn', 'l-1', 'spotify', 30);
      service.distributeRoyalties('track-earn', 1000, 'USD');

      const profile = service.getArtistProfile('earn-artist');
      expect(profile.totalEarnings).toBe(1000);
    });
  });

  describe('getTrackStats', () => {
    it('should return track statistics', () => {
      service.registerTrack({
        id: 'track-stats',
        title: 'Stats Track',
        artists: [{ artistId: 'stats-artist', name: 'Artist', share: 1.0 }],
        releaseDate: Date.now(),
      });

      service.recordPlay('track-stats', 'l-1', 'spotify', 30);
      service.recordPlay('track-stats', 'l-1', 'spotify', 30);
      service.recordPlay('track-stats', 'l-2', 'apple', 30);

      const stats = service.getTrackStats('track-stats');

      expect(stats.totalPlays).toBe(3);
      expect(stats.completedPlays).toBe(3);
      expect(stats.uniqueListeners).toBe(2);
      expect(stats.platformBreakdown['spotify']).toBe(2);
      expect(stats.platformBreakdown['apple']).toBe(1);
    });

    it('should throw for non-existent track', () => {
      expect(() => service.getTrackStats('fake')).toThrow('Track not found');
    });
  });

  describe('getArtistProfile', () => {
    it('should return comprehensive artist profile', () => {
      service.registerTrack({
        id: 'track-prof1',
        title: 'Profile Track 1',
        artists: [
          { artistId: 'prof-a', name: 'Profile Artist', share: 0.7 },
          { artistId: 'prof-b', name: 'Collaborator', share: 0.3 },
        ],
        releaseDate: Date.now(),
        genre: 'rock',
      });

      const profile = service.getArtistProfile('prof-a');

      expect(profile.artistId).toBe('prof-a');
      expect(profile.totalTracks).toBe(1);
      expect(profile.genres).toContain('rock');
      expect(profile.collaboratorCount).toBe(1);
      expect(profile.trustScore).toBeGreaterThan(0);
    });

    it('should return default profile for unknown artist', () => {
      const profile = service.getArtistProfile('unknown-artist');
      expect(profile.totalTracks).toBe(0);
      expect(profile.totalPlays).toBe(0);
      expect(profile.totalEarnings).toBe(0);
    });
  });

  describe('isArtistTrustworthy', () => {
    it('should return false for unknown artist', () => {
      expect(service.isArtistTrustworthy('unknown')).toBe(false);
    });

    it('should evaluate trust based on threshold', () => {
      service.registerTrack({
        id: 'track-trust',
        title: 'Trust',
        artists: [{ artistId: 'trust-artist', name: 'Artist', share: 1.0 }],
        releaseDate: Date.now(),
      });

      const result = service.isArtistTrustworthy('trust-artist');
      expect(typeof result).toBe('boolean');
    });
  });

  describe('disputeCollaboration', () => {
    it('should mark collaboration as disputed', () => {
      service.registerTrack({
        id: 'track-disp',
        title: 'Disputed',
        artists: [
          { artistId: 'disp-a', name: 'A', share: 0.5 },
          { artistId: 'disp-b', name: 'B', share: 0.5 },
        ],
        releaseDate: Date.now(),
      });

      service.disputeCollaboration('track-disp', 'disp-a', 'Unfair split');

      const collab = service.getCollaboration('track-disp');
      expect(collab!.status).toBe('disputed');
    });

    it('should throw for non-existent collaboration', () => {
      expect(() =>
        service.disputeCollaboration('fake-track', 'artist', 'reason'),
      ).toThrow('Collaboration not found');
    });
  });

  describe('getCollaboration', () => {
    it('should return undefined for non-existent track', () => {
      expect(service.getCollaboration('nonexistent')).toBeUndefined();
    });
  });

  describe('getMusicService', () => {
    it('should return singleton instance', () => {
      const s1 = getMusicService();
      const s2 = getMusicService();
      expect(s1).toBe(s2);
    });
  });
});
