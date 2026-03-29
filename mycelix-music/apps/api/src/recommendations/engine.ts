// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Music Recommendation Engine
 *
 * ML-based music discovery using:
 * - Collaborative filtering (users who liked X also liked Y)
 * - Content-based filtering (audio features, genre, mood)
 * - Contextual awareness (time, mood, activity)
 * - Hybrid approaches for best results
 */

import { Pool } from 'pg';
import { createBatchLoader } from '../resilience/request-coalescing';
import { getLogger } from '../logging';
import { getMetrics } from '../metrics';

const logger = getLogger();

/**
 * Audio features extracted from tracks
 */
export interface AudioFeatures {
  tempo: number;           // BPM (60-200)
  energy: number;          // 0-1 (calm to energetic)
  danceability: number;    // 0-1
  valence: number;         // 0-1 (sad to happy)
  acousticness: number;    // 0-1
  instrumentalness: number; // 0-1
  speechiness: number;     // 0-1
  loudness: number;        // dB (-60 to 0)
  key: number;             // 0-11 (pitch class)
  mode: number;            // 0 (minor) or 1 (major)
  duration: number;        // seconds
}

/**
 * User taste profile
 */
export interface TasteProfile {
  walletAddress: string;
  // Genre preferences (normalized weights)
  genres: Map<string, number>;
  // Average audio features from listening history
  avgFeatures: AudioFeatures;
  // Favorite artists
  topArtists: string[];
  // Listening patterns
  patterns: {
    preferredHours: number[];      // 0-23
    avgSessionLength: number;      // minutes
    skipRate: number;              // 0-1
    repeatRate: number;            // 0-1
    discoveryRate: number;         // 0-1 (new vs familiar)
  };
  // Last updated
  updatedAt: Date;
}

/**
 * Recommendation context
 */
export interface RecommendationContext {
  // Current time context
  timeOfDay: 'morning' | 'afternoon' | 'evening' | 'night';
  dayOfWeek: number;
  // User-specified context
  mood?: 'happy' | 'sad' | 'energetic' | 'calm' | 'focused';
  activity?: 'workout' | 'work' | 'party' | 'sleep' | 'commute';
  // Listening context
  currentSongId?: string;
  recentSongIds?: string[];
  // Preferences
  excludeArtists?: string[];
  excludeSongs?: string[];
  preferNew?: boolean;
}

/**
 * Recommendation result
 */
export interface Recommendation {
  songId: string;
  score: number;
  reasons: string[];
  source: 'collaborative' | 'content' | 'trending' | 'personalized' | 'contextual';
}

/**
 * Recommendation Engine
 */
export class RecommendationEngine {
  private tasteProfiles: Map<string, TasteProfile> = new Map();
  private songFeatures: Map<string, AudioFeatures> = new Map();
  private similarityCache: Map<string, Map<string, number>> = new Map();

  constructor(private pool: Pool) {
    // Register metrics
    const metrics = getMetrics();
    metrics.createHistogram('recommendation_latency_ms', 'Recommendation generation latency', ['type']);
    metrics.createCounter('recommendations_generated_total', 'Total recommendations generated', ['source']);
  }

  /**
   * Get personalized recommendations for a user
   */
  async getRecommendations(
    walletAddress: string,
    context: RecommendationContext = {},
    limit = 20
  ): Promise<Recommendation[]> {
    const startTime = Date.now();
    const normalizedAddress = walletAddress.toLowerCase();

    // Get or build taste profile
    let profile = this.tasteProfiles.get(normalizedAddress);
    if (!profile || Date.now() - profile.updatedAt.getTime() > 3600000) {
      profile = await this.buildTasteProfile(normalizedAddress);
      this.tasteProfiles.set(normalizedAddress, profile);
    }

    // Generate recommendations from multiple sources
    const [collaborative, contentBased, trending, contextual] = await Promise.all([
      this.collaborativeFiltering(normalizedAddress, profile, limit),
      this.contentBasedFiltering(profile, context, limit),
      this.getTrendingRecommendations(profile, limit),
      this.getContextualRecommendations(profile, context, limit),
    ]);

    // Merge and rank recommendations
    const merged = this.mergeRecommendations(
      [collaborative, contentBased, trending, contextual],
      context
    );

    // Filter out excluded songs/artists
    const filtered = merged.filter(rec => {
      if (context.excludeSongs?.includes(rec.songId)) return false;
      // Would need to check artist from song lookup
      return true;
    });

    // Diversify results
    const diversified = this.diversifyRecommendations(filtered, limit);

    const duration = Date.now() - startTime;
    getMetrics().observeHistogram('recommendation_latency_ms', duration, { type: 'personalized' });

    logger.debug('Generated recommendations', {
      wallet: normalizedAddress.slice(0, 10),
      count: diversified.length,
      duration,
    });

    return diversified;
  }

  /**
   * Build user taste profile from listening history
   */
  async buildTasteProfile(walletAddress: string): Promise<TasteProfile> {
    // Get listening history
    const historyResult = await this.pool.query(`
      SELECT
        p.song_id,
        p.duration as play_duration,
        p.played_at,
        s.genre,
        s.artist_address,
        s.duration as song_duration,
        EXTRACT(HOUR FROM p.played_at) as hour
      FROM plays p
      JOIN songs s ON p.song_id = s.id
      WHERE p.wallet_address = $1
      AND p.played_at > NOW() - INTERVAL '90 days'
      ORDER BY p.played_at DESC
      LIMIT 1000
    `, [walletAddress]);

    const plays = historyResult.rows;

    // Calculate genre preferences
    const genreCounts = new Map<string, number>();
    const artistCounts = new Map<string, number>();
    const hourCounts = new Array(24).fill(0);
    let totalSkips = 0;
    let totalRepeats = 0;
    const seenSongs = new Set<string>();

    for (const play of plays) {
      // Genre tracking
      if (play.genre) {
        genreCounts.set(play.genre, (genreCounts.get(play.genre) || 0) + 1);
      }

      // Artist tracking
      artistCounts.set(play.artist_address, (artistCounts.get(play.artist_address) || 0) + 1);

      // Hour tracking
      hourCounts[play.hour]++;

      // Skip detection (played less than 30% of song)
      if (play.play_duration && play.song_duration) {
        if (play.play_duration < play.song_duration * 0.3) {
          totalSkips++;
        }
      }

      // Repeat detection
      if (seenSongs.has(play.song_id)) {
        totalRepeats++;
      }
      seenSongs.add(play.song_id);
    }

    // Normalize genre weights
    const totalGenrePlays = Array.from(genreCounts.values()).reduce((a, b) => a + b, 0);
    const genres = new Map<string, number>();
    for (const [genre, count] of genreCounts) {
      genres.set(genre, count / totalGenrePlays);
    }

    // Get top artists
    const topArtists = Array.from(artistCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20)
      .map(([artist]) => artist);

    // Find preferred listening hours
    const avgHourPlays = hourCounts.reduce((a, b) => a + b, 0) / 24;
    const preferredHours = hourCounts
      .map((count, hour) => ({ hour, count }))
      .filter(h => h.count > avgHourPlays * 1.5)
      .map(h => h.hour);

    // Calculate average audio features (would need actual feature data)
    const avgFeatures: AudioFeatures = {
      tempo: 120,
      energy: 0.6,
      danceability: 0.5,
      valence: 0.5,
      acousticness: 0.3,
      instrumentalness: 0.2,
      speechiness: 0.1,
      loudness: -8,
      key: 0,
      mode: 1,
      duration: 210,
    };

    return {
      walletAddress,
      genres,
      avgFeatures,
      topArtists,
      patterns: {
        preferredHours,
        avgSessionLength: 45,
        skipRate: plays.length > 0 ? totalSkips / plays.length : 0.15,
        repeatRate: plays.length > 0 ? totalRepeats / plays.length : 0.3,
        discoveryRate: plays.length > 0 ? seenSongs.size / plays.length : 0.5,
      },
      updatedAt: new Date(),
    };
  }

  /**
   * Collaborative filtering - "users who liked X also liked Y"
   */
  async collaborativeFiltering(
    walletAddress: string,
    profile: TasteProfile,
    limit: number
  ): Promise<Recommendation[]> {
    // Find similar users based on listening overlap
    const similarUsersResult = await this.pool.query(`
      WITH user_songs AS (
        SELECT DISTINCT song_id FROM plays WHERE wallet_address = $1
      ),
      similar_users AS (
        SELECT
          p.wallet_address,
          COUNT(DISTINCT p.song_id) as overlap,
          COUNT(DISTINCT p.song_id)::float / (SELECT COUNT(*) FROM user_songs) as similarity
        FROM plays p
        JOIN user_songs us ON p.song_id = us.song_id
        WHERE p.wallet_address != $1
        GROUP BY p.wallet_address
        HAVING COUNT(DISTINCT p.song_id) >= 5
        ORDER BY similarity DESC
        LIMIT 50
      )
      SELECT
        s.id as song_id,
        s.title,
        COUNT(DISTINCT p.wallet_address) as recommender_count,
        AVG(su.similarity) as avg_similarity
      FROM similar_users su
      JOIN plays p ON p.wallet_address = su.wallet_address
      JOIN songs s ON p.song_id = s.id
      WHERE s.id NOT IN (SELECT song_id FROM plays WHERE wallet_address = $1)
      AND s.deleted_at IS NULL
      GROUP BY s.id, s.title
      ORDER BY recommender_count * avg_similarity DESC
      LIMIT $2
    `, [walletAddress, limit * 2]);

    return similarUsersResult.rows.map(row => ({
      songId: row.song_id,
      score: row.recommender_count * row.avg_similarity,
      reasons: [`Liked by ${row.recommender_count} listeners with similar taste`],
      source: 'collaborative' as const,
    }));
  }

  /**
   * Content-based filtering - similar audio features and metadata
   */
  async contentBasedFiltering(
    profile: TasteProfile,
    context: RecommendationContext,
    limit: number
  ): Promise<Recommendation[]> {
    // Get songs matching user's genre preferences
    const genreList = Array.from(profile.genres.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([genre]) => genre);

    if (genreList.length === 0) {
      return [];
    }

    const result = await this.pool.query(`
      SELECT
        s.id as song_id,
        s.title,
        s.genre,
        s.play_count,
        s.artist_address
      FROM songs s
      WHERE s.genre = ANY($1)
      AND s.deleted_at IS NULL
      AND s.artist_address = ANY($2)
      ORDER BY s.play_count DESC
      LIMIT $3
    `, [genreList, profile.topArtists.slice(0, 10), limit]);

    // Also get songs from similar genres but different artists (discovery)
    const discoveryResult = await this.pool.query(`
      SELECT
        s.id as song_id,
        s.title,
        s.genre,
        s.play_count
      FROM songs s
      WHERE s.genre = ANY($1)
      AND s.deleted_at IS NULL
      AND s.artist_address != ALL($2)
      AND s.play_count > 100
      ORDER BY RANDOM()
      LIMIT $3
    `, [genreList, profile.topArtists, limit]);

    const recommendations: Recommendation[] = [
      ...result.rows.map(row => ({
        songId: row.song_id,
        score: 0.8 + (profile.genres.get(row.genre) || 0) * 0.2,
        reasons: [`Matches your ${row.genre} preference`, 'From an artist you follow'],
        source: 'content' as const,
      })),
      ...discoveryResult.rows.map(row => ({
        songId: row.song_id,
        score: 0.6 + (profile.genres.get(row.genre) || 0) * 0.3,
        reasons: [`Discover new ${row.genre} artists`],
        source: 'content' as const,
      })),
    ];

    return recommendations;
  }

  /**
   * Get trending songs adjusted for user preferences
   */
  async getTrendingRecommendations(
    profile: TasteProfile,
    limit: number
  ): Promise<Recommendation[]> {
    const result = await this.pool.query(`
      SELECT
        s.id as song_id,
        s.title,
        s.genre,
        COUNT(p.id) as recent_plays,
        COUNT(DISTINCT p.wallet_address) as unique_listeners
      FROM songs s
      JOIN plays p ON s.id = p.song_id
      WHERE p.played_at > NOW() - INTERVAL '7 days'
      AND s.deleted_at IS NULL
      GROUP BY s.id, s.title, s.genre
      ORDER BY recent_plays DESC
      LIMIT $1
    `, [limit * 2]);

    return result.rows.map(row => {
      const genreBoost = profile.genres.get(row.genre) || 0.1;
      return {
        songId: row.song_id,
        score: 0.5 + genreBoost * 0.5,
        reasons: [`Trending with ${row.unique_listeners} listeners this week`],
        source: 'trending' as const,
      };
    });
  }

  /**
   * Context-aware recommendations
   */
  async getContextualRecommendations(
    profile: TasteProfile,
    context: RecommendationContext,
    limit: number
  ): Promise<Recommendation[]> {
    // Determine target audio features based on context
    const targetFeatures = this.getTargetFeatures(context);

    // Map mood/activity to genres
    const contextGenres = this.getContextGenres(context);

    if (contextGenres.length === 0) {
      return [];
    }

    const result = await this.pool.query(`
      SELECT
        s.id as song_id,
        s.title,
        s.genre,
        s.duration
      FROM songs s
      WHERE s.genre = ANY($1)
      AND s.deleted_at IS NULL
      ${context.activity === 'workout' ? 'AND s.duration BETWEEN 180 AND 300' : ''}
      ${context.activity === 'sleep' ? 'AND s.duration > 240' : ''}
      ORDER BY s.play_count DESC
      LIMIT $2
    `, [contextGenres, limit]);

    return result.rows.map(row => ({
      songId: row.song_id,
      score: 0.7,
      reasons: [this.getContextReason(context)],
      source: 'contextual' as const,
    }));
  }

  /**
   * Get target audio features for context
   */
  private getTargetFeatures(context: RecommendationContext): Partial<AudioFeatures> {
    const features: Partial<AudioFeatures> = {};

    switch (context.mood) {
      case 'happy':
        features.valence = 0.8;
        features.energy = 0.7;
        break;
      case 'sad':
        features.valence = 0.3;
        features.energy = 0.4;
        break;
      case 'energetic':
        features.energy = 0.9;
        features.danceability = 0.8;
        break;
      case 'calm':
        features.energy = 0.3;
        features.acousticness = 0.7;
        break;
      case 'focused':
        features.instrumentalness = 0.7;
        features.speechiness = 0.1;
        break;
    }

    switch (context.activity) {
      case 'workout':
        features.tempo = 140;
        features.energy = 0.9;
        break;
      case 'work':
        features.instrumentalness = 0.6;
        features.energy = 0.5;
        break;
      case 'party':
        features.danceability = 0.9;
        features.energy = 0.8;
        break;
      case 'sleep':
        features.energy = 0.2;
        features.acousticness = 0.8;
        break;
    }

    return features;
  }

  /**
   * Get genres appropriate for context
   */
  private getContextGenres(context: RecommendationContext): string[] {
    const genreMap: Record<string, string[]> = {
      workout: ['Electronic', 'Hip-Hop', 'Rock', 'Pop'],
      work: ['Ambient', 'Classical', 'Jazz', 'Lo-Fi'],
      party: ['Electronic', 'Pop', 'Hip-Hop', 'Dance'],
      sleep: ['Ambient', 'Classical', 'Acoustic'],
      commute: ['Pop', 'Rock', 'Hip-Hop', 'Electronic'],
    };

    const moodMap: Record<string, string[]> = {
      happy: ['Pop', 'Dance', 'Funk'],
      sad: ['Acoustic', 'Singer-Songwriter', 'Classical'],
      energetic: ['Electronic', 'Rock', 'Hip-Hop'],
      calm: ['Ambient', 'Jazz', 'Classical'],
      focused: ['Ambient', 'Classical', 'Lo-Fi'],
    };

    const genres = new Set<string>();

    if (context.activity && genreMap[context.activity]) {
      genreMap[context.activity].forEach(g => genres.add(g));
    }

    if (context.mood && moodMap[context.mood]) {
      moodMap[context.mood].forEach(g => genres.add(g));
    }

    return Array.from(genres);
  }

  /**
   * Get context-specific reason string
   */
  private getContextReason(context: RecommendationContext): string {
    if (context.activity) {
      return `Perfect for ${context.activity}`;
    }
    if (context.mood) {
      return `Matches your ${context.mood} mood`;
    }
    if (context.timeOfDay) {
      return `Great for ${context.timeOfDay} listening`;
    }
    return 'Recommended for you';
  }

  /**
   * Merge recommendations from multiple sources
   */
  private mergeRecommendations(
    sources: Recommendation[][],
    context: RecommendationContext
  ): Recommendation[] {
    const merged = new Map<string, Recommendation>();

    // Weight sources differently
    const weights = {
      collaborative: 0.35,
      content: 0.25,
      trending: 0.15,
      contextual: 0.25,
    };

    for (const recommendations of sources) {
      for (const rec of recommendations) {
        const weight = weights[rec.source];
        const weightedScore = rec.score * weight;

        const existing = merged.get(rec.songId);
        if (existing) {
          existing.score += weightedScore;
          existing.reasons = [...new Set([...existing.reasons, ...rec.reasons])];
        } else {
          merged.set(rec.songId, {
            ...rec,
            score: weightedScore,
          });
        }
      }
    }

    return Array.from(merged.values()).sort((a, b) => b.score - a.score);
  }

  /**
   * Diversify recommendations to avoid too much similarity
   */
  private diversifyRecommendations(
    recommendations: Recommendation[],
    limit: number
  ): Recommendation[] {
    const selected: Recommendation[] = [];
    const selectedSources = new Map<string, number>();

    for (const rec of recommendations) {
      if (selected.length >= limit) break;

      // Limit recommendations from single source
      const sourceCount = selectedSources.get(rec.source) || 0;
      if (sourceCount >= Math.ceil(limit / 3)) continue;

      selected.push(rec);
      selectedSources.set(rec.source, sourceCount + 1);
    }

    // Fill remaining slots if needed
    for (const rec of recommendations) {
      if (selected.length >= limit) break;
      if (!selected.find(s => s.songId === rec.songId)) {
        selected.push(rec);
      }
    }

    return selected;
  }

  /**
   * Get similar songs (for "more like this")
   */
  async getSimilarSongs(songId: string, limit = 20): Promise<Recommendation[]> {
    // Get song details
    const songResult = await this.pool.query(
      'SELECT * FROM songs WHERE id = $1',
      [songId]
    );

    if (songResult.rows.length === 0) {
      return [];
    }

    const song = songResult.rows[0];

    // Find similar songs by genre and artist connections
    const result = await this.pool.query(`
      WITH song_listeners AS (
        SELECT DISTINCT wallet_address FROM plays WHERE song_id = $1
      )
      SELECT
        s.id as song_id,
        s.title,
        s.genre,
        COUNT(DISTINCT p.wallet_address) as shared_listeners,
        CASE WHEN s.genre = $2 THEN 0.3 ELSE 0 END as genre_boost,
        CASE WHEN s.artist_address = $3 THEN 0.2 ELSE 0 END as artist_boost
      FROM songs s
      JOIN plays p ON s.id = p.song_id
      JOIN song_listeners sl ON p.wallet_address = sl.wallet_address
      WHERE s.id != $1
      AND s.deleted_at IS NULL
      GROUP BY s.id, s.title, s.genre, s.artist_address
      ORDER BY shared_listeners DESC
      LIMIT $4
    `, [songId, song.genre, song.artist_address, limit]);

    return result.rows.map(row => ({
      songId: row.song_id,
      score: row.shared_listeners * 0.1 + row.genre_boost + row.artist_boost,
      reasons: ['Listeners also enjoyed this'],
      source: 'content' as const,
    }));
  }

  /**
   * Get radio/endless mix based on seed songs
   */
  async getRadioMix(
    seedSongIds: string[],
    walletAddress: string,
    limit = 50
  ): Promise<Recommendation[]> {
    const recommendations: Recommendation[] = [];

    // Get similar songs for each seed
    for (const seedId of seedSongIds.slice(0, 5)) {
      const similar = await this.getSimilarSongs(seedId, 20);
      recommendations.push(...similar);
    }

    // Add some personalized recommendations
    const profile = await this.buildTasteProfile(walletAddress);
    const personalized = await this.contentBasedFiltering(profile, {}, 20);
    recommendations.push(...personalized);

    // Deduplicate and sort
    const unique = new Map<string, Recommendation>();
    for (const rec of recommendations) {
      if (!unique.has(rec.songId) && !seedSongIds.includes(rec.songId)) {
        unique.set(rec.songId, rec);
      }
    }

    return Array.from(unique.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  /**
   * Record feedback for improving recommendations
   */
  async recordFeedback(
    walletAddress: string,
    songId: string,
    feedback: 'like' | 'dislike' | 'skip' | 'save'
  ): Promise<void> {
    await this.pool.query(`
      INSERT INTO recommendation_feedback
        (wallet_address, song_id, feedback, created_at)
      VALUES ($1, $2, $3, NOW())
      ON CONFLICT (wallet_address, song_id)
      DO UPDATE SET feedback = $3, created_at = NOW()
    `, [walletAddress, songId, feedback]);

    // Invalidate taste profile cache
    this.tasteProfiles.delete(walletAddress.toLowerCase());

    getMetrics().incCounter('recommendation_feedback_total', { feedback });
  }
}

/**
 * Create recommendation engine
 */
export function createRecommendationEngine(pool: Pool): RecommendationEngine {
  return new RecommendationEngine(pool);
}

/**
 * Migration SQL for recommendations
 */
export const RECOMMENDATION_MIGRATION_SQL = `
-- Audio features table
CREATE TABLE IF NOT EXISTS song_features (
  song_id UUID PRIMARY KEY REFERENCES songs(id) ON DELETE CASCADE,
  tempo REAL,
  energy REAL,
  danceability REAL,
  valence REAL,
  acousticness REAL,
  instrumentalness REAL,
  speechiness REAL,
  loudness REAL,
  key INTEGER,
  mode INTEGER,
  analyzed_at TIMESTAMPTZ DEFAULT NOW()
);

-- User taste profiles (cached)
CREATE TABLE IF NOT EXISTS user_taste_profiles (
  wallet_address VARCHAR(42) PRIMARY KEY,
  profile JSONB NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Recommendation feedback
CREATE TABLE IF NOT EXISTS recommendation_feedback (
  wallet_address VARCHAR(42) NOT NULL,
  song_id UUID NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
  feedback VARCHAR(20) NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (wallet_address, song_id)
);

CREATE INDEX idx_feedback_wallet ON recommendation_feedback(wallet_address);
CREATE INDEX idx_feedback_song ON recommendation_feedback(song_id);
`;

export default RecommendationEngine;
