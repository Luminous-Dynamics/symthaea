// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Business Metrics
 *
 * Domain-specific metrics for the Mycelix music platform.
 */

import { Registry, Counter, Gauge, Histogram } from 'prom-client';
import { createCounter, createGauge, createHistogram } from './registry';

export interface BusinessMetrics {
  // Playback metrics
  playsTotal: Counter;
  playDuration: Histogram;
  streamsActive: Gauge;

  // User metrics
  usersTotal: Gauge;
  usersActive: Gauge;
  signupsTotal: Counter;

  // Content metrics
  songsTotal: Gauge;
  artistsTotal: Gauge;
  playlistsTotal: Gauge;
  uploadsTotal: Counter;
  uploadSize: Histogram;

  // Engagement metrics
  likesTotal: Counter;
  followsTotal: Counter;
  sharesTotal: Counter;
  commentsTotal: Counter;

  // Search metrics
  searchQueries: Counter;
  searchLatency: Histogram;
  searchResultsCount: Histogram;

  // Transaction metrics
  transactionsTotal: Counter;
  transactionAmount: Histogram;
  royaltiesDistributed: Counter;
}

/**
 * Create business metrics
 */
export function createBusinessMetrics(registry: Registry): BusinessMetrics {
  return {
    // Playback
    playsTotal: createCounter(
      registry,
      'plays_total',
      'Total number of song plays',
      ['genre', 'source'] // source: 'web', 'mobile', 'api'
    ),
    playDuration: createHistogram(
      registry,
      'play_duration_seconds',
      'Duration of song plays in seconds',
      ['completed'], // 'true' if played to end
      [10, 30, 60, 120, 180, 240, 300, 600]
    ),
    streamsActive: createGauge(
      registry,
      'streams_active',
      'Number of active streams'
    ),

    // Users
    usersTotal: createGauge(
      registry,
      'users_total',
      'Total number of registered users'
    ),
    usersActive: createGauge(
      registry,
      'users_active',
      'Number of active users (last 24h)'
    ),
    signupsTotal: createCounter(
      registry,
      'signups_total',
      'Total number of user signups',
      ['method'] // 'wallet', 'email', 'social'
    ),

    // Content
    songsTotal: createGauge(
      registry,
      'songs_total',
      'Total number of songs'
    ),
    artistsTotal: createGauge(
      registry,
      'artists_total',
      'Total number of artists'
    ),
    playlistsTotal: createGauge(
      registry,
      'playlists_total',
      'Total number of playlists'
    ),
    uploadsTotal: createCounter(
      registry,
      'uploads_total',
      'Total number of uploads',
      ['type', 'status'] // type: 'song', 'cover', status: 'success', 'failed'
    ),
    uploadSize: createHistogram(
      registry,
      'upload_size_bytes',
      'Size of uploads in bytes',
      ['type'],
      [1000000, 5000000, 10000000, 25000000, 50000000, 100000000] // 1MB to 100MB
    ),

    // Engagement
    likesTotal: createCounter(
      registry,
      'likes_total',
      'Total number of likes',
      ['target_type'] // 'song', 'playlist', 'comment'
    ),
    followsTotal: createCounter(
      registry,
      'follows_total',
      'Total number of follows',
      ['action'] // 'follow', 'unfollow'
    ),
    sharesTotal: createCounter(
      registry,
      'shares_total',
      'Total number of shares',
      ['platform'] // 'twitter', 'facebook', 'copy_link'
    ),
    commentsTotal: createCounter(
      registry,
      'comments_total',
      'Total number of comments',
      ['action'] // 'create', 'delete'
    ),

    // Search
    searchQueries: createCounter(
      registry,
      'search_queries_total',
      'Total number of search queries',
      ['type'] // 'full_text', 'autocomplete', 'similar'
    ),
    searchLatency: createHistogram(
      registry,
      'search_latency_seconds',
      'Search query latency in seconds',
      ['type'],
      [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]
    ),
    searchResultsCount: createHistogram(
      registry,
      'search_results_count',
      'Number of search results returned',
      ['type'],
      [0, 1, 5, 10, 25, 50, 100]
    ),

    // Transactions
    transactionsTotal: createCounter(
      registry,
      'transactions_total',
      'Total number of blockchain transactions',
      ['type', 'status'] // type: 'royalty', 'nft_mint', 'subscription', status: 'success', 'failed'
    ),
    transactionAmount: createHistogram(
      registry,
      'transaction_amount_wei',
      'Transaction amount in wei',
      ['type'],
      [1e15, 1e16, 1e17, 1e18, 1e19, 1e20] // 0.001 to 100 ETH
    ),
    royaltiesDistributed: createCounter(
      registry,
      'royalties_distributed_total',
      'Total royalties distributed (in wei)',
      ['currency'] // 'ETH', 'USDC'
    ),
  };
}
