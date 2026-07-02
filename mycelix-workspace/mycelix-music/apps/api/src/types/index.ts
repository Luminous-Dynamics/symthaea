// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Type Definitions Index
 *
 * Central export for shared type definitions.
 */

/**
 * Standard pagination parameters
 */
export interface PaginationParams {
  limit?: number;
  offset?: number;
  cursor?: string;
}

/**
 * Standard sort parameters
 */
export interface SortParams {
  sort?: string;
  order?: 'asc' | 'desc';
}

/**
 * Combined query parameters
 */
export interface QueryParams extends PaginationParams, SortParams {
  q?: string;
  [key: string]: unknown;
}

/**
 * Base entity with common fields
 */
export interface BaseEntity {
  id: string;
  created_at: Date;
  updated_at?: Date;
}

/**
 * Wallet address (Ethereum-style)
 */
export type WalletAddress = `0x${string}`;

/**
 * Payment model types
 */
export type PaymentModel = 'per_play' | 'subscription' | 'tip';

/**
 * Transaction status
 */
export type TransactionStatus = 'pending' | 'confirmed' | 'failed';

/**
 * Environment configuration
 */
export interface EnvironmentConfig {
  NODE_ENV: 'development' | 'staging' | 'production' | 'test';
  PORT: number;
  DATABASE_URL: string;
  REDIS_URL?: string;
  CORS_ORIGIN?: string;
  LOG_LEVEL?: 'debug' | 'info' | 'warn' | 'error';
}

/**
 * Health check response
 */
export interface HealthCheckResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  timestamp: string;
  checks: {
    database: boolean;
    redis?: boolean;
    ipfs?: boolean;
  };
}

/**
 * Analytics time range
 */
export interface TimeRange {
  start: Date;
  end: Date;
}

/**
 * Daily statistics
 */
export interface DailyStats {
  date: string;
  plays: number;
  earnings: string;
  uniqueListeners: number;
}

/**
 * Artist statistics
 */
export interface ArtistStats {
  artistAddress: WalletAddress;
  totalSongs: number;
  totalPlays: number;
  totalEarnings: string;
  topSongs: Array<{
    id: string;
    title: string;
    plays: number;
  }>;
}

/**
 * Genre statistics
 */
export interface GenreStats {
  genre: string;
  songCount: number;
  totalPlays: number;
  averageEarnings: string;
}
