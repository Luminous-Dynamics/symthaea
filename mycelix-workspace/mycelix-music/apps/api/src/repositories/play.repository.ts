// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Play Repository
 *
 * Handles all database operations for play events/transactions
 * with analytics-focused query methods.
 */

import { Pool, PoolClient } from 'pg';
import { BaseRepository, QueryOptions } from './base.repository';

export interface Play {
  id: string;
  song_id: string;
  listener_address: string;
  amount: string;
  transaction_hash?: string;
  status: 'pending' | 'confirmed' | 'failed';
  created_at: Date;
  confirmed_at?: Date;
}

export interface CreatePlayInput {
  song_id: string;
  listener_address: string;
  amount: string;
  transaction_hash?: string;
}

export interface PlayWithSong extends Play {
  song_title: string;
  song_artist: string;
  artist_address: string;
}

export interface TimeRange {
  start: Date;
  end: Date;
}

export class PlayRepository extends BaseRepository<Play> {
  constructor(pool: Pool) {
    super(pool, 'plays');
  }

  /**
   * Create a new play record
   */
  async createPlay(
    input: CreatePlayInput,
    client?: PoolClient
  ): Promise<Play> {
    const sql = `
      INSERT INTO ${this.tableName}
        (song_id, listener_address, amount, transaction_hash, status)
      VALUES ($1, $2, $3, $4, 'pending')
      RETURNING *
    `;
    const result = await this.query<Play>(
      sql,
      [input.song_id, input.listener_address, input.amount, input.transaction_hash],
      { client }
    );
    return result.rows[0];
  }

  /**
   * Confirm a play after blockchain confirmation
   */
  async confirmPlay(
    playId: string,
    transactionHash: string,
    options: QueryOptions = {}
  ): Promise<Play | null> {
    const sql = `
      UPDATE ${this.tableName}
      SET status = 'confirmed',
          transaction_hash = $2,
          confirmed_at = NOW()
      WHERE id = $1
      RETURNING *
    `;
    const result = await this.query<Play>(sql, [playId, transactionHash], options);
    return result.rows[0] || null;
  }

  /**
   * Mark a play as failed
   */
  async failPlay(
    playId: string,
    options: QueryOptions = {}
  ): Promise<Play | null> {
    const sql = `
      UPDATE ${this.tableName}
      SET status = 'failed'
      WHERE id = $1
      RETURNING *
    `;
    const result = await this.query<Play>(sql, [playId], options);
    return result.rows[0] || null;
  }

  /**
   * Get plays for a song
   */
  async findBySong(
    songId: string,
    limit = 50,
    offset = 0,
    options: QueryOptions = {}
  ): Promise<Play[]> {
    const sql = `
      SELECT * FROM ${this.tableName}
      WHERE song_id = $1 AND status = 'confirmed'
      ORDER BY created_at DESC
      LIMIT $2 OFFSET $3
    `;
    const result = await this.query<Play>(sql, [songId, limit, offset], options);
    return result.rows;
  }

  /**
   * Get plays by listener
   */
  async findByListener(
    listenerAddress: string,
    limit = 50,
    offset = 0,
    options: QueryOptions = {}
  ): Promise<PlayWithSong[]> {
    const sql = `
      SELECT
        p.*,
        s.title as song_title,
        s.artist as song_artist,
        s.artist_address
      FROM ${this.tableName} p
      JOIN songs s ON p.song_id = s.id
      WHERE p.listener_address = $1 AND p.status = 'confirmed'
      ORDER BY p.created_at DESC
      LIMIT $2 OFFSET $3
    `;
    const result = await this.query<PlayWithSong>(
      sql,
      [listenerAddress, limit, offset],
      options
    );
    return result.rows;
  }

  /**
   * Get recent plays (global feed)
   */
  async getRecentPlays(limit = 20): Promise<PlayWithSong[]> {
    const sql = `
      SELECT
        p.*,
        s.title as song_title,
        s.artist as song_artist,
        s.artist_address
      FROM ${this.tableName} p
      JOIN songs s ON p.song_id = s.id
      WHERE p.status = 'confirmed'
      ORDER BY p.created_at DESC
      LIMIT $1
    `;
    const result = await this.query<PlayWithSong>(sql, [limit]);
    return result.rows;
  }

  /**
   * Get plays within a time range
   */
  async getPlaysInRange(range: TimeRange): Promise<Play[]> {
    const sql = `
      SELECT * FROM ${this.tableName}
      WHERE status = 'confirmed'
        AND created_at >= $1
        AND created_at < $2
      ORDER BY created_at DESC
    `;
    const result = await this.query<Play>(sql, [range.start, range.end]);
    return result.rows;
  }

  /**
   * Get play count for a song in a time range
   */
  async countPlaysForSong(
    songId: string,
    range?: TimeRange
  ): Promise<number> {
    let sql = `
      SELECT COUNT(*) FROM ${this.tableName}
      WHERE song_id = $1 AND status = 'confirmed'
    `;
    const params: unknown[] = [songId];

    if (range) {
      sql += ` AND created_at >= $2 AND created_at < $3`;
      params.push(range.start, range.end);
    }

    const result = await this.query<{ count: string }>(sql, params);
    return parseInt(result.rows[0].count, 10);
  }

  /**
   * Get total earnings for an artist in a time range
   */
  async getArtistEarningsInRange(
    artistAddress: string,
    range: TimeRange
  ): Promise<string> {
    const sql = `
      SELECT COALESCE(SUM(p.amount), 0) as total
      FROM ${this.tableName} p
      JOIN songs s ON p.song_id = s.id
      WHERE s.artist_address = $1
        AND p.status = 'confirmed'
        AND p.created_at >= $2
        AND p.created_at < $3
    `;
    const result = await this.query<{ total: string }>(
      sql,
      [artistAddress, range.start, range.end]
    );
    return result.rows[0].total;
  }

  /**
   * Get daily play statistics
   */
  async getDailyStats(days = 30): Promise<{
    date: string;
    plays: number;
    earnings: string;
    unique_listeners: number;
  }[]> {
    const sql = `
      SELECT
        DATE(created_at) as date,
        COUNT(*) as plays,
        SUM(amount) as earnings,
        COUNT(DISTINCT listener_address) as unique_listeners
      FROM ${this.tableName}
      WHERE status = 'confirmed'
        AND created_at >= NOW() - INTERVAL '${days} days'
      GROUP BY DATE(created_at)
      ORDER BY date DESC
    `;
    const result = await this.query<{
      date: Date;
      plays: string;
      earnings: string;
      unique_listeners: string;
    }>(sql);

    return result.rows.map(row => ({
      date: row.date.toISOString().split('T')[0],
      plays: parseInt(row.plays, 10),
      earnings: row.earnings,
      unique_listeners: parseInt(row.unique_listeners, 10),
    }));
  }

  /**
   * Get hourly play statistics for the last 24 hours
   */
  async getHourlyStats(): Promise<{
    hour: string;
    plays: number;
    earnings: string;
  }[]> {
    const sql = `
      SELECT
        DATE_TRUNC('hour', created_at) as hour,
        COUNT(*) as plays,
        SUM(amount) as earnings
      FROM ${this.tableName}
      WHERE status = 'confirmed'
        AND created_at >= NOW() - INTERVAL '24 hours'
      GROUP BY DATE_TRUNC('hour', created_at)
      ORDER BY hour DESC
    `;
    const result = await this.query<{
      hour: Date;
      plays: string;
      earnings: string;
    }>(sql);

    return result.rows.map(row => ({
      hour: row.hour.toISOString(),
      plays: parseInt(row.plays, 10),
      earnings: row.earnings,
    }));
  }

  /**
   * Get top listeners by play count
   */
  async getTopListeners(limit = 10): Promise<{
    listener_address: string;
    play_count: number;
    total_spent: string;
  }[]> {
    const sql = `
      SELECT
        listener_address,
        COUNT(*) as play_count,
        SUM(amount) as total_spent
      FROM ${this.tableName}
      WHERE status = 'confirmed'
      GROUP BY listener_address
      ORDER BY play_count DESC
      LIMIT $1
    `;
    const result = await this.query<{
      listener_address: string;
      play_count: string;
      total_spent: string;
    }>(sql, [limit]);

    return result.rows.map(row => ({
      listener_address: row.listener_address,
      play_count: parseInt(row.play_count, 10),
      total_spent: row.total_spent,
    }));
  }

  /**
   * Get pending plays older than threshold (for cleanup/retry)
   */
  async getPendingPlays(olderThanMinutes = 30): Promise<Play[]> {
    const sql = `
      SELECT * FROM ${this.tableName}
      WHERE status = 'pending'
        AND created_at < NOW() - INTERVAL '${olderThanMinutes} minutes'
      ORDER BY created_at ASC
    `;
    const result = await this.query<Play>(sql);
    return result.rows;
  }
}
