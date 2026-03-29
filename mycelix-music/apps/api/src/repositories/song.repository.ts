// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Song Repository
 *
 * Handles all database operations for songs with
 * domain-specific query methods.
 */

import { Pool, PoolClient } from 'pg';
import { BaseRepository, PaginatedResult, QueryOptions, FilterOptions } from './base.repository';

export interface Song {
  id: string;
  title: string;
  artist: string;
  artist_address: string;
  genre: string;
  description?: string;
  ipfs_hash: string;
  song_hash?: string;
  cover_art?: string;
  audio_url?: string;
  payment_model: 'per_play' | 'subscription' | 'tip';
  claim_stream_id?: string;
  plays: number;
  earnings: string;
  created_at: Date;
  updated_at?: Date;
}

export interface CreateSongInput {
  title: string;
  artist: string;
  artist_address: string;
  genre: string;
  description?: string;
  ipfs_hash: string;
  song_hash?: string;
  cover_art?: string;
  audio_url?: string;
  payment_model: 'per_play' | 'subscription' | 'tip';
}

export interface SongSearchOptions {
  query?: string;
  genre?: string;
  artistAddress?: string;
  paymentModel?: string;
}

export class SongRepository extends BaseRepository<Song> {
  constructor(pool: Pool) {
    super(pool, 'songs');
  }

  /**
   * Find songs by artist address
   */
  async findByArtist(
    artistAddress: string,
    options: QueryOptions = {}
  ): Promise<Song[]> {
    const sql = `
      SELECT * FROM ${this.tableName}
      WHERE artist_address = $1
      ORDER BY created_at DESC
    `;
    const result = await this.query<Song>(sql, [artistAddress], options);
    return result.rows;
  }

  /**
   * Find songs by genre
   */
  async findByGenre(
    genre: string,
    limit = 50,
    options: QueryOptions = {}
  ): Promise<Song[]> {
    const sql = `
      SELECT * FROM ${this.tableName}
      WHERE genre = $1
      ORDER BY plays DESC, created_at DESC
      LIMIT $2
    `;
    const result = await this.query<Song>(sql, [genre, limit], options);
    return result.rows;
  }

  /**
   * Search songs with full-text search
   */
  async search(
    searchOptions: SongSearchOptions,
    limit = 50,
    offset = 0
  ): Promise<PaginatedResult<Song>> {
    const conditions: string[] = [];
    const params: unknown[] = [];
    let paramIndex = 1;

    if (searchOptions.query) {
      conditions.push(`(
        title ILIKE $${paramIndex} OR
        artist ILIKE $${paramIndex} OR
        description ILIKE $${paramIndex}
      )`);
      params.push(`%${searchOptions.query}%`);
      paramIndex++;
    }

    if (searchOptions.genre) {
      conditions.push(`genre = $${paramIndex}`);
      params.push(searchOptions.genre);
      paramIndex++;
    }

    if (searchOptions.artistAddress) {
      conditions.push(`artist_address = $${paramIndex}`);
      params.push(searchOptions.artistAddress);
      paramIndex++;
    }

    if (searchOptions.paymentModel) {
      conditions.push(`payment_model = $${paramIndex}`);
      params.push(searchOptions.paymentModel);
      paramIndex++;
    }

    const whereClause = conditions.length > 0
      ? `WHERE ${conditions.join(' AND ')}`
      : '';

    // Count total
    const countSql = `SELECT COUNT(*) FROM ${this.tableName} ${whereClause}`;
    const countResult = await this.query<{ count: string }>(countSql, params);
    const total = parseInt(countResult.rows[0].count, 10);

    // Get paginated data
    const dataSql = `
      SELECT * FROM ${this.tableName}
      ${whereClause}
      ORDER BY plays DESC, created_at DESC
      LIMIT $${paramIndex} OFFSET $${paramIndex + 1}
    `;
    const dataResult = await this.query<Song>(
      dataSql,
      [...params, limit, offset]
    );

    return {
      data: dataResult.rows,
      total,
      limit,
      offset,
      hasMore: offset + dataResult.rows.length < total,
    };
  }

  /**
   * Get top songs by plays
   */
  async getTopByPlays(limit = 10, options: QueryOptions = {}): Promise<Song[]> {
    const sql = `
      SELECT * FROM ${this.tableName}
      ORDER BY plays DESC
      LIMIT $1
    `;
    const result = await this.query<Song>(sql, [limit], options);
    return result.rows;
  }

  /**
   * Get top songs by earnings
   */
  async getTopByEarnings(limit = 10, options: QueryOptions = {}): Promise<Song[]> {
    const sql = `
      SELECT * FROM ${this.tableName}
      ORDER BY earnings DESC
      LIMIT $1
    `;
    const result = await this.query<Song>(sql, [limit], options);
    return result.rows;
  }

  /**
   * Get recently added songs
   */
  async getRecent(limit = 10, options: QueryOptions = {}): Promise<Song[]> {
    const sql = `
      SELECT * FROM ${this.tableName}
      ORDER BY created_at DESC
      LIMIT $1
    `;
    const result = await this.query<Song>(sql, [limit], options);
    return result.rows;
  }

  /**
   * Increment play count and add earnings
   */
  async recordPlay(
    songId: string,
    earnings: string,
    client?: PoolClient
  ): Promise<Song | null> {
    const sql = `
      UPDATE ${this.tableName}
      SET plays = plays + 1,
          earnings = earnings + $2::numeric,
          updated_at = NOW()
      WHERE id = $1
      RETURNING *
    `;
    const result = await this.query<Song>(
      sql,
      [songId, earnings],
      { client }
    );
    return result.rows[0] || null;
  }

  /**
   * Update claim stream ID after blockchain registration
   */
  async setClaimStreamId(
    songId: string,
    claimStreamId: string,
    options: QueryOptions = {}
  ): Promise<Song | null> {
    const sql = `
      UPDATE ${this.tableName}
      SET claim_stream_id = $2,
          updated_at = NOW()
      WHERE id = $1
      RETURNING *
    `;
    const result = await this.query<Song>(sql, [songId, claimStreamId], options);
    return result.rows[0] || null;
  }

  /**
   * Get genre distribution
   */
  async getGenreStats(): Promise<{ genre: string; count: number; total_plays: number }[]> {
    const sql = `
      SELECT
        genre,
        COUNT(*) as count,
        SUM(plays) as total_plays
      FROM ${this.tableName}
      GROUP BY genre
      ORDER BY count DESC
    `;
    const result = await this.query<{ genre: string; count: string; total_plays: string }>(sql);
    return result.rows.map(row => ({
      genre: row.genre,
      count: parseInt(row.count, 10),
      total_plays: parseInt(row.total_plays || '0', 10),
    }));
  }

  /**
   * Get artist statistics
   */
  async getArtistStats(artistAddress: string): Promise<{
    total_songs: number;
    total_plays: number;
    total_earnings: string;
  } | null> {
    const sql = `
      SELECT
        COUNT(*) as total_songs,
        COALESCE(SUM(plays), 0) as total_plays,
        COALESCE(SUM(earnings), 0) as total_earnings
      FROM ${this.tableName}
      WHERE artist_address = $1
    `;
    const result = await this.query<{
      total_songs: string;
      total_plays: string;
      total_earnings: string;
    }>(sql, [artistAddress]);

    const row = result.rows[0];
    if (!row || parseInt(row.total_songs, 10) === 0) {
      return null;
    }

    return {
      total_songs: parseInt(row.total_songs, 10),
      total_plays: parseInt(row.total_plays, 10),
      total_earnings: row.total_earnings,
    };
  }
}
