// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Migration: Add Analytics Views
 * Version: 2
 * Created: 2024-01-15
 *
 * Creates materialized views for analytics dashboards.
 */

import { PoolClient } from 'pg';
import { Migration } from './migrator';

const migration: Migration = {
  version: 2,
  name: 'add_analytics_views',

  async up(client: PoolClient): Promise<void> {
    // Daily play statistics view
    await client.query(`
      CREATE MATERIALIZED VIEW IF NOT EXISTS daily_play_stats AS
      SELECT
        DATE(created_at) as date,
        COUNT(*) as total_plays,
        SUM(amount) as total_earnings,
        COUNT(DISTINCT listener_address) as unique_listeners,
        COUNT(DISTINCT song_id) as unique_songs
      FROM plays
      WHERE status = 'confirmed'
      GROUP BY DATE(created_at)
      ORDER BY date DESC
    `);

    // Create index on the materialized view
    await client.query(`
      CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_play_stats_date
        ON daily_play_stats(date)
    `);

    // Artist statistics view
    await client.query(`
      CREATE MATERIALIZED VIEW IF NOT EXISTS artist_stats AS
      SELECT
        s.artist_address,
        s.artist,
        COUNT(DISTINCT s.id) as total_songs,
        COALESCE(SUM(s.plays), 0) as total_plays,
        COALESCE(SUM(s.earnings), 0) as total_earnings,
        MIN(s.created_at) as first_song_at,
        MAX(s.created_at) as latest_song_at
      FROM songs s
      GROUP BY s.artist_address, s.artist
      ORDER BY total_earnings DESC
    `);

    await client.query(`
      CREATE UNIQUE INDEX IF NOT EXISTS idx_artist_stats_address
        ON artist_stats(artist_address)
    `);

    // Genre statistics view
    await client.query(`
      CREATE MATERIALIZED VIEW IF NOT EXISTS genre_stats AS
      SELECT
        genre,
        COUNT(*) as song_count,
        SUM(plays) as total_plays,
        SUM(earnings) as total_earnings,
        AVG(plays) as avg_plays,
        AVG(earnings) as avg_earnings
      FROM songs
      GROUP BY genre
      ORDER BY song_count DESC
    `);

    await client.query(`
      CREATE UNIQUE INDEX IF NOT EXISTS idx_genre_stats_genre
        ON genre_stats(genre)
    `);

    // Top songs view (refreshed more frequently)
    await client.query(`
      CREATE MATERIALIZED VIEW IF NOT EXISTS top_songs AS
      SELECT
        id,
        title,
        artist,
        artist_address,
        genre,
        plays,
        earnings,
        created_at,
        RANK() OVER (ORDER BY plays DESC) as plays_rank,
        RANK() OVER (ORDER BY earnings DESC) as earnings_rank
      FROM songs
      ORDER BY plays DESC
      LIMIT 100
    `);

    await client.query(`
      CREATE UNIQUE INDEX IF NOT EXISTS idx_top_songs_id ON top_songs(id)
    `);

    // Create function to refresh materialized views
    await client.query(`
      CREATE OR REPLACE FUNCTION refresh_analytics_views()
      RETURNS void AS $$
      BEGIN
        REFRESH MATERIALIZED VIEW CONCURRENTLY daily_play_stats;
        REFRESH MATERIALIZED VIEW CONCURRENTLY artist_stats;
        REFRESH MATERIALIZED VIEW CONCURRENTLY genre_stats;
        REFRESH MATERIALIZED VIEW CONCURRENTLY top_songs;
      END;
      $$ LANGUAGE plpgsql
    `);
  },

  async down(client: PoolClient): Promise<void> {
    // Drop function
    await client.query(`DROP FUNCTION IF EXISTS refresh_analytics_views()`);

    // Drop materialized views
    await client.query(`DROP MATERIALIZED VIEW IF EXISTS top_songs`);
    await client.query(`DROP MATERIALIZED VIEW IF EXISTS genre_stats`);
    await client.query(`DROP MATERIALIZED VIEW IF EXISTS artist_stats`);
    await client.query(`DROP MATERIALIZED VIEW IF EXISTS daily_play_stats`);
  },
};

export default migration;
