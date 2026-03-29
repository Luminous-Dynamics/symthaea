// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Analytics Jobs
 *
 * Background jobs for analytics processing:
 * - Refresh materialized views
 * - Calculate daily/weekly stats
 * - Clean up old data
 */

import { Pool } from 'pg';
import { JobDefinition } from './scheduler';

/**
 * Create analytics refresh job
 */
export function createAnalyticsRefreshJob(pool: Pool): JobDefinition {
  return {
    name: 'analytics:refresh',
    schedule: {
      type: 'interval',
      value: 5 * 60 * 1000, // Every 5 minutes
    },
    timeout: 120000, // 2 minutes
    retries: 2,
    handler: async () => {
      console.log('[Job] Refreshing analytics views...');

      const client = await pool.connect();
      try {
        // Refresh materialized views
        await client.query('SELECT refresh_analytics_views()');
        console.log('[Job] Analytics views refreshed');
      } finally {
        client.release();
      }
    },
  };
}

/**
 * Create top songs cache refresh job
 */
export function createTopSongsCacheJob(pool: Pool): JobDefinition {
  return {
    name: 'cache:top-songs',
    schedule: {
      type: 'interval',
      value: 60 * 1000, // Every minute
    },
    timeout: 30000,
    handler: async () => {
      const client = await pool.connect();
      try {
        // Refresh top songs materialized view only
        await client.query('REFRESH MATERIALIZED VIEW CONCURRENTLY top_songs');
      } finally {
        client.release();
      }
    },
  };
}

/**
 * Create pending plays cleanup job
 */
export function createPendingPlaysCleanupJob(pool: Pool): JobDefinition {
  return {
    name: 'cleanup:pending-plays',
    schedule: {
      type: 'interval',
      value: 15 * 60 * 1000, // Every 15 minutes
    },
    timeout: 60000,
    handler: async () => {
      console.log('[Job] Cleaning up old pending plays...');

      const client = await pool.connect();
      try {
        // Mark plays pending for more than 1 hour as failed
        const result = await client.query(`
          UPDATE plays
          SET status = 'failed'
          WHERE status = 'pending'
            AND created_at < NOW() - INTERVAL '1 hour'
          RETURNING id
        `);

        if (result.rowCount && result.rowCount > 0) {
          console.log(`[Job] Marked ${result.rowCount} pending plays as failed`);
        }
      } finally {
        client.release();
      }
    },
  };
}

/**
 * Create daily stats aggregation job
 */
export function createDailyStatsJob(pool: Pool): JobDefinition {
  return {
    name: 'stats:daily-aggregate',
    schedule: {
      type: 'interval',
      value: 60 * 60 * 1000, // Every hour
    },
    timeout: 300000, // 5 minutes
    handler: async () => {
      console.log('[Job] Aggregating daily stats...');

      const client = await pool.connect();
      try {
        // Refresh daily play stats
        await client.query('REFRESH MATERIALIZED VIEW CONCURRENTLY daily_play_stats');

        // Refresh artist stats
        await client.query('REFRESH MATERIALIZED VIEW CONCURRENTLY artist_stats');

        // Refresh genre stats
        await client.query('REFRESH MATERIALIZED VIEW CONCURRENTLY genre_stats');

        console.log('[Job] Daily stats aggregated');
      } finally {
        client.release();
      }
    },
  };
}

/**
 * Create database vacuum job (for production)
 */
export function createVacuumJob(pool: Pool): JobDefinition {
  return {
    name: 'maintenance:vacuum',
    schedule: {
      type: 'interval',
      value: 24 * 60 * 60 * 1000, // Daily
    },
    timeout: 600000, // 10 minutes
    enabled: process.env.NODE_ENV === 'production',
    handler: async () => {
      console.log('[Job] Running database maintenance...');

      const client = await pool.connect();
      try {
        // Analyze tables for query optimization
        await client.query('ANALYZE songs');
        await client.query('ANALYZE plays');

        console.log('[Job] Database maintenance complete');
      } finally {
        client.release();
      }
    },
  };
}

/**
 * Create expired cache cleanup job
 */
export function createCacheCleanupJob(): JobDefinition {
  return {
    name: 'cleanup:cache',
    schedule: {
      type: 'interval',
      value: 10 * 60 * 1000, // Every 10 minutes
    },
    timeout: 30000,
    handler: async () => {
      // This is handled by Redis TTL or MemoryCache cleanup
      // Placeholder for any additional cache cleanup logic
      console.log('[Job] Cache cleanup check');
    },
  };
}

/**
 * Register all analytics jobs
 */
export function registerAnalyticsJobs(
  scheduler: import('./scheduler').JobScheduler,
  pool: Pool
): void {
  scheduler.register(createAnalyticsRefreshJob(pool));
  scheduler.register(createTopSongsCacheJob(pool));
  scheduler.register(createPendingPlaysCleanupJob(pool));
  scheduler.register(createDailyStatsJob(pool));
  scheduler.register(createVacuumJob(pool));
  scheduler.register(createCacheCleanupJob());

  console.log('[Scheduler] Analytics jobs registered');
}
