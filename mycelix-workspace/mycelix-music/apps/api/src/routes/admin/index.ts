// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Admin API Routes
 *
 * Protected administrative endpoints for system management.
 * Requires admin authentication.
 */

import { Router, Request, Response, NextFunction } from 'express';
import { Pool } from 'pg';
import { z } from 'zod';
import { validate } from '../../middleware/validate';
import { success, errors } from '../../utils/response';
import { getLogger } from '../../logging';
import { getMetrics, collectSystemMetrics } from '../../metrics';
import { getCircuitBreakerStats, resetAllCircuitBreakers } from '../../resilience/circuit-breaker';
import { getFeatureFlags } from '../../features';
import { getQueryMonitor } from '../../db/query-monitor';
import { getHealthChecker } from '../../health';

const logger = getLogger();

/**
 * Admin authentication middleware
 */
function adminAuth() {
  const adminAddresses = (process.env.ADMIN_ADDRESSES || '')
    .toLowerCase()
    .split(',')
    .filter(Boolean);

  const adminApiKey = process.env.ADMIN_API_KEY;

  return (req: Request, res: Response, next: NextFunction): void => {
    // Check API key
    const apiKey = req.headers['x-admin-key'] as string;
    if (apiKey && apiKey === adminApiKey) {
      return next();
    }

    // Check wallet address
    const walletAddress = req.auth?.address?.toLowerCase();
    if (walletAddress && adminAddresses.includes(walletAddress)) {
      return next();
    }

    errors.forbidden(res, 'Admin access required');
  };
}

/**
 * Create admin routes
 */
export function createAdminRoutes(pool: Pool): Router {
  const router = Router();

  // All admin routes require authentication
  router.use(adminAuth());

  /**
   * GET /admin/dashboard
   * Admin dashboard overview
   */
  router.get('/dashboard', async (req: Request, res: Response) => {
    try {
      // Collect system metrics
      collectSystemMetrics();

      // Get database stats
      const dbStats = await pool.query(`
        SELECT
          (SELECT COUNT(*) FROM songs WHERE deleted_at IS NULL) as total_songs,
          (SELECT COUNT(*) FROM plays) as total_plays,
          (SELECT COUNT(DISTINCT wallet_address) FROM plays) as unique_listeners,
          (SELECT COUNT(DISTINCT artist_address) FROM songs WHERE deleted_at IS NULL) as total_artists
      `);

      // Get recent activity
      const recentActivity = await pool.query(`
        SELECT
          'play' as type,
          p.id,
          s.title as song_title,
          p.wallet_address,
          p.played_at as timestamp
        FROM plays p
        JOIN songs s ON p.song_id = s.id
        ORDER BY p.played_at DESC
        LIMIT 10
      `);

      // Get circuit breaker status
      const circuitBreakers = getCircuitBreakerStats();

      // Get health status
      const health = await getHealthChecker().getDetailedHealth();

      success(res, {
        stats: dbStats.rows[0],
        recentActivity: recentActivity.rows,
        circuitBreakers,
        health,
        serverTime: new Date().toISOString(),
      });
    } catch (error) {
      logger.error('Admin dashboard error', error as Error);
      errors.internal(res);
    }
  });

  /**
   * GET /admin/users
   * List users with stats
   */
  router.get('/users', async (req: Request, res: Response) => {
    const limit = Math.min(parseInt(req.query.limit as string) || 50, 100);
    const offset = parseInt(req.query.offset as string) || 0;
    const search = req.query.search as string;

    try {
      let query = `
        SELECT
          wallet_address,
          COUNT(DISTINCT song_id) as unique_songs_played,
          COUNT(*) as total_plays,
          MAX(played_at) as last_active,
          MIN(played_at) as first_seen
        FROM plays
      `;

      const params: unknown[] = [];

      if (search) {
        query += ` WHERE wallet_address ILIKE $1`;
        params.push(`%${search}%`);
      }

      query += `
        GROUP BY wallet_address
        ORDER BY total_plays DESC
        LIMIT $${params.length + 1} OFFSET $${params.length + 2}
      `;

      params.push(limit, offset);

      const result = await pool.query(query, params);

      const countResult = await pool.query(`
        SELECT COUNT(DISTINCT wallet_address) as total FROM plays
        ${search ? 'WHERE wallet_address ILIKE $1' : ''}
      `, search ? [`%${search}%`] : []);

      success(res, {
        users: result.rows,
        pagination: {
          total: parseInt(countResult.rows[0].total),
          limit,
          offset,
          hasMore: offset + result.rows.length < parseInt(countResult.rows[0].total),
        },
      });
    } catch (error) {
      logger.error('Admin users error', error as Error);
      errors.internal(res);
    }
  });

  /**
   * GET /admin/artists
   * List artists with stats
   */
  router.get('/artists', async (req: Request, res: Response) => {
    const limit = Math.min(parseInt(req.query.limit as string) || 50, 100);
    const offset = parseInt(req.query.offset as string) || 0;

    try {
      const result = await pool.query(`
        SELECT
          s.artist_address,
          s.artist_name,
          COUNT(DISTINCT s.id) as song_count,
          COALESCE(SUM(s.play_count), 0) as total_plays,
          MAX(s.created_at) as last_upload,
          COUNT(DISTINCT p.wallet_address) as unique_listeners
        FROM songs s
        LEFT JOIN plays p ON s.id = p.song_id
        WHERE s.deleted_at IS NULL
        GROUP BY s.artist_address, s.artist_name
        ORDER BY total_plays DESC
        LIMIT $1 OFFSET $2
      `, [limit, offset]);

      const countResult = await pool.query(`
        SELECT COUNT(DISTINCT artist_address) as total FROM songs WHERE deleted_at IS NULL
      `);

      success(res, {
        artists: result.rows,
        pagination: {
          total: parseInt(countResult.rows[0].total),
          limit,
          offset,
          hasMore: offset + result.rows.length < parseInt(countResult.rows[0].total),
        },
      });
    } catch (error) {
      logger.error('Admin artists error', error as Error);
      errors.internal(res);
    }
  });

  /**
   * GET /admin/content/moderation
   * Content needing moderation
   */
  router.get('/content/moderation', async (req: Request, res: Response) => {
    try {
      const result = await pool.query(`
        SELECT
          s.*,
          COALESCE(r.report_count, 0) as report_count
        FROM songs s
        LEFT JOIN (
          SELECT song_id, COUNT(*) as report_count
          FROM content_reports
          WHERE resolved_at IS NULL
          GROUP BY song_id
        ) r ON s.id = r.song_id
        WHERE s.deleted_at IS NULL
        AND (r.report_count > 0 OR s.moderation_status = 'pending')
        ORDER BY r.report_count DESC NULLS LAST, s.created_at DESC
        LIMIT 50
      `);

      success(res, { items: result.rows });
    } catch (error) {
      logger.error('Admin moderation error', error as Error);
      errors.internal(res);
    }
  });

  /**
   * POST /admin/content/:id/moderate
   * Moderate content
   */
  router.post(
    '/content/:id/moderate',
    validate(z.object({
      action: z.enum(['approve', 'reject', 'flag', 'delete']),
      reason: z.string().optional(),
    })),
    async (req: Request, res: Response) => {
      const { id } = req.params;
      const { action, reason } = req.body;

      try {
        let status: string;
        switch (action) {
          case 'approve':
            status = 'approved';
            break;
          case 'reject':
            status = 'rejected';
            break;
          case 'flag':
            status = 'flagged';
            break;
          case 'delete':
            await pool.query(
              'UPDATE songs SET deleted_at = NOW() WHERE id = $1',
              [id]
            );
            success(res, { message: 'Content deleted' });
            return;
          default:
            status = 'pending';
        }

        await pool.query(`
          UPDATE songs
          SET
            moderation_status = $2,
            moderation_reason = $3,
            moderated_at = NOW(),
            moderated_by = $4
          WHERE id = $1
        `, [id, status, reason, req.auth?.address]);

        // Resolve any reports
        await pool.query(`
          UPDATE content_reports
          SET resolved_at = NOW(), resolution = $2
          WHERE song_id = $1 AND resolved_at IS NULL
        `, [id, action]);

        logger.info('Content moderated', {
          songId: id,
          action,
          moderator: req.auth?.address,
        });

        success(res, { message: `Content ${action}d` });
      } catch (error) {
        logger.error('Admin moderate error', error as Error);
        errors.internal(res);
      }
    }
  );

  /**
   * GET /admin/system/metrics
   * System metrics
   */
  router.get('/system/metrics', async (req: Request, res: Response) => {
    collectSystemMetrics();

    const queryMonitor = getQueryMonitor();

    success(res, {
      queries: {
        summary: queryMonitor.getSummary(),
        slowQueries: queryMonitor.getSlowQueries(20),
        topByTime: queryMonitor.getTopByTotalTime(10),
      },
      circuitBreakers: getCircuitBreakerStats(),
      memory: process.memoryUsage(),
      uptime: process.uptime(),
    });
  });

  /**
   * GET /admin/system/features
   * Feature flags status
   */
  router.get('/system/features', (req: Request, res: Response) => {
    const flags = getFeatureFlags();

    success(res, {
      flags: flags.getAllFlags(),
      summary: flags.getSummary(),
    });
  });

  /**
   * PUT /admin/system/features/:name
   * Update feature flag
   */
  router.put(
    '/system/features/:name',
    validate(z.object({
      enabled: z.boolean().optional(),
      percentage: z.number().min(0).max(100).optional(),
      targetAddresses: z.array(z.string()).optional(),
    })),
    (req: Request, res: Response) => {
      const { name } = req.params;
      const flags = getFeatureFlags();

      try {
        flags.updateFlag(name, req.body);
        logger.info('Feature flag updated', { flag: name, updates: req.body });
        success(res, flags.getFlag(name));
      } catch (error) {
        errors.notFound(res, 'Feature flag not found');
      }
    }
  );

  /**
   * POST /admin/system/circuit-breakers/reset
   * Reset all circuit breakers
   */
  router.post('/system/circuit-breakers/reset', (req: Request, res: Response) => {
    resetAllCircuitBreakers();
    logger.info('Circuit breakers reset', { by: req.auth?.address });
    success(res, { message: 'Circuit breakers reset' });
  });

  /**
   * GET /admin/analytics/overview
   * Analytics overview
   */
  router.get('/analytics/overview', async (req: Request, res: Response) => {
    const days = parseInt(req.query.days as string) || 30;

    try {
      const result = await pool.query(`
        SELECT
          date_trunc('day', played_at) as date,
          COUNT(*) as plays,
          COUNT(DISTINCT wallet_address) as unique_listeners,
          COUNT(DISTINCT song_id) as unique_songs
        FROM plays
        WHERE played_at > NOW() - INTERVAL '${days} days'
        GROUP BY date_trunc('day', played_at)
        ORDER BY date DESC
      `);

      const topSongs = await pool.query(`
        SELECT
          s.id,
          s.title,
          s.artist_name,
          COUNT(p.id) as plays
        FROM plays p
        JOIN songs s ON p.song_id = s.id
        WHERE p.played_at > NOW() - INTERVAL '${days} days'
        GROUP BY s.id, s.title, s.artist_name
        ORDER BY plays DESC
        LIMIT 10
      `);

      const topGenres = await pool.query(`
        SELECT
          s.genre,
          COUNT(p.id) as plays
        FROM plays p
        JOIN songs s ON p.song_id = s.id
        WHERE p.played_at > NOW() - INTERVAL '${days} days'
        AND s.genre IS NOT NULL
        GROUP BY s.genre
        ORDER BY plays DESC
        LIMIT 10
      `);

      success(res, {
        daily: result.rows,
        topSongs: topSongs.rows,
        topGenres: topGenres.rows,
        period: { days, from: new Date(Date.now() - days * 86400000) },
      });
    } catch (error) {
      logger.error('Admin analytics error', error as Error);
      errors.internal(res);
    }
  });

  /**
   * POST /admin/maintenance/cleanup
   * Run maintenance cleanup
   */
  router.post('/maintenance/cleanup', async (req: Request, res: Response) => {
    try {
      // Clean old soft-deleted records
      const deleted = await pool.query(`
        DELETE FROM songs
        WHERE deleted_at < NOW() - INTERVAL '30 days'
        RETURNING id
      `);

      // Clean old play records (optional, based on config)
      // await pool.query(`DELETE FROM plays WHERE played_at < NOW() - INTERVAL '1 year'`);

      // Vacuum analyze
      await pool.query('VACUUM ANALYZE songs');
      await pool.query('VACUUM ANALYZE plays');

      logger.info('Maintenance cleanup completed', {
        deletedSongs: deleted.rowCount,
        by: req.auth?.address,
      });

      success(res, {
        message: 'Cleanup completed',
        deletedSongs: deleted.rowCount,
      });
    } catch (error) {
      logger.error('Admin cleanup error', error as Error);
      errors.internal(res);
    }
  });

  /**
   * GET /admin/audit-log
   * View audit log
   */
  router.get('/audit-log', async (req: Request, res: Response) => {
    const limit = Math.min(parseInt(req.query.limit as string) || 100, 500);
    const offset = parseInt(req.query.offset as string) || 0;
    const action = req.query.action as string;

    try {
      let query = `
        SELECT * FROM audit_log
        ${action ? 'WHERE action = $1' : ''}
        ORDER BY created_at DESC
        LIMIT $${action ? 2 : 1} OFFSET $${action ? 3 : 2}
      `;

      const params = action
        ? [action, limit, offset]
        : [limit, offset];

      const result = await pool.query(query, params);

      success(res, {
        logs: result.rows,
        pagination: { limit, offset },
      });
    } catch (error) {
      logger.error('Admin audit log error', error as Error);
      errors.internal(res);
    }
  });

  return router;
}

export default createAdminRoutes;
