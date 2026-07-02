// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Artist Dashboard Routes
 *
 * API endpoints for artist analytics, revenue tracking,
 * and content management.
 */

import { Router, Request, Response } from 'express';
import { Pool } from 'pg';
import { z } from 'zod';
import { requireAuth, requireArtist } from '../../middleware/auth';
import { validateRequest } from '../../middleware/validation';
import { ArtistAnalyticsService, TimePeriod } from '../../services/artist-analytics.service';
import { responseEnvelope } from '../../utils/response';
import { getLogger } from '../../logging';

const logger = getLogger();

/**
 * Create dashboard routes
 */
export function createDashboardRoutes(pool: Pool): Router {
  const router = Router();
  const analytics = new ArtistAnalyticsService(pool);

  // All routes require artist authentication
  router.use(requireAuth, requireArtist);

  /**
   * GET /dashboard
   * Get full dashboard data
   */
  router.get(
    '/',
    validateRequest({
      query: z.object({
        period: z.enum(['24h', '7d', '30d', '90d', '1y', 'all']).optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const period = (req.query.period as TimePeriod) || '30d';

      const dashboard = await analytics.getDashboard(req.auth!.address, period);

      res.json(
        responseEnvelope({
          data: dashboard,
        })
      );
    }
  );

  /**
   * GET /dashboard/realtime
   * Get real-time stats
   */
  router.get('/realtime', async (req: Request, res: Response) => {
    const stats = await analytics.getRealTimeStats(req.auth!.address);

    res.json(
      responseEnvelope({
        data: stats,
      })
    );
  });

  /**
   * GET /dashboard/streams
   * Get detailed stream analytics
   */
  router.get(
    '/streams',
    validateRequest({
      query: z.object({
        period: z.enum(['24h', '7d', '30d', '90d', '1y', 'all']).optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const period = (req.query.period as TimePeriod) || '30d';

      const [streams, history] = await Promise.all([
        analytics.getStreamAnalytics(req.auth!.address, period),
        analytics.getStreamHistory(req.auth!.address, period),
      ]);

      res.json(
        responseEnvelope({
          data: {
            analytics: streams,
            history,
          },
        })
      );
    }
  );

  /**
   * GET /dashboard/revenue
   * Get revenue analytics
   */
  router.get(
    '/revenue',
    validateRequest({
      query: z.object({
        period: z.enum(['24h', '7d', '30d', '90d', '1y', 'all']).optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const period = (req.query.period as TimePeriod) || '30d';

      const [revenue, history] = await Promise.all([
        analytics.getRevenueAnalytics(req.auth!.address, period),
        analytics.getRevenueHistory(req.auth!.address, period),
      ]);

      res.json(
        responseEnvelope({
          data: {
            analytics: revenue,
            history,
          },
        })
      );
    }
  );

  /**
   * GET /dashboard/audience
   * Get audience demographics
   */
  router.get(
    '/audience',
    validateRequest({
      query: z.object({
        period: z.enum(['24h', '7d', '30d', '90d', '1y', 'all']).optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const period = (req.query.period as TimePeriod) || '30d';

      const demographics = await analytics.getAudienceDemographics(
        req.auth!.address,
        period
      );

      res.json(
        responseEnvelope({
          data: demographics,
        })
      );
    }
  );

  /**
   * GET /dashboard/songs
   * Get song performance
   */
  router.get(
    '/songs',
    validateRequest({
      query: z.object({
        period: z.enum(['24h', '7d', '30d', '90d', '1y', 'all']).optional(),
        limit: z.coerce.number().min(1).max(100).optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const period = (req.query.period as TimePeriod) || '30d';
      const limit = parseInt(req.query.limit as string) || 20;

      const songs = await analytics.getTopSongs(req.auth!.address, period, limit);

      res.json(
        responseEnvelope({
          data: songs,
        })
      );
    }
  );

  /**
   * GET /dashboard/engagement
   * Get engagement metrics
   */
  router.get(
    '/engagement',
    validateRequest({
      query: z.object({
        period: z.enum(['24h', '7d', '30d', '90d', '1y', 'all']).optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const period = (req.query.period as TimePeriod) || '30d';

      const [engagement, followerHistory] = await Promise.all([
        analytics.getEngagementMetrics(req.auth!.address, period),
        analytics.getFollowerHistory(req.auth!.address, period),
      ]);

      res.json(
        responseEnvelope({
          data: {
            metrics: engagement,
            followerHistory,
          },
        })
      );
    }
  );

  /**
   * GET /dashboard/content
   * Get artist's content (songs, albums)
   */
  router.get(
    '/content',
    validateRequest({
      query: z.object({
        type: z.enum(['songs', 'albums', 'all']).optional(),
        status: z.enum(['published', 'draft', 'scheduled', 'all']).optional(),
        limit: z.coerce.number().min(1).max(100).optional(),
        offset: z.coerce.number().min(0).optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const type = (req.query.type as string) || 'all';
      const status = (req.query.status as string) || 'all';
      const limit = parseInt(req.query.limit as string) || 20;
      const offset = parseInt(req.query.offset as string) || 0;

      let query = `
        SELECT
          s.id,
          s.title,
          s.genre,
          s.duration,
          s.cover_art_url,
          s.status,
          s.release_date,
          s.created_at,
          COUNT(p.id) as play_count,
          COUNT(DISTINCT p.wallet_address) as unique_listeners
        FROM songs s
        LEFT JOIN plays p ON s.id = p.song_id
        WHERE s.artist_address = $1
      `;

      const params: unknown[] = [req.auth!.address];

      if (status !== 'all') {
        params.push(status);
        query += ` AND s.status = $${params.length}`;
      }

      query += `
        GROUP BY s.id
        ORDER BY s.created_at DESC
        LIMIT $${params.length + 1} OFFSET $${params.length + 2}
      `;
      params.push(limit, offset);

      const result = await pool.query(query, params);

      // Get total count
      const countResult = await pool.query(
        `SELECT COUNT(*) FROM songs WHERE artist_address = $1 ${
          status !== 'all' ? 'AND status = $2' : ''
        }`,
        status !== 'all' ? [req.auth!.address, status] : [req.auth!.address]
      );

      res.json(
        responseEnvelope({
          data: result.rows,
          meta: {
            total: parseInt(countResult.rows[0].count),
            limit,
            offset,
          },
        })
      );
    }
  );

  /**
   * POST /dashboard/songs
   * Create a new song
   */
  router.post(
    '/songs',
    validateRequest({
      body: z.object({
        title: z.string().min(1).max(200),
        genre: z.string().optional(),
        description: z.string().optional(),
        audioUrl: z.string().url(),
        coverArtUrl: z.string().url().optional(),
        duration: z.number().positive(),
        releaseDate: z.string().datetime().optional(),
        status: z.enum(['draft', 'published', 'scheduled']).optional(),
        royaltySplits: z
          .array(
            z.object({
              address: z.string(),
              percentage: z.number().min(0).max(100),
            })
          )
          .optional(),
        tags: z.array(z.string()).optional(),
        isExplicit: z.boolean().optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const {
        title,
        genre,
        description,
        audioUrl,
        coverArtUrl,
        duration,
        releaseDate,
        status = 'draft',
        royaltySplits = [],
        tags = [],
        isExplicit = false,
      } = req.body;

      const result = await pool.query(
        `
        INSERT INTO songs (
          title, artist_address, artist_name, genre, description,
          audio_url, cover_art_url, duration, release_date, status,
          royalty_splits, tags, is_explicit
        )
        SELECT
          $1, $2, u.display_name, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
        FROM user_profiles u
        WHERE u.wallet_address = $2
        RETURNING *
        `,
        [
          title,
          req.auth!.address,
          genre,
          description,
          audioUrl,
          coverArtUrl,
          duration,
          releaseDate,
          status,
          JSON.stringify(royaltySplits),
          tags,
          isExplicit,
        ]
      );

      logger.info('Song created', {
        songId: result.rows[0].id,
        artistAddress: req.auth!.address,
      });

      res.status(201).json(
        responseEnvelope({
          data: result.rows[0],
        })
      );
    }
  );

  /**
   * PATCH /dashboard/songs/:id
   * Update a song
   */
  router.patch(
    '/songs/:id',
    validateRequest({
      params: z.object({
        id: z.string().uuid(),
      }),
      body: z.object({
        title: z.string().min(1).max(200).optional(),
        genre: z.string().optional(),
        description: z.string().optional(),
        coverArtUrl: z.string().url().optional(),
        releaseDate: z.string().datetime().optional(),
        status: z.enum(['draft', 'published', 'scheduled']).optional(),
        tags: z.array(z.string()).optional(),
        isExplicit: z.boolean().optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const { id } = req.params;
      const updates = req.body;

      // Verify ownership
      const songResult = await pool.query(
        'SELECT * FROM songs WHERE id = $1 AND artist_address = $2',
        [id, req.auth!.address]
      );

      if (songResult.rows.length === 0) {
        return res.status(404).json(
          responseEnvelope({
            error: {
              code: 'SONG_NOT_FOUND',
              message: 'Song not found or unauthorized',
            },
          })
        );
      }

      // Build update query dynamically
      const setClauses: string[] = [];
      const values: unknown[] = [];
      let paramIndex = 1;

      const fieldMap: Record<string, string> = {
        title: 'title',
        genre: 'genre',
        description: 'description',
        coverArtUrl: 'cover_art_url',
        releaseDate: 'release_date',
        status: 'status',
        tags: 'tags',
        isExplicit: 'is_explicit',
      };

      for (const [key, dbField] of Object.entries(fieldMap)) {
        if (updates[key] !== undefined) {
          setClauses.push(`${dbField} = $${paramIndex}`);
          values.push(updates[key]);
          paramIndex++;
        }
      }

      if (setClauses.length === 0) {
        return res.json(
          responseEnvelope({
            data: songResult.rows[0],
          })
        );
      }

      setClauses.push(`updated_at = NOW()`);

      const result = await pool.query(
        `UPDATE songs SET ${setClauses.join(', ')} WHERE id = $${paramIndex} RETURNING *`,
        [...values, id]
      );

      res.json(
        responseEnvelope({
          data: result.rows[0],
        })
      );
    }
  );

  /**
   * DELETE /dashboard/songs/:id
   * Delete a song
   */
  router.delete(
    '/songs/:id',
    validateRequest({
      params: z.object({
        id: z.string().uuid(),
      }),
    }),
    async (req: Request, res: Response) => {
      const { id } = req.params;

      // Verify ownership
      const result = await pool.query(
        'DELETE FROM songs WHERE id = $1 AND artist_address = $2 RETURNING id',
        [id, req.auth!.address]
      );

      if (result.rows.length === 0) {
        return res.status(404).json(
          responseEnvelope({
            error: {
              code: 'SONG_NOT_FOUND',
              message: 'Song not found or unauthorized',
            },
          })
        );
      }

      logger.info('Song deleted', {
        songId: id,
        artistAddress: req.auth!.address,
      });

      res.status(204).send();
    }
  );

  /**
   * GET /dashboard/payouts
   * Get payout history
   */
  router.get(
    '/payouts',
    validateRequest({
      query: z.object({
        limit: z.coerce.number().min(1).max(100).optional(),
        offset: z.coerce.number().min(0).optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const limit = parseInt(req.query.limit as string) || 20;
      const offset = parseInt(req.query.offset as string) || 0;

      const result = await pool.query(
        `
        SELECT
          id,
          amount,
          currency,
          tx_hash,
          status,
          created_at,
          processed_at
        FROM artist_payouts
        WHERE artist_address = $1
        ORDER BY created_at DESC
        LIMIT $2 OFFSET $3
        `,
        [req.auth!.address, limit, offset]
      );

      const countResult = await pool.query(
        'SELECT COUNT(*) FROM artist_payouts WHERE artist_address = $1',
        [req.auth!.address]
      );

      res.json(
        responseEnvelope({
          data: result.rows,
          meta: {
            total: parseInt(countResult.rows[0].count),
            limit,
            offset,
          },
        })
      );
    }
  );

  /**
   * POST /dashboard/payouts/request
   * Request a payout
   */
  router.post('/payouts/request', async (req: Request, res: Response) => {
    // Get pending earnings
    const earningsResult = await pool.query(
      `
      SELECT COALESCE(SUM(amount), 0) as pending
      FROM artist_earnings
      WHERE artist_address = $1 AND paid_out = false
      `,
      [req.auth!.address]
    );

    const pending = parseFloat(earningsResult.rows[0].pending);
    const minPayout = 10; // Minimum payout threshold

    if (pending < minPayout) {
      return res.status(400).json(
        responseEnvelope({
          error: {
            code: 'INSUFFICIENT_BALANCE',
            message: `Minimum payout is ${minPayout}. Current balance: ${pending.toFixed(2)}`,
          },
        })
      );
    }

    // Create payout request
    const result = await pool.query(
      `
      INSERT INTO artist_payouts (artist_address, amount, currency, status)
      VALUES ($1, $2, 'USDC', 'pending')
      RETURNING *
      `,
      [req.auth!.address, pending]
    );

    // Mark earnings as processing
    await pool.query(
      `
      UPDATE artist_earnings
      SET paid_out = true, payout_id = $2
      WHERE artist_address = $1 AND paid_out = false
      `,
      [req.auth!.address, result.rows[0].id]
    );

    logger.info('Payout requested', {
      artistAddress: req.auth!.address,
      amount: pending,
      payoutId: result.rows[0].id,
    });

    res.status(201).json(
      responseEnvelope({
        data: result.rows[0],
      })
    );
  });

  /**
   * GET /dashboard/notifications
   * Get artist notifications
   */
  router.get(
    '/notifications',
    validateRequest({
      query: z.object({
        unreadOnly: z.coerce.boolean().optional(),
        limit: z.coerce.number().min(1).max(100).optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const unreadOnly = req.query.unreadOnly === 'true';
      const limit = parseInt(req.query.limit as string) || 20;

      let query = `
        SELECT *
        FROM notifications
        WHERE wallet_address = $1
      `;

      if (unreadOnly) {
        query += ' AND read = false';
      }

      query += ' ORDER BY created_at DESC LIMIT $2';

      const result = await pool.query(query, [req.auth!.address, limit]);

      res.json(
        responseEnvelope({
          data: result.rows,
        })
      );
    }
  );

  /**
   * POST /dashboard/notifications/read
   * Mark notifications as read
   */
  router.post(
    '/notifications/read',
    validateRequest({
      body: z.object({
        notificationIds: z.array(z.string()).optional(),
        all: z.boolean().optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const { notificationIds, all } = req.body;

      if (all) {
        await pool.query(
          `UPDATE notifications SET read = true WHERE wallet_address = $1`,
          [req.auth!.address]
        );
      } else if (notificationIds && notificationIds.length > 0) {
        await pool.query(
          `UPDATE notifications SET read = true WHERE id = ANY($1) AND wallet_address = $2`,
          [notificationIds, req.auth!.address]
        );
      }

      res.json(
        responseEnvelope({
          data: { success: true },
        })
      );
    }
  );

  return router;
}

export default createDashboardRoutes;
