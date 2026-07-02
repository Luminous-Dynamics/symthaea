// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Analytics Routes
 * Handles artist and song analytics, earnings, and reporting
 */

import { Router, Request, Response } from 'express';
import { z } from 'zod';
import { ArtistAnalyticsService } from '../services/artist-analytics.service';
import { CacheService } from '../services/cache.service';
import { validateRequest } from '../middleware/validate';
import { requireAuth } from '../middleware/wallet-auth';
import { rateLimit } from '../middleware/rate-limit';
import { asyncHandler } from '../utils/async-handler';

const router = Router();

// Validation Schemas
const artistAnalyticsSchema = z.object({
  params: z.object({
    address: z.string().regex(/^0x[a-fA-F0-9]{40}$/),
  }),
  query: z.object({
    period: z.enum(['24h', '7d', '30d', '90d', '1y', 'all']).optional(),
    startDate: z.string().datetime().optional(),
    endDate: z.string().datetime().optional(),
  }),
});

const songAnalyticsSchema = z.object({
  params: z.object({
    id: z.string().uuid(),
  }),
  query: z.object({
    period: z.enum(['24h', '7d', '30d', '90d', '1y', 'all']).optional(),
    granularity: z.enum(['hour', 'day', 'week', 'month']).optional(),
  }),
});

const earningsSchema = z.object({
  params: z.object({
    address: z.string().regex(/^0x[a-fA-F0-9]{40}$/),
  }),
  query: z.object({
    period: z.enum(['24h', '7d', '30d', '90d', '1y', 'all']).optional(),
    currency: z.enum(['ETH', 'USD', 'EUR']).optional(),
  }),
});

// Initialize services
const analyticsService = new ArtistAnalyticsService();
const cacheService = CacheService.getInstance();

/**
 * GET /api/analytics/artist/:address
 * Get comprehensive artist analytics
 */
router.get(
  '/artist/:address',
  rateLimit({ windowMs: 60000, max: 30 }),
  validateRequest(artistAnalyticsSchema),
  asyncHandler(async (req: Request, res: Response) => {
    const { address } = req.params;
    const { period = '30d', startDate, endDate } = req.query;

    // Build cache key
    const cacheKey = `analytics:artist:${address}:${period}:${startDate || ''}:${endDate || ''}`;
    const cached = await cacheService.get(cacheKey);
    if (cached) {
      return res.json(JSON.parse(cached));
    }

    const analytics = await analyticsService.getArtistAnalytics(address, {
      period: period as string,
      startDate: startDate ? new Date(startDate as string) : undefined,
      endDate: endDate ? new Date(endDate as string) : undefined,
    });

    const response = {
      success: true,
      data: analytics,
    };

    // Cache for 5 minutes
    await cacheService.set(cacheKey, JSON.stringify(response), 300);

    res.json(response);
  })
);

/**
 * GET /api/analytics/artist/:address/earnings
 * Get artist earnings breakdown
 */
router.get(
  '/artist/:address/earnings',
  requireAuth,
  rateLimit({ windowMs: 60000, max: 30 }),
  validateRequest(earningsSchema),
  asyncHandler(async (req: Request, res: Response) => {
    const { address } = req.params;
    const { period = '30d', currency = 'ETH' } = req.query;

    // Verify the user owns this address or is admin
    if (req.user?.address.toLowerCase() !== address.toLowerCase() && !req.user?.isAdmin) {
      return res.status(403).json({
        error: 'Forbidden',
        message: 'You can only view your own earnings',
      });
    }

    const earnings = await analyticsService.getArtistEarnings(address, {
      period: period as string,
      currency: currency as string,
    });

    res.json({
      success: true,
      data: earnings,
    });
  })
);

/**
 * GET /api/analytics/artist/:address/top-songs
 * Get artist's top performing songs
 */
router.get(
  '/artist/:address/top-songs',
  rateLimit({ windowMs: 60000, max: 30 }),
  validateRequest(z.object({
    params: z.object({
      address: z.string().regex(/^0x[a-fA-F0-9]{40}$/),
    }),
    query: z.object({
      period: z.enum(['24h', '7d', '30d', '90d', '1y', 'all']).optional(),
      limit: z.string().transform(Number).pipe(z.number().int().min(1).max(50)).optional(),
      sortBy: z.enum(['plays', 'earnings', 'likes', 'shares']).optional(),
    }),
  })),
  asyncHandler(async (req: Request, res: Response) => {
    const { address } = req.params;
    const { period = '30d', limit = 10, sortBy = 'plays' } = req.query;

    const topSongs = await analyticsService.getArtistTopSongs(address, {
      period: period as string,
      limit: Number(limit),
      sortBy: sortBy as string,
    });

    res.json({
      success: true,
      data: topSongs,
    });
  })
);

/**
 * GET /api/analytics/artist/:address/listeners
 * Get artist listener demographics
 */
router.get(
  '/artist/:address/listeners',
  requireAuth,
  rateLimit({ windowMs: 60000, max: 20 }),
  asyncHandler(async (req: Request, res: Response) => {
    const { address } = req.params;

    // Verify ownership
    if (req.user?.address.toLowerCase() !== address.toLowerCase() && !req.user?.isAdmin) {
      return res.status(403).json({
        error: 'Forbidden',
        message: 'You can only view your own listener data',
      });
    }

    const listeners = await analyticsService.getListenerDemographics(address);

    res.json({
      success: true,
      data: listeners,
    });
  })
);

/**
 * GET /api/analytics/song/:id
 * Get song analytics
 */
router.get(
  '/song/:id',
  rateLimit({ windowMs: 60000, max: 50 }),
  validateRequest(songAnalyticsSchema),
  asyncHandler(async (req: Request, res: Response) => {
    const { id } = req.params;
    const { period = '30d', granularity = 'day' } = req.query;

    const cacheKey = `analytics:song:${id}:${period}:${granularity}`;
    const cached = await cacheService.get(cacheKey);
    if (cached) {
      return res.json(JSON.parse(cached));
    }

    const analytics = await analyticsService.getSongAnalytics(id, {
      period: period as string,
      granularity: granularity as string,
    });

    if (!analytics) {
      return res.status(404).json({
        error: 'Not Found',
        message: 'Song not found',
      });
    }

    const response = {
      success: true,
      data: analytics,
    };

    // Cache for 5 minutes
    await cacheService.set(cacheKey, JSON.stringify(response), 300);

    res.json(response);
  })
);

/**
 * GET /api/analytics/song/:id/plays
 * Get song play history with time series
 */
router.get(
  '/song/:id/plays',
  rateLimit({ windowMs: 60000, max: 50 }),
  asyncHandler(async (req: Request, res: Response) => {
    const { id } = req.params;
    const { period = '7d', granularity = 'hour' } = req.query;

    const plays = await analyticsService.getSongPlayTimeSeries(id, {
      period: period as string,
      granularity: granularity as string,
    });

    res.json({
      success: true,
      data: plays,
    });
  })
);

/**
 * GET /api/analytics/platform
 * Get platform-wide analytics (public stats)
 */
router.get(
  '/platform',
  rateLimit({ windowMs: 60000, max: 20 }),
  asyncHandler(async (req: Request, res: Response) => {
    const cacheKey = 'analytics:platform';
    const cached = await cacheService.get(cacheKey);
    if (cached) {
      return res.json(JSON.parse(cached));
    }

    const platformStats = await analyticsService.getPlatformStats();

    const response = {
      success: true,
      data: platformStats,
    };

    // Cache for 15 minutes
    await cacheService.set(cacheKey, JSON.stringify(response), 900);

    res.json(response);
  })
);

/**
 * GET /api/analytics/trending
 * Get trending content analytics
 */
router.get(
  '/trending',
  rateLimit({ windowMs: 60000, max: 30 }),
  asyncHandler(async (req: Request, res: Response) => {
    const { period = '24h', limit = 20, type = 'songs' } = req.query;

    const cacheKey = `analytics:trending:${type}:${period}:${limit}`;
    const cached = await cacheService.get(cacheKey);
    if (cached) {
      return res.json(JSON.parse(cached));
    }

    const trending = await analyticsService.getTrending({
      period: period as string,
      limit: Number(limit),
      type: type as 'songs' | 'artists' | 'playlists',
    });

    const response = {
      success: true,
      data: trending,
    };

    // Cache for 5 minutes
    await cacheService.set(cacheKey, JSON.stringify(response), 300);

    res.json(response);
  })
);

/**
 * POST /api/analytics/export
 * Export analytics data
 */
router.post(
  '/export',
  requireAuth,
  rateLimit({ windowMs: 3600000, max: 5 }), // 5 exports per hour
  asyncHandler(async (req: Request, res: Response) => {
    const { type, format = 'csv', dateRange } = req.body;
    const userAddress = req.user!.address;

    const exportData = await analyticsService.exportAnalytics(userAddress, {
      type,
      format,
      dateRange,
    });

    if (format === 'csv') {
      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', `attachment; filename=analytics-${type}-${Date.now()}.csv`);
      return res.send(exportData);
    }

    res.json({
      success: true,
      data: exportData,
    });
  })
);

/**
 * GET /api/analytics/realtime
 * Get real-time analytics (WebSocket upgrade available)
 */
router.get(
  '/realtime',
  requireAuth,
  rateLimit({ windowMs: 60000, max: 10 }),
  asyncHandler(async (req: Request, res: Response) => {
    const userAddress = req.user!.address;

    const realtimeStats = await analyticsService.getRealtimeStats(userAddress);

    res.json({
      success: true,
      data: realtimeStats,
      websocket: {
        available: true,
        endpoint: '/ws/analytics',
        token: await analyticsService.generateRealtimeToken(userAddress),
      },
    });
  })
);

export { router as analyticsRouter };
