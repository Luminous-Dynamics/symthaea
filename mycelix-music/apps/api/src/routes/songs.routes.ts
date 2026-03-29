// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Songs Routes
 * Handles song registration, retrieval, and management
 */

import { Router, Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import { SongService } from '../services/song.service';
import { CacheService } from '../services/cache.service';
import { validateRequest } from '../middleware/validate';
import { requireAuth } from '../middleware/wallet-auth';
import { rateLimit } from '../middleware/rate-limit';
import { idempotent } from '../middleware/idempotency';
import { asyncHandler } from '../utils/async-handler';

const router = Router();

// Validation Schemas
const createSongSchema = z.object({
  body: z.object({
    title: z.string().min(1).max(200),
    artistAddress: z.string().regex(/^0x[a-fA-F0-9]{40}$/),
    ipfsHash: z.string().min(46).max(64),
    duration: z.number().int().positive(),
    strategyId: z.number().int().nonnegative(),
    coverArt: z.string().url().optional(),
    genre: z.string().max(50).optional(),
    tags: z.array(z.string().max(30)).max(10).optional(),
    collaborators: z.array(z.object({
      address: z.string().regex(/^0x[a-fA-F0-9]{40}$/),
      role: z.string().max(50),
      splitPercentage: z.number().min(0).max(100),
    })).optional(),
    metadata: z.record(z.any()).optional(),
  }),
});

const listSongsSchema = z.object({
  query: z.object({
    artistAddress: z.string().regex(/^0x[a-fA-F0-9]{40}$/).optional(),
    genre: z.string().optional(),
    strategyId: z.string().optional(),
    search: z.string().optional(),
    sortBy: z.enum(['createdAt', 'plays', 'likes', 'title']).optional(),
    sortOrder: z.enum(['asc', 'desc']).optional(),
    limit: z.string().transform(Number).pipe(z.number().int().min(1).max(100)).optional(),
    offset: z.string().transform(Number).pipe(z.number().int().min(0)).optional(),
  }),
});

const songIdSchema = z.object({
  params: z.object({
    id: z.string().uuid(),
  }),
});

// Initialize services
const songService = new SongService();
const cacheService = CacheService.getInstance();

/**
 * POST /api/songs
 * Register a new song
 */
router.post(
  '/',
  requireAuth,
  rateLimit({ windowMs: 60000, max: 10 }),
  idempotent,
  validateRequest(createSongSchema),
  asyncHandler(async (req: Request, res: Response) => {
    const { artistAddress, ...songData } = req.body;

    // Verify the authenticated user owns this address
    if (req.user?.address.toLowerCase() !== artistAddress.toLowerCase()) {
      return res.status(403).json({
        error: 'Forbidden',
        message: 'You can only register songs for your own address',
      });
    }

    const song = await songService.createSong({
      ...songData,
      artistAddress,
    });

    // Invalidate cache
    await cacheService.invalidatePattern(`songs:list:*`);
    await cacheService.invalidatePattern(`artist:${artistAddress}:*`);

    res.status(201).json({
      success: true,
      data: song,
    });
  })
);

/**
 * GET /api/songs
 * List songs with filtering and pagination
 */
router.get(
  '/',
  rateLimit({ windowMs: 60000, max: 100 }),
  validateRequest(listSongsSchema),
  asyncHandler(async (req: Request, res: Response) => {
    const {
      artistAddress,
      genre,
      strategyId,
      search,
      sortBy = 'createdAt',
      sortOrder = 'desc',
      limit = 20,
      offset = 0,
    } = req.query;

    // Check cache
    const cacheKey = `songs:list:${JSON.stringify(req.query)}`;
    const cached = await cacheService.get(cacheKey);
    if (cached) {
      return res.json(JSON.parse(cached));
    }

    const result = await songService.listSongs({
      artistAddress: artistAddress as string,
      genre: genre as string,
      strategyId: strategyId ? parseInt(strategyId as string) : undefined,
      search: search as string,
      sortBy: sortBy as string,
      sortOrder: sortOrder as 'asc' | 'desc',
      limit: Number(limit),
      offset: Number(offset),
    });

    // Cache for 1 minute
    await cacheService.set(cacheKey, JSON.stringify(result), 60);

    res.json(result);
  })
);

/**
 * GET /api/songs/:id
 * Get a single song by ID
 */
router.get(
  '/:id',
  rateLimit({ windowMs: 60000, max: 200 }),
  validateRequest(songIdSchema),
  asyncHandler(async (req: Request, res: Response) => {
    const { id } = req.params;

    // Check cache
    const cacheKey = `songs:${id}`;
    const cached = await cacheService.get(cacheKey);
    if (cached) {
      return res.json(JSON.parse(cached));
    }

    const song = await songService.getSongById(id);

    if (!song) {
      return res.status(404).json({
        error: 'Not Found',
        message: 'Song not found',
      });
    }

    // Cache for 5 minutes
    await cacheService.set(cacheKey, JSON.stringify({ success: true, data: song }), 300);

    res.json({
      success: true,
      data: song,
    });
  })
);

/**
 * PATCH /api/songs/:id
 * Update song metadata
 */
router.patch(
  '/:id',
  requireAuth,
  rateLimit({ windowMs: 60000, max: 20 }),
  validateRequest(songIdSchema),
  asyncHandler(async (req: Request, res: Response) => {
    const { id } = req.params;
    const updates = req.body;

    // Get existing song
    const existingSong = await songService.getSongById(id);
    if (!existingSong) {
      return res.status(404).json({
        error: 'Not Found',
        message: 'Song not found',
      });
    }

    // Verify ownership
    if (req.user?.address.toLowerCase() !== existingSong.artistAddress.toLowerCase()) {
      return res.status(403).json({
        error: 'Forbidden',
        message: 'You can only update your own songs',
      });
    }

    const updatedSong = await songService.updateSong(id, updates);

    // Invalidate cache
    await cacheService.delete(`songs:${id}`);
    await cacheService.invalidatePattern(`songs:list:*`);

    res.json({
      success: true,
      data: updatedSong,
    });
  })
);

/**
 * DELETE /api/songs/:id
 * Delete a song (soft delete)
 */
router.delete(
  '/:id',
  requireAuth,
  rateLimit({ windowMs: 60000, max: 5 }),
  validateRequest(songIdSchema),
  asyncHandler(async (req: Request, res: Response) => {
    const { id } = req.params;

    // Get existing song
    const existingSong = await songService.getSongById(id);
    if (!existingSong) {
      return res.status(404).json({
        error: 'Not Found',
        message: 'Song not found',
      });
    }

    // Verify ownership
    if (req.user?.address.toLowerCase() !== existingSong.artistAddress.toLowerCase()) {
      return res.status(403).json({
        error: 'Forbidden',
        message: 'You can only delete your own songs',
      });
    }

    await songService.deleteSong(id);

    // Invalidate cache
    await cacheService.delete(`songs:${id}`);
    await cacheService.invalidatePattern(`songs:list:*`);
    await cacheService.invalidatePattern(`artist:${existingSong.artistAddress}:*`);

    res.json({
      success: true,
      message: 'Song deleted successfully',
    });
  })
);

/**
 * GET /api/songs/:id/stats
 * Get song statistics
 */
router.get(
  '/:id/stats',
  rateLimit({ windowMs: 60000, max: 100 }),
  validateRequest(songIdSchema),
  asyncHandler(async (req: Request, res: Response) => {
    const { id } = req.params;

    const stats = await songService.getSongStats(id);

    if (!stats) {
      return res.status(404).json({
        error: 'Not Found',
        message: 'Song not found',
      });
    }

    res.json({
      success: true,
      data: stats,
    });
  })
);

/**
 * POST /api/songs/:id/like
 * Like a song
 */
router.post(
  '/:id/like',
  requireAuth,
  rateLimit({ windowMs: 60000, max: 60 }),
  validateRequest(songIdSchema),
  asyncHandler(async (req: Request, res: Response) => {
    const { id } = req.params;
    const userAddress = req.user!.address;

    await songService.likeSong(id, userAddress);

    res.json({
      success: true,
      message: 'Song liked',
    });
  })
);

/**
 * DELETE /api/songs/:id/like
 * Unlike a song
 */
router.delete(
  '/:id/like',
  requireAuth,
  rateLimit({ windowMs: 60000, max: 60 }),
  validateRequest(songIdSchema),
  asyncHandler(async (req: Request, res: Response) => {
    const { id } = req.params;
    const userAddress = req.user!.address;

    await songService.unlikeSong(id, userAddress);

    res.json({
      success: true,
      message: 'Song unliked',
    });
  })
);

export { router as songsRouter };
