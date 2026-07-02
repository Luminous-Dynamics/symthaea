// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Songs Route (v2)
 *
 * Refactored songs API using the clean architecture:
 * - Zod validation for request validation
 * - Service layer for business logic
 * - Consistent response envelope
 * - Proper error handling
 * - Rate limiting
 * - Audit logging
 */

import { Router, Request, Response } from 'express';
import { getContainer } from '../../container';
import {
  validateBody,
  validateQuery,
  validateParams,
} from '../../middleware/validate';
import { asyncHandler, AppError } from '../../middleware/error-handler';
import { RateLimitPresets, rateLimit } from '../../middleware/rate-limit';
import { AuditHelpers } from '../../middleware/audit-logger';
import { cacheMiddleware, noCache } from '../../middleware/cache';
import {
  createSongSchema,
  updateSongSchema,
  songQuerySchema,
  idParamSchema,
  addressParamSchema,
  batchSongIdsSchema,
  setClaimStreamSchema,
} from '../../schemas';
import { success, paginated, created, noContent, errors } from '../../utils/response';

const router = Router();

/**
 * GET /songs
 * List and search songs
 */
router.get(
  '/',
  validateQuery(songQuerySchema),
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();
    const params = req.query as any;

    const result = await container.services.songs.searchSongs(params);

    paginated(res, result.data, {
      total: result.total,
      limit: result.limit,
      offset: result.offset,
      hasMore: result.hasMore,
      nextCursor: result.nextCursor,
    });
  })
);

/**
 * GET /songs/top
 * Get top songs by plays
 */
router.get(
  '/top',
  cacheMiddleware(getContainer().cache, { ttl: 60, tags: ['songs', 'analytics'] }),
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();
    const limit = parseInt(req.query.limit as string) || 10;

    const songs = await container.services.songs.getTopSongs(Math.min(limit, 50));

    success(res, songs);
  })
);

/**
 * GET /songs/recent
 * Get recently added songs
 */
router.get(
  '/recent',
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();
    const limit = parseInt(req.query.limit as string) || 10;

    const songs = await container.services.songs.getRecentSongs(Math.min(limit, 50));

    success(res, songs);
  })
);

/**
 * GET /songs/genres
 * Get genre statistics
 */
router.get(
  '/genres',
  cacheMiddleware(getContainer().cache, { ttl: 3600, tags: ['analytics'] }),
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();

    const stats = await container.services.songs.getGenreStats();

    success(res, stats);
  })
);

/**
 * POST /songs/batch
 * Get multiple songs by IDs
 */
router.post(
  '/batch',
  validateBody(batchSongIdsSchema),
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();
    const { ids } = req.body;

    const songs = await container.services.songs.getSongsByIds(ids);

    success(res, songs);
  })
);

/**
 * GET /songs/:id
 * Get a single song by ID
 */
router.get(
  '/:id',
  validateParams(idParamSchema),
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();
    const { id } = req.params;

    const song = await container.services.songs.getSongById(id);

    success(res, song);
  })
);

/**
 * POST /songs
 * Create a new song
 */
router.post(
  '/',
  noCache,
  rateLimit(getContainer().rateLimiters.strict, { keyPrefix: 'create-song:' }),
  validateBody(createSongSchema),
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();

    const song = await container.services.songs.createSong(req.body);

    // Audit log
    await AuditHelpers.songRegistered(req, song.id, song.title);

    created(res, song);
  })
);

/**
 * PATCH /songs/:id
 * Update a song
 */
router.patch(
  '/:id',
  noCache,
  validateParams(idParamSchema),
  validateBody(updateSongSchema),
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();
    const { id } = req.params;

    const song = await container.services.songs.updateSong(id, req.body);

    success(res, song);
  })
);

/**
 * DELETE /songs/:id
 * Delete a song
 */
router.delete(
  '/:id',
  noCache,
  validateParams(idParamSchema),
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();
    const { id } = req.params;

    await container.services.songs.deleteSong(id);

    noContent(res);
  })
);

/**
 * POST /songs/:id/claim-stream
 * Set the claim stream ID for a song
 */
router.post(
  '/:id/claim-stream',
  noCache,
  validateParams(idParamSchema),
  validateBody(setClaimStreamSchema),
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();
    const { id } = req.params;
    const { claim_stream_id } = req.body;

    const song = await container.services.songs.setClaimStreamId(id, claim_stream_id);

    // Audit log
    await AuditHelpers.claimCreated(req, claim_stream_id, id);

    success(res, song);
  })
);

/**
 * GET /songs/artist/:address
 * Get songs by artist address
 */
router.get(
  '/artist/:address',
  validateParams(addressParamSchema),
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();
    const { address } = req.params;

    const songs = await container.services.songs.getSongsByArtist(address);

    success(res, songs);
  })
);

/**
 * GET /songs/artist/:address/stats
 * Get artist statistics
 */
router.get(
  '/artist/:address/stats',
  validateParams(addressParamSchema),
  cacheMiddleware(getContainer().cache, { ttl: 300, tags: ['artists', 'analytics'] }),
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();
    const { address } = req.params;

    const stats = await container.services.songs.getArtistStats(address);

    success(res, stats);
  })
);

/**
 * GET /songs/genre/:genre
 * Get songs by genre
 */
router.get(
  '/genre/:genre',
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();
    const { genre } = req.params;
    const limit = parseInt(req.query.limit as string) || 50;

    const songs = await container.services.songs.getSongsByGenre(genre, Math.min(limit, 100));

    success(res, songs);
  })
);

export default router;
