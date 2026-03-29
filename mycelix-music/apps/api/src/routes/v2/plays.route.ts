// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Plays Route (v2)
 *
 * Refactored plays API for streaming/play events.
 */

import { Router, Request, Response } from 'express';
import { getContainer } from '../../container';
import {
  validateBody,
  validateQuery,
  validateParams,
} from '../../middleware/validate';
import { asyncHandler } from '../../middleware/error-handler';
import { rateLimit } from '../../middleware/rate-limit';
import { AuditHelpers } from '../../middleware/audit-logger';
import { noCache } from '../../middleware/cache';
import {
  createPlaySchema,
  confirmPlaySchema,
  playQuerySchema,
  playStatsQuerySchema,
  idParamSchema,
  addressParamSchema,
} from '../../schemas';
import { success, created, noContent } from '../../utils/response';

const router = Router();

/**
 * GET /plays
 * Get recent plays
 */
router.get(
  '/',
  validateQuery(playQuerySchema),
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();
    const limit = parseInt(req.query.limit as string) || 20;

    const plays = await container.services.plays.getRecentPlays(limit);

    success(res, plays);
  })
);

/**
 * GET /plays/stats
 * Get play statistics summary
 */
router.get(
  '/stats',
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();

    const stats = await container.services.plays.getStatsSummary();

    success(res, stats);
  })
);

/**
 * GET /plays/daily
 * Get daily statistics
 */
router.get(
  '/daily',
  validateQuery(playStatsQuerySchema),
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();
    const days = parseInt(req.query.days as string) || 30;

    const stats = await container.services.plays.getDailyStats(days);

    success(res, stats);
  })
);

/**
 * GET /plays/hourly
 * Get hourly statistics (last 24 hours)
 */
router.get(
  '/hourly',
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();

    const stats = await container.services.plays.getHourlyStats();

    success(res, stats);
  })
);

/**
 * GET /plays/top-listeners
 * Get top listeners
 */
router.get(
  '/top-listeners',
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();
    const limit = parseInt(req.query.limit as string) || 10;

    const listeners = await container.services.plays.getTopListeners(Math.min(limit, 50));

    success(res, listeners);
  })
);

/**
 * POST /plays
 * Record a new play
 */
router.post(
  '/',
  noCache,
  rateLimit(getContainer().rateLimiters.standard, { keyPrefix: 'record-play:' }),
  validateBody(createPlaySchema),
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();

    const { play, song } = await container.services.plays.recordPlay(req.body);

    // Audit log
    await AuditHelpers.paymentProcessed(req, req.body.amount, req.body.song_id);

    created(res, { play, song });
  })
);

/**
 * POST /plays/:id/confirm
 * Confirm a play after blockchain confirmation
 */
router.post(
  '/:id/confirm',
  noCache,
  validateParams(idParamSchema),
  validateBody(confirmPlaySchema),
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();
    const { id } = req.params;
    const { transaction_hash } = req.body;

    const play = await container.services.plays.confirmPlay(id, transaction_hash);

    success(res, play);
  })
);

/**
 * POST /plays/:id/fail
 * Mark a play as failed
 */
router.post(
  '/:id/fail',
  noCache,
  validateParams(idParamSchema),
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();
    const { id } = req.params;

    const play = await container.services.plays.failPlay(id);

    // Audit log
    await AuditHelpers.paymentFailed(req, 'Play marked as failed', play.song_id);

    success(res, play);
  })
);

/**
 * GET /plays/song/:id
 * Get plays for a song
 */
router.get(
  '/song/:id',
  validateParams(idParamSchema),
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();
    const { id } = req.params;
    const limit = parseInt(req.query.limit as string) || 50;
    const offset = parseInt(req.query.offset as string) || 0;

    const plays = await container.services.plays.getPlaysForSong(id, limit, offset);

    success(res, plays);
  })
);

/**
 * GET /plays/listener/:address
 * Get plays by listener
 */
router.get(
  '/listener/:address',
  validateParams(addressParamSchema),
  asyncHandler(async (req: Request, res: Response) => {
    const container = getContainer();
    const { address } = req.params;
    const limit = parseInt(req.query.limit as string) || 50;
    const offset = parseInt(req.query.offset as string) || 0;

    const plays = await container.services.plays.getPlaysByListener(address, limit, offset);

    success(res, plays);
  })
);

export default router;
