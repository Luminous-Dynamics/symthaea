// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * API Routes - Modular Route Organization
 * Refactored from monolithic index.ts for better maintainability
 */

import { Router } from 'express';
import { songsRouter } from './songs.routes';
import { analyticsRouter } from './analytics.routes';
import { uploadRouter } from './upload.routes';
import { playbackRouter } from './playback.routes';
import { socialRouter } from './social.routes';
import { collaborationRouter } from './collaboration.routes';
import { strategyRouter } from './strategy.routes';
import { userRouter } from './user.routes';
import { searchRouter } from './search.routes';
import { webhookRouter } from './webhook.routes';
import { healthRouter } from './health.routes';
import { adminRouter } from './admin.routes';

const router = Router();

// Health & Status (no auth required)
router.use('/health', healthRouter);

// Core API Routes
router.use('/songs', songsRouter);
router.use('/playback', playbackRouter);
router.use('/analytics', analyticsRouter);
router.use('/upload', uploadRouter);
router.use('/social', socialRouter);
router.use('/collaborations', collaborationRouter);
router.use('/strategies', strategyRouter);
router.use('/users', userRouter);
router.use('/search', searchRouter);

// Webhooks (external services)
router.use('/webhooks', webhookRouter);

// Admin routes (restricted)
router.use('/admin', adminRouter);

export { router as apiRouter };
export default router;
