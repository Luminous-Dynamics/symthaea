// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * API v2 Routes Index
 *
 * Aggregates all v2 API routes with the new clean architecture.
 */

import { Router } from 'express';
import songsRouter from './songs.route';
import playsRouter from './plays.route';

const router = Router();

// Mount routes
router.use('/songs', songsRouter);
router.use('/plays', playsRouter);

// Version info endpoint
router.get('/', (req, res) => {
  res.json({
    success: true,
    data: {
      version: 'v2',
      description: 'Mycelix Music API v2 with clean architecture',
      endpoints: {
        songs: '/api/v2/songs',
        plays: '/api/v2/plays',
      },
      features: [
        'Type-safe request validation with Zod',
        'Consistent response envelope',
        'Service layer abstraction',
        'Repository pattern for data access',
        'Rate limiting',
        'Audit logging',
        'Response caching',
      ],
    },
  });
});

export default router;
