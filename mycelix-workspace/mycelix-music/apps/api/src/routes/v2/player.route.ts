// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Player API Routes
 *
 * Playback control, queue management, and device sync.
 */

import { Router, Request, Response } from 'express';
import { Pool } from 'pg';
import { z } from 'zod';
import { requireAuth } from '../../middleware/auth';
import { validateRequest } from '../../middleware/validation';
import { PlayerService } from '../../services/player.service';
import { EventEmitter } from '../../events';
import { responseEnvelope } from '../../utils/response';

/**
 * Create player routes
 */
export function createPlayerRoutes(
  pool: Pool,
  events: EventEmitter,
  redisUrl?: string
): Router {
  const router = Router();
  const player = new PlayerService(pool, events, redisUrl);

  // All routes require authentication
  router.use(requireAuth);

  /**
   * GET /player
   * Get current player state
   */
  router.get(
    '/',
    validateRequest({
      query: z.object({
        deviceId: z.string().optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const deviceId = (req.query.deviceId as string) || 'default';

      const session = await player.getOrCreateSession(
        req.auth!.address,
        deviceId,
        {
          deviceType: 'web',
          deviceName: req.headers['user-agent']?.slice(0, 50) || 'Web Browser',
        }
      );

      res.json(
        responseEnvelope({
          data: {
            sessionId: session.sessionId,
            state: session.state,
            queue: session.queue,
            queuePosition: session.queuePosition,
            currentSong: session.queue[session.queuePosition] || null,
          },
        })
      );
    }
  );

  /**
   * PUT /player/play
   * Start playing a song
   */
  router.put(
    '/play',
    validateRequest({
      body: z.object({
        songId: z.string().uuid(),
        deviceId: z.string().optional(),
        source: z.enum(['library', 'search', 'playlist', 'radio', 'recommendation']).optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const { songId, deviceId = 'default', source = 'library' } = req.body;

      const session = await player.play(
        req.auth!.address,
        deviceId,
        songId,
        source
      );

      res.json(
        responseEnvelope({
          data: {
            state: session.state,
            queue: session.queue,
            queuePosition: session.queuePosition,
          },
        })
      );
    }
  );

  /**
   * PUT /player/pause
   * Pause playback
   */
  router.put(
    '/pause',
    validateRequest({
      body: z.object({
        deviceId: z.string().optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const deviceId = req.body.deviceId || 'default';

      const session = await player.pause(req.auth!.address, deviceId);

      res.json(
        responseEnvelope({
          data: {
            state: session.state,
          },
        })
      );
    }
  );

  /**
   * PUT /player/resume
   * Resume playback
   */
  router.put(
    '/resume',
    validateRequest({
      body: z.object({
        deviceId: z.string().optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const deviceId = req.body.deviceId || 'default';

      const session = await player.updateState(req.auth!.address, deviceId, {
        isPlaying: true,
      });

      res.json(
        responseEnvelope({
          data: {
            state: session.state,
          },
        })
      );
    }
  );

  /**
   * PUT /player/seek
   * Seek to position
   */
  router.put(
    '/seek',
    validateRequest({
      body: z.object({
        position: z.number().min(0),
        deviceId: z.string().optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const { position, deviceId = 'default' } = req.body;

      const session = await player.seek(req.auth!.address, deviceId, position);

      res.json(
        responseEnvelope({
          data: {
            state: session.state,
          },
        })
      );
    }
  );

  /**
   * POST /player/next
   * Skip to next song
   */
  router.post(
    '/next',
    validateRequest({
      body: z.object({
        deviceId: z.string().optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const deviceId = req.body.deviceId || 'default';

      const session = await player.next(req.auth!.address, deviceId);

      res.json(
        responseEnvelope({
          data: {
            state: session.state,
            queue: session.queue,
            queuePosition: session.queuePosition,
            currentSong: session.queue[session.queuePosition] || null,
          },
        })
      );
    }
  );

  /**
   * POST /player/previous
   * Go to previous song
   */
  router.post(
    '/previous',
    validateRequest({
      body: z.object({
        deviceId: z.string().optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const deviceId = req.body.deviceId || 'default';

      const session = await player.previous(req.auth!.address, deviceId);

      res.json(
        responseEnvelope({
          data: {
            state: session.state,
            queue: session.queue,
            queuePosition: session.queuePosition,
            currentSong: session.queue[session.queuePosition] || null,
          },
        })
      );
    }
  );

  /**
   * PUT /player/volume
   * Set volume
   */
  router.put(
    '/volume',
    validateRequest({
      body: z.object({
        volume: z.number().min(0).max(1),
        deviceId: z.string().optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const { volume, deviceId = 'default' } = req.body;

      const session = await player.updateState(req.auth!.address, deviceId, {
        volume,
      });

      res.json(
        responseEnvelope({
          data: {
            volume: session.state.volume,
          },
        })
      );
    }
  );

  /**
   * PUT /player/shuffle
   * Toggle shuffle
   */
  router.put(
    '/shuffle',
    validateRequest({
      body: z.object({
        shuffle: z.boolean(),
        deviceId: z.string().optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const { shuffle, deviceId = 'default' } = req.body;

      const session = await player.updateState(req.auth!.address, deviceId, {
        shuffle,
      });

      res.json(
        responseEnvelope({
          data: {
            shuffle: session.state.shuffle,
          },
        })
      );
    }
  );

  /**
   * PUT /player/repeat
   * Set repeat mode
   */
  router.put(
    '/repeat',
    validateRequest({
      body: z.object({
        repeat: z.enum(['off', 'one', 'all']),
        deviceId: z.string().optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const { repeat, deviceId = 'default' } = req.body;

      const session = await player.updateState(req.auth!.address, deviceId, {
        repeat,
      });

      res.json(
        responseEnvelope({
          data: {
            repeat: session.state.repeat,
          },
        })
      );
    }
  );

  /**
   * PUT /player/quality
   * Set audio quality
   */
  router.put(
    '/quality',
    validateRequest({
      body: z.object({
        quality: z.enum(['low', 'medium', 'high', 'lossless']),
        deviceId: z.string().optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const { quality, deviceId = 'default' } = req.body;

      const session = await player.updateState(req.auth!.address, deviceId, {
        quality,
      });

      res.json(
        responseEnvelope({
          data: {
            quality: session.state.quality,
          },
        })
      );
    }
  );

  // ==================== Queue Management ====================

  /**
   * GET /player/queue
   * Get current queue
   */
  router.get(
    '/queue',
    validateRequest({
      query: z.object({
        deviceId: z.string().optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const deviceId = (req.query.deviceId as string) || 'default';

      const session = await player.getOrCreateSession(req.auth!.address, deviceId);

      res.json(
        responseEnvelope({
          data: {
            queue: session.queue,
            position: session.queuePosition,
            currentSong: session.queue[session.queuePosition] || null,
          },
        })
      );
    }
  );

  /**
   * POST /player/queue
   * Add songs to queue
   */
  router.post(
    '/queue',
    validateRequest({
      body: z.object({
        songIds: z.array(z.string().uuid()).min(1).max(100),
        position: z.enum(['next', 'last']).optional(),
        deviceId: z.string().optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const { songIds, position = 'last', deviceId = 'default' } = req.body;

      const session = await player.addToQueue(
        req.auth!.address,
        deviceId,
        songIds,
        position
      );

      res.json(
        responseEnvelope({
          data: {
            queue: session.queue,
            addedCount: songIds.length,
          },
        })
      );
    }
  );

  /**
   * DELETE /player/queue/:itemId
   * Remove item from queue
   */
  router.delete(
    '/queue/:itemId',
    validateRequest({
      params: z.object({
        itemId: z.string(),
      }),
      query: z.object({
        deviceId: z.string().optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const { itemId } = req.params;
      const deviceId = (req.query.deviceId as string) || 'default';

      const session = await player.removeFromQueue(
        req.auth!.address,
        deviceId,
        itemId
      );

      res.json(
        responseEnvelope({
          data: {
            queue: session.queue,
          },
        })
      );
    }
  );

  /**
   * PUT /player/queue/reorder
   * Reorder queue items
   */
  router.put(
    '/queue/reorder',
    validateRequest({
      body: z.object({
        fromIndex: z.number().min(0),
        toIndex: z.number().min(0),
        deviceId: z.string().optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const { fromIndex, toIndex, deviceId = 'default' } = req.body;

      const session = await player.reorderQueue(
        req.auth!.address,
        deviceId,
        fromIndex,
        toIndex
      );

      res.json(
        responseEnvelope({
          data: {
            queue: session.queue,
          },
        })
      );
    }
  );

  /**
   * DELETE /player/queue
   * Clear queue
   */
  router.delete(
    '/queue',
    validateRequest({
      query: z.object({
        deviceId: z.string().optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const deviceId = (req.query.deviceId as string) || 'default';

      const session = await player.clearQueue(req.auth!.address, deviceId);

      res.json(
        responseEnvelope({
          data: {
            queue: session.queue,
          },
        })
      );
    }
  );

  // ==================== Device Management ====================

  /**
   * GET /player/devices
   * Get all user devices
   */
  router.get('/devices', async (req: Request, res: Response) => {
    const devices = await player.getDevices(req.auth!.address);

    res.json(
      responseEnvelope({
        data: devices,
      })
    );
  });

  /**
   * PUT /player/transfer
   * Transfer playback to another device
   */
  router.put(
    '/transfer',
    validateRequest({
      body: z.object({
        fromDeviceId: z.string(),
        toDeviceId: z.string(),
      }),
    }),
    async (req: Request, res: Response) => {
      const { fromDeviceId, toDeviceId } = req.body;

      const session = await player.transferPlayback(
        req.auth!.address,
        fromDeviceId,
        toDeviceId
      );

      res.json(
        responseEnvelope({
          data: {
            deviceId: session.deviceId,
            state: session.state,
          },
        })
      );
    }
  );

  // ==================== History ====================

  /**
   * GET /player/history
   * Get listening history
   */
  router.get(
    '/history',
    validateRequest({
      query: z.object({
        limit: z.coerce.number().min(1).max(100).optional(),
        offset: z.coerce.number().min(0).optional(),
      }),
    }),
    async (req: Request, res: Response) => {
      const limit = parseInt(req.query.limit as string) || 50;
      const offset = parseInt(req.query.offset as string) || 0;

      const result = await pool.query(
        `
        SELECT
          ls.song_id,
          ls.started_at,
          ls.duration as listen_duration,
          ls.completed,
          ls.source,
          s.title,
          s.artist_name,
          s.artist_address,
          s.cover_art_url,
          s.duration
        FROM listening_sessions ls
        JOIN songs s ON ls.song_id = s.id
        WHERE ls.wallet_address = $1
        ORDER BY ls.started_at DESC
        LIMIT $2 OFFSET $3
        `,
        [req.auth!.address, limit, offset]
      );

      res.json(
        responseEnvelope({
          data: result.rows,
          meta: {
            limit,
            offset,
          },
        })
      );
    }
  );

  return router;
}

export default createPlayerRoutes;
