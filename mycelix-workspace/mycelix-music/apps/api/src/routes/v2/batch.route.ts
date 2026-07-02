// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Batch Operations API
 *
 * Bulk create/update/delete endpoints for efficiency.
 * Supports transactions and partial success handling.
 */

import { Router, Request, Response } from 'express';
import { z } from 'zod';
import { Pool } from 'pg';
import { validate } from '../../middleware/validate';
import { walletAuth } from '../../middleware/wallet-auth';
import { idempotencyMiddleware, requireIdempotencyKey } from '../../middleware/idempotency';
import { success, errors, paginated } from '../../utils/response';
import { getLogger } from '../../logging';
import { getMetrics } from '../../metrics';

const logger = getLogger();

/**
 * Batch operation result
 */
interface BatchOperationResult<T = unknown> {
  index: number;
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
  };
}

/**
 * Batch response
 */
interface BatchResponse<T = unknown> {
  total: number;
  succeeded: number;
  failed: number;
  results: BatchOperationResult<T>[];
}

/**
 * Schemas
 */
const batchSongCreateSchema = z.object({
  songs: z.array(z.object({
    title: z.string().min(1).max(200),
    artistAddress: z.string().regex(/^0x[a-fA-F0-9]{40}$/),
    ipfsHash: z.string().min(46).max(64),
    genre: z.string().optional(),
    duration: z.number().int().positive().optional(),
    metadata: z.record(z.unknown()).optional(),
  })).min(1).max(100),
  options: z.object({
    stopOnError: z.boolean().default(false),
    transaction: z.boolean().default(true),
  }).optional(),
});

const batchSongUpdateSchema = z.object({
  updates: z.array(z.object({
    id: z.string().uuid(),
    data: z.object({
      title: z.string().min(1).max(200).optional(),
      genre: z.string().optional(),
      metadata: z.record(z.unknown()).optional(),
    }),
  })).min(1).max(100),
  options: z.object({
    stopOnError: z.boolean().default(false),
    transaction: z.boolean().default(true),
  }).optional(),
});

const batchSongDeleteSchema = z.object({
  ids: z.array(z.string().uuid()).min(1).max(100),
  options: z.object({
    hardDelete: z.boolean().default(false),
    transaction: z.boolean().default(true),
  }).optional(),
});

const batchPlayRecordSchema = z.object({
  plays: z.array(z.object({
    songId: z.string().uuid(),
    walletAddress: z.string().regex(/^0x[a-fA-F0-9]{40}$/),
    duration: z.number().int().nonnegative().optional(),
    timestamp: z.string().datetime().optional(),
    source: z.string().optional(),
  })).min(1).max(1000),
  options: z.object({
    skipDuplicates: z.boolean().default(true),
  }).optional(),
});

const batchGetSchema = z.object({
  ids: z.array(z.string().uuid()).min(1).max(100),
});

/**
 * Create batch routes
 */
export function createBatchRoutes(pool: Pool): Router {
  const router = Router();

  /**
   * POST /batch/songs
   * Batch create songs
   */
  router.post(
    '/songs',
    walletAuth({ skipInDev: true }),
    idempotencyMiddleware(),
    requireIdempotencyKey,
    validate(batchSongCreateSchema),
    async (req: Request, res: Response) => {
      const { songs, options = {} } = req.body;
      const { stopOnError = false, transaction = true } = options;

      const results: BatchOperationResult[] = [];
      let succeeded = 0;
      let failed = 0;

      const metrics = getMetrics();
      const startTime = Date.now();

      const executeOperations = async (client: Pool | any) => {
        for (let i = 0; i < songs.length; i++) {
          const song = songs[i];

          try {
            const result = await client.query(`
              INSERT INTO songs (title, artist_address, ipfs_hash, genre, duration, metadata)
              VALUES ($1, $2, $3, $4, $5, $6)
              RETURNING *
            `, [
              song.title,
              song.artistAddress.toLowerCase(),
              song.ipfsHash,
              song.genre,
              song.duration,
              song.metadata || {},
            ]);

            results.push({
              index: i,
              success: true,
              data: result.rows[0],
            });
            succeeded++;
          } catch (error) {
            const err = error as Error;
            results.push({
              index: i,
              success: false,
              error: {
                code: 'CREATE_FAILED',
                message: err.message,
              },
            });
            failed++;

            if (stopOnError) {
              throw err;
            }
          }
        }
      };

      try {
        if (transaction) {
          const client = await pool.connect();
          try {
            await client.query('BEGIN');
            await executeOperations(client);
            await client.query('COMMIT');
          } catch (error) {
            await client.query('ROLLBACK');
            throw error;
          } finally {
            client.release();
          }
        } else {
          await executeOperations(pool);
        }

        metrics.observeHistogram('batch_operation_duration_seconds',
          (Date.now() - startTime) / 1000,
          { operation: 'create_songs' }
        );

        const response: BatchResponse = {
          total: songs.length,
          succeeded,
          failed,
          results,
        };

        res.status(failed === songs.length ? 400 : succeeded === songs.length ? 201 : 207);
        success(res, response);
      } catch (error) {
        logger.error('Batch song create failed', error as Error);
        errors.internal(res, 'Batch operation failed');
      }
    }
  );

  /**
   * PATCH /batch/songs
   * Batch update songs
   */
  router.patch(
    '/songs',
    walletAuth({ skipInDev: true }),
    idempotencyMiddleware(),
    validate(batchSongUpdateSchema),
    async (req: Request, res: Response) => {
      const { updates, options = {} } = req.body;
      const { stopOnError = false, transaction = true } = options;

      const results: BatchOperationResult[] = [];
      let succeeded = 0;
      let failed = 0;

      const executeOperations = async (client: Pool | any) => {
        for (let i = 0; i < updates.length; i++) {
          const { id, data } = updates[i];

          try {
            // Build dynamic update query
            const setClauses: string[] = [];
            const values: unknown[] = [id];
            let paramIndex = 2;

            if (data.title) {
              setClauses.push(`title = $${paramIndex++}`);
              values.push(data.title);
            }
            if (data.genre) {
              setClauses.push(`genre = $${paramIndex++}`);
              values.push(data.genre);
            }
            if (data.metadata) {
              setClauses.push(`metadata = metadata || $${paramIndex++}`);
              values.push(JSON.stringify(data.metadata));
            }

            if (setClauses.length === 0) {
              results.push({
                index: i,
                success: false,
                error: { code: 'NO_UPDATES', message: 'No fields to update' },
              });
              failed++;
              continue;
            }

            setClauses.push('updated_at = NOW()');

            const result = await client.query(`
              UPDATE songs
              SET ${setClauses.join(', ')}
              WHERE id = $1 AND deleted_at IS NULL
              RETURNING *
            `, values);

            if (result.rows.length === 0) {
              results.push({
                index: i,
                success: false,
                error: { code: 'NOT_FOUND', message: 'Song not found' },
              });
              failed++;
            } else {
              results.push({
                index: i,
                success: true,
                data: result.rows[0],
              });
              succeeded++;
            }
          } catch (error) {
            const err = error as Error;
            results.push({
              index: i,
              success: false,
              error: { code: 'UPDATE_FAILED', message: err.message },
            });
            failed++;

            if (stopOnError) throw err;
          }
        }
      };

      try {
        if (transaction) {
          const client = await pool.connect();
          try {
            await client.query('BEGIN');
            await executeOperations(client);
            await client.query('COMMIT');
          } catch (error) {
            await client.query('ROLLBACK');
            throw error;
          } finally {
            client.release();
          }
        } else {
          await executeOperations(pool);
        }

        const response: BatchResponse = {
          total: updates.length,
          succeeded,
          failed,
          results,
        };

        res.status(failed === updates.length ? 400 : succeeded === updates.length ? 200 : 207);
        success(res, response);
      } catch (error) {
        logger.error('Batch song update failed', error as Error);
        errors.internal(res, 'Batch operation failed');
      }
    }
  );

  /**
   * DELETE /batch/songs
   * Batch delete songs
   */
  router.delete(
    '/songs',
    walletAuth({ skipInDev: true }),
    idempotencyMiddleware(),
    requireIdempotencyKey,
    validate(batchSongDeleteSchema),
    async (req: Request, res: Response) => {
      const { ids, options = {} } = req.body;
      const { hardDelete = false, transaction = true } = options;

      const results: BatchOperationResult[] = [];
      let succeeded = 0;
      let failed = 0;

      const executeOperations = async (client: Pool | any) => {
        for (let i = 0; i < ids.length; i++) {
          const id = ids[i];

          try {
            let result;
            if (hardDelete) {
              result = await client.query(
                'DELETE FROM songs WHERE id = $1 RETURNING id',
                [id]
              );
            } else {
              result = await client.query(
                'UPDATE songs SET deleted_at = NOW() WHERE id = $1 AND deleted_at IS NULL RETURNING id',
                [id]
              );
            }

            if (result.rows.length === 0) {
              results.push({
                index: i,
                success: false,
                error: { code: 'NOT_FOUND', message: 'Song not found' },
              });
              failed++;
            } else {
              results.push({ index: i, success: true, data: { id } });
              succeeded++;
            }
          } catch (error) {
            const err = error as Error;
            results.push({
              index: i,
              success: false,
              error: { code: 'DELETE_FAILED', message: err.message },
            });
            failed++;
          }
        }
      };

      try {
        if (transaction) {
          const client = await pool.connect();
          try {
            await client.query('BEGIN');
            await executeOperations(client);
            await client.query('COMMIT');
          } catch (error) {
            await client.query('ROLLBACK');
            throw error;
          } finally {
            client.release();
          }
        } else {
          await executeOperations(pool);
        }

        const response: BatchResponse = {
          total: ids.length,
          succeeded,
          failed,
          results,
        };

        res.status(failed === ids.length ? 400 : 200);
        success(res, response);
      } catch (error) {
        logger.error('Batch song delete failed', error as Error);
        errors.internal(res, 'Batch operation failed');
      }
    }
  );

  /**
   * POST /batch/plays
   * Batch record plays (high volume)
   */
  router.post(
    '/plays',
    idempotencyMiddleware(),
    validate(batchPlayRecordSchema),
    async (req: Request, res: Response) => {
      const { plays, options = {} } = req.body;
      const { skipDuplicates = true } = options;

      const startTime = Date.now();

      try {
        // Use COPY for high performance bulk insert
        const values = plays.map((play: any, index: number) => `(
          gen_random_uuid(),
          '${play.songId}',
          '${play.walletAddress.toLowerCase()}',
          ${play.duration || 0},
          '${play.timestamp || new Date().toISOString()}'::timestamptz,
          '${play.source || 'api'}'
        )`).join(',\n');

        const query = skipDuplicates
          ? `
            INSERT INTO plays (id, song_id, wallet_address, duration, played_at, source)
            VALUES ${values}
            ON CONFLICT DO NOTHING
            RETURNING id
          `
          : `
            INSERT INTO plays (id, song_id, wallet_address, duration, played_at, source)
            VALUES ${values}
            RETURNING id
          `;

        const result = await pool.query(query);

        // Update play counts
        const songIds = [...new Set(plays.map((p: any) => p.songId))];
        await pool.query(`
          UPDATE songs
          SET play_count = play_count + subquery.count
          FROM (
            SELECT song_id, COUNT(*) as count
            FROM plays
            WHERE song_id = ANY($1)
            AND played_at > NOW() - INTERVAL '1 minute'
            GROUP BY song_id
          ) as subquery
          WHERE songs.id = subquery.song_id
        `, [songIds]);

        getMetrics().observeHistogram('batch_operation_duration_seconds',
          (Date.now() - startTime) / 1000,
          { operation: 'record_plays' }
        );

        success(res, {
          total: plays.length,
          inserted: result.rowCount,
          skipped: plays.length - (result.rowCount || 0),
        }, 201);
      } catch (error) {
        logger.error('Batch play record failed', error as Error);
        errors.internal(res, 'Batch operation failed');
      }
    }
  );

  /**
   * POST /batch/get/songs
   * Batch get songs by IDs
   */
  router.post(
    '/get/songs',
    validate(batchGetSchema),
    async (req: Request, res: Response) => {
      const { ids } = req.body;

      try {
        const result = await pool.query(`
          SELECT * FROM songs
          WHERE id = ANY($1) AND deleted_at IS NULL
        `, [ids]);

        // Build response preserving order
        const songMap = new Map(result.rows.map(row => [row.id, row]));
        const results = ids.map((id: string, index: number) => {
          const song = songMap.get(id);
          return song
            ? { index, success: true, data: song }
            : { index, success: false, error: { code: 'NOT_FOUND', message: 'Song not found' } };
        });

        success(res, {
          total: ids.length,
          found: result.rows.length,
          results,
        });
      } catch (error) {
        logger.error('Batch get songs failed', error as Error);
        errors.internal(res, 'Batch operation failed');
      }
    }
  );

  return router;
}

export default createBatchRoutes;
