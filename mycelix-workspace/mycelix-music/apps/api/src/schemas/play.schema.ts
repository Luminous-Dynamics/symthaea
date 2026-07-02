// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Play Validation Schemas
 *
 * Zod schemas for play/streaming-related requests.
 */

import { z } from 'zod';
import {
  uuidSchema,
  walletAddressSchema,
  numericStringSchema,
  transactionStatusSchema,
  paginationSchema,
  dateRangeSchema,
} from './common.schema';

/**
 * Create play (record a song play) request
 */
export const createPlaySchema = z.object({
  song_id: uuidSchema,
  listener_address: walletAddressSchema,
  amount: numericStringSchema,
  transaction_hash: z
    .string()
    .regex(/^0x[a-fA-F0-9]{64}$/, 'Invalid transaction hash')
    .optional(),
});

/**
 * Confirm play request (after blockchain confirmation)
 */
export const confirmPlaySchema = z.object({
  transaction_hash: z
    .string()
    .regex(/^0x[a-fA-F0-9]{64}$/, 'Invalid transaction hash'),
});

/**
 * Play query parameters
 */
export const playQuerySchema = paginationSchema
  .merge(dateRangeSchema.partial())
  .extend({
    song_id: uuidSchema.optional(),
    listener_address: walletAddressSchema.optional(),
    artist_address: walletAddressSchema.optional(),
    status: transactionStatusSchema.optional(),
  });

/**
 * Play response schema
 */
export const playResponseSchema = z.object({
  id: uuidSchema,
  song_id: uuidSchema,
  listener_address: z.string(),
  amount: z.string(),
  transaction_hash: z.string().nullable(),
  status: transactionStatusSchema,
  created_at: z.string().datetime(),
  confirmed_at: z.string().datetime().nullable(),
});

/**
 * Play with song details response
 */
export const playWithSongResponseSchema = playResponseSchema.extend({
  song_title: z.string(),
  song_artist: z.string(),
  artist_address: z.string(),
});

/**
 * Batch play creation (for bulk imports)
 */
export const batchCreatePlaysSchema = z.object({
  plays: z.array(createPlaySchema).min(1).max(100),
});

/**
 * Play statistics query
 */
export const playStatsQuerySchema = z.object({
  period: z.enum(['hour', 'day', 'week', 'month']).default('day'),
  days: z
    .string()
    .optional()
    .transform((val) => (val ? parseInt(val, 10) : 30))
    .pipe(z.number().min(1).max(365)),
  artist_address: walletAddressSchema.optional(),
  song_id: uuidSchema.optional(),
});

// Type exports
export type CreatePlayInput = z.infer<typeof createPlaySchema>;
export type ConfirmPlayInput = z.infer<typeof confirmPlaySchema>;
export type PlayQueryParams = z.infer<typeof playQuerySchema>;
export type PlayResponse = z.infer<typeof playResponseSchema>;
export type PlayWithSongResponse = z.infer<typeof playWithSongResponseSchema>;
export type PlayStatsQuery = z.infer<typeof playStatsQuerySchema>;
