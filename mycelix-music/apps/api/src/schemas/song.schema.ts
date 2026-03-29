// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Song Validation Schemas
 *
 * Zod schemas for song-related requests and responses.
 */

import { z } from 'zod';
import {
  uuidSchema,
  walletAddressSchema,
  ipfsHashSchema,
  titleSchema,
  descriptionSchema,
  genreSchema,
  paymentModelSchema,
  urlSchema,
  paginationSchema,
  sortSchema,
  searchSchema,
  numericStringSchema,
} from './common.schema';

/**
 * Create song request body
 */
export const createSongSchema = z.object({
  title: titleSchema,
  artist: z.string().min(1).max(100).transform((val) => val.trim()),
  artist_address: walletAddressSchema,
  genre: genreSchema,
  description: descriptionSchema,
  ipfs_hash: ipfsHashSchema,
  song_hash: z.string().optional(),
  cover_art: urlSchema,
  audio_url: urlSchema,
  payment_model: paymentModelSchema.default('per_play'),
});

/**
 * Update song request body
 */
export const updateSongSchema = z.object({
  title: titleSchema.optional(),
  description: descriptionSchema,
  genre: genreSchema.optional(),
  cover_art: urlSchema,
  audio_url: urlSchema,
  payment_model: paymentModelSchema.optional(),
}).refine(
  (data) => Object.values(data).some((v) => v !== undefined),
  { message: 'At least one field must be provided for update' }
);

/**
 * Song query parameters (for listing/search)
 */
export const songQuerySchema = paginationSchema
  .merge(sortSchema)
  .merge(searchSchema)
  .extend({
    genre: genreSchema.optional(),
    artist_address: walletAddressSchema.optional(),
    payment_model: paymentModelSchema.optional(),
    min_plays: z
      .string()
      .optional()
      .transform((val) => (val ? parseInt(val, 10) : undefined))
      .pipe(z.number().min(0).optional()),
    max_plays: z
      .string()
      .optional()
      .transform((val) => (val ? parseInt(val, 10) : undefined))
      .pipe(z.number().min(0).optional()),
  });

/**
 * Song response schema (for validation/documentation)
 */
export const songResponseSchema = z.object({
  id: uuidSchema,
  title: z.string(),
  artist: z.string(),
  artist_address: z.string(),
  genre: z.string(),
  description: z.string().nullable(),
  ipfs_hash: z.string(),
  song_hash: z.string().nullable(),
  cover_art: z.string().nullable(),
  audio_url: z.string().nullable(),
  payment_model: paymentModelSchema,
  claim_stream_id: z.string().nullable(),
  plays: z.number(),
  earnings: z.string(),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime().nullable(),
});

/**
 * Batch song IDs request
 */
export const batchSongIdsSchema = z.object({
  ids: z.array(uuidSchema).min(1).max(100),
});

/**
 * Register song on blockchain request
 */
export const registerSongSchema = z.object({
  song_id: uuidSchema,
  transaction_hash: z.string().regex(/^0x[a-fA-F0-9]{64}$/, 'Invalid transaction hash'),
});

/**
 * Set claim stream ID request
 */
export const setClaimStreamSchema = z.object({
  claim_stream_id: z.string().min(1).max(200),
});

// Type exports
export type CreateSongInput = z.infer<typeof createSongSchema>;
export type UpdateSongInput = z.infer<typeof updateSongSchema>;
export type SongQueryParams = z.infer<typeof songQuerySchema>;
export type SongResponse = z.infer<typeof songResponseSchema>;
export type BatchSongIds = z.infer<typeof batchSongIdsSchema>;
