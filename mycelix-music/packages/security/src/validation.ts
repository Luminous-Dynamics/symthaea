// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Request Validation
 *
 * Zod-based request validation middleware.
 */

import { Request, Response, NextFunction, RequestHandler } from 'express';
import { z, ZodSchema, ZodError } from 'zod';

export interface ValidationSchema {
  body?: ZodSchema;
  query?: ZodSchema;
  params?: ZodSchema;
}

/**
 * Create a validation middleware
 */
export function validateRequest(schema: ValidationSchema): RequestHandler {
  return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      // Validate body
      if (schema.body) {
        req.body = await schema.body.parseAsync(req.body);
      }

      // Validate query
      if (schema.query) {
        req.query = await schema.query.parseAsync(req.query);
      }

      // Validate params
      if (schema.params) {
        req.params = await schema.params.parseAsync(req.params);
      }

      next();
    } catch (error) {
      if (error instanceof ZodError) {
        res.status(400).json({
          error: 'Validation Error',
          message: 'Invalid request data',
          details: error.errors.map((e) => ({
            path: e.path.join('.'),
            message: e.message,
            code: e.code,
          })),
        });
        return;
      }

      res.status(500).json({
        error: 'Internal Server Error',
        message: 'Validation failed',
      });
    }
  };
}

// Common validation schemas
export const commonSchemas = {
  // Pagination
  pagination: z.object({
    page: z.coerce.number().int().positive().default(1),
    limit: z.coerce.number().int().min(1).max(100).default(20),
  }),

  // Sorting
  sorting: z.object({
    sortBy: z.string().optional(),
    sortOrder: z.enum(['asc', 'desc']).default('desc'),
  }),

  // ID parameter
  idParam: z.object({
    id: z.string().uuid(),
  }),

  // Ethereum address
  addressParam: z.object({
    address: z.string().regex(/^0x[a-fA-F0-9]{40}$/),
  }),

  // Search query
  searchQuery: z.object({
    q: z.string().min(1).max(100),
    type: z.enum(['all', 'songs', 'artists', 'playlists']).optional(),
    limit: z.coerce.number().int().min(1).max(50).default(10),
  }),

  // Song upload
  songUpload: z.object({
    title: z.string().min(1).max(200),
    genre: z.string().min(1).max(50),
    tags: z.array(z.string().max(30)).max(10).optional(),
    description: z.string().max(2000).optional(),
    explicit: z.boolean().default(false),
    releaseDate: z.string().datetime().optional(),
  }),

  // User profile update
  profileUpdate: z.object({
    name: z.string().min(1).max(100).optional(),
    bio: z.string().max(500).optional(),
    website: z.string().url().optional(),
    twitter: z.string().max(50).optional(),
    instagram: z.string().max(50).optional(),
  }),

  // Playlist creation
  playlistCreate: z.object({
    name: z.string().min(1).max(100),
    description: z.string().max(500).optional(),
    isPublic: z.boolean().default(true),
  }),

  // Report submission
  reportSubmit: z.object({
    targetType: z.enum(['song', 'user', 'comment', 'playlist']),
    targetId: z.string(),
    reason: z.enum([
      'copyright',
      'harassment',
      'spam',
      'inappropriate',
      'other',
    ]),
    description: z.string().max(1000).optional(),
  }),
};
