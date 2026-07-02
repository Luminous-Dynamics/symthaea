// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * OpenAPI Specification
 *
 * Defines the complete API specification for Mycelix Music API.
 */

import { OpenAPIBuilder } from './generator';
import {
  createSongSchema,
  updateSongSchema,
  songQuerySchema,
  songResponseSchema,
  idParamSchema,
  addressParamSchema,
  createPlaySchema,
  playQuerySchema,
  playResponseSchema,
  paginationSchema,
} from '../schemas';
import { z } from 'zod';

/**
 * Standard API response wrapper
 */
const apiResponseSchema = <T extends z.ZodSchema>(dataSchema: T) =>
  z.object({
    success: z.boolean(),
    data: dataSchema.optional(),
    error: z.object({
      code: z.string(),
      message: z.string(),
      details: z.unknown().optional(),
    }).optional(),
    meta: z.object({
      pagination: z.object({
        total: z.number(),
        limit: z.number(),
        offset: z.number(),
        hasMore: z.boolean(),
      }).optional(),
      timing: z.object({
        startedAt: z.string(),
        duration: z.number(),
      }).optional(),
      requestId: z.string().optional(),
      version: z.string().optional(),
    }).optional(),
  });

/**
 * Error response schema
 */
const errorResponseSchema = z.object({
  success: z.literal(false),
  error: z.object({
    code: z.string(),
    message: z.string(),
    details: z.unknown().optional(),
  }),
});

/**
 * Build the OpenAPI specification
 */
export function buildOpenAPISpec(): ReturnType<OpenAPIBuilder['build']> {
  const builder = new OpenAPIBuilder({
    title: 'Mycelix Music API',
    version: '2.0.0',
    description: `
# Mycelix Music API

A decentralized music streaming platform API built on blockchain technology.

## Features
- Song management with IPFS storage
- Play tracking with cryptocurrency payments
- Artist analytics and earnings
- Real-time updates via WebSocket

## Authentication
Most endpoints are public. Write operations require wallet signature authentication.

## Rate Limiting
- Standard endpoints: 100 requests/minute
- Write endpoints: 10 requests/minute
- Authentication: 5 requests/5 minutes

## Response Format
All responses follow a consistent envelope:
\`\`\`json
{
  "success": true,
  "data": { ... },
  "meta": {
    "pagination": { ... },
    "timing": { ... }
  }
}
\`\`\`
    `.trim(),
    contact: {
      name: 'Mycelix Team',
      url: 'https://mycelix.io',
      email: 'api@mycelix.io',
    },
    license: {
      name: 'MIT',
      url: 'https://opensource.org/licenses/MIT',
    },
  });

  // Add servers
  builder.addServer('http://localhost:3100', 'Local development');
  builder.addServer('https://api.staging.mycelix.io', 'Staging environment');
  builder.addServer('https://api.mycelix.io', 'Production');

  // Add tags
  builder.addTag('Songs', 'Song management and discovery');
  builder.addTag('Plays', 'Play tracking and streaming');
  builder.addTag('Analytics', 'Statistics and analytics');
  builder.addTag('Health', 'Health checks and status');

  // Add security schemes
  builder.addSecurityScheme('walletSignature', {
    type: 'apiKey',
    in: 'header',
    name: 'X-Wallet-Signature',
    description: 'Ethereum wallet signature for authentication',
  });

  // Register reusable schemas
  builder.registerSchema('Song', songResponseSchema);
  builder.registerSchema('Play', playResponseSchema);
  builder.registerSchema('Error', errorResponseSchema);

  // ==================== SONGS ====================

  // GET /api/v2/songs
  builder.addRoute({
    method: 'get',
    path: '/api/v2/songs',
    summary: 'List songs',
    description: 'Get a paginated list of songs with optional filtering and search.',
    tags: ['Songs'],
    operationId: 'listSongs',
    query: songQuerySchema,
    response: apiResponseSchema(z.array(songResponseSchema)),
  });

  // GET /api/v2/songs/top
  builder.addRoute({
    method: 'get',
    path: '/api/v2/songs/top',
    summary: 'Get top songs',
    description: 'Get the most played songs.',
    tags: ['Songs', 'Analytics'],
    operationId: 'getTopSongs',
    query: z.object({ limit: z.string().optional() }),
    response: apiResponseSchema(z.array(songResponseSchema)),
  });

  // GET /api/v2/songs/recent
  builder.addRoute({
    method: 'get',
    path: '/api/v2/songs/recent',
    summary: 'Get recent songs',
    description: 'Get recently added songs.',
    tags: ['Songs'],
    operationId: 'getRecentSongs',
    query: z.object({ limit: z.string().optional() }),
    response: apiResponseSchema(z.array(songResponseSchema)),
  });

  // GET /api/v2/songs/genres
  builder.addRoute({
    method: 'get',
    path: '/api/v2/songs/genres',
    summary: 'Get genre statistics',
    description: 'Get statistics for all genres.',
    tags: ['Songs', 'Analytics'],
    operationId: 'getGenreStats',
    response: apiResponseSchema(z.array(z.object({
      genre: z.string(),
      count: z.number(),
      total_plays: z.number(),
    }))),
  });

  // POST /api/v2/songs/batch
  builder.addRoute({
    method: 'post',
    path: '/api/v2/songs/batch',
    summary: 'Get multiple songs',
    description: 'Get multiple songs by their IDs in a single request.',
    tags: ['Songs'],
    operationId: 'batchGetSongs',
    body: z.object({ ids: z.array(z.string().uuid()) }),
    response: apiResponseSchema(z.array(songResponseSchema)),
  });

  // GET /api/v2/songs/:id
  builder.addRoute({
    method: 'get',
    path: '/api/v2/songs/:id',
    summary: 'Get song by ID',
    description: 'Get a single song by its unique identifier.',
    tags: ['Songs'],
    operationId: 'getSongById',
    params: idParamSchema,
    response: apiResponseSchema(songResponseSchema),
    responses: {
      404: { description: 'Song not found' },
    },
  });

  // POST /api/v2/songs
  builder.addRoute({
    method: 'post',
    path: '/api/v2/songs',
    summary: 'Create song',
    description: 'Register a new song on the platform.',
    tags: ['Songs'],
    operationId: 'createSong',
    body: createSongSchema,
    response: apiResponseSchema(songResponseSchema),
    security: ['walletSignature'],
    responses: {
      201: { description: 'Song created successfully' },
      409: { description: 'Song with this IPFS hash already exists' },
      422: { description: 'Validation error' },
    },
  });

  // PATCH /api/v2/songs/:id
  builder.addRoute({
    method: 'patch',
    path: '/api/v2/songs/:id',
    summary: 'Update song',
    description: 'Update an existing song\'s metadata.',
    tags: ['Songs'],
    operationId: 'updateSong',
    params: idParamSchema,
    body: updateSongSchema,
    response: apiResponseSchema(songResponseSchema),
    security: ['walletSignature'],
    responses: {
      404: { description: 'Song not found' },
      422: { description: 'Validation error' },
    },
  });

  // DELETE /api/v2/songs/:id
  builder.addRoute({
    method: 'delete',
    path: '/api/v2/songs/:id',
    summary: 'Delete song',
    description: 'Delete a song (only if it has no plays).',
    tags: ['Songs'],
    operationId: 'deleteSong',
    params: idParamSchema,
    security: ['walletSignature'],
    responses: {
      204: { description: 'Song deleted successfully' },
      404: { description: 'Song not found' },
      409: { description: 'Cannot delete song with plays' },
    },
  });

  // GET /api/v2/songs/artist/:address
  builder.addRoute({
    method: 'get',
    path: '/api/v2/songs/artist/:address',
    summary: 'Get songs by artist',
    description: 'Get all songs by a specific artist address.',
    tags: ['Songs'],
    operationId: 'getSongsByArtist',
    params: addressParamSchema,
    response: apiResponseSchema(z.array(songResponseSchema)),
  });

  // GET /api/v2/songs/artist/:address/stats
  builder.addRoute({
    method: 'get',
    path: '/api/v2/songs/artist/:address/stats',
    summary: 'Get artist statistics',
    description: 'Get aggregated statistics for an artist.',
    tags: ['Songs', 'Analytics'],
    operationId: 'getArtistStats',
    params: addressParamSchema,
    response: apiResponseSchema(z.object({
      total_songs: z.number(),
      total_plays: z.number(),
      total_earnings: z.string(),
    })),
  });

  // ==================== PLAYS ====================

  // GET /api/v2/plays
  builder.addRoute({
    method: 'get',
    path: '/api/v2/plays',
    summary: 'Get recent plays',
    description: 'Get the most recent play events.',
    tags: ['Plays'],
    operationId: 'getRecentPlays',
    query: paginationSchema,
    response: apiResponseSchema(z.array(playResponseSchema)),
  });

  // GET /api/v2/plays/stats
  builder.addRoute({
    method: 'get',
    path: '/api/v2/plays/stats',
    summary: 'Get play statistics',
    description: 'Get aggregated play statistics.',
    tags: ['Plays', 'Analytics'],
    operationId: 'getPlayStats',
    response: apiResponseSchema(z.object({
      today: z.object({
        plays: z.number(),
        earnings: z.string(),
        unique_listeners: z.number(),
      }).nullable(),
      thisWeek: z.object({
        plays: z.number(),
        earnings: z.string(),
      }),
      thisMonth: z.object({
        plays: z.number(),
        earnings: z.string(),
      }),
    })),
  });

  // POST /api/v2/plays
  builder.addRoute({
    method: 'post',
    path: '/api/v2/plays',
    summary: 'Record play',
    description: 'Record a new song play with payment.',
    tags: ['Plays'],
    operationId: 'recordPlay',
    body: createPlaySchema,
    response: apiResponseSchema(z.object({
      play: playResponseSchema,
      song: songResponseSchema,
    })),
    security: ['walletSignature'],
    responses: {
      201: { description: 'Play recorded successfully' },
      404: { description: 'Song not found' },
      422: { description: 'Validation error' },
    },
  });

  // ==================== HEALTH ====================

  // GET /health
  builder.addRoute({
    method: 'get',
    path: '/health',
    summary: 'Health check',
    description: 'Get the overall health status of the API.',
    tags: ['Health'],
    operationId: 'healthCheck',
    response: z.object({
      status: z.enum(['healthy', 'degraded', 'unhealthy']),
      version: z.string(),
      uptime: z.number(),
      timestamp: z.string(),
      checks: z.record(z.object({
        status: z.enum(['healthy', 'degraded', 'unhealthy']),
        latency: z.number().optional(),
        message: z.string().optional(),
      })),
    }),
  });

  return builder.build();
}

export default buildOpenAPISpec;
