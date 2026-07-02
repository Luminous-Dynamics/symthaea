// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Music Platform - Documentation Generator
 *
 * Comprehensive documentation system with:
 * - OpenAPI 3.0 specification generation
 * - Architecture Decision Records (ADRs)
 * - Developer onboarding guides
 * - Operations runbooks
 * - API client SDK documentation
 */

import * as crypto from 'crypto';

// ============================================================================
// OPENAPI SPECIFICATION
// ============================================================================

interface OpenAPIInfo {
  title: string;
  version: string;
  description: string;
  termsOfService?: string;
  contact?: {
    name: string;
    url?: string;
    email?: string;
  };
  license?: {
    name: string;
    url?: string;
  };
}

interface OpenAPIServer {
  url: string;
  description?: string;
  variables?: Record<string, {
    default: string;
    enum?: string[];
    description?: string;
  }>;
}

interface OpenAPIPath {
  summary?: string;
  description?: string;
  get?: OpenAPIOperation;
  post?: OpenAPIOperation;
  put?: OpenAPIOperation;
  patch?: OpenAPIOperation;
  delete?: OpenAPIOperation;
  parameters?: OpenAPIParameter[];
}

interface OpenAPIOperation {
  operationId: string;
  summary?: string;
  description?: string;
  tags?: string[];
  parameters?: OpenAPIParameter[];
  requestBody?: OpenAPIRequestBody;
  responses: Record<string, OpenAPIResponse>;
  security?: Array<Record<string, string[]>>;
  deprecated?: boolean;
}

interface OpenAPIParameter {
  name: string;
  in: 'path' | 'query' | 'header' | 'cookie';
  description?: string;
  required?: boolean;
  deprecated?: boolean;
  schema: OpenAPISchema;
}

interface OpenAPIRequestBody {
  description?: string;
  required?: boolean;
  content: Record<string, { schema: OpenAPISchema }>;
}

interface OpenAPIResponse {
  description: string;
  headers?: Record<string, { schema: OpenAPISchema }>;
  content?: Record<string, { schema: OpenAPISchema }>;
}

interface OpenAPISchema {
  type?: string;
  format?: string;
  items?: OpenAPISchema;
  properties?: Record<string, OpenAPISchema>;
  required?: string[];
  enum?: any[];
  default?: any;
  description?: string;
  example?: any;
  $ref?: string;
}

/**
 * Generate complete OpenAPI specification
 */
function generateOpenAPISpec(): object {
  return {
    openapi: '3.0.3',
    info: {
      title: 'Mycelix Music API',
      version: '1.0.0',
      description: `
# Mycelix Music Platform API

The Mycelix Music API provides comprehensive access to music streaming, artist information,
playlists, and social features. This API follows RESTful conventions and uses JSON for
request and response bodies.

## Authentication

The API uses Bearer token authentication. Include your access token in the Authorization header:

\`\`\`
Authorization: Bearer <your-access-token>
\`\`\`

## Rate Limiting

API requests are rate-limited to ensure fair usage:
- Standard tier: 100 requests per minute
- Premium tier: 1000 requests per minute
- Enterprise tier: Custom limits

Rate limit headers are included in all responses:
- \`X-RateLimit-Limit\`: Maximum requests allowed
- \`X-RateLimit-Remaining\`: Requests remaining in current window
- \`X-RateLimit-Reset\`: Unix timestamp when the limit resets

## Pagination

List endpoints support pagination with the following query parameters:
- \`page\`: Page number (default: 1)
- \`pageSize\`: Items per page (default: 20, max: 100)

Paginated responses include a \`pagination\` object with metadata.

## Error Handling

Errors are returned with appropriate HTTP status codes and a JSON body:

\`\`\`json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {}
  }
}
\`\`\`
      `.trim(),
      termsOfService: 'https://mycelix.music/terms',
      contact: {
        name: 'Mycelix Developer Support',
        url: 'https://developers.mycelix.music',
        email: 'api@mycelix.music',
      },
      license: {
        name: 'MIT',
        url: 'https://opensource.org/licenses/MIT',
      },
    },

    servers: [
      {
        url: 'https://api.mycelix.music/v1',
        description: 'Production',
      },
      {
        url: 'https://staging-api.mycelix.music/v1',
        description: 'Staging',
      },
      {
        url: 'http://localhost:3000/api',
        description: 'Local Development',
      },
    ],

    tags: [
      { name: 'tracks', description: 'Music track operations' },
      { name: 'artists', description: 'Artist profile operations' },
      { name: 'albums', description: 'Album operations' },
      { name: 'playlists', description: 'Playlist management' },
      { name: 'users', description: 'User profile and preferences' },
      { name: 'search', description: 'Search across all content' },
      { name: 'streaming', description: 'Audio streaming' },
      { name: 'social', description: 'Social features' },
      { name: 'recommendations', description: 'Personalized recommendations' },
      { name: 'analytics', description: 'Listening analytics' },
    ],

    paths: generatePaths(),
    components: generateComponents(),

    security: [
      { bearerAuth: [] },
    ],
  };
}

function generatePaths(): Record<string, OpenAPIPath> {
  return {
    // Tracks
    '/tracks': {
      get: {
        operationId: 'listTracks',
        summary: 'List tracks',
        description: 'Retrieve a paginated list of tracks with optional filtering.',
        tags: ['tracks'],
        parameters: [
          { name: 'page', in: 'query', schema: { type: 'integer', default: 1 } },
          { name: 'pageSize', in: 'query', schema: { type: 'integer', default: 20 } },
          { name: 'artistId', in: 'query', schema: { type: 'string', format: 'uuid' } },
          { name: 'genre', in: 'query', schema: { type: 'string' } },
          { name: 'sort', in: 'query', schema: { type: 'string', enum: ['popularity', 'releaseDate', 'title'] } },
        ],
        responses: {
          '200': {
            description: 'Successful response',
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/TrackListResponse' },
              },
            },
          },
          '401': { $ref: '#/components/responses/Unauthorized' },
        },
        security: [{ bearerAuth: [] }],
      },
      post: {
        operationId: 'createTrack',
        summary: 'Create a track',
        description: 'Upload a new track to the platform.',
        tags: ['tracks'],
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: { $ref: '#/components/schemas/CreateTrackRequest' },
            },
          },
        },
        responses: {
          '201': {
            description: 'Track created',
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/TrackResponse' },
              },
            },
          },
          '400': { $ref: '#/components/responses/BadRequest' },
          '401': { $ref: '#/components/responses/Unauthorized' },
        },
        security: [{ bearerAuth: [] }],
      },
    },

    '/tracks/{trackId}': {
      parameters: [
        { name: 'trackId', in: 'path', required: true, schema: { type: 'string', format: 'uuid' } },
      ],
      get: {
        operationId: 'getTrack',
        summary: 'Get track details',
        tags: ['tracks'],
        responses: {
          '200': {
            description: 'Track details',
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/TrackResponse' },
              },
            },
          },
          '404': { $ref: '#/components/responses/NotFound' },
        },
      },
      put: {
        operationId: 'updateTrack',
        summary: 'Update track',
        tags: ['tracks'],
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: { $ref: '#/components/schemas/UpdateTrackRequest' },
            },
          },
        },
        responses: {
          '200': {
            description: 'Track updated',
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/TrackResponse' },
              },
            },
          },
        },
        security: [{ bearerAuth: [] }],
      },
      delete: {
        operationId: 'deleteTrack',
        summary: 'Delete track',
        tags: ['tracks'],
        responses: {
          '204': { description: 'Track deleted' },
          '404': { $ref: '#/components/responses/NotFound' },
        },
        security: [{ bearerAuth: [] }],
      },
    },

    '/tracks/{trackId}/stream': {
      parameters: [
        { name: 'trackId', in: 'path', required: true, schema: { type: 'string', format: 'uuid' } },
      ],
      get: {
        operationId: 'streamTrack',
        summary: 'Get streaming URL',
        description: 'Get a temporary URL to stream the track. URL expires after 1 hour.',
        tags: ['tracks', 'streaming'],
        parameters: [
          { name: 'quality', in: 'query', schema: { type: 'string', enum: ['low', 'medium', 'high', 'lossless'] } },
          { name: 'format', in: 'query', schema: { type: 'string', enum: ['mp3', 'aac', 'flac', 'opus'] } },
        ],
        responses: {
          '200': {
            description: 'Streaming URL',
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  properties: {
                    url: { type: 'string', format: 'uri' },
                    expiresAt: { type: 'string', format: 'date-time' },
                    format: { type: 'string' },
                    bitrate: { type: 'integer' },
                  },
                },
              },
            },
          },
        },
        security: [{ bearerAuth: [] }],
      },
    },

    // Artists
    '/artists': {
      get: {
        operationId: 'listArtists',
        summary: 'List artists',
        tags: ['artists'],
        parameters: [
          { name: 'page', in: 'query', schema: { type: 'integer' } },
          { name: 'pageSize', in: 'query', schema: { type: 'integer' } },
          { name: 'genre', in: 'query', schema: { type: 'string' } },
        ],
        responses: {
          '200': {
            description: 'Artist list',
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/ArtistListResponse' },
              },
            },
          },
        },
      },
    },

    '/artists/{artistId}': {
      parameters: [
        { name: 'artistId', in: 'path', required: true, schema: { type: 'string', format: 'uuid' } },
      ],
      get: {
        operationId: 'getArtist',
        summary: 'Get artist details',
        tags: ['artists'],
        responses: {
          '200': {
            description: 'Artist details',
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/ArtistResponse' },
              },
            },
          },
        },
      },
    },

    '/artists/{artistId}/tracks': {
      parameters: [
        { name: 'artistId', in: 'path', required: true, schema: { type: 'string', format: 'uuid' } },
      ],
      get: {
        operationId: 'getArtistTracks',
        summary: 'Get artist tracks',
        tags: ['artists', 'tracks'],
        responses: {
          '200': {
            description: 'Artist tracks',
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/TrackListResponse' },
              },
            },
          },
        },
      },
    },

    // Playlists
    '/playlists': {
      get: {
        operationId: 'listPlaylists',
        summary: 'List playlists',
        tags: ['playlists'],
        responses: {
          '200': {
            description: 'Playlist list',
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/PlaylistListResponse' },
              },
            },
          },
        },
      },
      post: {
        operationId: 'createPlaylist',
        summary: 'Create playlist',
        tags: ['playlists'],
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: { $ref: '#/components/schemas/CreatePlaylistRequest' },
            },
          },
        },
        responses: {
          '201': {
            description: 'Playlist created',
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/PlaylistResponse' },
              },
            },
          },
        },
        security: [{ bearerAuth: [] }],
      },
    },

    '/playlists/{playlistId}/tracks': {
      parameters: [
        { name: 'playlistId', in: 'path', required: true, schema: { type: 'string', format: 'uuid' } },
      ],
      post: {
        operationId: 'addTracksToPlaylist',
        summary: 'Add tracks to playlist',
        tags: ['playlists'],
        requestBody: {
          required: true,
          content: {
            'application/json': {
              schema: {
                type: 'object',
                properties: {
                  trackIds: { type: 'array', items: { type: 'string', format: 'uuid' } },
                  position: { type: 'integer' },
                },
                required: ['trackIds'],
              },
            },
          },
        },
        responses: {
          '200': { description: 'Tracks added' },
        },
        security: [{ bearerAuth: [] }],
      },
    },

    // Search
    '/search': {
      get: {
        operationId: 'search',
        summary: 'Search across all content',
        tags: ['search'],
        parameters: [
          { name: 'q', in: 'query', required: true, schema: { type: 'string' }, description: 'Search query' },
          { name: 'type', in: 'query', schema: { type: 'array', items: { type: 'string', enum: ['track', 'artist', 'album', 'playlist'] } } },
          { name: 'limit', in: 'query', schema: { type: 'integer', default: 20 } },
        ],
        responses: {
          '200': {
            description: 'Search results',
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/SearchResponse' },
              },
            },
          },
        },
      },
    },

    // Users
    '/users/me': {
      get: {
        operationId: 'getCurrentUser',
        summary: 'Get current user profile',
        tags: ['users'],
        responses: {
          '200': {
            description: 'User profile',
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/UserResponse' },
              },
            },
          },
        },
        security: [{ bearerAuth: [] }],
      },
    },

    '/users/me/top/tracks': {
      get: {
        operationId: 'getTopTracks',
        summary: 'Get user top tracks',
        tags: ['users', 'recommendations'],
        parameters: [
          { name: 'timeRange', in: 'query', schema: { type: 'string', enum: ['short_term', 'medium_term', 'long_term'] } },
          { name: 'limit', in: 'query', schema: { type: 'integer', default: 20 } },
        ],
        responses: {
          '200': {
            description: 'Top tracks',
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/TrackListResponse' },
              },
            },
          },
        },
        security: [{ bearerAuth: [] }],
      },
    },

    // Recommendations
    '/recommendations': {
      get: {
        operationId: 'getRecommendations',
        summary: 'Get personalized recommendations',
        tags: ['recommendations'],
        parameters: [
          { name: 'seedTracks', in: 'query', schema: { type: 'array', items: { type: 'string' } } },
          { name: 'seedArtists', in: 'query', schema: { type: 'array', items: { type: 'string' } } },
          { name: 'seedGenres', in: 'query', schema: { type: 'array', items: { type: 'string' } } },
          { name: 'limit', in: 'query', schema: { type: 'integer', default: 20 } },
        ],
        responses: {
          '200': {
            description: 'Recommended tracks',
            content: {
              'application/json': {
                schema: { $ref: '#/components/schemas/TrackListResponse' },
              },
            },
          },
        },
        security: [{ bearerAuth: [] }],
      },
    },
  };
}

function generateComponents(): object {
  return {
    schemas: {
      // Track schemas
      Track: {
        type: 'object',
        properties: {
          id: { type: 'string', format: 'uuid' },
          title: { type: 'string' },
          artistId: { type: 'string', format: 'uuid' },
          artist: { $ref: '#/components/schemas/ArtistSummary' },
          albumId: { type: 'string', format: 'uuid' },
          album: { $ref: '#/components/schemas/AlbumSummary' },
          duration: { type: 'integer', description: 'Duration in milliseconds' },
          trackNumber: { type: 'integer' },
          discNumber: { type: 'integer' },
          isExplicit: { type: 'boolean' },
          isrc: { type: 'string' },
          popularity: { type: 'integer', minimum: 0, maximum: 100 },
          previewUrl: { type: 'string', format: 'uri' },
          externalUrls: { type: 'object', additionalProperties: { type: 'string' } },
          createdAt: { type: 'string', format: 'date-time' },
          updatedAt: { type: 'string', format: 'date-time' },
        },
      },

      TrackResponse: {
        type: 'object',
        properties: {
          data: { $ref: '#/components/schemas/Track' },
          links: {
            type: 'object',
            properties: {
              self: { type: 'string', format: 'uri' },
            },
          },
        },
      },

      TrackListResponse: {
        type: 'object',
        properties: {
          data: { type: 'array', items: { $ref: '#/components/schemas/Track' } },
          pagination: { $ref: '#/components/schemas/Pagination' },
          links: { $ref: '#/components/schemas/PaginationLinks' },
        },
      },

      CreateTrackRequest: {
        type: 'object',
        required: ['title', 'artistId'],
        properties: {
          title: { type: 'string', minLength: 1, maxLength: 200 },
          artistId: { type: 'string', format: 'uuid' },
          albumId: { type: 'string', format: 'uuid' },
          duration: { type: 'integer' },
          genre: { type: 'string' },
          releaseDate: { type: 'string', format: 'date' },
          isExplicit: { type: 'boolean', default: false },
          isrc: { type: 'string', pattern: '^[A-Z]{2}[A-Z0-9]{3}\\d{7}$' },
        },
      },

      UpdateTrackRequest: {
        type: 'object',
        properties: {
          title: { type: 'string', minLength: 1, maxLength: 200 },
          genre: { type: 'string' },
          isExplicit: { type: 'boolean' },
        },
      },

      // Artist schemas
      Artist: {
        type: 'object',
        properties: {
          id: { type: 'string', format: 'uuid' },
          name: { type: 'string' },
          bio: { type: 'string' },
          genres: { type: 'array', items: { type: 'string' } },
          imageUrl: { type: 'string', format: 'uri' },
          followerCount: { type: 'integer' },
          verified: { type: 'boolean' },
          externalUrls: { type: 'object', additionalProperties: { type: 'string' } },
        },
      },

      ArtistSummary: {
        type: 'object',
        properties: {
          id: { type: 'string', format: 'uuid' },
          name: { type: 'string' },
          imageUrl: { type: 'string', format: 'uri' },
        },
      },

      ArtistResponse: {
        type: 'object',
        properties: {
          data: { $ref: '#/components/schemas/Artist' },
        },
      },

      ArtistListResponse: {
        type: 'object',
        properties: {
          data: { type: 'array', items: { $ref: '#/components/schemas/Artist' } },
          pagination: { $ref: '#/components/schemas/Pagination' },
        },
      },

      // Album schemas
      Album: {
        type: 'object',
        properties: {
          id: { type: 'string', format: 'uuid' },
          title: { type: 'string' },
          artistId: { type: 'string', format: 'uuid' },
          artist: { $ref: '#/components/schemas/ArtistSummary' },
          releaseDate: { type: 'string', format: 'date' },
          trackCount: { type: 'integer' },
          totalDuration: { type: 'integer' },
          imageUrl: { type: 'string', format: 'uri' },
          albumType: { type: 'string', enum: ['album', 'single', 'ep', 'compilation'] },
        },
      },

      AlbumSummary: {
        type: 'object',
        properties: {
          id: { type: 'string', format: 'uuid' },
          title: { type: 'string' },
          imageUrl: { type: 'string', format: 'uri' },
        },
      },

      // Playlist schemas
      Playlist: {
        type: 'object',
        properties: {
          id: { type: 'string', format: 'uuid' },
          name: { type: 'string' },
          description: { type: 'string' },
          ownerId: { type: 'string', format: 'uuid' },
          owner: { $ref: '#/components/schemas/UserSummary' },
          isPublic: { type: 'boolean' },
          isCollaborative: { type: 'boolean' },
          trackCount: { type: 'integer' },
          followerCount: { type: 'integer' },
          imageUrl: { type: 'string', format: 'uri' },
        },
      },

      PlaylistResponse: {
        type: 'object',
        properties: {
          data: { $ref: '#/components/schemas/Playlist' },
        },
      },

      PlaylistListResponse: {
        type: 'object',
        properties: {
          data: { type: 'array', items: { $ref: '#/components/schemas/Playlist' } },
          pagination: { $ref: '#/components/schemas/Pagination' },
        },
      },

      CreatePlaylistRequest: {
        type: 'object',
        required: ['name'],
        properties: {
          name: { type: 'string', minLength: 1, maxLength: 200 },
          description: { type: 'string', maxLength: 2000 },
          isPublic: { type: 'boolean', default: true },
        },
      },

      // User schemas
      User: {
        type: 'object',
        properties: {
          id: { type: 'string', format: 'uuid' },
          email: { type: 'string', format: 'email' },
          displayName: { type: 'string' },
          avatarUrl: { type: 'string', format: 'uri' },
          country: { type: 'string' },
          followerCount: { type: 'integer' },
          followingCount: { type: 'integer' },
          subscription: { type: 'string', enum: ['free', 'premium', 'family'] },
        },
      },

      UserSummary: {
        type: 'object',
        properties: {
          id: { type: 'string', format: 'uuid' },
          displayName: { type: 'string' },
          avatarUrl: { type: 'string', format: 'uri' },
        },
      },

      UserResponse: {
        type: 'object',
        properties: {
          data: { $ref: '#/components/schemas/User' },
        },
      },

      // Search schema
      SearchResponse: {
        type: 'object',
        properties: {
          tracks: { $ref: '#/components/schemas/TrackListResponse' },
          artists: { $ref: '#/components/schemas/ArtistListResponse' },
          albums: {
            type: 'object',
            properties: {
              data: { type: 'array', items: { $ref: '#/components/schemas/Album' } },
            },
          },
          playlists: { $ref: '#/components/schemas/PlaylistListResponse' },
        },
      },

      // Common schemas
      Pagination: {
        type: 'object',
        properties: {
          page: { type: 'integer' },
          pageSize: { type: 'integer' },
          totalItems: { type: 'integer' },
          totalPages: { type: 'integer' },
          hasNextPage: { type: 'boolean' },
          hasPrevPage: { type: 'boolean' },
        },
      },

      PaginationLinks: {
        type: 'object',
        properties: {
          self: { type: 'string', format: 'uri' },
          first: { type: 'string', format: 'uri' },
          last: { type: 'string', format: 'uri' },
          next: { type: 'string', format: 'uri' },
          prev: { type: 'string', format: 'uri' },
        },
      },

      Error: {
        type: 'object',
        properties: {
          error: {
            type: 'object',
            properties: {
              code: { type: 'string' },
              message: { type: 'string' },
              details: { type: 'object' },
              timestamp: { type: 'string', format: 'date-time' },
              requestId: { type: 'string' },
            },
            required: ['code', 'message'],
          },
        },
      },
    },

    responses: {
      BadRequest: {
        description: 'Bad request - Invalid input',
        content: {
          'application/json': {
            schema: { $ref: '#/components/schemas/Error' },
          },
        },
      },
      Unauthorized: {
        description: 'Unauthorized - Invalid or missing authentication',
        content: {
          'application/json': {
            schema: { $ref: '#/components/schemas/Error' },
          },
        },
      },
      Forbidden: {
        description: 'Forbidden - Insufficient permissions',
        content: {
          'application/json': {
            schema: { $ref: '#/components/schemas/Error' },
          },
        },
      },
      NotFound: {
        description: 'Not found - Resource does not exist',
        content: {
          'application/json': {
            schema: { $ref: '#/components/schemas/Error' },
          },
        },
      },
      RateLimited: {
        description: 'Too many requests - Rate limit exceeded',
        headers: {
          'Retry-After': {
            description: 'Seconds to wait before retrying',
            schema: { type: 'integer' },
          },
        },
        content: {
          'application/json': {
            schema: { $ref: '#/components/schemas/Error' },
          },
        },
      },
    },

    securitySchemes: {
      bearerAuth: {
        type: 'http',
        scheme: 'bearer',
        bearerFormat: 'JWT',
        description: 'JWT access token obtained from the auth endpoint',
      },
      apiKey: {
        type: 'apiKey',
        in: 'header',
        name: 'X-API-Key',
        description: 'API key for server-to-server communication',
      },
      oauth2: {
        type: 'oauth2',
        flows: {
          authorizationCode: {
            authorizationUrl: 'https://accounts.mycelix.music/authorize',
            tokenUrl: 'https://accounts.mycelix.music/token',
            scopes: {
              'user-read': 'Read user profile',
              'user-modify': 'Modify user profile',
              'playlist-read': 'Read playlists',
              'playlist-modify': 'Create and modify playlists',
              'streaming': 'Stream music',
              'user-library-read': 'Read user library',
              'user-library-modify': 'Modify user library',
            },
          },
        },
      },
    },
  };
}

// ============================================================================
// ARCHITECTURE DECISION RECORDS
// ============================================================================

interface ADR {
  id: string;
  title: string;
  status: 'proposed' | 'accepted' | 'deprecated' | 'superseded';
  date: string;
  context: string;
  decision: string;
  consequences: string[];
  alternatives?: string[];
  supersededBy?: string;
}

const architectureDecisions: ADR[] = [
  {
    id: 'ADR-001',
    title: 'Use TypeScript for Backend Services',
    status: 'accepted',
    date: '2024-01-15',
    context: `
We need to choose a primary language for backend development. Key considerations:
- Developer productivity and hiring pool
- Type safety and refactoring support
- Ecosystem maturity for music/streaming
- Performance requirements
    `,
    decision: `
We will use TypeScript as the primary language for backend services, with the option
to use Rust for performance-critical components (audio processing, streaming).
    `,
    consequences: [
      'Faster iteration on business logic and API development',
      'Shared types between frontend and backend',
      'Need to monitor performance and optimize where necessary',
      'Rust integration via napi-rs for compute-intensive operations',
    ],
    alternatives: [
      'Go - Better performance but less ecosystem for audio',
      'Rust - Better performance but slower development velocity',
      'Python - Good ML ecosystem but performance concerns',
    ],
  },
  {
    id: 'ADR-002',
    title: 'Event-Driven Architecture for Real-Time Features',
    status: 'accepted',
    date: '2024-01-20',
    context: `
The platform requires real-time features including:
- Live listening activity
- Collaborative playlists
- Real-time notifications
- Analytics streaming
    `,
    decision: `
Adopt an event-driven architecture using Kafka for durable event streaming
and WebSockets for client-facing real-time communication.
    `,
    consequences: [
      'Decoupled services enable independent scaling',
      'Event replay capability for debugging and analytics',
      'Increased operational complexity',
      'Need for idempotent event handlers',
    ],
  },
  {
    id: 'ADR-003',
    title: 'PostgreSQL with Read Replicas for Primary Database',
    status: 'accepted',
    date: '2024-01-22',
    context: `
Need a primary database that handles:
- Complex queries for music catalog
- Transactional integrity for payments
- Full-text search capabilities
- High read throughput
    `,
    decision: `
Use PostgreSQL as the primary database with read replicas for scaling reads.
Supplement with Elasticsearch for advanced search and Redis for caching.
    `,
    consequences: [
      'Strong consistency for critical operations',
      'Native full-text search for simple queries',
      'Read replicas handle dashboard and analytics queries',
      'Need to manage replication lag for read-after-write scenarios',
    ],
  },
  {
    id: 'ADR-004',
    title: 'Multi-Region Active-Active Deployment',
    status: 'accepted',
    date: '2024-02-01',
    context: `
Global user base requires:
- Low latency for all users
- High availability (99.99% target)
- Disaster recovery
- Data sovereignty compliance
    `,
    decision: `
Deploy in multiple regions with active-active configuration using
latency-based routing and automatic failover.
    `,
    consequences: [
      'Sub-100ms latency for most users globally',
      'Complexity in data synchronization',
      'Higher infrastructure costs',
      'Need for conflict resolution strategies',
    ],
  },
  {
    id: 'ADR-005',
    title: 'HLS and DASH for Adaptive Streaming',
    status: 'accepted',
    date: '2024-02-10',
    context: `
Audio streaming requirements:
- Adaptive bitrate for varying network conditions
- Wide device compatibility
- DRM support
- Efficient CDN caching
    `,
    decision: `
Support both HLS (for Apple devices) and DASH (for others) with
common segment format and adaptive bitrate manifests.
    `,
    consequences: [
      'Maximum device compatibility',
      'Efficient content delivery via CDN',
      'Dual-format manifest generation complexity',
      'Storage overhead for multiple bitrates',
    ],
  },
];

// ============================================================================
// DEVELOPER GUIDE GENERATOR
// ============================================================================

function generateDeveloperGuide(): string {
  return `
# Mycelix Music Platform - Developer Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Architecture Overview](#architecture-overview)
3. [Local Development](#local-development)
4. [API Development](#api-development)
5. [Testing](#testing)
6. [Deployment](#deployment)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

- Node.js 20+ (LTS recommended)
- Docker and Docker Compose
- Git
- PostgreSQL client (for migrations)

### Quick Start

\`\`\`bash
# Clone the repository
git clone https://github.com/mycelix/mycelix-music.git
cd mycelix-music

# Install dependencies
npm install

# Start infrastructure
docker-compose up -d postgres redis elasticsearch kafka

# Run migrations
npx prisma migrate dev

# Start development server
npm run dev
\`\`\`

The API will be available at \`http://localhost:3000\`.

---

## Architecture Overview

### System Components

\`\`\`
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    API Gateway                               │
│  (Authentication, Rate Limiting, Request Routing)           │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
   │ API     │  │ Worker  │  │ Stream  │
   │ Service │  │ Service │  │ Service │
   └────┬────┘  └────┬────┘  └────┬────┘
        │             │             │
        └─────────────┼─────────────┘
                      │
   ┌──────────────────┼──────────────────┐
   │                  │                  │
┌──▼──┐         ┌────▼────┐        ┌────▼────┐
│Redis│         │PostgreSQL│        │  Kafka  │
└─────┘         └──────────┘        └─────────┘
\`\`\`

### Key Services

| Service | Purpose | Port |
|---------|---------|------|
| API | HTTP REST and GraphQL API | 3000 |
| Worker | Background job processing | - |
| Streaming | Audio streaming and transcoding | 3001 |
| Search | Elasticsearch indexing | - |

---

## Local Development

### Environment Setup

1. Copy the example environment file:
\`\`\`bash
cp .env.example .env
\`\`\`

2. Configure required variables:
\`\`\`env
DATABASE_URL=postgresql://mycelix:secret@localhost:5432/mycelix
REDIS_URL=redis://localhost:6379
JWT_SECRET=your-development-secret
\`\`\`

### Running Services

\`\`\`bash
# Start all infrastructure
docker-compose up -d

# Start API in development mode (with hot reload)
npm run dev

# Start workers
npm run worker:dev

# Run tests in watch mode
npm run test:watch
\`\`\`

### Database Migrations

\`\`\`bash
# Create a new migration
npx prisma migrate dev --name add_user_preferences

# Apply migrations
npx prisma migrate deploy

# Reset database (careful!)
npx prisma migrate reset
\`\`\`

---

## API Development

### Creating a New Endpoint

1. Define the route in \`src/api/routes/\`:

\`\`\`typescript
// src/api/routes/genres.ts
import { Router } from '../router';

export function registerGenreRoutes(router: Router) {
  router.get('/genres', async (ctx) => {
    const genres = await ctx.services.genre.list();
    ctx.body = { data: genres };
  });
}
\`\`\`

2. Add validation schema:

\`\`\`typescript
const createGenreSchema = {
  name: { type: 'string', required: true, minLength: 1 },
  description: { type: 'string', maxLength: 500 },
};
\`\`\`

3. Register in the main router.

### Error Handling

Use the built-in error classes:

\`\`\`typescript
import { NotFoundError, ValidationError } from '../errors';

// Throwing errors
throw new NotFoundError('Track', trackId);
throw new ValidationError('Invalid genre', { field: 'genre' });
\`\`\`

---

## Testing

### Running Tests

\`\`\`bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Run specific test file
npm test -- src/services/track.test.ts

# Run in watch mode
npm run test:watch
\`\`\`

### Writing Tests

\`\`\`typescript
describe('TrackService', () => {
  let trackService: TrackService;

  beforeEach(() => {
    trackService = new TrackService(mockDb, mockCache);
  });

  it('should create a track', async () => {
    const track = await trackService.create({
      title: 'Test Track',
      artistId: 'artist-123',
    });

    expect(track.id).toBeDefined();
    expect(track.title).toBe('Test Track');
  });
});
\`\`\`

---

## Deployment

### Building for Production

\`\`\`bash
# Build TypeScript
npm run build

# Build Docker image
docker build -t mycelix/api:latest .
\`\`\`

### Kubernetes Deployment

\`\`\`bash
# Deploy using Helm
helm upgrade --install mycelix ./deploy/helm/mycelix \\
  --namespace mycelix \\
  --create-namespace \\
  --values ./deploy/helm/mycelix/values.production.yaml

# Check deployment status
kubectl get pods -n mycelix
kubectl logs -f deployment/mycelix-api -n mycelix
\`\`\`

---

## Troubleshooting

### Common Issues

**Database connection errors:**
\`\`\`bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check connection
psql -h localhost -U mycelix -d mycelix
\`\`\`

**Redis connection errors:**
\`\`\`bash
# Check Redis is running
docker-compose ps redis

# Test connection
redis-cli ping
\`\`\`

**Port already in use:**
\`\`\`bash
# Find process using port
lsof -i :3000

# Kill process
kill -9 <PID>
\`\`\`

### Debug Mode

Enable debug logging:
\`\`\`bash
DEBUG=mycelix:* npm run dev
\`\`\`

### Support

- GitHub Issues: https://github.com/mycelix/mycelix-music/issues
- Slack: #mycelix-dev
- Email: dev@mycelix.music
  `.trim();
}

// ============================================================================
// OPERATIONS RUNBOOK
// ============================================================================

function generateOperationsRunbook(): string {
  return `
# Mycelix Music Platform - Operations Runbook

## Table of Contents
1. [Incident Response](#incident-response)
2. [Common Alerts](#common-alerts)
3. [Scaling Procedures](#scaling-procedures)
4. [Deployment Procedures](#deployment-procedures)
5. [Backup and Recovery](#backup-and-recovery)
6. [Health Checks](#health-checks)

---

## Incident Response

### Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| SEV1 | Complete outage | 15 min | API down, no streaming |
| SEV2 | Major degradation | 30 min | High latency, partial outage |
| SEV3 | Minor issue | 2 hours | Single feature broken |
| SEV4 | Low priority | 24 hours | Non-critical bug |

### Incident Process

1. **Detect**: Alert triggered or user report
2. **Triage**: Determine severity level
3. **Communicate**: Update status page
4. **Investigate**: Check dashboards and logs
5. **Mitigate**: Apply temporary fix
6. **Resolve**: Deploy permanent fix
7. **Review**: Post-incident review

### Key Dashboards

- **Grafana Main**: https://grafana.mycelix.music/d/main
- **API Metrics**: https://grafana.mycelix.music/d/api
- **Streaming Metrics**: https://grafana.mycelix.music/d/streaming
- **Database Metrics**: https://grafana.mycelix.music/d/database

---

## Common Alerts

### High API Latency (p99 > 500ms)

**Symptoms:**
- API response times elevated
- User complaints about slow loading

**Investigation:**
\`\`\`bash
# Check current latency
kubectl top pods -n mycelix

# Check database queries
kubectl exec -it mycelix-postgres-0 -n mycelix -- \\
  psql -U mycelix -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"

# Check Redis
kubectl exec -it mycelix-redis-0 -n mycelix -- redis-cli info stats
\`\`\`

**Remediation:**
1. Scale up API pods if CPU > 80%
2. Check for slow database queries
3. Verify Redis cache hit rate
4. Check external service dependencies

### High Error Rate (5xx > 1%)

**Investigation:**
\`\`\`bash
# Check error logs
kubectl logs -l app=mycelix-api -n mycelix --since=10m | grep ERROR

# Check pod health
kubectl get pods -n mycelix

# Check recent deployments
kubectl rollout history deployment/mycelix-api -n mycelix
\`\`\`

**Remediation:**
1. Rollback if recent deployment
2. Scale pods if resource constrained
3. Check database connections
4. Verify external services

### Database Connection Pool Exhausted

**Investigation:**
\`\`\`bash
# Check active connections
kubectl exec -it mycelix-postgres-0 -n mycelix -- \\
  psql -U mycelix -c "SELECT count(*) FROM pg_stat_activity;"

# Check application logs for connection errors
kubectl logs -l app=mycelix-api -n mycelix | grep "connection"
\`\`\`

**Remediation:**
1. Restart pods with connection leaks
2. Increase connection pool size temporarily
3. Identify and fix queries holding connections

---

## Scaling Procedures

### Horizontal Scaling (API)

\`\`\`bash
# Scale API pods
kubectl scale deployment mycelix-api -n mycelix --replicas=10

# Or update HPA limits
kubectl patch hpa mycelix-api -n mycelix \\
  -p '{"spec":{"maxReplicas":100}}'
\`\`\`

### Vertical Scaling (Database)

1. Create RDS snapshot
2. Modify instance class
3. Apply during maintenance window
4. Verify application connectivity

### Cache Scaling

\`\`\`bash
# Scale Redis cluster
kubectl scale statefulset mycelix-redis -n mycelix --replicas=5
\`\`\`

---

## Deployment Procedures

### Standard Deployment

\`\`\`bash
# Deploy new version
helm upgrade mycelix ./deploy/helm/mycelix \\
  --namespace mycelix \\
  --set api.image.tag=v1.2.3 \\
  --wait

# Verify deployment
kubectl rollout status deployment/mycelix-api -n mycelix
\`\`\`

### Rollback Procedure

\`\`\`bash
# Rollback to previous revision
kubectl rollout undo deployment/mycelix-api -n mycelix

# Rollback to specific revision
kubectl rollout undo deployment/mycelix-api -n mycelix --to-revision=5
\`\`\`

### Blue-Green Deployment

1. Deploy new version to green environment
2. Run smoke tests
3. Switch traffic to green
4. Keep blue for quick rollback
5. Decommission blue after validation

---

## Backup and Recovery

### Database Backups

**Automated Backups:**
- Continuous WAL archiving to S3
- Daily snapshots retained for 30 days
- Cross-region replication enabled

**Manual Backup:**
\`\`\`bash
# Create manual snapshot
aws rds create-db-snapshot \\
  --db-instance-identifier mycelix-production \\
  --db-snapshot-identifier mycelix-manual-$(date +%Y%m%d)
\`\`\`

### Recovery Procedures

**Point-in-Time Recovery:**
\`\`\`bash
# Restore to specific time
aws rds restore-db-instance-to-point-in-time \\
  --source-db-instance-identifier mycelix-production \\
  --target-db-instance-identifier mycelix-recovery \\
  --restore-time 2024-01-15T10:00:00Z
\`\`\`

### Disaster Recovery

**RTO:** 4 hours
**RPO:** 1 hour

1. Failover to DR region
2. Update DNS records
3. Verify application functionality
4. Notify stakeholders

---

## Health Checks

### Application Health

\`\`\`bash
# Check API health
curl https://api.mycelix.music/api/health

# Expected response:
# {"status":"healthy","version":"1.2.3","timestamp":"..."}
\`\`\`

### Infrastructure Health

\`\`\`bash
# Check all pods
kubectl get pods -n mycelix

# Check node resources
kubectl top nodes

# Check persistent volumes
kubectl get pv
\`\`\`

### Service Dependencies

| Service | Health Endpoint | Expected |
|---------|-----------------|----------|
| API | /api/health | 200 OK |
| PostgreSQL | pg_isready | 0 exit |
| Redis | PING | PONG |
| Elasticsearch | /_cluster/health | green/yellow |
| Kafka | broker list | 3 brokers |

---

## Contact Information

| Role | Contact | Escalation |
|------|---------|------------|
| On-Call | oncall@mycelix.music | PagerDuty |
| Platform Team | platform@mycelix.music | Slack #platform |
| Database Admin | dba@mycelix.music | Slack #database |
| Security | security@mycelix.music | security@mycelix.music |
  `.trim();
}

// ============================================================================
// DOCUMENTATION GENERATOR CLASS
// ============================================================================

class DocumentationGenerator {
  /**
   * Generate OpenAPI specification
   */
  generateOpenAPI(): object {
    return generateOpenAPISpec();
  }

  /**
   * Get Architecture Decision Records
   */
  getADRs(): ADR[] {
    return architectureDecisions;
  }

  /**
   * Generate developer guide
   */
  generateDeveloperGuide(): string {
    return generateDeveloperGuide();
  }

  /**
   * Generate operations runbook
   */
  generateOperationsRunbook(): string {
    return generateOperationsRunbook();
  }

  /**
   * Generate all documentation as a bundle
   */
  generateAll(): {
    openapi: object;
    adrs: ADR[];
    developerGuide: string;
    operationsRunbook: string;
  } {
    return {
      openapi: this.generateOpenAPI(),
      adrs: this.getADRs(),
      developerGuide: this.generateDeveloperGuide(),
      operationsRunbook: this.generateOperationsRunbook(),
    };
  }

  /**
   * Export OpenAPI spec to JSON
   */
  exportOpenAPIJson(): string {
    return JSON.stringify(this.generateOpenAPI(), null, 2);
  }

  /**
   * Export ADRs to markdown
   */
  exportADRsMarkdown(): string {
    const adrs = this.getADRs();
    let markdown = '# Architecture Decision Records\n\n';

    for (const adr of adrs) {
      markdown += `## ${adr.id}: ${adr.title}\n\n`;
      markdown += `**Status:** ${adr.status}\n`;
      markdown += `**Date:** ${adr.date}\n\n`;
      markdown += `### Context\n${adr.context.trim()}\n\n`;
      markdown += `### Decision\n${adr.decision.trim()}\n\n`;
      markdown += `### Consequences\n`;
      for (const consequence of adr.consequences) {
        markdown += `- ${consequence}\n`;
      }
      if (adr.alternatives) {
        markdown += `\n### Alternatives Considered\n`;
        for (const alt of adr.alternatives) {
          markdown += `- ${alt}\n`;
        }
      }
      markdown += '\n---\n\n';
    }

    return markdown;
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export {
  DocumentationGenerator,
  generateOpenAPISpec,
  generateDeveloperGuide,
  generateOperationsRunbook,
  architectureDecisions,
  ADR,
  OpenAPIInfo,
  OpenAPIServer,
  OpenAPIPath,
  OpenAPIOperation,
  OpenAPISchema,
};

/**
 * Create the documentation system
 */
export function createDocumentationSystem(): DocumentationGenerator {
  return new DocumentationGenerator();
}
