// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * API Versioning Middleware
 *
 * Supports version specification via:
 * 1. URL path: /api/v1/songs
 * 2. Header: Accept-Version: v1
 * 3. Query parameter: ?api-version=v1
 *
 * Usage:
 *   import { versionedRouter, apiVersion } from './versioning';
 *
 *   // Version-specific routes
 *   app.use('/api/v1', versionedRouter('v1'));
 *   app.use('/api/v2', versionedRouter('v2'));
 *
 *   // Or with middleware
 *   app.get('/api/songs', apiVersion(['v1', 'v2']), (req, res) => {
 *     if (req.apiVersion === 'v2') {
 *       // v2 behavior
 *     }
 *   });
 */

import { Router, Request, Response, NextFunction } from 'express';

// Extend Express Request type
declare global {
  namespace Express {
    interface Request {
      apiVersion?: string;
    }
  }
}

export type ApiVersion = 'v1' | 'v2';

const SUPPORTED_VERSIONS: ApiVersion[] = ['v1', 'v2'];
const DEFAULT_VERSION: ApiVersion = 'v1';
const LATEST_VERSION: ApiVersion = 'v2';

/**
 * Extract API version from request
 */
export function extractVersion(req: Request): ApiVersion {
  // Check URL path first (/api/v1/...)
  const pathMatch = req.path.match(/^\/api\/(v\d+)\//);
  if (pathMatch && SUPPORTED_VERSIONS.includes(pathMatch[1] as ApiVersion)) {
    return pathMatch[1] as ApiVersion;
  }

  // Check Accept-Version header
  const headerVersion = req.headers['accept-version'] as string;
  if (headerVersion && SUPPORTED_VERSIONS.includes(headerVersion as ApiVersion)) {
    return headerVersion as ApiVersion;
  }

  // Check query parameter
  const queryVersion = req.query['api-version'] as string;
  if (queryVersion && SUPPORTED_VERSIONS.includes(queryVersion as ApiVersion)) {
    return queryVersion as ApiVersion;
  }

  return DEFAULT_VERSION;
}

/**
 * Middleware to attach API version to request
 */
export function versionMiddleware(req: Request, res: Response, next: NextFunction): void {
  req.apiVersion = extractVersion(req);

  // Add version to response headers
  res.setHeader('X-API-Version', req.apiVersion);
  res.setHeader('X-API-Latest-Version', LATEST_VERSION);

  // Add deprecation warning for v1
  if (req.apiVersion === 'v1') {
    res.setHeader('Deprecation', 'version="v1"');
    res.setHeader('Sunset', new Date(Date.now() + 180 * 24 * 60 * 60 * 1000).toUTCString()); // 180 days
    res.setHeader('Link', `</api/v2${req.path}>; rel="successor-version"`);
  }

  next();
}

/**
 * Middleware to require specific API versions
 */
export function apiVersion(allowedVersions: ApiVersion[]) {
  return (req: Request, res: Response, next: NextFunction): void => {
    const version = extractVersion(req);
    req.apiVersion = version;

    if (!allowedVersions.includes(version)) {
      res.status(400).json({
        error: 'Unsupported API version',
        version,
        supported: allowedVersions,
        latest: LATEST_VERSION,
      });
      return;
    }

    res.setHeader('X-API-Version', version);
    next();
  };
}

/**
 * Create a versioned router
 */
export function versionedRouter(version: ApiVersion): Router {
  const router = Router();

  router.use((req: Request, res: Response, next: NextFunction) => {
    req.apiVersion = version;
    res.setHeader('X-API-Version', version);

    if (version === 'v1') {
      res.setHeader('Deprecation', 'version="v1"');
    }

    next();
  });

  return router;
}

/**
 * Response transformer for version-specific formatting
 */
export function versionedResponse<T>(
  req: Request,
  data: T,
  transformers: Partial<Record<ApiVersion, (data: T) => unknown>>
): unknown {
  const version = req.apiVersion || DEFAULT_VERSION;
  const transformer = transformers[version];

  if (transformer) {
    return transformer(data);
  }

  return data;
}

/**
 * Example version transformers for songs
 */
export const songTransformers = {
  v1: (songs: any[]) => songs.map(song => ({
    id: song.id,
    title: song.title,
    artist: song.artist,
    artistAddress: song.artist_address,
    genre: song.genre,
    ipfsHash: song.ipfs_hash,
    paymentModel: song.payment_model,
    plays: song.plays,
    earnings: song.earnings,
    createdAt: song.created_at,
  })),

  v2: (songs: any[]) => songs.map(song => ({
    id: song.id,
    title: song.title,
    artist: {
      name: song.artist,
      address: song.artist_address,
    },
    metadata: {
      genre: song.genre,
      description: song.description,
      coverArt: song.cover_art,
      audioUrl: song.audio_url,
    },
    blockchain: {
      ipfsHash: song.ipfs_hash,
      songHash: song.song_hash,
      claimStreamId: song.claim_stream_id,
    },
    economics: {
      paymentModel: song.payment_model,
      plays: song.plays,
      earnings: String(song.earnings), // String for precision
    },
    timestamps: {
      createdAt: song.created_at,
    },
  })),
};

/**
 * Get version info for documentation
 */
export function getVersionInfo() {
  return {
    current: LATEST_VERSION,
    supported: SUPPORTED_VERSIONS,
    default: DEFAULT_VERSION,
    deprecated: ['v1'],
    changelog: {
      v2: [
        'Nested response structure for better organization',
        'String earnings for better precision',
        'Added blockchain metadata (songHash, claimStreamId)',
        'Added artist object structure',
      ],
      v1: [
        'Initial API version',
        'Flat response structure',
        'Deprecated: scheduled for removal in 6 months',
      ],
    },
  };
}

export { SUPPORTED_VERSIONS, DEFAULT_VERSION, LATEST_VERSION };
