// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Content Compression Middleware
 *
 * Brotli/gzip response compression with smart defaults.
 * Significant bandwidth savings for JSON responses.
 */

import { Request, Response, NextFunction } from 'express';
import { createBrotliCompress, createGzip, createDeflate, constants } from 'zlib';
import { Transform, pipeline } from 'stream';
import { getMetrics } from '../metrics';

/**
 * Compression configuration
 */
export interface CompressionConfig {
  /** Minimum response size to compress (bytes) */
  threshold: number;
  /** Compression level (1-9 for gzip, 1-11 for brotli) */
  level: number;
  /** Preferred encoding */
  preferredEncoding: 'br' | 'gzip' | 'deflate';
  /** Content types to compress */
  compressibleTypes: RegExp;
  /** Skip compression for specific paths */
  skipPaths?: RegExp[];
  /** Enable brotli */
  brotli: boolean;
  /** Memory level for zlib */
  memLevel?: number;
}

/**
 * Default configuration
 */
const defaultConfig: CompressionConfig = {
  threshold: 1024, // 1KB minimum
  level: 6,
  preferredEncoding: 'br',
  compressibleTypes: /^(text\/|application\/json|application\/javascript|application\/xml|image\/svg)/,
  brotli: true,
  memLevel: 8,
};

/**
 * Parse Accept-Encoding header
 */
function parseAcceptEncoding(header: string): Map<string, number> {
  const encodings = new Map<string, number>();

  if (!header) return encodings;

  header.split(',').forEach(part => {
    const [encoding, ...params] = part.trim().split(';');
    let quality = 1;

    for (const param of params) {
      const match = param.trim().match(/^q=([\d.]+)$/);
      if (match) {
        quality = parseFloat(match[1]);
      }
    }

    encodings.set(encoding.trim().toLowerCase(), quality);
  });

  return encodings;
}

/**
 * Select best encoding based on client support
 */
function selectEncoding(
  acceptEncoding: string,
  config: CompressionConfig
): 'br' | 'gzip' | 'deflate' | null {
  const encodings = parseAcceptEncoding(acceptEncoding);

  // Check for identity (no compression)
  if (encodings.get('identity') === 0) {
    // Client explicitly rejects identity, must compress
  } else if (encodings.get('*') === 0) {
    // Client rejects all encodings
    return null;
  }

  // Preferred order
  const order: Array<'br' | 'gzip' | 'deflate'> = config.brotli
    ? ['br', 'gzip', 'deflate']
    : ['gzip', 'deflate'];

  // Check preferred encoding first
  if (encodings.has(config.preferredEncoding) && encodings.get(config.preferredEncoding)! > 0) {
    if (config.preferredEncoding === 'br' && config.brotli) {
      return 'br';
    }
    if (config.preferredEncoding === 'gzip' || config.preferredEncoding === 'deflate') {
      return config.preferredEncoding;
    }
  }

  // Fall back to best available
  for (const encoding of order) {
    const quality = encodings.get(encoding);
    if (quality !== undefined && quality > 0) {
      if (encoding === 'br' && !config.brotli) continue;
      return encoding;
    }
  }

  return null;
}

/**
 * Create compressor stream
 */
function createCompressor(
  encoding: 'br' | 'gzip' | 'deflate',
  config: CompressionConfig
): Transform {
  switch (encoding) {
    case 'br':
      return createBrotliCompress({
        params: {
          [constants.BROTLI_PARAM_QUALITY]: Math.min(config.level, 11),
          [constants.BROTLI_PARAM_MODE]: constants.BROTLI_MODE_TEXT,
        },
      });

    case 'gzip':
      return createGzip({
        level: Math.min(config.level, 9),
        memLevel: config.memLevel,
      });

    case 'deflate':
      return createDeflate({
        level: Math.min(config.level, 9),
        memLevel: config.memLevel,
      });
  }
}

/**
 * Check if content type is compressible
 */
function isCompressible(contentType: string, config: CompressionConfig): boolean {
  if (!contentType) return false;
  return config.compressibleTypes.test(contentType);
}

/**
 * Compression middleware
 */
export function compressionMiddleware(config: Partial<CompressionConfig> = {}) {
  const cfg: CompressionConfig = { ...defaultConfig, ...config };
  const metrics = getMetrics();

  // Register metrics
  metrics.createCounter('compression_requests_total', 'Compressed responses', ['encoding']);
  metrics.createCounter('compression_bytes_saved', 'Bytes saved by compression', ['encoding']);

  return (req: Request, res: Response, next: NextFunction): void => {
    // Skip if no accept-encoding
    const acceptEncoding = req.headers['accept-encoding'] as string;
    if (!acceptEncoding) {
      return next();
    }

    // Skip for certain paths
    if (cfg.skipPaths?.some(pattern => pattern.test(req.path))) {
      return next();
    }

    // Select encoding
    const encoding = selectEncoding(acceptEncoding, cfg);
    if (!encoding) {
      return next();
    }

    // Store original methods
    const originalWrite = res.write.bind(res);
    const originalEnd = res.end.bind(res);

    let compressor: Transform | null = null;
    let compressionStarted = false;
    let originalLength = 0;
    let compressedLength = 0;
    const chunks: Buffer[] = [];

    // Check if we should compress after headers are set
    const shouldCompress = (): boolean => {
      // Already started
      if (compressionStarted) return true;

      // Check content type
      const contentType = res.getHeader('Content-Type') as string;
      if (!isCompressible(contentType, cfg)) {
        return false;
      }

      // Check content length
      const contentLength = res.getHeader('Content-Length');
      if (contentLength && parseInt(contentLength as string) < cfg.threshold) {
        return false;
      }

      // Check if already encoded
      if (res.getHeader('Content-Encoding')) {
        return false;
      }

      return true;
    };

    // Start compression
    const startCompression = (): void => {
      if (compressionStarted) return;
      compressionStarted = true;

      compressor = createCompressor(encoding, cfg);

      // Set headers
      res.setHeader('Content-Encoding', encoding);
      res.removeHeader('Content-Length');
      res.setHeader('Vary', 'Accept-Encoding');

      // Collect compressed data
      compressor.on('data', (chunk: Buffer) => {
        compressedLength += chunk.length;
        originalWrite(chunk);
      });

      compressor.on('end', () => {
        originalEnd();

        // Record metrics
        const saved = originalLength - compressedLength;
        if (saved > 0) {
          metrics.incCounter('compression_requests_total', { encoding });
          metrics.incCounter('compression_bytes_saved', { encoding }, saved);
        }
      });
    };

    // Override write
    res.write = function (chunk: any, ...args: any[]): boolean {
      if (chunk) {
        const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
        originalLength += buffer.length;
        chunks.push(buffer);
      }

      // Don't write yet, buffer until end
      return true;
    };

    // Override end
    res.end = function (chunk?: any, ...args: any[]): Response {
      if (chunk) {
        const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
        originalLength += buffer.length;
        chunks.push(buffer);
      }

      // Check if we should compress
      if (shouldCompress() && originalLength >= cfg.threshold) {
        startCompression();

        // Write all buffered chunks
        for (const buf of chunks) {
          compressor!.write(buf);
        }
        compressor!.end();
      } else {
        // Write uncompressed
        for (const buf of chunks) {
          originalWrite(buf);
        }
        originalEnd();
      }

      return res;
    };

    next();
  };
}

/**
 * Static compression presets
 */
export const compressionPresets = {
  /** Fast compression (lower CPU, larger output) */
  fast: {
    level: 1,
    brotli: false,
  },

  /** Balanced (default) */
  balanced: {
    level: 6,
    brotli: true,
  },

  /** Maximum compression (higher CPU, smaller output) */
  maximum: {
    level: 9,
    brotli: true,
  },

  /** API-optimized (fast brotli for JSON) */
  api: {
    level: 4,
    brotli: true,
    threshold: 512,
    compressibleTypes: /^application\/json/,
  },
};

/**
 * Compression stats helper
 */
export function getCompressionRatio(original: number, compressed: number): {
  ratio: number;
  percent: number;
  saved: number;
} {
  const ratio = compressed / original;
  const percent = ((original - compressed) / original) * 100;
  const saved = original - compressed;

  return { ratio, percent, saved };
}

/**
 * Manually compress content
 */
export async function compress(
  content: string | Buffer,
  encoding: 'br' | 'gzip' | 'deflate' = 'gzip',
  level = 6
): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    const compressor = createCompressor(encoding, {
      ...defaultConfig,
      level,
    });

    compressor.on('data', (chunk: Buffer) => chunks.push(chunk));
    compressor.on('end', () => resolve(Buffer.concat(chunks)));
    compressor.on('error', reject);

    compressor.write(content);
    compressor.end();
  });
}

export default compressionMiddleware;
