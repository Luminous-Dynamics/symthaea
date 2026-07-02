// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Cache Package
 *
 * Redis-backed caching with multiple strategies.
 */

export { CacheClient, createCacheClient, CacheOptions } from './client';
export { cacheMiddleware, CacheMiddlewareOptions } from './middleware';
export { cached, invalidate, CacheDecorator } from './decorators';
export {
  CachePattern,
  CachePatterns,
  buildKey,
  parseKey,
  keyPatterns
} from './patterns';
