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
