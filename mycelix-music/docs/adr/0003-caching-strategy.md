# ADR 0003: Caching Strategy

## Status
Accepted

## Context
The API serves data that varies in volatility:
- **High volatility**: Play counts, earnings (real-time updates)
- **Medium volatility**: Song lists, search results (minutes)
- **Low volatility**: Genre lists, static content (hours)

We need caching that:
- Reduces database load
- Supports distributed environments
- Allows targeted invalidation
- Works in development without Redis

## Decision
Implement a **cache abstraction layer** supporting multiple backends:

### Cache Service Architecture
```
CacheService
    ├── MemoryCache (development/testing)
    └── RedisCache (production)
```

### Caching Patterns

1. **Cache-Aside (Read-Through)**
   ```typescript
   const song = await cache.getOrSet(
     `song:${id}`,
     () => songRepo.findById(id),
     { ttl: 300, tags: ['songs'] }
   );
   ```

2. **Cache Tags for Invalidation**
   - `songs` - Invalidate all song-related caches
   - `plays` - Invalidate play statistics
   - `artists` - Invalidate artist data

3. **Stale-While-Revalidate (SWR)**
   - Return stale data immediately
   - Refresh in background
   - Useful for non-critical data

### TTL Guidelines
| Data Type | TTL | Rationale |
|-----------|-----|-----------|
| Song details | 5 min | Moderate update frequency |
| Song lists | 1 min | More dynamic |
| Artist stats | 5 min | Aggregated, less volatile |
| Genre list | 1 hour | Rarely changes |
| Top songs | 5 min | Updates with plays |

### HTTP Caching
- Set `Cache-Control` headers appropriately
- Support `ETag` and `If-None-Match`
- Return `304 Not Modified` when appropriate

## Consequences

### Positive
- Significant reduction in database queries
- Fast development setup (no Redis required)
- Tag-based invalidation simplifies cache management
- HTTP caching reduces bandwidth

### Negative
- Cache invalidation complexity
- Potential for stale data in edge cases
- Additional infrastructure (Redis) in production

### Mitigations
- Short TTLs for volatile data
- Aggressive invalidation on writes
- Include cache status in response headers for debugging
