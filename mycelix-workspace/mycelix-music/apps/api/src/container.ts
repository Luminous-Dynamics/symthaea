// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Dependency Injection Container
 *
 * Provides centralized dependency management for:
 * - Database connections
 * - Redis clients
 * - Repositories
 * - Services
 * - Rate limiters
 *
 * Supports lazy initialization and proper cleanup.
 */

import { Pool } from 'pg';
import Redis from 'ioredis';
import { SongRepository, PlayRepository, createRepositories, Repositories } from './repositories';
import { CacheService, createCacheService } from './services/cache.service';
import { SongService } from './services/song.service';
import { PlayService } from './services/play.service';
import { createRateLimiters, RateLimiter } from './middleware/rate-limit';

/**
 * Container configuration
 */
export interface ContainerConfig {
  database: {
    connectionString: string;
    maxConnections?: number;
    idleTimeout?: number;
  };
  redis?: {
    url: string;
    keyPrefix?: string;
  };
  cache?: {
    ttl?: number;
  };
}

/**
 * Rate limiters container
 */
export interface RateLimiters {
  standard: RateLimiter;
  strict: RateLimiter;
  auth: RateLimiter;
}

/**
 * Services container
 */
export interface Services {
  songs: SongService;
  plays: PlayService;
}

/**
 * Dependency injection container
 */
export class Container {
  private _pool: Pool | null = null;
  private _redis: Redis | null = null;
  private _repositories: Repositories | null = null;
  private _cache: CacheService | null = null;
  private _services: Services | null = null;
  private _rateLimiters: RateLimiters | null = null;
  private _initialized = false;

  constructor(private readonly config: ContainerConfig) {}

  /**
   * Initialize all dependencies
   */
  async initialize(): Promise<void> {
    if (this._initialized) return;

    // Initialize database pool
    this._pool = new Pool({
      connectionString: this.config.database.connectionString,
      max: this.config.database.maxConnections || 20,
      idleTimeoutMillis: this.config.database.idleTimeout || 30000,
    });

    // Test database connection
    const client = await this._pool.connect();
    await client.query('SELECT 1');
    client.release();

    // Initialize Redis if configured
    if (this.config.redis) {
      this._redis = new Redis(this.config.redis.url, {
        keyPrefix: this.config.redis.keyPrefix || 'mycelix:',
        lazyConnect: true,
      });
      await this._redis.connect();
    }

    // Initialize repositories
    this._repositories = createRepositories(this._pool);

    // Initialize cache
    this._cache = createCacheService(this._redis || undefined);

    // Initialize services
    this._services = {
      songs: new SongService(
        this._repositories.songs,
        this._cache,
      ),
      plays: new PlayService(
        this._repositories.plays,
        this._repositories.songs,
        this._cache,
      ),
    };

    // Initialize rate limiters
    this._rateLimiters = createRateLimiters(this._redis || undefined);

    this._initialized = true;
    console.log('Container initialized successfully');
  }

  /**
   * Shutdown all dependencies
   */
  async shutdown(): Promise<void> {
    if (!this._initialized) return;

    // Close Redis connection
    if (this._redis) {
      await this._redis.quit();
      this._redis = null;
    }

    // Close database pool
    if (this._pool) {
      await this._pool.end();
      this._pool = null;
    }

    this._repositories = null;
    this._cache = null;
    this._services = null;
    this._rateLimiters = null;
    this._initialized = false;

    console.log('Container shutdown complete');
  }

  /**
   * Get database pool
   */
  get pool(): Pool {
    this.ensureInitialized();
    return this._pool!;
  }

  /**
   * Get Redis client (may be null)
   */
  get redis(): Redis | null {
    return this._redis;
  }

  /**
   * Get repositories
   */
  get repositories(): Repositories {
    this.ensureInitialized();
    return this._repositories!;
  }

  /**
   * Get cache service
   */
  get cache(): CacheService {
    this.ensureInitialized();
    return this._cache!;
  }

  /**
   * Get services
   */
  get services(): Services {
    this.ensureInitialized();
    return this._services!;
  }

  /**
   * Get rate limiters
   */
  get rateLimiters(): RateLimiters {
    this.ensureInitialized();
    return this._rateLimiters!;
  }

  /**
   * Check if container is initialized
   */
  get isInitialized(): boolean {
    return this._initialized;
  }

  /**
   * Ensure container is initialized
   */
  private ensureInitialized(): void {
    if (!this._initialized) {
      throw new Error('Container not initialized. Call initialize() first.');
    }
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<{
    database: boolean;
    redis: boolean;
  }> {
    const health = {
      database: false,
      redis: false,
    };

    // Check database
    try {
      if (this._pool) {
        const client = await this._pool.connect();
        await client.query('SELECT 1');
        client.release();
        health.database = true;
      }
    } catch (error) {
      console.error('Database health check failed:', error);
    }

    // Check Redis
    try {
      if (this._redis) {
        await this._redis.ping();
        health.redis = true;
      } else {
        // Redis not configured, considered healthy
        health.redis = true;
      }
    } catch (error) {
      console.error('Redis health check failed:', error);
    }

    return health;
  }
}

/**
 * Create container from environment variables
 */
export function createContainerFromEnv(): Container {
  const config: ContainerConfig = {
    database: {
      connectionString: process.env.DATABASE_URL || 'postgresql://localhost:5432/mycelix',
      maxConnections: parseInt(process.env.DB_MAX_CONNECTIONS || '20', 10),
      idleTimeout: parseInt(process.env.DB_IDLE_TIMEOUT || '30000', 10),
    },
  };

  // Add Redis if configured
  if (process.env.REDIS_URL) {
    config.redis = {
      url: process.env.REDIS_URL,
      keyPrefix: process.env.REDIS_KEY_PREFIX || 'mycelix:',
    };
  }

  return new Container(config);
}

/**
 * Global container instance
 */
let _container: Container | null = null;

/**
 * Get or create the global container
 */
export function getContainer(): Container {
  if (!_container) {
    _container = createContainerFromEnv();
  }
  return _container;
}

/**
 * Set the global container (useful for testing)
 */
export function setContainer(container: Container): void {
  _container = container;
}

/**
 * Reset the global container
 */
export async function resetContainer(): Promise<void> {
  if (_container) {
    await _container.shutdown();
    _container = null;
  }
}
