// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Read Replica Routing
 *
 * Routes read-heavy queries to replicas for horizontal scaling.
 * Write operations always go to primary.
 */

import { Pool, PoolClient, QueryResult, QueryConfig } from 'pg';
import { getLogger } from '../logging';
import { getMetrics } from '../metrics';

const logger = getLogger();

/**
 * Replica configuration
 */
export interface ReplicaConfig {
  /** Primary database URL */
  primaryUrl: string;
  /** Replica database URLs */
  replicaUrls: string[];
  /** Pool size per connection */
  poolSize?: number;
  /** Load balancing strategy */
  strategy: 'round-robin' | 'random' | 'least-connections';
  /** Lag tolerance in milliseconds */
  maxLagMs?: number;
  /** Health check interval */
  healthCheckInterval?: number;
}

/**
 * Replica health status
 */
interface ReplicaHealth {
  url: string;
  pool: Pool;
  healthy: boolean;
  lagMs: number;
  lastCheck: Date;
  activeConnections: number;
  errorCount: number;
}

/**
 * Database Router
 */
export class DatabaseRouter {
  private primary: Pool;
  private replicas: ReplicaHealth[] = [];
  private currentReplicaIndex = 0;
  private healthCheckTimer?: NodeJS.Timeout;
  private config: Required<ReplicaConfig>;

  constructor(config: ReplicaConfig) {
    this.config = {
      poolSize: 10,
      maxLagMs: 5000,
      healthCheckInterval: 30000,
      ...config,
    };

    // Initialize primary
    this.primary = new Pool({
      connectionString: config.primaryUrl,
      max: this.config.poolSize,
    });

    // Initialize replicas
    for (const url of config.replicaUrls) {
      const pool = new Pool({
        connectionString: url,
        max: this.config.poolSize,
      });

      this.replicas.push({
        url,
        pool,
        healthy: true,
        lagMs: 0,
        lastCheck: new Date(),
        activeConnections: 0,
        errorCount: 0,
      });
    }

    // Register metrics
    const metrics = getMetrics();
    metrics.createCounter('db_queries_routed_total', 'Routed database queries', ['target']);
    metrics.createGauge('db_replica_lag_ms', 'Replica lag in milliseconds', ['replica']);
    metrics.createGauge('db_replica_healthy', 'Replica health status', ['replica']);

    // Start health checks
    if (this.replicas.length > 0) {
      this.startHealthChecks();
    }

    logger.info('Database router initialized', {
      primary: this.maskUrl(config.primaryUrl),
      replicas: config.replicaUrls.length,
    });
  }

  /**
   * Get pool for read operations
   */
  getReadPool(): Pool {
    const replica = this.selectReplica();
    if (replica) {
      getMetrics().incCounter('db_queries_routed_total', { target: 'replica' });
      return replica.pool;
    }

    // Fallback to primary if no healthy replicas
    getMetrics().incCounter('db_queries_routed_total', { target: 'primary' });
    return this.primary;
  }

  /**
   * Get pool for write operations
   */
  getWritePool(): Pool {
    getMetrics().incCounter('db_queries_routed_total', { target: 'primary' });
    return this.primary;
  }

  /**
   * Get primary pool directly
   */
  getPrimary(): Pool {
    return this.primary;
  }

  /**
   * Execute a read query (routed to replica)
   */
  async read<T extends QueryResult = QueryResult>(
    text: string,
    values?: unknown[]
  ): Promise<T> {
    const pool = this.getReadPool();
    return pool.query(text, values) as Promise<T>;
  }

  /**
   * Execute a write query (always primary)
   */
  async write<T extends QueryResult = QueryResult>(
    text: string,
    values?: unknown[]
  ): Promise<T> {
    return this.primary.query(text, values) as Promise<T>;
  }

  /**
   * Execute transaction (always primary)
   */
  async transaction<T>(
    fn: (client: PoolClient) => Promise<T>
  ): Promise<T> {
    const client = await this.primary.connect();
    try {
      await client.query('BEGIN');
      const result = await fn(client);
      await client.query('COMMIT');
      return result;
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }

  /**
   * Select a replica based on strategy
   */
  private selectReplica(): ReplicaHealth | null {
    const healthyReplicas = this.replicas.filter(
      r => r.healthy && r.lagMs <= this.config.maxLagMs
    );

    if (healthyReplicas.length === 0) {
      logger.warn('No healthy replicas available, falling back to primary');
      return null;
    }

    switch (this.config.strategy) {
      case 'round-robin':
        return this.selectRoundRobin(healthyReplicas);
      case 'random':
        return this.selectRandom(healthyReplicas);
      case 'least-connections':
        return this.selectLeastConnections(healthyReplicas);
      default:
        return healthyReplicas[0];
    }
  }

  /**
   * Round-robin selection
   */
  private selectRoundRobin(replicas: ReplicaHealth[]): ReplicaHealth {
    const replica = replicas[this.currentReplicaIndex % replicas.length];
    this.currentReplicaIndex++;
    return replica;
  }

  /**
   * Random selection
   */
  private selectRandom(replicas: ReplicaHealth[]): ReplicaHealth {
    return replicas[Math.floor(Math.random() * replicas.length)];
  }

  /**
   * Least connections selection
   */
  private selectLeastConnections(replicas: ReplicaHealth[]): ReplicaHealth {
    return replicas.reduce((min, r) =>
      r.activeConnections < min.activeConnections ? r : min
    );
  }

  /**
   * Start health check loop
   */
  private startHealthChecks(): void {
    this.healthCheckTimer = setInterval(
      () => this.checkReplicaHealth(),
      this.config.healthCheckInterval
    );

    // Initial check
    this.checkReplicaHealth();
  }

  /**
   * Check health of all replicas
   */
  private async checkReplicaHealth(): Promise<void> {
    const metrics = getMetrics();

    for (let i = 0; i < this.replicas.length; i++) {
      const replica = this.replicas[i];

      try {
        // Check connectivity and get lag
        const start = Date.now();
        const result = await replica.pool.query(`
          SELECT
            CASE
              WHEN pg_last_wal_receive_lsn() = pg_last_wal_replay_lsn()
              THEN 0
              ELSE EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) * 1000
            END AS lag_ms
        `);

        const lagMs = parseFloat(result.rows[0]?.lag_ms || '0');
        const latency = Date.now() - start;

        replica.lagMs = lagMs;
        replica.healthy = lagMs <= this.config.maxLagMs;
        replica.lastCheck = new Date();
        replica.errorCount = 0;

        // Get connection count
        const connResult = await replica.pool.query(
          'SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()'
        );
        replica.activeConnections = parseInt(connResult.rows[0].count);

        // Update metrics
        metrics.setGauge('db_replica_lag_ms', lagMs, { replica: this.getReplicaLabel(i) });
        metrics.setGauge('db_replica_healthy', replica.healthy ? 1 : 0, {
          replica: this.getReplicaLabel(i),
        });

        logger.debug('Replica health check passed', {
          replica: this.getReplicaLabel(i),
          lagMs,
          latency,
          connections: replica.activeConnections,
        });
      } catch (error) {
        replica.errorCount++;
        replica.healthy = replica.errorCount < 3; // Allow 2 failures
        replica.lastCheck = new Date();

        metrics.setGauge('db_replica_healthy', 0, { replica: this.getReplicaLabel(i) });

        logger.warn('Replica health check failed', {
          replica: this.getReplicaLabel(i),
          error: (error as Error).message,
          errorCount: replica.errorCount,
        });
      }
    }
  }

  /**
   * Get replica label for metrics
   */
  private getReplicaLabel(index: number): string {
    return `replica-${index + 1}`;
  }

  /**
   * Mask sensitive URL parts
   */
  private maskUrl(url: string): string {
    try {
      const parsed = new URL(url);
      return `${parsed.protocol}//${parsed.host}${parsed.pathname}`;
    } catch {
      return '***';
    }
  }

  /**
   * Get router statistics
   */
  getStats(): {
    primary: { healthy: boolean; connections: number };
    replicas: Array<{
      healthy: boolean;
      lagMs: number;
      connections: number;
      lastCheck: Date;
    }>;
  } {
    return {
      primary: {
        healthy: true,
        connections: this.primary.totalCount,
      },
      replicas: this.replicas.map(r => ({
        healthy: r.healthy,
        lagMs: r.lagMs,
        connections: r.activeConnections,
        lastCheck: r.lastCheck,
      })),
    };
  }

  /**
   * Shutdown all pools
   */
  async shutdown(): Promise<void> {
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
    }

    await Promise.all([
      this.primary.end(),
      ...this.replicas.map(r => r.pool.end()),
    ]);

    logger.info('Database router shut down');
  }
}

/**
 * Create router from environment
 */
export function createRouterFromEnv(): DatabaseRouter {
  const primaryUrl = process.env.DATABASE_URL;
  if (!primaryUrl) {
    throw new Error('DATABASE_URL is required');
  }

  const replicaUrls = (process.env.DATABASE_REPLICA_URLS || '')
    .split(',')
    .filter(Boolean);

  return new DatabaseRouter({
    primaryUrl,
    replicaUrls,
    strategy: (process.env.DB_ROUTING_STRATEGY as any) || 'round-robin',
    poolSize: parseInt(process.env.DB_POOL_SIZE || '10'),
    maxLagMs: parseInt(process.env.DB_MAX_LAG_MS || '5000'),
  });
}

export default DatabaseRouter;
