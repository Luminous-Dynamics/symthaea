// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Query Performance Monitor
 *
 * Monitors database query performance, identifies slow queries,
 * and provides insights for optimization.
 */

import { Pool, PoolClient, QueryResult, QueryConfig } from 'pg';
import { getMetrics } from '../metrics';

/**
 * Query execution record
 */
interface QueryRecord {
  query: string;
  params?: unknown[];
  duration: number;
  timestamp: Date;
  rowCount: number;
  cached: boolean;
  error?: string;
  stack?: string;
}

/**
 * Query statistics
 */
interface QueryStats {
  query: string;
  totalExecutions: number;
  totalDuration: number;
  avgDuration: number;
  minDuration: number;
  maxDuration: number;
  p95Duration: number;
  totalRows: number;
  errorCount: number;
  lastExecuted: Date;
}

/**
 * Monitor configuration
 */
interface MonitorConfig {
  /** Slow query threshold in milliseconds */
  slowQueryThreshold: number;
  /** Maximum query records to keep */
  maxRecords: number;
  /** Enable query logging */
  logQueries: boolean;
  /** Log slow queries to console */
  logSlowQueries: boolean;
  /** Enable explain analysis for slow queries */
  explainSlowQueries: boolean;
  /** Sampling rate (0-1) */
  sampleRate: number;
}

/**
 * Default configuration
 */
const defaultConfig: MonitorConfig = {
  slowQueryThreshold: 100, // 100ms
  maxRecords: 10000,
  logQueries: false,
  logSlowQueries: true,
  explainSlowQueries: false,
  sampleRate: 1,
};

/**
 * Query Performance Monitor
 */
export class QueryMonitor {
  private config: MonitorConfig;
  private records: QueryRecord[] = [];
  private statsMap: Map<string, QueryStats> = new Map();
  private slowQueries: QueryRecord[] = [];

  constructor(config: Partial<MonitorConfig> = {}) {
    this.config = { ...defaultConfig, ...config };
  }

  /**
   * Record a query execution
   */
  record(record: QueryRecord): void {
    // Sampling
    if (Math.random() > this.config.sampleRate) return;

    // Store record
    this.records.push(record);
    if (this.records.length > this.config.maxRecords) {
      this.records.shift();
    }

    // Update statistics
    this.updateStats(record);

    // Track slow queries
    if (record.duration >= this.config.slowQueryThreshold) {
      this.slowQueries.push(record);
      if (this.slowQueries.length > 100) {
        this.slowQueries.shift();
      }

      if (this.config.logSlowQueries) {
        this.logSlowQuery(record);
      }
    }

    // Update metrics
    const metrics = getMetrics();
    const operation = this.extractOperation(record.query);
    const table = this.extractTable(record.query);

    metrics.incCounter('db_queries_total', { operation, table });
    metrics.observeHistogram('db_query_duration_seconds', record.duration / 1000, {
      operation,
      table,
    });

    if (record.error) {
      metrics.incCounter('errors_total', { type: 'database', code: 'query_error' });
    }
  }

  /**
   * Update query statistics
   */
  private updateStats(record: QueryRecord): void {
    const normalized = this.normalizeQuery(record.query);
    let stats = this.statsMap.get(normalized);

    if (!stats) {
      stats = {
        query: normalized,
        totalExecutions: 0,
        totalDuration: 0,
        avgDuration: 0,
        minDuration: Infinity,
        maxDuration: 0,
        p95Duration: 0,
        totalRows: 0,
        errorCount: 0,
        lastExecuted: record.timestamp,
      };
      this.statsMap.set(normalized, stats);
    }

    stats.totalExecutions++;
    stats.totalDuration += record.duration;
    stats.avgDuration = stats.totalDuration / stats.totalExecutions;
    stats.minDuration = Math.min(stats.minDuration, record.duration);
    stats.maxDuration = Math.max(stats.maxDuration, record.duration);
    stats.totalRows += record.rowCount;
    stats.lastExecuted = record.timestamp;

    if (record.error) {
      stats.errorCount++;
    }
  }

  /**
   * Log slow query
   */
  private logSlowQuery(record: QueryRecord): void {
    console.warn(`🐢 Slow query detected (${record.duration.toFixed(2)}ms):`);
    console.warn(`   Query: ${record.query.slice(0, 200)}...`);
    if (record.params?.length) {
      console.warn(`   Params: ${JSON.stringify(record.params).slice(0, 100)}`);
    }
    if (record.stack) {
      const relevantStack = record.stack
        .split('\n')
        .filter(line => !line.includes('node_modules'))
        .slice(0, 3)
        .join('\n');
      console.warn(`   Stack: ${relevantStack}`);
    }
  }

  /**
   * Normalize query for aggregation
   */
  private normalizeQuery(query: string): string {
    return query
      // Remove parameter values
      .replace(/\$\d+/g, '$?')
      // Normalize whitespace
      .replace(/\s+/g, ' ')
      // Remove specific values in IN clauses
      .replace(/IN\s*\([^)]+\)/gi, 'IN (?)')
      // Remove specific UUIDs
      .replace(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi, '?')
      // Normalize numbers
      .replace(/\b\d+\b/g, '?')
      .trim()
      .toLowerCase();
  }

  /**
   * Extract SQL operation
   */
  private extractOperation(query: string): string {
    const normalized = query.trim().toUpperCase();
    if (normalized.startsWith('SELECT')) return 'select';
    if (normalized.startsWith('INSERT')) return 'insert';
    if (normalized.startsWith('UPDATE')) return 'update';
    if (normalized.startsWith('DELETE')) return 'delete';
    return 'other';
  }

  /**
   * Extract table name
   */
  private extractTable(query: string): string {
    const normalized = query.trim().toLowerCase();
    const fromMatch = normalized.match(/from\s+(\w+)/);
    if (fromMatch) return fromMatch[1];
    const intoMatch = normalized.match(/into\s+(\w+)/);
    if (intoMatch) return intoMatch[1];
    const updateMatch = normalized.match(/update\s+(\w+)/);
    if (updateMatch) return updateMatch[1];
    return 'unknown';
  }

  /**
   * Get slow queries
   */
  getSlowQueries(limit = 20): QueryRecord[] {
    return [...this.slowQueries]
      .sort((a, b) => b.duration - a.duration)
      .slice(0, limit);
  }

  /**
   * Get query statistics
   */
  getStats(): QueryStats[] {
    return Array.from(this.statsMap.values())
      .sort((a, b) => b.totalDuration - a.totalDuration);
  }

  /**
   * Get top queries by total time
   */
  getTopByTotalTime(limit = 10): QueryStats[] {
    return this.getStats()
      .sort((a, b) => b.totalDuration - a.totalDuration)
      .slice(0, limit);
  }

  /**
   * Get top queries by average time
   */
  getTopByAvgTime(limit = 10): QueryStats[] {
    return this.getStats()
      .filter(s => s.totalExecutions >= 5) // Need enough samples
      .sort((a, b) => b.avgDuration - a.avgDuration)
      .slice(0, limit);
  }

  /**
   * Get top queries by execution count
   */
  getTopByExecutions(limit = 10): QueryStats[] {
    return this.getStats()
      .sort((a, b) => b.totalExecutions - a.totalExecutions)
      .slice(0, limit);
  }

  /**
   * Get queries with errors
   */
  getQueriesWithErrors(): QueryStats[] {
    return this.getStats().filter(s => s.errorCount > 0);
  }

  /**
   * Get summary report
   */
  getSummary(): {
    totalQueries: number;
    totalDuration: number;
    avgDuration: number;
    slowQueries: number;
    errorCount: number;
    uniqueQueries: number;
  } {
    const stats = this.getStats();
    const totalQueries = stats.reduce((sum, s) => sum + s.totalExecutions, 0);
    const totalDuration = stats.reduce((sum, s) => sum + s.totalDuration, 0);
    const errorCount = stats.reduce((sum, s) => sum + s.errorCount, 0);

    return {
      totalQueries,
      totalDuration,
      avgDuration: totalQueries > 0 ? totalDuration / totalQueries : 0,
      slowQueries: this.slowQueries.length,
      errorCount,
      uniqueQueries: stats.length,
    };
  }

  /**
   * Clear all records
   */
  clear(): void {
    this.records = [];
    this.statsMap.clear();
    this.slowQueries = [];
  }
}

/**
 * Global monitor instance
 */
let monitor: QueryMonitor | null = null;

export function getQueryMonitor(): QueryMonitor {
  if (!monitor) {
    monitor = new QueryMonitor({
      slowQueryThreshold: parseInt(process.env.SLOW_QUERY_THRESHOLD || '100', 10),
      logSlowQueries: process.env.NODE_ENV !== 'test',
    });
  }
  return monitor;
}

export function resetQueryMonitor(): void {
  monitor?.clear();
  monitor = null;
}

/**
 * Wrap pool to monitor queries
 */
export function createMonitoredPool(pool: Pool): Pool {
  const queryMonitor = getQueryMonitor();
  const originalQuery = pool.query.bind(pool);

  // Wrap the query method
  const wrappedQuery = async function (
    this: Pool,
    textOrConfig: string | QueryConfig,
    values?: unknown[]
  ): Promise<QueryResult> {
    const start = process.hrtime.bigint();
    const query = typeof textOrConfig === 'string' ? textOrConfig : textOrConfig.text;
    const params = typeof textOrConfig === 'string' ? values : textOrConfig.values;

    // Capture stack trace for slow query debugging
    const stack = new Error().stack;

    try {
      const result = await originalQuery(textOrConfig as any, values);

      const end = process.hrtime.bigint();
      const duration = Number(end - start) / 1e6; // Convert to milliseconds

      queryMonitor.record({
        query,
        params,
        duration,
        timestamp: new Date(),
        rowCount: result.rowCount || 0,
        cached: false,
        stack,
      });

      return result;
    } catch (error) {
      const end = process.hrtime.bigint();
      const duration = Number(end - start) / 1e6;

      queryMonitor.record({
        query,
        params,
        duration,
        timestamp: new Date(),
        rowCount: 0,
        cached: false,
        error: (error as Error).message,
        stack,
      });

      throw error;
    }
  };

  // Override query method
  (pool as any).query = wrappedQuery;

  return pool;
}

/**
 * Create API endpoint for query stats
 */
export function createQueryStatsHandler() {
  return (req: any, res: any) => {
    const monitor = getQueryMonitor();

    res.json({
      summary: monitor.getSummary(),
      topByTotalTime: monitor.getTopByTotalTime(),
      topByAvgTime: monitor.getTopByAvgTime(),
      topByExecutions: monitor.getTopByExecutions(),
      slowQueries: monitor.getSlowQueries().map(q => ({
        query: q.query.slice(0, 200),
        duration: q.duration,
        timestamp: q.timestamp,
        rowCount: q.rowCount,
      })),
      queriesWithErrors: monitor.getQueriesWithErrors(),
    });
  };
}

/**
 * EXPLAIN ANALYZE helper
 */
export async function explainQuery(
  pool: Pool,
  query: string,
  params?: unknown[]
): Promise<{
  plan: string[];
  executionTime: number;
  planningTime: number;
}> {
  const explainQuery = `EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) ${query}`;

  const result = await pool.query(explainQuery, params);
  const plan = result.rows.map((row: any) => row['QUERY PLAN']);

  // Extract timing from plan
  let executionTime = 0;
  let planningTime = 0;

  for (const line of plan) {
    const execMatch = line.match(/Execution Time: ([\d.]+) ms/);
    if (execMatch) executionTime = parseFloat(execMatch[1]);

    const planMatch = line.match(/Planning Time: ([\d.]+) ms/);
    if (planMatch) planningTime = parseFloat(planMatch[1]);
  }

  return { plan, executionTime, planningTime };
}

/**
 * Suggest indexes based on query patterns
 */
export function suggestIndexes(stats: QueryStats[]): string[] {
  const suggestions: string[] = [];

  for (const stat of stats) {
    if (stat.avgDuration > 50 && stat.totalExecutions > 100) {
      // Check for WHERE clause patterns
      const whereMatch = stat.query.match(/where\s+(\w+)\s*=/i);
      if (whereMatch) {
        const column = whereMatch[1];
        const table = stat.query.match(/from\s+(\w+)/i)?.[1] || 'unknown';
        suggestions.push(`Consider index on ${table}(${column}) - avg ${stat.avgDuration.toFixed(0)}ms, ${stat.totalExecutions} executions`);
      }

      // Check for JOIN patterns
      const joinMatch = stat.query.match(/join\s+(\w+)\s+on\s+\w+\.(\w+)/gi);
      if (joinMatch) {
        suggestions.push(`Review JOIN performance in: ${stat.query.slice(0, 100)}`);
      }
    }
  }

  return [...new Set(suggestions)]; // Deduplicate
}

export default {
  getQueryMonitor,
  resetQueryMonitor,
  createMonitoredPool,
  createQueryStatsHandler,
  explainQuery,
  suggestIndexes,
};
