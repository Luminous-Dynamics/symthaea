// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * API Usage Analytics
 *
 * Track and analyze API usage patterns, error rates,
 * and client behavior. Helps inform API evolution decisions.
 */

import { Request, Response, NextFunction } from 'express';
import { Pool } from 'pg';
import { getLogger } from '../logging';
import { getMetrics } from '../metrics';

const logger = getLogger();

/**
 * API usage record
 */
interface UsageRecord {
  timestamp: Date;
  method: string;
  path: string;
  normalizedPath: string;
  statusCode: number;
  duration: number;
  requestSize: number;
  responseSize: number;
  clientId?: string;
  walletAddress?: string;
  userAgent?: string;
  ip: string;
  apiVersion: string;
  error?: string;
}

/**
 * Aggregated statistics
 */
interface EndpointStats {
  path: string;
  method: string;
  totalRequests: number;
  successCount: number;
  errorCount: number;
  avgDuration: number;
  p95Duration: number;
  p99Duration: number;
  avgRequestSize: number;
  avgResponseSize: number;
  uniqueClients: number;
  lastAccessed: Date;
}

/**
 * Client statistics
 */
interface ClientStats {
  clientId: string;
  totalRequests: number;
  uniqueEndpoints: number;
  errorRate: number;
  avgDuration: number;
  lastAccessed: Date;
  topEndpoints: { path: string; count: number }[];
}

/**
 * Analytics configuration
 */
export interface AnalyticsConfig {
  /** Sample rate (0-1) */
  sampleRate: number;
  /** Batch size for writes */
  batchSize: number;
  /** Flush interval in ms */
  flushInterval: number;
  /** Retention period in days */
  retentionDays: number;
  /** Enable detailed logging */
  verbose: boolean;
}

/**
 * Default configuration
 */
const defaultConfig: AnalyticsConfig = {
  sampleRate: 1.0,
  batchSize: 100,
  flushInterval: 10000,
  retentionDays: 90,
  verbose: false,
};

/**
 * API Usage Analytics
 */
export class ApiUsageAnalytics {
  private config: AnalyticsConfig;
  private buffer: UsageRecord[] = [];
  private flushTimer?: NodeJS.Timeout;
  private endpointStats: Map<string, {
    durations: number[];
    requests: number;
    errors: number;
    clients: Set<string>;
  }> = new Map();

  constructor(
    private pool: Pool,
    config: Partial<AnalyticsConfig> = {}
  ) {
    this.config = { ...defaultConfig, ...config };

    // Start flush timer
    this.flushTimer = setInterval(
      () => this.flush(),
      this.config.flushInterval
    );

    // Register metrics
    const metrics = getMetrics();
    metrics.createCounter('api_analytics_records_total', 'Total analytics records', []);
    metrics.createGauge('api_analytics_buffer_size', 'Analytics buffer size', []);
  }

  /**
   * Record API usage
   */
  record(record: UsageRecord): void {
    // Sampling
    if (Math.random() > this.config.sampleRate) return;

    this.buffer.push(record);
    getMetrics().setGauge('api_analytics_buffer_size', this.buffer.length, {});

    // Update in-memory stats
    this.updateStats(record);

    // Flush if buffer is full
    if (this.buffer.length >= this.config.batchSize) {
      this.flush();
    }
  }

  /**
   * Update in-memory statistics
   */
  private updateStats(record: UsageRecord): void {
    const key = `${record.method}:${record.normalizedPath}`;
    let stats = this.endpointStats.get(key);

    if (!stats) {
      stats = {
        durations: [],
        requests: 0,
        errors: 0,
        clients: new Set(),
      };
      this.endpointStats.set(key, stats);
    }

    stats.durations.push(record.duration);
    stats.requests++;

    if (record.statusCode >= 400) {
      stats.errors++;
    }

    if (record.clientId) {
      stats.clients.add(record.clientId);
    }

    // Keep only last 1000 durations for percentile calculations
    if (stats.durations.length > 1000) {
      stats.durations = stats.durations.slice(-1000);
    }
  }

  /**
   * Flush buffer to database
   */
  async flush(): Promise<void> {
    if (this.buffer.length === 0) return;

    const records = [...this.buffer];
    this.buffer = [];

    getMetrics().setGauge('api_analytics_buffer_size', 0, {});

    try {
      // Batch insert
      const values = records.map((r, i) => {
        const base = i * 12;
        return `($${base + 1}, $${base + 2}, $${base + 3}, $${base + 4}, $${base + 5}, $${base + 6}, $${base + 7}, $${base + 8}, $${base + 9}, $${base + 10}, $${base + 11}, $${base + 12})`;
      }).join(',');

      const params = records.flatMap(r => [
        r.timestamp,
        r.method,
        r.normalizedPath,
        r.statusCode,
        r.duration,
        r.requestSize,
        r.responseSize,
        r.clientId,
        r.walletAddress,
        r.userAgent?.slice(0, 500),
        r.ip,
        r.apiVersion,
      ]);

      await this.pool.query(`
        INSERT INTO api_usage_logs
          (timestamp, method, path, status_code, duration_ms, request_size, response_size, client_id, wallet_address, user_agent, ip, api_version)
        VALUES ${values}
      `, params);

      getMetrics().incCounter('api_analytics_records_total', {});

      if (this.config.verbose) {
        logger.debug(`Flushed ${records.length} analytics records`);
      }
    } catch (error) {
      logger.error('Failed to flush analytics', error as Error);
      // Put records back in buffer
      this.buffer = [...records, ...this.buffer].slice(0, this.config.batchSize * 10);
    }
  }

  /**
   * Get endpoint statistics
   */
  async getEndpointStats(options: {
    dateFrom?: Date;
    dateTo?: Date;
    limit?: number;
  } = {}): Promise<EndpointStats[]> {
    const { dateFrom, dateTo, limit = 50 } = options;

    let query = `
      SELECT
        path,
        method,
        COUNT(*) as total_requests,
        COUNT(*) FILTER (WHERE status_code < 400) as success_count,
        COUNT(*) FILTER (WHERE status_code >= 400) as error_count,
        AVG(duration_ms) as avg_duration,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_duration,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) as p99_duration,
        AVG(request_size) as avg_request_size,
        AVG(response_size) as avg_response_size,
        COUNT(DISTINCT client_id) as unique_clients,
        MAX(timestamp) as last_accessed
      FROM api_usage_logs
      WHERE 1=1
    `;

    const params: unknown[] = [];

    if (dateFrom) {
      params.push(dateFrom);
      query += ` AND timestamp >= $${params.length}`;
    }
    if (dateTo) {
      params.push(dateTo);
      query += ` AND timestamp <= $${params.length}`;
    }

    query += `
      GROUP BY path, method
      ORDER BY total_requests DESC
      LIMIT $${params.length + 1}
    `;
    params.push(limit);

    const result = await this.pool.query(query, params);

    return result.rows.map(row => ({
      path: row.path,
      method: row.method,
      totalRequests: parseInt(row.total_requests),
      successCount: parseInt(row.success_count),
      errorCount: parseInt(row.error_count),
      avgDuration: parseFloat(row.avg_duration),
      p95Duration: parseFloat(row.p95_duration),
      p99Duration: parseFloat(row.p99_duration),
      avgRequestSize: parseFloat(row.avg_request_size),
      avgResponseSize: parseFloat(row.avg_response_size),
      uniqueClients: parseInt(row.unique_clients),
      lastAccessed: row.last_accessed,
    }));
  }

  /**
   * Get client statistics
   */
  async getClientStats(clientId: string): Promise<ClientStats | null> {
    const result = await this.pool.query(`
      SELECT
        client_id,
        COUNT(*) as total_requests,
        COUNT(DISTINCT path) as unique_endpoints,
        AVG(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_rate,
        AVG(duration_ms) as avg_duration,
        MAX(timestamp) as last_accessed
      FROM api_usage_logs
      WHERE client_id = $1
      GROUP BY client_id
    `, [clientId]);

    if (result.rows.length === 0) return null;

    const row = result.rows[0];

    // Get top endpoints
    const topEndpoints = await this.pool.query(`
      SELECT path, COUNT(*) as count
      FROM api_usage_logs
      WHERE client_id = $1
      GROUP BY path
      ORDER BY count DESC
      LIMIT 5
    `, [clientId]);

    return {
      clientId: row.client_id,
      totalRequests: parseInt(row.total_requests),
      uniqueEndpoints: parseInt(row.unique_endpoints),
      errorRate: parseFloat(row.error_rate),
      avgDuration: parseFloat(row.avg_duration),
      lastAccessed: row.last_accessed,
      topEndpoints: topEndpoints.rows.map(r => ({
        path: r.path,
        count: parseInt(r.count),
      })),
    };
  }

  /**
   * Get error analysis
   */
  async getErrorAnalysis(options: {
    dateFrom?: Date;
    dateTo?: Date;
  } = {}): Promise<{
    byStatusCode: { code: number; count: number }[];
    byEndpoint: { path: string; errorCount: number; errorRate: number }[];
    byClient: { clientId: string; errorCount: number }[];
  }> {
    const { dateFrom, dateTo } = options;

    const dateFilter = dateFrom
      ? `AND timestamp >= '${dateFrom.toISOString()}'`
      : '';

    const [byStatusCode, byEndpoint, byClient] = await Promise.all([
      this.pool.query(`
        SELECT status_code as code, COUNT(*) as count
        FROM api_usage_logs
        WHERE status_code >= 400 ${dateFilter}
        GROUP BY status_code
        ORDER BY count DESC
      `),
      this.pool.query(`
        SELECT
          path,
          COUNT(*) FILTER (WHERE status_code >= 400) as error_count,
          AVG(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_rate
        FROM api_usage_logs
        WHERE 1=1 ${dateFilter}
        GROUP BY path
        HAVING COUNT(*) FILTER (WHERE status_code >= 400) > 0
        ORDER BY error_count DESC
        LIMIT 20
      `),
      this.pool.query(`
        SELECT client_id, COUNT(*) as error_count
        FROM api_usage_logs
        WHERE status_code >= 400 ${dateFilter}
        AND client_id IS NOT NULL
        GROUP BY client_id
        ORDER BY error_count DESC
        LIMIT 20
      `),
    ]);

    return {
      byStatusCode: byStatusCode.rows.map(r => ({
        code: parseInt(r.code),
        count: parseInt(r.count),
      })),
      byEndpoint: byEndpoint.rows.map(r => ({
        path: r.path,
        errorCount: parseInt(r.error_count),
        errorRate: parseFloat(r.error_rate),
      })),
      byClient: byClient.rows.map(r => ({
        clientId: r.client_id,
        errorCount: parseInt(r.error_count),
      })),
    };
  }

  /**
   * Get usage trends
   */
  async getUsageTrends(options: {
    dateFrom?: Date;
    dateTo?: Date;
    granularity?: 'hour' | 'day' | 'week';
  } = {}): Promise<{
    date: Date;
    requests: number;
    uniqueClients: number;
    avgDuration: number;
    errorRate: number;
  }[]> {
    const { granularity = 'day' } = options;

    const truncate = granularity === 'hour' ? 'hour' :
                     granularity === 'week' ? 'week' : 'day';

    const result = await this.pool.query(`
      SELECT
        date_trunc('${truncate}', timestamp) as date,
        COUNT(*) as requests,
        COUNT(DISTINCT client_id) as unique_clients,
        AVG(duration_ms) as avg_duration,
        AVG(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_rate
      FROM api_usage_logs
      WHERE timestamp > NOW() - INTERVAL '30 days'
      GROUP BY date_trunc('${truncate}', timestamp)
      ORDER BY date DESC
    `);

    return result.rows.map(row => ({
      date: row.date,
      requests: parseInt(row.requests),
      uniqueClients: parseInt(row.unique_clients),
      avgDuration: parseFloat(row.avg_duration),
      errorRate: parseFloat(row.error_rate),
    }));
  }

  /**
   * Cleanup old records
   */
  async cleanup(): Promise<number> {
    const result = await this.pool.query(`
      DELETE FROM api_usage_logs
      WHERE timestamp < NOW() - INTERVAL '${this.config.retentionDays} days'
    `);

    logger.info(`Cleaned up ${result.rowCount} old analytics records`);
    return result.rowCount || 0;
  }

  /**
   * Shutdown
   */
  async shutdown(): Promise<void> {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }
    await this.flush();
  }
}

/**
 * Express middleware for API analytics
 */
export function apiAnalyticsMiddleware(analytics: ApiUsageAnalytics) {
  return (req: Request, res: Response, next: NextFunction): void => {
    const startTime = Date.now();
    const requestSize = parseInt(req.headers['content-length'] || '0');

    res.on('finish', () => {
      const duration = Date.now() - startTime;
      const responseSize = parseInt(res.getHeader('content-length') as string || '0');

      analytics.record({
        timestamp: new Date(),
        method: req.method,
        path: req.path,
        normalizedPath: normalizePath(req.path),
        statusCode: res.statusCode,
        duration,
        requestSize,
        responseSize,
        clientId: (req as any).clientId || req.headers['x-client-id'] as string,
        walletAddress: req.auth?.address,
        userAgent: req.headers['user-agent'],
        ip: req.ip || req.socket.remoteAddress || '',
        apiVersion: (req as any).apiVersion || '2',
        error: res.statusCode >= 400 ? (res as any).errorMessage : undefined,
      });
    });

    next();
  };
}

/**
 * Normalize path for grouping
 */
function normalizePath(path: string): string {
  return path
    .replace(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi, ':id')
    .replace(/0x[a-fA-F0-9]{40}/g, ':address')
    .replace(/\/\d+(?=\/|$)/g, '/:id');
}

/**
 * Migration SQL
 */
export const API_ANALYTICS_MIGRATION_SQL = `
CREATE TABLE IF NOT EXISTS api_usage_logs (
  id BIGSERIAL PRIMARY KEY,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  method VARCHAR(10) NOT NULL,
  path VARCHAR(500) NOT NULL,
  status_code INTEGER NOT NULL,
  duration_ms INTEGER NOT NULL,
  request_size INTEGER DEFAULT 0,
  response_size INTEGER DEFAULT 0,
  client_id VARCHAR(100),
  wallet_address VARCHAR(42),
  user_agent VARCHAR(500),
  ip VARCHAR(45),
  api_version VARCHAR(10)
);

CREATE INDEX idx_api_usage_timestamp ON api_usage_logs(timestamp);
CREATE INDEX idx_api_usage_path ON api_usage_logs(path);
CREATE INDEX idx_api_usage_client ON api_usage_logs(client_id) WHERE client_id IS NOT NULL;
CREATE INDEX idx_api_usage_status ON api_usage_logs(status_code) WHERE status_code >= 400;
`;

export default ApiUsageAnalytics;
