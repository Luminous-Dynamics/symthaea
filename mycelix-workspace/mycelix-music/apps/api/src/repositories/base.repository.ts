// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Base Repository Pattern Implementation
 *
 * Provides a clean abstraction over data access with:
 * - Type-safe query building
 * - Consistent error handling
 * - Transaction support
 * - Query logging for debugging
 *
 * @example
 * class SongRepository extends BaseRepository<Song> {
 *   constructor(pool: Pool) {
 *     super(pool, 'songs');
 *   }
 * }
 */

import { Pool, PoolClient, QueryResult } from 'pg';

export interface QueryOptions {
  client?: PoolClient;
  timeout?: number;
}

export interface PaginationOptions {
  limit: number;
  offset: number;
  cursor?: string;
}

export interface SortOptions {
  field: string;
  direction: 'ASC' | 'DESC';
}

export interface FilterOptions {
  [key: string]: unknown;
}

export interface PaginatedResult<T> {
  data: T[];
  total: number;
  limit: number;
  offset: number;
  hasMore: boolean;
  nextCursor?: string;
}

export abstract class BaseRepository<T extends Record<string, unknown>> {
  protected readonly pool: Pool;
  protected readonly tableName: string;
  protected readonly primaryKey: string;

  constructor(pool: Pool, tableName: string, primaryKey = 'id') {
    this.pool = pool;
    this.tableName = tableName;
    this.primaryKey = primaryKey;
  }

  /**
   * Execute a raw query with parameterized values
   */
  protected async query<R = T>(
    sql: string,
    params: unknown[] = [],
    options: QueryOptions = {}
  ): Promise<QueryResult<R>> {
    const client = options.client || this.pool;
    const startTime = Date.now();

    try {
      const result = await client.query<R>(sql, params);

      // Log slow queries in development
      const duration = Date.now() - startTime;
      if (process.env.NODE_ENV === 'development' && duration > 100) {
        console.warn(`[slow-query] ${duration}ms: ${sql.slice(0, 100)}...`);
      }

      return result;
    } catch (error) {
      // Enhance error with query context
      const enhancedError = new RepositoryError(
        `Query failed: ${(error as Error).message}`,
        { sql: sql.slice(0, 200), table: this.tableName },
        error as Error
      );
      throw enhancedError;
    }
  }

  /**
   * Find a single record by primary key
   */
  async findById(id: string | number, options: QueryOptions = {}): Promise<T | null> {
    const sql = `SELECT * FROM ${this.tableName} WHERE ${this.primaryKey} = $1 LIMIT 1`;
    const result = await this.query<T>(sql, [id], options);
    return result.rows[0] || null;
  }

  /**
   * Find multiple records by IDs
   */
  async findByIds(ids: (string | number)[], options: QueryOptions = {}): Promise<T[]> {
    if (ids.length === 0) return [];

    const placeholders = ids.map((_, i) => `$${i + 1}`).join(', ');
    const sql = `SELECT * FROM ${this.tableName} WHERE ${this.primaryKey} IN (${placeholders})`;
    const result = await this.query<T>(sql, ids, options);
    return result.rows;
  }

  /**
   * Find all records with optional filtering, sorting, and pagination
   */
  async findAll(
    filters: FilterOptions = {},
    sort: SortOptions = { field: 'created_at', direction: 'DESC' },
    pagination: PaginationOptions = { limit: 50, offset: 0 }
  ): Promise<PaginatedResult<T>> {
    const { whereClause, params } = this.buildWhereClause(filters);
    const sortField = this.sanitizeFieldName(sort.field);
    const sortDir = sort.direction === 'ASC' ? 'ASC' : 'DESC';

    // Get total count
    const countSql = `SELECT COUNT(*) FROM ${this.tableName} ${whereClause}`;
    const countResult = await this.query<{ count: string }>(countSql, params);
    const total = parseInt(countResult.rows[0].count, 10);

    // Get paginated data
    const dataSql = `
      SELECT * FROM ${this.tableName}
      ${whereClause}
      ORDER BY ${sortField} ${sortDir}, ${this.primaryKey} ${sortDir}
      LIMIT ${pagination.limit}
      OFFSET ${pagination.offset}
    `;
    const dataResult = await this.query<T>(dataSql, params);

    return {
      data: dataResult.rows,
      total,
      limit: pagination.limit,
      offset: pagination.offset,
      hasMore: pagination.offset + dataResult.rows.length < total,
      nextCursor: this.generateCursor(dataResult.rows, sort.field),
    };
  }

  /**
   * Find one record matching filters
   */
  async findOne(filters: FilterOptions, options: QueryOptions = {}): Promise<T | null> {
    const { whereClause, params } = this.buildWhereClause(filters);
    const sql = `SELECT * FROM ${this.tableName} ${whereClause} LIMIT 1`;
    const result = await this.query<T>(sql, params, options);
    return result.rows[0] || null;
  }

  /**
   * Create a new record
   */
  async create(data: Partial<T>, options: QueryOptions = {}): Promise<T> {
    const entries = Object.entries(data).filter(([_, v]) => v !== undefined);
    const fields = entries.map(([k]) => this.sanitizeFieldName(k));
    const values = entries.map(([_, v]) => v);
    const placeholders = values.map((_, i) => `$${i + 1}`);

    const sql = `
      INSERT INTO ${this.tableName} (${fields.join(', ')})
      VALUES (${placeholders.join(', ')})
      RETURNING *
    `;

    const result = await this.query<T>(sql, values, options);
    return result.rows[0];
  }

  /**
   * Update a record by primary key
   */
  async update(id: string | number, data: Partial<T>, options: QueryOptions = {}): Promise<T | null> {
    const entries = Object.entries(data).filter(([_, v]) => v !== undefined);
    if (entries.length === 0) {
      return this.findById(id, options);
    }

    const setClauses = entries.map(([k], i) => `${this.sanitizeFieldName(k)} = $${i + 1}`);
    const values = [...entries.map(([_, v]) => v), id];

    const sql = `
      UPDATE ${this.tableName}
      SET ${setClauses.join(', ')}
      WHERE ${this.primaryKey} = $${values.length}
      RETURNING *
    `;

    const result = await this.query<T>(sql, values, options);
    return result.rows[0] || null;
  }

  /**
   * Upsert (insert or update) a record
   */
  async upsert(
    data: Partial<T>,
    conflictFields: string[],
    options: QueryOptions = {}
  ): Promise<T> {
    const entries = Object.entries(data).filter(([_, v]) => v !== undefined);
    const fields = entries.map(([k]) => this.sanitizeFieldName(k));
    const values = entries.map(([_, v]) => v);
    const placeholders = values.map((_, i) => `$${i + 1}`);

    const conflictClause = conflictFields.map(f => this.sanitizeFieldName(f)).join(', ');
    const updateClauses = fields
      .filter(f => !conflictFields.includes(f))
      .map(f => `${f} = EXCLUDED.${f}`);

    const sql = `
      INSERT INTO ${this.tableName} (${fields.join(', ')})
      VALUES (${placeholders.join(', ')})
      ON CONFLICT (${conflictClause}) DO UPDATE SET
        ${updateClauses.join(', ')}
      RETURNING *
    `;

    const result = await this.query<T>(sql, values, options);
    return result.rows[0];
  }

  /**
   * Delete a record by primary key
   */
  async delete(id: string | number, options: QueryOptions = {}): Promise<boolean> {
    const sql = `DELETE FROM ${this.tableName} WHERE ${this.primaryKey} = $1`;
    const result = await this.query(sql, [id], options);
    return result.rowCount !== null && result.rowCount > 0;
  }

  /**
   * Count records matching filters
   */
  async count(filters: FilterOptions = {}, options: QueryOptions = {}): Promise<number> {
    const { whereClause, params } = this.buildWhereClause(filters);
    const sql = `SELECT COUNT(*) FROM ${this.tableName} ${whereClause}`;
    const result = await this.query<{ count: string }>(sql, params, options);
    return parseInt(result.rows[0].count, 10);
  }

  /**
   * Check if a record exists
   */
  async exists(filters: FilterOptions, options: QueryOptions = {}): Promise<boolean> {
    const { whereClause, params } = this.buildWhereClause(filters);
    const sql = `SELECT 1 FROM ${this.tableName} ${whereClause} LIMIT 1`;
    const result = await this.query(sql, params, options);
    return result.rows.length > 0;
  }

  /**
   * Execute operations in a transaction
   */
  async transaction<R>(
    callback: (client: PoolClient) => Promise<R>
  ): Promise<R> {
    const client = await this.pool.connect();

    try {
      await client.query('BEGIN');
      const result = await callback(client);
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
   * Build WHERE clause from filters
   */
  protected buildWhereClause(filters: FilterOptions): {
    whereClause: string;
    params: unknown[];
  } {
    const conditions: string[] = [];
    const params: unknown[] = [];

    for (const [key, value] of Object.entries(filters)) {
      if (value === undefined || value === null) continue;

      const field = this.sanitizeFieldName(key);
      params.push(value);

      if (typeof value === 'string' && value.includes('%')) {
        conditions.push(`${field} ILIKE $${params.length}`);
      } else if (Array.isArray(value)) {
        const placeholders = value.map((_, i) => `$${params.length - 1 + i + 1}`);
        params.pop();
        params.push(...value);
        conditions.push(`${field} IN (${placeholders.join(', ')})`);
      } else {
        conditions.push(`${field} = $${params.length}`);
      }
    }

    const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(' AND ')}` : '';
    return { whereClause, params };
  }

  /**
   * Sanitize field name to prevent SQL injection
   */
  protected sanitizeFieldName(field: string): string {
    // Only allow alphanumeric and underscore
    if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(field)) {
      throw new Error(`Invalid field name: ${field}`);
    }
    return field;
  }

  /**
   * Generate cursor for pagination
   */
  protected generateCursor(rows: T[], sortField: string): string | undefined {
    if (rows.length === 0) return undefined;

    const lastRow = rows[rows.length - 1];
    const value = lastRow[sortField];
    const id = lastRow[this.primaryKey];

    if (value === undefined || id === undefined) return undefined;

    return Buffer.from(JSON.stringify({ value, id })).toString('base64');
  }

  /**
   * Parse cursor for pagination
   */
  protected parseCursor(cursor: string): { value: unknown; id: unknown } | null {
    try {
      const decoded = Buffer.from(cursor, 'base64').toString('utf8');
      return JSON.parse(decoded);
    } catch {
      return null;
    }
  }
}

/**
 * Custom error class for repository operations
 */
export class RepositoryError extends Error {
  public readonly context: Record<string, unknown>;
  public readonly cause?: Error;

  constructor(message: string, context: Record<string, unknown> = {}, cause?: Error) {
    super(message);
    this.name = 'RepositoryError';
    this.context = context;
    this.cause = cause;

    // Capture stack trace
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, RepositoryError);
    }
  }

  toJSON() {
    return {
      name: this.name,
      message: this.message,
      context: this.context,
      stack: this.stack,
    };
  }
}
