// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Database Migration System
 *
 * Simple, reliable migration system with:
 * - Version tracking in database
 * - Up/down migrations
 * - Transaction support per migration
 * - CLI interface
 */

import { Pool, PoolClient } from 'pg';
import * as fs from 'fs';
import * as path from 'path';

/**
 * Migration interface
 */
export interface Migration {
  version: number;
  name: string;
  up: (client: PoolClient) => Promise<void>;
  down: (client: PoolClient) => Promise<void>;
}

/**
 * Migration record from database
 */
interface MigrationRecord {
  version: number;
  name: string;
  applied_at: Date;
}

/**
 * Migration result
 */
export interface MigrationResult {
  version: number;
  name: string;
  direction: 'up' | 'down';
  duration: number;
  success: boolean;
  error?: string;
}

/**
 * Migrator class
 */
export class Migrator {
  private pool: Pool;
  private migrations: Migration[] = [];
  private tableName = 'schema_migrations';

  constructor(pool: Pool) {
    this.pool = pool;
  }

  /**
   * Register a migration
   */
  addMigration(migration: Migration): void {
    this.migrations.push(migration);
    // Keep sorted by version
    this.migrations.sort((a, b) => a.version - b.version);
  }

  /**
   * Register multiple migrations
   */
  addMigrations(migrations: Migration[]): void {
    for (const migration of migrations) {
      this.addMigration(migration);
    }
  }

  /**
   * Ensure migrations table exists
   */
  async ensureMigrationsTable(): Promise<void> {
    await this.pool.query(`
      CREATE TABLE IF NOT EXISTS ${this.tableName} (
        version INTEGER PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
      )
    `);
  }

  /**
   * Get applied migrations
   */
  async getAppliedMigrations(): Promise<MigrationRecord[]> {
    await this.ensureMigrationsTable();

    const result = await this.pool.query<MigrationRecord>(
      `SELECT version, name, applied_at FROM ${this.tableName} ORDER BY version`
    );

    return result.rows;
  }

  /**
   * Get pending migrations
   */
  async getPendingMigrations(): Promise<Migration[]> {
    const applied = await this.getAppliedMigrations();
    const appliedVersions = new Set(applied.map((m) => m.version));

    return this.migrations.filter((m) => !appliedVersions.has(m.version));
  }

  /**
   * Get current version
   */
  async getCurrentVersion(): Promise<number> {
    const applied = await this.getAppliedMigrations();
    return applied.length > 0 ? applied[applied.length - 1].version : 0;
  }

  /**
   * Run a single migration up
   */
  async runMigrationUp(migration: Migration): Promise<MigrationResult> {
    const startTime = Date.now();
    const client = await this.pool.connect();

    try {
      await client.query('BEGIN');

      // Run migration
      await migration.up(client);

      // Record migration
      await client.query(
        `INSERT INTO ${this.tableName} (version, name) VALUES ($1, $2)`,
        [migration.version, migration.name]
      );

      await client.query('COMMIT');

      return {
        version: migration.version,
        name: migration.name,
        direction: 'up',
        duration: Date.now() - startTime,
        success: true,
      };
    } catch (error) {
      await client.query('ROLLBACK');

      return {
        version: migration.version,
        name: migration.name,
        direction: 'up',
        duration: Date.now() - startTime,
        success: false,
        error: (error as Error).message,
      };
    } finally {
      client.release();
    }
  }

  /**
   * Run a single migration down
   */
  async runMigrationDown(migration: Migration): Promise<MigrationResult> {
    const startTime = Date.now();
    const client = await this.pool.connect();

    try {
      await client.query('BEGIN');

      // Run migration
      await migration.down(client);

      // Remove migration record
      await client.query(
        `DELETE FROM ${this.tableName} WHERE version = $1`,
        [migration.version]
      );

      await client.query('COMMIT');

      return {
        version: migration.version,
        name: migration.name,
        direction: 'down',
        duration: Date.now() - startTime,
        success: true,
      };
    } catch (error) {
      await client.query('ROLLBACK');

      return {
        version: migration.version,
        name: migration.name,
        direction: 'down',
        duration: Date.now() - startTime,
        success: false,
        error: (error as Error).message,
      };
    } finally {
      client.release();
    }
  }

  /**
   * Run all pending migrations
   */
  async migrateUp(): Promise<MigrationResult[]> {
    const pending = await this.getPendingMigrations();
    const results: MigrationResult[] = [];

    for (const migration of pending) {
      console.log(`Running migration ${migration.version}: ${migration.name}...`);
      const result = await this.runMigrationUp(migration);
      results.push(result);

      if (result.success) {
        console.log(`  ✓ Completed in ${result.duration}ms`);
      } else {
        console.error(`  ✗ Failed: ${result.error}`);
        break; // Stop on first failure
      }
    }

    return results;
  }

  /**
   * Rollback the last migration
   */
  async migrateDown(): Promise<MigrationResult | null> {
    const applied = await this.getAppliedMigrations();

    if (applied.length === 0) {
      console.log('No migrations to rollback');
      return null;
    }

    const lastApplied = applied[applied.length - 1];
    const migration = this.migrations.find((m) => m.version === lastApplied.version);

    if (!migration) {
      throw new Error(`Migration ${lastApplied.version} not found in registered migrations`);
    }

    console.log(`Rolling back migration ${migration.version}: ${migration.name}...`);
    const result = await this.runMigrationDown(migration);

    if (result.success) {
      console.log(`  ✓ Rolled back in ${result.duration}ms`);
    } else {
      console.error(`  ✗ Failed: ${result.error}`);
    }

    return result;
  }

  /**
   * Migrate to a specific version
   */
  async migrateTo(targetVersion: number): Promise<MigrationResult[]> {
    const currentVersion = await this.getCurrentVersion();
    const results: MigrationResult[] = [];

    if (targetVersion > currentVersion) {
      // Migrate up
      const pending = await this.getPendingMigrations();
      const toRun = pending.filter((m) => m.version <= targetVersion);

      for (const migration of toRun) {
        console.log(`Running migration ${migration.version}: ${migration.name}...`);
        const result = await this.runMigrationUp(migration);
        results.push(result);

        if (!result.success) {
          break;
        }
      }
    } else if (targetVersion < currentVersion) {
      // Migrate down
      const applied = await this.getAppliedMigrations();
      const toRollback = applied
        .filter((m) => m.version > targetVersion)
        .reverse();

      for (const record of toRollback) {
        const migration = this.migrations.find((m) => m.version === record.version);
        if (!migration) {
          throw new Error(`Migration ${record.version} not found`);
        }

        console.log(`Rolling back migration ${migration.version}: ${migration.name}...`);
        const result = await this.runMigrationDown(migration);
        results.push(result);

        if (!result.success) {
          break;
        }
      }
    }

    return results;
  }

  /**
   * Get migration status
   */
  async status(): Promise<{
    current: number;
    pending: number;
    applied: MigrationRecord[];
    pendingMigrations: { version: number; name: string }[];
  }> {
    const applied = await this.getAppliedMigrations();
    const pending = await this.getPendingMigrations();

    return {
      current: applied.length > 0 ? applied[applied.length - 1].version : 0,
      pending: pending.length,
      applied,
      pendingMigrations: pending.map((m) => ({ version: m.version, name: m.name })),
    };
  }
}

/**
 * Create migration file helper
 */
export function createMigrationTemplate(name: string): string {
  const timestamp = Date.now();
  const version = parseInt(timestamp.toString().slice(-8), 10);

  return `/**
 * Migration: ${name}
 * Version: ${version}
 * Created: ${new Date().toISOString()}
 */

import { PoolClient } from 'pg';
import { Migration } from '../migrator';

const migration: Migration = {
  version: ${version},
  name: '${name}',

  async up(client: PoolClient): Promise<void> {
    // Add your migration SQL here
    await client.query(\`
      -- Your SQL here
    \`);
  },

  async down(client: PoolClient): Promise<void> {
    // Add your rollback SQL here
    await client.query(\`
      -- Your rollback SQL here
    \`);
  },
};

export default migration;
`;
}

export default Migrator;
