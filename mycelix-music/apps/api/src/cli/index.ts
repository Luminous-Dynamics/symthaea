#!/usr/bin/env node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix API CLI
 *
 * Command-line interface for database operations, migrations,
 * and administrative tasks.
 *
 * Usage:
 *   npx ts-node src/cli/index.ts <command> [options]
 *
 * Commands:
 *   migrate:up      - Run pending migrations
 *   migrate:down    - Rollback last migration
 *   migrate:status  - Show migration status
 *   db:seed         - Seed the database
 *   db:reset        - Reset database (drop all, migrate, seed)
 *   cache:clear     - Clear all caches
 *   jobs:run        - Run a specific job
 *   jobs:list       - List all jobs
 *   openapi:export  - Export OpenAPI spec
 */

import { Pool } from 'pg';
import { createMigrator } from '../db/migrations';
import { seed } from '../db/seeds/seed-data';
import { buildOpenAPISpec } from '../openapi';
import { getScheduler, registerAnalyticsJobs } from '../jobs';
import * as fs from 'fs';
import * as path from 'path';

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
};

function log(message: string, color = colors.reset): void {
  console.log(`${color}${message}${colors.reset}`);
}

function success(message: string): void {
  log(`✓ ${message}`, colors.green);
}

function error(message: string): void {
  log(`✗ ${message}`, colors.red);
}

function info(message: string): void {
  log(`ℹ ${message}`, colors.blue);
}

function warn(message: string): void {
  log(`⚠ ${message}`, colors.yellow);
}

function header(message: string): void {
  console.log();
  log(`${colors.bright}${message}${colors.reset}`);
  log('─'.repeat(50));
}

/**
 * Get database connection
 */
function getPool(): Pool {
  const connectionString = process.env.DATABASE_URL || 'postgresql://localhost:5432/mycelix';
  return new Pool({ connectionString });
}

/**
 * Migration commands
 */
async function migrateUp(): Promise<void> {
  header('Running Migrations');

  const pool = getPool();
  const migrator = createMigrator(pool);

  try {
    const results = await migrator.migrateUp();

    if (results.length === 0) {
      info('No pending migrations');
    } else {
      const successful = results.filter(r => r.success).length;
      const failed = results.filter(r => !r.success).length;

      if (failed === 0) {
        success(`Applied ${successful} migration(s)`);
      } else {
        error(`${failed} migration(s) failed`);
      }
    }
  } finally {
    await pool.end();
  }
}

async function migrateDown(): Promise<void> {
  header('Rolling Back Migration');

  const pool = getPool();
  const migrator = createMigrator(pool);

  try {
    const result = await migrator.migrateDown();

    if (!result) {
      info('No migrations to rollback');
    } else if (result.success) {
      success(`Rolled back: ${result.name}`);
    } else {
      error(`Rollback failed: ${result.error}`);
    }
  } finally {
    await pool.end();
  }
}

async function migrateStatus(): Promise<void> {
  header('Migration Status');

  const pool = getPool();
  const migrator = createMigrator(pool);

  try {
    const status = await migrator.status();

    info(`Current version: ${status.current}`);
    info(`Pending migrations: ${status.pending}`);

    if (status.applied.length > 0) {
      console.log('\nApplied migrations:');
      for (const m of status.applied) {
        log(`  ${colors.green}✓${colors.reset} ${m.version}: ${m.name}`);
      }
    }

    if (status.pendingMigrations.length > 0) {
      console.log('\nPending migrations:');
      for (const m of status.pendingMigrations) {
        log(`  ${colors.yellow}○${colors.reset} ${m.version}: ${m.name}`);
      }
    }
  } finally {
    await pool.end();
  }
}

/**
 * Database commands
 */
async function dbSeed(): Promise<void> {
  header('Seeding Database');

  const pool = getPool();

  try {
    await seed(pool);
    success('Database seeded successfully');
  } catch (err) {
    error(`Seeding failed: ${(err as Error).message}`);
    throw err;
  } finally {
    await pool.end();
  }
}

async function dbReset(): Promise<void> {
  header('Resetting Database');

  warn('This will delete all data!');

  const pool = getPool();
  const migrator = createMigrator(pool);

  try {
    // Rollback all migrations
    info('Rolling back all migrations...');
    let status = await migrator.status();
    while (status.applied.length > 0) {
      await migrator.migrateDown();
      status = await migrator.status();
    }

    // Run all migrations
    info('Running migrations...');
    await migrator.migrateUp();

    // Seed database
    info('Seeding database...');
    await seed(pool);

    success('Database reset complete');
  } finally {
    await pool.end();
  }
}

/**
 * Cache commands
 */
async function cacheClear(): Promise<void> {
  header('Clearing Cache');

  // TODO: Implement with Redis client
  info('Cache clearing not implemented yet');
  warn('If using Redis, run: redis-cli FLUSHDB');
}

/**
 * Job commands
 */
async function jobsList(): Promise<void> {
  header('Available Jobs');

  const pool = getPool();
  const scheduler = getScheduler();

  try {
    registerAnalyticsJobs(scheduler, pool);

    const statuses = scheduler.getAllStatuses();

    for (const status of statuses) {
      const enabled = status.enabled ? colors.green + '●' : colors.red + '○';
      log(`${enabled} ${status.name}${colors.reset}`);
    }
  } finally {
    await pool.end();
  }
}

async function jobsRun(jobName: string): Promise<void> {
  header(`Running Job: ${jobName}`);

  const pool = getPool();
  const scheduler = getScheduler();

  try {
    registerAnalyticsJobs(scheduler, pool);

    const result = await scheduler.runNow(jobName);

    if (result.success) {
      success(`Job completed in ${result.duration}ms`);
    } else {
      error(`Job failed: ${result.error}`);
    }
  } catch (err) {
    error(`Failed to run job: ${(err as Error).message}`);
  } finally {
    await pool.end();
  }
}

/**
 * OpenAPI commands
 */
async function openapiExport(format: 'json' | 'yaml' = 'json'): Promise<void> {
  header('Exporting OpenAPI Spec');

  const spec = buildOpenAPISpec();
  const filename = `openapi.${format}`;
  const outputPath = path.join(process.cwd(), filename);

  let content: string;
  if (format === 'json') {
    content = JSON.stringify(spec, null, 2);
  } else {
    // Basic YAML export
    content = JSON.stringify(spec, null, 2); // TODO: Proper YAML
  }

  fs.writeFileSync(outputPath, content);
  success(`Exported to ${outputPath}`);
}

/**
 * Help
 */
function showHelp(): void {
  header('Mycelix API CLI');

  console.log(`
${colors.bright}Usage:${colors.reset}
  npx ts-node src/cli/index.ts <command> [options]

${colors.bright}Migration Commands:${colors.reset}
  migrate:up        Run pending migrations
  migrate:down      Rollback last migration
  migrate:status    Show migration status

${colors.bright}Database Commands:${colors.reset}
  db:seed           Seed the database with test data
  db:reset          Reset database (drop, migrate, seed)

${colors.bright}Cache Commands:${colors.reset}
  cache:clear       Clear all caches

${colors.bright}Job Commands:${colors.reset}
  jobs:list         List all available jobs
  jobs:run <name>   Run a specific job

${colors.bright}OpenAPI Commands:${colors.reset}
  openapi:export    Export OpenAPI spec (--format=json|yaml)

${colors.bright}Environment Variables:${colors.reset}
  DATABASE_URL      PostgreSQL connection string
  REDIS_URL         Redis connection string (optional)
`);
}

/**
 * Main entry point
 */
async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const command = args[0];

  try {
    switch (command) {
      case 'migrate:up':
        await migrateUp();
        break;

      case 'migrate:down':
        await migrateDown();
        break;

      case 'migrate:status':
        await migrateStatus();
        break;

      case 'db:seed':
        await dbSeed();
        break;

      case 'db:reset':
        await dbReset();
        break;

      case 'cache:clear':
        await cacheClear();
        break;

      case 'jobs:list':
        await jobsList();
        break;

      case 'jobs:run':
        if (!args[1]) {
          error('Job name required');
          process.exit(1);
        }
        await jobsRun(args[1]);
        break;

      case 'openapi:export':
        const format = args.includes('--yaml') ? 'yaml' : 'json';
        await openapiExport(format);
        break;

      case 'help':
      case '--help':
      case '-h':
        showHelp();
        break;

      default:
        if (command) {
          error(`Unknown command: ${command}`);
        }
        showHelp();
        process.exit(command ? 1 : 0);
    }
  } catch (err) {
    error(`Command failed: ${(err as Error).message}`);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

export { main };
