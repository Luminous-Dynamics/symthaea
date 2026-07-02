// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Migrations Index
 *
 * Registers all migrations with the migrator.
 */

import { Pool } from 'pg';
import Migrator from './migrator';
import initialSchema from './001_initial_schema';
import addAnalyticsViews from './002_add_analytics_views';

// All migrations in order
const migrations = [
  initialSchema,
  addAnalyticsViews,
];

/**
 * Create and configure migrator with all migrations
 */
export function createMigrator(pool: Pool): Migrator {
  const migrator = new Migrator(pool);
  migrator.addMigrations(migrations);
  return migrator;
}

/**
 * CLI entry point
 */
async function main(): Promise<void> {
  const command = process.argv[2];
  const connectionString = process.env.DATABASE_URL || 'postgresql://localhost:5432/mycelix';

  const pool = new Pool({ connectionString });
  const migrator = createMigrator(pool);

  try {
    switch (command) {
      case 'up':
        console.log('Running pending migrations...');
        const upResults = await migrator.migrateUp();
        if (upResults.length === 0) {
          console.log('No pending migrations');
        } else {
          console.log(`\nCompleted ${upResults.filter((r) => r.success).length} migrations`);
        }
        break;

      case 'down':
        console.log('Rolling back last migration...');
        const downResult = await migrator.migrateDown();
        if (!downResult) {
          console.log('No migrations to rollback');
        }
        break;

      case 'status':
        const status = await migrator.status();
        console.log('\nMigration Status:');
        console.log(`  Current version: ${status.current}`);
        console.log(`  Pending migrations: ${status.pending}`);
        if (status.applied.length > 0) {
          console.log('\nApplied migrations:');
          for (const m of status.applied) {
            console.log(`  ${m.version}: ${m.name} (${m.applied_at.toISOString()})`);
          }
        }
        if (status.pendingMigrations.length > 0) {
          console.log('\nPending migrations:');
          for (const m of status.pendingMigrations) {
            console.log(`  ${m.version}: ${m.name}`);
          }
        }
        break;

      case 'to':
        const version = parseInt(process.argv[3], 10);
        if (isNaN(version)) {
          console.error('Usage: migrate to <version>');
          process.exit(1);
        }
        console.log(`Migrating to version ${version}...`);
        await migrator.migrateTo(version);
        break;

      default:
        console.log('Usage: migrate <command>');
        console.log('Commands:');
        console.log('  up      - Run all pending migrations');
        console.log('  down    - Rollback the last migration');
        console.log('  status  - Show migration status');
        console.log('  to <n>  - Migrate to specific version');
        break;
    }
  } catch (error) {
    console.error('Migration error:', error);
    process.exit(1);
  } finally {
    await pool.end();
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

export { Migrator, migrations };
