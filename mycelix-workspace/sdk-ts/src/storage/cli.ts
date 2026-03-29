#!/usr/bin/env node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * UESS Storage CLI
 *
 * Command-line interface for storage operations:
 *   mycelix-storage stats        - Show storage statistics
 *   mycelix-storage health       - Run health checks
 *   mycelix-storage export       - Export data to JSON
 *   mycelix-storage import       - Import data from JSON
 *   mycelix-storage migrate      - Migrate between backends
 *   mycelix-storage metrics      - Show Prometheus metrics
 *   mycelix-storage list         - List stored keys
 *   mycelix-storage get <key>    - Get a specific item
 */

import {
  createEpistemicStorage,
  type EpistemicStorageConfig,
  type EpistemicStorageImpl,
} from './epistemic-storage.js';
import { exportBackend, importBundle, type ExportBundle } from './migration.js';

// =============================================================================
// CLI Helpers
// =============================================================================

function printUsage(): void {
  console.log(`
UESS Storage CLI

Usage: mycelix-storage <command> [options]

Commands:
  stats                Show storage statistics
  health               Run health checks on all backends
  metrics              Show Prometheus-format metrics
  list [pattern]       List stored keys (optional glob pattern)
  get <key>            Retrieve and display a stored item
  export [file]        Export all data to JSON (stdout or file)
  import <file>        Import data from JSON file
  migrate <src> <dst>  Migrate data between backends (memory|local|dht|ipfs)
  sync <src> <dst>     Sync two backends (push mode)
  info <key>           Show storage info for a key
  verify <key>         Verify data integrity

Options:
  --agent <id>         Agent ID (default: cli-agent)
  --pattern <glob>     Key pattern filter for export/migrate
  --skip-existing      Skip keys that already exist in target (default: true)
  --delete-source      Delete from source after migration
  --help               Show this help message
`);
}

function parseArgs(args: string[]): { command: string; positional: string[]; flags: Record<string, string | boolean> } {
  const flags: Record<string, string | boolean> = {};
  const positional: string[] = [];
  let command = '';

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (!command && !arg.startsWith('--')) {
      command = arg;
    } else if (arg.startsWith('--')) {
      const key = arg.slice(2);
      const next = args[i + 1];
      if (next && !next.startsWith('--')) {
        flags[key] = next;
        i++;
      } else {
        flags[key] = true;
      }
    } else {
      positional.push(arg);
    }
  }

  return { command, positional, flags };
}

// =============================================================================
// CLI Commands
// =============================================================================

async function cmdStats(storage: EpistemicStorageImpl): Promise<void> {
  const stats = await storage.getStats();
  console.log('Storage Statistics');
  console.log('==================');
  console.log(`Total items:        ${stats.totalItems}`);
  console.log(`Total size:         ${formatBytes(stats.totalSizeBytes)}`);
  console.log(`Cache hit rate:     ${(stats.cacheHitRate * 100).toFixed(1)}%`);
  console.log(`Avg retrieval:      ${stats.avgRetrievalTimeMs.toFixed(2)}ms`);
  console.log('');
  console.log('Items by Backend:');
  for (const [backend, count] of Object.entries(stats.itemsByBackend)) {
    if (count > 0) console.log(`  ${backend.padEnd(10)} ${count}`);
  }
  console.log('');
  console.log('Items by Materiality:');
  const mLabels = ['M0 Ephemeral', 'M1 Temporal', 'M2 Persistent', 'M3 Immutable'];
  for (const [level, count] of Object.entries(stats.itemsByMateriality)) {
    if (count > 0) console.log(`  ${mLabels[Number(level)].padEnd(16)} ${count}`);
  }
}

async function cmdHealth(storage: EpistemicStorageImpl): Promise<void> {
  const results = await storage.runHealthChecks();
  if (results.length === 0) {
    console.log('No health checks configured. Enable with observability.healthCheckIntervalMs.');
    return;
  }
  console.log('Health Checks');
  console.log('=============');
  for (const result of results) {
    const icon = result.status === 'healthy' ? '[OK]' : result.status === 'degraded' ? '[!!]' : '[XX]';
    const latency = result.latencyMs ? ` (${result.latencyMs}ms)` : '';
    console.log(`${icon} ${result.name}${latency}${result.message ? ' - ' + result.message : ''}`);
  }
}

async function cmdMetrics(storage: EpistemicStorageImpl): Promise<void> {
  const prom = storage.getPrometheusMetrics();
  if (!prom) {
    console.log('Metrics not enabled. Enable with observability.enableMetrics.');
    return;
  }
  console.log(prom);
}

async function cmdList(storage: EpistemicStorageImpl, pattern?: string): Promise<void> {
  // Access internal backends to list keys
  const backends = [
    { name: 'memory', backend: storage['memoryBackend'] },
    { name: 'local', backend: storage['localBackend'] },
  ];

  for (const { name, backend } of backends) {
    const keys = await backend.keys(pattern);
    if (keys.length > 0) {
      console.log(`[${name}] ${keys.length} keys:`);
      for (const key of keys) {
        console.log(`  ${key}`);
      }
    }
  }
}

async function cmdGet(storage: EpistemicStorageImpl, key: string): Promise<void> {
  const data = await storage.retrieve(key);
  if (!data) {
    console.error(`Key not found: ${key}`);
    process.exitCode = 1;
    return;
  }
  console.log(JSON.stringify({
    key,
    data: data.data,
    metadata: {
      cid: data.metadata.cid,
      storedAt: data.metadata.storedAt,
      version: data.metadata.version,
      createdBy: data.metadata.createdBy,
      classification: data.metadata.classification,
      schema: data.metadata.schema,
    },
    verified: data.verified,
  }, null, 2));
}

async function cmdExport(storage: EpistemicStorageImpl, file?: string, pattern?: string): Promise<void> {
  const localBackend = storage['localBackend'];
  const bundle = await exportBackend(localBackend, {
    keyPattern: pattern ?? '*',
    agentId: storage['agentId'],
    onProgress: (p) => {
      if (file) process.stderr.write(`\rExporting ${p.completed}/${p.total}...`);
    },
  });

  const json = JSON.stringify(bundle, null, 2);
  if (file) {
    const fs = await import('fs');
    fs.writeFileSync(file, json);
    console.log(`\nExported ${bundle.itemCount} items to ${file}`);
  } else {
    console.log(json);
  }
}

async function cmdImport(storage: EpistemicStorageImpl, file: string, skipExisting: boolean): Promise<void> {
  const fs = await import('fs');
  const json = fs.readFileSync(file, 'utf-8');
  const bundle = JSON.parse(json) as ExportBundle;

  const localBackend = storage['localBackend'];
  const result = await importBundle(localBackend, bundle, {
    skipExisting,
    onProgress: (p) => process.stderr.write(`\rImporting ${p.completed}/${p.total}...`),
  });

  console.log(`\nImported: ${result.migrated}, Skipped: ${result.skipped}, Failed: ${result.failed}`);
}

async function cmdInfo(storage: EpistemicStorageImpl, key: string): Promise<void> {
  const info = await storage.getStorageInfo(key);
  if (!info || !info.exists) {
    console.error(`Key not found: ${key}`);
    process.exitCode = 1;
    return;
  }
  console.log(JSON.stringify(info, null, 2));
}

async function cmdVerify(storage: EpistemicStorageImpl, key: string): Promise<void> {
  const result = await storage.verify(key);
  const icon = result.verified ? '[OK]' : '[FAIL]';
  console.log(`${icon} ${key}`);
  console.log(`  CID valid:     ${result.cidValid}`);
  console.log(`  Replication:   ${result.actualReplication}/${result.expectedReplication}`);
  if (result.errors.length > 0) {
    console.log(`  Errors:        ${result.errors.join(', ')}`);
  }
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

// =============================================================================
// Main
// =============================================================================

export async function main(argv: string[] = process.argv.slice(2)): Promise<void> {
  const { command, positional, flags } = parseArgs(argv);

  if (!command || flags['help']) {
    printUsage();
    return;
  }

  const agentId = (flags['agent'] as string) ?? 'cli-agent';
  const config: EpistemicStorageConfig = {
    agentId,
    observability: {
      enableMetrics: true,
      enableTracing: true,
    },
  };

  const storage = createEpistemicStorage(config);

  try {
    switch (command) {
      case 'stats':
        await cmdStats(storage);
        break;
      case 'health':
        await cmdHealth(storage);
        break;
      case 'metrics':
        await cmdMetrics(storage);
        break;
      case 'list':
        await cmdList(storage, positional[0]);
        break;
      case 'get':
        if (!positional[0]) { console.error('Usage: get <key>'); process.exitCode = 1; break; }
        await cmdGet(storage, positional[0]);
        break;
      case 'export':
        await cmdExport(storage, positional[0], flags['pattern'] as string | undefined);
        break;
      case 'import':
        if (!positional[0]) { console.error('Usage: import <file>'); process.exitCode = 1; break; }
        await cmdImport(storage, positional[0], flags['skip-existing'] !== false);
        break;
      case 'info':
        if (!positional[0]) { console.error('Usage: info <key>'); process.exitCode = 1; break; }
        await cmdInfo(storage, positional[0]);
        break;
      case 'verify':
        if (!positional[0]) { console.error('Usage: verify <key>'); process.exitCode = 1; break; }
        await cmdVerify(storage, positional[0]);
        break;
      default:
        console.error(`Unknown command: ${command}`);
        printUsage();
        process.exitCode = 1;
    }
  } finally {
    storage.dispose();
  }
}

// Auto-run when executed directly
const isDirectExecution = typeof process !== 'undefined' &&
  process.argv[1] &&
  (process.argv[1].endsWith('/cli.js') || process.argv[1].endsWith('/cli.ts'));

if (isDirectExecution) {
  main().catch((err) => {
    console.error(err);
    process.exitCode = 1;
  });
}

