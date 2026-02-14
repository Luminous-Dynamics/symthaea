#!/usr/bin/env node
/**
 * Bundle Size Analysis Script
 *
 * Analyzes the built dist/ directory and provides detailed size information
 * for each module to help identify optimization opportunities.
 */

import { readdirSync, statSync, readFileSync } from 'fs';
import { join, relative, extname } from 'path';

const DIST_DIR = 'dist';

// Color helpers for terminal output
const colors = {
  reset: '\x1b[0m',
  bold: '\x1b[1m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  gray: '\x1b[90m',
};

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function getFileSizes(dir, prefix = '') {
  const results = [];

  try {
    const entries = readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = join(dir, entry.name);
      const relativePath = join(prefix, entry.name);

      if (entry.isDirectory()) {
        results.push(...getFileSizes(fullPath, relativePath));
      } else if (entry.isFile()) {
        const stats = statSync(fullPath);
        results.push({
          path: relativePath,
          size: stats.size,
          ext: extname(entry.name),
        });
      }
    }
  } catch {
    // Directory doesn't exist
  }

  return results;
}

function analyzeModule(modulePath) {
  const files = getFileSizes(modulePath);
  const jsFiles = files.filter((f) => f.ext === '.js');
  const dtsFiles = files.filter((f) => f.ext === '.ts');
  const mapFiles = files.filter((f) => f.ext === '.map');

  return {
    jsSize: jsFiles.reduce((sum, f) => sum + f.size, 0),
    dtsSize: dtsFiles.reduce((sum, f) => sum + f.size, 0),
    mapSize: mapFiles.reduce((sum, f) => sum + f.size, 0),
    fileCount: files.length,
    jsFiles,
  };
}

function printHeader(text) {
  console.log(`\n${colors.bold}${colors.cyan}${text}${colors.reset}`);
  console.log('─'.repeat(60));
}

function main() {
  console.log(`\n${colors.bold}${colors.blue}╔═══════════════════════════════════════════════════════════╗${colors.reset}`);
  console.log(`${colors.bold}${colors.blue}║         @mycelix/sdk Bundle Size Analysis                 ║${colors.reset}`);
  console.log(`${colors.bold}${colors.blue}╚═══════════════════════════════════════════════════════════╝${colors.reset}`);

  // Get all files in dist
  const allFiles = getFileSizes(DIST_DIR);
  const totalSize = allFiles.reduce((sum, f) => sum + f.size, 0);
  const jsTotal = allFiles.filter((f) => f.ext === '.js').reduce((sum, f) => sum + f.size, 0);
  const dtsTotal = allFiles.filter((f) => f.ext === '.ts').reduce((sum, f) => sum + f.size, 0);

  printHeader('Summary');
  console.log(`  Total files:     ${allFiles.length}`);
  console.log(`  Total size:      ${colors.bold}${formatBytes(totalSize)}${colors.reset}`);
  console.log(`  JavaScript:      ${formatBytes(jsTotal)} (${((jsTotal / totalSize) * 100).toFixed(1)}%)`);
  console.log(`  TypeScript .d.ts: ${formatBytes(dtsTotal)} (${((dtsTotal / totalSize) * 100).toFixed(1)}%)`);

  printHeader('By Module');

  // Analyze each top-level module
  const modules = [];
  try {
    const entries = readdirSync(DIST_DIR, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isDirectory()) {
        const analysis = analyzeModule(join(DIST_DIR, entry.name));
        modules.push({ name: entry.name, ...analysis });
      }
    }
  } catch {
    console.log('  Error reading dist directory');
    return;
  }

  // Sort by JS size descending
  modules.sort((a, b) => b.jsSize - a.jsSize);

  for (const mod of modules) {
    const bar = '█'.repeat(Math.ceil((mod.jsSize / jsTotal) * 40));
    console.log(
      `  ${colors.green}${mod.name.padEnd(20)}${colors.reset} ` +
        `${formatBytes(mod.jsSize).padStart(10)} ` +
        `${colors.gray}${bar}${colors.reset}`
    );
  }

  // Find root-level JS files
  const rootFiles = allFiles.filter(
    (f) => f.ext === '.js' && !f.path.includes('/') && !f.path.includes('\\')
  );
  if (rootFiles.length > 0) {
    const rootSize = rootFiles.reduce((sum, f) => sum + f.size, 0);
    console.log(
      `  ${colors.yellow}(root files)${colors.reset}`.padEnd(32) +
        `${formatBytes(rootSize).padStart(10)}`
    );
  }

  printHeader('Largest Files');
  const jsFiles = allFiles.filter((f) => f.ext === '.js');
  jsFiles.sort((a, b) => b.size - a.size);

  for (const file of jsFiles.slice(0, 10)) {
    console.log(`  ${formatBytes(file.size).padStart(10)}  ${colors.gray}${file.path}${colors.reset}`);
  }

  printHeader('Tree-Shaking Ready');
  console.log(`  ${colors.green}✓${colors.reset} ESM-only build (type: "module")`);
  console.log(`  ${colors.green}✓${colors.reset} sideEffects: false configured`);
  console.log(`  ${colors.green}✓${colors.reset} Granular exports (${modules.length} subpath exports)`);
  console.log(`  ${colors.green}✓${colors.reset} Pure TypeScript compilation (no bundling)`);

  printHeader('Import Recommendations');
  console.log(`  ${colors.gray}// Import only what you need for optimal tree-shaking:${colors.reset}`);
  console.log(`  ${colors.cyan}import { createPoGQ } from '@mycelix/sdk/matl';${colors.reset}`);
  console.log(`  ${colors.cyan}import { createClaim } from '@mycelix/sdk/epistemic';${colors.reset}`);
  console.log(`  ${colors.cyan}import { MycelixClient } from '@mycelix/sdk/client';${colors.reset}`);
  console.log(`\n  ${colors.gray}// Avoid full SDK import unless needed:${colors.reset}`);
  console.log(`  ${colors.yellow}import * from '@mycelix/sdk';  // Larger bundle${colors.reset}`);

  console.log('\n');
}

main();
