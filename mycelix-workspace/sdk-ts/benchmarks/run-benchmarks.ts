#!/usr/bin/env tsx
/**
 * Benchmark Runner with Baseline Tracking
 *
 * Run all benchmarks and compare against stored baselines.
 * Usage:
 *   npx tsx benchmarks/run-benchmarks.ts           # Run and compare
 *   npx tsx benchmarks/run-benchmarks.ts --save    # Save as new baseline
 *   npx tsx benchmarks/run-benchmarks.ts --check   # CI mode (exit 1 on regression)
 */

import { execSync, spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { BaselineTracker, createResultFromVitest } from './baseline-tracker.js';

// ============================================================================
// Configuration
// ============================================================================

const BENCHMARK_PATTERN = 'benchmarks/*.bench.ts';
const BASELINE_PATH = 'benchmarks/.baseline.json';
const OUTPUT_PATH = 'benchmarks/.bench-results.json';

// ============================================================================
// Types
// ============================================================================

interface VitestBenchResult {
  name: string;
  hz: number;
  min: number;
  max: number;
  mean: number;
  p75: number;
  p99: number;
  p995: number;
  p999: number;
  rme: number;
  samples: number;
  sd: number;
}

interface VitestBenchOutput {
  files: Array<{
    filepath: string;
    groups: Array<{
      fullName: string;
      benchmarks: VitestBenchResult[];
    }>;
  }>;
}

// ============================================================================
// CLI Arguments
// ============================================================================

const args = process.argv.slice(2);
const shouldSave = args.includes('--save') || args.includes('-s');
const checkMode = args.includes('--check') || args.includes('-c');
const verbose = args.includes('--verbose') || args.includes('-v');
const help = args.includes('--help') || args.includes('-h');

if (help) {
  console.log(`
Benchmark Runner with Baseline Tracking

Usage:
  npx tsx benchmarks/run-benchmarks.ts [options]

Options:
  --save, -s      Save current results as new baseline
  --check, -c     CI mode: exit with code 1 if regressions detected
  --verbose, -v   Show detailed output
  --help, -h      Show this help message

Examples:
  # Run benchmarks and compare against baseline
  npx tsx benchmarks/run-benchmarks.ts

  # Save new baseline (after improvements)
  npx tsx benchmarks/run-benchmarks.ts --save

  # CI check (fails on regression)
  npx tsx benchmarks/run-benchmarks.ts --check
`);
  process.exit(0);
}

// ============================================================================
// Main Runner
// ============================================================================

async function runBenchmarks(): Promise<void> {
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('                    MYCELIX SDK BENCHMARKS                      ');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('');

  // Initialize tracker
  const tracker = new BaselineTracker({ baselinePath: BASELINE_PATH });
  const baselineInfo = tracker.baselineInfo;

  if (baselineInfo.exists) {
    console.log(`📊 Baseline: ${baselineInfo.benchmarkCount} benchmarks from ${baselineInfo.createdAt?.toISOString()}`);
    if (baselineInfo.commitHash) {
      console.log(`   Commit: ${baselineInfo.commitHash}`);
    }
  } else {
    console.log('📊 No baseline found. Will create one after this run.');
  }
  console.log('');

  // Run vitest bench
  console.log('🏃 Running benchmarks...');
  console.log('');

  try {
    // Run vitest bench with JSON output
    const result = execSync(
      `npx vitest bench --run --reporter=json ${BENCHMARK_PATTERN}`,
      {
        encoding: 'utf-8',
        stdio: verbose ? 'inherit' : 'pipe',
        maxBuffer: 50 * 1024 * 1024, // 50MB buffer for large output
      }
    );

    // Parse results if not verbose mode
    if (!verbose && result) {
      parseAndRecord(result, tracker);
    }
  } catch (error: unknown) {
    // Vitest might exit with non-zero for bench output format
    // Try to parse stdout anyway
    const execError = error as { stdout?: string; stderr?: string };
    if (execError.stdout) {
      parseAndRecord(execError.stdout, tracker);
    } else {
      console.error('Failed to run benchmarks:', execError.stderr || error);
      process.exit(1);
    }
  }

  console.log('');
  console.log(`📈 Recorded ${tracker.resultCount} benchmark results`);
  console.log('');

  // Save baseline if requested
  if (shouldSave || !baselineInfo.exists) {
    tracker.saveBaseline();
    console.log('');
    console.log('✅ Baseline saved successfully!');
    return;
  }

  // Generate report
  const report = tracker.generateReport();
  console.log(report);

  // Exit with error in check mode if regressions found
  if (checkMode && tracker.hasRegressions()) {
    const regressions = tracker.getRegressions();
    console.error(`\n❌ CI Check Failed: ${regressions.length} regression(s) detected`);
    console.error('   Fix the regressions or update the baseline with --save');
    process.exit(1);
  }
}

/**
 * Parse vitest bench output and record results
 */
function parseAndRecord(output: string, tracker: BaselineTracker): void {
  // Try to find JSON in output
  const jsonMatch = output.match(/\{[\s\S]*"files"[\s\S]*\}/);
  if (!jsonMatch) {
    // Fallback: parse console table output
    parseConsoleOutput(output, tracker);
    return;
  }

  try {
    const data = JSON.parse(jsonMatch[0]) as VitestBenchOutput;

    for (const file of data.files) {
      for (const group of file.groups) {
        const suiteName = group.fullName;

        for (const bench of group.benchmarks) {
          const result = createResultFromVitest(
            suiteName,
            bench.name,
            // Reconstruct samples from stats
            generateSamplesFromStats(bench)
          );

          // Override with actual values from vitest
          result.meanMs = bench.mean;
          result.stdDevMs = bench.sd;
          result.opsPerSec = bench.hz;
          result.iterations = bench.samples;

          tracker.recordResult(result);
        }
      }
    }
  } catch (error) {
    console.warn('Failed to parse JSON output, falling back to console parsing');
    parseConsoleOutput(output, tracker);
  }
}

/**
 * Generate synthetic samples from stats (for percentile calculation)
 */
function generateSamplesFromStats(stats: VitestBenchResult): number[] {
  // Generate approximate samples based on mean and std dev
  const samples: number[] = [];
  const n = stats.samples;

  for (let i = 0; i < n; i++) {
    // Box-Muller transform for normal distribution
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    samples.push(stats.mean + z * stats.sd);
  }

  return samples;
}

/**
 * Parse console table output from vitest bench
 */
function parseConsoleOutput(output: string, tracker: BaselineTracker): void {
  const lines = output.split('\n');
  let currentSuite = 'Unknown';

  for (const line of lines) {
    // Detect suite name (usually in PASS/FAIL lines or describe blocks)
    const suiteMatch = line.match(/(?:PASS|FAIL|describe)\s*[│|]\s*(.+?)(?:\s*[│|]|$)/i);
    if (suiteMatch) {
      currentSuite = suiteMatch[1].trim();
      continue;
    }

    // Parse benchmark line (table format)
    // Format: · benchName    xxx ops/sec ±x.xx% (xx runs)
    const benchMatch = line.match(/[·•]\s*(.+?)\s+([0-9,.]+)\s*ops\/sec\s*±([0-9.]+)%\s*\((\d+)\s*(?:runs|samples)\)/);
    if (benchMatch) {
      const [, name, opsStr, rmeStr, runsStr] = benchMatch;
      const opsPerSec = parseFloat(opsStr.replace(/,/g, ''));
      const rme = parseFloat(rmeStr);
      const runs = parseInt(runsStr, 10);

      const meanMs = opsPerSec > 0 ? 1000 / opsPerSec : 0;
      const stdDevMs = meanMs * (rme / 100);

      tracker.recordResult({
        name: name.trim(),
        suite: currentSuite,
        meanMs,
        stdDevMs,
        opsPerSec,
        iterations: runs,
        p50Ms: meanMs,
        p95Ms: meanMs * 1.1,
        p99Ms: meanMs * 1.2,
        timestamp: Date.now(),
      });
    }
  }
}

// ============================================================================
// Run
// ============================================================================

runBenchmarks().catch((error) => {
  console.error('Benchmark runner failed:', error);
  process.exit(1);
});
