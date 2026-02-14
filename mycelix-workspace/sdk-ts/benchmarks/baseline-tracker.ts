/**
 * Benchmark Baseline Tracker
 *
 * Provides baseline storage, comparison, and regression detection for benchmarks.
 * Integrates with Vitest's bench API.
 */

import * as fs from 'fs';
import * as path from 'path';

// ============================================================================
// Types
// ============================================================================

/**
 * Single benchmark result
 */
export interface BenchmarkResult {
  /** Name of the benchmark */
  name: string;
  /** Suite/group name */
  suite: string;
  /** Mean execution time in ms */
  meanMs: number;
  /** Standard deviation in ms */
  stdDevMs: number;
  /** Operations per second */
  opsPerSec: number;
  /** Number of iterations */
  iterations: number;
  /** Percentile values */
  p50Ms: number;
  p95Ms: number;
  p99Ms: number;
  /** Timestamp of the run */
  timestamp: number;
  /** Git commit hash (if available) */
  commitHash?: string;
}

/**
 * Stored baseline
 */
export interface Baseline {
  /** Version of baseline format */
  version: string;
  /** When baseline was created */
  createdAt: number;
  /** Last updated */
  updatedAt: number;
  /** Git commit when baseline was set */
  commitHash?: string;
  /** Environment info */
  environment: {
    node: string;
    platform: string;
    arch: string;
    cpus: number;
  };
  /** Benchmark results */
  results: Record<string, BenchmarkResult>;
}

/**
 * Comparison result between current and baseline
 */
export interface ComparisonResult {
  /** Benchmark name */
  name: string;
  /** Suite name */
  suite: string;
  /** Current mean time */
  currentMs: number;
  /** Baseline mean time */
  baselineMs: number;
  /** Percent change (positive = slower) */
  percentChange: number;
  /** Whether this is a regression */
  isRegression: boolean;
  /** Whether this is an improvement */
  isImprovement: boolean;
  /** Status description */
  status: 'regression' | 'improvement' | 'stable';
}

/**
 * Regression report
 */
export interface RegressionReport {
  /** Total benchmarks compared */
  totalBenchmarks: number;
  /** Number of regressions */
  regressions: number;
  /** Number of improvements */
  improvements: number;
  /** Number of stable */
  stable: number;
  /** Detailed comparisons */
  comparisons: ComparisonResult[];
  /** Threshold used for detection */
  threshold: number;
  /** Report timestamp */
  timestamp: number;
}

// ============================================================================
// Configuration
// ============================================================================

/** Default regression threshold (10% slower = regression) */
export const DEFAULT_REGRESSION_THRESHOLD = 0.10;

/** Default improvement threshold (10% faster = improvement) */
export const DEFAULT_IMPROVEMENT_THRESHOLD = 0.10;

/** Baseline file path */
export const DEFAULT_BASELINE_PATH = 'benchmarks/.baseline.json';

// ============================================================================
// Baseline Storage
// ============================================================================

/**
 * Baseline Tracker class
 */
export class BaselineTracker {
  private baselinePath: string;
  private baseline: Baseline | null = null;
  private currentResults: Map<string, BenchmarkResult> = new Map();
  private regressionThreshold: number;
  private improvementThreshold: number;

  constructor(options: {
    baselinePath?: string;
    regressionThreshold?: number;
    improvementThreshold?: number;
  } = {}) {
    this.baselinePath = options.baselinePath ?? DEFAULT_BASELINE_PATH;
    this.regressionThreshold = options.regressionThreshold ?? DEFAULT_REGRESSION_THRESHOLD;
    this.improvementThreshold = options.improvementThreshold ?? DEFAULT_IMPROVEMENT_THRESHOLD;
    this.loadBaseline();
  }

  /**
   * Load baseline from file
   */
  private loadBaseline(): void {
    try {
      if (fs.existsSync(this.baselinePath)) {
        const data = fs.readFileSync(this.baselinePath, 'utf-8');
        this.baseline = JSON.parse(data);
      }
    } catch (error) {
      console.warn(`Failed to load baseline: ${error}`);
      this.baseline = null;
    }
  }

  /**
   * Save current results as baseline
   */
  saveBaseline(): void {
    const resultsObj: Record<string, BenchmarkResult> = {};
    this.currentResults.forEach((value, key) => {
      resultsObj[key] = value;
    });

    const baseline: Baseline = {
      version: '1.0.0',
      createdAt: this.baseline?.createdAt ?? Date.now(),
      updatedAt: Date.now(),
      commitHash: this.getGitCommit(),
      environment: this.getEnvironment(),
      results: resultsObj,
    };

    const dir = path.dirname(this.baselinePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    fs.writeFileSync(this.baselinePath, JSON.stringify(baseline, null, 2));
    this.baseline = baseline;
    console.log(`Baseline saved with ${this.currentResults.size} benchmarks`);
  }

  /**
   * Record a benchmark result
   */
  recordResult(result: BenchmarkResult): void {
    const key = `${result.suite}::${result.name}`;
    this.currentResults.set(key, result);
  }

  /**
   * Record results from Vitest bench output
   */
  recordVitestResults(benchFile: {
    name: string;
    groups: Array<{
      name: string;
      benchmarks: Array<{
        name: string;
        samples: number[];
      }>;
    }>;
  }): void {
    for (const group of benchFile.groups) {
      for (const bench of group.benchmarks) {
        const samples = bench.samples;
        if (samples.length === 0) continue;

        const sorted = [...samples].sort((a, b) => a - b);
        const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
        const variance = samples.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / samples.length;
        const stdDev = Math.sqrt(variance);

        const result: BenchmarkResult = {
          name: bench.name,
          suite: group.name,
          meanMs: mean,
          stdDevMs: stdDev,
          opsPerSec: mean > 0 ? 1000 / mean : 0,
          iterations: samples.length,
          p50Ms: sorted[Math.floor(sorted.length * 0.5)],
          p95Ms: sorted[Math.floor(sorted.length * 0.95)],
          p99Ms: sorted[Math.floor(sorted.length * 0.99)],
          timestamp: Date.now(),
          commitHash: this.getGitCommit(),
        };

        this.recordResult(result);
      }
    }
  }

  /**
   * Compare current results against baseline
   */
  compare(): RegressionReport {
    const comparisons: ComparisonResult[] = [];

    this.currentResults.forEach((current, key) => {
      const baseline = this.baseline?.results[key];

      if (!baseline) {
        // New benchmark, no baseline to compare
        return;
      }

      const percentChange = (current.meanMs - baseline.meanMs) / baseline.meanMs;
      const isRegression = percentChange > this.regressionThreshold;
      const isImprovement = percentChange < -this.improvementThreshold;

      comparisons.push({
        name: current.name,
        suite: current.suite,
        currentMs: current.meanMs,
        baselineMs: baseline.meanMs,
        percentChange,
        isRegression,
        isImprovement,
        status: isRegression ? 'regression' : isImprovement ? 'improvement' : 'stable',
      });
    });

    const regressions = comparisons.filter((c) => c.isRegression).length;
    const improvements = comparisons.filter((c) => c.isImprovement).length;
    const stable = comparisons.length - regressions - improvements;

    return {
      totalBenchmarks: comparisons.length,
      regressions,
      improvements,
      stable,
      comparisons,
      threshold: this.regressionThreshold,
      timestamp: Date.now(),
    };
  }

  /**
   * Generate human-readable report
   */
  generateReport(): string {
    const report = this.compare();
    const lines: string[] = [];

    lines.push('═══════════════════════════════════════════════════════════════');
    lines.push('                    BENCHMARK REGRESSION REPORT                 ');
    lines.push('═══════════════════════════════════════════════════════════════');
    lines.push('');
    lines.push(`Total Benchmarks: ${report.totalBenchmarks}`);
    lines.push(`Regressions:      ${report.regressions} (${this.formatPercent(report.regressions / report.totalBenchmarks)})`);
    lines.push(`Improvements:     ${report.improvements} (${this.formatPercent(report.improvements / report.totalBenchmarks)})`);
    lines.push(`Stable:           ${report.stable} (${this.formatPercent(report.stable / report.totalBenchmarks)})`);
    lines.push(`Threshold:        ±${this.formatPercent(report.threshold)}`);
    lines.push('');

    // Group by suite
    const bySuite = new Map<string, ComparisonResult[]>();
    for (const c of report.comparisons) {
      const existing = bySuite.get(c.suite) ?? [];
      existing.push(c);
      bySuite.set(c.suite, existing);
    }

    bySuite.forEach((comparisons, suite) => {
      lines.push(`─── ${suite} ───`);

      // Sort: regressions first, then improvements, then stable
      const sorted = [...comparisons].sort((a, b) => {
        if (a.isRegression !== b.isRegression) return a.isRegression ? -1 : 1;
        if (a.isImprovement !== b.isImprovement) return a.isImprovement ? -1 : 1;
        return Math.abs(b.percentChange) - Math.abs(a.percentChange);
      });

      for (const c of sorted) {
        const icon = c.isRegression ? '🔴' : c.isImprovement ? '🟢' : '⚪';
        const sign = c.percentChange > 0 ? '+' : '';
        const pct = this.formatPercent(c.percentChange);
        const time = `${c.currentMs.toFixed(2)}ms (was ${c.baselineMs.toFixed(2)}ms)`;
        lines.push(`  ${icon} ${c.name}`);
        lines.push(`     ${sign}${pct} | ${time}`);
      }
      lines.push('');
    });

    if (report.regressions > 0) {
      lines.push('⚠️  WARNING: Performance regressions detected!');
      lines.push('   Review the benchmarks above marked with 🔴');
    } else if (report.improvements > 0) {
      lines.push('✅ Great! Some benchmarks improved with no regressions.');
    } else {
      lines.push('✅ All benchmarks are stable.');
    }

    return lines.join('\n');
  }

  /**
   * Check if there are any regressions
   */
  hasRegressions(): boolean {
    return this.compare().regressions > 0;
  }

  /**
   * Get list of regressions
   */
  getRegressions(): ComparisonResult[] {
    return this.compare().comparisons.filter((c) => c.isRegression);
  }

  /**
   * Get environment info
   */
  private getEnvironment(): Baseline['environment'] {
    return {
      node: process.version,
      platform: process.platform,
      arch: process.arch,
      cpus: require('os').cpus().length,
    };
  }

  /**
   * Get git commit hash
   */
  private getGitCommit(): string | undefined {
    try {
      const { execSync } = require('child_process');
      return execSync('git rev-parse --short HEAD', { encoding: 'utf-8' }).trim();
    } catch {
      return undefined;
    }
  }

  /**
   * Format percentage
   */
  private formatPercent(value: number): string {
    return `${(value * 100).toFixed(1)}%`;
  }

  /**
   * Get current results count
   */
  get resultCount(): number {
    return this.currentResults.size;
  }

  /**
   * Get baseline info
   */
  get baselineInfo(): { exists: boolean; benchmarkCount: number; createdAt?: Date; commitHash?: string } {
    if (!this.baseline) {
      return { exists: false, benchmarkCount: 0 };
    }
    return {
      exists: true,
      benchmarkCount: Object.keys(this.baseline.results).length,
      createdAt: new Date(this.baseline.createdAt),
      commitHash: this.baseline.commitHash,
    };
  }
}

// ============================================================================
// Vitest Integration Helpers
// ============================================================================

/**
 * Create a benchmark result from Vitest task result
 */
export function createResultFromVitest(
  suite: string,
  name: string,
  samples: number[]
): BenchmarkResult {
  if (samples.length === 0) {
    return {
      name,
      suite,
      meanMs: 0,
      stdDevMs: 0,
      opsPerSec: 0,
      iterations: 0,
      p50Ms: 0,
      p95Ms: 0,
      p99Ms: 0,
      timestamp: Date.now(),
    };
  }

  const sorted = [...samples].sort((a, b) => a - b);
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const variance = samples.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / samples.length;
  const stdDev = Math.sqrt(variance);

  return {
    name,
    suite,
    meanMs: mean,
    stdDevMs: stdDev,
    opsPerSec: mean > 0 ? 1000 / mean : 0,
    iterations: samples.length,
    p50Ms: sorted[Math.floor(sorted.length * 0.5)] ?? 0,
    p95Ms: sorted[Math.floor(sorted.length * 0.95)] ?? 0,
    p99Ms: sorted[Math.floor(sorted.length * 0.99)] ?? 0,
    timestamp: Date.now(),
  };
}

// ============================================================================
// CLI Commands (for use in scripts)
// ============================================================================

/**
 * Save current benchmark results as baseline
 */
export function saveBaseline(): void {
  const tracker = new BaselineTracker();
  tracker.saveBaseline();
}

/**
 * Print regression report
 */
export function printReport(): void {
  const tracker = new BaselineTracker();
  console.log(tracker.generateReport());
}

/**
 * Check for regressions (returns exit code)
 */
export function checkRegressions(): number {
  const tracker = new BaselineTracker();
  const report = tracker.compare();

  console.log(tracker.generateReport());

  if (report.regressions > 0) {
    console.error(`\n❌ Found ${report.regressions} regression(s)!`);
    return 1;
  }

  console.log('\n✅ No regressions detected.');
  return 0;
}

// ============================================================================
// Singleton instance for easy use
// ============================================================================

let _defaultTracker: BaselineTracker | null = null;

/**
 * Get default tracker instance
 */
export function getTracker(): BaselineTracker {
  if (!_defaultTracker) {
    _defaultTracker = new BaselineTracker();
  }
  return _defaultTracker;
}

/**
 * Reset default tracker
 */
export function resetTracker(): void {
  _defaultTracker = null;
}

// ============================================================================
// Exports
// ============================================================================

export default BaselineTracker;
