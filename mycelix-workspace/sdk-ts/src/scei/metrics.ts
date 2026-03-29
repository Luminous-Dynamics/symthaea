// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk SCEI Observability & Metrics
 *
 * Centralized metrics collection for all SCEI modules.
 * Supports Prometheus-style metrics export and structured logging.
 *
 * @packageDocumentation
 * @module scei/metrics
 */

// ============================================================================
// METRIC TYPES
// ============================================================================

export type MetricType = 'counter' | 'gauge' | 'histogram' | 'summary';

export interface MetricLabels {
  [key: string]: string | number;
}

export interface MetricDefinition {
  name: string;
  help: string;
  type: MetricType;
  labels?: string[];
}

export interface CounterValue {
  value: number;
  labels: MetricLabels;
}

export interface GaugeValue {
  value: number;
  labels: MetricLabels;
}

export interface HistogramValue {
  count: number;
  sum: number;
  buckets: Map<number, number>; // bucket upper bound -> count
  labels: MetricLabels;
}

export interface SummaryValue {
  count: number;
  sum: number;
  quantiles: Map<number, number>; // quantile -> value
  labels: MetricLabels;
}

// ============================================================================
// SCEI METRIC DEFINITIONS
// ============================================================================

/**
 * SCEI Metric Definitions
 *
 * Naming conventions (Prometheus-compliant):
 * - Counters: Always end with `_total`
 * - Gauges: No suffix, represent current value
 * - Histograms: End with unit suffix (e.g., `_ms`, `_seconds`, `_bytes`)
 * - All names prefixed with `scei_` for namespace
 */
export const SCEI_METRICS: Record<string, MetricDefinition> = {
  // ==========================================================================
  // CALIBRATION METRICS
  // ==========================================================================
  calibration_predictions_total: {
    name: 'scei_calibration_predictions_total',
    help: 'Total number of predictions recorded',
    type: 'counter',
    labels: ['domain', 'source'],
  },
  calibration_resolutions_total: {
    name: 'scei_calibration_resolutions_total',
    help: 'Total number of predictions resolved',
    type: 'counter',
    labels: ['domain', 'outcome'],
  },
  calibration_brier_score: {
    name: 'scei_calibration_brier_score',
    help: 'Current Brier score by domain',
    type: 'gauge',
    labels: ['domain'],
  },
  calibration_report_duration_ms: {
    name: 'scei_calibration_report_duration_ms',
    help: 'Time to generate calibration report in milliseconds',
    type: 'histogram',
    labels: ['scope_type'],
  },
  calibration_report_cache_hits_total: {
    name: 'scei_calibration_report_cache_hits_total',
    help: 'Total cache hits for calibration reports',
    type: 'counter',
    labels: ['scope_type'],
  },
  calibration_report_cache_misses_total: {
    name: 'scei_calibration_report_cache_misses_total',
    help: 'Total cache misses for calibration reports',
    type: 'counter',
    labels: ['scope_type'],
  },
  calibration_insufficient_data_total: {
    name: 'scei_calibration_insufficient_data_total',
    help: 'Total calibration reports skipped due to insufficient data',
    type: 'counter',
    labels: ['scope_type'],
  },
  calibration_overall_error: {
    name: 'scei_calibration_overall_error',
    help: 'Overall calibration error by scope',
    type: 'gauge',
    labels: ['scope_type'],
  },

  // ==========================================================================
  // METABOLISM METRICS
  // ==========================================================================
  metabolism_claims_active: {
    name: 'scei_metabolism_claims_active',
    help: 'Current number of active claims by phase',
    type: 'gauge',
    labels: ['phase', 'domain'],
  },
  metabolism_births_total: {
    name: 'scei_metabolism_births_total',
    help: 'Total claim births',
    type: 'counter',
    labels: ['domain'],
  },
  metabolism_deaths_total: {
    name: 'scei_metabolism_deaths_total',
    help: 'Total claim deaths',
    type: 'counter',
    labels: ['domain', 'reason'],
  },
  metabolism_health_distribution: {
    name: 'scei_metabolism_health_distribution',
    help: 'Distribution of claim health scores',
    type: 'histogram',
    labels: ['domain'],
  },
  metabolism_lifespan_ms: {
    name: 'scei_metabolism_lifespan_ms',
    help: 'Claim lifespan in milliseconds',
    type: 'histogram',
    labels: ['domain', 'death_reason'],
  },
  metabolism_active_claims: {
    name: 'scei_metabolism_active_claims',
    help: 'Current number of active claims',
    type: 'gauge',
    labels: [],
  },
  metabolism_tombstones: {
    name: 'scei_metabolism_tombstones',
    help: 'Current number of tombstones',
    type: 'gauge',
    labels: [],
  },
  metabolism_claim_health: {
    name: 'scei_metabolism_claim_health',
    help: 'Individual claim health scores',
    type: 'gauge',
    labels: ['claim_id'],
  },
  metabolism_evidence_added_total: {
    name: 'scei_metabolism_evidence_added_total',
    help: 'Total evidence items added to claims',
    type: 'counter',
    labels: ['domain'],
  },

  // ==========================================================================
  // PROPAGATION METRICS
  // ==========================================================================
  propagation_requests_total: {
    name: 'scei_propagation_requests_total',
    help: 'Total propagation requests by outcome',
    type: 'counter',
    labels: ['outcome'],
  },
  propagation_affected_claims: {
    name: 'scei_propagation_affected_claims',
    help: 'Number of claims affected per propagation',
    type: 'histogram',
    labels: [],
  },
  propagation_depth: {
    name: 'scei_propagation_depth',
    help: 'Propagation depth reached',
    type: 'histogram',
    labels: [],
  },
  propagation_duration_ms: {
    name: 'scei_propagation_duration_ms',
    help: 'Propagation execution time in milliseconds',
    type: 'histogram',
    labels: ['outcome'],
  },
  propagation_rate_limit_hits_total: {
    name: 'scei_propagation_rate_limit_hits_total',
    help: 'Total rate limit violations',
    type: 'counter',
    labels: ['type'],
  },
  propagation_quarantine_queue: {
    name: 'scei_propagation_quarantine_queue',
    help: 'Current quarantine queue size',
    type: 'gauge',
    labels: [],
  },
  propagation_completed_total: {
    name: 'scei_propagation_completed_total',
    help: 'Total completed propagation operations',
    type: 'counter',
    labels: ['update_type'],
  },
  propagation_blocked_total: {
    name: 'scei_propagation_blocked_total',
    help: 'Total blocked propagation operations',
    type: 'counter',
    labels: ['reason', 'update_type'],
  },

  // ==========================================================================
  // DISCOVERY METRICS
  // ==========================================================================
  discovery_gaps_detected_total: {
    name: 'scei_discovery_gaps_detected_total',
    help: 'Total gaps detected',
    type: 'counter',
    labels: ['type', 'domain', 'severity'],
  },
  discovery_gaps_resolved_total: {
    name: 'scei_discovery_gaps_resolved_total',
    help: 'Total gaps resolved',
    type: 'counter',
    labels: ['type', 'method'],
  },
  discovery_scan_duration_ms: {
    name: 'scei_discovery_scan_duration_ms',
    help: 'Gap detection scan duration in milliseconds',
    type: 'histogram',
    labels: ['domain'],
  },
  discovery_active_gaps: {
    name: 'scei_discovery_active_gaps',
    help: 'Current number of active gaps by type',
    type: 'gauge',
    labels: ['type'],
  },
  discovery_scans_total: {
    name: 'scei_discovery_scans_total',
    help: 'Total gap detection scans completed',
    type: 'counter',
    labels: ['domain'],
  },
  discovery_unresolved_gaps: {
    name: 'scei_discovery_unresolved_gaps',
    help: 'Current number of unresolved gaps',
    type: 'gauge',
    labels: [],
  },

  // ==========================================================================
  // ALGEBRA METRICS
  // ==========================================================================
  algebra_operations_total: {
    name: 'scei_algebra_operations_total',
    help: 'Total algebra operations performed',
    type: 'counter',
    labels: ['operation'],
  },
  algebra_confidence_adjustments: {
    name: 'scei_algebra_confidence_adjustments',
    help: 'Confidence value adjustments',
    type: 'histogram',
    labels: ['operation'],
  },

  // ==========================================================================
  // EVENT BUS METRICS
  // ==========================================================================
  eventbus_events_total: {
    name: 'scei_eventbus_events_total',
    help: 'Total events emitted through the event bus',
    type: 'counter',
    labels: ['type', 'source'],
  },
  eventbus_listener_errors_total: {
    name: 'scei_eventbus_listener_errors_total',
    help: 'Total listener errors',
    type: 'counter',
    labels: [],
  },
  eventbus_listener_timeouts_total: {
    name: 'scei_eventbus_listener_timeouts_total',
    help: 'Total listener timeouts',
    type: 'counter',
    labels: [],
  },
  eventbus_validation_failures_total: {
    name: 'scei_eventbus_validation_failures_total',
    help: 'Total event payload validation failures',
    type: 'counter',
    labels: ['type'],
  },
  eventbus_active_listeners: {
    name: 'scei_eventbus_active_listeners',
    help: 'Current number of active event listeners',
    type: 'gauge',
    labels: [],
  },

  // ==========================================================================
  // PERSISTENCE METRICS
  // ==========================================================================
  persistence_operations_total: {
    name: 'scei_persistence_operations_total',
    help: 'Total persistence operations',
    type: 'counter',
    labels: ['operation', 'namespace'],
  },
  persistence_operation_duration_ms: {
    name: 'scei_persistence_operation_duration_ms',
    help: 'Persistence operation duration in milliseconds',
    type: 'histogram',
    labels: ['operation', 'namespace'],
  },
  persistence_items: {
    name: 'scei_persistence_items',
    help: 'Current number of items in storage by namespace',
    type: 'gauge',
    labels: ['namespace'],
  },
  persistence_expired_purged_total: {
    name: 'scei_persistence_expired_purged_total',
    help: 'Total expired items purged',
    type: 'counter',
    labels: [],
  },

  // ==========================================================================
  // LEGACY METRIC ALIASES (for backward compatibility)
  // ==========================================================================
  // These map old metric names to new ones for smooth migration
  calibration_predictions_recorded: {
    name: 'scei_calibration_predictions_total',
    help: '[DEPRECATED] Use calibration_predictions_total',
    type: 'counter',
    labels: ['domain', 'source'],
  },
  metabolism_claims_total: {
    name: 'scei_metabolism_claims_active',
    help: '[DEPRECATED] Use metabolism_claims_active',
    type: 'gauge',
    labels: ['phase', 'domain'],
  },
  metabolism_claims_born: {
    name: 'scei_metabolism_births_total',
    help: '[DEPRECATED] Use metabolism_births_total',
    type: 'counter',
    labels: ['domain'],
  },
  metabolism_claims_died: {
    name: 'scei_metabolism_deaths_total',
    help: '[DEPRECATED] Use metabolism_deaths_total',
    type: 'counter',
    labels: ['reason'],
  },
  metabolism_tombstone_count: {
    name: 'scei_metabolism_tombstones',
    help: '[DEPRECATED] Use metabolism_tombstones',
    type: 'gauge',
    labels: [],
  },
  metabolism_claim_lifetime_ms: {
    name: 'scei_metabolism_lifespan_ms',
    help: '[DEPRECATED] Use metabolism_lifespan_ms',
    type: 'histogram',
    labels: ['reason'],
  },
  metabolism_lifespan_seconds: {
    name: 'scei_metabolism_lifespan_ms',
    help: '[DEPRECATED] Use metabolism_lifespan_ms',
    type: 'histogram',
    labels: ['domain', 'death_reason'],
  },
  calibration_report_cache_hits: {
    name: 'scei_calibration_report_cache_hits_total',
    help: '[DEPRECATED] Use calibration_report_cache_hits_total',
    type: 'counter',
    labels: ['scope_type'],
  },
  calibration_report_cache_misses: {
    name: 'scei_calibration_report_cache_misses_total',
    help: '[DEPRECATED] Use calibration_report_cache_misses_total',
    type: 'counter',
    labels: ['scope_type'],
  },
  calibration_insufficient_data: {
    name: 'scei_calibration_insufficient_data_total',
    help: '[DEPRECATED] Use calibration_insufficient_data_total',
    type: 'counter',
    labels: ['scope_type'],
  },
  propagation_rate_limit_hits: {
    name: 'scei_propagation_rate_limit_hits_total',
    help: '[DEPRECATED] Use propagation_rate_limit_hits_total',
    type: 'counter',
    labels: ['type'],
  },
  propagation_quarantine_queue_size: {
    name: 'scei_propagation_quarantine_queue',
    help: '[DEPRECATED] Use propagation_quarantine_queue',
    type: 'gauge',
    labels: [],
  },
  propagation_completed: {
    name: 'scei_propagation_completed_total',
    help: '[DEPRECATED] Use propagation_completed_total',
    type: 'counter',
    labels: ['update_type'],
  },
  propagation_blocked: {
    name: 'scei_propagation_blocked_total',
    help: '[DEPRECATED] Use propagation_blocked_total',
    type: 'counter',
    labels: ['reason', 'update_type'],
  },
  discovery_scans_completed: {
    name: 'scei_discovery_scans_total',
    help: '[DEPRECATED] Use discovery_scans_total',
    type: 'counter',
    labels: ['domain'],
  },
  discovery_gaps_found: {
    name: 'scei_discovery_gaps_detected_total',
    help: '[DEPRECATED] Use discovery_gaps_detected_total',
    type: 'counter',
    labels: ['domain'],
  },
  discovery_total_gaps: {
    name: 'scei_discovery_active_gaps',
    help: '[DEPRECATED] Use discovery_active_gaps',
    type: 'gauge',
    labels: [],
  },
  discovery_gaps_resolved: {
    name: 'scei_discovery_gaps_resolved_total',
    help: '[DEPRECATED] Use discovery_gaps_resolved_total',
    type: 'counter',
    labels: ['method', 'gap_type'],
  },
};

// ============================================================================
// METRICS COLLECTOR
// ============================================================================

/**
 * SCEI Metrics Collector
 *
 * Centralized metrics collection with Prometheus-compatible export.
 *
 * @example
 * ```typescript
 * const metrics = getSCEIMetrics();
 *
 * // Increment counter
 * metrics.incrementCounter('calibration_predictions_total', { domain: 'science' });
 *
 * // Set gauge
 * metrics.setGauge('propagation_quarantine_queue_size', 5);
 *
 * // Record histogram
 * metrics.recordHistogram('propagation_duration_ms', 150, { outcome: 'completed' });
 *
 * // Export for Prometheus
 * const prometheusOutput = metrics.exportPrometheus();
 * ```
 */
export class SCEIMetricsCollector {
  private counters: Map<string, Map<string, CounterValue>> = new Map();
  private gauges: Map<string, Map<string, GaugeValue>> = new Map();
  private histograms: Map<string, Map<string, HistogramValue>> = new Map();

  // Default histogram buckets
  private defaultBuckets = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000];

  constructor() {
    // Initialize metric storage for all defined metrics
    for (const [key, def] of Object.entries(SCEI_METRICS)) {
      switch (def.type) {
        case 'counter':
          this.counters.set(key, new Map());
          break;
        case 'gauge':
          this.gauges.set(key, new Map());
          break;
        case 'histogram':
        case 'summary':
          this.histograms.set(key, new Map());
          break;
      }
    }
  }

  // ==========================================================================
  // COUNTER OPERATIONS
  // ==========================================================================

  /**
   * Increment a counter
   */
  incrementCounter(name: string, labels: MetricLabels = {}, value: number = 1): void {
    const labelKey = this.labelsToKey(labels);
    const counterMap = this.counters.get(name);

    if (!counterMap) {
      console.warn(`Unknown counter metric: ${name}`);
      return;
    }

    const existing = counterMap.get(labelKey);
    if (existing) {
      existing.value += value;
    } else {
      counterMap.set(labelKey, { value, labels });
    }
  }

  /**
   * Get counter value
   */
  getCounter(name: string, labels: MetricLabels = {}): number {
    const labelKey = this.labelsToKey(labels);
    return this.counters.get(name)?.get(labelKey)?.value ?? 0;
  }

  // ==========================================================================
  // GAUGE OPERATIONS
  // ==========================================================================

  /**
   * Set a gauge value
   */
  setGauge(name: string, value: number, labels: MetricLabels = {}): void {
    const labelKey = this.labelsToKey(labels);
    const gaugeMap = this.gauges.get(name);

    if (!gaugeMap) {
      console.warn(`Unknown gauge metric: ${name}`);
      return;
    }

    gaugeMap.set(labelKey, { value, labels });
  }

  /**
   * Increment a gauge
   */
  incrementGauge(name: string, labels: MetricLabels = {}, delta: number = 1): void {
    const current = this.getGauge(name, labels);
    this.setGauge(name, current + delta, labels);
  }

  /**
   * Decrement a gauge
   */
  decrementGauge(name: string, labels: MetricLabels = {}, delta: number = 1): void {
    const current = this.getGauge(name, labels);
    this.setGauge(name, current - delta, labels);
  }

  /**
   * Get gauge value
   */
  getGauge(name: string, labels: MetricLabels = {}): number {
    const labelKey = this.labelsToKey(labels);
    return this.gauges.get(name)?.get(labelKey)?.value ?? 0;
  }

  // ==========================================================================
  // HISTOGRAM OPERATIONS
  // ==========================================================================

  /**
   * Record a histogram observation
   */
  recordHistogram(name: string, value: number, labels: MetricLabels = {}): void {
    const labelKey = this.labelsToKey(labels);
    const histMap = this.histograms.get(name);

    if (!histMap) {
      console.warn(`Unknown histogram metric: ${name}`);
      return;
    }

    let hist = histMap.get(labelKey);
    if (!hist) {
      hist = {
        count: 0,
        sum: 0,
        buckets: new Map(this.defaultBuckets.map((b) => [b, 0])),
        labels,
      };
      histMap.set(labelKey, hist);
    }

    hist.count++;
    hist.sum += value;

    // Update buckets
    for (const bucket of this.defaultBuckets) {
      if (value <= bucket) {
        hist.buckets.set(bucket, (hist.buckets.get(bucket) ?? 0) + 1);
      }
    }
  }

  /**
   * Get histogram statistics
   */
  getHistogramStats(
    name: string,
    labels: MetricLabels = {}
  ): { count: number; sum: number; avg: number; p50: number; p95: number; p99: number } | null {
    const labelKey = this.labelsToKey(labels);
    const hist = this.histograms.get(name)?.get(labelKey);

    if (!hist || hist.count === 0) {
      return null;
    }

    // Approximate percentiles from buckets
    const findPercentile = (p: number): number => {
      const target = hist.count * p;
      let cumulative = 0;
      for (const [bucket, count] of hist.buckets) {
        cumulative += count;
        if (cumulative >= target) {
          return bucket;
        }
      }
      return this.defaultBuckets[this.defaultBuckets.length - 1];
    };

    return {
      count: hist.count,
      sum: hist.sum,
      avg: hist.sum / hist.count,
      p50: findPercentile(0.5),
      p95: findPercentile(0.95),
      p99: findPercentile(0.99),
    };
  }

  // ==========================================================================
  // TIMING HELPERS
  // ==========================================================================

  /**
   * Time an operation and record to histogram
   */
  async timeAsync<T>(
    name: string,
    labels: MetricLabels,
    operation: () => Promise<T>
  ): Promise<T> {
    const start = performance.now();
    try {
      return await operation();
    } finally {
      const duration = performance.now() - start;
      this.recordHistogram(name, duration, labels);
    }
  }

  /**
   * Time a sync operation and record to histogram
   */
  timeSync<T>(name: string, labels: MetricLabels, operation: () => T): T {
    const start = performance.now();
    try {
      return operation();
    } finally {
      const duration = performance.now() - start;
      this.recordHistogram(name, duration, labels);
    }
  }

  /**
   * Create a timer that can be stopped manually
   */
  startTimer(name: string, labels: MetricLabels = {}): () => number {
    const start = performance.now();
    return () => {
      const duration = performance.now() - start;
      this.recordHistogram(name, duration, labels);
      return duration;
    };
  }

  // ==========================================================================
  // EXPORT
  // ==========================================================================

  /**
   * Export metrics in Prometheus format
   */
  exportPrometheus(): string {
    const lines: string[] = [];

    // Export counters
    for (const [name, values] of this.counters) {
      const def = SCEI_METRICS[name];
      if (def) {
        lines.push(`# HELP ${def.name} ${def.help}`);
        lines.push(`# TYPE ${def.name} counter`);
      }
      for (const [, counter] of values) {
        const labelStr = this.labelsToPrometheus(counter.labels);
        lines.push(`${def?.name ?? name}${labelStr} ${counter.value}`);
      }
    }

    // Export gauges
    for (const [name, values] of this.gauges) {
      const def = SCEI_METRICS[name];
      if (def) {
        lines.push(`# HELP ${def.name} ${def.help}`);
        lines.push(`# TYPE ${def.name} gauge`);
      }
      for (const [, gauge] of values) {
        const labelStr = this.labelsToPrometheus(gauge.labels);
        lines.push(`${def?.name ?? name}${labelStr} ${gauge.value}`);
      }
    }

    // Export histograms
    for (const [name, values] of this.histograms) {
      const def = SCEI_METRICS[name];
      if (def) {
        lines.push(`# HELP ${def.name} ${def.help}`);
        lines.push(`# TYPE ${def.name} histogram`);
      }
      for (const [, hist] of values) {
        const baseName = def?.name ?? name;
        const labelStr = this.labelsToPrometheus(hist.labels);

        // Bucket values
        for (const [bucket, count] of hist.buckets) {
          const bucketLabels =
            labelStr === '' ? `{le="${bucket}"}` : labelStr.replace('}', `,le="${bucket}"}`);
          lines.push(`${baseName}_bucket${bucketLabels} ${count}`);
        }

        // +Inf bucket
        const infLabels =
          labelStr === '' ? `{le="+Inf"}` : labelStr.replace('}', `,le="+Inf"}`);
        lines.push(`${baseName}_bucket${infLabels} ${hist.count}`);

        // Sum and count
        lines.push(`${baseName}_sum${labelStr} ${hist.sum}`);
        lines.push(`${baseName}_count${labelStr} ${hist.count}`);
      }
    }

    return lines.join('\n');
  }

  /**
   * Export metrics as JSON (for debugging/dashboards)
   */
  exportJSON(): SCEIMetricsSnapshot {
    const counters: Record<string, CounterValue[]> = {};
    const gauges: Record<string, GaugeValue[]> = {};
    const histograms: Record<
      string,
      Array<{ labels: MetricLabels; count: number; sum: number; avg: number }>
    > = {};

    for (const [name, values] of this.counters) {
      counters[name] = Array.from(values.values());
    }

    for (const [name, values] of this.gauges) {
      gauges[name] = Array.from(values.values());
    }

    for (const [name, values] of this.histograms) {
      histograms[name] = Array.from(values.values()).map((h) => ({
        labels: h.labels,
        count: h.count,
        sum: h.sum,
        avg: h.count > 0 ? h.sum / h.count : 0,
      }));
    }

    return {
      timestamp: Date.now(),
      counters,
      gauges,
      histograms,
    };
  }

  // ==========================================================================
  // RESET
  // ==========================================================================

  /**
   * Reset all metrics
   */
  reset(): void {
    for (const map of this.counters.values()) map.clear();
    for (const map of this.gauges.values()) map.clear();
    for (const map of this.histograms.values()) map.clear();
  }

  // ==========================================================================
  // HELPERS
  // ==========================================================================

  private labelsToKey(labels: MetricLabels): string {
    const sorted = Object.entries(labels).sort(([a], [b]) => a.localeCompare(b));
    return JSON.stringify(sorted);
  }

  private labelsToPrometheus(labels: MetricLabels): string {
    const entries = Object.entries(labels);
    if (entries.length === 0) return '';

    const labelPairs = entries.map(([k, v]) => `${k}="${v}"`).join(',');
    return `{${labelPairs}}`;
  }
}

export interface SCEIMetricsSnapshot {
  timestamp: number;
  counters: Record<string, CounterValue[]>;
  gauges: Record<string, GaugeValue[]>;
  histograms: Record<string, Array<{ labels: MetricLabels; count: number; sum: number; avg: number }>>;
}

// ============================================================================
// SINGLETON
// ============================================================================

let instance: SCEIMetricsCollector | null = null;

/**
 * Get the global SCEI metrics collector
 */
export function getSCEIMetrics(): SCEIMetricsCollector {
  if (!instance) {
    instance = new SCEIMetricsCollector();
  }
  return instance;
}

/**
 * Reset the global metrics collector (for testing)
 */
export function resetSCEIMetrics(): void {
  if (instance) {
    instance.reset();
  }
  instance = null;
}
