/**
 * @mycelix/sdk Calibration Engine
 *
 * The keystone of Self-Correcting Epistemic Infrastructure.
 * Answers: "When we said 70%, were we right 70% of the time?"
 *
 * Without calibration, the system becomes a confident bullshitter.
 *
 * @packageDocumentation
 * @module calibration/engine
 */

import {
  type CalibrationBucket,
  type CalibrationConfig,
  type CalibrationReport,
  type CalibrationScope,
  type ResolutionRecord,
  type ResolutionEvidence,
  type AdjustedConfidence,
  type CredibilityCheck,
  DEFAULT_CALIBRATION_CONFIG,
  InsufficientCalibrationDataError,
} from './types.js';
import {
  getSCEIEventBus,
  createSCEIEvent,
  type SCEIEventBus,
} from '../scei/event-bus.js';
import { getSCEIMetrics, type SCEIMetricsCollector } from '../scei/metrics.js';
import {
  getSCEIPersistence,
  type SCEIPersistenceManager,
} from '../scei/persistence.js';

import type { SourceType } from '../knowledge/index.js';

// ============================================================================
// CALIBRATION ENGINE
// ============================================================================

/**
 * Calibration Engine
 *
 * Tracks claim resolutions and computes calibration metrics.
 * Gates high-confidence claims and propagation on calibration history.
 *
 * @example
 * ```typescript
 * const engine = new CalibrationEngine();
 *
 * // Record resolutions as claims are resolved
 * await engine.recordResolution(claim, 'correct', evidence);
 *
 * // Generate calibration report
 * const report = await engine.generateReport({ type: 'domain', domain: 'climate' });
 * console.log(`Brier score: ${report.brierScore}`);
 *
 * // Check if a confidence claim is credible
 * const check = engine.isConfidenceCredible(0.9, 'climate');
 * if (!check.credible) {
 *   console.log(`Reduce confidence: ${check.recommendation}`);
 * }
 * ```
 */
export class CalibrationEngine {
  private resolutionHistory: ResolutionRecord[] = [];
  private reportCache: Map<string, CalibrationReport> = new Map();
  private config: CalibrationConfig;
  private initialized: boolean = false;

  // SCEI Infrastructure
  private eventBus: SCEIEventBus;
  private metrics: SCEIMetricsCollector;
  private persistence: SCEIPersistenceManager;

  constructor(config: Partial<CalibrationConfig> = {}) {
    this.config = { ...DEFAULT_CALIBRATION_CONFIG, ...config };
    this.eventBus = getSCEIEventBus();
    this.metrics = getSCEIMetrics();
    this.persistence = getSCEIPersistence();
  }

  // ==========================================================================
  // PERSISTENCE
  // ==========================================================================

  /**
   * Load resolution history from persistence layer
   *
   * Call this on startup to restore previous session's data.
   *
   * @example
   * ```typescript
   * const engine = new CalibrationEngine({ enablePersistence: true });
   * await engine.loadFromPersistence();
   * ```
   */
  async loadFromPersistence(): Promise<number> {
    if (!this.config.enablePersistence) {
      return 0;
    }

    try {
      const records = await this.persistence.queryCalibrationRecords<ResolutionRecord>({
        sortBy: 'createdAt',
        sortOrder: 'asc',
      });

      this.resolutionHistory = records;
      this.initialized = true;

      this.metrics.incrementCounter('persistence_operations_total', {
        namespace: 'calibration:records',
        operation: 'load',
      });
      this.metrics.setGauge('calibration_loaded_records', records.length);

      return records.length;
    } catch (error) {
      console.error('CalibrationEngine: Failed to load from persistence:', error);
      return 0;
    }
  }

  /**
   * Persist a resolution record
   */
  private async persistRecord(record: ResolutionRecord): Promise<void> {
    if (!this.config.enablePersistence) {
      return;
    }

    try {
      await this.persistence.saveCalibrationRecord(record.claimId, record, {
        ttl: this.config.persistenceTtlMs || undefined,
        tags: [record.domain, record.outcome],
      });

      this.metrics.incrementCounter('persistence_operations_total', {
        namespace: 'calibration:records',
        operation: 'save',
      });
    } catch (error) {
      console.error('CalibrationEngine: Failed to persist record:', error);
      // Don't throw - persistence failure shouldn't break the engine
    }
  }

  /**
   * Check if persistence is enabled and data has been loaded
   */
  isPersistenceReady(): boolean {
    return this.config.enablePersistence && this.initialized;
  }

  // ==========================================================================
  // RECORDING
  // ==========================================================================

  /**
   * Record a claim resolution for calibration tracking
   *
   * @param claim - The resolved claim
   * @param outcome - Whether the claim was correct, incorrect, or ambiguous
   * @param evidence - Evidence for the resolution
   */
  async recordResolution(
    claim: { id: string; confidence: number; domain?: string; sources?: string[] },
    outcome: 'correct' | 'incorrect' | 'ambiguous',
    evidence: ResolutionEvidence[] = []
  ): Promise<void> {
    // Ambiguous outcomes are recorded but don't contribute to calibration
    const record: ResolutionRecord = {
      claimId: claim.id,
      statedConfidence: claim.confidence,
      outcome,
      domain: claim.domain ?? 'unknown',
      sourceType: this.extractSourceType(claim),
      resolvedAt: Date.now(),
      evidence,
    };

    this.resolutionHistory.push(record);

    // Persist to storage
    await this.persistRecord(record);

    // Invalidate relevant cache entries
    this.invalidateCache(record);

    // Emit SCEI event
    const event = createSCEIEvent('calibration:prediction_resolved', 'calibration', {
      claimId: claim.id,
      predictedConfidence: claim.confidence,
      outcome,
      domain: record.domain,
      wasCorrect: outcome === 'correct',
    });
    await this.eventBus.emit(event);

    // Record metrics
    this.metrics.incrementCounter('calibration_predictions_total', {
      domain: record.domain,
      outcome,
    });
  }

  // ==========================================================================
  // REPORTING
  // ==========================================================================

  /**
   * Generate calibration report for a scope
   *
   * @param scope - What to compute calibration for
   * @throws InsufficientCalibrationDataError if not enough data
   */
  async generateReport(scope: CalibrationScope): Promise<CalibrationReport> {
    const startTime = Date.now();
    const cacheKey = this.scopeToKey(scope);
    const cached = this.reportCache.get(cacheKey);

    // Return cached if fresh (< 1 hour old)
    if (cached && Date.now() - cached.generatedAt < 60 * 60 * 1000) {
      this.metrics.incrementCounter('calibration_report_cache_hits', {
        scope_type: scope.type,
      });
      return cached;
    }

    this.metrics.incrementCounter('calibration_report_cache_misses', {
      scope_type: scope.type,
    });

    // Filter to relevant records (exclude ambiguous)
    const relevantRecords = this.filterByScope(scope).filter(
      (r) => r.outcome !== 'ambiguous'
    );

    if (relevantRecords.length < this.config.minSampleSize) {
      this.metrics.incrementCounter('calibration_insufficient_data', {
        scope_type: scope.type,
      });
      throw new InsufficientCalibrationDataError(
        this.config.minSampleSize,
        relevantRecords.length
      );
    }

    // Build calibration buckets
    const buckets = this.buildBuckets(relevantRecords);

    // Calculate metrics
    const brierScore = this.calculateBrierScore(relevantRecords);
    const logLoss = this.calculateLogLoss(relevantRecords);
    const inflationBias = this.detectInflation(buckets);
    const overallError = this.weightedCalibrationError(buckets);

    const report: CalibrationReport = {
      scope,
      timeWindow: this.getTimeWindow(relevantRecords),
      buckets,
      brierScore,
      logLoss,
      overallCalibrationError: overallError,
      inflationBias,
      reliabilityBySource: this.reliabilityBySource(relevantRecords),
      trend: this.detectTrend(scope),
      sampleSize: relevantRecords.length,
      generatedAt: Date.now(),
    };

    this.reportCache.set(cacheKey, report);

    // Record generation time metric
    const duration = Date.now() - startTime;
    this.metrics.recordHistogram('calibration_report_duration_ms', duration, {
      scope_type: scope.type,
    });

    // Record calibration quality metrics
    this.metrics.setGauge('calibration_brier_score', brierScore, {
      scope_type: scope.type,
    });
    this.metrics.setGauge('calibration_overall_error', overallError, {
      scope_type: scope.type,
    });

    return report;
  }

  /**
   * Get the latest report for a scope, or null if insufficient data
   */
  async getLatestReport(scope: CalibrationScope): Promise<CalibrationReport | null> {
    try {
      return await this.generateReport(scope);
    } catch (e) {
      if (e instanceof InsufficientCalibrationDataError) {
        return null;
      }
      throw e;
    }
  }

  // ==========================================================================
  // CONFIDENCE ADJUSTMENT
  // ==========================================================================

  /**
   * Adjust confidence based on historical calibration
   *
   * If we're systematically overconfident in domain X,
   * new claims in domain X get their confidence reduced.
   */
  async adjustConfidence(
    statedConfidence: number,
    domain: string
  ): Promise<AdjustedConfidence> {
    const report = await this.getLatestReport({ type: 'domain', domain });

    if (!report || report.buckets.length === 0) {
      return {
        adjustedConfidence: statedConfidence,
        uncertainty: 0.3, // High uncertainty due to no calibration history
        calibrationAvailable: false,
        reason: 'No calibration history for this domain',
      };
    }

    // Find the relevant bucket
    const bucket = this.findBucket(report.buckets, statedConfidence);

    if (bucket.claimCount < 5) {
      return {
        adjustedConfidence: statedConfidence,
        uncertainty: 0.25,
        calibrationAvailable: true,
        reason: `Limited data in confidence range (${bucket.claimCount} claims)`,
        originalConfidence: statedConfidence,
        bucketStats: bucket,
      };
    }

    // Apply calibration adjustment
    // If we claimed 80% but were only right 60% of the time, adjust toward 60%
    const adjustmentFactor =
      bucket.expectedAccuracy > 0
        ? bucket.actualAccuracy / bucket.expectedAccuracy
        : 1;

    const adjustedConfidence = statedConfidence * adjustmentFactor;

    return {
      adjustedConfidence: Math.min(0.99, Math.max(0.01, adjustedConfidence)),
      uncertainty: bucket.calibrationError,
      calibrationAvailable: true,
      reason: `Adjusted based on ${bucket.claimCount} historical claims in this confidence range`,
      originalConfidence: statedConfidence,
      bucketStats: bucket,
    };
  }

  // ==========================================================================
  // CREDIBILITY GATES
  // ==========================================================================

  /**
   * Gate function: Is this confidence claim credible given calibration history?
   *
   * @param statedConfidence - The confidence being claimed
   * @param domain - The domain of the claim
   * @param errorThreshold - Maximum acceptable calibration error (default: 0.15)
   */
  isConfidenceCredible(
    statedConfidence: number,
    domain: string,
    errorThreshold: number = this.config.highConfidenceErrorThreshold
  ): CredibilityCheck {
    const cacheKey = `domain:${domain}`;
    const report = this.reportCache.get(cacheKey);

    if (!report) {
      return {
        credible: false,
        reason: 'No calibration history',
        recommendation: 'Treat as low-confidence until calibration data accumulates',
      };
    }

    const bucket = this.findBucket(report.buckets, statedConfidence);

    // High-confidence claims require good calibration
    if (statedConfidence > 0.8 && bucket.calibrationError > errorThreshold) {
      return {
        credible: false,
        reason: `Historical calibration error (${(bucket.calibrationError * 100).toFixed(1)}%) exceeds threshold for high-confidence claims`,
        recommendation: 'Adjust confidence or gather more evidence',
        suggestedConfidence: bucket.actualAccuracy,
      };
    }

    // General check
    if (bucket.calibrationError > errorThreshold * 2) {
      return {
        credible: false,
        reason: `Calibration error (${(bucket.calibrationError * 100).toFixed(1)}%) is high`,
        recommendation: 'Consider calibration-adjusted confidence',
        suggestedConfidence: bucket.actualAccuracy,
      };
    }

    return {
      credible: true,
      reason: 'Within acceptable calibration bounds',
      historicalAccuracy: bucket.actualAccuracy,
    };
  }

  /**
   * Check if propagation from this claim should be allowed
   */
  canPropagate(
    claim: { confidence: number; domain?: string },
    domain: string = claim.domain ?? 'unknown'
  ): { allowed: boolean; reason: string } {
    const check = this.isConfidenceCredible(claim.confidence, domain);

    if (!check.credible && claim.confidence > 0.7) {
      return {
        allowed: false,
        reason: `Propagation blocked: ${check.reason}`,
      };
    }

    return {
      allowed: true,
      reason: 'Calibration check passed',
    };
  }

  // ==========================================================================
  // PRIVATE METHODS
  // ==========================================================================

  private buildBuckets(records: ResolutionRecord[]): CalibrationBucket[] {
    const buckets: CalibrationBucket[] = [];

    for (let i = 0; i < this.config.bucketCount; i++) {
      const low = i / this.config.bucketCount;
      const high = (i + 1) / this.config.bucketCount;

      const inBucket = records.filter(
        (r) => r.statedConfidence >= low && r.statedConfidence < high
      );

      const correctCount = inBucket.filter((r) => r.outcome === 'correct').length;
      const actualAccuracy = inBucket.length > 0 ? correctCount / inBucket.length : 0;
      const expectedAccuracy = (low + high) / 2;

      buckets.push({
        confidenceRange: [low, high],
        claimCount: inBucket.length,
        correctCount,
        actualAccuracy,
        expectedAccuracy,
        calibrationError: Math.abs(actualAccuracy - expectedAccuracy),
      });
    }

    return buckets;
  }

  private calculateBrierScore(records: ResolutionRecord[]): number {
    if (records.length === 0) return 1; // Worst possible

    const sumSquaredError = records.reduce((sum, r) => {
      const outcome = r.outcome === 'correct' ? 1 : 0;
      return sum + Math.pow(r.statedConfidence - outcome, 2);
    }, 0);

    return sumSquaredError / records.length;
  }

  private calculateLogLoss(records: ResolutionRecord[]): number {
    if (records.length === 0) return Infinity;

    const epsilon = 1e-15; // Prevent log(0)

    const sumLogLoss = records.reduce((sum, r) => {
      const p = Math.max(epsilon, Math.min(1 - epsilon, r.statedConfidence));
      const y = r.outcome === 'correct' ? 1 : 0;
      return sum - (y * Math.log(p) + (1 - y) * Math.log(1 - p));
    }, 0);

    return sumLogLoss / records.length;
  }

  private detectInflation(buckets: CalibrationBucket[]): number {
    // Positive = overconfident (stated > actual)
    // Negative = underconfident (stated < actual)

    let totalWeightedBias = 0;
    let totalWeight = 0;

    for (const bucket of buckets) {
      if (bucket.claimCount > 0) {
        const bias = bucket.expectedAccuracy - bucket.actualAccuracy;
        totalWeightedBias += bias * bucket.claimCount;
        totalWeight += bucket.claimCount;
      }
    }

    return totalWeight > 0 ? totalWeightedBias / totalWeight : 0;
  }

  private weightedCalibrationError(buckets: CalibrationBucket[]): number {
    let totalError = 0;
    let totalWeight = 0;

    for (const bucket of buckets) {
      if (bucket.claimCount > 0) {
        totalError += bucket.calibrationError * bucket.claimCount;
        totalWeight += bucket.claimCount;
      }
    }

    return totalWeight > 0 ? totalError / totalWeight : 0;
  }

  private reliabilityBySource(
    records: ResolutionRecord[]
  ): Map<SourceType, number> {
    const bySource = new Map<SourceType, { correct: number; total: number }>();

    for (const record of records) {
      const stats = bySource.get(record.sourceType) ?? { correct: 0, total: 0 };
      stats.total++;
      if (record.outcome === 'correct') stats.correct++;
      bySource.set(record.sourceType, stats);
    }

    const reliability = new Map<SourceType, number>();
    for (const [source, stats] of bySource) {
      reliability.set(source, stats.total > 0 ? stats.correct / stats.total : 0);
    }

    return reliability;
  }

  private detectTrend(scope: CalibrationScope): 'improving' | 'stable' | 'degrading' {
    const records = this.filterByScope(scope).filter((r) => r.outcome !== 'ambiguous');

    if (records.length < 60) return 'stable'; // Not enough data

    // Compare recent half to older half
    const sorted = [...records].sort((a, b) => a.resolvedAt - b.resolvedAt);
    const midpoint = Math.floor(sorted.length / 2);

    const olderBuckets = this.buildBuckets(sorted.slice(0, midpoint));
    const recentBuckets = this.buildBuckets(sorted.slice(midpoint));

    const olderError = this.weightedCalibrationError(olderBuckets);
    const recentError = this.weightedCalibrationError(recentBuckets);

    const improvement = olderError - recentError;

    if (improvement > 0.05) return 'improving';
    if (improvement < -0.05) return 'degrading';
    return 'stable';
  }

  private filterByScope(scope: CalibrationScope): ResolutionRecord[] {
    switch (scope.type) {
      case 'global':
        return this.resolutionHistory;
      case 'domain':
        return this.resolutionHistory.filter((r) => r.domain === scope.domain);
      case 'source':
        return this.resolutionHistory.filter((r) => r.sourceType === scope.sourceType);
      case 'agent':
        // Would need agent info in ResolutionRecord
        return this.resolutionHistory;
      case 'claimType':
        // Would need claim type in ResolutionRecord
        return this.resolutionHistory;
    }
  }

  private findBucket(
    buckets: CalibrationBucket[],
    confidence: number
  ): CalibrationBucket {
    const index = Math.min(
      Math.floor(confidence * buckets.length),
      buckets.length - 1
    );
    return buckets[index];
  }

  private getTimeWindow(records: ResolutionRecord[]): { start: number; end: number } {
    if (records.length === 0) {
      const now = Date.now();
      return { start: now, end: now };
    }

    const times = records.map((r) => r.resolvedAt);
    return {
      start: Math.min(...times),
      end: Math.max(...times),
    };
  }

  private scopeToKey(scope: CalibrationScope): string {
    switch (scope.type) {
      case 'global':
        return 'global';
      case 'domain':
        return `domain:${scope.domain}`;
      case 'source':
        return `source:${scope.sourceType}`;
      case 'agent':
        return `agent:${scope.agentId}`;
      case 'claimType':
        return `claimType:${scope.claimType}`;
    }
  }

  private invalidateCache(record: ResolutionRecord): void {
    // Invalidate global and domain-specific caches
    this.reportCache.delete('global');
    this.reportCache.delete(`domain:${record.domain}`);
    this.reportCache.delete(`source:${record.sourceType}`);
  }

  private extractSourceType(claim: { sources?: string[] }): SourceType {
    // Simple heuristic - would be more sophisticated in practice
    if (!claim.sources || claim.sources.length === 0) {
      return 'Other';
    }

    const source = claim.sources[0].toLowerCase();

    if (source.includes('arxiv') || source.includes('doi')) {
      return 'AcademicPaper';
    }
    if (source.includes('census') || source.includes('.gov')) {
      return 'DataSet';
    }
    if (source.includes('market') || source.includes('prediction')) {
      return 'PredictionMarket';
    }

    return 'Other';
  }

  // ==========================================================================
  // STATISTICS
  // ==========================================================================

  /**
   * Get summary statistics
   */
  getStats(): CalibrationStats {
    const total = this.resolutionHistory.length;
    const correct = this.resolutionHistory.filter((r) => r.outcome === 'correct').length;
    const incorrect = this.resolutionHistory.filter(
      (r) => r.outcome === 'incorrect'
    ).length;
    const ambiguous = this.resolutionHistory.filter(
      (r) => r.outcome === 'ambiguous'
    ).length;

    const domains = new Set(this.resolutionHistory.map((r) => r.domain));

    return {
      totalResolutions: total,
      correctCount: correct,
      incorrectCount: incorrect,
      ambiguousCount: ambiguous,
      overallAccuracy: total > 0 ? correct / (correct + incorrect) : 0,
      domainsTracked: domains.size,
      oldestResolution:
        total > 0
          ? Math.min(...this.resolutionHistory.map((r) => r.resolvedAt))
          : null,
      newestResolution:
        total > 0
          ? Math.max(...this.resolutionHistory.map((r) => r.resolvedAt))
          : null,
    };
  }
}

/**
 * Summary statistics for the calibration engine
 */
export interface CalibrationStats {
  totalResolutions: number;
  correctCount: number;
  incorrectCount: number;
  ambiguousCount: number;
  overallAccuracy: number;
  domainsTracked: number;
  oldestResolution: number | null;
  newestResolution: number | null;
}

// Singleton instance
let instance: CalibrationEngine | null = null;

/**
 * Get the global calibration engine instance
 */
export function getCalibrationEngine(
  config?: Partial<CalibrationConfig>
): CalibrationEngine {
  if (!instance) {
    instance = new CalibrationEngine(config);
  }
  return instance;
}

/**
 * Reset the global calibration engine (for testing)
 */
export function resetCalibrationEngine(): void {
  instance = null;
}
