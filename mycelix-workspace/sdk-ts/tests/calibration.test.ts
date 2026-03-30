// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Calibration Engine Tests
 *
 * Tests for the keystone of Self-Correcting Epistemic Infrastructure.
 * Verifies Brier score, log loss, bucket calibration, and confidence adjustment.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  CalibrationEngine,
  getCalibrationEngine,
  resetCalibrationEngine,
  type CalibrationStats,
} from '../src/calibration/engine.js';
import {
  InsufficientCalibrationDataError,
  type CalibrationScope,
  type CalibrationBucket,
} from '../src/calibration/types.js';

describe('Calibration Engine - Recording', () => {
  let engine: CalibrationEngine;

  beforeEach(() => {
    engine = new CalibrationEngine();
  });

  it('should record a correct resolution', async () => {
    await engine.recordResolution(
      { id: 'claim-1', confidence: 0.8, domain: 'science' },
      'correct',
      []
    );

    const stats = engine.getStats();
    expect(stats.totalResolutions).toBe(1);
    expect(stats.correctCount).toBe(1);
    expect(stats.incorrectCount).toBe(0);
  });

  it('should record an incorrect resolution', async () => {
    await engine.recordResolution(
      { id: 'claim-1', confidence: 0.9, domain: 'politics' },
      'incorrect',
      []
    );

    const stats = engine.getStats();
    expect(stats.totalResolutions).toBe(1);
    expect(stats.correctCount).toBe(0);
    expect(stats.incorrectCount).toBe(1);
  });

  it('should record an ambiguous resolution', async () => {
    await engine.recordResolution(
      { id: 'claim-1', confidence: 0.5, domain: 'philosophy' },
      'ambiguous',
      []
    );

    const stats = engine.getStats();
    expect(stats.totalResolutions).toBe(1);
    expect(stats.ambiguousCount).toBe(1);
  });

  it('should track multiple domains', async () => {
    await engine.recordResolution({ id: '1', confidence: 0.7, domain: 'science' }, 'correct');
    await engine.recordResolution({ id: '2', confidence: 0.8, domain: 'politics' }, 'incorrect');
    await engine.recordResolution({ id: '3', confidence: 0.6, domain: 'economics' }, 'correct');

    const stats = engine.getStats();
    expect(stats.domainsTracked).toBe(3);
  });
});

describe('Calibration Engine - Brier Score', () => {
  let engine: CalibrationEngine;

  beforeEach(() => {
    engine = new CalibrationEngine({ minSampleSize: 5 });
  });

  it('should calculate perfect Brier score for perfect predictions', async () => {
    // Claim 100% confident and correct = 0 error
    // Claim 0% confident and incorrect = 0 error
    for (let i = 0; i < 10; i++) {
      await engine.recordResolution(
        { id: `claim-${i}`, confidence: 1.0, domain: 'test' },
        'correct'
      );
    }

    const report = await engine.generateReport({ type: 'global' });
    expect(report.brierScore).toBe(0);
  });

  it('should calculate worst Brier score for worst predictions', async () => {
    // Claim 100% confident but wrong = max error
    for (let i = 0; i < 10; i++) {
      await engine.recordResolution(
        { id: `claim-${i}`, confidence: 1.0, domain: 'test' },
        'incorrect'
      );
    }

    const report = await engine.generateReport({ type: 'global' });
    expect(report.brierScore).toBe(1);
  });

  it('should calculate moderate Brier score for 50% predictions', async () => {
    // Claim 50% confident - Brier depends on outcomes
    for (let i = 0; i < 10; i++) {
      await engine.recordResolution(
        { id: `claim-${i}`, confidence: 0.5, domain: 'test' },
        i % 2 === 0 ? 'correct' : 'incorrect'
      );
    }

    const report = await engine.generateReport({ type: 'global' });
    // For 50% confidence: (0.5-1)^2 = 0.25 for correct, (0.5-0)^2 = 0.25 for incorrect
    expect(report.brierScore).toBe(0.25);
  });

  it('should calculate Brier score for realistic distribution', async () => {
    // 80% confident predictions that are 80% correct
    for (let i = 0; i < 100; i++) {
      await engine.recordResolution(
        { id: `claim-${i}`, confidence: 0.8, domain: 'test' },
        i < 80 ? 'correct' : 'incorrect'
      );
    }

    const report = await engine.generateReport({ type: 'global' });
    // For 80% conf, 80% correct: 0.8*(0.8-1)^2 + 0.2*(0.8-0)^2 = 0.8*0.04 + 0.2*0.64 = 0.16
    expect(report.brierScore).toBeCloseTo(0.16, 2);
  });
});

describe('Calibration Engine - Log Loss', () => {
  let engine: CalibrationEngine;

  beforeEach(() => {
    engine = new CalibrationEngine({ minSampleSize: 5 });
  });

  it('should calculate low log loss for good predictions', async () => {
    // High confidence correct predictions
    for (let i = 0; i < 10; i++) {
      await engine.recordResolution(
        { id: `claim-${i}`, confidence: 0.95, domain: 'test' },
        'correct'
      );
    }

    const report = await engine.generateReport({ type: 'global' });
    expect(report.logLoss).toBeLessThan(0.1);
  });

  it('should calculate high log loss for bad predictions', async () => {
    // High confidence wrong predictions
    for (let i = 0; i < 10; i++) {
      await engine.recordResolution(
        { id: `claim-${i}`, confidence: 0.95, domain: 'test' },
        'incorrect'
      );
    }

    const report = await engine.generateReport({ type: 'global' });
    expect(report.logLoss).toBeGreaterThan(2);
  });
});

describe('Calibration Engine - Calibration Buckets', () => {
  let engine: CalibrationEngine;

  beforeEach(() => {
    engine = new CalibrationEngine({ minSampleSize: 5, bucketCount: 10 });
  });

  it('should create correct number of buckets', async () => {
    // Add data across confidence ranges
    for (let i = 0; i < 50; i++) {
      const confidence = i / 50;
      await engine.recordResolution(
        { id: `claim-${i}`, confidence, domain: 'test' },
        Math.random() < confidence ? 'correct' : 'incorrect'
      );
    }

    const report = await engine.generateReport({ type: 'global' });
    expect(report.buckets.length).toBe(10);
  });

  it('should correctly calculate bucket accuracy', async () => {
    // All claims at 0.75 confidence, 75% correct
    for (let i = 0; i < 100; i++) {
      await engine.recordResolution(
        { id: `claim-${i}`, confidence: 0.75, domain: 'test' },
        i < 75 ? 'correct' : 'incorrect'
      );
    }

    const report = await engine.generateReport({ type: 'global' });

    // Find the bucket containing 0.75
    const bucket = report.buckets.find(
      (b) => b.confidenceRange[0] <= 0.75 && b.confidenceRange[1] > 0.75
    );

    expect(bucket).toBeDefined();
    expect(bucket!.actualAccuracy).toBeCloseTo(0.75, 2);
    expect(bucket!.calibrationError).toBeLessThan(0.05); // Well calibrated
  });

  it('should detect overconfidence', async () => {
    // Claims at 90% confidence but only 60% correct
    for (let i = 0; i < 100; i++) {
      await engine.recordResolution(
        { id: `claim-${i}`, confidence: 0.9, domain: 'overconfident' },
        i < 60 ? 'correct' : 'incorrect'
      );
    }

    const report = await engine.generateReport({ type: 'domain', domain: 'overconfident' });

    // Should show positive inflation bias (stated > actual)
    expect(report.inflationBias).toBeGreaterThan(0.2);
  });

  it('should detect underconfidence', async () => {
    // Claims at 50% confidence but 80% correct
    for (let i = 0; i < 100; i++) {
      await engine.recordResolution(
        { id: `claim-${i}`, confidence: 0.5, domain: 'underconfident' },
        i < 80 ? 'correct' : 'incorrect'
      );
    }

    const report = await engine.generateReport({ type: 'domain', domain: 'underconfident' });

    // Should show negative inflation bias (stated < actual)
    expect(report.inflationBias).toBeLessThan(-0.2);
  });
});

describe('Calibration Engine - Confidence Adjustment', () => {
  let engine: CalibrationEngine;

  beforeEach(() => {
    engine = new CalibrationEngine({ minSampleSize: 5 });
  });

  it('should not adjust confidence without calibration history', async () => {
    const adjusted = await engine.adjustConfidence(0.8, 'unknown-domain');

    expect(adjusted.adjustedConfidence).toBe(0.8);
    expect(adjusted.calibrationAvailable).toBe(false);
    expect(adjusted.uncertainty).toBeGreaterThan(0.2);
  });

  it('should reduce confidence for overconfident domain', async () => {
    // Build up calibration history showing overconfidence
    for (let i = 0; i < 100; i++) {
      await engine.recordResolution(
        { id: `claim-${i}`, confidence: 0.85, domain: 'overconfident' },
        i < 60 ? 'correct' : 'incorrect' // Only 60% correct despite 85% confidence
      );
    }

    // Force report generation to populate cache
    await engine.generateReport({ type: 'domain', domain: 'overconfident' });

    const adjusted = await engine.adjustConfidence(0.85, 'overconfident');

    expect(adjusted.adjustedConfidence).toBeLessThan(0.85);
    expect(adjusted.calibrationAvailable).toBe(true);
  });

  it('should increase confidence for underconfident domain', async () => {
    // Build up calibration history showing underconfidence
    for (let i = 0; i < 100; i++) {
      await engine.recordResolution(
        { id: `claim-${i}`, confidence: 0.55, domain: 'underconfident' },
        i < 85 ? 'correct' : 'incorrect' // 85% correct despite only 55% confidence
      );
    }

    await engine.generateReport({ type: 'domain', domain: 'underconfident' });

    const adjusted = await engine.adjustConfidence(0.55, 'underconfident');

    expect(adjusted.adjustedConfidence).toBeGreaterThan(0.55);
    expect(adjusted.calibrationAvailable).toBe(true);
  });
});

describe('Calibration Engine - Credibility Gates', () => {
  let engine: CalibrationEngine;

  beforeEach(() => {
    engine = new CalibrationEngine({ minSampleSize: 5, highConfidenceErrorThreshold: 0.15 });
  });

  it('should reject high confidence when no calibration history', () => {
    const check = engine.isConfidenceCredible(0.95, 'unknown');

    expect(check.credible).toBe(false);
    expect(check.reason).toContain('No calibration history');
  });

  it('should accept well-calibrated high confidence', async () => {
    // Build well-calibrated history
    for (let i = 0; i < 100; i++) {
      await engine.recordResolution(
        { id: `claim-${i}`, confidence: 0.9, domain: 'calibrated' },
        i < 90 ? 'correct' : 'incorrect'
      );
    }

    await engine.generateReport({ type: 'domain', domain: 'calibrated' });

    const check = engine.isConfidenceCredible(0.9, 'calibrated');

    expect(check.credible).toBe(true);
  });

  it('should reject poorly-calibrated high confidence', async () => {
    // Build poorly calibrated history
    for (let i = 0; i < 100; i++) {
      await engine.recordResolution(
        { id: `claim-${i}`, confidence: 0.9, domain: 'uncalibrated' },
        i < 50 ? 'correct' : 'incorrect' // Only 50% correct at 90% confidence
      );
    }

    await engine.generateReport({ type: 'domain', domain: 'uncalibrated' });

    const check = engine.isConfidenceCredible(0.9, 'uncalibrated');

    expect(check.credible).toBe(false);
    expect(check.suggestedConfidence).toBeDefined();
    expect(check.suggestedConfidence!).toBeLessThan(0.9);
  });
});

describe('Calibration Engine - Propagation Gating', () => {
  let engine: CalibrationEngine;

  beforeEach(() => {
    engine = new CalibrationEngine({ minSampleSize: 5 });
  });

  it('should block propagation for high-confidence poorly-calibrated claims', async () => {
    // Build poorly calibrated history
    for (let i = 0; i < 100; i++) {
      await engine.recordResolution(
        { id: `claim-${i}`, confidence: 0.85, domain: 'blocked' },
        i < 40 ? 'correct' : 'incorrect'
      );
    }

    await engine.generateReport({ type: 'domain', domain: 'blocked' });

    const result = engine.canPropagate({ confidence: 0.85, domain: 'blocked' });

    expect(result.allowed).toBe(false);
  });

  it('should allow propagation for low-confidence claims even without calibration', () => {
    const result = engine.canPropagate({ confidence: 0.3, domain: 'unknown' });

    expect(result.allowed).toBe(true);
  });
});

describe('Calibration Engine - Trend Detection', () => {
  let engine: CalibrationEngine;

  beforeEach(() => {
    engine = new CalibrationEngine({ minSampleSize: 30 });
  });

  it('should detect improving calibration', async () => {
    // Early predictions: poorly calibrated
    for (let i = 0; i < 50; i++) {
      await engine.recordResolution(
        { id: `early-${i}`, confidence: 0.8, domain: 'improving' },
        i < 30 ? 'correct' : 'incorrect' // Only 60% correct
      );
    }

    // Later predictions: well calibrated
    for (let i = 0; i < 50; i++) {
      await engine.recordResolution(
        { id: `late-${i}`, confidence: 0.8, domain: 'improving' },
        i < 40 ? 'correct' : 'incorrect' // 80% correct
      );
    }

    const report = await engine.generateReport({ type: 'domain', domain: 'improving' });

    expect(report.trend).toBe('improving');
  });
});

describe('Calibration Engine - Singleton', () => {
  beforeEach(() => {
    resetCalibrationEngine();
  });

  it('should return same instance', () => {
    const engine1 = getCalibrationEngine();
    const engine2 = getCalibrationEngine();

    expect(engine1).toBe(engine2);
  });

  it('should reset correctly', async () => {
    const engine1 = getCalibrationEngine();
    await engine1.recordResolution({ id: '1', confidence: 0.5 }, 'correct');

    resetCalibrationEngine();

    const engine2 = getCalibrationEngine();
    expect(engine2.getStats().totalResolutions).toBe(0);
  });
});

describe('Calibration Engine - Error Handling', () => {
  let engine: CalibrationEngine;

  beforeEach(() => {
    engine = new CalibrationEngine({ minSampleSize: 20 });
  });

  it('should throw InsufficientCalibrationDataError when not enough data', async () => {
    // Add only 5 resolutions but require 20
    for (let i = 0; i < 5; i++) {
      await engine.recordResolution({ id: `claim-${i}`, confidence: 0.7 }, 'correct');
    }

    await expect(engine.generateReport({ type: 'global' })).rejects.toThrow(
      InsufficientCalibrationDataError
    );
  });

  it('should return null from getLatestReport when insufficient data', async () => {
    const report = await engine.getLatestReport({ type: 'global' });
    expect(report).toBeNull();
  });
});

describe('Calibration Engine - Source Type Extraction', () => {
  let engine: CalibrationEngine;

  beforeEach(() => {
    engine = new CalibrationEngine({ minSampleSize: 5 });
  });

  it('should extract AcademicPaper from arxiv sources', async () => {
    for (let i = 0; i < 10; i++) {
      await engine.recordResolution(
        { id: `claim-${i}`, confidence: 0.8, sources: ['arxiv:2301.00001'] },
        'correct'
      );
    }

    const report = await engine.generateReport({ type: 'global' });
    expect(report.reliabilityBySource.get('AcademicPaper')).toBeDefined();
  });

  it('should extract DataSet from .gov sources', async () => {
    for (let i = 0; i < 10; i++) {
      await engine.recordResolution(
        { id: `claim-${i}`, confidence: 0.8, sources: ['census.gov/data/population'] },
        'correct'
      );
    }

    const report = await engine.generateReport({ type: 'global' });
    expect(report.reliabilityBySource.get('DataSet')).toBeDefined();
  });
});
