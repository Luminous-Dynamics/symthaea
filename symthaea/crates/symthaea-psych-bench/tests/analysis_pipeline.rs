// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! E2E tests for the analysis pipeline modules.
//!
//! Exercises PsychometricReport, NormativeReport, CrossDomainMatrix,
//! SatBattery, and ReliabilityBattery on real benchmark output.

use symthaea_psych_bench::benchmarks::executive::*;
use symthaea_psych_bench::benchmarks::worm::*;
use symthaea_psych_bench::harness::{
    BenchmarkConfig, BenchmarkReport, CrossDomainMatrix, NormativeReport, PsychBenchmark,
    PsychometricReport, ReliabilityBattery, SatBattery,
};

/// Build a minimal `BenchmarkReport` from a set of benchmarks.
fn make_report(benchmarks: &[&dyn PsychBenchmark]) -> BenchmarkReport {
    let config = BenchmarkConfig {
        dimension: 128,
        trials_per_condition: 5,
        seed: 42,
        trial_trace: true,
        ..Default::default()
    };

    let mut report = BenchmarkReport::new();
    for bench in benchmarks {
        report.results.push(bench.run(&config));
    }
    report
}

#[test]
fn test_psychometric_report_from_report() {
    let stroop = StroopBenchmark;
    let flanker = FlankerBenchmark;
    let nback = NBackBenchmark;
    let benchmarks: Vec<&dyn PsychBenchmark> = vec![&stroop, &flanker, &nback];
    let report = make_report(&benchmarks);

    let psych = PsychometricReport::from_report(&report);

    // Summary fields should be finite
    assert!(psych.summary.overall_score.is_finite());
    assert!(psych.summary.n_benchmarks > 0);
    assert!(psych.summary.n_metrics > 0);
    assert!(!psych.summary.strongest_domain.is_empty());
    assert!(!psych.summary.weakest_domain.is_empty());

    // Benchmark details populated
    assert!(!psych.benchmark_details.is_empty());
    for detail in &psych.benchmark_details {
        assert!(detail.key_value.is_finite());
        assert!(!detail.key_metric.is_empty());
    }

    // Markdown report contains expected sections
    let md = psych.to_markdown();
    assert!(md.contains("Psychometric Assessment Report"));
    assert!(md.contains("Summary"));
}

#[test]
fn test_normative_report_from_report() {
    let stroop = StroopBenchmark;
    let wcst = WisconsinCardSortingBenchmark;
    let benchmarks: Vec<&dyn PsychBenchmark> = vec![&stroop, &wcst];
    let report = make_report(&benchmarks);

    let norm = NormativeReport::from_report(&report);

    // z-scores should be finite
    assert!(norm.overall_mean_z.is_finite());
    assert!(norm.overall_percentile.is_finite());
    assert!(norm.overall_percentile >= 0.0 && norm.overall_percentile <= 100.0);

    for score in &norm.scores {
        assert!(score.z_score.is_finite());
        assert!(score.percentile >= 0.0 && score.percentile <= 100.0);
        assert!(score.human_sd > 0.0);
        assert!(!score.interpretation.is_empty());
    }
}

#[test]
fn test_cross_domain_matrix_from_single_report() {
    let stroop = StroopBenchmark;
    let flanker = FlankerBenchmark;
    let wcst = WisconsinCardSortingBenchmark;
    let nback = NBackBenchmark;
    let benchmarks: Vec<&dyn PsychBenchmark> = vec![&stroop, &flanker, &wcst, &nback];
    let report = make_report(&benchmarks);

    let matrix = CrossDomainMatrix::from_single_report(&report);

    // Correlations should be finite when trace data is present
    for corr in &matrix.correlations {
        assert!(
            corr.r.is_finite(),
            "r not finite for {} vs {}",
            corr.domain_a,
            corr.domain_b
        );
        assert!(corr.n > 0);
        assert!(corr.p_value.is_finite());
    }

    // Predictions, if any, should have finite coefficients
    for pred in &matrix.predictions {
        assert!(pred.r_squared.is_finite());
    }
}

#[test]
fn test_sat_battery_run() {
    let stroop = StroopBenchmark;
    let flanker = FlankerBenchmark;
    let nback = NBackBenchmark;

    let config = BenchmarkConfig {
        dimension: 128,
        trials_per_condition: 3,
        seed: 42,
        ..Default::default()
    };
    let pressures = [0.0, 0.5, 1.0];
    let benchmarks: Vec<&dyn PsychBenchmark> = vec![&stroop, &flanker, &nback];

    let battery = SatBattery::run(&benchmarks, &config, &pressures);

    assert_eq!(battery.curves.len(), 3);
    for curve in &battery.curves {
        assert!(!curve.benchmark.is_empty());
        assert_eq!(curve.points.len(), 3);
        // Fit parameters should be finite
        assert!(curve.fit.asymptote.is_finite());
        assert!(curve.fit.rate.is_finite());
        assert!(curve.fit.intercept.is_finite());
        assert!(
            curve.fit.r_squared.is_finite(),
            "R² not finite for {}",
            curve.benchmark
        );
        // R² should be in [0,1] for a valid fit (or possibly negative for bad fits)
        // Just assert finiteness; the range can be negative for degenerate fits
    }
}

#[test]
fn test_reliability_battery_run() {
    let stroop = StroopBenchmark;
    let flanker = FlankerBenchmark;
    let nback = NBackBenchmark;

    let config = BenchmarkConfig {
        dimension: 128,
        trials_per_condition: 3,
        seed: 42,
        ..Default::default()
    };
    let benchmarks: Vec<&dyn PsychBenchmark> = vec![&stroop, &flanker, &nback];

    let battery = ReliabilityBattery::run(&benchmarks, &config, 3, 10);

    assert_eq!(battery.results.len(), 3);
    for result in &battery.results {
        assert!(!result.benchmark.is_empty());
        assert!(!result.metric.is_empty());
        // ICC should be finite and in [-1,1]
        assert!(
            result.icc.is_finite(),
            "ICC not finite for {}",
            result.benchmark
        );
        assert!(
            result.icc >= -1.0 && result.icc <= 1.0,
            "ICC {} out of range for {}",
            result.icc,
            result.benchmark
        );
        // SEM should be finite and non-negative
        assert!(result.sem.is_finite());
        assert!(result.sem >= 0.0);
    }
}
