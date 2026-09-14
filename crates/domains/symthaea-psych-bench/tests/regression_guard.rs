// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Psych-bench regression guard: explicit direction-aware snapshot comparison.
//!
//! These tests exercise snapshot regression semantics only; they do not require
//! the `symthaea-backend` feature. Metric direction is declared explicitly in a
//! test-only manifest and is never inferred from metric names.
//!
//! To update the baseline:
//! ```text
//! UPDATE_SNAPSHOT=1 cargo test -p symthaea-psych-bench \
//!     --test regression_guard -- --ignored
//! ```

use std::collections::BTreeMap;
use symthaea_psych_bench::harness::report::MetricValue;
use symthaea_psych_bench::harness::snapshot::RegressionSnapshot;
use symthaea_psych_bench::regression_contract::{
    ComparisonDisposition, MetricComparisonPolicy, MetricKey, RegressionThresholds,
};
use symthaea_psych_bench::regression_policy_manifest::{
    MetricPolicyEntry, MetricPolicyManifest, compare_snapshots_with_manifest,
};

/// Build a minimal snapshot from hand-specified benchmark/metric/value triples.
fn synthetic_snapshot(name: &str, data: &[(&str, &str, f64)]) -> RegressionSnapshot {
    let mut metrics: BTreeMap<String, BTreeMap<String, MetricValue>> = BTreeMap::new();
    for &(bench, metric, val) in data {
        metrics.entry(bench.to_string()).or_default().insert(
            metric.to_string(),
            MetricValue {
                mean: val,
                std_dev: val.abs() * 0.05,
                n: 20,
                ci_lower: val * 0.95,
                ci_upper: val * 1.05,
            },
        );
    }
    RegressionSnapshot {
        name: name.to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        git_hash: None,
        config_summary: "regression guard test".to_string(),
        schema_version: symthaea_psych_bench::harness::snapshot::SNAPSHOT_SCHEMA_VERSION
            .to_string(),
        metrics,
    }
}

/// Explicit test-fixture policy catalog. This is not a production benchmark
/// policy manifest and grants no authority outside this regression-guard fixture.
fn regression_guard_manifest() -> MetricPolicyManifest {
    MetricPolicyManifest::new(
        "regression-guard-fixture-v1",
        1,
        vec![
            MetricPolicyEntry::new(
                MetricKey::new("NBack", "accuracy"),
                MetricComparisonPolicy::higher("guard.nback.accuracy", 1),
            ),
            MetricPolicyEntry::new(
                MetricKey::new("ChangeDetection", "accuracy"),
                MetricComparisonPolicy::higher("guard.change-detection.accuracy", 1),
            ),
            MetricPolicyEntry::new(
                MetricKey::new("Stroop", "interference"),
                MetricComparisonPolicy::lower("guard.stroop.interference", 1),
            ),
            MetricPolicyEntry::new(
                MetricKey::new("Flanker", "congruency_effect"),
                MetricComparisonPolicy::lower("guard.flanker.congruency-effect", 1),
            ),
            MetricPolicyEntry::new(
                MetricKey::new("FalseBelief", "accuracy"),
                MetricComparisonPolicy::higher("guard.false-belief.accuracy", 1),
            ),
            MetricPolicyEntry::new(
                MetricKey::new("Butlin", "composite_score"),
                MetricComparisonPolicy::higher("guard.butlin.composite-score", 1),
            ),
        ],
    )
}

fn regression_guard_thresholds() -> RegressionThresholds {
    RegressionThresholds::new(0.05, 0.10).expect("fixed regression-guard thresholds are valid")
}

fn fixture_baseline() -> RegressionSnapshot {
    synthetic_snapshot(
        "v0.5.0-baseline",
        &[
            ("NBack", "accuracy", 0.75),
            ("ChangeDetection", "accuracy", 0.80),
            ("Stroop", "interference", 0.10),
            ("Flanker", "congruency_effect", 0.12),
            ("FalseBelief", "accuracy", 0.70),
            ("Butlin", "composite_score", 0.65),
        ],
    )
}

fn fixture_current() -> RegressionSnapshot {
    synthetic_snapshot(
        "current-run",
        &[
            ("NBack", "accuracy", 0.76),
            ("ChangeDetection", "accuracy", 0.81),
            ("Stroop", "interference", 0.09),
            ("Flanker", "congruency_effect", 0.11),
            ("FalseBelief", "accuracy", 0.72),
            ("Butlin", "composite_score", 0.67),
        ],
    )
}

#[test]
fn test_direction_aware_fixture_policies() {
    let baseline = fixture_baseline();
    let current = fixture_current();
    let report = compare_snapshots_with_manifest(
        &baseline,
        &current,
        &regression_guard_manifest(),
        regression_guard_thresholds(),
    )
    .expect("fixture manifest and snapshot schemas should be valid");

    assert!(!report.has_blocking_integrity_failure());
    assert_eq!(report.summary.total_required, 6);

    let stroop = report
        .results
        .iter()
        .find(|result| result.key == MetricKey::new("Stroop", "interference"))
        .expect("Stroop result present");
    let flanker = report
        .results
        .iter()
        .find(|result| result.key == MetricKey::new("Flanker", "congruency_effect"))
        .expect("Flanker result present");
    assert_eq!(stroop.disposition, ComparisonDisposition::Pass);
    assert_eq!(flanker.disposition, ComparisonDisposition::Pass);
    assert!(stroop.delta.expect("valid delta") < 0.0);
    assert!(flanker.delta.expect("valid delta") < 0.0);
}

#[test]
#[ignore]
fn test_no_critical_regressions() {
    let snapshot_dir = RegressionSnapshot::snapshot_path();
    let baseline_path = snapshot_dir.join("v0.5.0-baseline.json");

    if std::env::var("UPDATE_SNAPSHOT").is_ok() {
        let baseline = fixture_baseline();
        std::fs::create_dir_all(&snapshot_dir).expect("create snapshot dir");
        baseline
            .save(&baseline_path)
            .expect("save baseline snapshot");
        println!("Saved baseline to {}", baseline_path.display());
        return;
    }

    if !baseline_path.exists() {
        eprintln!(
            "No baseline found at {}. Run with UPDATE_SNAPSHOT=1 to generate.",
            baseline_path.display()
        );
        return;
    }

    let baseline = RegressionSnapshot::load(&baseline_path).expect("load baseline");
    let current = fixture_current();
    let report = compare_snapshots_with_manifest(
        &baseline,
        &current,
        &regression_guard_manifest(),
        regression_guard_thresholds(),
    )
    .expect("baseline must be schema-compatible and fully covered by fixture policies");

    if report.has_blocking_integrity_failure() {
        let summary = serde_json::to_string_pretty(&report).expect("serialize regression report");
        panic!("Blocking regression/integrity failure detected!\n{summary}");
    }
}

#[test]
fn test_snapshot_round_trip() {
    let original = synthetic_snapshot(
        "round-trip-test",
        &[
            ("NBack", "accuracy", 0.80),
            ("Stroop", "interference", 0.08),
        ],
    );

    let dir = std::env::temp_dir();
    let path = dir.join("psych_bench_roundtrip_test.json");
    original.save(&path).expect("save snapshot");

    let loaded = RegressionSnapshot::load(&path).expect("load snapshot");
    assert_eq!(loaded.name, "round-trip-test");

    let report = compare_snapshots_with_manifest(
        &original,
        &loaded,
        &regression_guard_manifest(),
        regression_guard_thresholds(),
    )
    .expect("round-trip snapshot should be comparable");
    assert!(
        !report.has_blocking_integrity_failure(),
        "Round-trip comparison should have no blocking failures"
    );
    assert_eq!(report.summary.total_required, 2);
    assert!(
        report
            .results
            .iter()
            .all(|result| result.disposition == ComparisonDisposition::Pass)
    );

    let _ = std::fs::remove_file(&path);
}
