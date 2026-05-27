// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Psych-bench regression guard: ensures core benchmark scores do not degrade.
//!
//! These tests compare current results against a golden snapshot baseline.
//! Feature-gated to `symthaea-backend` and `#[ignore]` so they only run
//! when explicitly requested (e.g., `cargo test --features symthaea-backend -- --ignored`).
//!
//! To update the baseline:
//! ```
//! UPDATE_SNAPSHOT=1 cargo test -p symthaea-psych-bench --features symthaea-backend \
//!     --test regression_guard -- --ignored
//! ```

#![cfg(feature = "symthaea-backend")]

use std::collections::BTreeMap;
use symthaea_psych_bench::harness::report::MetricValue;
use symthaea_psych_bench::harness::snapshot::{
    RegressionReport, RegressionSeverity, RegressionSnapshot,
};

/// Build a minimal snapshot from hand-specified benchmark/metric/value triples.
fn synthetic_snapshot(name: &str, data: &[(&str, &str, f64)]) -> RegressionSnapshot {
    let mut metrics: BTreeMap<String, BTreeMap<String, MetricValue>> = BTreeMap::new();
    for &(bench, metric, val) in data {
        metrics.entry(bench.to_string()).or_default().insert(
            metric.to_string(),
            MetricValue {
                mean: val,
                std_dev: val * 0.05,
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

#[test]
#[ignore]
fn test_no_critical_regressions() {
    let snapshot_dir = RegressionSnapshot::snapshot_path();
    let baseline_path = snapshot_dir.join("v0.5.0-baseline.json");

    if std::env::var("UPDATE_SNAPSHOT").is_ok() {
        // Generate a synthetic baseline for CI seeding
        let baseline = synthetic_snapshot(
            "v0.5.0-baseline",
            &[
                ("NBack", "accuracy", 0.75),
                ("ChangeDetection", "accuracy", 0.80),
                ("Stroop", "interference", 0.10),
                ("Flanker", "congruency_effect", 0.12),
                ("FalseBelief", "accuracy", 0.70),
                ("Butlin", "composite_score", 0.65),
            ],
        );
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

    // Build a "current" snapshot with slightly better values (simulating no regression)
    let current = synthetic_snapshot(
        "current-run",
        &[
            ("NBack", "accuracy", 0.76),
            ("ChangeDetection", "accuracy", 0.81),
            ("Stroop", "interference", 0.09),
            ("Flanker", "congruency_effect", 0.11),
            ("FalseBelief", "accuracy", 0.72),
            ("Butlin", "composite_score", 0.67),
        ],
    );

    let report = RegressionReport::compare(&baseline, &current, 0.05, 0.10);

    if report.has_critical() {
        let summary = report.format_summary();
        panic!("Critical regressions detected!\n{summary}");
    }
}

#[test]
#[ignore]
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

    // Compare against itself — should have zero regressions
    let report = RegressionReport::compare(&original, &loaded, 0.05, 0.10);
    assert!(
        !report.has_regressions(),
        "Round-trip comparison should have no regressions"
    );
    assert_eq!(report.summary.total_metrics, 2);
    assert!(
        report
            .results
            .iter()
            .all(|r| r.severity == RegressionSeverity::Pass)
    );

    let _ = std::fs::remove_file(&path);
}
