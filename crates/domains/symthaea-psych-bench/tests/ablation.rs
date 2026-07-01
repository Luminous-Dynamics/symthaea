// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Ablation test: verify that different configurations produce distinct results.

use symthaea_psych_bench::benchmarks::worm::NBackBenchmark;
use symthaea_psych_bench::harness::{AblationPreset, PsychBenchmark};

#[test]
fn ablation_full_vs_reduced_wm() {
    let full = AblationPreset::FullConsciousness.to_config(42);
    let reduced = AblationPreset::ReducedWm.to_config(42);

    let configs = [full, reduced];
    let results = NBackBenchmark.run_ablation(&configs);

    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0].config_label.as_deref(),
        Some("Full Consciousness")
    );
    assert_eq!(results[1].config_label.as_deref(), Some("Reduced WM (K=3)"));

    // Both should produce metrics
    assert!(!results[0].metrics.is_empty());
    assert!(!results[1].metrics.is_empty());

    // The two configs should produce different metric values
    // (reduced WM capacity should affect N-back performance)
    let full_metrics: Vec<f64> = results[0].metrics.values().map(|v| v.mean).collect();
    let reduced_metrics: Vec<f64> = results[1].metrics.values().map(|v| v.mean).collect();
    assert_ne!(
        full_metrics, reduced_metrics,
        "Full and reduced WM should produce different results"
    );
}

#[test]
fn ablation_all_presets_run() {
    let configs: Vec<_> = AblationPreset::all()
        .iter()
        .map(|p| p.to_config(42))
        .collect();

    let results = NBackBenchmark.run_ablation(&configs);
    assert_eq!(results.len(), 6);

    for result in &results {
        assert!(!result.metrics.is_empty());
        for (key, val) in &result.metrics {
            assert!(
                val.mean.is_finite(),
                "Preset '{}' metric '{}' is not finite",
                result.config_label.as_deref().unwrap_or("?"),
                key
            );
        }
    }
}

#[test]
fn ablation_report_json_roundtrip() {
    let config = AblationPreset::FullConsciousness.to_config(42);
    let results = NBackBenchmark.run_ablation(&[config]);

    let json = serde_json::to_string(&results).expect("serialize");
    let parsed: Vec<symthaea_psych_bench::harness::BenchmarkResult> =
        serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].benchmark, results[0].benchmark);
}
