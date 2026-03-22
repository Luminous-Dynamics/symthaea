// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the external benchmark harness and psych-bench infrastructure.
//!
//! Tests cover:
//! - External benchmark manifest parsing
//! - Running psych-bench subsets and validating output
//! - Regression snapshot loading and validation
//! - Report generation, forest plots, and composite scores
//! - Ravens Progressive Matrices (fluid intelligence)
//! - ARC fluid reasoning benchmarks

use std::collections::BTreeMap;
use std::path::Path;

use symthaea_psych_bench::benchmarks::attention::AttentionalBlinkBenchmark;
use symthaea_psych_bench::benchmarks::executive::{
    FlankerBenchmark, RavensProgressiveMatricesBenchmark, StroopBenchmark,
    WisconsinCardSortingBenchmark,
};
use symthaea_psych_bench::benchmarks::inhibition::GoNoGoBenchmark;
use symthaea_psych_bench::benchmarks::metacognition::MetacognitiveCalibrationBenchmark;
use symthaea_psych_bench::benchmarks::reasoning::ArcFluidBenchmark;
use symthaea_psych_bench::benchmarks::tombench::FalseBeliefBenchmark;
use symthaea_psych_bench::benchmarks::worm::{DigitSpanBenchmark, NBackBenchmark};
use symthaea_psych_bench::harness::baselines::BaselineCollection;
use symthaea_psych_bench::harness::snapshot::{
    RegressionReport, RegressionSnapshot, SNAPSHOT_SCHEMA_VERSION,
};
use symthaea_psych_bench::harness::{
    BenchmarkConfig, BenchmarkReport, BenchmarkResult, PsychBenchmark,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Minimal config for fast integration tests (low dim, few trials).
fn integration_config() -> BenchmarkConfig {
    BenchmarkConfig {
        dimension: 256,
        trials_per_condition: 5,
        seed: 42,
        ..Default::default()
    }
}

/// Assert all metrics in a result are finite and the result is non-empty.
fn assert_valid_result(result: &BenchmarkResult) {
    assert!(
        !result.metrics.is_empty(),
        "{}: expected at least one metric",
        result.benchmark
    );
    for (key, val) in &result.metrics {
        assert!(
            val.mean.is_finite(),
            "{}: metric '{}' mean is not finite: {}",
            result.benchmark,
            key,
            val.mean
        );
        assert!(
            val.std_dev.is_finite(),
            "{}: metric '{}' std_dev is not finite: {}",
            result.benchmark,
            key,
            val.std_dev
        );
        assert!(
            val.ci_lower.is_finite(),
            "{}: metric '{}' ci_lower is not finite",
            result.benchmark,
            key,
        );
        assert!(
            val.ci_upper.is_finite(),
            "{}: metric '{}' ci_upper is not finite",
            result.benchmark,
            key,
        );
    }
}

// ---------------------------------------------------------------------------
// 1. Benchmark manifest parsing
// ---------------------------------------------------------------------------

/// Represents the structure of `benchmarks/manifest.json`.
#[derive(serde::Deserialize)]
struct Manifest {
    version: u32,
    description: String,
    benchmarks: Vec<ManifestBenchmark>,
}

#[derive(serde::Deserialize)]
struct ManifestBenchmark {
    id: String,
    name: String,
    #[serde(rename = "type")]
    bench_type: String,
    root: String,
    run: String,
    ready: String,
    results: String,
    #[allow(dead_code)]
    local_only: bool,
}

#[test]
fn test_benchmark_manifest_parses() {
    // Walk up from crate root to find the repo-level manifest.json
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo_root = crate_dir
        .parent() // crates/
        .and_then(|p| p.parent()) // symthaea/
        .expect("cannot find repo root from CARGO_MANIFEST_DIR");
    let manifest_path = repo_root.join("benchmarks").join("manifest.json");

    if !manifest_path.exists() {
        // In CI environments the benchmarks directory might not exist.
        eprintln!(
            "Skipping test_benchmark_manifest_parses: {} does not exist",
            manifest_path.display()
        );
        return;
    }

    let contents = std::fs::read_to_string(&manifest_path).expect("failed to read manifest.json");
    let manifest: Manifest =
        serde_json::from_str(&contents).expect("failed to parse manifest.json");

    // Structural assertions
    assert_eq!(manifest.version, 1, "expected manifest version 1");
    assert!(
        !manifest.description.is_empty(),
        "manifest description should be non-empty"
    );
    assert!(
        !manifest.benchmarks.is_empty(),
        "manifest should list at least one benchmark"
    );

    // Verify each entry has required fields populated
    for bench in &manifest.benchmarks {
        assert!(!bench.id.is_empty(), "benchmark id should be non-empty");
        assert!(!bench.name.is_empty(), "benchmark name should be non-empty");
        assert_eq!(
            bench.bench_type, "external",
            "all manifest benchmarks should be type 'external'"
        );
        assert!(
            !bench.root.is_empty(),
            "benchmark root path should be non-empty"
        );
        assert!(
            !bench.run.is_empty(),
            "benchmark run script should be non-empty"
        );
        assert!(
            !bench.ready.is_empty(),
            "benchmark ready marker should be non-empty"
        );
        assert!(
            !bench.results.is_empty(),
            "benchmark results path should be non-empty"
        );
    }

    // Verify known benchmark IDs are present
    let ids: Vec<&str> = manifest.benchmarks.iter().map(|b| b.id.as_str()).collect();
    assert!(
        ids.contains(&"arc-agi-2"),
        "manifest should contain arc-agi-2, found: {:?}",
        ids
    );
    assert!(
        ids.contains(&"gaia-dev"),
        "manifest should contain gaia-dev, found: {:?}",
        ids
    );
}

// ---------------------------------------------------------------------------
// 2. Psych-bench runs a subset and produces valid MetricValues
// ---------------------------------------------------------------------------

#[test]
fn test_psych_bench_runs_subset() {
    let config = integration_config();

    // Run 3 representative benchmarks from different domains
    let stroop_result = StroopBenchmark.run(&config);
    let go_nogo_result = GoNoGoBenchmark.run(&config);
    let flanker_result = FlankerBenchmark.run(&config);

    // Validate all three produce valid metrics
    assert_valid_result(&stroop_result);
    assert_valid_result(&go_nogo_result);
    assert_valid_result(&flanker_result);

    // Domain-specific assertions
    assert!(
        stroop_result.metrics.contains_key("stroop_effect"),
        "Stroop result should contain 'stroop_effect' metric"
    );
    assert!(
        go_nogo_result.metrics.contains_key("nogo_accuracy"),
        "GoNoGo result should contain 'nogo_accuracy' metric"
    );
    assert!(
        flanker_result.metrics.contains_key("flanker_effect"),
        "Flanker result should contain 'flanker_effect' metric"
    );

    // Stroop effect should be non-negative (incongruent is harder than congruent)
    let stroop_effect = stroop_result.metrics["stroop_effect"].mean;
    assert!(
        stroop_effect >= 0.0,
        "stroop_effect should be non-negative, got {}",
        stroop_effect
    );

    // GoNoGo accuracy should be in [0, 1]
    let nogo_acc = go_nogo_result.metrics["nogo_accuracy"].mean;
    assert!(
        (0.0..=1.0).contains(&nogo_acc),
        "nogo_accuracy should be in [0,1], got {}",
        nogo_acc
    );
}

// ---------------------------------------------------------------------------
// 3. Regression snapshot loading
// ---------------------------------------------------------------------------

#[test]
fn test_psych_bench_regression_snapshot_loads() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let baselines_dir = crate_dir.join("baselines");

    // Try to load the latest baseline snapshot
    let snapshot_path = baselines_dir.join("v0.6.0.json");
    if !snapshot_path.exists() {
        // Fall back to v0.5.2 or any available version
        let fallback = baselines_dir.join("v0.5.2.json");
        if !fallback.exists() {
            eprintln!(
                "Skipping test_psych_bench_regression_snapshot_loads: no baseline files found"
            );
            return;
        }
        let snapshot =
            RegressionSnapshot::load(&fallback).expect("failed to load fallback baseline snapshot");
        assert!(
            !snapshot.name.is_empty(),
            "snapshot name should be non-empty"
        );
        assert!(
            !snapshot.metrics.is_empty(),
            "snapshot should contain benchmark metrics"
        );
        return;
    }

    let snapshot =
        RegressionSnapshot::load(&snapshot_path).expect("failed to load v0.6.0 baseline snapshot");

    // Structural validation
    assert!(
        !snapshot.name.is_empty(),
        "snapshot name should be non-empty"
    );
    assert!(
        !snapshot.metrics.is_empty(),
        "snapshot should contain benchmark metrics"
    );
    assert_eq!(
        snapshot.schema_version, SNAPSHOT_SCHEMA_VERSION,
        "snapshot schema version should match current"
    );

    // Validate the snapshot passes structural integrity checks
    snapshot
        .validate()
        .expect("baseline snapshot should pass validation");

    // Verify expected benchmark domains are present
    let bench_names: Vec<&str> = snapshot.metrics.keys().map(|k| k.as_str()).collect();
    let has_executive = bench_names
        .iter()
        .any(|n| n.contains("Executive") || n.contains("Stroop"));
    let has_worm = bench_names
        .iter()
        .any(|n| n.contains("WorM") || n.contains("NBack") || n.contains("N-back"));
    assert!(
        has_executive || has_worm,
        "baseline should contain Executive or WorM benchmarks, found: {:?}",
        bench_names
    );

    // Verify at least 10 benchmark entries (the full battery has 40+)
    assert!(
        snapshot.metrics.len() >= 10,
        "baseline should have at least 10 benchmarks, found {}",
        snapshot.metrics.len()
    );

    // Verify all metric values in the snapshot are finite
    for (bench, metrics) in &snapshot.metrics {
        for (metric_name, val) in metrics {
            assert!(
                val.mean.is_finite(),
                "snapshot {}/{}: mean is not finite",
                bench,
                metric_name
            );
            assert!(
                val.std_dev.is_finite(),
                "snapshot {}/{}: std_dev is not finite",
                bench,
                metric_name
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Report generation
// ---------------------------------------------------------------------------

#[test]
fn test_psych_bench_report_generation() {
    let config = integration_config();
    let mut report = BenchmarkReport::new();

    // Run benchmarks from 3 different domains
    report.add(StroopBenchmark.run(&config));
    report.add(NBackBenchmark.run(&config));
    report.add(FalseBeliefBenchmark.run(&config));

    // Report should contain all 3 results
    assert_eq!(
        report.results.len(),
        3,
        "report should contain exactly 3 results"
    );

    // Timestamp should be populated
    assert!(
        !report.timestamp.is_empty(),
        "report timestamp should be non-empty"
    );

    // Summary should be non-empty and contain benchmark names
    let summary = report.summary();
    assert!(!summary.is_empty(), "report summary should be non-empty");
    assert!(
        summary.contains("Stroop"),
        "summary should mention Stroop benchmark"
    );
    assert!(
        summary.contains("N-back") || summary.contains("NBack"),
        "summary should mention N-back benchmark"
    );

    // JSON serialization should work
    let json = report.to_json().expect("report should serialize to JSON");
    assert!(!json.is_empty(), "JSON output should be non-empty");
    assert!(
        json.contains("Stroop"),
        "JSON should contain benchmark names"
    );

    // CSV serialization should work
    let csv = report.to_csv().expect("report should serialize to CSV");
    assert!(!csv.is_empty(), "CSV output should be non-empty");
    assert!(
        csv.contains("benchmark"),
        "CSV should contain header with 'benchmark'"
    );

    // Paper summary should contain table headers
    let paper = report.paper_summary();
    assert!(
        paper.contains("% Human"),
        "paper summary should have '% Human' header"
    );

    // Cognitive profile should produce domain scores
    let profile = report.cognitive_profile();
    // At least some domains should have baseline comparisons
    // (Stroop effect has baselines, NBack has baselines)
    if !profile.is_empty() {
        for (domain, score) in &profile {
            assert!(
                *score >= 0.0 && *score <= 1.0,
                "profile score for {} out of [0,1] range: {}",
                domain,
                score
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Forest plot data
// ---------------------------------------------------------------------------

#[test]
fn test_psych_bench_forest_plot_data() {
    let config = integration_config();
    let mut report = BenchmarkReport::new();

    // Run benchmarks that have known baselines
    report.add(StroopBenchmark.run(&config));
    report.add(NBackBenchmark.run(&config));
    report.add(DigitSpanBenchmark.run(&config));
    report.add(GoNoGoBenchmark.run(&config));

    let forest_rows = report.forest_plot_data();

    // Should have rows (all 4 benchmarks have baselines)
    assert!(
        !forest_rows.is_empty(),
        "forest plot should have at least one row"
    );

    for row in &forest_rows {
        // All numeric fields should be finite
        assert!(
            row.agent_mean.is_finite(),
            "agent_mean should be finite for {}",
            row.benchmark
        );
        assert!(
            row.human_mean.is_finite(),
            "human_mean should be finite for {}",
            row.benchmark
        );
        assert!(
            row.ci_lower.is_finite(),
            "ci_lower should be finite for {}",
            row.benchmark
        );
        assert!(
            row.ci_upper.is_finite(),
            "ci_upper should be finite for {}",
            row.benchmark
        );
        assert!(
            row.ratio.is_finite(),
            "ratio should be finite for {}",
            row.benchmark
        );

        // Cohen's d should be finite when present
        if let Some(d) = row.cohens_d {
            assert!(
                d.is_finite(),
                "cohens_d should be finite for {}",
                row.benchmark
            );
        }

        // z-score should be finite when present
        if let Some(z) = row.z_score {
            assert!(
                z.is_finite(),
                "z_score should be finite for {}",
                row.benchmark
            );
        }

        // Domain and benchmark should be non-empty strings
        assert!(!row.domain.is_empty(), "domain should be non-empty");
        assert!(!row.benchmark.is_empty(), "benchmark should be non-empty");
        assert!(!row.metric.is_empty(), "metric should be non-empty");
    }

    // Forest plot CSV should be well-formed
    let csv = report.forest_plot_csv();
    assert!(
        csv.starts_with("domain,benchmark,metric"),
        "CSV should start with header row"
    );

    // ASCII forest plot should contain visual elements
    let ascii = report.forest_plot_ascii();
    assert!(
        ascii.contains("Effect"),
        "ASCII plot should contain 'Effect' header"
    );
}

// ---------------------------------------------------------------------------
// 6. Composite scores
// ---------------------------------------------------------------------------

#[test]
fn test_psych_bench_composite_scores() {
    let config = integration_config();
    let mut report = BenchmarkReport::new();

    // Run enough benchmarks across domains to produce composite scores.
    // We need benchmarks with z-score-enabled baselines (baselines with SD).
    report.add(StroopBenchmark.run(&config));
    report.add(FlankerBenchmark.run(&config));
    report.add(WisconsinCardSortingBenchmark.run(&config));
    report.add(NBackBenchmark.run(&config));
    report.add(DigitSpanBenchmark.run(&config));
    report.add(GoNoGoBenchmark.run(&config));
    report.add(FalseBeliefBenchmark.run(&config));
    report.add(MetacognitiveCalibrationBenchmark.run(&config));
    report.add(AttentionalBlinkBenchmark.run(&config));

    let composites = report.composite_scores();

    // At least some domains should have composite scores
    // (Executive has Stroop/Flanker/WCST with SD baselines)
    assert!(
        !composites.is_empty(),
        "should have at least one domain composite score"
    );

    for (domain, cs) in &composites {
        // z-score should be finite
        assert!(
            cs.mean_z.is_finite(),
            "composite z-score for {} should be finite, got {}",
            domain,
            cs.mean_z
        );

        // n_benchmarks should be >= 1
        assert!(
            cs.n_benchmarks >= 1,
            "composite for {} should have at least 1 benchmark, got {}",
            domain,
            cs.n_benchmarks
        );

        // Benchmark names should be non-empty
        assert!(
            !cs.benchmarks.is_empty(),
            "composite for {} should list contributing benchmarks",
            domain
        );

        // z-scores should be in a reasonable range (typically -5 to +5)
        assert!(
            cs.mean_z > -10.0 && cs.mean_z < 10.0,
            "composite z-score for {} is extreme: {}",
            domain,
            cs.mean_z
        );
    }

    // Format composites should produce non-empty output
    let formatted = report.format_composites();
    assert!(
        !formatted.is_empty(),
        "formatted composites should be non-empty"
    );
    assert!(
        formatted.contains("z"),
        "formatted composites should reference z-scores"
    );
}

// ---------------------------------------------------------------------------
// 7. Ravens Progressive Matrices
// ---------------------------------------------------------------------------

#[test]
fn test_ravens_matrices_runs() {
    let config = BenchmarkConfig {
        dimension: 512,
        trials_per_condition: 5,
        seed: 42,
        ..Default::default()
    };

    let result = RavensProgressiveMatricesBenchmark.run(&config);
    assert_valid_result(&result);

    // Ravens should produce overall_accuracy
    assert!(
        result.metrics.contains_key("overall_accuracy"),
        "Ravens should have 'overall_accuracy' metric, found: {:?}",
        result.metrics.keys().collect::<Vec<_>>()
    );

    let overall_accuracy = result.metrics["overall_accuracy"].mean;
    assert!(
        overall_accuracy >= 0.0 && overall_accuracy <= 1.0,
        "overall_accuracy should be in [0,1], got {}",
        overall_accuracy
    );
    assert!(
        overall_accuracy.is_finite(),
        "overall_accuracy should be finite"
    );

    // Should also have difficulty tier breakdowns
    assert!(
        result.metrics.contains_key("easy_accuracy"),
        "Ravens should have 'easy_accuracy' metric"
    );

    let easy_acc = result.metrics["easy_accuracy"].mean;
    assert!(
        easy_acc >= 0.0 && easy_acc <= 1.0,
        "easy_accuracy should be in [0,1], got {}",
        easy_acc
    );

    // Easy accuracy should generally be >= overall accuracy (by design)
    // Use a soft check with tolerance since this is stochastic with few trials
    assert!(
        easy_acc >= overall_accuracy - 0.3,
        "easy_accuracy ({}) should not be much lower than overall_accuracy ({})",
        easy_acc,
        overall_accuracy
    );

    // Verify provenance is present
    let prov = RavensProgressiveMatricesBenchmark.provenance();
    assert!(prov.is_some(), "Ravens should have provenance metadata");
    let prov = prov.unwrap();
    assert!(
        prov.paradigm.contains("Raven") || prov.paradigm.contains("Progressive"),
        "Ravens provenance paradigm should reference Raven/Progressive, got: {}",
        prov.paradigm
    );
}

// ---------------------------------------------------------------------------
// 8. ARC Fluid Reasoning benchmark
// ---------------------------------------------------------------------------

#[test]
fn test_arc_fluid_reasoning_runs() {
    let config = BenchmarkConfig {
        dimension: 256,
        trials_per_condition: 3,
        seed: 42,
        ..Default::default()
    };

    let result = ArcFluidBenchmark.run(&config);
    assert_valid_result(&result);

    // ARC Fluid should produce transfer_accuracy (primary metric)
    assert!(
        result.metrics.contains_key("transfer_accuracy"),
        "ARC Fluid should have 'transfer_accuracy' metric, found: {:?}",
        result.metrics.keys().collect::<Vec<_>>()
    );

    let transfer_acc = result.metrics["transfer_accuracy"].mean;
    assert!(
        transfer_acc >= 0.0 && transfer_acc <= 1.0,
        "transfer_accuracy should be in [0,1], got {}",
        transfer_acc
    );

    // Should also have rule_consistency
    assert!(
        result.metrics.contains_key("rule_consistency"),
        "ARC Fluid should have 'rule_consistency' metric"
    );

    let rule_consistency = result.metrics["rule_consistency"].mean;
    assert!(
        rule_consistency >= 0.0 && rule_consistency <= 1.0,
        "rule_consistency should be in [0,1], got {}",
        rule_consistency
    );

    // Verify provenance
    let prov = ArcFluidBenchmark.provenance();
    assert!(prov.is_some(), "ARC Fluid should have provenance metadata");
}

// ---------------------------------------------------------------------------
// 9. Baseline comparisons are structurally correct
// ---------------------------------------------------------------------------

#[test]
fn test_baseline_collection_structural_integrity() {
    let bl = BaselineCollection::all();

    // Verify each baseline map is non-empty
    let maps: Vec<(&str, &BTreeMap<&str, _>)> = vec![
        ("worm", &bl.worm),
        ("cogbench", &bl.cogbench),
        ("tombench", &bl.tombench),
        ("memory_agent", &bl.memory_agent),
        ("executive", &bl.executive),
        ("metacognition", &bl.metacognition),
        ("affect", &bl.affect),
        ("creativity", &bl.creativity),
        ("butlin", &bl.butlin),
        ("inhibition", &bl.inhibition),
        ("attention", &bl.attention),
        ("reasoning", &bl.reasoning),
    ];

    for (name, map) in &maps {
        assert!(
            !map.is_empty(),
            "baseline collection '{}' should be non-empty",
            name
        );

        // Verify all baseline values are finite
        for (key, baseline) in *map {
            assert!(
                baseline.value.is_finite(),
                "baseline {}/{}: value should be finite, got {}",
                name,
                key,
                baseline.value
            );
            if let Some(sd) = baseline.sd {
                assert!(
                    sd.is_finite() && sd >= 0.0,
                    "baseline {}/{}: sd should be finite and non-negative, got {}",
                    name,
                    key,
                    sd
                );
            }
            assert!(
                !baseline.source.is_empty(),
                "baseline {}/{}: source should be non-empty",
                name,
                key
            );
            assert!(
                !baseline.population.is_empty(),
                "baseline {}/{}: population should be non-empty",
                name,
                key
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 10. Regression comparison across two snapshots
// ---------------------------------------------------------------------------

#[test]
fn test_regression_comparison_workflow() {
    let config = integration_config();
    let mut report = BenchmarkReport::new();

    // Run a few benchmarks to build a baseline snapshot
    report.add(StroopBenchmark.run(&config));
    report.add(NBackBenchmark.run(&config));

    let baseline_snapshot = RegressionSnapshot::from_report(&report, "baseline-test");
    assert!(baseline_snapshot.validate().is_ok());

    // Run the same benchmarks with identical config (deterministic seed)
    let mut report2 = BenchmarkReport::new();
    report2.add(StroopBenchmark.run(&config));
    report2.add(NBackBenchmark.run(&config));

    let current_snapshot = RegressionSnapshot::from_report(&report2, "current-test");

    // Compare: with identical seeds, there should be zero regressions
    let regression = RegressionReport::compare(&baseline_snapshot, &current_snapshot, 0.05, 0.10);

    assert_eq!(
        regression.baseline_name, "baseline-test",
        "regression report should reference baseline name"
    );
    assert_eq!(
        regression.current_name, "current-test",
        "regression report should reference current name"
    );
    assert!(
        regression.summary.total_metrics > 0,
        "regression report should have compared at least one metric"
    );
    assert!(
        !regression.has_critical(),
        "identical-seed runs should produce no critical regressions"
    );
    assert!(
        !regression.has_regressions(),
        "identical-seed runs should produce no regressions"
    );

    // Format summary should be non-empty and indicate no regressions
    let summary = regression.format_summary();
    assert!(
        summary.contains("No regressions"),
        "summary should indicate no regressions: {}",
        summary
    );
}

// ---------------------------------------------------------------------------
// 11. Snapshot JSON roundtrip from live benchmarks
// ---------------------------------------------------------------------------

#[test]
fn test_snapshot_json_roundtrip_from_live_run() {
    let config = integration_config();
    let mut report = BenchmarkReport::new();
    report.add(StroopBenchmark.run(&config));

    let snapshot = RegressionSnapshot::from_report(&report, "roundtrip-test")
        .with_git_hash("test-hash-abc123".to_string());

    // Serialize to JSON
    let json = snapshot
        .to_json()
        .expect("snapshot should serialize to JSON");

    // Deserialize back
    let loaded =
        RegressionSnapshot::from_json(&json).expect("snapshot should deserialize from JSON");

    // Verify round-trip fidelity
    assert_eq!(loaded.name, "roundtrip-test");
    assert_eq!(loaded.git_hash.as_deref(), Some("test-hash-abc123"));
    assert_eq!(loaded.schema_version, SNAPSHOT_SCHEMA_VERSION);
    assert_eq!(loaded.metrics.len(), snapshot.metrics.len());

    // Verify metric values survived the round-trip
    for (bench, bench_metrics) in &snapshot.metrics {
        let loaded_metrics = loaded
            .metrics
            .get(bench)
            .unwrap_or_else(|| panic!("missing benchmark '{}' after roundtrip", bench));
        for (metric_name, val) in bench_metrics {
            let loaded_val = loaded_metrics.get(metric_name).unwrap_or_else(|| {
                panic!("missing metric '{}/{}' after roundtrip", bench, metric_name)
            });
            assert!(
                (val.mean - loaded_val.mean).abs() < 1e-10,
                "mean mismatch for {}/{}: {} vs {}",
                bench,
                metric_name,
                val.mean,
                loaded_val.mean
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 12. Multiple benchmarks with baseline comparisons
// ---------------------------------------------------------------------------

#[test]
fn test_find_comparisons_across_domains() {
    let config = integration_config();
    let bl = BaselineCollection::all();
    let report = BenchmarkReport::new();

    // Run benchmarks from several domains and verify comparisons exist
    let stroop = StroopBenchmark.run(&config);
    let nback = NBackBenchmark.run(&config);
    let digit = DigitSpanBenchmark.run(&config);

    let stroop_comps = report.find_comparisons(&stroop, &bl);
    let nback_comps = report.find_comparisons(&nback, &bl);
    let digit_comps = report.find_comparisons(&digit, &bl);

    // Stroop should have comparisons (stroop_effect has baseline)
    assert!(
        !stroop_comps.is_empty(),
        "Stroop should have baseline comparisons"
    );

    // NBack should have comparisons (nback_2::accuracy has baseline)
    assert!(
        !nback_comps.is_empty(),
        "NBack should have baseline comparisons"
    );

    // DigitSpan should have comparisons (forward_span has baseline)
    assert!(
        !digit_comps.is_empty(),
        "DigitSpan should have baseline comparisons"
    );

    // Verify comparison fields are populated correctly
    for (metric_key, comp) in &stroop_comps {
        assert!(
            comp.human_value.is_finite(),
            "human_value should be finite for metric '{}'",
            metric_key
        );
        assert!(
            comp.ratio.is_finite(),
            "ratio should be finite for metric '{}'",
            metric_key
        );
        assert!(
            !comp.source.is_empty(),
            "source should be non-empty for metric '{}'",
            metric_key
        );
        assert!(
            !comp.population.is_empty(),
            "population should be non-empty for metric '{}'",
            metric_key
        );
    }
}
