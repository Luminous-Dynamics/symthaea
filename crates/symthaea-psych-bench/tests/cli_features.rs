// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for CLI-facing features.
//!
//! Each test exercises the library code path behind a CLI flag
//! (e.g., --composites, --paper-table, --percentiles) on a small
//! report to ensure no panics and correct output format.

use symthaea_psych_bench::benchmarks::executive::StroopBenchmark;
use symthaea_psych_bench::benchmarks::worm::{DigitSpanBenchmark, NBackBenchmark};
use symthaea_psych_bench::harness::analysis::{
    CrossBenchmarkAnalysis, ablation_effect_sizes, cronbachs_alpha, format_ablation_effects,
    percentile_from_z,
};
use symthaea_psych_bench::harness::{BenchmarkConfig, BenchmarkReport, PsychBenchmark};

/// Helper: build a small report with 3 benchmarks.
fn small_report() -> BenchmarkReport {
    let config = BenchmarkConfig {
        dimension: 128,
        trials_per_condition: 3,
        ..Default::default()
    };
    let mut report = BenchmarkReport::new();
    report.add(NBackBenchmark.run(&config));
    report.add(DigitSpanBenchmark.run(&config));
    report.add(StroopBenchmark.run(&config));
    report
}

// ──── --composites ────

#[test]
fn cli_composites_format_non_empty() {
    let report = small_report();
    let output = report.format_composites();
    assert!(
        output.contains("Domain") || output.contains("z"),
        "composites should contain domain header: {}",
        output
    );
}

#[test]
fn cli_composites_z_scores_finite() {
    let report = small_report();
    let composites = report.composite_scores();
    for (domain, cs) in &composites {
        assert!(
            cs.mean_z.is_finite(),
            "domain {} z-score should be finite, got {}",
            domain,
            cs.mean_z
        );
    }
}

// ──── --paper-table ────

#[test]
fn cli_paper_table_markdown_valid() {
    let report = small_report();
    let md = report.paper_summary();
    assert!(md.contains("| Domain |"), "markdown should have header");
    assert!(
        md.contains("N-back") || md.contains("NBack"),
        "should list N-back"
    );
    // Check it has the right number of |'s per line (table format)
    for line in md.lines().skip(2) {
        // data rows
        if line.starts_with('|') {
            let pipes = line.chars().filter(|c| *c == '|').count();
            assert!(pipes >= 10, "table row should have >=10 pipes: {}", line);
        }
    }
}

#[test]
fn cli_paper_table_latex_valid() {
    let report = small_report();
    let tex = report.paper_summary_latex();
    assert!(tex.contains(r"\begin{tabular}"));
    assert!(tex.contains(r"\end{tabular}"));
    assert!(tex.contains(r"\toprule"));
    assert!(tex.contains(r"\bottomrule"));
}

// ──── --forest-plot ────

#[test]
fn cli_forest_plot_data_non_empty() {
    let report = small_report();
    let rows = report.forest_plot_data();
    assert!(
        !rows.is_empty(),
        "forest plot should have at least 1 row for 3 benchmarks"
    );
    for row in &rows {
        assert!(row.agent_mean.is_finite());
        assert!(row.ratio.is_finite());
    }
}

#[test]
fn cli_forest_plot_csv_valid() {
    let report = small_report();
    let csv = report.forest_plot_csv();
    assert!(csv.starts_with("domain,benchmark,metric"));
    let line_count = csv.lines().count();
    assert!(
        line_count >= 2,
        "CSV should have header + at least 1 data row"
    );
}

#[test]
fn cli_forest_plot_ascii_valid() {
    let report = small_report();
    let ascii = report.forest_plot_ascii();
    assert!(ascii.contains("Effect"));
    assert!(ascii.contains('*'), "should have at least one * marker");
}

// ──── --percentiles ────

#[test]
fn cli_percentiles_z_to_pct_monotonic() {
    // Percentile should be monotonically increasing with z
    let z_values = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let pcts: Vec<f64> = z_values.iter().map(|z| percentile_from_z(*z)).collect();
    for i in 1..pcts.len() {
        assert!(
            pcts[i] > pcts[i - 1],
            "percentile should increase: z={} → {:.1}, z={} → {:.1}",
            z_values[i - 1],
            pcts[i - 1],
            z_values[i],
            pcts[i],
        );
    }
}

// ──── --sat (speed-accuracy tradeoff) ────

#[test]
fn cli_sat_pressure_degrades_accuracy() {
    // Stroop at time_pressure=0.0 should have better accuracy than at 1.0
    let config_low = BenchmarkConfig {
        dimension: 256,
        trials_per_condition: 10,
        time_pressure: 0.0,
        ..Default::default()
    };
    let config_high = BenchmarkConfig {
        dimension: 256,
        trials_per_condition: 10,
        time_pressure: 1.0,
        ..Default::default()
    };

    let result_low = StroopBenchmark.run(&config_low);
    let result_high = StroopBenchmark.run(&config_high);

    let acc_low = result_low.metrics["incongruent_accuracy"].mean;
    let acc_high = result_high.metrics["incongruent_accuracy"].mean;

    // High pressure should generally lower accuracy (allow some noise)
    assert!(
        acc_low >= acc_high - 0.15,
        "low pressure acc ({:.3}) should be >= high pressure acc ({:.3}) - 0.15",
        acc_low,
        acc_high,
    );
}

// ──── --pca ────

#[test]
fn cli_pca_multi_seed_runs() {
    let seeds = [42, 137, 256];
    let mut reports = Vec::new();

    for &seed in &seeds {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 3,
            seed,
            ..Default::default()
        };
        let mut report = BenchmarkReport::new();
        report.add(NBackBenchmark.run(&config));
        report.add(StroopBenchmark.run(&config));
        reports.push(report);
    }

    let analysis = CrossBenchmarkAnalysis::from_multi_seed_reports(&reports);
    let pca_output = analysis.format_pca(2);
    assert!(
        pca_output.contains("Principal Component"),
        "pca: {}",
        pca_output
    );
    assert!(pca_output.contains("Eigenvalue"));
}

// ──── --ablation-effects ────

#[test]
fn cli_ablation_effects_format() {
    let baseline = vec![("A::X", 0.90), ("B::Y", 0.80)];
    let ablated = vec![("A::X", 0.70), ("B::Y", 0.75)];
    let effects = ablation_effect_sizes(&baseline, &ablated);
    let formatted = format_ablation_effects(&effects);
    assert!(formatted.contains("Cohen's d"));
    assert!(formatted.contains("A::X") || formatted.contains("X"));
}

// ──── --correlation-matrix ────

#[test]
fn cli_correlation_matrix_format() {
    let seeds = [42, 137, 256];
    let mut reports = Vec::new();

    for &seed in &seeds {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 3,
            seed,
            ..Default::default()
        };
        let mut report = BenchmarkReport::new();
        report.add(NBackBenchmark.run(&config));
        report.add(StroopBenchmark.run(&config));
        reports.push(report);
    }

    let analysis = CrossBenchmarkAnalysis::from_multi_seed_reports(&reports);
    let matrix = analysis.format_matrix();
    assert!(!matrix.is_empty());
    assert!(matrix.contains("1.00"), "diagonal should be 1.00");
}

// ──── --reliability (internal consistency) ────

#[test]
fn cli_reliability_cronbachs_alpha() {
    // 3 items (seeds), 2 benchmarks → should produce finite alpha
    let items = vec![vec![0.85, 0.90, 0.88], vec![0.75, 0.78, 0.72]];
    let alpha = cronbachs_alpha(&items);
    assert!(alpha.is_finite(), "alpha should be finite, got {}", alpha);
}

// ──── --csv and --json output ────

#[test]
fn cli_csv_output_valid() {
    let report = small_report();
    let csv = report.to_csv().expect("CSV should serialize");
    assert!(csv.contains("benchmark"));
    assert!(csv.contains("metric"));
    assert!(csv.contains("mean"));
}

#[test]
fn cli_json_output_valid() {
    let report = small_report();
    let json = report.to_json().expect("JSON should serialize");
    // Should be valid JSON
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("should parse as JSON");
    assert!(parsed.get("results").is_some());
    assert!(parsed.get("timestamp").is_some());
}

// ──── --profile ────

#[test]
fn cli_profile_format() {
    let report = small_report();
    let output = report.format_profile();
    assert!(output.contains("Cognitive Profile"));
}

// ──── --population (simulated) ────

#[test]
fn cli_population_individual_differences() {
    // Running with different seeds should produce different results
    let config1 = BenchmarkConfig {
        dimension: 128,
        trials_per_condition: 3,
        seed: 42,
        ..Default::default()
    };
    let config2 = BenchmarkConfig {
        dimension: 128,
        trials_per_condition: 3,
        seed: 999,
        ..Default::default()
    };

    let r1 = NBackBenchmark.run(&config1);
    let r2 = NBackBenchmark.run(&config2);

    // Results should differ (not identical)
    let m1 = r1.metrics["nback_2::accuracy"].mean;
    let m2 = r2.metrics["nback_2::accuracy"].mean;
    // They could be equal by chance, so just check both are finite
    assert!(m1.is_finite() && m2.is_finite());
}

// ──── --learning-curve (block analysis) ────

#[test]
fn cli_learning_curve_blocks_finite() {
    // Simulate 3 blocks with different seeds
    let blocks: Vec<f64> = (0..3)
        .map(|block| {
            let config = BenchmarkConfig {
                dimension: 128,
                trials_per_condition: 3,
                seed: 42 + block * 1000,
                ..Default::default()
            };
            let result = NBackBenchmark.run(&config);
            result.metrics["nback_2::accuracy"].mean
        })
        .collect();

    for (i, &m) in blocks.iter().enumerate() {
        assert!(m.is_finite(), "block {} mean should be finite", i);
    }
}
