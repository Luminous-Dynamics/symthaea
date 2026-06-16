// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration test: verify the full HTML report pipeline produces a valid,
//! complete report containing all sections.
//!
//! Runs a small set of benchmarks, generates the HTML report, and asserts
//! structural properties (sections present, valid HTML, no panics).

use symthaea_psych_bench::benchmarks::executive::*;
use symthaea_psych_bench::benchmarks::inhibition::*;
use symthaea_psych_bench::benchmarks::sustained_attention::*;
use symthaea_psych_bench::benchmarks::worm::*;
use symthaea_psych_bench::harness::html;
use symthaea_psych_bench::harness::report::BenchmarkReport;
use symthaea_psych_bench::harness::{BenchmarkConfig, PsychBenchmark};

#[test]
fn test_html_full_pipeline() {
    let config = BenchmarkConfig {
        dimension: 128,
        trials_per_condition: 3,
        seed: 42,
        ..Default::default()
    };

    // Run a representative set of benchmarks across multiple domains
    let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
        Box::new(StroopBenchmark),
        Box::new(FlankerBenchmark),
        Box::new(DualTaskBenchmark),
        Box::new(NBackBenchmark),
        Box::new(DigitSpanBenchmark),
        Box::new(GoNoGoBenchmark),
        Box::new(SartBenchmark),
        Box::new(PvtBenchmark),
    ];

    let mut report = BenchmarkReport::new();
    for bench in &benchmarks {
        report.add(bench.run(&config));
    }

    let html_output = html::generate_report(&report, false);

    // Structural validity
    assert!(html_output.contains("<!DOCTYPE html>"), "missing DOCTYPE");
    assert!(html_output.contains("</html>"), "missing closing html tag");
    assert!(html_output.contains("<table>"), "missing tables");

    // All expected sections present
    assert!(
        html_output.contains("Cognitive Profile"),
        "missing cognitive profile section"
    );
    assert!(
        html_output.contains("Results by Domain"),
        "missing domain tables section"
    );
    assert!(
        html_output.contains("Effect Size Forest Plot"),
        "missing forest plot section"
    );
    assert!(
        html_output.contains("Composite Scores"),
        "missing composite scores section"
    );
    assert!(
        html_output.contains("Psychometric Summary"),
        "missing psychometric summary section"
    );
    assert!(
        html_output.contains("Normative Comparison"),
        "missing normative comparison section"
    );

    // Benchmark names should appear in domain tables
    assert!(html_output.contains("Stroop"), "missing Stroop in output");
    assert!(html_output.contains("Flanker"), "missing Flanker in output");
    assert!(html_output.contains("N-back"), "missing N-back in output");

    // SVG charts should be present
    assert!(html_output.contains("<svg"), "missing SVG elements");
    assert!(html_output.contains("<circle"), "missing SVG circles");

    // Footer present
    assert!(
        html_output.contains("symthaea-psych-bench"),
        "missing footer attribution"
    );
}

#[test]
fn test_html_reliability_section_integration() {
    use symthaea_psych_bench::harness::html::write_reliability_section;
    use symthaea_psych_bench::harness::reliability_analysis::ReliabilityBattery;

    let config = BenchmarkConfig {
        dimension: 128,
        trials_per_condition: 3,
        seed: 42,
        ..Default::default()
    };

    let benchmarks: Vec<Box<dyn PsychBenchmark>> =
        vec![Box::new(StroopBenchmark), Box::new(FlankerBenchmark)];
    let bench_refs: Vec<&dyn PsychBenchmark> = benchmarks.iter().map(|b| b.as_ref()).collect();

    let battery = ReliabilityBattery::run(&bench_refs, &config, 3, 10);

    let mut html = String::new();
    write_reliability_section(&mut html, &battery);

    assert!(
        html.contains("Test-Retest Reliability"),
        "missing reliability heading"
    );
    assert!(html.contains("ICC"), "missing ICC column");
    assert!(html.contains("Stroop"), "missing Stroop in reliability");
    assert!(html.contains("Flanker"), "missing Flanker in reliability");
}
