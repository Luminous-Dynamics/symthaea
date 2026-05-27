// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Substrate Prediction Validation Runner
//!
//! Executes all testable predictions against empirical psych-bench results
//! and reports evidence level changes.
//!
//! ```bash
//! cargo run --example validate_substrate_predictions
//! ```

use symthaea_core::hdc::substrate_validation::SubstrateValidationFramework;
use symthaea_psych_bench::benchmarks::substrate::validation::SubstrateValidationBenchmark;
use symthaea_psych_bench::harness::PsychBenchmark;
use symthaea_psych_bench::harness::config::BenchmarkConfig;
use symthaea_psych_bench::harness::report::MetricValue;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("     SUBSTRATE PREDICTION VALIDATION — Phase 5");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Snapshot evidence levels before validation
    let before = SubstrateValidationFramework::new();
    println!("EVIDENCE LEVELS (before validation):");
    for summary in before.prediction_summary() {
        println!(
            "  {:<12} {:?} (confidence: {:.2}, {} predictions)",
            summary.substrate,
            summary.evidence_level,
            summary.honest_confidence,
            summary.total_predictions,
        );
    }
    println!();

    // Run the validation benchmark
    println!("Running empirical benchmarks...");
    let bench = SubstrateValidationBenchmark;
    let config = BenchmarkConfig::default();
    let result = bench.run(&config);

    // Print results
    println!();
    println!("PREDICTION RESULTS:");
    println!("───────────────────────────────────────────────────────────────");
    for line in result.notes.lines() {
        println!("  {}", line);
    }
    println!();

    // Summary metrics
    let tested = match result.metrics.get("predictions_tested") {
        Some(MetricValue::Score(n)) => *n as usize,
        _ => 0,
    };
    let passed = match result.metrics.get("predictions_passed") {
        Some(MetricValue::Score(n)) => *n as usize,
        _ => 0,
    };
    let pass_rate = match result.metrics.get("pass_rate") {
        Some(MetricValue::Float(r)) => *r,
        _ => 0.0,
    };

    println!("SUMMARY:");
    println!("  Predictions tested: {}", tested);
    println!("  Predictions passed: {}", passed);
    println!("  Pass rate: {:.1}%", pass_rate * 100.0);
    println!("  Overall: {}", if result.passed { "PASS" } else { "FAIL" });
    println!();

    // Evidence level changes
    println!("EVIDENCE LEVELS (after validation):");
    for key in result.metrics.keys() {
        if key.ends_with("_evidence_level") {
            let substrate = key.trim_end_matches("_evidence_level");
            let level = match result.metrics.get(key) {
                Some(MetricValue::Label(l)) => l.as_str(),
                _ => "Unknown",
            };
            let confidence = match result
                .metrics
                .get(&format!("{}_honest_confidence", substrate))
            {
                Some(MetricValue::Float(c)) => *c,
                _ => 0.0,
            };

            // Find before level
            let before_level = before
                .prediction_summary()
                .iter()
                .find(|s| s.substrate == substrate)
                .map(|s| format!("{:?}", s.evidence_level))
                .unwrap_or_default();

            let changed = before_level != level;
            println!(
                "  {:<12} {} → {} (confidence: {:.2}) {}",
                substrate,
                before_level,
                level,
                confidence,
                if changed { "⬆ UPGRADED" } else { "" },
            );
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("NOTE: Evidence upgrades are internal to this run. Higher levels");
    println!("(Experimental, Validated) require external replication and peer");
    println!("review — they cannot be reached by internal benchmarks alone.");
    println!("═══════════════════════════════════════════════════════════════");
}