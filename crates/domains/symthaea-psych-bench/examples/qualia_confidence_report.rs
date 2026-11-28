// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Run seven theory-inspired synthetic probes and print their heuristic conformance summary.
//! This summary is not a measure or probability estimate of consciousness.
//!
//! Usage:
//!   cargo run -p symthaea-psych-bench --example qualia_confidence_report
//!   cargo run -p symthaea-psych-bench --example qualia_confidence_report -- --seed 123
//!   cargo run -p symthaea-psych-bench --example qualia_confidence_report -- --json

use symthaea_psych_bench::benchmarks::qualia_confidence::{
    BistablePerceptionBenchmark, GwtAsphyxiationBenchmark, MetacognitiveIgnitionBenchmark,
    PerturbationalComplexityBenchmark, PhaseTransitionBenchmark, QualiaConfidenceScore,
    SomaticInterferenceBenchmark, UnconsciousPrimingBenchmark,
};
use symthaea_psych_bench::harness::PsychBenchmark;
use symthaea_psych_bench::harness::config::BenchmarkConfig;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let seed = args
        .iter()
        .position(|a| a == "--seed")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(42);

    let json_output = args.iter().any(|a| a == "--json");

    let config = BenchmarkConfig {
        seed,
        ..Default::default()
    };

    eprintln!("Running theory-probe diagnostics (seed={seed})...\n");

    let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
        Box::new(GwtAsphyxiationBenchmark),
        Box::new(PhaseTransitionBenchmark),
        Box::new(PerturbationalComplexityBenchmark),
        Box::new(SomaticInterferenceBenchmark),
        Box::new(BistablePerceptionBenchmark),
        Box::new(UnconsciousPrimingBenchmark),
        Box::new(MetacognitiveIgnitionBenchmark),
    ];

    let results: Vec<_> = benchmarks.iter().map(|b| b.run(&config)).collect();
    let score = QualiaConfidenceScore::from_results(&results);

    if json_output {
        println!("{{");
        println!("  \"seed\": {seed},");
        println!("  \"summary_type\": \"theory_probe_heuristic_index\",");
        println!("  \"consciousness_measure\": false,");
        println!("  \"theory_probe_heuristic_index\": {:.4},", score.composite);
        println!("  \"heuristic_bucket\": \"{}\",", score.level);
        // Deprecated compatibility fields: retain until downstream consumers migrate.
        println!("  \"composite\": {:.4},", score.composite);
        println!("  \"level\": \"{}\",", score.level);
        println!(
            "  \"predictions_met\": {}/{}",
            score.predictions_met, score.predictions_total
        );
        println!("  \"indicators\": [");
        for (i, ind) in score.indicators.iter().enumerate() {
            let comma = if i + 1 < score.indicators.len() {
                ","
            } else {
                ""
            };
            println!(
                "    {{\"name\": \"{}\", \"score\": {:.4}, \"raw\": {:.4}, \"met\": {}}}{}",
                ind.name, ind.score, ind.raw_value, ind.prediction_met, comma
            );
        }
        println!("  ]");
        println!("}}");
    } else {
        print!("{}", score.format_report());
    }
}
