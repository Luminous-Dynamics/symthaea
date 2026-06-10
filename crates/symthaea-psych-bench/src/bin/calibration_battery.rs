// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Minimal 9-benchmark calibration battery.
//!
//! Runs the benchmark subset needed for full 9-transmitter calibration,
//! computes NormativeReport, and emits JSON z-scores to stdout.
//!
//! Usage:
//!   calibration_battery [--seed N] [--trials N]
//!
//! Output (stdout):
//!   [{"benchmark":"Executive::Stroop","metric":"stroop_effect","z_score":-0.42}, ...]

use symthaea_psych_bench::benchmarks::executive::{
    DualTaskBenchmark, FlankerBenchmark, StroopBenchmark,
};
use symthaea_psych_bench::benchmarks::inhibition::{GoNoGoBenchmark, StopSignalBenchmark};
use symthaea_psych_bench::benchmarks::social::{RmeBenchmark, UltimatumGameBenchmark};
use symthaea_psych_bench::benchmarks::sustained_attention::{CptBenchmark, PvtBenchmark};
use symthaea_psych_bench::benchmarks::worm::NBackBenchmark;
use symthaea_psych_bench::harness::{
    BenchmarkConfig, BenchmarkReport, NormativeReport, PsychBenchmark,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse --seed and --trials from CLI
    let seed = args
        .windows(2)
        .find(|w| w[0] == "--seed")
        .and_then(|w| w[1].parse::<u64>().ok())
        .unwrap_or(42);

    let trials = args
        .windows(2)
        .find(|w| w[0] == "--trials")
        .and_then(|w| w[1].parse::<usize>().ok())
        .unwrap_or(5);

    let config = BenchmarkConfig {
        seed,
        trials_per_condition: trials,
        ..Default::default()
    };

    // The 10-benchmark calibration battery:
    // DA: Stroop + Flanker (interference)
    // ACh: N-back (working memory)
    // NE: StopSignal + GoNoGo (inhibition)
    // 5-HT: CPT (sustained attention)
    // GABA: DualTask (executive multitasking)
    // Oxytocin: UltimatumGame + RME (social cognition)
    // Glutamate: PVT (lapse_rate)
    // Adenosine: PVT (vigilance_decrement)
    // ECB: no direct benchmark — derived from allostatic self-assessment
    let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
        Box::new(StroopBenchmark),
        Box::new(FlankerBenchmark),
        Box::new(NBackBenchmark),
        Box::new(StopSignalBenchmark),
        Box::new(GoNoGoBenchmark),
        Box::new(CptBenchmark),
        Box::new(DualTaskBenchmark),
        Box::new(UltimatumGameBenchmark),
        Box::new(RmeBenchmark),
        Box::new(PvtBenchmark),
    ];

    let mut report = BenchmarkReport::new();
    for bench in &benchmarks {
        let result = bench.run(&config);
        report.add(result);
    }

    let normative = NormativeReport::from_report(&report);

    // Emit JSON z-scores
    #[derive(serde::Serialize)]
    struct ZScoreEntry {
        benchmark: String,
        metric: String,
        z_score: f64,
    }

    let entries: Vec<ZScoreEntry> = normative
        .scores
        .iter()
        .map(|s| ZScoreEntry {
            benchmark: s.benchmark.clone(),
            metric: s.metric.clone(),
            z_score: s.z_score,
        })
        .collect();

    let json = serde_json::to_string(&entries).expect("JSON serialization should not fail");
    println!("{json}");
}
