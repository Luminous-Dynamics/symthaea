// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use clap::Parser;
use std::collections::HashMap;
use symthaea_fractal_time_lab::floquet_time_crystal::{TimeCrystalDetector, TimeCrystalSimulator};
use symthaea_fractal_time_lab::hofstadter::HofstadterGenerator;
use symthaea_fractal_time_lab::metrics::{ExperimentScorecard, scorecards_to_json_array};
use symthaea_fractal_time_lab::multiscale_phi::{
    BoxCoveringCoarseGrainer, CoarseGrainer, MultiScalePhi, SpectralCoarseGrainer,
};
use symthaea_fractal_time_lab::null_models::NullModels;
use symthaea_fractal_time_lab::report::scorecards_to_markdown_report;
use symthaea_fractal_time_lab::runner::{BenchmarkConfig, BenchmarkRun, run_benchmark_run};
use symthaea_fractal_time_lab::topological_analysis::TopologicalAnalyzer;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Run Symthaea Fractal Time Lab benchmark suite"
)]
struct Args {
    #[arg(short, long, default_value_t = 42)]
    seed: u64,
    #[arg(short, long, default_value_t = 32)]
    trials: usize,
    #[arg(long)]
    json: bool,
}

fn main() {
    let args = Args::parse();
    let config = BenchmarkConfig {
        seed: args.seed,
        trials: args.trials,
    };

    // Note: This example now runs a subset of the full suite.
    // In production, use symthaea_fractal_time_lab::run_all_benchmarks(config)
    let run = run_benchmark_run(config);
    let mut scorecards = run.scorecards;

    // --- Experiment 4: Topological Persistence ---
    {
        let analyzer = TopologicalAnalyzer;
        let graph = NullModels::hierarchical_graph(16, 2);
        let spectral = SpectralCoarseGrainer;

        let b1_orig = analyzer.betti_1_proxy(&graph);
        let g_coarse = spectral.coarse_grain(&graph).unwrap();
        let b1_coarse = analyzer.betti_1_proxy(&g_coarse);

        let primary_score = if b1_orig > 0 {
            (b1_coarse as f64) / (b1_orig as f64)
        } else {
            1.0
        };

        scorecards.push(ExperimentScorecard::new(
            "Topological Betti-1 Persistence",
            "Hierarchical graph preserves cycle-structure across spectral coarse-graining.",
            primary_score,
            &[1.0], // Toy null baseline
            1,
            args.seed,
            0.5,
            "Exploratory: Topological persistence of graph cycles.",
        ));
    }

    if args.json {
        println!("{}", scorecards_to_json_array(&scorecards));
    } else {
        let report_run = BenchmarkRun::new(config, scorecards);
        println!("{}", scorecards_to_markdown_report(&report_run));
    }
}
