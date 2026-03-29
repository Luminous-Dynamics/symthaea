// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dose-response ablation analysis.
//!
//! Varies key parameters (WM capacity, HDC dimension, planning horizon,
//! temperature) and measures how representative benchmarks respond.
//! Computes Spearman correlation (monotonicity score) per parameter x benchmark.
//!
//! Usage:
//!   cargo run -p symthaea-psych-bench --example dose_response_ablation

use symthaea_psych_bench::benchmarks::cogbench::BartBenchmark;
use symthaea_psych_bench::benchmarks::executive::WisconsinCardSortingBenchmark;
use symthaea_psych_bench::benchmarks::tombench::FalseBeliefBenchmark;
use symthaea_psych_bench::benchmarks::worm::NBackBenchmark;
use symthaea_psych_bench::harness::analysis::spearman_correlation;
use symthaea_psych_bench::harness::report::key_metric_for_benchmark;
use symthaea_psych_bench::harness::{BenchmarkConfig, PsychBenchmark};

struct DoseParameter {
    name: &'static str,
    doses: Vec<f64>,
    apply: fn(&mut BenchmarkConfig, f64),
}

fn main() {
    let parameters = vec![
        DoseParameter {
            name: "wm_capacity",
            doses: vec![1.0, 2.0, 3.0, 5.0, 7.0],
            apply: |cfg, v| cfg.working_memory_capacity = v as usize,
        },
        DoseParameter {
            name: "hdc_dim",
            doses: vec![32.0, 64.0, 128.0, 256.0, 512.0],
            apply: |cfg, v| cfg.dimension = v as usize,
        },
        DoseParameter {
            name: "planning_horizon",
            doses: vec![0.0, 1.0, 2.0, 3.0],
            apply: |cfg, v| cfg.planning_horizon = v as usize,
        },
        DoseParameter {
            name: "temperature",
            doses: vec![0.1, 0.5, 1.0, 2.0, 5.0],
            apply: |cfg, v| cfg.action_temperature = v,
        },
    ];

    let benchmarks: Vec<Box<dyn PsychBenchmark>> = vec![
        Box::new(NBackBenchmark),
        Box::new(WisconsinCardSortingBenchmark),
        Box::new(BartBenchmark),
        Box::new(FalseBeliefBenchmark),
    ];

    eprintln!(
        "Dose-response ablation: {} parameters x {} benchmarks",
        parameters.len(),
        benchmarks.len()
    );

    // Dose-response table
    println!("\n## Dose-Response Results\n");

    for param in &parameters {
        println!("### Parameter: {}\n", param.name);
        print!("| {:12} |", "Dose");
        for bench in &benchmarks {
            let short = bench.name().split("::").last().unwrap_or(bench.name());
            print!(" {:>12} |", &short[..short.len().min(12)]);
        }
        println!();

        print!("|{:-<14}|", "");
        for _ in &benchmarks {
            print!("{:-<14}|", "");
        }
        println!();

        let mut bench_values: Vec<Vec<f64>> = vec![Vec::new(); benchmarks.len()];

        for &dose in &param.doses {
            print!("| {:>12} |", format!("{}", dose));

            for (bi, bench) in benchmarks.iter().enumerate() {
                let mut config = BenchmarkConfig {
                    dimension: 256,
                    trials_per_condition: 5,
                    ..Default::default()
                };
                (param.apply)(&mut config, dose);

                let result = bench.run(&config);
                let key = key_metric_for_benchmark(bench.name());
                let val = result.metrics.get(key).map(|m| m.mean).unwrap_or(f64::NAN);

                bench_values[bi].push(val);
                print!(" {:>12.3} |", val);
            }
            println!();
        }

        // Monotonicity summary
        println!();
        print!("| {:12} |", "Monotonicity");
        for (bi, bench) in benchmarks.iter().enumerate() {
            let vals = &bench_values[bi];
            let doses = &param.doses;
            let min_len = vals.len().min(doses.len());
            let rho = if min_len >= 3 {
                spearman_correlation(&doses[..min_len], &vals[..min_len])
            } else {
                0.0
            };
            let _ = bench; // used to iterate
            print!(" {:>12.3} |", rho);
        }
        println!("\n");
    }

    eprintln!("Done.");
}
