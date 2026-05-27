// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Benchmark suite for causal reasoning module.
//!
//! Tests performance of:
//! - d-separation on various graph sizes
//! - PC algorithm for causal discovery
//! - Effect estimation methods
//! - IV and mediation analysis
//!
//! Run with: cargo run --example benchmark_causal_reasoning --features reasoning_engine

#[cfg(feature = "reasoning_engine")]
use std::time::Instant;

#[cfg(feature = "reasoning_engine")]
use symthaea::consciousness::counterfactual::{
    CausalDAG, CausalQuery, CounterfactualReasoner, EffectEstimator, IVEstimator,
    MediationAnalysis, ObservationalData, PCAlgorithm, TimeSeriesCausalDiscovery, TimeSeriesData,
};

#[cfg(feature = "reasoning_engine")]
fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("                 CAUSAL REASONING BENCHMARK SUITE                  ");
    println!("═══════════════════════════════════════════════════════════════════\n");

    bench_dseparation();
    bench_pc_algorithm();
    bench_effect_estimation();
    bench_iv_estimation();
    bench_mediation_analysis();
    bench_time_series_discovery();

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                        BENCHMARK COMPLETE                         ");
    println!("═══════════════════════════════════════════════════════════════════");
}

#[cfg(not(feature = "reasoning_engine"))]
fn main() {
    println!("This benchmark requires the 'reasoning_engine' feature.");
    println!(
        "Run with: cargo run --example benchmark_causal_reasoning --features reasoning_engine"
    );
}

#[cfg(feature = "reasoning_engine")]
fn bench_dseparation() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│                    D-SEPARATION BENCHMARK                       │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    for n in [5, 10, 15, 20] {
        // Create chain DAG: 0 → 1 → 2 → ... → n-1
        let nodes: Vec<String> = (0..n).map(|i| format!("X{}", i)).collect();
        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let dag = CausalDAG::new(nodes, edges);

        let mut cond_set = std::collections::HashSet::new();
        cond_set.insert(n / 2);

        let start = Instant::now();
        let iterations = 10000;

        for _ in 0..iterations {
            let _ = dag.is_d_separated(0, n - 1, &cond_set);
        }

        let elapsed = start.elapsed();
        let per_op = elapsed.as_nanos() / iterations as u128;

        println!(
            "  DAG size {:>2} nodes: {:>6} ns/op ({} iterations)",
            n, per_op, iterations
        );
    }
    println!();
}

#[cfg(feature = "reasoning_engine")]
fn bench_pc_algorithm() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│                    PC ALGORITHM BENCHMARK                       │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    for n_vars in [3, 5, 7] {
        let n_obs = 200;

        // Generate sample data
        let mut data = ObservationalData::new((0..n_vars).map(|i| format!("X{}", i)).collect());

        for i in 0..n_obs {
            let obs: Vec<f64> = (0..n_vars)
                .map(|j| (i + j) as f64 / n_obs as f64 + (i * j % 7) as f64 / 10.0)
                .collect();
            data.add_observation(obs);
        }

        let pc = PCAlgorithm::new();

        let start = Instant::now();
        let iterations = 10;

        for _ in 0..iterations {
            let _ = pc.discover(&data);
        }

        let elapsed = start.elapsed();
        let per_op = elapsed.as_micros() / iterations as u128;

        println!(
            "  {} variables, {} observations: {:>6} µs/discovery",
            n_vars, n_obs, per_op
        );
    }
    println!();
}

#[cfg(feature = "reasoning_engine")]
fn bench_effect_estimation() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│                  EFFECT ESTIMATION BENCHMARK                    │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let dag = CausalDAG::new(
        vec!["X".into(), "Y".into(), "Z".into()],
        vec![(0, 1), (2, 0), (2, 1)],
    );

    for n_obs in [100, 500, 1000, 5000] {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into(), "Z".into()]);

        for i in 0..n_obs {
            let z = (i % 5) as f64 / 4.0;
            let x = if z + 0.1 * (i % 3) as f64 / 3.0 > 0.4 {
                1.0
            } else {
                0.0
            };
            let y = 1.5 * x + 0.8 * z + 0.02 * (i % 7) as f64 / 7.0;
            data.add_observation(vec![x, y, z]);
        }

        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };

        let estimator = EffectEstimator::new();

        let start = Instant::now();
        let iterations = 100;

        for _ in 0..iterations {
            let _ = estimator.estimate(&dag, &query, &data);
        }

        let elapsed = start.elapsed();
        let per_op = elapsed.as_micros() / iterations as u128;

        println!("  {} observations: {:>6} µs/estimation", n_obs, per_op);
    }
    println!();
}

#[cfg(feature = "reasoning_engine")]
fn bench_iv_estimation() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│                    IV ESTIMATION BENCHMARK                      │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    for n_obs in [100, 500, 1000] {
        let mut data = ObservationalData::new(vec!["Z".into(), "X".into(), "Y".into()]);

        for i in 0..n_obs {
            let z = (i % 2) as f64;
            let u = (i % 5) as f64 / 5.0;
            let x = 0.5 * z + 0.3 * u + 0.05 * (i % 7) as f64 / 7.0;
            let y = 1.5 * x + 0.4 * u + 0.05 * (i % 11) as f64 / 11.0;
            data.add_observation(vec![z, x, y]);
        }

        let start = Instant::now();
        let iterations = 1000;

        for _ in 0..iterations {
            let _ = IVEstimator::estimate_2sls(&data, 0, 1, 2);
        }

        let elapsed = start.elapsed();
        let per_op = elapsed.as_micros() / iterations as u128;

        println!(
            "  {} observations (2SLS): {:>6} µs/estimation",
            n_obs, per_op
        );
    }
    println!();
}

#[cfg(feature = "reasoning_engine")]
fn bench_mediation_analysis() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│                  MEDIATION ANALYSIS BENCHMARK                   │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let dag = CausalDAG::new(
        vec!["X".into(), "M".into(), "Y".into()],
        vec![(0, 1), (1, 2), (0, 2)],
    );

    for n_obs in [100, 500, 1000] {
        let mut data = ObservationalData::new(vec!["X".into(), "M".into(), "Y".into()]);

        for i in 0..n_obs {
            let x = (i % 2) as f64;
            let m = 0.5 * x + 0.1 * (i % 7) as f64 / 7.0;
            let y = 0.3 * x + 0.4 * m + 0.1 * (i % 11) as f64 / 11.0;
            data.add_observation(vec![x, m, y]);
        }

        let analysis = MediationAnalysis::new(&dag, 0, 1, 2);

        let start = Instant::now();
        let iterations = 1000;

        for _ in 0..iterations {
            let _ = analysis.analyze(&data);
        }

        let elapsed = start.elapsed();
        let per_op = elapsed.as_micros() / iterations as u128;

        println!("  {} observations: {:>6} µs/analysis", n_obs, per_op);
    }
    println!();
}

#[cfg(feature = "reasoning_engine")]
fn bench_time_series_discovery() {
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│               TIME-SERIES DISCOVERY BENCHMARK                   │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    for n_timepoints in [100, 500, 1000] {
        let mut data = TimeSeriesData::new(vec!["X".into(), "Y".into(), "Z".into()]);

        for t in 0..n_timepoints {
            let x = (t as f64 * 0.1).sin();
            let y = if t > 0 {
                0.7 * ((t - 1) as f64 * 0.1).sin()
            } else {
                0.0
            };
            let z = (t as f64 * 0.05).cos();
            data.add_observation(vec![x, y, z]);
        }

        let discovery = TimeSeriesCausalDiscovery::new(3);

        let start = Instant::now();
        let iterations = 10;

        for _ in 0..iterations {
            let _ = discovery.discover(&data);
        }

        let elapsed = start.elapsed();
        let per_op = elapsed.as_micros() / iterations as u128;

        println!(
            "  {} timepoints, 3 vars: {:>6} µs/discovery",
            n_timepoints, per_op
        );
    }
    println!();
}