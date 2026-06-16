// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Performance benchmarks for Hodge Laplacian operations.
//!
//! Measures the cost of the key operations in the cross-scale symmetry
//! pipeline: complex construction, Laplacian computation, Betti numbers,
//! Hodge decomposition, and fraction extraction.

use std::hint::black_box;
use std::time::Instant;
use symthaea_hodge::{HodgeLaplacian, SimplicialComplex};

/// Build a random-ish Rips complex with `n` vertices and edge probability `p`.
fn make_rips_complex(n: usize, p: f64, seed: u64) -> (SimplicialComplex, Vec<f64>) {
    let mut complex = SimplicialComplex::new();
    let mut edge_signal = Vec::new();
    let mut rng_state = seed;

    for i in 0..n {
        complex.add_simplex(vec![i]);
    }

    for i in 0..n {
        for j in (i + 1)..n {
            // Simple LCG pseudo-random
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
            if r < p {
                complex.add_simplex(vec![i, j]);
                edge_signal.push(r);
                // Add triangles with lower probability
                for k in (j + 1)..n {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let r2 = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
                    if r2 < p * 0.5 {
                        complex.add_simplex(vec![i, j, k]);
                    }
                }
            }
        }
    }

    (complex, edge_signal)
}

fn bench_one(label: &str, n: usize, p: f64) {
    let (complex, edge_signal) = make_rips_complex(n, p, 42);
    let n_edges = complex.count(1);
    let n_triangles = complex.count(2);

    // Bench: HodgeLaplacian construction
    let t0 = Instant::now();
    let laplacian = HodgeLaplacian::new(complex);
    let t_construct = t0.elapsed();

    // Bench: Betti numbers
    let t1 = Instant::now();
    let betti = laplacian.betti_numbers();
    let t_betti = t1.elapsed();

    // Bench: Hodge decomposition on edge signal
    let t2 = Instant::now();
    let decomp = if n_edges >= 3 && edge_signal.len() == n_edges {
        // Center signal
        let mean = edge_signal.iter().sum::<f64>() / edge_signal.len() as f64;
        let centered: Vec<f64> = edge_signal.iter().map(|s| s - mean).collect();
        laplacian.hodge_decompose(1, &centered)
    } else {
        None
    };
    let t_decompose = t2.elapsed();

    // Bench: fraction extraction
    let t3 = Instant::now();
    let fracs = decomp.as_ref().map(|d| black_box(d.fractions()));
    let t_fractions = t3.elapsed();

    println!(
        "{:<20} n={:<4} edges={:<6} tri={:<6} β₀={} β₁={} | construct={:>8.2}ms betti={:>8.2}ms decompose={:>8.2}ms fracs={:>6.2}µs | harm={:.4}",
        label,
        n,
        n_edges,
        n_triangles,
        betti.get(0),
        betti.get(1),
        t_construct.as_secs_f64() * 1000.0,
        t_betti.as_secs_f64() * 1000.0,
        t_decompose.as_secs_f64() * 1000.0,
        t_fractions.as_secs_f64() * 1e6,
        fracs.map(|(_, _, h)| h).unwrap_or(f64::NAN),
    );
}

fn main() {
    println!("=== Hodge Laplacian Performance Benchmarks ===\n");
    println!(
        "{:<20} {:<10} {:<12} {:<12} {:>5} {:>5} | {:>12} {:>12} {:>12} {:>10} | {:>8}",
        "Label",
        "N",
        "Edges",
        "Triangles",
        "β₀",
        "β₁",
        "Construct",
        "Betti",
        "Decompose",
        "Fracs",
        "Harmonic"
    );
    println!("{}", "-".repeat(130));

    // Small complexes (typical moral topology window: 16-64 scenarios)
    bench_one("small_sparse", 8, 0.3);
    bench_one("small_dense", 8, 0.7);
    bench_one("medium_sparse", 16, 0.3);
    bench_one("medium_dense", 16, 0.7);
    bench_one("production_sparse", 32, 0.3);
    bench_one("production_dense", 32, 0.5);
    bench_one("large_sparse", 64, 0.2);
    bench_one("large_dense", 64, 0.4);

    println!("\n=== Multi-Scale Sweep Benchmark ===\n");

    // Measure full persistence-weighted sweep (10 scales)
    let n = 32;
    let num_scales = 10;
    let t_total = Instant::now();

    for s in 0..num_scales {
        let p = s as f64 / (num_scales - 1) as f64;
        let p = 0.1 + p * 0.6; // Range [0.1, 0.7]
        let (complex, signal) = make_rips_complex(n, p, 100 + s as u64);
        let n_edges = complex.count(1);
        if n_edges >= 3 && signal.len() == n_edges {
            let laplacian = HodgeLaplacian::new(complex);
            let mean = signal.iter().sum::<f64>() / signal.len() as f64;
            let centered: Vec<f64> = signal.iter().map(|v| v - mean).collect();
            if let Some(d) = laplacian.hodge_decompose(1, &centered) {
                black_box(d.fractions());
            }
        }
    }

    let sweep_time = t_total.elapsed();
    println!(
        "Full {num_scales}-scale sweep on n={n}: {:.2}ms total ({:.2}ms/scale)",
        sweep_time.as_secs_f64() * 1000.0,
        sweep_time.as_secs_f64() * 1000.0 / num_scales as f64,
    );

    // Production-size sweep
    let n = 64;
    let t_total = Instant::now();
    for s in 0..num_scales {
        let p = 0.1 + (s as f64 / (num_scales - 1) as f64) * 0.6;
        let (complex, signal) = make_rips_complex(n, p, 200 + s as u64);
        let n_edges = complex.count(1);
        if n_edges >= 3 && signal.len() == n_edges {
            let laplacian = HodgeLaplacian::new(complex);
            let mean = signal.iter().sum::<f64>() / signal.len() as f64;
            let centered: Vec<f64> = signal.iter().map(|v| v - mean).collect();
            if let Some(d) = laplacian.hodge_decompose(1, &centered) {
                black_box(d.fractions());
            }
        }
    }
    let sweep_time_64 = t_total.elapsed();
    println!(
        "Full {num_scales}-scale sweep on n={n}: {:.2}ms total ({:.2}ms/scale)",
        sweep_time_64.as_secs_f64() * 1000.0,
        sweep_time_64.as_secs_f64() * 1000.0 / num_scales as f64,
    );

    println!("\n=== Budget Assessment ===\n");
    let cycle_budget_ms = 50.0; // 20Hz = 50ms per cycle
    let cadence_min = 30;
    let amortized_ms = sweep_time.as_secs_f64() * 1000.0 / cadence_min as f64;
    println!(
        "n=32 amortized over {cadence_min} cycles: {amortized_ms:.3}ms/cycle ({:.1}% of {cycle_budget_ms}ms budget)",
        amortized_ms / cycle_budget_ms * 100.0,
    );
    let amortized_ms_64 = sweep_time_64.as_secs_f64() * 1000.0 / cadence_min as f64;
    println!(
        "n=64 amortized over {cadence_min} cycles: {amortized_ms_64:.3}ms/cycle ({:.1}% of {cycle_budget_ms}ms budget)",
        amortized_ms_64 / cycle_budget_ms * 100.0,
    );
}
