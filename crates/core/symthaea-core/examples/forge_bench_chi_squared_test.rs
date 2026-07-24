// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lightweight timing harness for `symthaea-forge`'s fitness gate, second
//! real target after `entropy_histogram` -- `chi_squared_test` has more
//! structural variety (a closure with a conditional, comparisons, and
//! arithmetic in a Wilson-Hilferty normal approximation) and, unlike
//! `entropy_histogram`, no already-hand-optimized sibling implementation
//! competing for the same job, so it's a fairer test of the widened
//! mutation operator set (comparison/arithmetic/boolean swaps + literal
//! perturbation).
//!
//! Same median-of-N wall-clock design as `forge_bench_entropy_histogram.rs`
//! -- see that file's docs for why this isn't a full `criterion` benchmark.
//! Prints `FORGE_BENCH_RESULT: <median_nanoseconds_per_call>`.

use std::time::Instant;
use symthaea_core::hdc::statistics::chi_squared_test;

const N_CATEGORIES: usize = 12;
const REPEATS: usize = 51; // odd, so the median is a single sample

fn main() {
    // Fixed input: every run (baseline and every candidate) measures the
    // exact same data, so the only thing that can change the timing (or
    // correctness, if a mutation slips past the compile gate but produces
    // a wrong-but-plausible result) is the mutation itself.
    let observed: Vec<f64> = (0..N_CATEGORIES).map(|i| 40.0 + (i as f64 * 3.7)).collect();
    let expected: Vec<f64> = vec![50.0; N_CATEGORIES];

    // Warm up.
    for _ in 0..1000 {
        std::hint::black_box(chi_squared_test(
            std::hint::black_box(&observed),
            std::hint::black_box(&expected),
            std::hint::black_box(0.05),
        ));
    }

    let mut medians_ns = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let start = Instant::now();
        for _ in 0..1000 {
            std::hint::black_box(chi_squared_test(
                std::hint::black_box(&observed),
                std::hint::black_box(&expected),
                std::hint::black_box(0.05),
            ));
        }
        let elapsed = start.elapsed();
        medians_ns.push(elapsed.as_nanos() as f64 / 1000.0);
    }
    medians_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = medians_ns[medians_ns.len() / 2];

    println!("FORGE_BENCH_RESULT: {median}");
}
