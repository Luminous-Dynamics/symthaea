// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lightweight timing harness for `symthaea-forge`'s fitness gate.
//!
//! Deliberately not a full `criterion` benchmark: a `symthaea-forge` search
//! run may evaluate dozens of candidates, and criterion's statistical
//! sampling (warmup + many measured iterations, ~3-5s minimum per
//! function) would make that prohibitively slow. This is a simple
//! median-of-N wall-clock timer instead -- adequate for *ranking*
//! candidates against each other and against the baseline, which is all
//! the search loop needs; `entropy_methods.rs`'s real criterion benchmark
//! remains the source of truth for publishable performance numbers.
//!
//! Prints a single machine-parseable line:
//! `FORGE_BENCH_RESULT: <median_nanoseconds_per_call>`
//! which `symthaea_forge::fitness::run_benchmark` looks for.

use std::time::Instant;
use symthaea_core::consciousness_metrics::ContinuousEntropyEstimator;
use symthaea_core::hdc::unified_hv::ContinuousHV;

const DIMENSION: usize = 16_384;
const NUM_VECTORS: usize = 32;
const REPEATS: usize = 21; // odd, so the median is a single sample

fn main() {
    let estimator = ContinuousEntropyEstimator::fast();
    // Fixed seeds: every run (baseline and every candidate) measures the
    // exact same inputs, so the only thing that can change the timing is
    // the mutation itself.
    let vectors: Vec<ContinuousHV> = (0..NUM_VECTORS)
        .map(|i| ContinuousHV::random(DIMENSION, i as u64))
        .collect();

    // Warm up (page faults, branch predictor, etc.) before any measured run.
    for hv in &vectors {
        std::hint::black_box(estimator.entropy(hv));
    }

    let mut medians_ns = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let start = Instant::now();
        for hv in &vectors {
            std::hint::black_box(estimator.entropy(hv));
        }
        let elapsed = start.elapsed();
        medians_ns.push(elapsed.as_nanos() as f64 / NUM_VECTORS as f64);
    }
    medians_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = medians_ns[medians_ns.len() / 2];

    println!("FORGE_BENCH_RESULT: {median}");
}
