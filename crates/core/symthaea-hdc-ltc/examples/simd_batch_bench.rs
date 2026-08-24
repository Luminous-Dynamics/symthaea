// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Engineering benchmark for repeated HDC retrieval.
//!
//! Run with:
//! `cargo run -p symthaea-hdc-ltc --example simd_batch_bench --release --features simd`
//!
//! This is not a scientific result. It compares the existing scalar query loop
//! with a candidate set prepared once using contiguous storage and cached norms.

use std::hint::black_box;
use std::time::Instant;
use symthaea_hdc_ltc::{ContinuousHV, PreparedContinuousHvSet, simd_backend};

fn main() {
    println!("backend={:?}", simd_backend());
    println!("dim,candidates,queries,scalar_ms,prepared_ms,speedup,winner_match");

    for dim in [256usize, 512, 1024, 2048, 4096, 8192, 16_384] {
        for candidate_count in [32usize, 128, 512] {
            let candidates: Vec<_> = (0..candidate_count)
                .map(|i| ContinuousHV::new_random(dim, 10_000 + i as u64))
                .collect();
            let queries: Vec<_> = (0..64)
                .map(|i| ContinuousHV::new_random(dim, 50_000 + i))
                .collect();
            let prepared = PreparedContinuousHvSet::new(&candidates);

            let scalar_started = Instant::now();
            let mut scalar_winners = Vec::with_capacity(queries.len());
            for query in &queries {
                let mut best_index = 0usize;
                let mut best_score = f32::NEG_INFINITY;
                for (index, candidate) in candidates.iter().enumerate() {
                    let score = black_box(query.similarity(black_box(candidate)));
                    if score > best_score {
                        best_score = score;
                        best_index = index;
                    }
                }
                scalar_winners.push((best_index, best_score));
            }
            let scalar_elapsed = scalar_started.elapsed();

            let prepared_started = Instant::now();
            let mut scores = vec![0.0f32; candidate_count];
            let mut prepared_winners = Vec::with_capacity(queries.len());
            for query in &queries {
                prepared.similarities_into(black_box(query), &mut scores);
                let (best_index, best_score) = scores
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .expect("candidate set is non-empty");
                prepared_winners.push((best_index, best_score));
                black_box(&scores);
            }
            let prepared_elapsed = prepared_started.elapsed();

            let winner_match = scalar_winners
                .iter()
                .zip(&prepared_winners)
                .all(|(a, b)| a.0 == b.0);
            let speedup = scalar_elapsed.as_secs_f64() / prepared_elapsed.as_secs_f64();

            println!(
                "{dim},{candidate_count},{},{:.3},{:.3},{:.3},{}",
                queries.len(),
                scalar_elapsed.as_secs_f64() * 1000.0,
                prepared_elapsed.as_secs_f64() * 1000.0,
                speedup,
                winner_match
            );
        }
    }
}
