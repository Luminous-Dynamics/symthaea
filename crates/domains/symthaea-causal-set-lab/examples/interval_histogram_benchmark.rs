// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use std::time::Instant;
use symthaea_causal_set_lab::CausalSet2D;

fn checksum(histogram: &[u64]) -> u128 {
    histogram
        .iter()
        .enumerate()
        .map(|(index, count)| (index as u128 + 1) * u128::from(*count))
        .sum()
}

fn main() -> Result<(), Box<dyn Error>> {
    let sizes = [128_usize, 256, 512, 1024];
    println!("epistemic_status=EXACT_ALGORITHM_COMPARISON_NOT_A_PHYSICS_RESULT");
    println!("seed_policy=seed=0x51A7+N; proper_time=1.0");
    println!("algorithm_a=persistent_prefix_segment_tree");
    println!("algorithm_b=generic_successor_predecessor_bitset_reference");
    println!("N,related_pairs,bin_checksum,persistent_us,bitset_us,exact_match");

    for event_count in sizes {
        let causet =
            CausalSet2D::sprinkle_fixed_cardinality(0x51A7 + event_count as u64, event_count, 1.0)?;

        let start = Instant::now();
        let persistent = causet.interval_size_histogram();
        let persistent_elapsed = start.elapsed();

        let start = Instant::now();
        let bitset = causet.interval_size_histogram_bitset_reference();
        let bitset_elapsed = start.elapsed();

        let equal = persistent == bitset;
        if !equal {
            return Err(format!(
                "interval histogram mismatch at N={event_count}: persistent={:?}, bitset={:?}",
                persistent, bitset
            )
            .into());
        }

        println!(
            "{event_count},{},{},{},{},{}",
            causet.related_pair_count(),
            checksum(&persistent),
            persistent_elapsed.as_micros(),
            bitset_elapsed.as_micros(),
            equal
        );
    }

    println!(
        "complexity_note=persistent O(N^2 log N) time O(N log N) memory; bitset O(R*ceil(N/64)) interval-intersection work O(N^2) bits"
    );
    println!(
        "timing_note=single-process microbenchmark; repeat and report machine/compiler before performance claims"
    );
    Ok(())
}
