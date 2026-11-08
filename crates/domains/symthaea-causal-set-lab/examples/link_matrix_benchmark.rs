// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use std::time::{Duration, Instant};
use symthaea_causal_set_lab::{CausalSet2D, LinkMatrix};

fn median_duration(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn measure<F, T>(mut operation: F) -> (T, Duration)
where
    F: FnMut() -> T,
{
    let mut durations = Vec::new();
    let mut result = None;
    for _ in 0..5 {
        let start = Instant::now();
        result = Some(operation());
        durations.push(start.elapsed());
    }
    (result.expect("five repetitions produce a result"), median_duration(durations))
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Symthaea causal-set link-matrix differential benchmark");
    println!("epistemic_status=EXACT_LINK_RELATION_ALGORITHM_COMPARISON_NOT_A_PROPAGATOR_OR_PHYSICS_RESULT");
    println!("bruteforce=direct_open_interval_scan");
    println!("optimized=causal_successor_predecessor_bitset_intersection");
    println!("timing=median_of_5_single_process_runs");
    println!("N,causal_pairs,links,two_link_paths,longest_link_path,oracle_us,bitset_us,exact_match");

    for event_count in [128_usize, 256, 512, 1024, 2048, 4096] {
        let causet =
            CausalSet2D::sprinkle_fixed_cardinality(0xCA55_0000 + event_count as u64, event_count, 3.0)?;

        let (optimized, optimized_time) =
            measure(|| LinkMatrix::from_causet_bitset(&causet).expect("within bitset budget"));

        let (oracle_time, exact_match, two_link_paths) = if event_count <= 512 {
            let (oracle, oracle_time) =
                measure(|| LinkMatrix::from_causet_bruteforce(&causet).expect("within oracle limit"));
            let exact_match = optimized == oracle;
            let two_link_paths = optimized.total_two_link_paths();
            (Some(oracle_time), exact_match, two_link_paths)
        } else {
            (None, true, 0)
        };

        if !optimized.is_acyclic() {
            return Err("optimized link matrix contains a cycle".into());
        }
        if event_count <= 512 && !exact_match {
            return Err(format!("bitset and brute-force link matrices differ at N={event_count}").into());
        }

        let related_pairs = causet.related_pair_count();
        let longest_link_path = optimized.longest_link_path_length();
        let oracle_micros = oracle_time
            .map(|duration| duration.as_micros().to_string())
            .unwrap_or_else(|| "not-run".to_string());

        println!(
            "{event_count},{related_pairs},{},{two_link_paths},{longest_link_path},{oracle_micros},{},{}",
            optimized.link_count(),
            optimized_time.as_micros(),
            if event_count <= 512 { exact_match } else { "not-compared".parse().unwrap_or(false) },
        );
    }

    println!("Brute-force differential equality is evaluated only where its declared N<=512 oracle gate permits.");
    println!("Two-link path totals are omitted for N>512 to bound the diagnostic workload.");
    Ok(())
}
