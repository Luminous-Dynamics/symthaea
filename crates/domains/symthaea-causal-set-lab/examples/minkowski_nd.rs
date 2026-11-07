// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::error::Error;
use symthaea_causal_set_lab::causal_set_nd::{CausalSetNd, expected_ordering_fraction};

fn main() -> Result<(), Box<dyn Error>> {
    let seed = 0xC0FFEE;
    let event_count = 1_500;
    let proper_time = 1.0;

    println!("Symthaea causal-set lab — flat Minkowski dimension sweep");
    println!("epistemic_status=FINITE_SPRINKLING_KINEMATICS_NOT_QUANTUM_GRAVITY");
    println!("seed={seed}");
    println!("events_per_dimension={event_count}");
    println!("proper_time={proper_time}");
    println!("D,volume,related_pairs,observed_r,expected_r,MM_dimension,longest_chain");

    for dimension in 2..=4 {
        let causet =
            CausalSetNd::sprinkle_fixed_cardinality(seed, event_count, dimension, proper_time)?;
        let observed = causet.ordering_fraction().expect("N >= 2");
        println!(
            "{dimension},{:.12},{},{:.9},{:.9},{:.6},{}",
            CausalSetNd::minkowski_interval_volume(dimension, proper_time).unwrap(),
            causet.related_pair_count(),
            observed,
            expected_ordering_fraction(dimension as f64).unwrap(),
            causet.estimate_myrheim_meyer_dimension(8.0).unwrap(),
            causet.longest_chain_length(),
        );
    }

    println!(
        "The dimension sweep checks finite-sample estimators against flat-spacetime references."
    );
    println!("It does not test curved-spacetime reconstruction or causal-set quantum dynamics.");
    Ok(())
}
