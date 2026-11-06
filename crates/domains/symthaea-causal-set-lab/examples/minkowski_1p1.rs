// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_causal_set_lab::CausalSet2D;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let seed = 0xC0FFEE;
    let event_count = 1_500;
    let proper_time = 1.0;
    let causet = CausalSet2D::sprinkle_fixed_cardinality(seed, event_count, proper_time)?;

    println!("Symthaea causal-set lab — 1+1D Minkowski kinematics");
    println!("epistemic_status=CONDITIONAL_SPRINKLING_KINEMATICS_NOT_QUANTUM_GRAVITY");
    println!("seed={seed}");
    println!("events={}", causet.len());
    println!("interval_volume={:.12}", CausalSet2D::minkowski_interval_volume_1p1(proper_time).unwrap());
    println!("related_pairs={}", causet.related_pair_count());
    println!("ordering_fraction={:.8}", causet.ordering_fraction().unwrap());
    println!("myrheim_meyer_dimension={:.6}", causet.estimate_myrheim_meyer_dimension(8.0).unwrap());
    println!("longest_chain={}", causet.longest_chain_length());
    println!("longest_chain_over_sqrt_n={:.6}", causet.longest_chain_length() as f64 / (causet.len() as f64).sqrt());

    let boosted = causet.lorentz_boosted(1.7)?;
    println!("boost_rapidity=1.7");
    println!("boosted_related_pairs={}", boosted.related_pair_count());
    println!("boosted_longest_chain={}", boosted.longest_chain_length());

    Ok(())
}
