// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! End-to-end IIT 4.0 Eq. 4 cause-likelihood sweep for a fixed proper system
//! embedded in increasingly large sparse ring substrates.
//!
//! Run with:
//! cargo run --release --locked -p symthaea-iit --example eq4_sparse_background_sweep
//!
//! The candidate system is always the first four units. Only the external
//! background grows, from four to twelve units. This measures the production
//! log-domain min-fill Eq. 4 path, not the complete system-partition search.

use std::time::{Duration, Instant};

use symthaea_iit::BinaryCausalModel;

const CANDIDATE: [usize; 4] = [0, 1, 2, 3];
const SUBSTRATE_SIZES: [usize; 3] = [8, 12, 16];
const REPEATS: usize = 9;

fn ring_model(n_units: usize) -> Result<BinaryCausalModel, String> {
    if n_units <= CANDIDATE.len() || n_units > 16 {
        return Err(format!(
            "substrate size must be in {}..=16, got {n_units}",
            CANDIDATE.len() + 1
        ));
    }
    let state_count = 1usize << n_units;
    let mut tpm = Vec::with_capacity(state_count * n_units);
    let mut connectivity = vec![false; n_units * n_units];

    for target in 0..n_units {
        let previous = (target + n_units - 1) % n_units;
        connectivity[target * n_units + target] = true;
        connectivity[previous * n_units + target] = true;
    }

    for past_state in 0..state_count {
        for target in 0..n_units {
            let previous = (target + n_units - 1) % n_units;
            let self_on = ((past_state >> target) & 1) as f64;
            let previous_on = ((past_state >> previous) & 1) as f64;
            tpm.push(0.15 + 0.25 * self_on + 0.30 * previous_on);
        }
    }

    BinaryCausalModel::new(n_units, tpm, connectivity)
        .map_err(|error| format!("invalid {n_units}-unit ring fixture: {error}"))
}

fn median_duration(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn run() -> Result<(), String> {
    let current_state_for = |n_units: usize| {
        (0..n_units)
            .map(|unit| (unit * 7 + 3) % 5 < 2)
            .collect::<Vec<_>>()
    };
    println!(
        "substrate_units,candidate_units,background_units,global_state_count,candidate_state_count,repeats,median_ns,min_ns,output_sum,validated"
    );

    for n_units in SUBSTRATE_SIZES {
        let model = ring_model(n_units)?;
        let state = current_state_for(n_units);

        let warmup = model
            .forward_cause_likelihood_iit4(
                &CANDIDATE,
                &CANDIDATE,
                &state,
                &CANDIDATE,
            )
            .map_err(|error| format!("Eq4 warm-up failed at n={n_units}: {error}"))?;
        let expected_len = 1usize << CANDIDATE.len();
        if warmup.len() != expected_len
            || warmup.iter().any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(format!("invalid Eq4 warm-up result at n={n_units}"));
        }

        let mut samples = Vec::with_capacity(REPEATS);
        let mut output_sum = 0.0;
        for _ in 0..REPEATS {
            let start = Instant::now();
            let result = model
                .forward_cause_likelihood_iit4(
                    &CANDIDATE,
                    &CANDIDATE,
                    &state,
                    &CANDIDATE,
                )
                .map_err(|error| format!("Eq4 contraction failed at n={n_units}: {error}"))?;
            samples.push(start.elapsed());
            if result.len() != expected_len
                || result.iter().any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(format!("invalid Eq4 output at n={n_units}"));
            }
            let checksum = result.iter().sum::<f64>();
            if (checksum - warmup.iter().sum::<f64>()).abs() > 1e-12 {
                return Err(format!("Eq4 output changed across trials at n={n_units}"));
            }
            output_sum = checksum;
        }

        let median_ns = median_duration(&mut samples).as_nanos();
        let min_ns = samples.iter().copied().min().unwrap_or_default().as_nanos();
        println!(
            "{},{},{},{},{},{},{},{},{:.17},{}",
            n_units,
            CANDIDATE.len(),
            n_units - CANDIDATE.len(),
            1usize << n_units,
            expected_len,
            REPEATS,
            median_ns,
            min_ns,
            output_sum,
            true,
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Eq4 sparse-background benchmark failed: {error}");
        std::process::exit(1);
    }
}
