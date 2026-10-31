// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sparse local-CPT Eq. 4 scale sweep over ring substrates.
//!
//! Run with:
//! cargo run --release --locked -p symthaea-iit --example sparse_eq4_scale_sweep
//!
//! This measures the proper-system Eq. 4 cause-likelihood primitive over sparse
//! systems. It is not a full system phi_s, partition search, or PyPhi comparison.

use std::time::{Duration, Instant};

use symthaea_iit::{SparseBinaryCausalModel, SparseMechanismCpt};

const CANDIDATE: [usize; 4] = [0, 1, 2, 3];
const SUBSTRATE_SIZES: [usize; 5] = [64, 256, 512, 1_024, 2_048];
const FACTOR_ENTRY_BUDGET: usize = 1 << 16;

fn ring_model(n_units: usize) -> Result<SparseBinaryCausalModel, String> {
    let mechanisms = (0..n_units)
        .map(|target| {
            let previous = (target + n_units - 1) % n_units;
            let (parents, probabilities) = if previous < target {
                // Scope [previous, target], first parent is the low bit.
                (vec![previous, target], vec![0.15, 0.45, 0.40, 0.70])
            } else {
                // Scope [target, previous], first parent is the low bit.
                (vec![target, previous], vec![0.15, 0.40, 0.45, 0.70])
            };
            SparseMechanismCpt::new(target, parents, probabilities)
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    SparseBinaryCausalModel::new(n_units, mechanisms)
        .map_err(|error| error.to_string())
}

fn median_duration(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn parse_repeats() -> Result<usize, String> {
    let mut args = std::env::args().skip(1);
    let mut repeats = 7usize;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--repeats" => {
                let value = args.next().ok_or("--repeats requires a positive integer")?;
                repeats = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid repeat count: {value}"))?;
                if repeats == 0 || repeats > 100 {
                    return Err("--repeats must be in 1..=100".to_string());
                }
            }
            "--help" | "-h" => {
                println!("sparse_eq4_scale_sweep [--repeats N]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(repeats)
}

fn run() -> Result<(), String> {
    let repeats = parse_repeats()?;
    println!(
        "substrate_units,candidate_units,background_units,stored_local_cpt_cells,parent_references,repeats,posterior_width,target_width_max,posterior_predicted_peak_entries,target_predicted_peak_entries,predicted_peak_entries,observed_peak_entries,factor_multiplications,factor_entry_budget,median_ns,min_ns,output_sum,validated"
    );

    for n_units in SUBSTRATE_SIZES {
        let model = ring_model(n_units)?;
        let state = (0..n_units)
            .map(|unit| (unit * 7 + 3) % 5 < 2)
            .collect::<Vec<_>>();
        let stored_local_cpt_cells = (0..n_units)
            .map(|target| model.mechanism(target).unwrap().probability_table().len())
            .sum::<usize>();
        let parent_references = (0..n_units)
            .map(|target| model.mechanism(target).unwrap().parent_units().len())
            .sum::<usize>();

        let warmup = model
            .forward_cause_likelihood_iit4(&CANDIDATE, &state, FACTOR_ENTRY_BUDGET)
            .map_err(|error| format!("Eq4 warm-up failed at n={n_units}: {error}"))?;
        if warmup.forward_likelihood.len() != (1usize << CANDIDATE.len())
            || warmup.forward_likelihood.iter().any(|value| !value.is_finite() || *value < 0.0)
            || warmup.peak_factor_entries > FACTOR_ENTRY_BUDGET
            || warmup.predicted_peak_factor_entries != warmup.peak_factor_entries
        {
            return Err(format!("invalid warm-up result at n={n_units}"));
        }

        let mut samples = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let start = Instant::now();
            let report = model
                .forward_cause_likelihood_iit4(&CANDIDATE, &state, FACTOR_ENTRY_BUDGET)
                .map_err(|error| format!("Eq4 contraction failed at n={n_units}: {error}"))?;
            samples.push(start.elapsed());
            if report.forward_likelihood != warmup.forward_likelihood
                || report.posterior_induced_width != warmup.posterior_induced_width
                || report.maximum_target_induced_width != warmup.maximum_target_induced_width
                || report.posterior_predicted_peak_factor_entries != warmup.posterior_predicted_peak_factor_entries
                || report.maximum_target_predicted_peak_factor_entries != warmup.maximum_target_predicted_peak_factor_entries
                || report.predicted_peak_factor_entries != warmup.predicted_peak_factor_entries
                || report.peak_factor_entries != warmup.peak_factor_entries
                || report.factor_multiplications != warmup.factor_multiplications
            {
                return Err(format!("deterministic replay invariant failed at n={n_units}"));
            }
        }

        let median_ns = median_duration(&mut samples).as_nanos();
        let min_ns = samples.iter().copied().min().unwrap_or_default().as_nanos();
        let output_sum = warmup.forward_likelihood.iter().sum::<f64>();
        println!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.17},{}",
            n_units,
            CANDIDATE.len(),
            n_units - CANDIDATE.len(),
            stored_local_cpt_cells,
            parent_references,
            repeats,
            warmup.posterior_induced_width,
            warmup.maximum_target_induced_width,
            warmup.posterior_predicted_peak_factor_entries,
            warmup.maximum_target_predicted_peak_factor_entries,
            warmup.predicted_peak_factor_entries,
            warmup.peak_factor_entries,
            warmup.factor_multiplications,
            FACTOR_ENTRY_BUDGET,
            median_ns,
            min_ns,
            output_sum,
            output_sum.is_finite(),
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("sparse Eq4 scale sweep failed: {error}");
        std::process::exit(1);
    }
}
