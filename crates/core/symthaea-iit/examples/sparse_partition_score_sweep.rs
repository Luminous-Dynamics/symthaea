// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fixed-state sparse IIT4.0 (2023) directed-partition score scale sweep.
//!
//! Run with:
//! cargo run --release --locked -p symthaea-iit --example sparse_partition_score_sweep
//!
//! This evaluates one fixed partition and one fixed cause/effect state pair.
//! It is a per-partition score benchmark, not a full phi_s search or a PyPhi
//! end-to-end comparison.

use std::time::{Duration, Instant};

use symthaea_iit::{
    enumerate_iit4_system_partitions, Iit4SystemState, SparseBinaryCausalModel,
    SparseMechanismCpt,
};

const CANDIDATE: [usize; 4] = [0, 1, 2, 3];
const SUBSTRATE_SIZES: [usize; 5] = [64, 256, 512, 1_024, 2_048];
const FACTOR_ENTRY_BUDGET: usize = 1 << 16;

fn ring_model(n_units: usize) -> Result<SparseBinaryCausalModel, String> {
    let mechanisms = (0..n_units)
        .map(|target| {
            let previous = (target + n_units - 1) % n_units;
            let (parents, probabilities) = if previous < target {
                (vec![previous, target], vec![0.15, 0.45, 0.40, 0.70])
            } else {
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
    let mut repeats = 5usize;
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
                println!("sparse_partition_score_sweep [--repeats N]");
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
        "substrate_units,candidate_units,background_units,local_cpt_cells,parent_references,severed_matrix_entries,repeats,factor_entry_budget,median_ns,min_ns,phi,signed_phi,normalized_phi,cause_signed_phi,effect_signed_phi,uncut_eq4_receipt_sha256,partitioned_eq4_receipt_sha256,partition_score_receipt_sha256,validated"
    );

    for n_units in SUBSTRATE_SIZES {
        let model = ring_model(n_units)?;
        let current_state = (0..n_units)
            .map(|unit| (unit * 7 + 3) % 5 < 2)
            .collect::<Vec<_>>();
        let candidate_partitions = enumerate_iit4_system_partitions(&CANDIDATE, n_units, false, 1_000)
            .map_err(|error| format!("partition enumeration failed at n={n_units}: {error}"))?;
        let partition = candidate_partitions
            .into_iter()
            .next()
            .ok_or_else(|| format!("no directed partition fixture at n={n_units}"))?;
        let severed_entries = partition.cut_matrix().iter().filter(|cut| **cut).count();
        if severed_entries == 0 {
            return Err(format!("partition has no severed edges at n={n_units}"));
        }

        let system_state = Iit4SystemState {
            cause_specified_state: vec![true, false, true, false],
            effect_specified_state: vec![false, true, false, true],
        };
        let uncut_cause = model
            .forward_cause_likelihood_iit4(&CANDIDATE, &current_state, FACTOR_ENTRY_BUDGET)
            .map_err(|error| format!("uncut Eq4 failed at n={n_units}: {error}"))?;
        let cut_cause = model
            .partitioned_forward_cause_likelihood_iit4(
                &partition,
                &current_state,
                FACTOR_ENTRY_BUDGET,
            )
            .map_err(|error| format!("partitioned Eq4 failed at n={n_units}: {error}"))?;
        let warmup_score = model
            .score_iit4_system_partition(
                &partition,
                &current_state,
                &system_state,
                FACTOR_ENTRY_BUDGET,
            )
            .map_err(|error| format!("warm-up score failed at n={n_units}: {error}"))?;

        let mut samples = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let start = Instant::now();
            let score = model
                .score_iit4_system_partition(
                    &partition,
                    &current_state,
                    &system_state,
                    FACTOR_ENTRY_BUDGET,
                )
                .map_err(|error| format!("partition score failed at n={n_units}: {error}"))?;
            samples.push(start.elapsed());
            if score != warmup_score {
                return Err(format!("deterministic score replay failed at n={n_units}"));
            }
        }

        let local_cpt_cells = (0..n_units)
            .map(|target| model.mechanism(target).unwrap().probability_table().len())
            .sum::<usize>();
        let parent_references = (0..n_units)
            .map(|target| model.mechanism(target).unwrap().parent_units().len())
            .sum::<usize>();
        let median_ns = median_duration(&mut samples).as_nanos();
        let min_ns = samples.iter().copied().min().unwrap_or_default().as_nanos();
        let uncut_receipt = uncut_cause
            .receipt_sha256()
            .map_err(|error| format!("uncut Eq4 receipt failed at n={n_units}: {error}"))?;
        let partitioned_receipt = cut_cause
            .receipt_sha256()
            .map_err(|error| format!("partitioned Eq4 receipt failed at n={n_units}: {error}"))?;
        let score_receipt = warmup_score
            .receipt_sha256()
            .map_err(|error| format!("partition score receipt failed at n={n_units}: {error}"))?;
        let validated = score_is_finite(&warmup_score.score)
            && uncut_cause.forward_likelihood.iter().all(|value| value.is_finite())
            && cut_cause.forward_likelihood.iter().all(|value| value.is_finite())
            && uncut_receipt.len() == 64
            && partitioned_receipt.len() == 64
            && score_receipt.len() == 64;
        println!(
            "{},{},{},{},{},{},{},{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{},{},{},{}",
            n_units,
            CANDIDATE.len(),
            n_units - CANDIDATE.len(),
            local_cpt_cells,
            parent_references,
            severed_entries,
            repeats,
            FACTOR_ENTRY_BUDGET,
            median_ns,
            min_ns,
            warmup_score.score.phi,
            warmup_score.score.signed_phi,
            warmup_score.score.normalized_phi,
            warmup_score.score.cause.signed_phi,
            warmup_score.score.effect.signed_phi,
            uncut_receipt,
            partitioned_receipt,
            score_receipt,
            validated
        );
        if !validated {
            return Err(format!("non-finite/unvalidated output at n={n_units}"));
        }
    }
    Ok(())
}

fn score_is_finite(score: &symthaea_iit::SystemPhiPartitionScore) -> bool {
    [
        score.phi,
        score.signed_phi,
        score.normalization_factor,
        score.signed_normalized_phi,
        score.normalized_phi,
        score.cause.selectivity,
        score.cause.forward_probability,
        score.cause.partitioned_forward_probability,
        score.cause.signed_phi,
        score.effect.selectivity,
        score.effect.forward_probability,
        score.effect.partitioned_forward_probability,
        score.effect.signed_phi,
    ]
    .iter()
    .all(|value| value.is_finite())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("sparse partition-score sweep failed: {error}");
        std::process::exit(1);
    }
}
