// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact sparse IIT 4.0 (2023) system-search benchmark for scoped inference reuse.
//!
//! Compares the production analyzer (one uncut posterior + one set of cut
//! repertoires per partition) with a deliberately uncached exhaustive loop
//! that independently calls the single-partition API for every state pair.
//! Both paths use the same sparse local-CPT engine and the same mathematical
//! model, partition family, tie policy, precision, and work budgets.
//!
//! This is an internal algorithm comparison, not a PyPhi benchmark and not a
//! claim of full IIT conformance. The output is emitted only after every
//! per-state-pair winner and score agrees.
//!
//! Run:
//! cargo run --release --locked -p symthaea-iit --example sparse_system_search_reuse_sweep
//! cargo run --release --locked -p symthaea-iit --example sparse_system_search_reuse_sweep -- --repeats 7
//!
//! Optional:
//! --repeats N       Number of measured repetitions (default 5, range 1..=25)
//! --max-units N     Largest sparse substrate in the fixed ring sweep (default 128)

use std::time::{Duration, Instant};

use symthaea_iit::{
    enumerate_iit4_system_partitions, Iit4SystemPhiSearchConfig, Iit4SystemState,
    SparseBinaryCausalModel, SparseMechanismCpt,
};

const CANDIDATE: [usize; 4] = [0, 1, 2, 3];
const SUBSTRATE_SIZES: [usize; 5] = [8, 16, 32, 64, 128];
const MAX_FACTOR_ENTRIES: usize = 1 << 16;

#[derive(Debug)]
struct NaiveStatePairWinner {
    system_state: Iit4SystemState,
    partition_index: usize,
    score: symthaea_iit::SystemPhiPartitionScore,
}

#[derive(Debug)]
struct NaiveSearchResult {
    state_pair_winners: Vec<NaiveStatePairWinner>,
    partition_count: usize,
    state_pair_count: usize,
    partition_score_evaluations: usize,
}

fn ring_model(n_units: usize) -> Result<SparseBinaryCausalModel, String> {
    let mechanisms = (0..n_units)
        .map(|target| {
            // The first four units form a directed cycle. Unit 0 additionally
            // observes the external predecessor when a background is present.
            // This keeps the candidate strongly connected while embedding it
            // in larger sparse substrates.
            let mut parents = if target == 0 {
                let mut scope = vec![0, 3];
                if n_units > 4 {
                    scope.push(n_units - 1);
                    scope.sort_unstable();
                }
                scope
            } else {
                vec![target - 1, target]
            };
            parents.sort_unstable();
            let entries = 1usize << parents.len();
            let probabilities = (0..entries)
                .map(|assignment| {
                    let weighted = parents.iter().enumerate().fold(
                        0.0,
                        |sum, (slot, parent)| {
                            let on = ((assignment >> slot) & 1) != 0;
                            let weight = 0.07 + ((*parent * 3 + target * 5) % 7) as f64 * 0.01;
                            sum + if on { weight } else { -weight }
                        },
                    );
                    1.0 / (1.0 + (-weighted).exp())
                })
                .collect::<Vec<_>>();
            SparseMechanismCpt::new(target, parents, probabilities)
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    SparseBinaryCausalModel::new(n_units, mechanisms).map_err(|error| error.to_string())
}

fn search_config() -> Iit4SystemPhiSearchConfig {
    Iit4SystemPhiSearchConfig {
        decimal_precision: 6,
        max_partition_combinations: 2_000,
        max_state_pair_evaluations: 1_024,
        max_partition_score_evaluations: 1_000_000,
        include_total_cut: false,
    }
}

fn pyphi_eq(left: f64, right: f64, epsilon: f64) -> bool {
    if !left.is_finite() || !right.is_finite() {
        return left == right;
    }
    let tolerance = (epsilon * left.abs().max(right.abs())).max(epsilon);
    (left - right).abs() <= tolerance
}

// Mirrors the default PyPhi SIA partition ranking used by the library:
// minimum normalized phi, then maximum nonnegative phi, then cut-matrix lex.
fn select_winner(
    partitions: &[symthaea_iit::DirectedSystemPartition],
    scores: &[symthaea_iit::SystemPhiPartitionScore],
    epsilon: f64,
) -> Result<usize, String> {
    if partitions.is_empty() || partitions.len() != scores.len() {
        return Err("partition/score dimensions differ or are empty".to_string());
    }
    let minimum_normalized = scores
        .iter()
        .map(|score| score.normalized_phi)
        .min_by(f64::total_cmp)
        .ok_or_else(|| "no partition scores".to_string())?;
    let normalized_ties = scores
        .iter()
        .enumerate()
        .filter(|(_, score)| pyphi_eq(score.normalized_phi, minimum_normalized, epsilon))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let maximum_phi = normalized_ties
        .iter()
        .map(|index| scores[*index].phi)
        .max_by(f64::total_cmp)
        .ok_or_else(|| "no normalized-score ties".to_string())?;
    normalized_ties
        .into_iter()
        .filter(|index| pyphi_eq(scores[*index].phi, maximum_phi, epsilon))
        .min_by(|left, right| partitions[*left].cut_matrix().cmp(partitions[*right].cut_matrix()))
        .ok_or_else(|| "no final MIP tie candidate".to_string())
}

fn uncached_exhaustive_search(
    model: &SparseBinaryCausalModel,
    current_state: &[bool],
    config: Iit4SystemPhiSearchConfig,
) -> Result<NaiveSearchResult, String> {
    let intrinsic = model
        .system_intrinsic_specification_iit4(
            &CANDIDATE,
            current_state,
            config.decimal_precision,
            MAX_FACTOR_ENTRIES,
        )
        .map_err(|error| format!("intrinsic specification failed: {error}"))?;
    let state_pairs = intrinsic.state_pairs();
    if state_pairs.len() > config.max_state_pair_evaluations {
        return Err(format!(
            "state-pair budget exceeded: {} > {}",
            state_pairs.len(),
            config.max_state_pair_evaluations
        ));
    }
    let partitions = enumerate_iit4_system_partitions(
        &CANDIDATE,
        model.n_units(),
        config.include_total_cut,
        config.max_partition_combinations,
    )
    .map_err(|error| format!("partition enumeration failed: {error}"))?;
    let score_evaluations = partitions
        .len()
        .checked_mul(state_pairs.len())
        .ok_or_else(|| "partition-score counter overflow".to_string())?;
    if score_evaluations > config.max_partition_score_evaluations {
        return Err(format!(
            "partition-score budget exceeded: {score_evaluations} > {}",
            config.max_partition_score_evaluations
        ));
    }

    let epsilon = 10.0_f64.powi(-(config.decimal_precision as i32));
    let mut state_pair_winners = Vec::with_capacity(state_pairs.len());
    for system_state in state_pairs {
        let mut scores = Vec::with_capacity(partitions.len());
        for partition in &partitions {
            let report = model
                .score_iit4_system_partition(
                    partition,
                    current_state,
                    &system_state,
                    MAX_FACTOR_ENTRIES,
                )
                .map_err(|error| format!("uncached partition score failed: {error}"))?;
            // Enforce the same result serialization/finite-value boundary as
            // the optimized analyzer, even though this is only a benchmark.
            let _receipt = report
                .receipt_sha256()
                .map_err(|error| format!("uncached score receipt failed: {error}"))?;
            let score = report.score;
            if !score.phi.is_finite()
                || !score.signed_phi.is_finite()
                || !score.normalized_phi.is_finite()
            {
                return Err("uncached path produced a non-finite score".to_string());
            }
            scores.push(score);
        }
        let winner_index = select_winner(&partitions, &scores, epsilon)?;
        state_pair_winners.push(NaiveStatePairWinner {
            system_state,
            partition_index: winner_index,
            score: scores[winner_index].clone(),
        });
    }

    Ok(NaiveSearchResult {
        state_pair_count: state_pair_winners.len(),
        partition_count: partitions.len(),
        partition_score_evaluations: score_evaluations,
        state_pair_winners,
    })
}

fn verify_equal(
    model: &SparseBinaryCausalModel,
    current_state: &[bool],
    config: Iit4SystemPhiSearchConfig,
) -> Result<(usize, usize, usize, Option<f64>, symthaea_iit::Iit4SystemPhiStatus), String> {
    let optimized = model
        .analyze_iit4_2023_system_phi_s(
            &CANDIDATE,
            current_state,
            config,
            MAX_FACTOR_ENTRIES,
        )
        .map_err(|error| format!("optimized system search failed: {error}"))?;
    let naive = uncached_exhaustive_search(model, current_state, config)?;
    if !optimized.complete {
        return Err("optimized path returned an incomplete search".to_string());
    }
    if naive.state_pair_winners.is_empty() {
        return Err("benchmark fixture unexpectedly produced a null/no-state-pair analysis".to_string());
    }
    if optimized.state_pair_results.len() != naive.state_pair_winners.len() {
        return Err("state-pair result count mismatch".to_string());
    }
    if optimized.partition_score_evaluations != naive.partition_score_evaluations {
        return Err("partition-score work-count mismatch".to_string());
    }

    let partitions = enumerate_iit4_system_partitions(
        &CANDIDATE,
        model.n_units(),
        config.include_total_cut,
        config.max_partition_combinations,
    )
    .map_err(|error| error.to_string())?;
    let epsilon = 10.0_f64.powi(-(config.decimal_precision as i32));
    let mut maximum_phi = 0.0_f64;
    for (optimized_pair, naive_pair) in optimized
        .state_pair_results
        .iter()
        .zip(&naive.state_pair_winners)
    {
        if optimized_pair.system_state != naive_pair.system_state {
            return Err("cause/effect state-pair order mismatch".to_string());
        }
        let optimized_cut = optimized_pair.winning_partition.cut_matrix();
        let naive_cut = partitions
            .get(naive_pair.partition_index)
            .ok_or_else(|| "naive winner index outside partition set".to_string())?
            .cut_matrix();
        if optimized_cut != naive_cut {
            return Err("state-pair minimum-information partition differs".to_string());
        }
        let left = &optimized_pair.score;
        let right = &naive_pair.score;
        for (name, a, b) in [
            ("phi", left.phi, right.phi),
            ("signed_phi", left.signed_phi, right.signed_phi),
            ("normalized_phi", left.normalized_phi, right.normalized_phi),
            ("cause_phi", left.cause.signed_phi, right.cause.signed_phi),
            ("effect_phi", left.effect.signed_phi, right.effect.signed_phi),
        ] {
            if (a - b).abs() > 1e-12 {
                return Err(format!("winner {name} mismatch: optimized={a:.17}, naive={b:.17}"));
            }
        }
        maximum_phi = maximum_phi.max(left.phi);
    }
    let tied_state_pair_indices = naive
        .state_pair_winners
        .iter()
        .enumerate()
        .filter(|(_, winner)| pyphi_eq(winner.score.phi, maximum_phi, epsilon))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let (expected_status, expected_phi_s, expected_selected_index) =
        if tied_state_pair_indices.len() == 1 {
            (
                symthaea_iit::Iit4SystemPhiStatus::ExactUniqueStatePair,
                Some(maximum_phi),
                Some(tied_state_pair_indices[0]),
            )
        } else if tied_state_pair_indices
            .iter()
            .all(|index| naive.state_pair_winners[*index].score.phi == 0.0)
        {
            (
                symthaea_iit::Iit4SystemPhiStatus::UnresolvedCanonicalTie,
                Some(0.0),
                None,
            )
        } else {
            (
                symthaea_iit::Iit4SystemPhiStatus::UnresolvedCompositionTie,
                None,
                None,
            )
        };
    if optimized.status != expected_status
        || optimized.tied_state_pair_indices != tied_state_pair_indices
        || optimized.selected_state_pair_index != expected_selected_index
    {
        return Err("system-level tied-state resolution differs".to_string());
    }
    match (optimized.phi_s, expected_phi_s) {
        (Some(actual), Some(expected)) if (actual - expected).abs() <= 1e-12 => {}
        (None, None) => {}
        (actual, expected) => {
            return Err(format!("system phi_s differs: optimized={actual:?}, uncached={expected:?}"));
        }
    }

    Ok((
        naive.partition_count,
        naive.state_pair_count,
        naive.partition_score_evaluations,
        optimized.phi_s,
        optimized.status,
    ))
}

fn median_duration(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn percentile_duration(samples: &mut [Duration], percentile: f64) -> Duration {
    samples.sort_unstable();
    let rank = ((samples.len() as f64 * percentile).ceil() as usize)
        .saturating_sub(1)
        .min(samples.len() - 1);
    samples[rank]
}

fn parse_args() -> Result<(usize, usize), String> {
    let mut repeats = 5usize;
    let mut max_units = *SUBSTRATE_SIZES.last().unwrap();
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--repeats" => {
                let value = args.next().ok_or("--repeats requires a positive integer")?;
                repeats = value.parse().map_err(|_| format!("invalid repeats: {value}"))?;
                if !(1..=25).contains(&repeats) {
                    return Err("--repeats must be in 1..=25".to_string());
                }
            }
            "--max-units" => {
                let value = args.next().ok_or("--max-units requires a positive integer")?;
                max_units = value.parse().map_err(|_| format!("invalid max units: {value}"))?;
                if ![8, 16, 32, 64, 128].contains(&max_units) {
                    return Err("--max-units must be one of 8, 16, 32, 64, 128".to_string());
                }
            }
            "--help" | "-h" => {
                println!("sparse_system_search_reuse_sweep [--repeats N] [--max-units N]");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok((repeats, max_units))
}

fn run() -> Result<(), String> {
    let (repeats, max_units) = parse_args()?;
    println!(
        "substrate_units,candidate_units,partitions,state_pairs,partition_score_evaluations,repeats,naive_median_ns,naive_p95_ns,reused_median_ns,reused_p95_ns,median_speedup,uncut_posterior_preparations_naive,uncut_posterior_preparations_reused,partition_repertoire_preparations_naive,partition_repertoire_preparations_reused,phi_s,result_status,validated"
    );

    let config = search_config();
    for n_units in SUBSTRATE_SIZES.into_iter().filter(|n| *n <= max_units) {
        let model = ring_model(n_units)?;
        let current_state = (0..n_units)
            .map(|unit| (unit * 7 + 3) % 5 < 2)
            .collect::<Vec<_>>();

        // Warm both paths and verify exact same selected partition and score.
        let (partition_count, state_pair_count, score_evaluations, phi_s, result_status) =
            verify_equal(&model, &current_state, config)?;

        let mut naive_samples = Vec::with_capacity(repeats);
        let mut reused_samples = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let start = Instant::now();
            let _ = uncached_exhaustive_search(&model, &current_state, config)?;
            naive_samples.push(start.elapsed());

            let start = Instant::now();
            let result = model
                .analyze_iit4_2023_system_phi_s(
                    &CANDIDATE,
                    &current_state,
                    config,
                    MAX_FACTOR_ENTRIES,
                )
                .map_err(|error| format!("optimized search failed: {error}"))?;
            reused_samples.push(start.elapsed());
            if !result.complete {
                return Err(format!("incomplete optimized result at n={n_units}"));
            }
        }

        let naive_median = median_duration(&mut naive_samples.clone());
        let naive_p95 = percentile_duration(&mut naive_samples.clone(), 0.95);
        let reused_median = median_duration(&mut reused_samples.clone());
        let reused_p95 = percentile_duration(&mut reused_samples.clone(), 0.95);
        let speedup = if reused_median.as_nanos() == 0 {
            0.0
        } else {
            naive_median.as_nanos() as f64 / reused_median.as_nanos() as f64
        };

        // Counts follow the two loop structures, not elapsed-time inference:
        // the naive path recomputes the baseline for each partition × state pair;
        // the reused path computes the baseline once and partition terms once/cut.
        let naive_posterior_preparations = score_evaluations;
        let reused_posterior_preparations = 1usize;
        let naive_partition_terms = score_evaluations;
        let reused_partition_terms = partition_count;

        println!(
            "{},{},{},{},{},{},{},{},{},{},{:.4},{},{},{},{},{},{:?},{}",
            n_units,
            CANDIDATE.len(),
            partition_count,
            state_pair_count,
            score_evaluations,
            repeats,
            naive_median.as_nanos(),
            naive_p95.as_nanos(),
            reused_median.as_nanos(),
            reused_p95.as_nanos(),
            speedup,
            naive_posterior_preparations,
            reused_posterior_preparations,
            naive_partition_terms,
            reused_partition_terms,
            phi_s.map(|value| format!("{value:.17}")).unwrap_or_else(|| "UNRESOLVED".to_string()),
            result_status,
            true
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("sparse system-search reuse sweep failed: {error}");
        std::process::exit(1);
    }
}
