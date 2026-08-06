// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phase 2.2C-iii-a (`SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`): instrumentation-only
//! trace of the FEP replay's internal state, to check whether Phase 2.2C-i's unexplained
//! "trait-shaped channels score worse than constant/noise" finding is the *same*
//! `PrecisionEstimator` long-constant-signal runaway this exact research program already found
//! and confirmed twice on the `ecological` family (`rung5_convergence_probe.rs` and its
//! independent 2026-07-27 reconfirmation at a different `sample_size`) -- a slowly-drifting
//! trait signal is close to "long-constant" for long stretches, so this is a high-prior
//! hypothesis to check *before* any model-construction change (precision ablation, diagonal
//! coupling), not a new guess.
//!
//! **Deliberately makes NO changes to model behavior** -- this file reimplements
//! `FepCensusOnlyGenerator`/`FepCensusPlusTraitGenerator`'s exact replay loop directly (same
//! normalization, same masking, same `perceive`/`observe_transition` calls) rather than modifying
//! those already-committed, already-tested rung types, so nothing about Phase 2.2C-i's or Phase
//! 2.2B-ii's results is touched. It only adds print statements at chosen tick milestones during
//! one continuous replay through a single seed's recorded history.
//!
//! Five arms traced: `census_1d` (the 1D reference), and four state_dim=2 arms sharing the
//! *same* population channel construction as Phase 2.2C-i's ablation: `constant`,
//! `duplicate_census`, `real_noisy_trait`, `privileged_trait` (shuffled/independent-noise
//! omitted here -- the goal is diagnosing *why* trait-shaped content hurts, and shuffled/noise
//! were already shown in Phase 2.2C-i to sit close to constant/duplicate, so they're lower
//! priority for this specific mechanistic question).
//!
//! Prerequisite: `cargo run --release --example evolutionary_rescue_generate_worlds -p
//! symthaea-futures-ensemble` first (reuses the same fixtures as Phase 2.2C-i).
//!
//! Run: `cargo run --release --example evolutionary_rescue_precision_trace -p symthaea-futures-ensemble`

#[path = "support/evolutionary_rescue_common.rs"]
mod common;

use common::SerializedWorld;
use symthaea_futures_ensemble::evolutionary_rescue::{TRAIT_CEILING, TRAIT_FLOOR};
use symthaea_futures_state::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation, mask_observation,
};
use symthaea_futures_symtropy::evolutionary_rescue::EvolutionaryRescueObservation;

/// A handful of representative test seeds -- tracing all 5 would be noisy to read; these three
/// span the observed first_extinction_tick range (earliest, middle, latest among the 5 test
/// seeds: 66=10700, 99=10680 earliest, 111=10723 latest).
const TRACE_SEEDS: [u64; 3] = [99, 66, 111];

/// Ticks (as a fraction of each world's own first_extinction_tick) at which to print a snapshot
/// -- fractions rather than absolute ticks so the milestones land at comparable *relative* points
/// in each seed's own trajectory despite slightly different extinction timing.
const SNAPSHOT_FRACTIONS: [f64; 6] = [0.05, 0.25, 0.5, 0.75, 0.9, 0.99];

fn normalize_trait(v: f64) -> f64 {
    ((v - TRAIT_FLOOR) / (TRAIT_CEILING - TRAIT_FLOOR)).clamp(0.0, 1.0)
}

fn matrix_diag_offdiag_abs_mean(m: &[Vec<f64>]) -> (f64, f64) {
    let mut diag = Vec::new();
    let mut offdiag = Vec::new();
    for (i, row) in m.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if i == j {
                diag.push(v.abs());
            } else {
                offdiag.push(v.abs());
            }
        }
    }
    let mean = |v: &[f64]| {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    (mean(&diag), mean(&offdiag))
}

/// Replays one arm's history through a fresh agent, printing a snapshot of internal state at
/// each tick in `snapshot_ticks`. `second_channel` is `None` for the 1D reference.
fn trace_arm(
    label: &str,
    history: &[EvolutionaryRescueObservation],
    second_channel: Option<&dyn Fn(usize, &EvolutionaryRescueObservation) -> f64>,
    snapshot_ticks: &[u64],
) {
    let state_dim = if second_channel.is_some() { 2 } else { 1 };
    let config = ActiveInferenceAgentConfig {
        state_dim,
        obs_dim: state_dim,
        num_actions: 1,
        ..ActiveInferenceAgentConfig::default()
    };
    let mut agent = ActiveInferenceAgent::new(config);

    let reference = history
        .first()
        .and_then(|o| o.sample)
        .map(|s| s.sampled_alive_count as f64)
        .filter(|&r| r > 0.0)
        .unwrap_or(1.0);

    let mut prev_belief = agent.belief.clone();
    let mut next_snapshot_idx = 0usize;

    println!("\n-- arm: {label} ({state_dim}D) --");
    for (i, obs) in history.iter().enumerate() {
        let pop_value = obs
            .sample
            .map(|s| (s.sampled_alive_count as f64 / reference).clamp(0.0, 1.0))
            .unwrap_or(0.5);
        let pop_visible = if obs.sample.is_some() { 1.0 } else { 0.0 };

        let (values, visibility) = match second_channel {
            None => (vec![pop_value], vec![pop_visible]),
            Some(f) => {
                let raw = f(i, obs);
                let second_value = normalize_trait(raw);
                (vec![pop_value, second_value], vec![pop_visible, 1.0])
            }
        };

        let raw_obs = Observation::new(values, 1.0, "trace");
        let masked = mask_observation(&raw_obs, &agent.belief, &visibility);
        agent.perceive(&masked);
        let new_belief = agent.belief.clone();
        agent.observe_transition(&prev_belief, 0, &new_belief, &masked);
        prev_belief = new_belief;

        if next_snapshot_idx < snapshot_ticks.len() && obs.tick >= snapshot_ticks[next_snapshot_idx]
        {
            let (lik_diag, lik_offdiag) =
                matrix_diag_offdiag_abs_mean(&agent.model.likelihood_matrix);
            let (trans_diag, trans_offdiag) =
                matrix_diag_offdiag_abs_mean(&agent.model.transition_matrices[0]);
            println!(
                "  tick={:6}  belief.mean={:?}  belief.precision={:?}  \
                 sensory_precision={:.4}  prior_precision={:.4}  \
                 likelihood[diag|offdiag]={:.4}|{:.4}  transition[diag|offdiag]={:.4}|{:.4}",
                obs.tick,
                agent
                    .belief
                    .mean
                    .iter()
                    .map(|v| format!("{v:.4}"))
                    .collect::<Vec<_>>(),
                agent
                    .belief
                    .precision
                    .iter()
                    .map(|v| format!("{v:.4}"))
                    .collect::<Vec<_>>(),
                agent.precision.sensory_precision,
                agent.precision.prior_precision,
                lik_diag,
                lik_offdiag,
                trans_diag,
                trans_offdiag,
            );
            next_snapshot_idx += 1;
        }
    }
}

fn main() {
    println!("== Phase 2.2C-iii-a: precision/transition-matrix instrumentation trace ==");
    println!(
        "(no model-behavior changes -- read-only replay, reimplemented directly, not via the rung crate)\n"
    );

    for &seed in &TRACE_SEEDS {
        let world: SerializedWorld = common::load_world(seed);
        let last_tick = world
            .first_extinction_tick
            .unwrap_or(world.trajectory.len() as u64 - 1);
        let snapshot_ticks: Vec<u64> = SNAPSHOT_FRACTIONS
            .iter()
            .map(|&f| ((last_tick as f64) * f) as u64)
            .collect();

        println!(
            "=== seed={seed} first_extinction_tick={:?} snapshot_ticks={snapshot_ticks:?} ===",
            world.first_extinction_tick
        );

        let history = &world.census_obs[..=(last_tick as usize).min(world.census_obs.len() - 1)];

        trace_arm("census_1d", history, None, &snapshot_ticks);

        let midpoint = (TRAIT_FLOOR + TRAIT_CEILING) / 2.0;
        trace_arm("constant", history, Some(&|_, _| midpoint), &snapshot_ticks);

        let reference = history
            .first()
            .and_then(|o| o.sample)
            .map(|s| s.sampled_alive_count as f64)
            .filter(|&r| r > 0.0)
            .unwrap_or(1.0);
        trace_arm(
            "duplicate_census",
            history,
            Some(&move |_, obs| {
                let pop_fraction = obs
                    .sample
                    .map(|s| (s.sampled_alive_count as f64 / reference).clamp(0.0, 1.0))
                    .unwrap_or(0.5);
                TRAIT_FLOOR + pop_fraction * (TRAIT_CEILING - TRAIT_FLOOR)
            }),
            &snapshot_ticks,
        );

        let noisy_history = &world.noisy_obs[..history.len()];
        trace_arm(
            "real_noisy_trait",
            noisy_history,
            Some(&|i: usize, _: &EvolutionaryRescueObservation| {
                noisy_history[i]
                    .sample
                    .and_then(|s| s.observed_mean_forage_efficiency)
                    .unwrap_or(TRAIT_FLOOR)
            }),
            &snapshot_ticks,
        );

        let privileged_history = &world.privileged_obs[..history.len()];
        trace_arm(
            "privileged_trait",
            privileged_history,
            Some(&|i: usize, _: &EvolutionaryRescueObservation| {
                privileged_history[i]
                    .sample
                    .and_then(|s| s.observed_mean_forage_efficiency)
                    .unwrap_or(TRAIT_FLOOR)
            }),
            &snapshot_ticks,
        );

        println!();
    }
}
