// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phase 2.2C-iii-d (`SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`): the causal confirmation
//! Phase 2.2C-iii's instrumentation trace called for. That trace found `transition_matrices`/
//! `likelihood_matrix` degenerating toward `diag≈offdiag≈0.5` in every state_dim=2 arm -- the
//! model learns to average its two belief dimensions together regardless of content, which is
//! consistent with (but not yet proven to *cause*) every arm's relative Brier score.
//!
//! **Corrected design (2026-07-27, same day as the first attempt).** The first version forced
//! diagonal-only coupling *throughout training*, which hit a genuine, disclosed confound: the
//! population channel's trained self-transition weight (~0.99 under that constraint) compounds
//! over 300 unclamped `predict_next_state` calls into near-zero regardless of content -- an
//! artifact of training under an unfamiliar constraint, not evidence about the diag=offdiag
//! mechanism. This version isolates the extrapolation step specifically: **train every arm
//! completely normally (stock), then force diagonal-only coupling *only* for the 300-step
//! extrapolation loop**, using whatever the stock training run already converged to. This tests
//! "does cross-channel blending during extrapolation specifically corrupt the forecast" without
//! ever disturbing training dynamics -- the self-transition weight comes from ordinary,
//! already-verified-stable training, so the compounding-artifact confound cannot recur here by
//! construction.
//!
//! Three conditions per arm: `stock` (unchanged), `diag_during_training` (the first attempt's
//! design, kept for reference), `diag_at_forecast_only` (this pass's actual causal test).
//!
//! **Deliberately does NOT touch `symthaea-fep`.** `likelihood_matrix`/`transition_matrices` are
//! already `pub` fields, so all three conditions are implemented entirely from this script.
//!
//! Prerequisite: `cargo run --release --example evolutionary_rescue_generate_worlds -p
//! symthaea-futures-ensemble` first (reuses the Phase 2.2C-i fixtures).
//!
//! Run: `cargo run --release --example evolutionary_rescue_diagonal_coupling_ablation -p symthaea-futures-ensemble`

#[path = "support/evolutionary_rescue_common.rs"]
mod common;

use common::SerializedWorld;
use symthaea_futures_calibration::{BrierScore, ScoringRule};
use symthaea_futures_core::{Horizon, OutcomeRegion};
use symthaea_futures_ensemble::evolutionary_rescue::{TRAIT_CEILING, TRAIT_FLOOR};
use symthaea_futures_state::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation, mask_observation,
};
use symthaea_futures_symtropy::evolutionary_rescue::EvolutionaryRescueObservation;

const HORIZON: u64 = 300;
const CHECKPOINT_STRIDE: u64 = 300;
const POPULATION_COLLAPSE_WITHIN_HORIZON: &str = "evolutionary_rescue_collapse_within_horizon";

#[derive(Clone, Copy, PartialEq, Eq)]
enum DiagonalMode {
    Stock,
    DuringTraining,
    AtForecastOnly,
}

fn normalize_trait(v: f64) -> f64 {
    ((v - TRAIT_FLOOR) / (TRAIT_CEILING - TRAIT_FLOOR)).clamp(0.0, 1.0)
}

fn boolean_distribution(
    issued_at_tick: u64,
    horizon: Horizon,
    p_true: f64,
) -> symthaea_futures_core::ForecastDistribution {
    let p_true = p_true.clamp(0.0, 1.0);
    symthaea_futures_core::ForecastDistribution::try_from_raw(
        issued_at_tick,
        horizon,
        symthaea_futures_core::OutcomeSpaceId(POPULATION_COLLAPSE_WITHIN_HORIZON.to_string()),
        vec![
            (p_true, OutcomeRegion::Boolean(true), Vec::new()),
            (1.0 - p_true, OutcomeRegion::Boolean(false), Vec::new()),
        ],
        0.0,
    )
    .expect("clamped complementary boolean masses are valid by construction")
}

/// Zeros every off-diagonal entry of a square matrix in place.
fn clamp_diagonal(m: &mut [Vec<f64>]) {
    for (i, row) in m.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            if i != j {
                *v = 0.0;
            }
        }
    }
}

/// Replays a state_dim=2 history through a fresh agent and returns a forecast for
/// `checkpoint + horizon`, at every checkpoint, as `(p_true, actual)` pairs. `mode` controls
/// *when* (if ever) off-diagonal coupling gets zeroed -- see module docs for why this matters.
fn run_arm(
    history_full: &[EvolutionaryRescueObservation],
    second_channel: impl Fn(usize, &EvolutionaryRescueObservation) -> f64,
    trajectory: &[bool],
    mode: DiagonalMode,
) -> Vec<(f64, bool)> {
    let reference = history_full
        .first()
        .and_then(|o| o.sample)
        .map(|s| s.sampled_alive_count as f64)
        .filter(|&r| r > 0.0)
        .unwrap_or(1.0);

    let mut results = Vec::new();
    let mut checkpoint = 0u64;

    while checkpoint + HORIZON < trajectory.len() as u64 {
        let idx = checkpoint as usize;
        let history = &history_full[..=idx];

        let config = ActiveInferenceAgentConfig {
            state_dim: 2,
            obs_dim: 2,
            num_actions: 1,
            ..ActiveInferenceAgentConfig::default()
        };
        let mut agent = ActiveInferenceAgent::new(config);
        if mode == DiagonalMode::DuringTraining {
            clamp_diagonal(&mut agent.model.likelihood_matrix);
            clamp_diagonal(&mut agent.model.transition_matrices[0]);
        }
        let mut prev_belief = agent.belief.clone();

        // Training: fully normal for Stock and AtForecastOnly. Only DuringTraining re-clamps
        // after every step, matching the first attempt's (confounded) design.
        for (i, obs) in history.iter().enumerate() {
            let pop_value = obs
                .sample
                .map(|s| (s.sampled_alive_count as f64 / reference).clamp(0.0, 1.0))
                .unwrap_or(0.5);
            let pop_visible = if obs.sample.is_some() { 1.0 } else { 0.0 };
            let second_value = normalize_trait(second_channel(i, obs));

            let raw_obs = Observation::new(vec![pop_value, second_value], 1.0, "diagonal_ablation");
            let masked = mask_observation(&raw_obs, &agent.belief, &[pop_visible, 1.0]);
            agent.perceive(&masked);
            let new_belief = agent.belief.clone();
            agent.observe_transition(&prev_belief, 0, &new_belief, &masked);
            if mode == DiagonalMode::DuringTraining {
                clamp_diagonal(&mut agent.model.likelihood_matrix);
                clamp_diagonal(&mut agent.model.transition_matrices[0]);
            }
            prev_belief = new_belief;
        }

        // The corrected causal test: clamp ONLY here, after normal training has already
        // converged, right before extrapolation -- the trained diagonal self-transition weight
        // was never subjected to the diagonal-only constraint, so it carries whatever value
        // ordinary (stock) training produced, avoiding the first attempt's compounding artifact.
        if mode == DiagonalMode::AtForecastOnly {
            clamp_diagonal(&mut agent.model.likelihood_matrix);
            clamp_diagonal(&mut agent.model.transition_matrices[0]);
        }

        if mode != DiagonalMode::Stock && std::env::var("TRACE_DIAG").is_ok() && checkpoint == 0 {
            eprintln!(
                "  [diag trace] post-training transition[0][0]={:.4} transition[1][1]={:.4} \
                 belief.mean={:?}",
                agent.model.transition_matrices[0][0][0],
                agent.model.transition_matrices[0][1][1],
                agent.belief.mean
            );
        }

        // Per-step clamping: found necessary after the first "forecast-only" run produced the
        // same near-ceiling Brier as the confounded "during-training" condition. Root cause: the
        // diagonal self-transition weight ALONE (even under ordinary stock training, e.g.
        // ~0.77-0.89 traced directly) is nowhere near 1.0 -- it's the *combination* with
        // off-diagonal terms that keeps the full matrix's 300-step extrapolation well-behaved.
        // Removing off-diagonal coupling at ANY point exposes that raw diagonal value to 300
        // rounds of unclamped compounding. Clamping every step, uniformly across all three
        // modes (including stock), makes this a fair test of "does cross-channel coupling help
        // once every condition is prevented from diverging into physically meaningless
        // probability values" rather than an artifact of which condition's raw eigenvalue
        // happens to survive 300 unclamped multiplications.
        let mut projected = agent.belief.clone();
        for _ in 0..HORIZON {
            projected = agent.model.predict_next_state(&projected, 0);
            for m in projected.mean.iter_mut() {
                *m = m.clamp(0.0, 1.0);
            }
        }
        let projected_fraction = projected
            .mean
            .first()
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let p_true = (1.0 - projected_fraction).clamp(0.0, 1.0);
        let actual = trajectory[(checkpoint + HORIZON) as usize];
        results.push((p_true, actual));

        checkpoint += CHECKPOINT_STRIDE;
    }

    results
}

fn mean_brier(pairs: &[(f64, bool)]) -> f64 {
    if pairs.is_empty() {
        return f64::NAN;
    }
    let sum: f64 = pairs
        .iter()
        .map(|&(p, a)| {
            BrierScore
                .score(
                    &boolean_distribution(0, Horizon(HORIZON), p),
                    &OutcomeRegion::Boolean(a),
                )
                .expect("scoring a validated forecast cannot fail")
                .get()
        })
        .sum();
    sum / pairs.len() as f64
}

type PredictionPairs = Vec<(f64, bool)>;

fn main() {
    println!("== Phase 2.2C-iii-d (corrected): diagonal-coupling ablation ==");
    println!(
        "Three conditions: stock, diag-forced throughout training (the first, confounded \
         attempt), diag-forced only at forecast time (the corrected causal test).\n"
    );

    let midpoint = (TRAIT_FLOOR + TRAIT_CEILING) / 2.0;

    let mut totals: std::collections::HashMap<&str, [PredictionPairs; 3]> =
        std::collections::HashMap::new();
    for name in ["constant", "duplicate_census", "privileged_trait"] {
        totals.insert(name, [Vec::new(), Vec::new(), Vec::new()]);
    }

    for &seed in &common::TEST_SEEDS {
        let world: SerializedWorld = common::load_world(seed);
        let reference = world
            .census_obs
            .first()
            .and_then(|o| o.sample)
            .map(|s| s.sampled_alive_count as f64)
            .filter(|&r| r > 0.0)
            .unwrap_or(1.0);

        let const_fn = |_: usize, _: &EvolutionaryRescueObservation| midpoint;
        let dup_fn = |_: usize, obs: &EvolutionaryRescueObservation| {
            let pop_fraction = obs
                .sample
                .map(|s| (s.sampled_alive_count as f64 / reference).clamp(0.0, 1.0))
                .unwrap_or(0.5);
            TRAIT_FLOOR + pop_fraction * (TRAIT_CEILING - TRAIT_FLOOR)
        };
        let priv_fn = |i: usize, _: &EvolutionaryRescueObservation| {
            world.privileged_obs[i]
                .sample
                .and_then(|s| s.observed_mean_forage_efficiency)
                .unwrap_or(TRAIT_FLOOR)
        };

        for (mode_idx, mode) in [
            (0usize, DiagonalMode::Stock),
            (1usize, DiagonalMode::DuringTraining),
            (2usize, DiagonalMode::AtForecastOnly),
        ] {
            totals.get_mut("constant").unwrap()[mode_idx].extend(run_arm(
                &world.census_obs,
                const_fn,
                &world.trajectory,
                mode,
            ));
            totals.get_mut("duplicate_census").unwrap()[mode_idx].extend(run_arm(
                &world.census_obs,
                dup_fn,
                &world.trajectory,
                mode,
            ));
            totals.get_mut("privileged_trait").unwrap()[mode_idx].extend(run_arm(
                &world.privileged_obs,
                priv_fn,
                &world.trajectory,
                mode,
            ));
        }
    }

    println!(
        "{:20} {:>10} {:>18} {:>18} {:>10}",
        "arm", "stock", "diag-training", "diag-forecast-only", "delta(fc)"
    );
    println!("{}", "-".repeat(82));
    for name in ["constant", "duplicate_census", "privileged_trait"] {
        let pairs = &totals[name];
        let stock_brier = mean_brier(&pairs[0]);
        let training_brier = mean_brier(&pairs[1]);
        let forecast_only_brier = mean_brier(&pairs[2]);
        println!(
            "{name:20} {stock_brier:10.4} {training_brier:18.4} {forecast_only_brier:18.4} {:+10.4}",
            forecast_only_brier - stock_brier
        );
    }

    println!(
        "\nPredicted if the diag=offdiag degeneracy causally hurts extrapolation specifically: \
         privileged_trait's diag-forecast-only Brier should improve (negative delta) relative to \
         its stock run, while duplicate_census's delta should stay small (it never relied on \
         off-diagonal blending). diag-training is printed for reference only -- it's the first \
         attempt's confounded condition, not a valid comparison point on its own."
    );
}
