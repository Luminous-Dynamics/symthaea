// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Root cause behind `tests/hoffman_action_selection_resource_insensitivity.rs`'s finding, per
//! `HOFFMAN_INTERFACE_THEORY_PLAN_2026-07-22.md`'s investigation into `symthaea-fep`'s
//! `ActiveInferenceAgent`.
//!
//! A direct trace (bypassing `Organism`/perception entirely -- belief set directly) found belief
//! *does* correctly diverge with the true resource level (confirmed separately: two organisms at
//! resource_level 0.05 vs 0.95 converge to belief means around 0.15-0.28 and 0.82-0.94
//! respectively). The insensitivity is not a perception bug.
//!
//! It's in `select_action`'s expected-free-energy pragmatic term. `GenerativeModel::new`'s
//! per-action transition dynamics for a 2-action, 2-state (`[resource, energy]`) agent (matching
//! `Organism`'s `Action::{Forage,Rest}` / `STATE_DIM=2`) give, using `predict_next_state`'s
//! `next_mean[to] = Σ_from transition[from][to] * belief[from]`:
//!
//! - Forage's predicted resource dimension: `0.9·r_belief + 0.2·e_belief` (amplifies resource)
//! - Rest's predicted resource dimension:   `0.7·r_belief + 0.1·e_belief` (dampens it)
//!
//! `Organism::new`'s goal preference is `[0.5, set_point]` -- a **moderate** resource reading,
//! not "as high as possible." Because Forage's dynamics amplify whatever the current resource
//! belief is, Forage's predicted observation overshoots the moderate 0.5 target at *both* low and
//! high resource beliefs, while Rest's damped dynamics stay closer to the target across the whole
//! range. Net effect, confirmed both by direct formula evaluation and by the live
//! `ExpectedFreeEnergyComputer`: **Rest's pragmatic value is lower (better) than Forage's at every
//! resource belief tested, 0.05 through 0.95 -- never crosses over.** The gap is merely smallest
//! (Forage least disadvantaged) near resource belief ≈ 0.2-0.35, not reversed.
//!
//! This is the real mechanism behind both Hoffman experiments' negative results: the goal
//! preference that gates action selection was never calibrated to the actual fitness-relevant
//! quantity (real energy gain scales monotonically with true resource level); it targets an
//! arbitrary moderate reading instead. Resolving resource detail more precisely cannot pay off
//! when the decision it would inform is structurally biased toward one action regardless of that
//! detail. Fixing this (e.g. recalibrating the preference to target high resource, or otherwise
//! making the goal structure fitness-aligned) would be a real change to `symthaea-fep`'s
//! `ActiveInferenceAgent` or to how `Organism::new` sets its goals -- a materially larger,
//! separately-scoped change, not attempted here.

use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig};

const RESOURCE_LEVELS: &[f64] = &[0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95];
const ENERGY_BELIEF: f64 = 0.8;
const FORAGE: usize = 0;
const REST: usize = 1;

fn probabilities_at_belief(resource_belief: f64, energy_belief: f64) -> Vec<f64> {
    let cfg = ActiveInferenceAgentConfig {
        state_dim: 2,
        obs_dim: 2,
        num_actions: 2,
        action_temperature: 1.0,
        ..Default::default()
    };
    let mut agent = ActiveInferenceAgent::new(cfg);
    agent.set_goals(vec![0.5, 0.8], 2.0); // matches Organism::new's goal preference
    agent.belief.mean = vec![resource_belief, energy_belief];
    agent.select_action().action_probabilities
}

#[test]
fn rest_pragmatically_dominates_forage_across_the_full_resource_belief_range() {
    for &r in RESOURCE_LEVELS {
        let probs = probabilities_at_belief(r, ENERGY_BELIEF);
        assert!(
            probs[REST] > probs[FORAGE],
            "expected Rest to be favored over Forage at resource_belief={r}, got \
             forage={:.4} rest={:.4}",
            probs[FORAGE],
            probs[REST]
        );
    }
}

#[test]
fn forage_probability_is_non_monotonic_peaking_near_the_arbitrary_preference_point() {
    // Forage is *least* disadvantaged near resource_belief ~0.2-0.35, not at high resource --
    // the tell that the goal structure targets a moderate reading, not "more is better."
    let probs: Vec<f64> = RESOURCE_LEVELS
        .iter()
        .map(|&r| probabilities_at_belief(r, ENERGY_BELIEF)[FORAGE])
        .collect();
    let peak_idx = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let peak_r = RESOURCE_LEVELS[peak_idx];
    assert!(
        (0.2..=0.35).contains(&peak_r),
        "expected Forage probability to peak near resource_belief 0.2-0.35, peaked at {peak_r} \
         instead: {probs:?}"
    );
    // And it must be strictly lower at both extremes than at the peak -- non-monotonic, not a
    // monotonically decreasing artifact of some unrelated effect.
    assert!(
        probs[peak_idx] > probs[0] && probs[peak_idx] > probs[RESOURCE_LEVELS.len() - 1],
        "expected the peak to exceed both extremes: {probs:?}"
    );
}
