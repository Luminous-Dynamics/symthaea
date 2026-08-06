// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MA-001R Gate 0, per
//! `ALIFE_MA001R_SOCIAL_PHYSICAL_COUPLING_PLAN_2026-07-26.md` §2: a direct, `Organism`-independent
//! confirmation that `TemporalDifferenceLearner::update_model` can move an off-diagonal transition
//! coefficient connecting a "social" dimension to a "physical" one, in a maximally friendly
//! synthetic setup. Expected to pass, per reading `td_learning.rs` directly (its transition-matrix
//! update loops over every `(i, j)` pair, unlike `GenerativeModel::learn`'s diagonal-only Hebbian
//! path) -- verified here rather than merely assumed. If this ever failed, MA-001R's fuller
//! single-agent probe would be unnecessary: the null would already have a one-line mechanistic
//! explanation.

use symthaea_fep::{
    GenerativeModel, HiddenState, Observation, TemporalDifferenceLearner,
    TemporalDifferenceLearningConfig,
};

const STATE_DIM: usize = 6; // matches symthaea-alife's social_enabled Organism layout
const OBS_DIM: usize = 6;
const NUM_ACTIONS: usize = 3;
const SOCIAL_DIM: usize = 2; // "partner_present" in the real 6-dim layout
const PHYSICAL_DIM: usize = 0; // "resource_level"
const ACTION: usize = 2; // Action::Transfer's index

fn state(mean: [f64; STATE_DIM]) -> HiddenState {
    HiddenState {
        mean: mean.to_vec(),
        precision: vec![1.0; STATE_DIM],
        mode_probs: vec![1.0],
        current_mode: 0,
    }
}

fn obs(mean: [f64; OBS_DIM]) -> Observation {
    Observation::new(mean.to_vec(), 1.0, "gate0")
}

#[test]
fn td_learning_moves_the_social_to_physical_off_diagonal_coefficient() {
    let mut model = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
    let mut learner = TemporalDifferenceLearner::new(
        TemporalDifferenceLearningConfig::default(),
        NUM_ACTIONS,
        STATE_DIM,
        OBS_DIM,
    );

    let initial_coefficient = model.transition_matrices[ACTION][SOCIAL_DIM][PHYSICAL_DIM];

    // Present a perfectly consistent co-occurrence, alternating high/low, many times: when the
    // social dimension is high, the physical dimension is high next; when low, low next. All
    // other dimensions held at a neutral constant so the only real signal is dim SOCIAL_DIM ->
    // dim PHYSICAL_DIM.
    for step in 0..500u64 {
        let high = step % 2 == 0;
        let social_val = if high { 0.9 } else { 0.1 };
        let physical_val = if high { 0.9 } else { 0.1 };

        let old_state = state([social_val, 0.5, social_val, 0.5, 0.5, 0.5]);
        let new_state = state([physical_val, 0.5, social_val, 0.5, 0.5, 0.5]);
        let observation = obs([physical_val, 0.5, social_val, 0.5, 0.5, 0.5]);

        let td_error =
            learner.observe_transition(&old_state, ACTION, &new_state, &observation, &model, step);
        learner.update_model(
            &mut model,
            &old_state,
            ACTION,
            &new_state,
            &observation,
            td_error,
        );
    }

    let final_coefficient = model.transition_matrices[ACTION][SOCIAL_DIM][PHYSICAL_DIM];

    assert!(
        (final_coefficient - initial_coefficient).abs() > 0.05,
        "TD learning should move the social->physical off-diagonal coefficient measurably under \
         a perfectly consistent 500-step co-occurrence: initial={initial_coefficient}, \
         final={final_coefficient}"
    );
}

#[test]
fn hebbian_only_path_transition_update_is_blind_to_the_actual_physical_outcome() {
    // Correction to an earlier, wrong version of this test: GenerativeModel::learn's
    // transition-matrix branch DOES change transition_matrices[action][SOCIAL_DIM][PHYSICAL_DIM]
    // under this exposure sequence (initial=0.025 -> final=0.0226 was measured directly) -- but
    // only as a side effect of row-renormalization following a diagonal increment
    // (transition_matrices[idx][i][i] += ... when state.mean[i] > 0.5), which rescales every
    // entry in row i, diagonal and off-diagonal alike. Confirmed by reading generative_model.rs
    // directly: the transition-matrix branch reads only `state.mean` and `action` -- it never
    // reads `observation` at all. So the real, precise, testable claim is not "never moves" but
    // "moves identically regardless of what physical outcome actually followed" -- i.e. it is
    // mechanically incapable of encoding any social->physical relationship, since the update that
    // touches the transition matrix never sees the physical observation in the first place. Two
    // models fed the identical belief-state/action sequence but wildly different observations
    // (one correlated with the social dimension, one a flat constant) must end up with
    // bit-for-bit identical transition matrices.
    let mut model_correlated = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
    let mut model_constant = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);

    for step in 0..500u64 {
        let high = step % 2 == 0;
        let social_val = if high { 0.9 } else { 0.1 };
        let physical_val = if high { 0.9 } else { 0.1 };
        let belief_state = state([social_val, 0.5, social_val, 0.5, 0.5, 0.5]);

        let obs_correlated = obs([physical_val, 0.5, social_val, 0.5, 0.5, 0.5]);
        model_correlated.learn(&belief_state, &obs_correlated, Some(ACTION));

        let obs_constant = obs([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        model_constant.learn(&belief_state, &obs_constant, Some(ACTION));
    }

    assert_eq!(
        model_correlated.transition_matrices[ACTION], model_constant.transition_matrices[ACTION],
        "Hebbian-only learn()'s transition-matrix update must be identical regardless of the \
         actual physical observation, since it never reads `observation` -- any off-diagonal \
         movement it produces is a renormalization artifact of the diagonal update, not a \
         social->physical relationship learned from the outcome"
    );
}
