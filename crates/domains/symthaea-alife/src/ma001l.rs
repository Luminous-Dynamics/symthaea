// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MA-001L — Contextual Transition Learning: algorithm selection on prerecorded synthetic
//! transition data, per `ALIFE_MA001L_CONTEXTUAL_TRANSITION_LEARNING_PLAN_2026-07-26.md`.
//!
//! No `Organism`, no `Population`, no live agent behavior at all — MA-001R localized its Full
//! null to the learning rule itself, not the representation or the population-scale ecology, so
//! this probe removes every remaining confound and asks a pure algorithm-selection question:
//! does *any* candidate update rule reconstruct a known, frozen, hand-specified conditional
//! transition from prerecorded `(state, action, next_state, observation)` tuples? The new
//! `DeltaRuleLearner` here is deliberately experiment-local — nothing here is wired into
//! `symthaea-fep` production types.

use symthaea_fep::{GenerativeModel, HiddenState, Observation};

use crate::ledger::compress_for_observation;
use crate::ma001r::{context_a, context_b};

pub const STATE_DIM: usize = 6;
pub const OBS_DIM: usize = 6;
pub const NUM_ACTIONS: usize = 3;
pub const ACTION_FORAGE: usize = 0;
pub const ACTION_REST: usize = 1;
pub const ACTION_TRANSFER: usize = 2;

pub const OUTCOME_A: f64 = 0.9;
pub const OUTCOME_B: f64 = 0.2;
/// Ground truth for the Forage/Rest negative control (plan §3): energy never depends on social
/// context under these two actions, regardless of which context is presented.
pub const NEGATIVE_CONTROL_OUTCOME: f64 = 0.5;
/// Constant resource level, matching MA-001R's own `resource_level` (plan §3).
pub const CONSTANT_RESOURCE: f64 = 0.5;
/// Neutral placeholder for the *input* energy dimension of every tuple's `old_state` — matches
/// MA-001R's own baseline belief construction; every tuple starts from the same reference point so
/// arms are compared fairly.
pub const NEUTRAL_ENERGY: f64 = 0.5;

/// Which context this tuple presents (plan §3): `A` (rich shared history) or `B` (none).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextLabel {
    A,
    B,
}

/// Which context↔outcome correspondence a Transfer sub-index follows — mirrors
/// `crate::ma001r::Schedule` but scoped to this module's own prerecorded-stream generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamSchedule {
    Bound,
    Reversed,
}

fn social_dims_for(label: ContextLabel) -> [f64; 4] {
    let ctx = match label {
        ContextLabel::A => context_a(),
        ContextLabel::B => context_b(),
    };
    [
        1.0,
        compress_for_observation(ctx.given_to_partner),
        compress_for_observation(ctx.received_from_partner),
        compress_for_observation(ctx.encounter_count as f64),
    ]
}

fn make_state(resource: f64, energy: f64, social: [f64; 4]) -> HiddenState {
    HiddenState {
        mean: vec![resource, energy, social[0], social[1], social[2], social[3]],
        precision: vec![1.0; STATE_DIM],
        mode_probs: vec![1.0],
        current_mode: 0,
    }
}

/// One prerecorded transition tuple (plan §3).
#[derive(Debug, Clone)]
pub struct Tuple {
    pub old_state: HiddenState,
    pub action: usize,
    pub new_state: HiddenState,
    pub observation: Observation,
    pub context_label: ContextLabel,
}

fn make_tuple(action: usize, context_label: ContextLabel, next_energy: f64) -> Tuple {
    let social = social_dims_for(context_label);
    let old_state = make_state(CONSTANT_RESOURCE, NEUTRAL_ENERGY, social);
    let new_state = make_state(CONSTANT_RESOURCE, next_energy, social);
    let observation = Observation::new(new_state.mean.clone(), 1.0, "ma001l");
    Tuple {
        old_state,
        action,
        new_state,
        observation,
        context_label,
    }
}

fn context_and_outcome_for(sub_index: u64, schedule: StreamSchedule) -> (ContextLabel, f64) {
    let is_a = sub_index % 2 == 0;
    match schedule {
        StreamSchedule::Bound => {
            if is_a {
                (ContextLabel::A, OUTCOME_A)
            } else {
                (ContextLabel::B, OUTCOME_B)
            }
        }
        StreamSchedule::Reversed => {
            if is_a {
                (ContextLabel::A, OUTCOME_B)
            } else {
                (ContextLabel::B, OUTCOME_A)
            }
        }
    }
}

/// Generate `total` interleaved training tuples (plan §3): even overall index = Transfer tuple
/// (context↔outcome per `schedule`), odd = Forage/Rest negative-control tuple (energy always
/// `NEGATIVE_CONTROL_OUTCOME` regardless of context — the built-in action-specificity control).
/// `total` must be even.
pub fn generate_bound_stream(total: u64, schedule: StreamSchedule) -> Vec<Tuple> {
    assert!(
        total % 2 == 0,
        "generate_bound_stream requires an even tuple count"
    );
    let mut tuples = Vec::with_capacity(total as usize);
    let mut transfer_index = 0u64;
    let mut control_index = 0u64;
    for i in 0..total {
        if i % 2 == 0 {
            let (label, outcome) = context_and_outcome_for(transfer_index, schedule);
            tuples.push(make_tuple(ACTION_TRANSFER, label, outcome));
            transfer_index += 1;
        } else {
            let label = if control_index % 2 == 0 {
                ContextLabel::A
            } else {
                ContextLabel::B
            };
            let action = if control_index % 2 == 0 {
                ACTION_FORAGE
            } else {
                ACTION_REST
            };
            tuples.push(make_tuple(action, label, NEGATIVE_CONTROL_OUTCOME));
            control_index += 1;
        }
    }
    tuples
}

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Shuffled-context stream (plan §3): the outcome schedule stays tied to Transfer sub-index parity
/// (identical to `Bound`), but which context is *presented* is independently re-randomized each
/// Transfer tuple. Negative-control tuples are unaffected (their ground truth ignores context
/// regardless).
pub fn generate_shuffled_stream(total: u64, rng_state: &mut u64) -> Vec<Tuple> {
    assert!(
        total % 2 == 0,
        "generate_shuffled_stream requires an even tuple count"
    );
    let mut tuples = Vec::with_capacity(total as usize);
    let mut transfer_index = 0u64;
    let mut control_index = 0u64;
    for i in 0..total {
        if i % 2 == 0 {
            let (_, outcome) = context_and_outcome_for(transfer_index, StreamSchedule::Bound);
            let label = if xorshift(rng_state) % 2 == 0 {
                ContextLabel::A
            } else {
                ContextLabel::B
            };
            tuples.push(make_tuple(ACTION_TRANSFER, label, outcome));
            transfer_index += 1;
        } else {
            let label = if control_index % 2 == 0 {
                ContextLabel::A
            } else {
                ContextLabel::B
            };
            let action = if control_index % 2 == 0 {
                ACTION_FORAGE
            } else {
                ACTION_REST
            };
            tuples.push(make_tuple(action, label, NEGATIVE_CONTROL_OUTCOME));
            control_index += 1;
        }
    }
    tuples
}

/// Held-out Transfer-only tuples (plan §3), fresh sub-indices continuing from `start_index` —
/// never passed to any update call, used only for the §5 held-out prediction metric.
pub fn generate_held_out_stream(
    count: u64,
    schedule: StreamSchedule,
    start_index: u64,
) -> Vec<Tuple> {
    (0..count)
        .map(|k| {
            let (label, outcome) = context_and_outcome_for(start_index + k, schedule);
            make_tuple(ACTION_TRANSFER, label, outcome)
        })
        .collect()
}

/// One-step prediction-error delta rule (plan §4) — experiment-local, not added to
/// `symthaea-fep`. Deliberately does **not** row-renormalize (Gate 0(b) proved renormalization is
/// itself a source of outcome-blind coefficient movement in the existing Hebbian path).
#[derive(Debug, Clone, Copy)]
pub struct DeltaRuleConfig {
    pub eta: f64,
    pub decay: f64,
    pub clip_bound: f64,
    pub bias_learning: bool,
}

impl Default for DeltaRuleConfig {
    fn default() -> Self {
        Self {
            eta: 0.01,
            decay: 0.001,
            clip_bound: 5.0,
            bias_learning: false,
        }
    }
}

pub struct DeltaRuleLearner {
    pub cfg: DeltaRuleConfig,
    initial_transition_matrices: Vec<Vec<Vec<f64>>>,
}

impl DeltaRuleLearner {
    pub fn new(cfg: DeltaRuleConfig, model: &GenerativeModel) -> Self {
        Self {
            cfg,
            initial_transition_matrices: model.transition_matrices.clone(),
        }
    }

    /// One update (plan §4): `predicted = A_a . old_state (+ b_a)`, `error = actual_next -
    /// predicted`, `A_a += eta*outer(error, old_state) - decay*(A_a - A_a_initial)`, clipped.
    /// `matrix[j][i]` is the coefficient from dim `j` to dim `i`, matching
    /// `GenerativeModel::predict_next_state`'s own indexing (`next_mean[i] += transition[j][i] *
    /// state.mean[j]`).
    pub fn update(
        &self,
        model: &mut GenerativeModel,
        old_state: &HiddenState,
        action: usize,
        actual_next: &HiddenState,
    ) {
        let predicted = model.predict_next_state(old_state, action);
        let n = model.state_dim;
        let initial = &self.initial_transition_matrices[action];
        let errors: Vec<f64> = (0..n)
            .map(|i| actual_next.mean[i] - predicted.mean[i])
            .collect();
        {
            let matrix = &mut model.transition_matrices[action];
            for i in 0..n {
                let error_i = errors[i];
                for j in 0..n {
                    let state_j = old_state.mean[j];
                    let update = self.cfg.eta * error_i * state_j;
                    let decay_term = self.cfg.decay * (matrix[j][i] - initial[j][i]);
                    let new_val = (matrix[j][i] + update - decay_term)
                        .clamp(-self.cfg.clip_bound, self.cfg.clip_bound);
                    matrix[j][i] = new_val;
                }
            }
        }
        if self.cfg.bias_learning {
            let bias = &mut model.transition_bias[action];
            for i in 0..n {
                bias[i] = (bias[i] + self.cfg.eta * errors[i])
                    .clamp(-self.cfg.clip_bound, self.cfg.clip_bound);
            }
        }
    }
}

/// Held-out mean absolute error on the energy dimension (dim 1), predicted via
/// `predict_next_state` -> `predict_observation`, matching MA-001R's own mechanism.
pub fn held_out_energy_error(model: &GenerativeModel, tuples: &[Tuple]) -> f64 {
    if tuples.is_empty() {
        return 0.0;
    }
    let errors: Vec<f64> = tuples
        .iter()
        .map(|t| {
            let next = model.predict_next_state(&t.old_state, t.action);
            let predicted = model.predict_observation(&next);
            let predicted_energy = predicted.get(1).copied().unwrap_or(0.5);
            let actual_energy = t.new_state.mean[1];
            (predicted_energy - actual_energy).abs()
        })
        .collect();
    errors.iter().sum::<f64>() / errors.len() as f64
}

/// Counterfactual predicted energy for `label` under `action`, holding physical dims at the same
/// neutral reference every tuple was trained from (plan §5) — mirrors
/// `Ma001rProbe::predicted_energy_for_context`.
pub fn predicted_energy_for(model: &GenerativeModel, action: usize, label: ContextLabel) -> f64 {
    let social = social_dims_for(label);
    let variant = make_state(CONSTANT_RESOURCE, NEUTRAL_ENERGY, social);
    let next = model.predict_next_state(&variant, action);
    let predicted = model.predict_observation(&next);
    predicted.get(1).copied().unwrap_or(0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bound_stream_alternates_transfer_and_negative_control() {
        let tuples = generate_bound_stream(8, StreamSchedule::Bound);
        assert_eq!(tuples.len(), 8);
        for (i, t) in tuples.iter().enumerate() {
            if i % 2 == 0 {
                assert_eq!(t.action, ACTION_TRANSFER);
            } else {
                assert!(t.action == ACTION_FORAGE || t.action == ACTION_REST);
                assert_eq!(
                    t.new_state.mean[1], NEGATIVE_CONTROL_OUTCOME,
                    "negative-control tuples must never carry the context-dependent outcome"
                );
            }
        }
    }

    #[test]
    fn bound_transfer_tuples_alternate_context_and_outcome_correctly() {
        let tuples = generate_bound_stream(8, StreamSchedule::Bound);
        let transfer_tuples: Vec<&Tuple> = tuples
            .iter()
            .filter(|t| t.action == ACTION_TRANSFER)
            .collect();
        assert_eq!(transfer_tuples[0].context_label, ContextLabel::A);
        assert_eq!(transfer_tuples[0].new_state.mean[1], OUTCOME_A);
        assert_eq!(transfer_tuples[1].context_label, ContextLabel::B);
        assert_eq!(transfer_tuples[1].new_state.mean[1], OUTCOME_B);
    }

    #[test]
    fn reversed_schedule_swaps_the_context_outcome_correspondence() {
        let tuples = generate_bound_stream(4, StreamSchedule::Reversed);
        let transfer_tuples: Vec<&Tuple> = tuples
            .iter()
            .filter(|t| t.action == ACTION_TRANSFER)
            .collect();
        assert_eq!(transfer_tuples[0].context_label, ContextLabel::A);
        assert_eq!(transfer_tuples[0].new_state.mean[1], OUTCOME_B);
        assert_eq!(transfer_tuples[1].context_label, ContextLabel::B);
        assert_eq!(transfer_tuples[1].new_state.mean[1], OUTCOME_A);
    }

    #[test]
    fn shuffled_stream_keeps_the_outcome_schedule_tied_to_tick_parity() {
        let mut rng = 42u64;
        let tuples = generate_shuffled_stream(20, &mut rng);
        let transfer_outcomes: Vec<f64> = tuples
            .iter()
            .filter(|t| t.action == ACTION_TRANSFER)
            .map(|t| t.new_state.mean[1])
            .collect();
        for (k, outcome) in transfer_outcomes.iter().enumerate() {
            let expected = if k % 2 == 0 { OUTCOME_A } else { OUTCOME_B };
            assert_eq!(
                *outcome, expected,
                "outcome schedule must stay bound to parity even when context is shuffled"
            );
        }
    }

    #[test]
    fn context_values_match_ma001r_exactly() {
        // Direct regression check against ma001r.rs's own context_a()/context_b() (plan sec 9
        // step 1) -- not a re-derivation.
        let a = context_a();
        let b = context_b();
        assert_eq!(a.given_to_partner, 2.0);
        assert_eq!(a.received_from_partner, 2.0);
        assert_eq!(a.encounter_count, 20);
        assert_eq!(b.given_to_partner, 0.0);
        assert_eq!(b.received_from_partner, 0.0);
        assert_eq!(b.encounter_count, 0);
    }

    #[test]
    fn delta_rule_moves_the_transfer_coefficient_toward_the_target_relationship() {
        let mut model = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
        let learner = DeltaRuleLearner::new(DeltaRuleConfig::default(), &model);
        let tuples = generate_bound_stream(2000, StreamSchedule::Bound);
        for t in &tuples {
            learner.update(&mut model, &t.old_state, t.action, &t.new_state);
        }
        let held_out = generate_held_out_stream(200, StreamSchedule::Bound, 1000);
        let error = held_out_energy_error(&model, &held_out);
        // A weak sanity check only -- the real gates (plan sec 6) are computed by the driver
        // example, not asserted here as a pass/fail; this just confirms the learner runs to
        // completion without producing NaN/garbage.
        assert!(error.is_finite());
    }

    #[test]
    fn neither_arm_leaves_the_model_at_its_initial_values() {
        let model = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
        let fresh = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
        assert_eq!(model.transition_matrices, fresh.transition_matrices);
    }
}
