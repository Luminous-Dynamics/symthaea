// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cryptographic commitment for standardized prescribed-action FEP prediction.
//!
//! The legacy frozen-snapshot `u64` digest is intentionally retained as a small
//! deterministic replay identifier. It is not collision-resistant enough to be
//! the scientific identity of a learned EUREKA subject. This module provides a
//! separate BLAKE3 commitment over the exact prediction-relevant state after the
//! same transient reset used by held-out snapshotting.

use std::fmt::Write as _;

use crate::ActiveInferenceAgent;

/// Revision of the canonical semantic byte grammar used for the commitment.
pub const FEP_FROZEN_PREDICTION_COMMITMENT_REVISION: &str =
    "symthaea-fep-frozen-prediction-commitment-v1";

/// Collision-resistant identity of one standardized learned predictor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrozenPredictionCommitment {
    digest: [u8; 32],
}

impl FrozenPredictionCommitment {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.digest
    }

    pub fn to_hex(self) -> String {
        let mut output = String::with_capacity(64);
        for byte in self.digest {
            write!(&mut output, "{byte:02x}").expect("writing hex into String cannot fail");
        }
        output
    }
}

/// Commit the exact state used by prescribed held-out prediction after
/// standardizing transient inference state with `ActiveInferenceAgent::reset`.
///
/// The commitment intentionally covers:
///
/// - the full active-inference configuration, including TD configuration;
/// - the learned generative model parameters;
/// - the standardized belief state;
/// - the standardized precision state;
/// - reset free-energy-calculator state;
/// - whether a TD learner remains present after reset;
/// - explicit absence/presence of previous-state / pending-action history.
///
/// It intentionally excludes stochastic action-selection RNG because EUREKA's
/// prescribed-action path never calls `select_action()`. It also excludes
/// telemetry/statistics whose values cannot affect `perceive(); act(action)`
/// after reset. The revision string binds this semantic scope so future changes
/// require a new commitment revision rather than silently reusing v1.
pub fn frozen_prediction_commitment(agent: &ActiveInferenceAgent) -> FrozenPredictionCommitment {
    let mut standardized = agent.clone();
    standardized.reset();

    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        FEP_FROZEN_PREDICTION_COMMITMENT_REVISION.as_bytes(),
    );

    let config = &standardized.config;
    push_usize(&mut bytes, config.state_dim);
    push_usize(&mut bytes, config.obs_dim);
    push_usize(&mut bytes, config.num_actions);
    push_usize(&mut bytes, config.inference_iterations);
    push_f64(&mut bytes, config.belief_learning_rate);
    push_usize(&mut bytes, config.planning_horizon);
    push_f64(&mut bytes, config.action_temperature);
    bytes.push(u8::from(config.enable_model_learning));
    bytes.push(u8::from(config.enable_td_learning));

    let td = &config.td_config;
    push_f64(&mut bytes, td.initial_learning_rate);
    push_f64(&mut bytes, td.min_learning_rate);
    push_f64(&mut bytes, td.learning_rate_decay);
    push_f64(&mut bytes, td.gamma);
    push_f64(&mut bytes, td.lambda);
    bytes.push(u8::from(td.use_eligibility_traces));
    push_f64(&mut bytes, td.trace_decay);
    push_usize(&mut bytes, td.max_transition_history);
    push_f64(&mut bytes, td.confidence_decay);
    push_f64(&mut bytes, td.min_confidence);

    let model = &standardized.model;
    push_usize(&mut bytes, model.state_dim);
    push_usize(&mut bytes, model.obs_dim);
    push_usize(&mut bytes, model.num_actions);
    push_f64(&mut bytes, model.observation_precision);
    push_f64(&mut bytes, model.transition_precision);
    push_f64(&mut bytes, model.learning_rate);
    push_matrix(&mut bytes, &model.likelihood_matrix);
    push_tensor3(&mut bytes, &model.transition_matrices);
    push_matrix(&mut bytes, &model.transition_bias);
    push_f64_slice(&mut bytes, &model.prior_mean);
    push_f64_slice(&mut bytes, &model.prior_precision);

    let belief = &standardized.belief;
    push_f64_slice(&mut bytes, &belief.mean);
    push_f64_slice(&mut bytes, &belief.precision);
    push_f64_slice(&mut bytes, &belief.mode_probs);
    push_usize(&mut bytes, belief.current_mode);

    let precision = &standardized.precision;
    push_f64(&mut bytes, precision.sensory_precision);
    push_f64(&mut bytes, precision.prior_precision);
    push_f64(&mut bytes, precision.state_precision);
    push_f64(&mut bytes, precision.action_precision);
    push_f64(&mut bytes, precision.learning_rate);
    push_usize(&mut bytes, precision.max_history);
    push_usize(&mut bytes, precision.history.len());
    for snapshot in &precision.history {
        push_f64(&mut bytes, snapshot.sensory);
        push_f64(&mut bytes, snapshot.prior);
        push_f64(&mut bytes, snapshot.state);
        push_f64(&mut bytes, snapshot.action);
        bytes.extend_from_slice(&snapshot.timestamp.to_le_bytes());
    }

    let free_energy = &standardized.free_energy_calc;
    push_usize(&mut bytes, free_energy.max_history);
    push_f64(&mut bytes, free_energy.running_average);
    push_usize(&mut bytes, free_energy.history.len());
    for value in &free_energy.history {
        push_f64(&mut bytes, *value);
    }

    // `perceive()` branches on whether a TD learner exists: its presence can
    // suppress direct model learning even on the first observation.
    bytes.push(u8::from(standardized.td_learner.is_some()));

    // Snapshot reset semantics normally require both to be absent before the
    // first held-out observation. Encode actual state so an accidental future
    // reset change cannot silently satisfy this v1 byte grammar.
    bytes.push(u8::from(standardized.previous_state.is_some()));
    match standardized.last_action {
        Some(action) => {
            bytes.push(1);
            push_usize(&mut bytes, action);
        }
        None => bytes.push(0),
    }

    FrozenPredictionCommitment {
        digest: *blake3::hash(&bytes).as_bytes(),
    }
}

fn push_usize(bytes: &mut Vec<u8>, value: usize) {
    bytes.extend_from_slice(&(value as u64).to_le_bytes());
}

fn push_f64(bytes: &mut Vec<u8>, value: f64) {
    bytes.extend_from_slice(&value.to_bits().to_le_bytes());
}

fn push_f64_slice(bytes: &mut Vec<u8>, values: &[f64]) {
    push_usize(bytes, values.len());
    for value in values {
        push_f64(bytes, *value);
    }
}

fn push_matrix(bytes: &mut Vec<u8>, matrix: &[Vec<f64>]) {
    push_usize(bytes, matrix.len());
    for row in matrix {
        push_f64_slice(bytes, row);
    }
}

fn push_tensor3(bytes: &mut Vec<u8>, tensor: &[Vec<Vec<f64>>]) {
    push_usize(bytes, tensor.len());
    for matrix in tensor {
        push_matrix(bytes, matrix);
    }
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    push_usize(bytes, value.len());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ActiveInferenceAgentConfig;

    fn agent() -> ActiveInferenceAgent {
        ActiveInferenceAgent::new(ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim: 4,
            num_actions: 4,
            ..Default::default()
        })
    }

    #[test]
    fn identical_learned_predictors_have_identical_commitments() {
        let a = agent();
        let b = a.clone();
        assert_eq!(
            frozen_prediction_commitment(&a),
            frozen_prediction_commitment(&b)
        );
    }

    #[test]
    fn learned_model_change_changes_commitment() {
        let a = agent();
        let mut b = a.clone();
        b.model.transition_matrices[0][0][0] += 0.01;
        assert_ne!(
            frozen_prediction_commitment(&a),
            frozen_prediction_commitment(&b)
        );
    }

    #[test]
    fn transition_bias_change_changes_commitment() {
        let a = agent();
        let mut b = a.clone();
        b.model.transition_bias[2][3] = 0.125;
        assert_ne!(
            frozen_prediction_commitment(&a),
            frozen_prediction_commitment(&b)
        );
    }

    #[test]
    fn config_change_changes_commitment() {
        let a = agent();
        let mut b = a.clone();
        b.config.belief_learning_rate += 0.001;
        assert_ne!(
            frozen_prediction_commitment(&a),
            frozen_prediction_commitment(&b)
        );
    }

    #[test]
    fn transient_belief_is_standardized_before_commitment() {
        let a = agent();
        let mut b = a.clone();
        b.belief.mean[0] = 0.987;
        b.belief.precision[0] = 3.5;
        b.previous_state = Some(b.belief.clone());
        b.last_action = Some(2);
        assert_eq!(
            frozen_prediction_commitment(&a),
            frozen_prediction_commitment(&b)
        );
    }

    #[test]
    fn td_presence_after_reset_is_part_of_prediction_identity() {
        let mut a = agent();
        let mut b = a.clone();
        a.config.enable_td_learning = false;
        b.config.enable_td_learning = false;
        a.td_learner = None;
        // Deliberately leave b.td_learner present to model a stale state that
        // current `reset()` does not clear when the config flag is false.
        assert_ne!(
            frozen_prediction_commitment(&a),
            frozen_prediction_commitment(&b)
        );
    }

    #[test]
    fn hex_form_is_full_blake3_width() {
        let commitment = frozen_prediction_commitment(&agent());
        assert_eq!(commitment.as_bytes().len(), 32);
        assert_eq!(commitment.to_hex().len(), 64);
    }
}
