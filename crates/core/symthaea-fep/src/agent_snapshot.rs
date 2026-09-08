// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned persistence contract for exact ActiveInferenceAgent continuation.
//!
//! The live [`crate::ActiveInferenceAgent`] deliberately does not derive serde. Persisted bytes
//! are not executable cognition merely because they deserialize: a raw [`ActiveInferenceAgentSnapshotV1`]
//! must pass structural and numeric validation before it becomes a
//! [`ValidatedActiveInferenceAgentSnapshotV1`] capability.
//!
//! This module defines the contract only. A later wiring tranche may teach the live agent to emit
//! and restore this snapshot, including its private timestamp and action-selection RNG state. Until
//! that wiring and split-run equivalence are qualified, this module is not a claim that exact live
//! agent resume is established.

use serde::{Deserialize, Serialize};

use crate::{
    ActiveInferenceAgentConfig, ActiveInferenceAgentStats, ExpectedFreeEnergyComputer,
    FreeEnergyCalculator, FreeEnergyComponents, GenerativeModel, HiddenState, PrecisionEstimator,
    StateTransition, TemporalDifferenceLearner,
};

/// Persistable complete causal state intended for one `ActiveInferenceAgent` continuation boundary.
///
/// Fields are crate-private so external deserialization does not expose a mutable bag of
/// authority-bearing internals. External callers can persist the raw value but must consume it via
/// [`Self::validate`] before it can authorize future live-agent restoration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveInferenceAgentSnapshotV1 {
    pub(crate) config: ActiveInferenceAgentConfig,
    pub(crate) belief: HiddenState,
    pub(crate) previous_state: Option<HiddenState>,
    pub(crate) last_action: Option<usize>,
    pub(crate) model: GenerativeModel,
    pub(crate) free_energy_calc: FreeEnergyCalculator,
    pub(crate) precision: PrecisionEstimator,
    pub(crate) efe_computer: ExpectedFreeEnergyComputer,
    pub(crate) td_learner: Option<TemporalDifferenceLearner>,
    pub(crate) last_fe_components: Option<FreeEnergyComponents>,
    pub(crate) stats: ActiveInferenceAgentStats,
    pub(crate) timestamp: u64,
    pub(crate) rng_state: u64,
}

/// Non-serializable capability produced only by validating a raw persisted snapshot.
#[derive(Debug, Clone)]
pub struct ValidatedActiveInferenceAgentSnapshotV1 {
    snapshot: ActiveInferenceAgentSnapshotV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActiveInferenceAgentSnapshotErrorV1 {
    ZeroDimension { field: &'static str },
    LengthMismatch {
        field: &'static str,
        expected: usize,
        observed: usize,
    },
    EmptyModes { field: &'static str },
    ModeOutOfBounds {
        field: &'static str,
        mode: usize,
        modes: usize,
    },
    ActionOutOfBounds {
        field: &'static str,
        action: usize,
        num_actions: usize,
    },
    NonFinite { field: &'static str },
    NonPositive { field: &'static str },
    Negative { field: &'static str },
    HistoryTooLong {
        field: &'static str,
        len: usize,
        max: usize,
    },
    TdPresenceMismatch {
        enabled: bool,
        learner_present: bool,
    },
    EligibilityPresenceMismatch {
        enabled: bool,
        traces_present: bool,
    },
    TimestampPerceptionMismatch {
        timestamp: u64,
        perception_cycles: u64,
    },
    ZeroRngState,
}

impl ActiveInferenceAgentSnapshotV1 {
    /// Consume untrusted persistence and produce a validated capability only after all v1
    /// structural/numeric invariants pass.
    pub fn validate(
        self,
    ) -> Result<ValidatedActiveInferenceAgentSnapshotV1, ActiveInferenceAgentSnapshotErrorV1> {
        validate_snapshot(&self)?;
        Ok(ValidatedActiveInferenceAgentSnapshotV1 { snapshot: self })
    }
}

impl ValidatedActiveInferenceAgentSnapshotV1 {
    /// Raw serializable representation for persistence. Revalidate after loading.
    pub fn as_snapshot(&self) -> &ActiveInferenceAgentSnapshotV1 {
        &self.snapshot
    }

    /// Consume the validated capability and return its persistence representation.
    pub fn into_snapshot(self) -> ActiveInferenceAgentSnapshotV1 {
        self.snapshot
    }

    pub fn timestamp(&self) -> u64 {
        self.snapshot.timestamp
    }

    pub fn perception_cycles(&self) -> u64 {
        self.snapshot.stats.perception_cycles
    }
}

fn validate_snapshot(
    snapshot: &ActiveInferenceAgentSnapshotV1,
) -> Result<(), ActiveInferenceAgentSnapshotErrorV1> {
    let config = &snapshot.config;
    require_nonzero("config.state_dim", config.state_dim)?;
    require_nonzero("config.obs_dim", config.obs_dim)?;
    require_nonzero("config.num_actions", config.num_actions)?;
    require_nonzero("config.inference_iterations", config.inference_iterations)?;
    require_nonzero("config.planning_horizon", config.planning_horizon)?;
    require_finite("config.belief_learning_rate", config.belief_learning_rate)?;
    require_nonnegative("config.belief_learning_rate", config.belief_learning_rate)?;
    require_positive("config.action_temperature", config.action_temperature)?;

    validate_hidden_state("belief", &snapshot.belief, config.state_dim)?;
    if let Some(previous) = &snapshot.previous_state {
        validate_hidden_state("previous_state", previous, config.state_dim)?;
    }
    if let Some(action) = snapshot.last_action {
        require_action("last_action", action, config.num_actions)?;
    }

    validate_model(&snapshot.model, config.state_dim, config.obs_dim, config.num_actions)?;
    validate_free_energy_calc(&snapshot.free_energy_calc)?;
    validate_precision(&snapshot.precision)?;
    validate_efe(&snapshot.efe_computer, config.obs_dim, config.num_actions)?;

    let learner_present = snapshot.td_learner.is_some();
    if learner_present != config.enable_td_learning {
        return Err(ActiveInferenceAgentSnapshotErrorV1::TdPresenceMismatch {
            enabled: config.enable_td_learning,
            learner_present,
        });
    }
    if let Some(td) = &snapshot.td_learner {
        validate_td(td, config.state_dim, config.obs_dim, config.num_actions)?;
    }

    if let Some(fe) = &snapshot.last_fe_components {
        validate_fe_components("last_fe_components", fe)?;
    }
    validate_stats(&snapshot.stats)?;

    if snapshot.timestamp != snapshot.stats.perception_cycles {
        return Err(
            ActiveInferenceAgentSnapshotErrorV1::TimestampPerceptionMismatch {
                timestamp: snapshot.timestamp,
                perception_cycles: snapshot.stats.perception_cycles,
            },
        );
    }
    if snapshot.rng_state == 0 {
        return Err(ActiveInferenceAgentSnapshotErrorV1::ZeroRngState);
    }

    Ok(())
}

fn validate_hidden_state(
    field: &'static str,
    state: &HiddenState,
    state_dim: usize,
) -> Result<(), ActiveInferenceAgentSnapshotErrorV1> {
    require_len(field, state.mean.len(), state_dim)?;
    require_len(field, state.precision.len(), state_dim)?;
    if state.mode_probs.is_empty() {
        return Err(ActiveInferenceAgentSnapshotErrorV1::EmptyModes { field });
    }
    if state.current_mode >= state.mode_probs.len() {
        return Err(ActiveInferenceAgentSnapshotErrorV1::ModeOutOfBounds {
            field,
            mode: state.current_mode,
            modes: state.mode_probs.len(),
        });
    }
    require_finite_slice(field, &state.mean)?;
    require_finite_slice(field, &state.precision)?;
    for &value in &state.precision {
        require_positive(field, value)?;
    }
    require_finite_slice(field, &state.mode_probs)?;
    let mut total = 0.0;
    for &probability in &state.mode_probs {
        require_nonnegative(field, probability)?;
        total += probability;
    }
    require_positive(field, total)?;
    Ok(())
}

fn validate_model(
    model: &GenerativeModel,
    state_dim: usize,
    obs_dim: usize,
    num_actions: usize,
) -> Result<(), ActiveInferenceAgentSnapshotErrorV1> {
    if model.state_dim != state_dim {
        return Err(ActiveInferenceAgentSnapshotErrorV1::LengthMismatch {
            field: "model.state_dim",
            expected: state_dim,
            observed: model.state_dim,
        });
    }
    if model.obs_dim != obs_dim {
        return Err(ActiveInferenceAgentSnapshotErrorV1::LengthMismatch {
            field: "model.obs_dim",
            expected: obs_dim,
            observed: model.obs_dim,
        });
    }
    if model.num_actions != num_actions {
        return Err(ActiveInferenceAgentSnapshotErrorV1::LengthMismatch {
            field: "model.num_actions",
            expected: num_actions,
            observed: model.num_actions,
        });
    }

    require_matrix("model.likelihood_matrix", &model.likelihood_matrix, state_dim, obs_dim)?;
    require_cube(
        "model.transition_matrices",
        &model.transition_matrices,
        num_actions,
        state_dim,
        state_dim,
    )?;
    require_matrix(
        "model.transition_bias",
        &model.transition_bias,
        num_actions,
        state_dim,
    )?;
    require_len("model.prior_mean", model.prior_mean.len(), state_dim)?;
    require_len("model.prior_precision", model.prior_precision.len(), state_dim)?;
    require_finite_slice("model.prior_mean", &model.prior_mean)?;
    require_finite_slice("model.prior_precision", &model.prior_precision)?;
    for &value in &model.prior_precision {
        require_positive("model.prior_precision", value)?;
    }
    require_positive("model.observation_precision", model.observation_precision)?;
    require_positive("model.transition_precision", model.transition_precision)?;
    require_finite("model.learning_rate", model.learning_rate)?;
    require_nonnegative("model.learning_rate", model.learning_rate)?;
    Ok(())
}

fn validate_free_energy_calc(
    calc: &FreeEnergyCalculator,
) -> Result<(), ActiveInferenceAgentSnapshotErrorV1> {
    require_nonzero("free_energy_calc.max_history", calc.max_history)?;
    if calc.history.len() > calc.max_history {
        return Err(ActiveInferenceAgentSnapshotErrorV1::HistoryTooLong {
            field: "free_energy_calc.history",
            len: calc.history.len(),
            max: calc.max_history,
        });
    }
    for &value in &calc.history {
        require_finite("free_energy_calc.history", value)?;
    }
    require_finite("free_energy_calc.running_average", calc.running_average)
}

fn validate_precision(
    precision: &PrecisionEstimator,
) -> Result<(), ActiveInferenceAgentSnapshotErrorV1> {
    require_positive("precision.sensory_precision", precision.sensory_precision)?;
    require_positive("precision.prior_precision", precision.prior_precision)?;
    require_positive("precision.state_precision", precision.state_precision)?;
    require_positive("precision.action_precision", precision.action_precision)?;
    require_finite("precision.learning_rate", precision.learning_rate)?;
    require_nonnegative("precision.learning_rate", precision.learning_rate)?;
    require_nonzero("precision.max_history", precision.max_history)?;
    if precision.history.len() > precision.max_history {
        return Err(ActiveInferenceAgentSnapshotErrorV1::HistoryTooLong {
            field: "precision.history",
            len: precision.history.len(),
            max: precision.max_history,
        });
    }
    for entry in &precision.history {
        require_positive("precision.history.sensory", entry.sensory)?;
        require_positive("precision.history.prior", entry.prior)?;
        require_positive("precision.history.state", entry.state)?;
        require_positive("precision.history.action", entry.action)?;
    }
    Ok(())
}

fn validate_efe(
    efe: &ExpectedFreeEnergyComputer,
    obs_dim: usize,
    num_actions: usize,
) -> Result<(), ActiveInferenceAgentSnapshotErrorV1> {
    require_finite("efe.pragmatic_weight", efe.pragmatic_weight)?;
    require_finite("efe.epistemic_weight", efe.epistemic_weight)?;
    require_finite("efe.novelty_weight", efe.novelty_weight)?;
    require_len("efe.preferences", efe.preferences.len(), obs_dim)?;
    require_finite_slice("efe.preferences", &efe.preferences)?;
    require_finite("efe.preference_precision", efe.preference_precision)?;
    require_nonnegative("efe.preference_precision", efe.preference_precision)?;
    if let Some(overrides) = &efe.precision_overrides {
        require_len("efe.precision_overrides", overrides.len(), obs_dim)?;
        require_finite_slice("efe.precision_overrides", overrides)?;
        for &value in overrides {
            require_nonnegative("efe.precision_overrides", value)?;
        }
    }
    for &action in &efe.action_history {
        require_action("efe.action_history", action, num_actions)?;
    }
    Ok(())
}

fn validate_td(
    td: &TemporalDifferenceLearner,
    state_dim: usize,
    obs_dim: usize,
    num_actions: usize,
) -> Result<(), ActiveInferenceAgentSnapshotErrorV1> {
    require_finite("td.current_learning_rate", td.current_learning_rate)?;
    require_nonnegative("td.current_learning_rate", td.current_learning_rate)?;
    require_finite("td.avg_td_error", td.avg_td_error)?;
    require_finite("td.avg_prediction_accuracy", td.avg_prediction_accuracy)?;
    require_len("td.value_weights", td.value_weights.len(), state_dim)?;
    require_finite_slice("td.value_weights", &td.value_weights)?;
    require_finite("td.value_bias", td.value_bias)?;
    require_finite("td.prev_value", td.prev_value)?;

    require_finite("td.config.initial_learning_rate", td.config.initial_learning_rate)?;
    require_nonnegative("td.config.initial_learning_rate", td.config.initial_learning_rate)?;
    require_finite("td.config.min_learning_rate", td.config.min_learning_rate)?;
    require_nonnegative("td.config.min_learning_rate", td.config.min_learning_rate)?;
    require_finite("td.config.learning_rate_decay", td.config.learning_rate_decay)?;
    require_nonnegative("td.config.learning_rate_decay", td.config.learning_rate_decay)?;
    require_finite("td.config.gamma", td.config.gamma)?;
    require_nonnegative("td.config.gamma", td.config.gamma)?;
    require_finite("td.config.lambda", td.config.lambda)?;
    require_nonnegative("td.config.lambda", td.config.lambda)?;
    require_finite("td.config.trace_decay", td.config.trace_decay)?;
    require_nonnegative("td.config.trace_decay", td.config.trace_decay)?;
    require_nonzero("td.config.max_transition_history", td.config.max_transition_history)?;
    require_finite("td.config.confidence_decay", td.config.confidence_decay)?;
    require_nonnegative("td.config.confidence_decay", td.config.confidence_decay)?;
    require_positive("td.config.min_confidence", td.config.min_confidence)?;

    if td.transition_history.len() > td.config.max_transition_history {
        return Err(ActiveInferenceAgentSnapshotErrorV1::HistoryTooLong {
            field: "td.transition_history",
            len: td.transition_history.len(),
            max: td.config.max_transition_history,
        });
    }
    for transition in &td.transition_history {
        validate_transition(transition, state_dim, obs_dim, num_actions)?;
    }

    let traces_present = td.eligibility_traces.is_some();
    if traces_present != td.config.use_eligibility_traces {
        return Err(
            ActiveInferenceAgentSnapshotErrorV1::EligibilityPresenceMismatch {
                enabled: td.config.use_eligibility_traces,
                traces_present,
            },
        );
    }
    if let Some(traces) = &td.eligibility_traces {
        require_cube(
            "td.eligibility.transition_traces",
            &traces.transition_traces,
            num_actions,
            state_dim,
            state_dim,
        )?;
        require_matrix(
            "td.eligibility.likelihood_traces",
            &traces.likelihood_traces,
            state_dim,
            obs_dim,
        )?;
        require_finite("td.eligibility.lambda", traces.lambda)?;
        require_finite("td.eligibility.gamma", traces.gamma)?;
    }

    let confidence = &td.confidence_tracker;
    require_cube(
        "td.confidence.transition_confidence",
        &confidence.transition_confidence,
        num_actions,
        state_dim,
        state_dim,
    )?;
    require_matrix(
        "td.confidence.likelihood_confidence",
        &confidence.likelihood_confidence,
        state_dim,
        obs_dim,
    )?;
    require_u64_cube(
        "td.confidence.transition_counts",
        &confidence.transition_counts,
        num_actions,
        state_dim,
        state_dim,
    )?;
    require_u64_matrix(
        "td.confidence.likelihood_counts",
        &confidence.likelihood_counts,
        state_dim,
        obs_dim,
    )?;
    require_finite("td.confidence.decay_rate", confidence.decay_rate)?;
    require_nonnegative("td.confidence.decay_rate", confidence.decay_rate)?;
    require_positive("td.confidence.min_confidence", confidence.min_confidence)?;
    Ok(())
}

fn validate_transition(
    transition: &StateTransition,
    state_dim: usize,
    obs_dim: usize,
    num_actions: usize,
) -> Result<(), ActiveInferenceAgentSnapshotErrorV1> {
    validate_hidden_state("td.transition.old_state", &transition.old_state, state_dim)?;
    validate_hidden_state("td.transition.new_state", &transition.new_state, state_dim)?;
    require_action("td.transition.action", transition.action, num_actions)?;
    require_len("td.transition.observation", transition.observation.values.len(), obs_dim)?;
    require_finite_slice("td.transition.observation", &transition.observation.values)?;
    require_positive("td.transition.observation.precision", transition.observation.precision)?;
    require_finite("td.transition.td_error", transition.td_error)
}

fn validate_fe_components(
    field: &'static str,
    fe: &FreeEnergyComponents,
) -> Result<(), ActiveInferenceAgentSnapshotErrorV1> {
    require_finite(field, fe.total)?;
    require_finite(field, fe.accuracy)?;
    require_finite(field, fe.complexity)?;
    require_finite(field, fe.surprise)?;
    require_finite(field, fe.prediction_error)
}

fn validate_stats(
    stats: &ActiveInferenceAgentStats,
) -> Result<(), ActiveInferenceAgentSnapshotErrorV1> {
    require_finite("stats.avg_free_energy", stats.avg_free_energy)?;
    require_finite("stats.avg_prediction_error", stats.avg_prediction_error)?;
    require_finite("stats.avg_precision", stats.avg_precision)?;
    require_finite("stats.exploration_rate", stats.exploration_rate)?;
    require_finite("stats.avg_td_error", stats.avg_td_error)?;
    require_finite("stats.transition_accuracy", stats.transition_accuracy)?;
    Ok(())
}

fn require_matrix(
    field: &'static str,
    matrix: &[Vec<f64>],
    rows: usize,
    cols: usize,
) -> Result<(), ActiveInferenceAgentSnapshotErrorV1> {
    require_len(field, matrix.len(), rows)?;
    for row in matrix {
        require_len(field, row.len(), cols)?;
        require_finite_slice(field, row)?;
    }
    Ok(())
}

fn require_cube(
    field: &'static str,
    cube: &[Vec<Vec<f64>>],
    outer: usize,
    rows: usize,
    cols: usize,
) -> Result<(), ActiveInferenceAgentSnapshotErrorV1> {
    require_len(field, cube.len(), outer)?;
    for matrix in cube {
        require_matrix(field, matrix, rows, cols)?;
    }
    Ok(())
}

fn require_u64_matrix(
    field: &'static str,
    matrix: &[Vec<u64>],
    rows: usize,
    cols: usize,
) -> Result<(), ActiveInferenceAgentSnapshotErrorV1> {
    require_len(field, matrix.len(), rows)?;
    for row in matrix {
        require_len(field, row.len(), cols)?;
    }
    Ok(())
}

fn require_u64_cube(
    field: &'static str,
    cube: &[Vec<Vec<u64>>],
    outer: usize,
    rows: usize,
    cols: usize,
) -> Result<(), ActiveInferenceAgentSnapshotErrorV1> {
    require_len(field, cube.len(), outer)?;
    for matrix in cube {
        require_u64_matrix(field, matrix, rows, cols)?;
    }
    Ok(())
}

fn require_len(
    field: &'static str,
    observed: usize,
    expected: usize,
) -> Result<(), ActiveInferenceAgentSnapshotErrorV1> {
    if observed != expected {
        return Err(ActiveInferenceAgentSnapshotErrorV1::LengthMismatch {
            field,
            expected,
            observed,
        });
    }
    Ok(())
}

fn require_action(
    field: &'static str,
    action: usize,
    num_actions: usize,
) -> Result<(), ActiveInferenceAgentSnapshotErrorV1> {
    if action >= num_actions {
        return Err(ActiveInferenceAgentSnapshotErrorV1::ActionOutOfBounds {
            field,
            action,
            num_actions,
        });
    }
    Ok(())
}

fn require_nonzero(
    field: &'static str,
    value: usize,
) -> Result<(), ActiveInferenceAgentSnapshotErrorV1> {
    if value == 0 {
        return Err(ActiveInferenceAgentSnapshotErrorV1::ZeroDimension { field });
    }
    Ok(())
}

fn require_finite_slice(
    field: &'static str,
    values: &[f64],
) -> Result<(), ActiveInferenceAgentSnapshotErrorV1> {
    for &value in values {
        require_finite(field, value)?;
    }
    Ok(())
}

fn require_finite(
    field: &'static str,
    value: f64,
) -> Result<(), ActiveInferenceAgentSnapshotErrorV1> {
    if !value.is_finite() {
        return Err(ActiveInferenceAgentSnapshotErrorV1::NonFinite { field });
    }
    Ok(())
}

fn require_positive(
    field: &'static str,
    value: f64,
) -> Result<(), ActiveInferenceAgentSnapshotErrorV1> {
    require_finite(field, value)?;
    if value <= 0.0 {
        return Err(ActiveInferenceAgentSnapshotErrorV1::NonPositive { field });
    }
    Ok(())
}

fn require_nonnegative(
    field: &'static str,
    value: f64,
) -> Result<(), ActiveInferenceAgentSnapshotErrorV1> {
    require_finite(field, value)?;
    if value < 0.0 {
        return Err(ActiveInferenceAgentSnapshotErrorV1::Negative { field });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_snapshot() -> ActiveInferenceAgentSnapshotV1 {
        let config = ActiveInferenceAgentConfig {
            state_dim: 2,
            obs_dim: 2,
            num_actions: 2,
            ..Default::default()
        };
        ActiveInferenceAgentSnapshotV1 {
            belief: HiddenState::new(config.state_dim),
            previous_state: None,
            last_action: None,
            model: GenerativeModel::new(config.state_dim, config.obs_dim, config.num_actions),
            free_energy_calc: FreeEnergyCalculator::new(500),
            precision: PrecisionEstimator::new(),
            efe_computer: ExpectedFreeEnergyComputer::new(config.obs_dim),
            td_learner: Some(TemporalDifferenceLearner::new(
                config.td_config.clone(),
                config.num_actions,
                config.state_dim,
                config.obs_dim,
            )),
            last_fe_components: None,
            stats: ActiveInferenceAgentStats::default(),
            timestamp: 0,
            rng_state: 0x9E37_79B9_7F4A_7C15,
            config,
        }
    }

    #[test]
    fn valid_snapshot_creates_nonserializable_validation_capability() {
        let validated = valid_snapshot().validate().expect("valid snapshot");
        assert_eq!(validated.timestamp(), 0);
        assert_eq!(validated.perception_cycles(), 0);
    }

    #[test]
    fn zero_rng_state_fails_closed() {
        let mut snapshot = valid_snapshot();
        snapshot.rng_state = 0;
        assert_eq!(
            snapshot.validate().unwrap_err(),
            ActiveInferenceAgentSnapshotErrorV1::ZeroRngState
        );
    }

    #[test]
    fn timestamp_must_match_perception_cycle_authority() {
        let mut snapshot = valid_snapshot();
        snapshot.timestamp = 3;
        snapshot.stats.perception_cycles = 2;
        assert_eq!(
            snapshot.validate().unwrap_err(),
            ActiveInferenceAgentSnapshotErrorV1::TimestampPerceptionMismatch {
                timestamp: 3,
                perception_cycles: 2,
            }
        );
    }

    #[test]
    fn impossible_model_shape_fails_before_capability_creation() {
        let mut snapshot = valid_snapshot();
        snapshot.model.likelihood_matrix.pop();
        assert!(matches!(
            snapshot.validate(),
            Err(ActiveInferenceAgentSnapshotErrorV1::LengthMismatch {
                field: "model.likelihood_matrix",
                ..
            })
        ));
    }

    #[test]
    fn invalid_action_history_fails_closed() {
        let mut snapshot = valid_snapshot();
        snapshot.efe_computer.action_history.push_back(snapshot.config.num_actions);
        assert_eq!(
            snapshot.validate().unwrap_err(),
            ActiveInferenceAgentSnapshotErrorV1::ActionOutOfBounds {
                field: "efe.action_history",
                action: 2,
                num_actions: 2,
            }
        );
    }

    #[test]
    fn td_presence_must_match_agent_configuration() {
        let mut snapshot = valid_snapshot();
        snapshot.td_learner = None;
        assert_eq!(
            snapshot.validate().unwrap_err(),
            ActiveInferenceAgentSnapshotErrorV1::TdPresenceMismatch {
                enabled: true,
                learner_present: false,
            }
        );
    }

    #[test]
    fn persisted_snapshot_type_does_not_bypass_validation_after_clone() {
        let raw = valid_snapshot();
        let raw_copy = raw.clone();
        let first = raw.validate().expect("first validation");
        let second = raw_copy.validate().expect("second validation");
        assert_eq!(first.timestamp(), second.timestamp());
        assert_eq!(first.perception_cycles(), second.perception_cycles());
    }
}
