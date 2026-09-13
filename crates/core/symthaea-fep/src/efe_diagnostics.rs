// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Measurement-only common-currency diagnostics for expected free energy.
//!
//! V1 composes three canonical quantities in natural-log information units:
//!
//! - relative pragmatic preference cost in nats;
//! - expected hidden-state information gain in nats;
//! - expected transition-parameter information gain in nats.
//!
//! The historical committed-action frequency bonus is exposed separately as a
//! unitless legacy heuristic and is deliberately excluded from the canonical
//! diagnostic score. Nothing in this module changes live action scoring, policy
//! probabilities, RNG state, commitment history, beliefs, or learned evidence.

use crate::free_energy::ExpectedFreeEnergyComputer;
use crate::generative_model::GenerativeModel;
use crate::state_information_gain::{
    HiddenStateInformationGainError, linear_gaussian_hidden_state_information_gain_nats_v1,
};
use crate::td_learning::ModelConfidenceTracker;
use crate::transition_information_gain::TransitionParameterInformationGainError;
use crate::types::HiddenState;

/// Schema identity for the measurement-only common-currency decomposition.
pub const EXPECTED_FREE_ENERGY_DIAGNOSTIC_SCHEMA_V1: u16 = 1;

/// One action's measurement-only expected-free-energy decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpectedFreeEnergyDecompositionV1 {
    pub schema_version: u16,
    pub action: usize,
    /// Exact source-state identity used to query transition-parameter evidence.
    pub transition_from_state: usize,
    /// Relative negative log preference likelihood in nats.
    ///
    /// V1 uses `0.5 * precision * squared_error` per preferred observation
    /// dimension and omits Gaussian normalization terms because preference
    /// precisions are fixed across candidate actions. A zero precision means the
    /// dimension carries no preference and contributes exactly zero.
    pub pragmatic_cost_nats: f64,
    /// Expected information gain about the future hidden state from the future
    /// observation under this action.
    pub hidden_state_information_gain_nats: f64,
    /// Expected information gain about this action/source-state transition row's
    /// unknown categorical parameters from one additional confirmed outcome.
    pub transition_parameter_information_gain_nats: f64,
    /// Canonical measurement-only minimization score in nats.
    ///
    /// Positive information gain lowers the score by construction:
    ///
    /// `pragmatic_cost - hidden_state_IG - transition_parameter_IG`.
    ///
    /// This field is diagnostic only and is not consumed by live policy.
    pub canonical_diagnostic_score_nats: f64,
    /// Historical action-frequency exploration bonus, explicitly unitless and
    /// excluded from `canonical_diagnostic_score_nats`.
    pub legacy_exploration_heuristic_unitless: f64,
    /// The predicted observation used to compute the pragmatic term, retained so
    /// the diagnostic can be independently recomputed from an evidence receipt.
    pub expected_observation: Vec<f64>,
}

/// Fail-closed errors for V1 common-currency diagnostics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExpectedFreeEnergyDiagnosticError {
    ActionOutOfRange {
        action: usize,
        num_actions: usize,
    },
    TransitionFromStateOutOfRange {
        from_state: usize,
        state_dim: usize,
    },
    PreferenceLengthMismatch {
        expected: usize,
        actual: usize,
    },
    PrecisionOverrideLengthMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidPreferenceValue {
        index: usize,
        value: f64,
    },
    InvalidPreferencePrecision {
        index: usize,
        value: f64,
    },
    NonFiniteExpectedObservation {
        index: usize,
        value: f64,
    },
    TransitionEvidenceActionCountMismatch {
        expected: usize,
        actual: usize,
    },
    TransitionEvidenceSourceStateCountMismatch {
        action: usize,
        expected: usize,
        actual: usize,
    },
    TransitionEvidenceOutcomeCountMismatch {
        action: usize,
        from_state: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteCanonicalComponent,
    HiddenStateInformationGain(HiddenStateInformationGainError),
    TransitionParameterInformationGain(TransitionParameterInformationGainError),
}

fn pragmatic_relative_negative_log_likelihood_nats_v1(
    expected_observation: &[f64],
    efe: &ExpectedFreeEnergyComputer,
) -> Result<f64, ExpectedFreeEnergyDiagnosticError> {
    if efe.preferences.len() != expected_observation.len() {
        return Err(ExpectedFreeEnergyDiagnosticError::PreferenceLengthMismatch {
            expected: expected_observation.len(),
            actual: efe.preferences.len(),
        });
    }

    let precision_overrides = if let Some(overrides) = efe.precision_overrides.as_ref() {
        if overrides.len() != efe.preferences.len() {
            return Err(
                ExpectedFreeEnergyDiagnosticError::PrecisionOverrideLengthMismatch {
                    expected: efe.preferences.len(),
                    actual: overrides.len(),
                },
            );
        }
        Some(overrides.as_slice())
    } else {
        None
    };

    let mut cost = 0.0;
    for (index, (&observation, &preference)) in expected_observation
        .iter()
        .zip(efe.preferences.iter())
        .enumerate()
    {
        if !observation.is_finite() {
            return Err(
                ExpectedFreeEnergyDiagnosticError::NonFiniteExpectedObservation {
                    index,
                    value: observation,
                },
            );
        }
        if !preference.is_finite() {
            return Err(ExpectedFreeEnergyDiagnosticError::InvalidPreferenceValue {
                index,
                value: preference,
            });
        }

        let precision = precision_overrides
            .map(|overrides| overrides[index])
            .unwrap_or(efe.preference_precision);
        if !precision.is_finite() || precision < 0.0 {
            return Err(
                ExpectedFreeEnergyDiagnosticError::InvalidPreferencePrecision {
                    index,
                    value: precision,
                },
            );
        }

        let error = observation - preference;
        cost += 0.5 * precision * error * error;
    }

    if !cost.is_finite() {
        return Err(ExpectedFreeEnergyDiagnosticError::NonFiniteCanonicalComponent);
    }
    Ok(cost)
}

fn validate_transition_evidence_alignment(
    evidence: &ModelConfidenceTracker,
    model: &GenerativeModel,
    action: usize,
    from_state: usize,
) -> Result<(), ExpectedFreeEnergyDiagnosticError> {
    if evidence.transition_counts.len() != model.num_actions {
        return Err(
            ExpectedFreeEnergyDiagnosticError::TransitionEvidenceActionCountMismatch {
                expected: model.num_actions,
                actual: evidence.transition_counts.len(),
            },
        );
    }

    let action_rows = &evidence.transition_counts[action];
    if action_rows.len() != model.state_dim {
        return Err(
            ExpectedFreeEnergyDiagnosticError::TransitionEvidenceSourceStateCountMismatch {
                action,
                expected: model.state_dim,
                actual: action_rows.len(),
            },
        );
    }

    let outcome_counts = &action_rows[from_state];
    if outcome_counts.len() != model.state_dim {
        return Err(
            ExpectedFreeEnergyDiagnosticError::TransitionEvidenceOutcomeCountMismatch {
                action,
                from_state,
                expected: model.state_dim,
                actual: outcome_counts.len(),
            },
        );
    }

    Ok(())
}

/// Compute the V1 common-currency decomposition for one action without changing
/// live expected-free-energy behavior.
///
/// `transition_from_state` is explicit rather than inferred from the belief. It
/// must identify the exact evidence row whose transition-parameter uncertainty
/// is being queried; latent-state decoding/tie-breaking is outside this theorem.
pub fn expected_free_energy_decomposition_v1(
    action: usize,
    transition_from_state: usize,
    state: &HiddenState,
    model: &GenerativeModel,
    transition_evidence: &ModelConfidenceTracker,
    live_efe: &ExpectedFreeEnergyComputer,
) -> Result<ExpectedFreeEnergyDecompositionV1, ExpectedFreeEnergyDiagnosticError> {
    if action >= model.num_actions {
        return Err(ExpectedFreeEnergyDiagnosticError::ActionOutOfRange {
            action,
            num_actions: model.num_actions,
        });
    }
    if transition_from_state >= model.state_dim {
        return Err(
            ExpectedFreeEnergyDiagnosticError::TransitionFromStateOutOfRange {
                from_state: transition_from_state,
                state_dim: model.state_dim,
            },
        );
    }

    validate_transition_evidence_alignment(
        transition_evidence,
        model,
        action,
        transition_from_state,
    )?;

    let hidden_state_information_gain_nats =
        linear_gaussian_hidden_state_information_gain_nats_v1(state, model, action)
            .map_err(ExpectedFreeEnergyDiagnosticError::HiddenStateInformationGain)?;

    let transition_parameter_information_gain_nats = transition_evidence
        .transition_parameter_information_gain_nats_v1(action, transition_from_state)
        .map_err(ExpectedFreeEnergyDiagnosticError::TransitionParameterInformationGain)?;

    let predicted_state = model.predict_next_state(state, action);
    let expected_observation = model.predict_observation(&predicted_state);
    let pragmatic_cost_nats =
        pragmatic_relative_negative_log_likelihood_nats_v1(&expected_observation, live_efe)?;

    let canonical_diagnostic_score_nats = pragmatic_cost_nats
        - hidden_state_information_gain_nats
        - transition_parameter_information_gain_nats;
    if !canonical_diagnostic_score_nats.is_finite()
        || !hidden_state_information_gain_nats.is_finite()
        || !transition_parameter_information_gain_nats.is_finite()
    {
        return Err(ExpectedFreeEnergyDiagnosticError::NonFiniteCanonicalComponent);
    }

    let committed_count = live_efe
        .action_history
        .iter()
        .filter(|candidate| **candidate == action)
        .count();
    let legacy_exploration_heuristic_unitless = 1.0 / (1.0 + committed_count as f64);

    Ok(ExpectedFreeEnergyDecompositionV1 {
        schema_version: EXPECTED_FREE_ENERGY_DIAGNOSTIC_SCHEMA_V1,
        action,
        transition_from_state,
        pragmatic_cost_nats,
        hidden_state_information_gain_nats,
        transition_parameter_information_gain_nats,
        canonical_diagnostic_score_nats,
        legacy_exploration_heuristic_unitless,
        expected_observation,
    })
}
