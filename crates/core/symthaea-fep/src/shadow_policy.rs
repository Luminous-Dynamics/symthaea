// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Read-only shadow comparison between the live legacy FEP policy and the
//! common-currency diagnostic policy.
//!
//! This module is deliberately observational. It accepts `&ActiveInferenceAgent`,
//! samples no action, mutates no RNG/history/model/belief state, and returns a
//! typed `Unavailable` result when the canonical evidence path cannot be bound.

use crate::agent::ActiveInferenceAgent;
use crate::efe_diagnostics::{
    ExpectedFreeEnergyDecompositionV1, ExpectedFreeEnergyDiagnosticError,
    expected_free_energy_decomposition_v1,
};
use crate::efe_policy_probe::{
    DiagnosticPolicyProbeError, DiagnosticPolicyProbeV1, diagnostic_policy_probabilities_v1,
};
use crate::types::HiddenState;

/// Schema identity for the shadow-policy receipt.
pub const FEP_SHADOW_POLICY_SCHEMA_V1: u16 = 1;
/// Schema identity for the transition-evidence source-state adapter.
pub const TRANSITION_EVIDENCE_SOURCE_STATE_SCHEMA_V1: u16 = 1;

/// Explicit identity of the transition-evidence source-state row used by the
/// common-currency parameter-information term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransitionEvidenceSourceStateV1 {
    pub schema_version: u16,
    pub index: usize,
}

/// One action under the unchanged live/legacy EFE score and deterministic
/// pre-sampling probability transform.
#[derive(Debug, Clone, PartialEq)]
pub struct LegacyShadowActionV1 {
    pub action: usize,
    pub expected_free_energy: f64,
    pub probability: f64,
}

/// Available shadow receipt from one immutable pre-decision snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct FepShadowPolicyReceiptV1 {
    pub schema_version: u16,
    pub transition_source_state: TransitionEvidenceSourceStateV1,
    pub action_temperature: f64,
    pub legacy_actions: Vec<LegacyShadowActionV1>,
    pub canonical_components: Vec<ExpectedFreeEnergyDecompositionV1>,
    pub canonical_policy: DiagnosticPolicyProbeV1,
}

/// Typed reasons why canonical shadow telemetry cannot be produced.
#[derive(Debug, Clone, PartialEq)]
pub enum FepShadowPolicyUnavailableV1 {
    EmptyStateDomain,
    BeliefDimensionMismatch { expected: usize, actual: usize },
    NonFiniteBeliefMean { index: usize, value: f64 },
    EmptyActionDomain,
    ModelActionCountMismatch { configured: usize, model: usize },
    MissingTransitionEvidence,
    InvalidActionTemperature { value: f64 },
    NonFiniteLegacyScore { action: usize, value: f64 },
    InvalidLegacyNormalization { sum: f64 },
    Diagnostic(ExpectedFreeEnergyDiagnosticError),
    PolicyProbe(DiagnosticPolicyProbeError),
}

/// Shadow evaluation is always a typed telemetry result rather than a mutation
/// or implicit fallback to fabricated evidence.
#[derive(Debug, Clone, PartialEq)]
pub enum FepShadowPolicyEvaluationV1 {
    Available(FepShadowPolicyReceiptV1),
    Unavailable(FepShadowPolicyUnavailableV1),
}

/// Reproduce the finite-state source-row semantics of the existing TD evidence
/// update: choose the **last** maximal hidden-state mean component.
///
/// V1 deliberately tightens malformed-state handling relative to the legacy TD
/// implementation: empty or non-finite state vectors fail closed rather than
/// silently becoming evidence row zero. For valid finite states, ties retain the
/// existing last-max behavior.
pub fn transition_evidence_source_state_v1(
    state: &HiddenState,
    expected_state_dim: usize,
) -> Result<TransitionEvidenceSourceStateV1, FepShadowPolicyUnavailableV1> {
    if expected_state_dim == 0 {
        return Err(FepShadowPolicyUnavailableV1::EmptyStateDomain);
    }
    if state.mean.len() != expected_state_dim {
        return Err(FepShadowPolicyUnavailableV1::BeliefDimensionMismatch {
            expected: expected_state_dim,
            actual: state.mean.len(),
        });
    }

    for (index, value) in state.mean.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(FepShadowPolicyUnavailableV1::NonFiniteBeliefMean {
                index,
                value,
            });
        }
    }

    let mut best_index = 0usize;
    let mut best_value = state.mean[0];
    for (index, value) in state.mean.iter().copied().enumerate().skip(1) {
        // `max_by` returns the later element on equality; `>=` preserves that
        // production tie behavior for finite values.
        if value >= best_value {
            best_value = value;
            best_index = index;
        }
    }

    Ok(TransitionEvidenceSourceStateV1 {
        schema_version: TRANSITION_EVIDENCE_SOURCE_STATE_SCHEMA_V1,
        index: best_index,
    })
}

fn legacy_shadow_probabilities_v1(
    agent: &ActiveInferenceAgent,
) -> Result<Vec<LegacyShadowActionV1>, FepShadowPolicyUnavailableV1> {
    if agent.config.num_actions == 0 {
        return Err(FepShadowPolicyUnavailableV1::EmptyActionDomain);
    }
    if agent.model.num_actions != agent.config.num_actions {
        return Err(FepShadowPolicyUnavailableV1::ModelActionCountMismatch {
            configured: agent.config.num_actions,
            model: agent.model.num_actions,
        });
    }
    let temperature = agent.config.action_temperature;
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(FepShadowPolicyUnavailableV1::InvalidActionTemperature {
            value: temperature,
        });
    }

    let mut scores = Vec::with_capacity(agent.config.num_actions);
    let mut minimum = f64::INFINITY;
    for action in 0..agent.config.num_actions {
        let result = agent
            .efe_computer
            .compute(action, &agent.belief, &agent.model);
        if !result.total.is_finite() {
            return Err(FepShadowPolicyUnavailableV1::NonFiniteLegacyScore {
                action,
                value: result.total,
            });
        }
        minimum = minimum.min(result.total);
        scores.push((action, result.total));
    }

    // Algebraically identical to the live agent's stabilized softmax over -G.
    let masses: Vec<f64> = scores
        .iter()
        .map(|(_, score)| (-(score - minimum) / temperature).exp())
        .collect();
    let sum: f64 = masses.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(FepShadowPolicyUnavailableV1::InvalidLegacyNormalization { sum });
    }

    Ok(scores
        .into_iter()
        .zip(masses)
        .map(|((action, expected_free_energy), mass)| LegacyShadowActionV1 {
            action,
            expected_free_energy,
            probability: mass / sum,
        })
        .collect())
}

/// Produce legacy and canonical policy telemetry from one immutable live-agent
/// snapshot without affecting behavior.
pub fn shadow_policy_evaluation_v1(agent: &ActiveInferenceAgent) -> FepShadowPolicyEvaluationV1 {
    let transition_source_state = match transition_evidence_source_state_v1(
        &agent.belief,
        agent.model.state_dim,
    ) {
        Ok(value) => value,
        Err(reason) => return FepShadowPolicyEvaluationV1::Unavailable(reason),
    };

    let transition_evidence = match agent.td_learner.as_ref() {
        Some(learner) => &learner.confidence_tracker,
        None => {
            return FepShadowPolicyEvaluationV1::Unavailable(
                FepShadowPolicyUnavailableV1::MissingTransitionEvidence,
            );
        }
    };

    let legacy_actions = match legacy_shadow_probabilities_v1(agent) {
        Ok(value) => value,
        Err(reason) => return FepShadowPolicyEvaluationV1::Unavailable(reason),
    };

    let mut canonical_components = Vec::with_capacity(agent.config.num_actions);
    for action in 0..agent.config.num_actions {
        let component = match expected_free_energy_decomposition_v1(
            action,
            transition_source_state.index,
            &agent.belief,
            &agent.model,
            transition_evidence,
            &agent.efe_computer,
        ) {
            Ok(value) => value,
            Err(error) => {
                return FepShadowPolicyEvaluationV1::Unavailable(
                    FepShadowPolicyUnavailableV1::Diagnostic(error),
                );
            }
        };
        canonical_components.push(component);
    }

    let canonical_policy = match diagnostic_policy_probabilities_v1(
        &canonical_components,
        agent.config.action_temperature,
    ) {
        Ok(value) => value,
        Err(error) => {
            return FepShadowPolicyEvaluationV1::Unavailable(
                FepShadowPolicyUnavailableV1::PolicyProbe(error),
            );
        }
    };

    FepShadowPolicyEvaluationV1::Available(FepShadowPolicyReceiptV1 {
        schema_version: FEP_SHADOW_POLICY_SCHEMA_V1,
        transition_source_state,
        action_temperature: agent.config.action_temperature,
        legacy_actions,
        canonical_components,
        canonical_policy,
    })
}
