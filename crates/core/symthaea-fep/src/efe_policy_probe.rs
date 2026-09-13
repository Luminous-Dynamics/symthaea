// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic policy-probability probe for common-currency EFE diagnostics.
//!
//! This module transforms already-computed diagnostic minimization scores into
//! softmax probabilities. It is deliberately disconnected from the live active-
//! inference agent: no action is sampled, no RNG is touched, and no policy state
//! or evidence is mutated.

use std::collections::HashSet;

use crate::efe_diagnostics::{
    EXPECTED_FREE_ENERGY_DIAGNOSTIC_SCHEMA_V1, ExpectedFreeEnergyDecompositionV1,
};

/// Schema identity for the deterministic policy-probe transform.
pub const EFE_POLICY_PROBE_SCHEMA_V1: u16 = 1;

/// One action's probability under the deterministic diagnostic softmax.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagnosticActionProbabilityV1 {
    pub action: usize,
    pub canonical_score_nats: f64,
    pub probability: f64,
}

/// Deterministic probability-vector receipt.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagnosticPolicyProbeV1 {
    pub schema_version: u16,
    /// Positive finite softmax temperature. At `1.0`, log odds are exactly the
    /// negative difference in canonical diagnostic scores.
    pub action_temperature: f64,
    pub actions: Vec<DiagnosticActionProbabilityV1>,
}

/// Fail-closed validation errors for V1 policy-probe queries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiagnosticPolicyProbeError {
    EmptyCandidateSet,
    InvalidTemperature { value: f64 },
    UnsupportedDiagnosticSchema {
        action: usize,
        expected: u16,
        actual: u16,
    },
    DuplicateAction { action: usize },
    NonFiniteCanonicalScore { action: usize, value: f64 },
    InvalidNormalization { sum: f64 },
}

/// Convert common-currency diagnostic minimization scores into deterministic
/// softmax probabilities.
///
/// The transform matches the live agent's sign convention:
///
/// `P(a) ∝ exp(-G(a) / temperature)`.
///
/// Candidate order cannot affect the mathematical result; action identity is
/// carried explicitly in the receipt. This function does not sample an action.
pub fn diagnostic_policy_probabilities_v1(
    candidates: &[ExpectedFreeEnergyDecompositionV1],
    action_temperature: f64,
) -> Result<DiagnosticPolicyProbeV1, DiagnosticPolicyProbeError> {
    if candidates.is_empty() {
        return Err(DiagnosticPolicyProbeError::EmptyCandidateSet);
    }
    if !action_temperature.is_finite() || action_temperature <= 0.0 {
        return Err(DiagnosticPolicyProbeError::InvalidTemperature {
            value: action_temperature,
        });
    }

    let mut seen_actions = HashSet::with_capacity(candidates.len());
    let mut minimum_score = f64::INFINITY;
    for candidate in candidates {
        if candidate.schema_version != EXPECTED_FREE_ENERGY_DIAGNOSTIC_SCHEMA_V1 {
            return Err(DiagnosticPolicyProbeError::UnsupportedDiagnosticSchema {
                action: candidate.action,
                expected: EXPECTED_FREE_ENERGY_DIAGNOSTIC_SCHEMA_V1,
                actual: candidate.schema_version,
            });
        }
        if !seen_actions.insert(candidate.action) {
            return Err(DiagnosticPolicyProbeError::DuplicateAction {
                action: candidate.action,
            });
        }
        if !candidate.canonical_diagnostic_score_nats.is_finite() {
            return Err(DiagnosticPolicyProbeError::NonFiniteCanonicalScore {
                action: candidate.action,
                value: candidate.canonical_diagnostic_score_nats,
            });
        }
        minimum_score = minimum_score.min(candidate.canonical_diagnostic_score_nats);
    }

    // Stable softmax of negative minimization scores. Subtracting the minimum
    // score keeps every exponent <= 0 without changing ratios.
    let unnormalized: Vec<f64> = candidates
        .iter()
        .map(|candidate| {
            (-(candidate.canonical_diagnostic_score_nats - minimum_score) / action_temperature)
                .exp()
        })
        .collect();
    let sum: f64 = unnormalized.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(DiagnosticPolicyProbeError::InvalidNormalization { sum });
    }

    let actions = candidates
        .iter()
        .zip(unnormalized)
        .map(|(candidate, mass)| DiagnosticActionProbabilityV1 {
            action: candidate.action,
            canonical_score_nats: candidate.canonical_diagnostic_score_nats,
            probability: mass / sum,
        })
        .collect();

    Ok(DiagnosticPolicyProbeV1 {
        schema_version: EFE_POLICY_PROBE_SCHEMA_V1,
        action_temperature,
        actions,
    })
}
