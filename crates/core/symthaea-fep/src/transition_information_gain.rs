// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Measurement-only transition-parameter information gain.
//!
//! This module estimates one-step expected information gain about a categorical
//! transition row under a fixed symmetric Dirichlet(1) prior. It is deliberately
//! disconnected from expected-free-energy scoring and policy selection: this is
//! parameter uncertainty ("what would happen if I did that?"), not hidden-state
//! salience, pragmatic value, action novelty history, or evidence of agency.

use crate::td_learning::ModelConfidenceTracker;

/// Schema identity for the measurement-only V1 estimator.
pub const TRANSITION_PARAMETER_INFORMATION_GAIN_SCHEMA_V1: u16 = 1;

/// The V1 estimator uses one unit pseudocount for every possible successor.
///
/// Keeping this fixed avoids turning the first construct-validity tranche into a
/// tunable exploration bonus. A future estimator may expose a provenance-bound
/// prior, but doing so requires a separate calibration theorem.
pub const TRANSITION_PARAMETER_INFORMATION_GAIN_PSEUDOCOUNT_V1: u64 = 1;

/// Fail-closed errors for transition-parameter information-gain queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionParameterInformationGainError {
    /// Requested action has no transition-evidence row.
    ActionOutOfRange {
        action: usize,
        num_actions: usize,
    },
    /// Requested source state has no transition-evidence row for the action.
    FromStateOutOfRange {
        from_state: usize,
        state_count: usize,
    },
    /// A transition row has no possible successor states.
    EmptyOutcomeDomain,
    /// Adding counts and fixed pseudocounts exceeded the internal exact integer domain.
    ConcentrationOverflow,
}

/// Harmonic number H_n for positive integer `n`.
///
/// Dirichlet(1 + count) concentrations are positive integers, which lets V1 use
/// the exact identity `psi(n + 1) = H_n - gamma` without depending on a special-
/// functions crate. Small n is summed directly; large n uses the Euler-Maclaurin
/// expansion, whose omitted terms are negligible at the switch point for this
/// measurement's f64 precision.
fn harmonic_number(n: u128) -> f64 {
    debug_assert!(n > 0);

    const DIRECT_SUM_LIMIT: u128 = 4096;
    if n <= DIRECT_SUM_LIMIT {
        return (1..=n).map(|k| 1.0 / k as f64).sum();
    }

    const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
    let x = n as f64;
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    let inv4 = inv2 * inv2;
    let inv6 = inv4 * inv2;
    let inv8 = inv4 * inv4;

    x.ln() + EULER_GAMMA + 0.5 * inv - inv2 / 12.0 + inv4 / 120.0 - inv6 / 252.0
        + inv8 / 240.0
}

/// Expected one-step information gain about one categorical transition row.
///
/// `outcome_counts[j]` is the number of confirmed transitions from one fixed
/// `(action, from_state)` pair into successor state `j`. V1 places a symmetric
/// Dirichlet(1) prior over the successor probabilities, then computes
/// `I(theta; next_state)` in nats for one additional observation.
///
/// For concentration vector `alpha`, with `A = sum(alpha)`:
///
/// ```text
/// IG = sum_j p_j * [ ln(A / alpha_j) + H(alpha_j) - H(A) ]
/// p_j = alpha_j / A
/// alpha_j = 1 + count_j
/// ```
///
/// This is query-only. It does not mutate counts, confidences, novelty history,
/// preferences, the generative model, or action-selection state.
pub fn dirichlet_transition_parameter_information_gain_nats_v1(
    outcome_counts: &[u64],
) -> Result<f64, TransitionParameterInformationGainError> {
    if outcome_counts.is_empty() {
        return Err(TransitionParameterInformationGainError::EmptyOutcomeDomain);
    }

    let mut concentrations = Vec::with_capacity(outcome_counts.len());
    let mut total: u128 = 0;
    for &count in outcome_counts {
        let alpha = u128::from(count)
            .checked_add(u128::from(
                TRANSITION_PARAMETER_INFORMATION_GAIN_PSEUDOCOUNT_V1,
            ))
            .ok_or(TransitionParameterInformationGainError::ConcentrationOverflow)?;
        total = total
            .checked_add(alpha)
            .ok_or(TransitionParameterInformationGainError::ConcentrationOverflow)?;
        concentrations.push(alpha);
    }

    let total_f = total as f64;
    let h_total = harmonic_number(total);
    let mut information_gain = 0.0;

    for alpha in concentrations {
        let alpha_f = alpha as f64;
        let predictive_probability = alpha_f / total_f;
        let conditional_kl =
            total_f.ln() - alpha_f.ln() + harmonic_number(alpha) - h_total;
        information_gain += predictive_probability * conditional_kl;
    }

    // The analytic quantity is non-negative. For very large concentrations,
    // floating-point cancellation can produce a tiny negative residue.
    Ok(information_gain.max(0.0))
}

impl ModelConfidenceTracker {
    /// Query V1 expected transition-parameter information gain for one action
    /// and source state, in nats.
    ///
    /// Unlike the legacy confidence-update path, invalid indices are never
    /// clamped to another action/state. Evidence identity is part of the theorem,
    /// so malformed queries fail closed.
    pub fn transition_parameter_information_gain_nats_v1(
        &self,
        action: usize,
        from_state: usize,
    ) -> Result<f64, TransitionParameterInformationGainError> {
        let action_rows = self.transition_counts.get(action).ok_or(
            TransitionParameterInformationGainError::ActionOutOfRange {
                action,
                num_actions: self.transition_counts.len(),
            },
        )?;
        let outcome_counts = action_rows.get(from_state).ok_or(
            TransitionParameterInformationGainError::FromStateOutOfRange {
                from_state,
                state_count: action_rows.len(),
            },
        )?;

        dirichlet_transition_parameter_information_gain_nats_v1(outcome_counts)
    }
}
