// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Measurement-only hidden-state information gain for the generic FEP model.
//!
//! V1 treats the current hidden-state belief as a diagonal Gaussian, propagates
//! its full covariance through one action-conditioned linear transition, adds the
//! scalar transition noise implied by `transition_precision`, and evaluates the
//! linear observation channel with scalar noise from `observation_precision`.
//!
//! This is deliberately disconnected from live expected-free-energy scoring. It
//! measures Gaussian hidden-state salience in nats; it is not parameter novelty,
//! pragmatic value, an action-frequency bonus, or evidence of agency.

use crate::{GenerativeModel, HiddenState};

/// Schema identity for the measurement-only linear-Gaussian salience estimator.
pub const HIDDEN_STATE_INFORMATION_GAIN_SCHEMA_V1: u16 = 1;

/// Fail-closed validation errors for V1 hidden-state information gain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HiddenStateInformationGainError {
    EmptyStateDomain,
    InternalActionStorageMismatch {
        expected: usize,
        actual: usize,
    },
    ActionOutOfRange {
        action: usize,
        num_actions: usize,
    },
    StatePrecisionLengthMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidStatePrecision {
        index: usize,
        value: f64,
    },
    InvalidTransitionPrecision {
        value: f64,
    },
    InvalidObservationPrecision {
        value: f64,
    },
    TransitionRowCountMismatch {
        expected: usize,
        actual: usize,
    },
    TransitionColumnCountMismatch {
        row: usize,
        expected: usize,
        actual: usize,
    },
    LikelihoodRowCountMismatch {
        expected: usize,
        actual: usize,
    },
    LikelihoodColumnCountMismatch {
        row: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteTransitionValue {
        row: usize,
        column: usize,
    },
    NonFiniteLikelihoodValue {
        row: usize,
        column: usize,
    },
    NonPositiveDefiniteObservationCovariance {
        index: usize,
        pivot: f64,
    },
}

/// Expected information gain about the next hidden state from one future
/// observation under action `action`, in nats.
///
/// V1 uses the linear-Gaussian identity
///
/// ```text
/// Sigma_next = B_a^T Sigma_current B_a + Q
/// I(s_next; o | a) = 1/2 * ln det(I + R^-1 A^T Sigma_next A)
/// ```
///
/// where:
/// - `Sigma_current = diag(1 / state.precision)`;
/// - `B_a` is the selected transition matrix;
/// - `Q = I / transition_precision`;
/// - `A` is the likelihood matrix;
/// - `R = I / observation_precision`.
///
/// The additive transition bias changes the predicted mean only and therefore
/// correctly does not enter this covariance-only information quantity.
///
/// This function is query-only and performs exact shape/finite checks rather
/// than silently truncating or clamping malformed model state.
pub fn linear_gaussian_hidden_state_information_gain_nats_v1(
    state: &HiddenState,
    model: &GenerativeModel,
    action: usize,
) -> Result<f64, HiddenStateInformationGainError> {
    let state_dim = model.state_dim;
    let obs_dim = model.obs_dim;

    if state_dim == 0 {
        return Err(HiddenStateInformationGainError::EmptyStateDomain);
    }
    if model.transition_matrices.len() != model.num_actions {
        return Err(
            HiddenStateInformationGainError::InternalActionStorageMismatch {
                expected: model.num_actions,
                actual: model.transition_matrices.len(),
            },
        );
    }
    let transition = model.transition_matrices.get(action).ok_or(
        HiddenStateInformationGainError::ActionOutOfRange {
            action,
            num_actions: model.num_actions,
        },
    )?;

    if state.precision.len() != state_dim {
        return Err(
            HiddenStateInformationGainError::StatePrecisionLengthMismatch {
                expected: state_dim,
                actual: state.precision.len(),
            },
        );
    }
    for (index, precision) in state.precision.iter().copied().enumerate() {
        if !precision.is_finite() || precision <= 0.0 {
            return Err(HiddenStateInformationGainError::InvalidStatePrecision {
                index,
                value: precision,
            });
        }
    }

    if !model.transition_precision.is_finite() || model.transition_precision <= 0.0 {
        return Err(
            HiddenStateInformationGainError::InvalidTransitionPrecision {
                value: model.transition_precision,
            },
        );
    }
    if !model.observation_precision.is_finite() || model.observation_precision <= 0.0 {
        return Err(
            HiddenStateInformationGainError::InvalidObservationPrecision {
                value: model.observation_precision,
            },
        );
    }

    if transition.len() != state_dim {
        return Err(
            HiddenStateInformationGainError::TransitionRowCountMismatch {
                expected: state_dim,
                actual: transition.len(),
            },
        );
    }
    for (row_index, row) in transition.iter().enumerate() {
        if row.len() != state_dim {
            return Err(
                HiddenStateInformationGainError::TransitionColumnCountMismatch {
                    row: row_index,
                    expected: state_dim,
                    actual: row.len(),
                },
            );
        }
        if let Some((column, _)) = row
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(HiddenStateInformationGainError::NonFiniteTransitionValue {
                row: row_index,
                column,
            });
        }
    }

    if model.likelihood_matrix.len() != state_dim {
        return Err(HiddenStateInformationGainError::LikelihoodRowCountMismatch {
            expected: state_dim,
            actual: model.likelihood_matrix.len(),
        });
    }
    for (row_index, row) in model.likelihood_matrix.iter().enumerate() {
        if row.len() != obs_dim {
            return Err(
                HiddenStateInformationGainError::LikelihoodColumnCountMismatch {
                    row: row_index,
                    expected: obs_dim,
                    actual: row.len(),
                },
            );
        }
        if let Some((column, _)) = row
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(HiddenStateInformationGainError::NonFiniteLikelihoodValue {
                row: row_index,
                column,
            });
        }
    }

    // No observation channel means no possible hidden-state information gain.
    if obs_dim == 0 {
        return Ok(0.0);
    }

    let current_variance: Vec<f64> = state.precision.iter().map(|p| 1.0 / p).collect();
    let process_variance = 1.0 / model.transition_precision;

    // Full covariance propagation: B^T diag(var) B + Q.
    let mut predicted_covariance = vec![vec![0.0; state_dim]; state_dim];
    for i in 0..state_dim {
        for j in 0..state_dim {
            let mut covariance = if i == j { process_variance } else { 0.0 };
            for k in 0..state_dim {
                covariance += transition[k][i] * current_variance[k] * transition[k][j];
            }
            predicted_covariance[i][j] = covariance;
        }
    }

    // Observation signal covariance A^T Sigma_next A.
    let mut observation_signal_covariance = vec![vec![0.0; obs_dim]; obs_dim];
    for i in 0..obs_dim {
        for j in 0..obs_dim {
            let mut covariance = 0.0;
            for left_state in 0..state_dim {
                for right_state in 0..state_dim {
                    covariance += model.likelihood_matrix[left_state][i]
                        * predicted_covariance[left_state][right_state]
                        * model.likelihood_matrix[right_state][j];
                }
            }
            observation_signal_covariance[i][j] = covariance;
        }
    }

    // M = I + R^-1 A^T Sigma_next A.  M is positive definite for valid
    // positive precisions; Cholesky gives 1/2 ln det(M) as sum ln(diag(L)).
    let mut cholesky = vec![vec![0.0; obs_dim]; obs_dim];
    let mut information_gain = 0.0;
    for i in 0..obs_dim {
        for j in 0..=i {
            let mut value = if i == j { 1.0 } else { 0.0 }
                + model.observation_precision * observation_signal_covariance[i][j];
            for k in 0..j {
                value -= cholesky[i][k] * cholesky[j][k];
            }

            if i == j {
                if !value.is_finite() || value <= 0.0 {
                    return Err(
                        HiddenStateInformationGainError::NonPositiveDefiniteObservationCovariance {
                            index: i,
                            pivot: value,
                        },
                    );
                }
                let diagonal = value.sqrt();
                cholesky[i][j] = diagonal;
                information_gain += diagonal.ln();
            } else {
                cholesky[i][j] = value / cholesky[j][j];
            }
        }
    }

    Ok(information_gain.max(0.0))
}
