// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exact shared-covariance binary Gaussian decision diagnostics.
//!
//! This is the first bounded milestone for correlated multivariate observations.
//! For two hypotheses with a common positive-definite covariance and symmetric
//! 0-1 decision loss, expected posterior Bayes risk equals binary Gaussian
//! classification error and has a closed form in the Mahalanobis distance.
//! No multivariate numerical integration is required.
//!
//! The module deliberately does not implement unequal covariance, more than two
//! hypotheses, arbitrary loss matrices, or general multivariate EIG. Those need
//! a separately qualified numerical integration method. A caller-supplied risk
//! score is only a score: this module does not treat it as a calibrated
//! probability, compose it across actions, execute probes, or authorize edits.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use thiserror::Error;

const MAX_GAUSSIAN_DIMENSIONS: usize = 16;
const MAX_DIAGNOSTIC_ACTIONS: usize = 64;
const VALIDATION_TOLERANCE: f64 = 1e-10;
const DECISION_TOLERANCE: f64 = 1e-10;
/// Error envelope documented for the A&S 7.1.26 normal-CDF approximation.
pub const NORMAL_CDF_APPROXIMATION_ERROR_ENVELOPE: f64 = 7.5e-8;

/// One hypothesis' action-conditioned multivariate Gaussian means.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultivariateGaussianDecisionHypothesis {
    pub id: String,
    pub prior_probability: f64,
    /// One mean vector per action, using that action's declared feature order.
    pub mean_by_action: Vec<Vec<f64>>,
    /// This initial exact kernel requires [0, 1] for H0 and [1, 0] for H1.
    pub loss_by_final_decision: Vec<f64>,
}

/// One jointly acquired observation protocol with covariance shared by both
/// hypotheses. The full covariance is used; channels are never factorized.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultivariateGaussianDiagnosticAction {
    pub id: String,
    pub feature_ids: Vec<String>,
    pub covariance: Vec<Vec<f64>>,
    pub observation_cost: f64,
    /// Caller-defined score in [0, 1], not necessarily a calibrated probability.
    pub risk_score: f64,
}

/// Decision weights and the hard per-action score limit.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MultivariateGaussianDecisionDiagnosticConfig {
    pub observation_cost_weight: f64,
    pub risk_score_weight: f64,
    pub maximum_risk_score: f64,
}

impl Default for MultivariateGaussianDecisionDiagnosticConfig {
    fn default() -> Self {
        Self {
            observation_cost_weight: 1.0,
            risk_score_weight: 1.0,
            maximum_risk_score: 1.0,
        }
    }
}

/// Analytic evaluation of one joint Gaussian diagnostic action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultivariateGaussianDiagnosticActionEvaluation {
    pub action_index: usize,
    pub action_id: String,
    pub mahalanobis_squared: f64,
    pub expected_posterior_bayes_risk: f64,
    pub expected_value_of_sample_information: f64,
    pub observation_cost: f64,
    pub risk_score: f64,
    pub eligible: bool,
    pub total_score: Option<f64>,
    pub normal_cdf_approximation_error_envelope: f64,
}

/// Result of the exact shared-covariance binary diagnostic calculation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultivariateGaussianDecisionDiagnosticEvaluation {
    pub task_specification_digest: String,
    /// Binds priors, means, full covariances, feature order, costs, scores and config.
    pub problem_digest: String,
    pub evaluation_digest: String,
    pub prior_bayes_risk: f64,
    pub no_probe_score: f64,
    pub selected_action_index: Option<usize>,
    pub actions: Vec<MultivariateGaussianDiagnosticActionEvaluation>,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum MultivariateGaussianDecisionDiagnosticError {
    #[error("exact binary Gaussian diagnostics require exactly two hypotheses")]
    HypothesisCount,
    #[error("at least one diagnostic action is required")]
    EmptyActionSet,
    #[error("at most {maximum} diagnostic actions are supported, received {actual}")]
    TooManyActions { maximum: usize, actual: usize },
    #[error("hypothesis {hypothesis_index} has an empty or duplicate identifier")]
    InvalidHypothesisId { hypothesis_index: usize },
    #[error("hypothesis {hypothesis_index} has an invalid prior probability")]
    InvalidPrior { hypothesis_index: usize },
    #[error("hypothesis prior probabilities sum to {sum}, expected 1")]
    PriorMassNotNormalized { sum: f64 },
    #[error("hypothesis {hypothesis_index} must have the exact binary 0-1 loss vector")]
    UnsupportedLossStructure { hypothesis_index: usize },
    #[error("action {action_index} has an empty or duplicate identifier")]
    InvalidActionId { action_index: usize },
    #[error("hypothesis {hypothesis_index} has {actual} mean vectors, expected {expected}")]
    HypothesisActionCount {
        hypothesis_index: usize,
        actual: usize,
        expected: usize,
    },
    #[error("action {action_index} has an invalid or duplicate feature identifier")]
    InvalidFeatureId { action_index: usize },
    #[error("action {action_index} has unsupported dimension {actual}; maximum is {maximum}")]
    UnsupportedDimension {
        action_index: usize,
        actual: usize,
        maximum: usize,
    },
    #[error("action {action_index} has a dimension mismatch in hypothesis {hypothesis_index}")]
    MeanDimensionMismatch {
        action_index: usize,
        hypothesis_index: usize,
    },
    #[error("action {action_index} has a non-finite mean")]
    NonFiniteMean { action_index: usize },
    #[error("action {action_index} covariance is not square with the declared dimension")]
    CovarianceShape { action_index: usize },
    #[error("action {action_index} covariance is non-finite or not exactly symmetric")]
    InvalidCovarianceSymmetry { action_index: usize },
    #[error("action {action_index} covariance is not positive definite")]
    CovarianceNotPositiveDefinite { action_index: usize },
    #[error("action {action_index} has an invalid observation cost")]
    InvalidObservationCost { action_index: usize },
    #[error("action {action_index} risk score is outside [0, 1]")]
    InvalidRiskScore { action_index: usize },
    #[error("decision configuration is invalid")]
    InvalidConfig,
    #[error("task specification digest must be 64 lowercase hexadecimal characters")]
    InvalidTaskSpecificationDigest,
    #[error("Mahalanobis distance calculation was non-finite")]
    NonFiniteMahalanobisDistance,
    #[error("analytic Bayes-risk calculation was non-finite")]
    NonFiniteBayesRisk,
}

/// Evaluate correlated multivariate diagnostic actions using the exact
/// shared-covariance binary Gaussian Bayes-risk formula.
///
/// The two hypotheses must have strictly positive priors and symmetric 0-1
/// classification losses. Each action supplies one full SPD covariance shared
/// by both hypotheses and a mean vector for each hypothesis. Expected posterior
/// Bayes risk is:
///
/// p0 * Phi(-(log(p0/p1) + d^2/2)/d) + p1 * Phi((log(p0/p1) - d^2/2)/d)
///
/// where d^2 = (mu1 - mu0)^T Sigma^-1 (mu1 - mu0).
///
/// The standard normal CDF uses the documented A&S 7.1.26 approximation. This
/// is an analytic special-case research kernel, not a general multivariate
/// Gaussian integrator. Model information gain is intentionally not reported;
/// it must not be conflated with this downstream decision-value calculation.
pub fn evaluate_multivariate_gaussian_decision_diagnostics(
    hypotheses: &[MultivariateGaussianDecisionHypothesis],
    actions: &[MultivariateGaussianDiagnosticAction],
    config: MultivariateGaussianDecisionDiagnosticConfig,
    task_specification_digest: &str,
) -> Result<
    MultivariateGaussianDecisionDiagnosticEvaluation,
    MultivariateGaussianDecisionDiagnosticError,
> {
    validate_config(config)?;
    validate_task_digest(task_specification_digest)?;
    if actions.is_empty() {
        return Err(MultivariateGaussianDecisionDiagnosticError::EmptyActionSet);
    }
    validate_hypotheses(hypotheses, actions.len())?;
    validate_actions(hypotheses, actions)?;

    let prior_h0 = hypotheses[0].prior_probability;
    let prior_h1 = hypotheses[1].prior_probability;
    let prior_bayes_risk = prior_h0.min(prior_h1);
    let no_probe_score = prior_bayes_risk;
    let mut best_score = no_probe_score;
    let mut selected_action_index = None;
    let mut evaluations = Vec::with_capacity(actions.len());

    for (action_index, action) in actions.iter().enumerate() {
        let delta: Vec<f64> = hypotheses[1].mean_by_action[action_index]
            .iter()
            .zip(&hypotheses[0].mean_by_action[action_index])
            .map(|(right, left)| right - left)
            .collect();
        let mahalanobis_squared = mahalanobis_squared(&delta, &action.covariance, action_index)?;
        let expected_posterior_bayes_risk =
            binary_shared_covariance_bayes_risk(prior_h0, prior_h1, mahalanobis_squared)?;
        let bounded_posterior_risk = expected_posterior_bayes_risk.min(prior_bayes_risk);
        let expected_value_of_sample_information =
            (prior_bayes_risk - bounded_posterior_risk).max(0.0);
        let eligible = action.risk_score <= config.maximum_risk_score;
        let total_score = if eligible {
            Some(
                bounded_posterior_risk
                    + config.observation_cost_weight * action.observation_cost
                    + config.risk_score_weight * action.risk_score,
            )
        } else {
            None
        };

        if let Some(score) = total_score {
            if !score.is_finite() {
                return Err(MultivariateGaussianDecisionDiagnosticError::NonFiniteBayesRisk);
            }
            // Strict improvement preserves no-probe on ties and stable action ordering.
            if score + DECISION_TOLERANCE < best_score {
                best_score = score;
                selected_action_index = Some(action_index);
            }
        }

        evaluations.push(MultivariateGaussianDiagnosticActionEvaluation {
            action_index,
            action_id: action.id.clone(),
            mahalanobis_squared,
            expected_posterior_bayes_risk: bounded_posterior_risk,
            expected_value_of_sample_information,
            observation_cost: action.observation_cost,
            risk_score: action.risk_score,
            eligible,
            total_score,
            normal_cdf_approximation_error_envelope: NORMAL_CDF_APPROXIMATION_ERROR_ENVELOPE,
        });
    }

    let problem_digest = compute_problem_digest(
        hypotheses,
        actions,
        config,
        task_specification_digest,
    );
    let evaluation_digest = compute_evaluation_digest(
        &problem_digest,
        prior_bayes_risk,
        no_probe_score,
        selected_action_index,
        &evaluations,
    );

    Ok(MultivariateGaussianDecisionDiagnosticEvaluation {
        task_specification_digest: task_specification_digest.to_owned(),
        problem_digest,
        evaluation_digest,
        prior_bayes_risk,
        no_probe_score,
        selected_action_index,
        actions: evaluations,
    })
}

fn validate_config(
    config: MultivariateGaussianDecisionDiagnosticConfig,
) -> Result<(), MultivariateGaussianDecisionDiagnosticError> {
    if !config.observation_cost_weight.is_finite()
        || config.observation_cost_weight < 0.0
        || !config.risk_score_weight.is_finite()
        || config.risk_score_weight < 0.0
        || !config.maximum_risk_score.is_finite()
        || !(0.0..=1.0).contains(&config.maximum_risk_score)
    {
        return Err(MultivariateGaussianDecisionDiagnosticError::InvalidConfig);
    }
    Ok(())
}

fn validate_task_digest(
    digest: &str,
) -> Result<(), MultivariateGaussianDecisionDiagnosticError> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(MultivariateGaussianDecisionDiagnosticError::InvalidTaskSpecificationDigest);
    }
    Ok(())
}

fn validate_hypotheses(
    hypotheses: &[MultivariateGaussianDecisionHypothesis],
    action_count: usize,
) -> Result<(), MultivariateGaussianDecisionDiagnosticError> {
    if hypotheses.len() != 2 {
        return Err(MultivariateGaussianDecisionDiagnosticError::HypothesisCount);
    }
    let mut ids = HashSet::new();
    for (hypothesis_index, hypothesis) in hypotheses.iter().enumerate() {
        if hypothesis.id.trim().is_empty() || !ids.insert(hypothesis.id.as_str()) {
            return Err(MultivariateGaussianDecisionDiagnosticError::InvalidHypothesisId {
                hypothesis_index,
            });
        }
        if !hypothesis.prior_probability.is_finite()
            || hypothesis.prior_probability <= 0.0
            || hypothesis.prior_probability > 1.0
        {
            return Err(MultivariateGaussianDecisionDiagnosticError::InvalidPrior {
                hypothesis_index,
            });
        }
        if hypothesis.mean_by_action.len() != action_count {
            return Err(MultivariateGaussianDecisionDiagnosticError::HypothesisActionCount {
                hypothesis_index,
                actual: hypothesis.mean_by_action.len(),
                expected: action_count,
            });
        }
        let expected_loss = if hypothesis_index == 0 {
            [0.0, 1.0]
        } else {
            [1.0, 0.0]
        };
        if hypothesis.loss_by_final_decision.len() != 2
            || hypothesis
                .loss_by_final_decision
                .iter()
                .zip(expected_loss)
                .any(|(actual, expected)| *actual != expected)
        {
            return Err(
                MultivariateGaussianDecisionDiagnosticError::UnsupportedLossStructure {
                    hypothesis_index,
                },
            );
        }
    }
    let prior_mass = hypotheses[0].prior_probability + hypotheses[1].prior_probability;
    if !prior_mass.is_finite() || (prior_mass - 1.0).abs() > VALIDATION_TOLERANCE {
        return Err(MultivariateGaussianDecisionDiagnosticError::PriorMassNotNormalized {
            sum: prior_mass,
        });
    }
    Ok(())
}

fn validate_actions(
    hypotheses: &[MultivariateGaussianDecisionHypothesis],
    actions: &[MultivariateGaussianDiagnosticAction],
) -> Result<(), MultivariateGaussianDecisionDiagnosticError> {
    if actions.is_empty() {
        return Err(MultivariateGaussianDecisionDiagnosticError::EmptyActionSet);
    }
    if actions.len() > MAX_DIAGNOSTIC_ACTIONS {
        return Err(MultivariateGaussianDecisionDiagnosticError::TooManyActions {
            maximum: MAX_DIAGNOSTIC_ACTIONS,
            actual: actions.len(),
        });
    }
    let mut action_ids = HashSet::new();
    for (action_index, action) in actions.iter().enumerate() {
        if action.id.trim().is_empty() || !action_ids.insert(action.id.as_str()) {
            return Err(MultivariateGaussianDecisionDiagnosticError::InvalidActionId {
                action_index,
            });
        }
        if !action.observation_cost.is_finite() || action.observation_cost < 0.0 {
            return Err(MultivariateGaussianDecisionDiagnosticError::InvalidObservationCost {
                action_index,
            });
        }
        if !action.risk_score.is_finite() || !(0.0..=1.0).contains(&action.risk_score) {
            return Err(MultivariateGaussianDecisionDiagnosticError::InvalidRiskScore {
                action_index,
            });
        }
        let dimension = action.feature_ids.len();
        if dimension == 0 || dimension > MAX_GAUSSIAN_DIMENSIONS {
            return Err(MultivariateGaussianDecisionDiagnosticError::UnsupportedDimension {
                action_index,
                actual: dimension,
                maximum: MAX_GAUSSIAN_DIMENSIONS,
            });
        }
        let mut feature_ids = HashSet::new();
        if action
            .feature_ids
            .iter()
            .any(|feature| feature.trim().is_empty() || !feature_ids.insert(feature.as_str()))
        {
            return Err(MultivariateGaussianDecisionDiagnosticError::InvalidFeatureId {
                action_index,
            });
        }
        for (hypothesis_index, hypothesis) in hypotheses.iter().enumerate() {
            let mean = &hypothesis.mean_by_action[action_index];
            if mean.len() != dimension {
                return Err(
                    MultivariateGaussianDecisionDiagnosticError::MeanDimensionMismatch {
                        action_index,
                        hypothesis_index,
                    },
                );
            }
            if mean.iter().any(|value| !value.is_finite()) {
                return Err(MultivariateGaussianDecisionDiagnosticError::NonFiniteMean {
                    action_index,
                });
            }
        }
        validate_covariance(&action.covariance, dimension, action_index)?;
    }
    Ok(())
}

fn validate_covariance(
    covariance: &[Vec<f64>],
    dimension: usize,
    action_index: usize,
) -> Result<(), MultivariateGaussianDecisionDiagnosticError> {
    if covariance.len() != dimension || covariance.iter().any(|row| row.len() != dimension) {
        return Err(MultivariateGaussianDecisionDiagnosticError::CovarianceShape {
            action_index,
        });
    }
    for row in 0..dimension {
        for column in 0..dimension {
            if !covariance[row][column].is_finite()
                || covariance[row][column] != covariance[column][row]
            {
                return Err(
                    MultivariateGaussianDecisionDiagnosticError::InvalidCovarianceSymmetry {
                        action_index,
                    },
                );
            }
        }
    }
    cholesky_factor(covariance, action_index).map(|_| ())
}

fn cholesky_factor(
    covariance: &[Vec<f64>],
    action_index: usize,
) -> Result<Vec<Vec<f64>>, MultivariateGaussianDecisionDiagnosticError> {
    let dimension = covariance.len();
    let mut lower = vec![vec![0.0; dimension]; dimension];
    for row in 0..dimension {
        for column in 0..=row {
            let mut value = covariance[row][column];
            for inner in 0..column {
                value -= lower[row][inner] * lower[column][inner];
            }
            if row == column {
                if !value.is_finite() || value <= 0.0 {
                    return Err(
                        MultivariateGaussianDecisionDiagnosticError::CovarianceNotPositiveDefinite {
                            action_index,
                        },
                    );
                }
                lower[row][column] = value.sqrt();
            } else {
                let diagonal = lower[column][column];
                let entry = value / diagonal;
                if !entry.is_finite() {
                    return Err(
                        MultivariateGaussianDecisionDiagnosticError::CovarianceNotPositiveDefinite {
                            action_index,
                        },
                    );
                }
                lower[row][column] = entry;
            }
        }
    }
    Ok(lower)
}

fn mahalanobis_squared(
    delta: &[f64],
    covariance: &[Vec<f64>],
    action_index: usize,
) -> Result<f64, MultivariateGaussianDecisionDiagnosticError> {
    let lower = cholesky_factor(covariance, action_index)?;
    let mut whitened = vec![0.0; delta.len()];
    for row in 0..delta.len() {
        let mut value = delta[row];
        for column in 0..row {
            value -= lower[row][column] * whitened[column];
        }
        whitened[row] = value / lower[row][row];
    }
    let squared = whitened.iter().map(|value| value * value).sum::<f64>();
    if !squared.is_finite() || squared < 0.0 {
        return Err(MultivariateGaussianDecisionDiagnosticError::NonFiniteMahalanobisDistance);
    }
    Ok(squared)
}

fn binary_shared_covariance_bayes_risk(
    prior_h0: f64,
    prior_h1: f64,
    mahalanobis_squared: f64,
) -> Result<f64, MultivariateGaussianDecisionDiagnosticError> {
    if !mahalanobis_squared.is_finite() || mahalanobis_squared < 0.0 {
        return Err(MultivariateGaussianDecisionDiagnosticError::NonFiniteMahalanobisDistance);
    }
    if mahalanobis_squared <= f64::EPSILON {
        return Ok(prior_h0.min(prior_h1));
    }
    let distance = mahalanobis_squared.sqrt();
    let threshold = prior_h0.ln() - prior_h1.ln();
    let h0_error = standard_normal_cdf(-(threshold + 0.5 * mahalanobis_squared) / distance);
    let h1_error = standard_normal_cdf((threshold - 0.5 * mahalanobis_squared) / distance);
    let risk = prior_h0 * h0_error + prior_h1 * h1_error;
    if !risk.is_finite() || risk < 0.0 {
        return Err(MultivariateGaussianDecisionDiagnosticError::NonFiniteBayesRisk);
    }
    if risk > prior_h0.min(prior_h1) + NORMAL_CDF_APPROXIMATION_ERROR_ENVELOPE {
        return Err(MultivariateGaussianDecisionDiagnosticError::NonFiniteBayesRisk);
    }
    Ok(risk.min(prior_h0.min(prior_h1)))
}

/// A&S 7.1.26 rational approximation to the standard normal CDF.
/// The error envelope is exposed in every action result rather than hidden.
fn standard_normal_cdf(value: f64) -> f64 {
    if value == 0.0 {
        return 0.5;
    }
    if value < 0.0 {
        return 1.0 - standard_normal_cdf(-value);
    }
    let t = 1.0 / (1.0 + 0.2316419 * value);
    let polynomial = t
        * (0.319381530
            + t * (-0.356563782
                + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let density = 0.3989422804014327 * (-0.5 * value * value).exp();
    (1.0 - density * polynomial).clamp(0.0, 1.0)
}

fn compute_problem_digest(
    hypotheses: &[MultivariateGaussianDecisionHypothesis],
    actions: &[MultivariateGaussianDiagnosticAction],
    config: MultivariateGaussianDecisionDiagnosticConfig,
    task_specification_digest: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-multivariate-gaussian-binary-decision-v1\0");
    hash_string(&mut hasher, "task_specification_digest", task_specification_digest);
    for value in [
        config.observation_cost_weight,
        config.risk_score_weight,
        config.maximum_risk_score,
    ] {
        hasher.update(value.to_bits().to_le_bytes());
    }
    hasher.update((hypotheses.len() as u64).to_le_bytes());
    for hypothesis in hypotheses {
        hash_string(&mut hasher, "hypothesis_id", &hypothesis.id);
        hasher.update(hypothesis.prior_probability.to_bits().to_le_bytes());
        for loss in &hypothesis.loss_by_final_decision {
            hasher.update(loss.to_bits().to_le_bytes());
        }
        for mean in &hypothesis.mean_by_action {
            hasher.update((mean.len() as u64).to_le_bytes());
            for value in mean {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }
    }
    hasher.update((actions.len() as u64).to_le_bytes());
    for action in actions {
        hash_string(&mut hasher, "action_id", &action.id);
        hasher.update((action.feature_ids.len() as u64).to_le_bytes());
        for feature in &action.feature_ids {
            hash_string(&mut hasher, "feature_id", feature);
        }
        for row in &action.covariance {
            for value in row {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }
        hasher.update(action.observation_cost.to_bits().to_le_bytes());
        hasher.update(action.risk_score.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn compute_evaluation_digest(
    problem_digest: &str,
    prior_bayes_risk: f64,
    no_probe_score: f64,
    selected_action_index: Option<usize>,
    actions: &[MultivariateGaussianDiagnosticActionEvaluation],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-multivariate-gaussian-binary-evaluation-v1\0");
    hasher.update(NORMAL_CDF_APPROXIMATION_ERROR_ENVELOPE.to_bits().to_le_bytes());
    hash_string(&mut hasher, "normal_cdf_method", "A&S-7.1.26");
    hash_string(&mut hasher, "problem_digest", problem_digest);
    hasher.update(prior_bayes_risk.to_bits().to_le_bytes());
    hasher.update(no_probe_score.to_bits().to_le_bytes());
    match selected_action_index {
        Some(index) => {
            hasher.update([1]);
            hasher.update((index as u64).to_le_bytes());
        }
        None => hasher.update([0]),
    }
    for action in actions {
        hasher.update((action.action_index as u64).to_le_bytes());
        hash_string(&mut hasher, "action_id", &action.action_id);
        for value in [
            action.mahalanobis_squared,
            action.expected_posterior_bayes_risk,
            action.expected_value_of_sample_information,
            action.observation_cost,
            action.risk_score,
        ] {
            hasher.update(value.to_bits().to_le_bytes());
        }
        hasher.update([u8::from(action.eligible)]);
        match action.total_score {
            Some(score) => {
                hasher.update([1]);
                hasher.update(score.to_bits().to_le_bytes());
            }
            None => hasher.update([0]),
        }
    }
    format!("{:x}", hasher.finalize())
}

fn hash_string(hasher: &mut Sha256, field: &str, value: &str) {
    hasher.update((field.len() as u64).to_le_bytes());
    hasher.update(field.as_bytes());
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(
        means: [Vec<f64>; 2],
        covariance: Vec<Vec<f64>>,
        priors: [f64; 2],
    ) -> (
        Vec<MultivariateGaussianDecisionHypothesis>,
        Vec<MultivariateGaussianDiagnosticAction>,
    ) {
        let dimension = means[0].len();
        let hypotheses = vec![
            MultivariateGaussianDecisionHypothesis {
                id: "H0".to_owned(),
                prior_probability: priors[0],
                mean_by_action: vec![means[0].clone()],
                loss_by_final_decision: vec![0.0, 1.0],
            },
            MultivariateGaussianDecisionHypothesis {
                id: "H1".to_owned(),
                prior_probability: priors[1],
                mean_by_action: vec![means[1].clone()],
                loss_by_final_decision: vec![1.0, 0.0],
            },
        ];
        let actions = vec![MultivariateGaussianDiagnosticAction {
            id: "joint-sensor".to_owned(),
            feature_ids: (0..dimension).map(|index| format!("sensor-{index}")).collect(),
            covariance,
            observation_cost: 0.0,
            risk_score: 0.0,
        }];
        (hypotheses, actions)
    }

    fn evaluate(
        hypotheses: &[MultivariateGaussianDecisionHypothesis],
        actions: &[MultivariateGaussianDiagnosticAction],
    ) -> MultivariateGaussianDecisionDiagnosticEvaluation {
        evaluate_multivariate_gaussian_decision_diagnostics(
            hypotheses,
            actions,
            MultivariateGaussianDecisionDiagnosticConfig::default(),
            &"a".repeat(64),
        )
        .expect("valid shared-covariance binary fixture")
    }

    #[test]
    fn correlated_two_dimensional_oracle_matches_mahalanobis_formula() {
        // Sigma has eigenvalue 1.5 along [1, 1]. Delta=[sqrt(3),sqrt(3)]
        // therefore d^2=4 and equal-prior Bayes error is Phi(-1).
        let root_three = 3.0_f64.sqrt();
        let (hypotheses, actions) = fixture(
            [vec![0.0, 0.0], vec![root_three, root_three]],
            vec![vec![1.0, 0.5], vec![0.5, 1.0]],
            [0.5, 0.5],
        );
        let result = evaluate(&hypotheses, &actions);
        assert!((result.actions[0].mahalanobis_squared - 4.0).abs() < 1e-12);
        assert!((result.actions[0].expected_posterior_bayes_risk - 0.1586552539).abs() < 1e-7);
        assert!(result.actions[0].expected_value_of_sample_information > 0.3);
    }

    #[test]
    fn correlated_three_dimensional_oracle_matches_mahalanobis_formula() {
        // Sigma has eigenvalue 2 along [1,1,1]. Delta is chosen for d^2=4.
        let component = (8.0_f64 / 3.0).sqrt();
        let (hypotheses, actions) = fixture(
            [vec![0.0; 3], vec![component; 3]],
            vec![
                vec![1.0, 0.5, 0.5],
                vec![0.5, 1.0, 0.5],
                vec![0.5, 0.5, 1.0],
            ],
            [0.5, 0.5],
        );
        let result = evaluate(&hypotheses, &actions);
        assert!((result.actions[0].mahalanobis_squared - 4.0).abs() < 1e-12);
        assert!((result.actions[0].expected_posterior_bayes_risk - 0.1586552539).abs() < 1e-7);
    }

    #[test]
    fn diagonal_covariance_matches_one_dimensional_gaussian_reference() {
        let (hypotheses, actions) = fixture(
            [vec![0.0, 0.0], vec![2.0, 0.0]],
            vec![vec![1.0, 0.0], vec![0.0, 4.0]],
            [0.5, 0.5],
        );
        let result = evaluate(&hypotheses, &actions);
        assert!((result.actions[0].mahalanobis_squared - 4.0).abs() < 1e-12);
        assert!((result.actions[0].expected_posterior_bayes_risk - 0.1586552539).abs() < 1e-7);
    }

    #[test]
    fn ignoring_positive_sensor_correlation_is_an_overconfidence_negative_control() {
        let root_three = 3.0_f64.sqrt();
        let (hypotheses, correlated_actions) = fixture(
            [vec![0.0, 0.0], vec![root_three, root_three]],
            vec![vec![1.0, 0.5], vec![0.5, 1.0]],
            [0.5, 0.5],
        );
        let correct = evaluate(&hypotheses, &correlated_actions);

        // Deliberately wrong control: retain both marginal variances but erase
        // the positive covariance. It inflates d^2 from 4 to 6 and reports
        // spuriously lower Bayes error.
        let mut independent_actions = correlated_actions.clone();
        independent_actions[0].covariance =
            vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let independent = evaluate(&hypotheses, &independent_actions);
        assert!((correct.actions[0].mahalanobis_squared - 4.0).abs() < 1e-12);
        assert!((independent.actions[0].mahalanobis_squared - 6.0).abs() < 1e-12);
        assert!(
            independent.actions[0].expected_posterior_bayes_risk
                < correct.actions[0].expected_posterior_bayes_risk
        );
        assert_ne!(correct.problem_digest, independent.problem_digest);
    }

    #[test]
    fn covariance_permutation_does_not_change_decision_values() {
        let (hypotheses, actions) = fixture(
            [vec![0.0, 0.0, 0.0], vec![1.0, 2.0, -1.0]],
            vec![
                vec![2.0, 0.2, -0.1],
                vec![0.2, 1.5, 0.3],
                vec![-0.1, 0.3, 1.2],
            ],
            [0.3, 0.7],
        );
        let original = evaluate(&hypotheses, &actions);
        let permutation = [2, 0, 1];
        let mut permuted_hypotheses = hypotheses.clone();
        for hypothesis in &mut permuted_hypotheses {
            hypothesis.mean_by_action[0] = permutation
                .iter()
                .map(|index| hypothesis.mean_by_action[0][*index])
                .collect();
        }
        let mut permuted_actions = actions.clone();
        permuted_actions[0].feature_ids = permutation
            .iter()
            .map(|index| actions[0].feature_ids[*index].clone())
            .collect();
        permuted_actions[0].covariance = permutation
            .iter()
            .map(|row| {
                permutation
                    .iter()
                    .map(|column| actions[0].covariance[*row][*column])
                    .collect()
            })
            .collect();
        let permuted = evaluate(&permuted_hypotheses, &permuted_actions);
        assert!((original.actions[0].mahalanobis_squared
            - permuted.actions[0].mahalanobis_squared)
            .abs()
            < 1e-12);
        assert!((original.actions[0].expected_posterior_bayes_risk
            - permuted.actions[0].expected_posterior_bayes_risk)
            .abs()
            < 1e-12);
    }

    #[test]
    fn identical_means_have_zero_evsi_and_keep_no_probe() {
        let (hypotheses, actions) = fixture(
            [vec![1.0, -2.0], vec![1.0, -2.0]],
            vec![vec![1.0, 0.25], vec![0.25, 2.0]],
            [0.4, 0.6],
        );
        let result = evaluate(&hypotheses, &actions);
        assert_eq!(result.actions[0].mahalanobis_squared, 0.0);
        assert_eq!(result.actions[0].expected_value_of_sample_information, 0.0);
        assert_eq!(result.selected_action_index, None);
    }

    #[test]
    fn covariance_that_is_singular_or_not_symmetric_fails_closed() {
        let (hypotheses, mut actions) = fixture(
            [vec![0.0, 0.0], vec![1.0, 1.0]],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
            [0.5, 0.5],
        );
        assert!(matches!(
            evaluate_multivariate_gaussian_decision_diagnostics(
                &hypotheses,
                &actions,
                Default::default(),
                &"a".repeat(64),
            ),
            Err(MultivariateGaussianDecisionDiagnosticError::CovarianceNotPositiveDefinite { .. })
        ));
        actions[0].covariance[0][1] = 0.3;
        assert!(matches!(
            evaluate_multivariate_gaussian_decision_diagnostics(
                &hypotheses,
                &actions,
                Default::default(),
                &"a".repeat(64),
            ),
            Err(MultivariateGaussianDecisionDiagnosticError::InvalidCovarianceSymmetry { .. })
        ));
    }

    #[test]
    fn invalid_dimensions_non_finite_values_and_bad_digest_fail_closed() {
        let (hypotheses, mut actions) = fixture(
            [vec![0.0, 0.0], vec![1.0, 1.0]],
            vec![vec![1.0, 0.2], vec![0.2, 1.0]],
            [0.5, 0.5],
        );
        actions[0].covariance[0][0] = f64::NAN;
        assert!(matches!(
            evaluate_multivariate_gaussian_decision_diagnostics(
                &hypotheses,
                &actions,
                Default::default(),
                &"a".repeat(64),
            ),
            Err(MultivariateGaussianDecisionDiagnosticError::InvalidCovarianceSymmetry { .. })
        ));
        actions[0].covariance = vec![vec![1.0]];
        assert!(matches!(
            evaluate_multivariate_gaussian_decision_diagnostics(
                &hypotheses,
                &actions,
                Default::default(),
                &"a".repeat(64),
            ),
            Err(MultivariateGaussianDecisionDiagnosticError::CovarianceShape { .. })
        ));
        actions[0].covariance = vec![vec![1.0, 0.2], vec![0.2, 1.0]];
        assert!(matches!(
            evaluate_multivariate_gaussian_decision_diagnostics(
                &hypotheses,
                &actions,
                Default::default(),
                "bad-digest",
            ),
            Err(MultivariateGaussianDecisionDiagnosticError::InvalidTaskSpecificationDigest)
        ));
    }

    #[test]
    fn cost_and_risk_score_can_decline_an_informative_joint_measurement() {
        let (hypotheses, mut actions) = fixture(
            [vec![0.0, 0.0], vec![3.0, 3.0]],
            vec![vec![1.0, 0.8], vec![0.8, 1.0]],
            [0.5, 0.5],
        );
        actions[0].observation_cost = 0.2;
        actions[0].risk_score = 0.4;
        let result = evaluate_multivariate_gaussian_decision_diagnostics(
            &hypotheses,
            &actions,
            MultivariateGaussianDecisionDiagnosticConfig {
                observation_cost_weight: 1.0,
                risk_score_weight: 1.0,
                maximum_risk_score: 0.3,
            },
            &"a".repeat(64),
        )
        .expect("valid but ineligible action");
        assert!(result.actions[0].expected_value_of_sample_information > 0.0);
        assert!(!result.actions[0].eligible);
        assert_eq!(result.actions[0].total_score, None);
        assert_eq!(result.selected_action_index, None);
    }

    #[test]
    fn observation_cost_can_decline_an_eligible_informative_measurement() {
        let (hypotheses, mut actions) = fixture(
            [vec![0.0, 0.0], vec![3.0, 3.0]],
            vec![vec![1.0, 0.8], vec![0.8, 1.0]],
            [0.5, 0.5],
        );
        actions[0].observation_cost = 1.0;
        let result = evaluate_multivariate_gaussian_decision_diagnostics(
            &hypotheses,
            &actions,
            MultivariateGaussianDecisionDiagnosticConfig::default(),
            &"a".repeat(64),
        )
        .expect("valid informative but expensive action");
        assert!(result.actions[0].eligible);
        assert!(result.actions[0].expected_value_of_sample_information > 0.0);
        assert!(result.actions[0].total_score.unwrap() > result.no_probe_score);
        assert_eq!(result.selected_action_index, None);
    }

    #[test]
    fn input_digest_binds_covariance_feature_order_and_task() {
        let (hypotheses, actions) = fixture(
            [vec![0.0, 0.0], vec![1.0, 1.0]],
            vec![vec![1.0, 0.2], vec![0.2, 1.0]],
            [0.5, 0.5],
        );
        let original = evaluate(&hypotheses, &actions);
        let mut changed = actions.clone();
        changed[0].covariance[0][1] = 0.4;
        changed[0].covariance[1][0] = 0.4;
        let changed = evaluate(&hypotheses, &changed);
        assert_ne!(original.problem_digest, changed.problem_digest);
        assert_ne!(original.evaluation_digest, changed.evaluation_digest);
        let changed_task = evaluate_multivariate_gaussian_decision_diagnostics(
            &hypotheses,
            &actions,
            Default::default(),
            &"b".repeat(64),
        )
        .expect("valid changed task digest");
        assert_ne!(original.problem_digest, changed_task.problem_digest);
    }

    #[test]
    fn joint_covariance_and_signal_scaling_preserve_mahalanobis_decision() {
        let root_three = 3.0_f64.sqrt();
        let (hypotheses, actions) = fixture(
            [vec![0.0, 0.0], vec![root_three, root_three]],
            vec![vec![1.0, 0.5], vec![0.5, 1.0]],
            [0.5, 0.5],
        );
        let baseline = evaluate(&hypotheses, &actions);
        for covariance_scale in [1e-100_f64, 1e100_f64] {
            let signal_scale = covariance_scale.sqrt();
            let mut scaled_hypotheses = hypotheses.clone();
            for hypothesis in &mut scaled_hypotheses {
                for value in &mut hypothesis.mean_by_action[0] {
                    *value *= signal_scale;
                }
            }
            let mut scaled_actions = actions.clone();
            for row in &mut scaled_actions[0].covariance {
                for value in row {
                    *value *= covariance_scale;
                }
            }
            let scaled = evaluate(&scaled_hypotheses, &scaled_actions);
            assert!(
                (scaled.actions[0].mahalanobis_squared
                    - baseline.actions[0].mahalanobis_squared)
                    .abs()
                    < 1e-10
            );
            assert!(
                (scaled.actions[0].expected_posterior_bayes_risk
                    - baseline.actions[0].expected_posterior_bayes_risk)
                    .abs()
                    < 1e-10
            );
        }
    }

    #[test]
    fn near_singular_but_positive_covariance_is_finite_and_matches_eigen_reference() {
        let epsilon = 1e-10_f64;
        let covariance = vec![
            vec![1.0, 1.0 - epsilon],
            vec![1.0 - epsilon, 1.0],
        ];
        // [1, 1] is an eigenvector with eigenvalue 2 - epsilon, so
        // delta^T Sigma^-1 delta = 2 / (2 - epsilon).
        let expected_d_squared = 2.0 / (2.0 - epsilon);
        let (hypotheses, actions) = fixture(
            [vec![0.0, 0.0], vec![1.0, 1.0]],
            covariance,
            [0.5, 0.5],
        );
        let result = evaluate(&hypotheses, &actions);
        assert!(result.actions[0].mahalanobis_squared.is_finite());
        assert!(
            (result.actions[0].mahalanobis_squared - expected_d_squared).abs() < 1e-6
        );
        assert!(result.actions[0].expected_posterior_bayes_risk.is_finite());
    }

    #[test]
    fn extreme_but_valid_priors_keep_risk_finite_and_bounded() {
        for rare_prior in [1e-12_f64, 1e-100_f64, 1e-300_f64] {
            let (hypotheses, actions) = fixture(
                [vec![0.0, 0.0], vec![8.0, 8.0]],
                vec![vec![1.0, 0.2], vec![0.2, 1.0]],
                [rare_prior, 1.0 - rare_prior],
            );
            let result = evaluate(&hypotheses, &actions);
            let risk = result.actions[0].expected_posterior_bayes_risk;
            assert!(risk.is_finite());
            assert!(risk >= 0.0);
            assert!(risk <= rare_prior + NORMAL_CDF_APPROXIMATION_ERROR_ENVELOPE);
            assert!(result.actions[0].expected_value_of_sample_information.is_finite());
        }
    }

    #[test]
    fn normal_cdf_tail_fixtures_stay_inside_declared_empirical_envelope() {
        // Reference values are conventional high-precision normal-CDF values.
        // This finite fixture is a regression test, not a proof of the global envelope.
        let fixtures = [
            (-8.0, 6.22096057427178e-16),
            (-4.0, 3.167124183311986e-5),
            (-2.0, 0.02275013194817921),
            (0.0, 0.5),
            (2.0, 0.9772498680518208),
            (4.0, 0.9999683287581669),
            (8.0, 1.0 - 6.22096057427178e-16),
        ];
        for (value, expected) in fixtures {
            let error = (standard_normal_cdf(value) - expected).abs();
            assert!(
                error <= NORMAL_CDF_APPROXIMATION_ERROR_ENVELOPE,
                "normal CDF error {error} exceeded the declared envelope at {value}"
            );
        }
    }

}
