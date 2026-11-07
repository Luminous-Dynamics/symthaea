// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Bayesian model reduction for categorical likelihoods with Dirichlet priors.
//!
//! For a full model with Dirichlet prior concentrations alpha, posterior
//! concentrations a, and a prior-restricted model with concentrations alpha_m,
//! the reduced posterior concentrations are
//!
//!   a_m = a + alpha_m - alpha
//!
//! and the log marginal-likelihood ratio is
//!
//!   log p(data | m) / p(data | full)
//!     = log B(alpha) + log B(a_m) - log B(a) - log B(alpha_m),
//!
//! where B is the multivariate beta function. Independent likelihood columns
//! factorize, so their log ratios add. This is the Dirichlet BMR construction
//! used for categorical active-inference likelihood tensors.
//!
//! Every Dirichlet column carries an explicit support mask. Concentrations on
//! active coordinates must be strictly positive; inactive coordinates must be
//! exactly zero and denote a lower-dimensional simplex face. The full prior
//! and posterior must use identical support; reduced-model support may only be
//! a subset of the full support. Any observed count in a coordinate excluded
//! by a reduced model makes that model's evidence exactly zero. Implicit zeros,
//! empty support, mismatched supports, and non-integrable reduced posteriors
//! are rejected. Small positive shrinkage concentrations (e.g. 1/32) are
//! supported when the resulting reduced posterior remains positive.
//!
//! This is post-observation model evidence, not prospective expected
//! information gain and not a complete expected-free-energy policy.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

const PROBABILITY_TOLERANCE: f64 = 1e-9;
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_312e-7,
];

/// Dirichlet concentrations for independent categorical likelihood columns.
///
/// Each inner vector is one categorical distribution (one column of a
/// likelihood tensor). Column order, outcome order, factor identities and
/// tensor shape must be bound by parameterization_digest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirichletConcentrationTensor {
    /// SHA-256 digest of the canonical ordered parameterization.
    pub parameterization_digest: String,
    /// Per-column concentration vectors.
    /// Active coordinates must be positive finite; inactive coordinates must be exactly zero.
    pub columns: Vec<Vec<f64>>,
    /// Explicit support mask. Full-model tensors use all true; reduced models
    /// may mark excluded coordinates false to represent a proper lower-dimensional
    /// Dirichlet on a simplex face.
    pub active_support: Vec<Vec<bool>>,
}

/// A prior-restricted Dirichlet model within a shared full-model parameterization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReducedDirichletModelHypothesis {
    pub model_id: String,
    pub prior_probability: f64,
    pub parameter_prior: DirichletConcentrationTensor,
}

/// Evidence and posterior probability for one reduced Dirichlet model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReducedDirichletModelEvidence {
    pub model_id: String,
    pub prior_probability: f64,
    /// Log p(data | reduced model) / p(data | full model), in nats.
    pub log_evidence_ratio: f64,
    pub posterior_probability: f64,
}

/// Posterior over prior-restricted categorical models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirichletBayesianModelReductionResult {
    pub models: Vec<ReducedDirichletModelEvidence>,
    pub posterior_model_entropy_nats: f64,
    pub most_probable_model_id: String,
    /// Log posterior odds of the winner against the highest-probability runner-up.
    /// None when only one model has non-zero posterior mass.
    pub log_posterior_odds_over_runner_up: Option<f64>,
}

/// Validation and numerical failures for Dirichlet Bayesian model reduction.
#[derive(Debug, Clone, PartialEq)]
pub enum DirichletBayesianModelReductionError {
    EmptyColumnSet {
        distribution: &'static str,
    },
    EmptyColumn {
        distribution: &'static str,
        column: usize,
    },
    EmptySupport {
        distribution: &'static str,
        column: usize,
    },
    SupportMismatch {
        distribution: &'static str,
        column: usize,
        outcome: usize,
    },
    ColumnCountMismatch {
        distribution: &'static str,
        expected: usize,
        actual: usize,
    },
    OutcomeCountMismatch {
        distribution: &'static str,
        column: usize,
        expected: usize,
        actual: usize,
    },
    InvalidConcentration {
        distribution: &'static str,
        column: usize,
        outcome: usize,
        value: f64,
        active: bool,
    },
    InvalidSupportMask {
        distribution: &'static str,
        column: usize,
        expected: usize,
        actual: usize,
    },
    EmptyReducedSupport {
        model_id: String,
        column: usize,
    },
    PosteriorBelowPrior {
        model_id: String,
        column: usize,
        outcome: usize,
        increment: f64,
    },
    InvalidParameterizationDigest {
        distribution: &'static str,
    },
    ParameterizationMismatch {
        distribution: &'static str,
        expected: String,
        actual: String,
    },
    EmptyModelSet,
    EmptyModelId {
        index: usize,
    },
    DuplicateModelId(String),
    InvalidModelPrior {
        model_id: String,
        probability: f64,
    },
    ModelPriorMassNotNormalized {
        sum: f64,
    },
    NonPositiveReducedPosteriorConcentration {
        model_id: String,
        column: usize,
        outcome: usize,
        concentration: f64,
    },
    NonFiniteLogEvidence {
        model_id: String,
        column: usize,
    },
    InvalidPosteriorNormalization,
}

impl std::fmt::Display for DirichletBayesianModelReductionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyColumnSet { distribution } => {
                write!(f, "{distribution} has no Dirichlet columns")
            }
            Self::EmptyColumn {
                distribution,
                column,
            } => {
                write!(f, "{distribution} column {column} is empty")
            }
            Self::EmptySupport {
                distribution,
                column,
            } => {
                write!(f, "{distribution} column {column} has no active outcomes")
            }
            Self::SupportMismatch {
                distribution,
                column,
                outcome,
            } => write!(
                f,
                "{distribution} support at column {column}, outcome {outcome} is incompatible with the full model"
            ),
            Self::ColumnCountMismatch {
                distribution,
                expected,
                actual,
            } => write!(
                f,
                "{distribution} has {actual} columns, expected {expected}"
            ),
            Self::OutcomeCountMismatch {
                distribution,
                column,
                expected,
                actual,
            } => write!(
                f,
                "{distribution} column {column} has {actual} outcomes, expected {expected}"
            ),
            Self::InvalidConcentration {
                distribution,
                column,
                outcome,
                value,
                active,
            } => write!(
                f,
                "{distribution} concentration at column {column}, outcome {outcome} is invalid for active={active}: {value}"
            ),
            Self::InvalidSupportMask {
                distribution,
                column,
                expected,
                actual,
            } => write!(
                f,
                "{distribution} support mask at column {column} has length {actual}, expected {expected}"
            ),
            Self::EmptyReducedSupport { model_id, column } => {
                write!(
                    f,
                    "model {model_id} has empty active support at column {column}"
                )
            }
            Self::PosteriorBelowPrior {
                model_id,
                column,
                outcome,
                increment,
            } => write!(
                f,
                "model {model_id} has posterior-minus-prior increment {increment} at column {column}, outcome {outcome}"
            ),
            Self::InvalidParameterizationDigest { distribution } => {
                write!(f, "{distribution} has an invalid parameterization digest")
            }
            Self::ParameterizationMismatch {
                distribution,
                expected,
                actual,
            } => write!(
                f,
                "{distribution} parameterization digest {actual} does not match {expected}"
            ),
            Self::EmptyModelSet => write!(f, "at least one reduced model is required"),
            Self::EmptyModelId { index } => {
                write!(f, "reduced model at index {index} has an empty ID")
            }
            Self::DuplicateModelId(id) => write!(f, "duplicate reduced model ID: {id}"),
            Self::InvalidModelPrior {
                model_id,
                probability,
            } => write!(
                f,
                "model {model_id} has invalid prior probability {probability}"
            ),
            Self::ModelPriorMassNotNormalized { sum } => {
                write!(f, "model prior mass sums to {sum}, expected 1")
            }
            Self::NonPositiveReducedPosteriorConcentration {
                model_id,
                column,
                outcome,
                concentration,
            } => write!(
                f,
                "model {model_id} reduced posterior concentration at column {column}, outcome {outcome} is not positive: {concentration}"
            ),
            Self::NonFiniteLogEvidence { model_id, column } => write!(
                f,
                "model {model_id} has a non-finite log evidence ratio at column {column}"
            ),
            Self::InvalidPosteriorNormalization => {
                write!(f, "reduced-model posterior could not be normalized")
            }
        }
    }
}

impl std::error::Error for DirichletBayesianModelReductionError {}

/// Compare Dirichlet reduced models sharing one ordered likelihood parameterization.
///
/// full_posterior is the posterior under the Bayesian model average / full
/// model. Each reduced model differs only in its prior concentration tensor.
/// Evidence ratios are relative to the full model and include supplied model
/// prior probabilities when computing posterior model probabilities.
pub fn reduce_dirichlet_models(
    full_posterior: &DirichletConcentrationTensor,
    full_prior: &DirichletConcentrationTensor,
    reduced_models: &[ReducedDirichletModelHypothesis],
) -> Result<DirichletBayesianModelReductionResult, DirichletBayesianModelReductionError> {
    let column_count = full_posterior.columns.len();
    validate_tensor(full_posterior, "full posterior", None, true)?;
    validate_tensor(full_prior, "full prior", Some(full_posterior), true)?;
    ensure_support_equal(full_posterior, full_prior, "full prior")?;
    ensure_parameterization_matches(
        &full_posterior.parameterization_digest,
        "full prior",
        &full_prior.parameterization_digest,
    )?;

    if reduced_models.is_empty() {
        return Err(DirichletBayesianModelReductionError::EmptyModelSet);
    }

    let mut model_ids = BTreeSet::new();
    let mut prior_sum = 0.0;
    for (index, model) in reduced_models.iter().enumerate() {
        if model.model_id.trim().is_empty() {
            return Err(DirichletBayesianModelReductionError::EmptyModelId { index });
        }
        if !model_ids.insert(model.model_id.clone()) {
            return Err(DirichletBayesianModelReductionError::DuplicateModelId(
                model.model_id.clone(),
            ));
        }
        if !model.prior_probability.is_finite() || !(0.0..=1.0).contains(&model.prior_probability) {
            return Err(DirichletBayesianModelReductionError::InvalidModelPrior {
                model_id: model.model_id.clone(),
                probability: model.prior_probability,
            });
        }
        prior_sum += model.prior_probability;
        for (column, support) in model.parameter_prior.active_support.iter().enumerate() {
            if !support.iter().any(|active| *active) {
                return Err(DirichletBayesianModelReductionError::EmptyReducedSupport {
                    model_id: model.model_id.clone(),
                    column,
                });
            }
        }
        validate_tensor(
            &model.parameter_prior,
            "reduced prior",
            Some(full_posterior),
            true,
        )?;
        ensure_support_is_subset(full_posterior, &model.parameter_prior, "reduced prior")?;
        ensure_parameterization_matches(
            &full_posterior.parameterization_digest,
            "reduced prior",
            &model.parameter_prior.parameterization_digest,
        )?;
    }
    if (prior_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(
            DirichletBayesianModelReductionError::ModelPriorMassNotNormalized { sum: prior_sum },
        );
    }

    let mut log_weights = Vec::with_capacity(reduced_models.len());
    let mut log_evidence_ratios = Vec::with_capacity(reduced_models.len());
    for model in reduced_models {
        let log_ratio = dirichlet_log_evidence_ratio(
            full_posterior,
            full_prior,
            &model.parameter_prior,
            &model.model_id,
        )?;
        log_evidence_ratios.push(log_ratio);
        log_weights.push(if model.prior_probability == 0.0 {
            f64::NEG_INFINITY
        } else {
            model.prior_probability.ln() + log_ratio
        });
    }

    let max_log_weight = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_log_weight.is_finite() {
        return Err(DirichletBayesianModelReductionError::InvalidPosteriorNormalization);
    }
    let unnormalized = log_weights
        .iter()
        .map(|value| (value - max_log_weight).exp())
        .collect::<Vec<_>>();
    let normalizer = unnormalized.iter().sum::<f64>();
    if !normalizer.is_finite() || normalizer <= 0.0 {
        return Err(DirichletBayesianModelReductionError::InvalidPosteriorNormalization);
    }
    let posterior = unnormalized
        .iter()
        .map(|value| value / normalizer)
        .collect::<Vec<_>>();

    let posterior_model_entropy_nats = posterior
        .iter()
        .copied()
        .filter(|probability| *probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum::<f64>();

    let mut order = (0..reduced_models.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| posterior[*right].total_cmp(&posterior[*left]));
    let best_index = order[0];
    let runner_up = order
        .iter()
        .copied()
        .skip(1)
        .find(|index| posterior[*index] > 0.0);
    let log_posterior_odds_over_runner_up =
        runner_up.map(|runner_index| log_weights[best_index] - log_weights[runner_index]);

    let models = reduced_models
        .iter()
        .zip(log_evidence_ratios.iter().copied())
        .zip(posterior.iter().copied())
        .map(
            |((model, log_evidence_ratio), posterior_probability)| ReducedDirichletModelEvidence {
                model_id: model.model_id.clone(),
                prior_probability: model.prior_probability,
                log_evidence_ratio,
                posterior_probability,
            },
        )
        .collect();

    debug_assert_eq!(column_count, full_prior.columns.len());
    Ok(DirichletBayesianModelReductionResult {
        models,
        posterior_model_entropy_nats,
        most_probable_model_id: reduced_models[best_index].model_id.clone(),
        log_posterior_odds_over_runner_up,
    })
}

fn dirichlet_log_evidence_ratio(
    full_posterior: &DirichletConcentrationTensor,
    full_prior: &DirichletConcentrationTensor,
    reduced_prior: &DirichletConcentrationTensor,
    model_id: &str,
) -> Result<f64, DirichletBayesianModelReductionError> {
    let mut total = 0.0;
    for column in 0..full_posterior.columns.len() {
        let posterior = &full_posterior.columns[column];
        let full = &full_prior.columns[column];
        let full_support = &full_prior.active_support[column];
        let reduced = &reduced_prior.columns[column];
        let reduced_support = &reduced_prior.active_support[column];
        let mut full_prior_active = Vec::new();
        let mut full_posterior_active = Vec::new();
        let mut reduced_prior_active = Vec::new();
        let mut reduced_posterior_active = Vec::new();

        for outcome in 0..posterior.len() {
            if !full_support[outcome] {
                // The full model assigns exactly zero probability to this outcome.
                // Support equality guarantees both prior and posterior concentrations
                // are zero, and support-subset validation prevents reduced models
                // from activating it.
                continue;
            }

            let posterior_increment = posterior[outcome] - full[outcome];
            full_prior_active.push(full[outcome]);
            full_posterior_active.push(posterior[outcome]);

            if !reduced_support[outcome] {
                if posterior_increment < 0.0 {
                    return Err(DirichletBayesianModelReductionError::PosteriorBelowPrior {
                        model_id: model_id.to_owned(),
                        column,
                        outcome,
                        increment: posterior_increment,
                    });
                }
                if posterior_increment > 0.0 {
                    // The data contain an outcome forbidden by this reduced model.
                    // Its marginal likelihood is exactly zero.
                    return Ok(f64::NEG_INFINITY);
                }
                continue;
            }

            let value = posterior[outcome] + reduced[outcome] - full[outcome];
            if !value.is_finite() || value <= 0.0 {
                return Err(
                    DirichletBayesianModelReductionError::NonPositiveReducedPosteriorConcentration {
                        model_id: model_id.to_owned(),
                        column,
                        outcome,
                        concentration: value,
                    },
                );
            }
            reduced_prior_active.push(reduced[outcome]);
            reduced_posterior_active.push(value);
        }

        if reduced_prior_active.is_empty() {
            return Err(DirichletBayesianModelReductionError::EmptyReducedSupport {
                model_id: model_id.to_owned(),
                column,
            });
        }
        let contribution = log_multivariate_beta(&full_prior_active)
            + log_multivariate_beta(&reduced_posterior_active)
            - log_multivariate_beta(&full_posterior_active)
            - log_multivariate_beta(&reduced_prior_active);
        if !contribution.is_finite() {
            return Err(DirichletBayesianModelReductionError::NonFiniteLogEvidence {
                model_id: model_id.to_owned(),
                column,
            });
        }
        total += contribution;
        if !total.is_finite() {
            return Err(DirichletBayesianModelReductionError::NonFiniteLogEvidence {
                model_id: model_id.to_owned(),
                column,
            });
        }
    }
    Ok(total)
}

fn validate_tensor(
    tensor: &DirichletConcentrationTensor,
    distribution: &'static str,
    expected: Option<&DirichletConcentrationTensor>,
    allow_support_restriction: bool,
) -> Result<(), DirichletBayesianModelReductionError> {
    if !is_sha256_digest(&tensor.parameterization_digest) {
        return Err(
            DirichletBayesianModelReductionError::InvalidParameterizationDigest { distribution },
        );
    }
    if tensor.columns.is_empty() {
        return Err(DirichletBayesianModelReductionError::EmptyColumnSet { distribution });
    }
    if tensor.active_support.len() != tensor.columns.len() {
        return Err(DirichletBayesianModelReductionError::ColumnCountMismatch {
            distribution,
            expected: tensor.columns.len(),
            actual: tensor.active_support.len(),
        });
    }
    if let Some(expected) = expected {
        if tensor.columns.len() != expected.columns.len() {
            return Err(DirichletBayesianModelReductionError::ColumnCountMismatch {
                distribution,
                expected: expected.columns.len(),
                actual: tensor.columns.len(),
            });
        }
    }
    for (column_index, column) in tensor.columns.iter().enumerate() {
        if column.is_empty() {
            return Err(DirichletBayesianModelReductionError::EmptyColumn {
                distribution,
                column: column_index,
            });
        }
        if let Some(expected) = expected {
            let expected_len = expected.columns[column_index].len();
            if column.len() != expected_len {
                return Err(DirichletBayesianModelReductionError::OutcomeCountMismatch {
                    distribution,
                    column: column_index,
                    expected: expected_len,
                    actual: column.len(),
                });
            }
        }
        let support = &tensor.active_support[column_index];
        if support.len() != column.len() {
            return Err(DirichletBayesianModelReductionError::InvalidSupportMask {
                distribution,
                column: column_index,
                expected: column.len(),
                actual: support.len(),
            });
        }
        for (outcome_index, (concentration, active)) in column
            .iter()
            .copied()
            .zip(support.iter().copied())
            .enumerate()
        {
            let valid = if active {
                concentration.is_finite() && concentration > 0.0
            } else {
                allow_support_restriction && concentration == 0.0
            };
            if !valid {
                return Err(DirichletBayesianModelReductionError::InvalidConcentration {
                    distribution,
                    column: column_index,
                    outcome: outcome_index,
                    value: concentration,
                    active,
                });
            }
        }
        if !support.iter().any(|active| *active) {
            if allow_support_restriction {
                return Err(DirichletBayesianModelReductionError::EmptySupport {
                    distribution,
                    column: column_index,
                });
            }
            return Err(DirichletBayesianModelReductionError::InvalidSupportMask {
                distribution,
                column: column_index,
                expected: column.len(),
                actual: 0,
            });
        }
        if !allow_support_restriction && support.iter().any(|active| !active) {
            return Err(DirichletBayesianModelReductionError::InvalidSupportMask {
                distribution,
                column: column_index,
                expected: column.len(),
                actual: support.iter().filter(|active| **active).count(),
            });
        }
    }
    Ok(())
}

/// The full posterior and prior must describe exactly the same categorical
/// simplex faces; learning changes concentrations, not model support.
fn ensure_support_equal(
    full_posterior: &DirichletConcentrationTensor,
    full_prior: &DirichletConcentrationTensor,
    distribution: &'static str,
) -> Result<(), DirichletBayesianModelReductionError> {
    for column in 0..full_prior.active_support.len() {
        for outcome in 0..full_prior.active_support[column].len() {
            if full_prior.active_support[column][outcome]
                != full_posterior.active_support[column][outcome]
            {
                return Err(DirichletBayesianModelReductionError::SupportMismatch {
                    distribution,
                    column,
                    outcome,
                });
            }
        }
    }
    Ok(())
}

/// Reduced models may remove support, but may not introduce outcomes that
/// are impossible under the full-model likelihood family.
fn ensure_support_is_subset(
    full_model: &DirichletConcentrationTensor,
    reduced_model: &DirichletConcentrationTensor,
    distribution: &'static str,
) -> Result<(), DirichletBayesianModelReductionError> {
    for column in 0..full_model.active_support.len() {
        for outcome in 0..full_model.active_support[column].len() {
            if reduced_model.active_support[column][outcome]
                && !full_model.active_support[column][outcome]
            {
                return Err(DirichletBayesianModelReductionError::SupportMismatch {
                    distribution,
                    column,
                    outcome,
                });
            }
        }
    }
    Ok(())
}

fn ensure_parameterization_matches(
    expected: &str,
    distribution: &'static str,
    actual: &str,
) -> Result<(), DirichletBayesianModelReductionError> {
    if actual != expected {
        return Err(
            DirichletBayesianModelReductionError::ParameterizationMismatch {
                distribution,
                expected: expected.to_owned(),
                actual: actual.to_owned(),
            },
        );
    }
    Ok(())
}

fn is_sha256_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn log_multivariate_beta(concentrations: &[f64]) -> f64 {
    concentrations
        .iter()
        .map(|value| log_gamma_positive(*value))
        .sum::<f64>()
        - log_gamma_positive(concentrations.iter().sum::<f64>())
}

/// Lanczos approximation for log(Gamma(x)), with reflection for 0 < x < 1/2.
fn log_gamma_positive(value: f64) -> f64 {
    if value < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * value).sin().ln()
            - log_gamma_positive(1.0 - value);
    }

    let shifted = value - 1.0;
    let mut series = LANCZOS_COEFFICIENTS[0];
    for (index, coefficient) in LANCZOS_COEFFICIENTS.iter().copied().enumerate().skip(1) {
        series += coefficient / (shifted + index as f64);
    }
    let t = shifted + LANCZOS_G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (shifted + 0.5) * t.ln() - t + series.ln()
}

/// One possible observation under an action, with its full-model predictive
/// probability and the expected Dirichlet sufficient-statistic increment.
///
/// The increment is typically the outer product of the predicted outcome
/// modality and posterior expectations over its parent state factors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirichletPredictedOutcome {
    pub outcome_id: String,
    pub full_model_predictive_probability: f64,
    pub posterior_count_increment: Vec<Vec<f64>>,
}

/// All possible outcomes for one candidate action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirichletActionPrediction {
    pub action_id: String,
    pub outcomes: Vec<DirichletPredictedOutcome>,
}

/// Posterior model belief induced by one predicted outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirichletOutcomeModelBelief {
    pub outcome_id: String,
    pub predictive_probability: f64,
    pub full_model_predictive_probability: f64,
    pub posterior_model_entropy_nats: f64,
    pub posterior_model_probabilities: Vec<(String, f64)>,
}

/// Prospective expected information gain about model structure for one action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirichletModelInformationGain {
    pub action_id: String,
    pub current_model_entropy_nats: f64,
    pub expected_posterior_model_entropy_nats: f64,
    pub model_information_gain_nats: f64,
    /// Must be approximately one; retained as an auditable coherence diagnostic.
    pub predictive_probability_mass: f64,
    pub outcomes: Vec<DirichletOutcomeModelBelief>,
}

/// Fail-closed validation errors for prospective Dirichlet model information gain.
#[derive(Debug, Clone, PartialEq)]
pub enum DirichletModelInformationGainError {
    ModelReduction(DirichletBayesianModelReductionError),
    EmptyActionId,
    EmptyOutcomeSet,
    EmptyOutcomeId {
        index: usize,
    },
    DuplicateOutcomeId(String),
    InvalidPredictiveProbability {
        outcome_id: String,
        probability: f64,
    },
    PredictiveProbabilityMassNotNormalized {
        sum: f64,
    },
    CountIncrementColumnCountMismatch {
        outcome_id: String,
        expected: usize,
        actual: usize,
    },
    CountIncrementOutcomeCountMismatch {
        outcome_id: String,
        column: usize,
        expected: usize,
        actual: usize,
    },
    InvalidCountIncrement {
        outcome_id: String,
        column: usize,
        outcome: usize,
        increment: f64,
    },
    ModelAveragePriorMismatch {
        column: usize,
        outcome: usize,
        expected: f64,
        actual: f64,
    },
    InvalidModelInformationGain {
        value: f64,
    },
}

impl std::fmt::Display for DirichletModelInformationGainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelReduction(error) => write!(f, "Dirichlet BMR failed: {error}"),
            Self::EmptyActionId => write!(f, "action ID must not be empty"),
            Self::EmptyOutcomeSet => write!(f, "an action must have at least one possible outcome"),
            Self::EmptyOutcomeId { index } => {
                write!(f, "predicted outcome at index {index} has an empty ID")
            }
            Self::DuplicateOutcomeId(id) => write!(f, "duplicate predicted outcome ID: {id}"),
            Self::InvalidPredictiveProbability {
                outcome_id,
                probability,
            } => write!(
                f,
                "outcome {outcome_id} has invalid full-model predictive probability {probability}"
            ),
            Self::PredictiveProbabilityMassNotNormalized { sum } => {
                write!(
                    f,
                    "full-model predictive outcome mass sums to {sum}, expected 1"
                )
            }
            Self::CountIncrementColumnCountMismatch {
                outcome_id,
                expected,
                actual,
            } => write!(
                f,
                "outcome {outcome_id} count increment has {actual} columns, expected {expected}"
            ),
            Self::CountIncrementOutcomeCountMismatch {
                outcome_id,
                column,
                expected,
                actual,
            } => write!(
                f,
                "outcome {outcome_id} count increment column {column} has {actual} entries, expected {expected}"
            ),
            Self::InvalidCountIncrement {
                outcome_id,
                column,
                outcome,
                increment,
            } => write!(
                f,
                "outcome {outcome_id} has invalid count increment {increment} at column {column}, coordinate {outcome}"
            ),
            Self::ModelAveragePriorMismatch {
                column,
                outcome,
                expected,
                actual,
            } => write!(
                f,
                "full prior at column {column}, coordinate {outcome} is {actual}, but the model-prior-weighted reduced priors average to {expected}"
            ),
            Self::InvalidModelInformationGain { value } => {
                write!(f, "prospective model information gain is invalid: {value}")
            }
        }
    }
}

impl std::error::Error for DirichletModelInformationGainError {}

/// Evaluate the one-step expected information gain about model structure.
///
/// full_posterior is the current posterior under the Bayesian model-average
/// parent model. For each possible outcome, posterior_count_increment is
/// added to those full-model counts, and BMR computes the posterior over
/// reduced models without refitting. The full prior must equal the
/// model-probability-weighted mean of the reduced priors (the Bayesian model
/// average constraint in Friston et al., 2026).
///
/// The caller supplies each action outcome's predictive probability under
/// the full Bayesian-model-average parent model. Those probabilities are used
/// directly to average post-outcome model entropy and must sum to one. BMR
/// evidence ratios update model identity after a hypothetical outcome; they
/// do not redefine or reweight the outcome probabilities a second time.
/// The caller is responsible for deriving predictive probabilities and
/// sufficient-statistic increments from the same generative-model family,
/// parent beliefs, action, and ordered parameterization. This module validates
/// the Dirichlet parameterization/support and probability mass, but it does
/// not itself implement a general observation likelihood or perception model.
///
/// This function scores model information only. Parameter and state EIG,
/// preference/risk, ambiguity, action cost, and multi-step planning remain
/// separate terms in a complete expected-free-energy policy.
pub fn evaluate_dirichlet_model_information_gain(
    full_posterior: &DirichletConcentrationTensor,
    full_prior: &DirichletConcentrationTensor,
    reduced_models: &[ReducedDirichletModelHypothesis],
    action: &DirichletActionPrediction,
) -> Result<DirichletModelInformationGain, DirichletModelInformationGainError> {
    validate_tensor(full_posterior, "full posterior", None, true)
        .map_err(DirichletModelInformationGainError::ModelReduction)?;
    validate_tensor(full_prior, "full prior", Some(full_posterior), true)
        .map_err(DirichletModelInformationGainError::ModelReduction)?;
    ensure_support_equal(full_posterior, full_prior, "full prior")
        .map_err(DirichletModelInformationGainError::ModelReduction)?;
    ensure_parameterization_matches(
        &full_posterior.parameterization_digest,
        "full prior",
        &full_prior.parameterization_digest,
    )
    .map_err(DirichletModelInformationGainError::ModelReduction)?;

    // This is the current posterior over model identity, not the original
    // model prior. BMR evaluates it from evidence accumulated so far.
    let current_model_belief = reduce_dirichlet_models(full_posterior, full_prior, reduced_models)
        .map_err(DirichletModelInformationGainError::ModelReduction)?;
    validate_model_average_prior(full_prior, reduced_models)?;

    if action.action_id.trim().is_empty() {
        return Err(DirichletModelInformationGainError::EmptyActionId);
    }
    if action.outcomes.is_empty() {
        return Err(DirichletModelInformationGainError::EmptyOutcomeSet);
    }

    let mut outcome_ids = BTreeSet::new();
    let mut full_probability_mass = 0.0;
    for (index, outcome) in action.outcomes.iter().enumerate() {
        if outcome.outcome_id.trim().is_empty() {
            return Err(DirichletModelInformationGainError::EmptyOutcomeId { index });
        }
        if !outcome_ids.insert(outcome.outcome_id.clone()) {
            return Err(DirichletModelInformationGainError::DuplicateOutcomeId(
                outcome.outcome_id.clone(),
            ));
        }
        let probability = outcome.full_model_predictive_probability;
        if !probability.is_finite() || probability <= 0.0 || probability > 1.0 {
            return Err(
                DirichletModelInformationGainError::InvalidPredictiveProbability {
                    outcome_id: outcome.outcome_id.clone(),
                    probability,
                },
            );
        }
        full_probability_mass += probability;
        if outcome.posterior_count_increment.len() != full_posterior.columns.len() {
            return Err(
                DirichletModelInformationGainError::CountIncrementColumnCountMismatch {
                    outcome_id: outcome.outcome_id.clone(),
                    expected: full_posterior.columns.len(),
                    actual: outcome.posterior_count_increment.len(),
                },
            );
        }
        for (column, (increment, posterior_column)) in outcome
            .posterior_count_increment
            .iter()
            .zip(full_posterior.columns.iter())
            .enumerate()
        {
            if increment.len() != posterior_column.len() {
                return Err(
                    DirichletModelInformationGainError::CountIncrementOutcomeCountMismatch {
                        outcome_id: outcome.outcome_id.clone(),
                        column,
                        expected: posterior_column.len(),
                        actual: increment.len(),
                    },
                );
            }
            for (coordinate, value) in increment.iter().copied().enumerate() {
                if !value.is_finite() || value < 0.0 {
                    return Err(DirichletModelInformationGainError::InvalidCountIncrement {
                        outcome_id: outcome.outcome_id.clone(),
                        column,
                        outcome: coordinate,
                        increment: value,
                    });
                }
            }
        }
    }
    if (full_probability_mass - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(
            DirichletModelInformationGainError::PredictiveProbabilityMassNotNormalized {
                sum: full_probability_mass,
            },
        );
    }

    let current_model_entropy_nats = current_model_belief.posterior_model_entropy_nats;

    let mut outcomes = Vec::with_capacity(action.outcomes.len());
    let mut expected_posterior_model_entropy_nats = 0.0;

    for outcome in &action.outcomes {
        let mut hypothetical_posterior = full_posterior.clone();
        for (column, increments) in outcome.posterior_count_increment.iter().enumerate() {
            for (coordinate, increment) in increments.iter().copied().enumerate() {
                hypothetical_posterior.columns[column][coordinate] += increment;
            }
        }

        let reduced = reduce_dirichlet_models(&hypothetical_posterior, full_prior, reduced_models)
            .map_err(DirichletModelInformationGainError::ModelReduction)?;

        // Eq. 10-11: expectation is under the predictive outcome distribution
        // of the current full / Bayesian-model-average parent model. BMR supplies
        // the current and post-outcome model beliefs; it does not redefine the
        // action's observation probabilities.
        let predictive_probability = outcome.full_model_predictive_probability;
        expected_posterior_model_entropy_nats +=
            predictive_probability * reduced.posterior_model_entropy_nats;
        outcomes.push(DirichletOutcomeModelBelief {
            outcome_id: outcome.outcome_id.clone(),
            predictive_probability,
            full_model_predictive_probability: outcome.full_model_predictive_probability,
            posterior_model_entropy_nats: reduced.posterior_model_entropy_nats,
            posterior_model_probabilities: reduced
                .models
                .into_iter()
                .map(|model| (model.model_id, model.posterior_probability))
                .collect(),
        });
    }

    let raw_information_gain = current_model_entropy_nats - expected_posterior_model_entropy_nats;
    if !raw_information_gain.is_finite() || raw_information_gain < -1e-8 {
        return Err(
            DirichletModelInformationGainError::InvalidModelInformationGain {
                value: raw_information_gain,
            },
        );
    }
    Ok(DirichletModelInformationGain {
        action_id: action.action_id.clone(),
        current_model_entropy_nats,
        expected_posterior_model_entropy_nats,
        model_information_gain_nats: raw_information_gain.max(0.0),
        predictive_probability_mass: full_probability_mass,
        outcomes,
    })
}

fn validate_model_average_prior(
    full_prior: &DirichletConcentrationTensor,
    reduced_models: &[ReducedDirichletModelHypothesis],
) -> Result<(), DirichletModelInformationGainError> {
    for column in 0..full_prior.columns.len() {
        for coordinate in 0..full_prior.columns[column].len() {
            let expected = reduced_models
                .iter()
                .map(|model| {
                    model.prior_probability * model.parameter_prior.columns[column][coordinate]
                })
                .sum::<f64>();
            let actual = full_prior.columns[column][coordinate];
            let tolerance = 1e-8 * expected.abs().max(actual.abs()).max(1.0);
            if (expected - actual).abs() > tolerance {
                return Err(
                    DirichletModelInformationGainError::ModelAveragePriorMismatch {
                        column,
                        outcome: coordinate,
                        expected,
                        actual,
                    },
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn tensor(columns: &[&[f64]], parameterization: char) -> DirichletConcentrationTensor {
        DirichletConcentrationTensor {
            parameterization_digest: digest(parameterization),
            columns: columns.iter().map(|column| column.to_vec()).collect(),
            active_support: columns
                .iter()
                .map(|column| vec![true; column.len()])
                .collect(),
        }
    }

    fn tensor_with_support(
        columns: &[&[f64]],
        active_support: &[&[bool]],
        parameterization: char,
    ) -> DirichletConcentrationTensor {
        DirichletConcentrationTensor {
            parameterization_digest: digest(parameterization),
            columns: columns.iter().map(|column| column.to_vec()).collect(),
            active_support: active_support
                .iter()
                .map(|support| support.to_vec())
                .collect(),
        }
    }

    #[test]
    fn identical_reduced_and_full_priors_have_zero_log_evidence_ratio() {
        let full_prior = tensor(&[&[1.0, 1.0], &[0.5, 1.5, 2.0]], 'a');
        let posterior = tensor(&[&[4.0, 2.0], &[1.5, 2.5, 3.0]], 'a');
        let result = reduce_dirichlet_models(
            &posterior,
            &full_prior,
            &[ReducedDirichletModelHypothesis {
                model_id: "full".into(),
                prior_probability: 1.0,
                parameter_prior: full_prior.clone(),
            }],
        )
        .expect("identical priors are valid");
        assert!(result.models[0].log_evidence_ratio.abs() < 1e-12);
        assert!((result.models[0].posterior_probability - 1.0).abs() < 1e-12);
        assert_eq!(result.most_probable_model_id, "full");
    }

    #[test]
    fn beta_bmr_matches_independent_beta_binomial_evidence_ratio() {
        // Full Beta(1,1) prior plus observations (3 successes, 1 failure)
        // has evidence B(4,2)/B(1,1) = 1/20.
        // Reduced Beta(2,1) prior has evidence B(5,2)/B(2,1) = 1/15.
        // The reduced/full ratio is therefore 4/3.
        let posterior = tensor(&[&[4.0, 2.0]], 'a');
        let full_prior = tensor(&[&[1.0, 1.0]], 'a');
        let reduced_prior = tensor(&[&[2.0, 1.0]], 'a');
        let result = reduce_dirichlet_models(
            &posterior,
            &full_prior,
            &[ReducedDirichletModelHypothesis {
                model_id: "reduced".into(),
                prior_probability: 1.0,
                parameter_prior: reduced_prior,
            }],
        )
        .expect("proper reduced posterior");
        assert!((result.models[0].log_evidence_ratio - (4.0_f64 / 3.0).ln()).abs() < 1e-11);
    }

    #[test]
    fn independent_likelihood_columns_have_additive_log_evidence_ratios() {
        let posterior = tensor(&[&[4.0, 2.0], &[5.0, 3.0]], 'a');
        let full_prior = tensor(&[&[1.0, 1.0], &[1.0, 2.0]], 'a');
        let reduced = tensor(&[&[2.0, 1.0], &[1.5, 2.5]], 'a');
        let joint = reduce_dirichlet_models(
            &posterior,
            &full_prior,
            &[ReducedDirichletModelHypothesis {
                model_id: "joint".into(),
                prior_probability: 1.0,
                parameter_prior: reduced.clone(),
            }],
        )
        .expect("joint reduction");
        let first = reduce_dirichlet_models(
            &tensor(&[&[4.0, 2.0]], 'a'),
            &tensor(&[&[1.0, 1.0]], 'a'),
            &[ReducedDirichletModelHypothesis {
                model_id: "first".into(),
                prior_probability: 1.0,
                parameter_prior: tensor(&[&[2.0, 1.0]], 'a'),
            }],
        )
        .expect("first column");
        let second = reduce_dirichlet_models(
            &tensor(&[&[5.0, 3.0]], 'a'),
            &tensor(&[&[1.0, 2.0]], 'a'),
            &[ReducedDirichletModelHypothesis {
                model_id: "second".into(),
                prior_probability: 1.0,
                parameter_prior: tensor(&[&[1.5, 2.5]], 'a'),
            }],
        )
        .expect("second column");
        let expected = first.models[0].log_evidence_ratio + second.models[0].log_evidence_ratio;
        assert!((joint.models[0].log_evidence_ratio - expected).abs() < 1e-12);
    }

    #[test]
    fn model_prior_odds_are_combined_with_dirichlet_evidence() {
        let posterior = tensor(&[&[4.0, 2.0]], 'a');
        let full_prior = tensor(&[&[1.0, 1.0]], 'a');
        let result = reduce_dirichlet_models(
            &posterior,
            &full_prior,
            &[
                ReducedDirichletModelHypothesis {
                    model_id: "full".into(),
                    prior_probability: 0.5,
                    parameter_prior: full_prior.clone(),
                },
                ReducedDirichletModelHypothesis {
                    model_id: "reduced".into(),
                    prior_probability: 0.5,
                    parameter_prior: tensor(&[&[2.0, 1.0]], 'a'),
                },
            ],
        )
        .expect("valid model priors");
        assert_eq!(result.most_probable_model_id, "reduced");
        assert!(result.log_posterior_odds_over_runner_up.unwrap() > 0.0);
    }

    #[test]
    fn malformed_dimensions_digests_and_model_prior_mass_are_rejected() {
        let posterior = tensor(&[&[4.0, 2.0]], 'a');
        let full_prior = tensor(&[&[1.0, 1.0]], 'a');
        let wrong_outcome_count = reduce_dirichlet_models(
            &posterior,
            &full_prior,
            &[ReducedDirichletModelHypothesis {
                model_id: "bad-shape".into(),
                prior_probability: 1.0,
                parameter_prior: tensor(&[&[1.0, 1.0, 1.0]], 'a'),
            }],
        );
        assert!(matches!(
            wrong_outcome_count,
            Err(DirichletBayesianModelReductionError::OutcomeCountMismatch { .. })
        ));

        let wrong_digest = reduce_dirichlet_models(
            &posterior,
            &full_prior,
            &[ReducedDirichletModelHypothesis {
                model_id: "bad-coordinate-system".into(),
                prior_probability: 1.0,
                parameter_prior: tensor(&[&[2.0, 1.0]], 'b'),
            }],
        );
        assert!(matches!(
            wrong_digest,
            Err(DirichletBayesianModelReductionError::ParameterizationMismatch { .. })
        ));

        let invalid_prior_mass = reduce_dirichlet_models(
            &posterior,
            &full_prior,
            &[
                ReducedDirichletModelHypothesis {
                    model_id: "a".into(),
                    prior_probability: 0.2,
                    parameter_prior: full_prior.clone(),
                },
                ReducedDirichletModelHypothesis {
                    model_id: "b".into(),
                    prior_probability: 0.2,
                    parameter_prior: full_prior.clone(),
                },
            ],
        );
        assert!(matches!(
            invalid_prior_mass,
            Err(DirichletBayesianModelReductionError::ModelPriorMassNotNormalized { .. })
        ));
    }

    #[test]
    fn full_model_support_masks_allow_structural_zero_outcomes() {
        let full_prior = tensor_with_support(
            &[&[1.0, 0.0, 0.0], &[0.0, 1.0, 1.0]],
            &[&[true, false, false], &[false, true, true]],
            'a',
        );
        let posterior = tensor_with_support(
            &[&[4.0, 0.0, 0.0], &[0.0, 3.0, 1.0]],
            &[&[true, false, false], &[false, true, true]],
            'a',
        );
        let reduced_prior = tensor_with_support(
            &[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]],
            &[&[true, false, false], &[false, true, false]],
            'a',
        );
        let result = reduce_dirichlet_models(
            &posterior,
            &full_prior,
            &[ReducedDirichletModelHypothesis {
                model_id: "no-incorrect-feedback".into(),
                prior_probability: 1.0,
                parameter_prior: reduced_prior,
            }],
        )
        .expect("full and reduced models have valid explicit support");
        // The second column has two observed correct outcomes and no incorrect
        // outcomes. The reduced model fixes P(correct)=1, so BF = 3.
        assert!((result.models[0].log_evidence_ratio - 3.0_f64.ln()).abs() < 1e-11);
    }

    #[test]
    fn full_support_mismatch_and_reduced_support_expansion_are_rejected() {
        let prior = tensor_with_support(&[&[1.0, 0.0, 0.0]], &[&[true, false, false]], 'a');
        let mismatched_posterior =
            tensor_with_support(&[&[2.0, 1.0, 0.0]], &[&[true, true, false]], 'a');
        assert!(matches!(
            reduce_dirichlet_models(
                &mismatched_posterior,
                &prior,
                &[ReducedDirichletModelHypothesis {
                    model_id: "bad-full-support".into(),
                    prior_probability: 1.0,
                    parameter_prior: prior.clone(),
                }]
            ),
            Err(DirichletBayesianModelReductionError::SupportMismatch {
                distribution: "full prior",
                ..
            })
        ));

        let posterior = tensor_with_support(&[&[2.0, 0.0, 0.0]], &[&[true, false, false]], 'a');
        let expanded_reduced_support = tensor(&[&[1.0, 1.0, 1.0]], 'a');
        assert!(matches!(
            reduce_dirichlet_models(
                &posterior,
                &prior,
                &[ReducedDirichletModelHypothesis {
                    model_id: "expanded-support".into(),
                    prior_probability: 1.0,
                    parameter_prior: expanded_reduced_support,
                }]
            ),
            Err(DirichletBayesianModelReductionError::SupportMismatch {
                distribution: "reduced prior",
                ..
            })
        ));
    }

    #[test]
    fn full_column_with_empty_support_is_rejected() {
        let empty = tensor_with_support(&[&[0.0, 0.0]], &[&[false, false]], 'a');
        let result = reduce_dirichlet_models(
            &empty,
            &empty,
            &[ReducedDirichletModelHypothesis {
                model_id: "empty-full-support".into(),
                prior_probability: 1.0,
                parameter_prior: empty.clone(),
            }],
        );
        assert!(matches!(
            result,
            Err(DirichletBayesianModelReductionError::EmptySupport {
                distribution: "full posterior",
                column: 0
            })
        ));
    }

    #[test]
    fn improper_zero_concentrations_and_incompatible_reduced_posterior_are_rejected() {
        let posterior = tensor(&[&[1.5, 1.5]], 'a');
        let full_prior = tensor(&[&[1.0, 1.0]], 'a');
        let zero_prior = tensor(&[&[0.0, 1.0]], 'a');
        let zero_result = reduce_dirichlet_models(
            &posterior,
            &full_prior,
            &[ReducedDirichletModelHypothesis {
                model_id: "zero-support".into(),
                prior_probability: 1.0,
                parameter_prior: zero_prior,
            }],
        );
        assert!(matches!(
            zero_result,
            Err(DirichletBayesianModelReductionError::InvalidConcentration { .. })
        ));

        let incompatible = tensor(&[&[0.1, 1.0]], 'a');
        let result = reduce_dirichlet_models(
            &tensor(&[&[0.2, 1.1]], 'a'),
            &full_prior,
            &[ReducedDirichletModelHypothesis {
                model_id: "posterior-support-conflict".into(),
                prior_probability: 1.0,
                parameter_prior: incompatible,
            }],
        );
        assert!(matches!(
            result,
            Err(
                DirichletBayesianModelReductionError::NonPositiveReducedPosteriorConcentration { .. }
            )
        ));
    }

    #[test]
    fn explicit_support_restriction_has_finite_evidence_when_excluded_outcome_is_unobserved() {
        let full_prior = tensor(&[&[1.0, 1.0]], 'a');
        let posterior = tensor(&[&[4.0, 1.0]], 'a');
        let reduced_prior = tensor_with_support(&[&[2.0, 0.0]], &[&[true, false]], 'a');
        let result = reduce_dirichlet_models(
            &posterior,
            &full_prior,
            &[ReducedDirichletModelHypothesis {
                model_id: "one-outcome-support".into(),
                prior_probability: 1.0,
                parameter_prior: reduced_prior,
            }],
        )
        .expect("proper support-restricted model");
        // Full Beta(1,1) assigns evidence 1/4 to three successes and no failures.
        // The reduced model fixes P(success)=1, so its evidence is 1 and BF=4.
        assert!((result.models[0].log_evidence_ratio - 4.0_f64.ln()).abs() < 1e-11);
    }

    #[test]
    fn observing_an_excluded_outcome_gives_the_reduced_model_zero_evidence() {
        let full_prior = tensor(&[&[1.0, 1.0]], 'a');
        let posterior = tensor(&[&[4.0, 2.0]], 'a');
        let reduced_prior = tensor_with_support(&[&[2.0, 0.0]], &[&[true, false]], 'a');
        let result = reduce_dirichlet_models(
            &posterior,
            &full_prior,
            &[
                ReducedDirichletModelHypothesis {
                    model_id: "full".into(),
                    prior_probability: 0.5,
                    parameter_prior: full_prior.clone(),
                },
                ReducedDirichletModelHypothesis {
                    model_id: "forbids-observed-outcome".into(),
                    prior_probability: 0.5,
                    parameter_prior: reduced_prior,
                },
            ],
        )
        .expect("full model retains positive evidence");
        assert_eq!(result.models[1].log_evidence_ratio, f64::NEG_INFINITY);
        assert_eq!(result.models[1].posterior_probability, 0.0);
        assert_eq!(result.models[0].posterior_probability, 1.0);
    }

    #[test]
    fn empty_reduced_support_is_rejected() {
        let full_prior = tensor(&[&[1.0, 1.0]], 'a');
        let posterior = tensor(&[&[1.0, 1.0]], 'a');
        let empty = tensor_with_support(&[&[0.0, 0.0]], &[&[false, false]], 'a');
        let result = reduce_dirichlet_models(
            &posterior,
            &full_prior,
            &[ReducedDirichletModelHypothesis {
                model_id: "empty-support".into(),
                prior_probability: 1.0,
                parameter_prior: empty,
            }],
        );
        assert!(matches!(
            result,
            Err(DirichletBayesianModelReductionError::EmptyReducedSupport { .. })
        ));
    }

    fn symmetric_beta_model_family() -> (
        DirichletConcentrationTensor,
        Vec<ReducedDirichletModelHypothesis>,
    ) {
        let full = tensor(&[&[2.5, 2.5]], 'a');
        let models = vec![
            ReducedDirichletModelHypothesis {
                model_id: "success-favouring".into(),
                prior_probability: 0.5,
                parameter_prior: tensor(&[&[4.0, 1.0]], 'a'),
            },
            ReducedDirichletModelHypothesis {
                model_id: "failure-favouring".into(),
                prior_probability: 0.5,
                parameter_prior: tensor(&[&[1.0, 4.0]], 'a'),
            },
        ];
        (full, models)
    }

    #[test]
    fn prospective_dirichlet_model_eig_matches_beta_binomial_information_gain() {
        let (full_prior, models) = symmetric_beta_model_family();
        let action = DirichletActionPrediction {
            action_id: "inspect-outcome".into(),
            outcomes: vec![
                DirichletPredictedOutcome {
                    outcome_id: "success".into(),
                    full_model_predictive_probability: 0.5,
                    posterior_count_increment: vec![vec![1.0, 0.0]],
                },
                DirichletPredictedOutcome {
                    outcome_id: "failure".into(),
                    full_model_predictive_probability: 0.5,
                    posterior_count_increment: vec![vec![0.0, 1.0]],
                },
            ],
        };
        let result =
            evaluate_dirichlet_model_information_gain(&full_prior, &full_prior, &models, &action)
                .expect("coherent one-step BMR model-EIG prediction");

        let expected_posterior_entropy = -(0.8_f64 * 0.8_f64.ln() + 0.2_f64 * 0.2_f64.ln());
        let expected_information_gain = std::f64::consts::LN_2 - expected_posterior_entropy;
        assert!((result.current_model_entropy_nats - std::f64::consts::LN_2).abs() < 1e-12);
        assert!(
            (result.expected_posterior_model_entropy_nats - expected_posterior_entropy).abs()
                < 1e-10
        );
        assert!((result.model_information_gain_nats - expected_information_gain).abs() < 1e-10);
        assert!((result.predictive_probability_mass - 1.0).abs() < 1e-12);
        assert!((result.outcomes[0].predictive_probability - 0.5).abs() < 1e-12);
        assert!((result.outcomes[1].predictive_probability - 0.5).abs() < 1e-12);
        assert!((result.outcomes[0].posterior_model_probabilities[0].1 - 0.8).abs() < 1e-10);
        assert!((result.outcomes[1].posterior_model_probabilities[0].1 - 0.2).abs() < 1e-10);
    }

    #[test]
    fn uninformative_outcomes_have_zero_prospective_model_eig() {
        let (full_prior, models) = symmetric_beta_model_family();
        let action = DirichletActionPrediction {
            action_id: "uninformative-action".into(),
            outcomes: vec![
                DirichletPredictedOutcome {
                    outcome_id: "outcome-a".into(),
                    full_model_predictive_probability: 0.3,
                    posterior_count_increment: vec![vec![0.0, 0.0]],
                },
                DirichletPredictedOutcome {
                    outcome_id: "outcome-b".into(),
                    full_model_predictive_probability: 0.7,
                    posterior_count_increment: vec![vec![0.0, 0.0]],
                },
            ],
        };
        let result =
            evaluate_dirichlet_model_information_gain(&full_prior, &full_prior, &models, &action)
                .expect("uninformative but normalized action prediction");
        assert!(result.model_information_gain_nats.abs() < 1e-12);
        assert!((result.outcomes[0].posterior_model_probabilities[0].1 - 0.5).abs() < 1e-12);
        assert!((result.outcomes[1].posterior_model_probabilities[0].1 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn prospective_model_eig_is_invariant_to_exact_duplicate_model_labels() {
        let full_prior = tensor(&[&[1.5, 1.5]], 'a');
        let duplicated_models = vec![
            ReducedDirichletModelHypothesis {
                model_id: "red-labelled-a".into(),
                prior_probability: 0.2,
                parameter_prior: tensor(&[&[2.0, 1.0]], 'a'),
            },
            ReducedDirichletModelHypothesis {
                model_id: "red-labelled-b".into(),
                prior_probability: 0.3,
                parameter_prior: tensor(&[&[2.0, 1.0]], 'a'),
            },
            ReducedDirichletModelHypothesis {
                model_id: "blue".into(),
                prior_probability: 0.5,
                parameter_prior: tensor(&[&[1.0, 2.0]], 'a'),
            },
        ];
        let quotient_models = vec![
            ReducedDirichletModelHypothesis {
                model_id: "red-equivalence-class".into(),
                prior_probability: 0.5,
                parameter_prior: tensor(&[&[2.0, 1.0]], 'a'),
            },
            ReducedDirichletModelHypothesis {
                model_id: "blue-equivalence-class".into(),
                prior_probability: 0.5,
                parameter_prior: tensor(&[&[1.0, 2.0]], 'a'),
            },
        ];
        let action = DirichletActionPrediction {
            action_id: "one-observation".into(),
            outcomes: vec![
                DirichletPredictedOutcome {
                    outcome_id: "success".into(),
                    full_model_predictive_probability: 0.5,
                    posterior_count_increment: vec![vec![1.0, 0.0]],
                },
                DirichletPredictedOutcome {
                    outcome_id: "failure".into(),
                    full_model_predictive_probability: 0.5,
                    posterior_count_increment: vec![vec![0.0, 1.0]],
                },
            ],
        };

        let labelled = evaluate_dirichlet_model_information_gain(
            &full_prior,
            &full_prior,
            &duplicated_models,
            &action,
        )
        .expect("labelled model EIG");
        let quotient = evaluate_dirichlet_model_information_gain(
            &full_prior,
            &full_prior,
            &quotient_models,
            &action,
        )
        .expect("quotient model EIG");

        assert!(
            (labelled.model_information_gain_nats - quotient.model_information_gain_nats).abs()
                < 1e-12
        );
        assert!(
            (labelled.expected_posterior_model_entropy_nats
                - quotient.expected_posterior_model_entropy_nats
                - 0.2_f64 * 0.2_f64.ln()
                - 0.3_f64 * 0.3_f64.ln()
                + 0.5_f64 * 0.5_f64.ln())
            .abs()
                < 1e-10
        );

        for (labelled_outcome, quotient_outcome) in labelled.outcomes.iter().zip(&quotient.outcomes)
        {
            let labelled_red_mass = labelled_outcome
                .posterior_model_probabilities
                .iter()
                .filter(|(model_id, _)| model_id.starts_with("red-labelled-"))
                .map(|(_, probability)| *probability)
                .sum::<f64>();
            let quotient_red_mass = quotient_outcome
                .posterior_model_probabilities
                .iter()
                .find(|(model_id, _)| model_id == "red-equivalence-class")
                .map(|(_, probability)| *probability)
                .expect("red quotient posterior");
            assert!((labelled_red_mass - quotient_red_mass).abs() < 1e-12);
        }
    }

    #[test]
    fn prospective_model_eig_rejects_non_bma_priors_and_bad_predictive_mass() {
        let (full_prior, models) = symmetric_beta_model_family();
        let action = DirichletActionPrediction {
            action_id: "bad-prediction".into(),
            outcomes: vec![
                DirichletPredictedOutcome {
                    outcome_id: "success".into(),
                    full_model_predictive_probability: 0.6,
                    posterior_count_increment: vec![vec![1.0, 0.0]],
                },
                DirichletPredictedOutcome {
                    outcome_id: "failure".into(),
                    full_model_predictive_probability: 0.3,
                    posterior_count_increment: vec![vec![0.0, 1.0]],
                },
            ],
        };
        assert!(matches!(
            evaluate_dirichlet_model_information_gain(&full_prior, &full_prior, &models, &action,),
            Err(DirichletModelInformationGainError::PredictiveProbabilityMassNotNormalized { .. })
        ));

        let mismatched_parent = tensor(&[&[2.6, 2.4]], 'a');
        let coherent_action = DirichletActionPrediction {
            action_id: "coherent-shape-but-wrong-parent".into(),
            outcomes: vec![
                DirichletPredictedOutcome {
                    outcome_id: "success".into(),
                    full_model_predictive_probability: 0.5,
                    posterior_count_increment: vec![vec![1.0, 0.0]],
                },
                DirichletPredictedOutcome {
                    outcome_id: "failure".into(),
                    full_model_predictive_probability: 0.5,
                    posterior_count_increment: vec![vec![0.0, 1.0]],
                },
            ],
        };
        assert!(matches!(
            evaluate_dirichlet_model_information_gain(
                &mismatched_parent,
                &mismatched_parent,
                &models,
                &coherent_action,
            ),
            Err(DirichletModelInformationGainError::ModelAveragePriorMismatch { .. })
        ));
    }

    #[test]
    fn parameter_order_permutation_is_invariant_when_digest_tracks_same_ordering() {
        let first = reduce_dirichlet_models(
            &tensor(&[&[4.0, 2.0]], 'a'),
            &tensor(&[&[1.0, 1.0]], 'a'),
            &[ReducedDirichletModelHypothesis {
                model_id: "model".into(),
                prior_probability: 1.0,
                parameter_prior: tensor(&[&[2.0, 1.0]], 'a'),
            }],
        )
        .expect("first order");
        let permuted = reduce_dirichlet_models(
            &tensor(&[&[2.0, 4.0]], 'b'),
            &tensor(&[&[1.0, 1.0]], 'b'),
            &[ReducedDirichletModelHypothesis {
                model_id: "model".into(),
                prior_probability: 1.0,
                parameter_prior: tensor(&[&[1.0, 2.0]], 'b'),
            }],
        )
        .expect("permuted order");
        assert!(
            (first.models[0].log_evidence_ratio - permuted.models[0].log_evidence_ratio).abs()
                < 1e-12
        );
    }

    #[test]
    fn log_gamma_and_multivariate_beta_match_known_values() {
        assert!((log_gamma_positive(1.0)).abs() < 1e-14);
        assert!((log_gamma_positive(5.0) - 24.0_f64.ln()).abs() < 1e-13);
        assert!(log_multivariate_beta(&[1.0, 1.0]).abs() < 1e-13);
        assert!((log_multivariate_beta(&[2.0, 1.0]) - (0.5_f64).ln()).abs() < 1e-13);
    }
}
