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
//! This implementation requires proper Dirichlet distributions with strictly
//! positive concentrations. Exact-zero concentrations imply lower-dimensional
//! support / structural point masses, not an ordinary proper Dirichlet; those
//! require an explicit support-restricted model representation and are
//! rejected here. Small positive shrinkage concentrations (e.g. 1/32) are
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
    EmptyColumnSet { distribution: &'static str },
    EmptyColumn {
        distribution: &'static str,
        column: usize,
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
    EmptyModelId { index: usize },
    DuplicateModelId(String),
    InvalidModelPrior {
        model_id: String,
        probability: f64,
    },
    ModelPriorMassNotNormalized { sum: f64 },
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
            Self::EmptyColumn { distribution, column } => {
                write!(f, "{distribution} column {column} is empty")
            }
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
                write!(f, "model {model_id} has empty active support at column {column}")
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
) -> Result<
    DirichletBayesianModelReductionResult,
    DirichletBayesianModelReductionError,
> {
    let column_count = full_posterior.columns.len();
    validate_tensor(full_posterior, "full posterior", None, false)?;
    validate_tensor(full_prior, "full prior", Some(full_posterior), false)?;
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
        if !model.prior_probability.is_finite()
            || !(0.0..=1.0).contains(&model.prior_probability)
        {
            return Err(DirichletBayesianModelReductionError::InvalidModelPrior {
                model_id: model.model_id.clone(),
                probability: model.prior_probability,
            });
        }
        prior_sum += model.prior_probability;
        validate_tensor(
            &model.parameter_prior,
            "reduced prior",
            Some(full_posterior),
            true,
        )?;
        ensure_parameterization_matches(
            &full_posterior.parameterization_digest,
            "reduced prior",
            &model.parameter_prior.parameterization_digest,
        )?;
    }
    if (prior_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(DirichletBayesianModelReductionError::ModelPriorMassNotNormalized {
            sum: prior_sum,
        });
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
            |((model, log_evidence_ratio), posterior_probability)| {
                ReducedDirichletModelEvidence {
                    model_id: model.model_id.clone(),
                    prior_probability: model.prior_probability,
                    log_evidence_ratio,
                    posterior_probability,
                }
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
        let reduced = &reduced_prior.columns[column];
        let reduced_support = &reduced_prior.active_support[column];
        let mut reduced_posterior = Vec::with_capacity(posterior.len());
        let mut reduced_prior_active = Vec::new();
        let mut reduced_posterior_active = Vec::new();

        for outcome in 0..posterior.len() {
            let posterior_increment = posterior[outcome] - full[outcome];
            if !reduced_support[outcome] {
                if posterior_increment < -1e-9 {
                    return Err(DirichletBayesianModelReductionError::PosteriorBelowPrior {
                        model_id: model_id.to_owned(),
                        column,
                        outcome,
                        increment: posterior_increment,
                    });
                }
                if posterior_increment > 1e-9 {
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
            reduced_posterior.push(value);
        }

        if reduced_prior_active.is_empty() {
            return Err(DirichletBayesianModelReductionError::EmptyReducedSupport {
                model_id: model_id.to_owned(),
                column,
            });
        }
        let contribution = log_multivariate_beta(full)
            + log_multivariate_beta(&reduced_posterior_active)
            - log_multivariate_beta(posterior)
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
        return Err(DirichletBayesianModelReductionError::InvalidParameterizationDigest {
            distribution,
        });
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
        for (outcome_index, (concentration, active)) in
            column.iter().copied().zip(support.iter().copied()).enumerate()
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

fn ensure_parameterization_matches(
    expected: &str,
    distribution: &'static str,
    actual: &str,
) -> Result<(), DirichletBayesianModelReductionError> {
    if actual != expected {
        return Err(DirichletBayesianModelReductionError::ParameterizationMismatch {
            distribution,
            expected: expected.to_owned(),
            actual: actual.to_owned(),
        });
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
    0.5 * (2.0 * std::f64::consts::PI).ln()
        + (shifted + 0.5) * t.ln()
        - t
        + series.ln()
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
        assert!(
            (result.models[0].log_evidence_ratio - (4.0_f64 / 3.0).ln()).abs() < 1e-11
        );
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
            Err(DirichletBayesianModelReductionError::NonPositiveReducedPosteriorConcentration { .. })
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
