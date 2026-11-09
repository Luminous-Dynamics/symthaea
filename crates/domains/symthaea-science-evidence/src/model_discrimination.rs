// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Open-world-aware experimental ranking for discrete theory discrimination.
//!
//! The objective is expected information gain about a *discrete model label*,
//! not parameter uncertainty. Candidate experiments are evaluated from
//! calibrated scalar Gaussian predictive distributions using deterministic
//! composite Simpson quadrature.
//!
//! This is deliberately not a full nonlinear BOED/SBI implementation. It is a
//! small, auditable reference kernel for known predictive distributions.
//! Missing predictions, an unresolved open-world model, non-Gaussian noise,
//! or uncalibrated measurement noise are never converted into convenient
//! numeric likelihoods.

use super::theory_prediction::{
    MeasurementNoiseModel, NoiseCalibrationStatus, OTHER_UNRESOLVED_MODEL_ID, PredictionDisposition,
};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

/// Upper bound for the discrete model set in the scalar reference kernel.
pub const MAX_DISCRIMINATION_HYPOTHESES: usize = 16;
/// Upper bound for a single candidate ranking request.
pub const MAX_DISCRIMINATION_EXPERIMENTS: usize = 128;
/// Upper bound for prior-sensitivity scenarios.
pub const MAX_MODEL_PRIOR_SCENARIOS: usize = 8;
/// Maximum coarse Simpson intervals; the refined pass uses twice this count.
pub const MAX_EIG_QUADRATURE_INTERVALS: usize = 1024;

/// Whether the named model set is open-world or has an explicit completeness
/// argument. The unresolved open-world alternative is never assigned a
/// predictive distribution by this module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelUniverseCoverage {
    OpenWorld,
    DeclaredClosed { completeness_reference: String },
}

/// Versioned named model included in a discrete model-identity design.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscriminationHypothesis {
    pub model_id: String,
    /// Full 40-character lowercase Git commit SHA for the model implementation.
    pub source_revision_sha: String,
    /// Stable reference to the model equations, assumptions or specification.
    pub specification_reference: String,
}

/// One declared prior scenario. Probabilities are normalized internally;
/// zero-probability hypotheses remain in the model coverage contract.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelPriorScenario {
    pub scenario_id: String,
    /// Relative importance of this prior scenario in the weighted mean.
    pub scenario_weight: f64,
    /// One non-negative prior weight per DiscriminationHypothesis, in order.
    pub probabilities: Vec<f64>,
}

/// Quantitative predictive distribution for one model under one candidate
/// experiment. Predictive sigma represents model/parameter uncertainty only;
/// calibrated measurement sigma is supplied by CandidateExperiment.
///
/// When embedded in a `DetectionAwareCandidateExperiment`, the predictive
/// mean and sigma describe the value distribution **conditional on D=1**.
/// They must not be treated as an unconditional distribution when detection
/// itself truncates or selection-biases the value. Such non-Gaussian/censored
/// channels are outside this reference model and must block scoring.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianModelPrediction {
    pub model_id: String,
    pub disposition: PredictionDisposition,
    pub predictive_mean: Option<f64>,
    pub predictive_sigma: Option<f64>,
    pub derivation_reference: Option<String>,
}

/// One scalar candidate experiment. The likelihood variance for model i is
/// predictive_sigma_i^2 + measurement_sigma^2. The two uncertainty sources
/// are treated as independent in this initial reference model.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateExperiment {
    pub experiment_id: String,
    pub observable_id: String,
    pub domain_id: String,
    pub units: String,
    pub measurement_noise_model: MeasurementNoiseModel,
    pub measurement_sigma: Option<f64>,
    pub noise_calibration: NoiseCalibrationStatus,
    pub calibration_reference: Option<String>,
    /// Probability this experiment returns a usable observation. Missingness
    /// is assumed uninformative in this reference model, so adjusted EIG is
    /// this probability multiplied by conditional model-identity EIG.
    pub availability_probability: f64,
    pub cost: f64,
    /// Optional mutual-exclusion group for downstream portfolio selection.
    pub exclusive_group: Option<String>,
    /// Exactly one typed prediction record is required per named hypothesis.
    pub predictions: Vec<GaussianModelPrediction>,
}

/// Model-conditional probability that an instrument returns a usable detection.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelDetectionProbability {
    pub model_id: String,
    /// P(D=1 | model, instrument executed), in [0, 1].
    pub probability_detected: f64,
    /// Versioned physics/instrument derivation or calibration reference.
    pub derivation_reference: String,
}

/// Candidate experiment with an explicit model-dependent detection channel.
///
/// The wrapped candidate's `availability_probability` remains the
/// model-independent probability that the experiment is executed successfully.
/// `detection_probabilities` describe P(D=1 | model, executed). When D=1 the
/// wrapped scalar Gaussian measurement model applies; when D=0 this first
/// version observes only the binary non-detection outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectionAwareCandidateExperiment {
    pub experiment: CandidateExperiment,
    /// Exactly one calibrated/derived detection probability per named model.
    pub detection_probabilities: Vec<ModelDetectionProbability>,
}

/// Quadrature and resource limits for model-identity EIG estimation.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelDiscriminationConfig {
    pub model_universe: ModelUniverseCoverage,
    pub prior_scenarios: Vec<ModelPriorScenario>,
    /// Coarse Simpson intervals. A second pass uses twice this count.
    pub quadrature_intervals: usize,
    /// Candidate is not ranked unless |EIG(2N)-EIG(N)| is at or below this.
    pub quadrature_tolerance_nats: f64,
}

/// Why a candidate experiment cannot be scored quantitatively.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExperimentBlocker {
    PredictionNotQuantitative {
        model_id: String,
        disposition: PredictionDisposition,
    },
    PredictionMissingMean {
        model_id: String,
    },
    PredictionMissingUncertainty {
        model_id: String,
    },
    PredictionMissingDerivation {
        model_id: String,
    },
    NonGaussianMeasurementNoise,
    MeasurementNoiseUncalibrated,
    MeasurementSigmaMissing,
    CalibrationReferenceMissing,
}

/// Numerical status for one candidate. A non-comparable result is not a
/// zero-EIG result; its score fields remain None.
#[derive(Clone, Debug, PartialEq)]
pub enum ExperimentDiscriminationStatus {
    Scored,
    NotComparable {
        blockers: Vec<ExperimentBlocker>,
    },
    QuadratureNotConverged {
        maximum_error_nats: f64,
        tolerance_nats: f64,
    },
}

/// EIG estimate under one explicit model-prior scenario.
#[derive(Clone, Debug, PartialEq)]
pub struct PriorScenarioEIG {
    pub scenario_id: String,
    pub prior_entropy_nats: f64,
    /// EIG conditional on obtaining a usable observation.
    pub coarse_eig_nats: f64,
    /// EIG conditional on obtaining a usable observation.
    pub refined_eig_nats: f64,
    /// Availability-adjusted EIG under the uninformative-missingness model.
    pub availability_adjusted_eig_nats: f64,
    /// Expected posterior entropy after accounting for missing observations.
    pub expected_posterior_entropy_nats: f64,
    pub quadrature_error_nats: f64,
    pub converged: bool,
}

/// EIG-based ranking record for one candidate experiment.
#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentDiscriminationScore {
    pub experiment_id: String,
    pub observable_id: String,
    pub cost: f64,
    pub availability_probability: f64,
    pub status: ExperimentDiscriminationStatus,
    pub prior_scenario_scores: Vec<PriorScenarioEIG>,
    /// Conditional EIG assuming a usable observation is obtained.
    pub worst_case_model_identity_eig_nats: Option<f64>,
    /// Conditional EIG assuming a usable observation is obtained.
    pub weighted_mean_model_identity_eig_nats: Option<f64>,
    /// Worst-case EIG after accounting for the probability of a usable observation.
    pub availability_adjusted_worst_case_model_identity_eig_nats: Option<f64>,
    /// Weighted-mean EIG after accounting for the probability of a usable observation.
    pub availability_adjusted_weighted_mean_model_identity_eig_nats: Option<f64>,
    /// Availability-adjusted worst-case EIG per cost.
    pub worst_case_eig_per_cost: Option<f64>,
    /// Availability-adjusted weighted-mean EIG per cost.
    pub weighted_mean_eig_per_cost: Option<f64>,
    /// True means OTHER_UNRESOLVED and any unenumerated model are outside the
    /// numerical EIG calculation; the report does not claim open-world closure.
    pub open_world_unresolved_excluded: bool,
}

/// Ranked candidate experiments with explicit model-universe coverage.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelDiscriminationReport {
    pub model_ids: Vec<String>,
    pub model_universe: ModelUniverseCoverage,
    pub prior_scenario_ids: Vec<String>,
    pub experiments: Vec<ExperimentDiscriminationScore>,
}

/// EIG decomposition for one prior scenario with model-dependent detection.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectionAwarePriorScenarioEIG {
    pub scenario_id: String,
    pub prior_entropy_nats: f64,
    /// P(D=1) averaged over the current model prior.
    pub expected_detection_probability: f64,
    /// Mutual information I(M; D) from the detection/non-detection indicator.
    pub detection_indicator_eig_nats: f64,
    /// I(M; Y | D=1), using the model prior conditioned on detection.
    pub conditional_detected_value_eig_nats: Option<f64>,
    /// I(M; D,Y) conditional on the instrument executing successfully.
    pub total_eig_given_execution_nats: f64,
    /// Expected information after the model-independent execution probability.
    pub availability_adjusted_total_eig_nats: f64,
    pub expected_posterior_entropy_nats: f64,
    /// Coarse/refined quadrature difference for I(M;Y | D=1), before
    /// multiplying by the detection or execution probabilities.
    pub quadrature_error_nats: f64,
    /// The conditional quadrature error scaled by P(D=1) and execution
    /// probability, bounding its contribution to operational EIG.
    pub operational_quadrature_error_bound_nats: f64,
    pub converged: bool,
}

/// Cost-ranked result for one candidate with an explicit detection channel.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectionAwareExperimentScore {
    pub experiment_id: String,
    pub observable_id: String,
    pub cost: f64,
    /// Model-independent execution probability from the wrapped candidate.
    pub execution_probability: f64,
    pub status: ExperimentDiscriminationStatus,
    pub prior_scenario_scores: Vec<DetectionAwarePriorScenarioEIG>,
    pub worst_case_total_eig_nats: Option<f64>,
    pub weighted_mean_total_eig_nats: Option<f64>,
    pub worst_case_total_eig_per_cost: Option<f64>,
    pub weighted_mean_total_eig_per_cost: Option<f64>,
    pub open_world_unresolved_excluded: bool,
}

/// Model-discrimination report for a Bernoulli detection channel followed by
/// the existing scalar Gaussian measurement channel.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectionAwareModelDiscriminationReport {
    pub model_ids: Vec<String>,
    pub model_universe: ModelUniverseCoverage,
    pub prior_scenario_ids: Vec<String>,
    pub experiments: Vec<DetectionAwareExperimentScore>,
}

/// Malformed inputs are errors; scientifically incomplete predictions remain
/// typed NotComparable results.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ModelDiscriminationError {
    #[error("at least two named model hypotheses are required")]
    TooFewHypotheses,
    #[error("model set has {count} hypotheses; maximum is {maximum}")]
    TooManyHypotheses { count: usize, maximum: usize },
    #[error("candidate set is empty")]
    NoExperiments,
    #[error("candidate set has {count} experiments; maximum is {maximum}")]
    TooManyExperiments { count: usize, maximum: usize },
    #[error("at least one prior scenario is required")]
    NoPriorScenarios,
    #[error("prior scenario count {count} exceeds maximum {maximum}")]
    TooManyPriorScenarios { count: usize, maximum: usize },
    #[error("hypothesis ID at index {0} is empty")]
    EmptyHypothesisId(usize),
    #[error("duplicate hypothesis ID: {0}")]
    DuplicateHypothesisId(String),
    #[error("OTHER_UNRESOLVED must not be assigned a numeric model likelihood")]
    UnresolvedHypothesisMustRemainOpen,
    #[error("hypothesis {model_id} has an invalid source revision SHA")]
    InvalidSourceRevision { model_id: String },
    #[error("hypothesis {model_id} has no model specification reference")]
    MissingSpecificationReference { model_id: String },
    #[error("prior scenario ID at index {0} is empty")]
    EmptyPriorScenarioId(usize),
    #[error("duplicate prior scenario ID: {0}")]
    DuplicatePriorScenarioId(String),
    #[error("prior scenario {scenario_id} has {actual} probabilities; expected {expected}")]
    PriorProbabilityCount {
        scenario_id: String,
        expected: usize,
        actual: usize,
    },
    #[error("prior scenario {scenario_id} has invalid probability at index {index}")]
    InvalidPriorProbability { scenario_id: String, index: usize },
    #[error("prior scenario {0} has zero total probability")]
    ZeroPriorProbability(String),
    #[error("prior scenario {0} must have a finite positive scenario weight")]
    InvalidPriorScenarioWeight(String),
    #[error("closed model universe requires a non-empty completeness reference")]
    MissingCompletenessReference,
    #[error("experiment ID at index {0} is empty")]
    EmptyExperimentId(usize),
    #[error("duplicate experiment ID: {0}")]
    DuplicateExperimentId(String),
    #[error("experiment {experiment_id} has an empty {field}")]
    EmptyExperimentField {
        experiment_id: String,
        field: &'static str,
    },
    #[error("experiment {0} must have a finite positive cost")]
    InvalidExperimentCost(String),
    #[error("experiment {0} availability probability must be finite and in [0, 1]")]
    InvalidAvailabilityProbability(String),
    #[error("experiment {0} has an empty mutual-exclusion group")]
    EmptyExclusiveGroup(String),
    #[error("experiment {experiment_id} has {actual} prediction rows; expected {expected}")]
    PredictionCount {
        experiment_id: String,
        expected: usize,
        actual: usize,
    },
    #[error("experiment {experiment_id} has duplicate prediction for model {model_id}")]
    DuplicateModelPrediction {
        experiment_id: String,
        model_id: String,
    },
    #[error("experiment {experiment_id} is missing prediction row for model {model_id}")]
    MissingModelPrediction {
        experiment_id: String,
        model_id: String,
    },
    #[error("experiment {experiment_id}, model {model_id} has non-finite predictive mean")]
    NonFinitePredictiveMean {
        experiment_id: String,
        model_id: String,
    },
    #[error("experiment {experiment_id}, model {model_id} has invalid predictive sigma")]
    InvalidPredictiveSigma {
        experiment_id: String,
        model_id: String,
    },
    #[error("quantitative prediction for {experiment_id}/{model_id} is missing its mean")]
    QuantitativePredictionMissingMean {
        experiment_id: String,
        model_id: String,
    },
    #[error(
        "quantitative prediction for {experiment_id}/{model_id} is missing its predictive uncertainty"
    )]
    QuantitativePredictionMissingUncertainty {
        experiment_id: String,
        model_id: String,
    },
    #[error("non-quantitative prediction for {experiment_id}/{model_id} carries numeric fields")]
    NonQuantitativePredictionHasNumbers {
        experiment_id: String,
        model_id: String,
    },
    #[error("experiment {experiment_id} has invalid measurement sigma")]
    InvalidMeasurementSigma { experiment_id: String },
    #[error("quadrature intervals must be even and between 64 and {maximum}")]
    InvalidQuadratureIntervals { maximum: usize },
    #[error("quadrature tolerance must be finite and positive")]
    InvalidQuadratureTolerance,
    #[error("Gaussian KL inputs must be finite with positive predictive and measurement sigmas")]
    InvalidGaussianKlInputs,
    #[error("experiment {experiment_id} has {actual} detection probabilities; expected {expected}")]
    DetectionProbabilityCount {
        experiment_id: String,
        expected: usize,
        actual: usize,
    },
    #[error("experiment {experiment_id} has duplicate detection probability for model {model_id}")]
    DuplicateDetectionProbability {
        experiment_id: String,
        model_id: String,
    },
    #[error("experiment {experiment_id} is missing detection probability for model {model_id}")]
    MissingDetectionProbability {
        experiment_id: String,
        model_id: String,
    },
    #[error(
        "experiment {experiment_id}, model {model_id} has detection probability outside [0, 1]"
    )]
    InvalidDetectionProbability {
        experiment_id: String,
        model_id: String,
    },
    #[error(
        "experiment {experiment_id}, model {model_id} is missing a detection derivation reference"
    )]
    MissingDetectionDerivation {
        experiment_id: String,
        model_id: String,
    },
    #[error(
        "experiment {experiment_id} contains a detection probability for unknown model {model_id}"
    )]
    UnexpectedDetectionProbability {
        experiment_id: String,
        model_id: String,
    },
    #[error("detection-channel inputs are dimensionally inconsistent or numerically invalid")]
    InvalidDetectionChannel,
    #[error("detection-channel mutual information exceeded prior entropy for experiment {0}")]
    DetectionInformationExceedsPriorEntropy(String),
    #[error("numerical overflow while evaluating experiment {0}")]
    NumericalOverflow(String),
}

impl ModelDiscriminationConfig {
    fn validate(
        &self,
        hypotheses: &[DiscriminationHypothesis],
    ) -> Result<(), ModelDiscriminationError> {
        if hypotheses.len() < 2 {
            return Err(ModelDiscriminationError::TooFewHypotheses);
        }
        if hypotheses.len() > MAX_DISCRIMINATION_HYPOTHESES {
            return Err(ModelDiscriminationError::TooManyHypotheses {
                count: hypotheses.len(),
                maximum: MAX_DISCRIMINATION_HYPOTHESES,
            });
        }
        if self.prior_scenarios.is_empty() {
            return Err(ModelDiscriminationError::NoPriorScenarios);
        }
        if self.prior_scenarios.len() > MAX_MODEL_PRIOR_SCENARIOS {
            return Err(ModelDiscriminationError::TooManyPriorScenarios {
                count: self.prior_scenarios.len(),
                maximum: MAX_MODEL_PRIOR_SCENARIOS,
            });
        }
        if self.quadrature_intervals < 64
            || self.quadrature_intervals > MAX_EIG_QUADRATURE_INTERVALS
            || self.quadrature_intervals % 2 != 0
        {
            return Err(ModelDiscriminationError::InvalidQuadratureIntervals {
                maximum: MAX_EIG_QUADRATURE_INTERVALS,
            });
        }
        if !self.quadrature_tolerance_nats.is_finite() || self.quadrature_tolerance_nats <= 0.0 {
            return Err(ModelDiscriminationError::InvalidQuadratureTolerance);
        }
        if let ModelUniverseCoverage::DeclaredClosed {
            completeness_reference,
        } = &self.model_universe
            && completeness_reference.trim().is_empty()
        {
            return Err(ModelDiscriminationError::MissingCompletenessReference);
        }

        let mut model_ids = BTreeSet::new();
        for (index, hypothesis) in hypotheses.iter().enumerate() {
            if hypothesis.model_id.trim().is_empty() {
                return Err(ModelDiscriminationError::EmptyHypothesisId(index));
            }
            if hypothesis.model_id == OTHER_UNRESOLVED_MODEL_ID {
                return Err(ModelDiscriminationError::UnresolvedHypothesisMustRemainOpen);
            }
            if !model_ids.insert(hypothesis.model_id.as_str()) {
                return Err(ModelDiscriminationError::DuplicateHypothesisId(
                    hypothesis.model_id.clone(),
                ));
            }
            if !is_lower_hex_git_sha(&hypothesis.source_revision_sha) {
                return Err(ModelDiscriminationError::InvalidSourceRevision {
                    model_id: hypothesis.model_id.clone(),
                });
            }
            if hypothesis.specification_reference.trim().is_empty() {
                return Err(ModelDiscriminationError::MissingSpecificationReference {
                    model_id: hypothesis.model_id.clone(),
                });
            }
        }

        let mut scenario_ids = BTreeSet::new();
        for (index, scenario) in self.prior_scenarios.iter().enumerate() {
            if scenario.scenario_id.trim().is_empty() {
                return Err(ModelDiscriminationError::EmptyPriorScenarioId(index));
            }
            if !scenario_ids.insert(scenario.scenario_id.as_str()) {
                return Err(ModelDiscriminationError::DuplicatePriorScenarioId(
                    scenario.scenario_id.clone(),
                ));
            }
            if !scenario.scenario_weight.is_finite() || scenario.scenario_weight <= 0.0 {
                return Err(ModelDiscriminationError::InvalidPriorScenarioWeight(
                    scenario.scenario_id.clone(),
                ));
            }
            if scenario.probabilities.len() != hypotheses.len() {
                return Err(ModelDiscriminationError::PriorProbabilityCount {
                    scenario_id: scenario.scenario_id.clone(),
                    expected: hypotheses.len(),
                    actual: scenario.probabilities.len(),
                });
            }
            let mut maximum_probability = 0.0_f64;
            for (probability_index, probability) in
                scenario.probabilities.iter().copied().enumerate()
            {
                if !probability.is_finite() || probability < 0.0 {
                    return Err(ModelDiscriminationError::InvalidPriorProbability {
                        scenario_id: scenario.scenario_id.clone(),
                        index: probability_index,
                    });
                }
                maximum_probability = maximum_probability.max(probability);
            }
            if maximum_probability <= 0.0 {
                return Err(ModelDiscriminationError::ZeroPriorProbability(
                    scenario.scenario_id.clone(),
                ));
            }
        }
        Ok(())
    }
}

/// Rank scalar candidate experiments by expected information gain about the
/// discrete model identity.
///
/// The prior-scenario objective is robust: experiments are sorted first by
/// worst-case model-identity EIG per cost, then by weighted-mean EIG per cost.
/// An open-world run explicitly excludes OTHER_UNRESOLVED from numeric
/// likelihoods. It therefore ranks discrimination among the supplied named
/// hypotheses; it cannot prove that the selected hypothesis is true.
/// Analytic directed KL divergence between two scalar Gaussian predictive
/// distributions after adding the same independent calibrated measurement
/// noise to both.
///
/// The predictive distributions are
///
/// `p = Normal(left_mean, left_predictive_sigma^2 + measurement_sigma^2)`
///
/// `q = Normal(right_mean, right_predictive_sigma^2 + measurement_sigma^2)`.
///
/// Returns `D_KL(p || q)` in nats. This is an expected log-likelihood-ratio
/// objective, **not** model-identity EIG, posterior entropy reduction or
/// model-recovery accuracy. This closed-form reference requires the predictive
/// and measurement noise sources to be independent and Gaussian.
pub fn gaussian_predictive_kl_nats(
    left_mean: f64,
    left_predictive_sigma: f64,
    right_mean: f64,
    right_predictive_sigma: f64,
    measurement_sigma: f64,
) -> Result<f64, ModelDiscriminationError> {
    if !left_mean.is_finite()
        || !right_mean.is_finite()
        || !left_predictive_sigma.is_finite()
        || !right_predictive_sigma.is_finite()
        || !measurement_sigma.is_finite()
        || left_predictive_sigma <= 0.0
        || right_predictive_sigma <= 0.0
        || measurement_sigma <= 0.0
    {
        return Err(ModelDiscriminationError::InvalidGaussianKlInputs);
    }

    let left_sigma = left_predictive_sigma.hypot(measurement_sigma);
    let right_sigma = right_predictive_sigma.hypot(measurement_sigma);
    let mean_difference = left_mean - right_mean;
    let sigma_ratio = left_sigma / right_sigma;
    let standardized_difference = mean_difference / right_sigma;
    let ratio_squared = sigma_ratio * sigma_ratio;
    let difference_squared = standardized_difference * standardized_difference;
    let value =
        right_sigma.ln() - left_sigma.ln() + 0.5 * (ratio_squared + difference_squared - 1.0);

    if !left_sigma.is_finite()
        || !right_sigma.is_finite()
        || !mean_difference.is_finite()
        || !sigma_ratio.is_finite()
        || !standardized_difference.is_finite()
        || !value.is_finite()
    {
        return Err(ModelDiscriminationError::InvalidGaussianKlInputs);
    }

    // The analytic value is non-negative; clamp only tiny floating-point
    // cancellation around identical distributions.
    Ok(value.max(0.0))
}

pub fn rank_model_discriminating_experiments(
    hypotheses: &[DiscriminationHypothesis],
    experiments: &[CandidateExperiment],
    config: &ModelDiscriminationConfig,
) -> Result<ModelDiscriminationReport, ModelDiscriminationError> {
    config.validate(hypotheses)?;
    if experiments.is_empty() {
        return Err(ModelDiscriminationError::NoExperiments);
    }
    if experiments.len() > MAX_DISCRIMINATION_EXPERIMENTS {
        return Err(ModelDiscriminationError::TooManyExperiments {
            count: experiments.len(),
            maximum: MAX_DISCRIMINATION_EXPERIMENTS,
        });
    }

    let mut experiment_ids = BTreeSet::new();
    let mut scores = Vec::with_capacity(experiments.len());
    for (index, experiment) in experiments.iter().enumerate() {
        validate_experiment(experiment, hypotheses, index, &mut experiment_ids)?;
        scores.push(score_experiment(experiment, config)?);
    }

    scores.sort_by(|left, right| {
        let left_scored = matches!(left.status, ExperimentDiscriminationStatus::Scored);
        let right_scored = matches!(right.status, ExperimentDiscriminationStatus::Scored);
        right_scored
            .cmp(&left_scored)
            .then_with(|| {
                right
                    .worst_case_eig_per_cost
                    .unwrap_or(f64::NEG_INFINITY)
                    .total_cmp(&left.worst_case_eig_per_cost.unwrap_or(f64::NEG_INFINITY))
            })
            .then_with(|| {
                right
                    .weighted_mean_eig_per_cost
                    .unwrap_or(f64::NEG_INFINITY)
                    .total_cmp(&left.weighted_mean_eig_per_cost.unwrap_or(f64::NEG_INFINITY))
            })
            .then_with(|| left.experiment_id.cmp(&right.experiment_id))
    });

    Ok(ModelDiscriminationReport {
        model_ids: hypotheses
            .iter()
            .map(|hypothesis| hypothesis.model_id.clone())
            .collect(),
        model_universe: config.model_universe.clone(),
        prior_scenario_ids: config
            .prior_scenarios
            .iter()
            .map(|scenario| scenario.scenario_id.clone())
            .collect(),
        experiments: scores,
    })
}

/// Rank experiments when the detection/non-detection outcome can itself
/// depend on the physical model.
///
/// The total information is decomposed as:
/// `I(M;D,Y) = I(M;D) + P(D=1) I(M;Y | D=1)`.
///
/// The measurement-value EIG reuses the existing Gaussian quadrature kernel
/// with the prior reweighted by each model's detection probability. The
/// wrapped `availability_probability` is treated as a separate, model-independent execution probability. A failed execution is assumed
/// uninformative; model-dependent detection only applies after execution.
///
/// This is not a censored-value likelihood or a model for different failure
/// causes. Those outcomes must be represented as explicit likelihood
/// categories before they can contribute information.
pub fn rank_detection_aware_model_discriminating_experiments(
    hypotheses: &[DiscriminationHypothesis],
    experiments: &[DetectionAwareCandidateExperiment],
    config: &ModelDiscriminationConfig,
) -> Result<DetectionAwareModelDiscriminationReport, ModelDiscriminationError> {
    config.validate(hypotheses)?;
    if experiments.is_empty() {
        return Err(ModelDiscriminationError::NoExperiments);
    }
    if experiments.len() > MAX_DISCRIMINATION_EXPERIMENTS {
        return Err(ModelDiscriminationError::TooManyExperiments {
            count: experiments.len(),
            maximum: MAX_DISCRIMINATION_EXPERIMENTS,
        });
    }

    let mut experiment_ids = BTreeSet::new();
    let mut scores = Vec::with_capacity(experiments.len());
    for (index, candidate) in experiments.iter().enumerate() {
        let experiment = &candidate.experiment;
        validate_experiment(experiment, hypotheses, index, &mut experiment_ids)?;

        // Reuse all of the existing measurement comparability gates. The
        // wrapped score's old uninformative-missingness EIG is not used here.
        let base_score = score_experiment(experiment, config)?;
        let open_world_unresolved_excluded =
            matches!(config.model_universe, ModelUniverseCoverage::OpenWorld);
        if let ExperimentDiscriminationStatus::NotComparable { blockers } = base_score.status {
            scores.push(DetectionAwareExperimentScore {
                experiment_id: experiment.experiment_id.clone(),
                observable_id: experiment.observable_id.clone(),
                cost: experiment.cost,
                execution_probability: experiment.availability_probability,
                status: ExperimentDiscriminationStatus::NotComparable { blockers },
                prior_scenario_scores: Vec::new(),
                worst_case_total_eig_nats: None,
                weighted_mean_total_eig_nats: None,
                worst_case_total_eig_per_cost: None,
                weighted_mean_total_eig_per_cost: None,
                open_world_unresolved_excluded,
            });
            continue;
        }

        let detection_by_model = validate_detection_probabilities(candidate, hypotheses)?;
        let measurement_sigma = experiment.measurement_sigma.expect("comparability gate");
        // Prediction rows are keyed by model ID, not by caller-supplied vector
        // position. Canonicalize to the hypothesis order before pairing them
        // with the prior and detection probabilities.
        let prediction_by_model: BTreeMap<&str, (f64, f64)> = experiment
            .predictions
            .iter()
            .map(|prediction| {
                (
                    prediction.model_id.as_str(),
                    (
                        prediction.predictive_mean.expect("comparability gate"),
                        prediction.predictive_sigma.expect("comparability gate"),
                    ),
                )
            })
            .collect();
        let normal_predictions: Vec<(f64, f64)> = hypotheses
            .iter()
            .map(|hypothesis| prediction_by_model[hypothesis.model_id.as_str()])
            .collect();
        let detection_probabilities: Vec<f64> = hypotheses
            .iter()
            .map(|hypothesis| detection_by_model[&hypothesis.model_id])
            .collect();

        let max_scenario_weight = config
            .prior_scenarios
            .iter()
            .map(|scenario| scenario.scenario_weight)
            .fold(0.0_f64, f64::max);
        let normalized_weights: Vec<f64> = config
            .prior_scenarios
            .iter()
            .map(|scenario| scenario.scenario_weight / max_scenario_weight)
            .collect();
        let total_scenario_weight: f64 = normalized_weights.iter().sum();

        let mut prior_scenario_scores = Vec::with_capacity(config.prior_scenarios.len());
        let mut worst_case = f64::INFINITY;
        let mut weighted_sum = 0.0;
        let mut maximum_error = 0.0_f64;

        for (scenario, scenario_weight) in config
            .prior_scenarios
            .iter()
            .zip(normalized_weights.iter().copied())
        {
            let prior = normalize_probabilities(&scenario.probabilities);
            let prior_entropy = discrete_entropy(&prior);
            let expected_detection_probability: f64 = prior
                .iter()
                .zip(&detection_probabilities)
                .map(|(probability, detection)| probability * detection)
                .sum::<f64>()
                .clamp(0.0, 1.0);

            let detection_indicator_eig = binary_channel_mutual_information(
                &prior,
                &detection_probabilities,
                expected_detection_probability,
            )?;
            let detected_prior = if expected_detection_probability > 0.0 {
                prior
                    .iter()
                    .zip(&detection_probabilities)
                    .map(|(probability, detection)| {
                        probability * detection / expected_detection_probability
                    })
                    .collect::<Vec<_>>()
            } else {
                vec![0.0; prior.len()]
            };
            let (coarse_detected_eig, refined_detected_eig) = if expected_detection_probability
                > 0.0
            {
                let coarse = gaussian_model_identity_eig(
                    &normal_predictions,
                    measurement_sigma,
                    &detected_prior,
                    config.quadrature_intervals,
                )
                .ok_or_else(|| {
                    ModelDiscriminationError::NumericalOverflow(experiment.experiment_id.clone())
                })?;
                let refined = gaussian_model_identity_eig(
                    &normal_predictions,
                    measurement_sigma,
                    &detected_prior,
                    config.quadrature_intervals * 2,
                )
                .ok_or_else(|| {
                    ModelDiscriminationError::NumericalOverflow(experiment.experiment_id.clone())
                })?;
                (coarse, refined)
            } else {
                (0.0, 0.0)
            };

            let total_given_execution =
                detection_indicator_eig + expected_detection_probability * refined_detected_eig;
            let execution_adjusted = experiment.availability_probability * total_given_execution;
            // Gate on the conditional detected-value integral itself. A
            // small detection/execution probability must not hide an
            // under-resolved conditional likelihood calculation.
            let quadrature_error = (refined_detected_eig - coarse_detected_eig).abs();
            let operational_quadrature_error_bound_nats = experiment.availability_probability
                * expected_detection_probability
                * quadrature_error;
            let converged = quadrature_error <= config.quadrature_tolerance_nats;
            if total_given_execution > prior_entropy + 1.0e-8
                || execution_adjusted > prior_entropy + 1.0e-8
                || !total_given_execution.is_finite()
                || !execution_adjusted.is_finite()
            {
                return Err(
                    ModelDiscriminationError::DetectionInformationExceedsPriorEntropy(
                        experiment.experiment_id.clone(),
                    ),
                );
            }
            let execution_adjusted = execution_adjusted.min(prior_entropy);
            let expected_posterior_entropy = (prior_entropy - execution_adjusted).max(0.0);
            maximum_error = maximum_error.max(quadrature_error);
            worst_case = worst_case.min(execution_adjusted);
            weighted_sum += scenario_weight * execution_adjusted;
            prior_scenario_scores.push(DetectionAwarePriorScenarioEIG {
                scenario_id: scenario.scenario_id.clone(),
                prior_entropy_nats: prior_entropy,
                expected_detection_probability,
                detection_indicator_eig_nats: detection_indicator_eig,
                conditional_detected_value_eig_nats: (expected_detection_probability > 0.0)
                    .then_some(refined_detected_eig),
                total_eig_given_execution_nats: total_given_execution,
                availability_adjusted_total_eig_nats: execution_adjusted,
                expected_posterior_entropy_nats: expected_posterior_entropy,
                quadrature_error_nats: quadrature_error,
                operational_quadrature_error_bound_nats,
                converged,
            });
        }

        if prior_scenario_scores.iter().any(|score| !score.converged) {
            scores.push(DetectionAwareExperimentScore {
                experiment_id: experiment.experiment_id.clone(),
                observable_id: experiment.observable_id.clone(),
                cost: experiment.cost,
                execution_probability: experiment.availability_probability,
                status: ExperimentDiscriminationStatus::QuadratureNotConverged {
                    maximum_error_nats: maximum_error,
                    tolerance_nats: config.quadrature_tolerance_nats,
                },
                prior_scenario_scores,
                worst_case_total_eig_nats: None,
                weighted_mean_total_eig_nats: None,
                worst_case_total_eig_per_cost: None,
                weighted_mean_total_eig_per_cost: None,
                open_world_unresolved_excluded,
            });
            continue;
        }

        let worst_case = worst_case.max(0.0);
        let weighted_mean = (weighted_sum / total_scenario_weight).max(0.0);
        scores.push(DetectionAwareExperimentScore {
            experiment_id: experiment.experiment_id.clone(),
            observable_id: experiment.observable_id.clone(),
            cost: experiment.cost,
            execution_probability: experiment.availability_probability,
            status: ExperimentDiscriminationStatus::Scored,
            prior_scenario_scores,
            worst_case_total_eig_nats: Some(worst_case),
            weighted_mean_total_eig_nats: Some(weighted_mean),
            worst_case_total_eig_per_cost: Some(worst_case / experiment.cost),
            weighted_mean_total_eig_per_cost: Some(weighted_mean / experiment.cost),
            open_world_unresolved_excluded,
        });
    }

    scores.sort_by(|left, right| {
        let left_scored = matches!(left.status, ExperimentDiscriminationStatus::Scored);
        let right_scored = matches!(right.status, ExperimentDiscriminationStatus::Scored);
        right_scored
            .cmp(&left_scored)
            .then_with(|| {
                right
                    .worst_case_total_eig_per_cost
                    .unwrap_or(f64::NEG_INFINITY)
                    .total_cmp(
                        &left
                            .worst_case_total_eig_per_cost
                            .unwrap_or(f64::NEG_INFINITY),
                    )
            })
            .then_with(|| left.experiment_id.cmp(&right.experiment_id))
    });

    Ok(DetectionAwareModelDiscriminationReport {
        model_ids: hypotheses
            .iter()
            .map(|hypothesis| hypothesis.model_id.clone())
            .collect(),
        model_universe: config.model_universe.clone(),
        prior_scenario_ids: config
            .prior_scenarios
            .iter()
            .map(|scenario| scenario.scenario_id.clone())
            .collect(),
        experiments: scores,
    })
}

fn validate_detection_probabilities(
    candidate: &DetectionAwareCandidateExperiment,
    hypotheses: &[DiscriminationHypothesis],
) -> Result<BTreeMap<String, f64>, ModelDiscriminationError> {
    let experiment_id = &candidate.experiment.experiment_id;
    if candidate.detection_probabilities.len() != hypotheses.len() {
        return Err(ModelDiscriminationError::DetectionProbabilityCount {
            experiment_id: experiment_id.clone(),
            expected: hypotheses.len(),
            actual: candidate.detection_probabilities.len(),
        });
    }
    let expected: BTreeSet<&str> = hypotheses
        .iter()
        .map(|hypothesis| hypothesis.model_id.as_str())
        .collect();
    let mut actual = BTreeMap::new();
    for detection in &candidate.detection_probabilities {
        if !expected.contains(detection.model_id.as_str()) {
            return Err(ModelDiscriminationError::UnexpectedDetectionProbability {
                experiment_id: experiment_id.clone(),
                model_id: detection.model_id.clone(),
            });
        }
        if actual.contains_key(&detection.model_id) {
            return Err(ModelDiscriminationError::DuplicateDetectionProbability {
                experiment_id: experiment_id.clone(),
                model_id: detection.model_id.clone(),
            });
        }
        if !detection.probability_detected.is_finite()
            || !(0.0..=1.0).contains(&detection.probability_detected)
        {
            return Err(ModelDiscriminationError::InvalidDetectionProbability {
                experiment_id: experiment_id.clone(),
                model_id: detection.model_id.clone(),
            });
        }
        if detection.derivation_reference.trim().is_empty() {
            return Err(ModelDiscriminationError::MissingDetectionDerivation {
                experiment_id: experiment_id.clone(),
                model_id: detection.model_id.clone(),
            });
        }
        actual.insert(detection.model_id.clone(), detection.probability_detected);
    }
    for hypothesis in hypotheses {
        if !actual.contains_key(&hypothesis.model_id) {
            return Err(ModelDiscriminationError::MissingDetectionProbability {
                experiment_id: experiment_id.clone(),
                model_id: hypothesis.model_id.clone(),
            });
        }
    }
    Ok(actual)
}

fn binary_channel_mutual_information(
    prior: &[f64],
    detection_probabilities: &[f64],
    marginal_detection_probability: f64,
) -> Result<f64, ModelDiscriminationError> {
    if prior.len() != detection_probabilities.len()
        || !marginal_detection_probability.is_finite()
        || !(0.0..=1.0).contains(&marginal_detection_probability)
    {
        return Err(ModelDiscriminationError::InvalidDetectionChannel);
    }
    let conditional_entropy: f64 = prior
        .iter()
        .zip(detection_probabilities)
        .map(|(weight, probability)| weight * binary_entropy(*probability))
        .sum();
    let information = binary_entropy(marginal_detection_probability) - conditional_entropy;
    if !information.is_finite() || information < -1.0e-9 {
        return Err(ModelDiscriminationError::InvalidDetectionChannel);
    }
    Ok(information.max(0.0))
}

fn binary_entropy(probability: f64) -> f64 {
    let detected = if probability > 0.0 {
        -probability * probability.ln()
    } else {
        0.0
    };
    let not_detected = if probability < 1.0 {
        -(1.0 - probability) * (1.0 - probability).ln()
    } else {
        0.0
    };
    detected + not_detected
}

fn validate_experiment(
    experiment: &CandidateExperiment,
    hypotheses: &[DiscriminationHypothesis],
    index: usize,
    experiment_ids: &mut BTreeSet<String>,
) -> Result<(), ModelDiscriminationError> {
    if experiment.experiment_id.trim().is_empty() {
        return Err(ModelDiscriminationError::EmptyExperimentId(index));
    }
    if !experiment_ids.insert(experiment.experiment_id.clone()) {
        return Err(ModelDiscriminationError::DuplicateExperimentId(
            experiment.experiment_id.clone(),
        ));
    }
    for (field_name, value) in [
        ("observable_id", experiment.observable_id.as_str()),
        ("domain_id", experiment.domain_id.as_str()),
        ("units", experiment.units.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(ModelDiscriminationError::EmptyExperimentField {
                experiment_id: experiment.experiment_id.clone(),
                field: field_name,
            });
        }
    }
    if !experiment.cost.is_finite() || experiment.cost <= 0.0 {
        return Err(ModelDiscriminationError::InvalidExperimentCost(
            experiment.experiment_id.clone(),
        ));
    }
    if !experiment.availability_probability.is_finite()
        || !(0.0..=1.0).contains(&experiment.availability_probability)
    {
        return Err(ModelDiscriminationError::InvalidAvailabilityProbability(
            experiment.experiment_id.clone(),
        ));
    }
    if experiment
        .exclusive_group
        .as_ref()
        .is_some_and(|group| group.trim().is_empty())
    {
        return Err(ModelDiscriminationError::EmptyExclusiveGroup(
            experiment.experiment_id.clone(),
        ));
    }
    if experiment.predictions.len() != hypotheses.len() {
        return Err(ModelDiscriminationError::PredictionCount {
            experiment_id: experiment.experiment_id.clone(),
            expected: hypotheses.len(),
            actual: experiment.predictions.len(),
        });
    }

    let expected: BTreeSet<&str> = hypotheses
        .iter()
        .map(|hypothesis| hypothesis.model_id.as_str())
        .collect();
    let mut actual = BTreeSet::new();
    for prediction in &experiment.predictions {
        if !actual.insert(prediction.model_id.as_str()) {
            return Err(ModelDiscriminationError::DuplicateModelPrediction {
                experiment_id: experiment.experiment_id.clone(),
                model_id: prediction.model_id.clone(),
            });
        }
        match prediction.disposition {
            PredictionDisposition::Quantitative => {
                let mean = prediction.predictive_mean.ok_or_else(|| {
                    ModelDiscriminationError::QuantitativePredictionMissingMean {
                        experiment_id: experiment.experiment_id.clone(),
                        model_id: prediction.model_id.clone(),
                    }
                })?;
                let sigma = prediction.predictive_sigma.ok_or_else(|| {
                    ModelDiscriminationError::QuantitativePredictionMissingUncertainty {
                        experiment_id: experiment.experiment_id.clone(),
                        model_id: prediction.model_id.clone(),
                    }
                })?;
                if !mean.is_finite() {
                    return Err(ModelDiscriminationError::NonFinitePredictiveMean {
                        experiment_id: experiment.experiment_id.clone(),
                        model_id: prediction.model_id.clone(),
                    });
                }
                if !sigma.is_finite() || sigma < 0.0 {
                    return Err(ModelDiscriminationError::InvalidPredictiveSigma {
                        experiment_id: experiment.experiment_id.clone(),
                        model_id: prediction.model_id.clone(),
                    });
                }
            }
            _ => {
                if prediction.predictive_mean.is_some() || prediction.predictive_sigma.is_some() {
                    return Err(
                        ModelDiscriminationError::NonQuantitativePredictionHasNumbers {
                            experiment_id: experiment.experiment_id.clone(),
                            model_id: prediction.model_id.clone(),
                        },
                    );
                }
            }
        }
    }
    for model_id in expected {
        if !actual.contains(model_id) {
            return Err(ModelDiscriminationError::MissingModelPrediction {
                experiment_id: experiment.experiment_id.clone(),
                model_id: model_id.to_string(),
            });
        }
    }

    if let Some(sigma) = experiment.measurement_sigma
        && (!sigma.is_finite() || sigma <= 0.0)
    {
        return Err(ModelDiscriminationError::InvalidMeasurementSigma {
            experiment_id: experiment.experiment_id.clone(),
        });
    }
    Ok(())
}

fn score_experiment(
    experiment: &CandidateExperiment,
    config: &ModelDiscriminationConfig,
) -> Result<ExperimentDiscriminationScore, ModelDiscriminationError> {
    let mut blockers = Vec::new();
    if experiment.measurement_noise_model != MeasurementNoiseModel::Gaussian {
        blockers.push(ExperimentBlocker::NonGaussianMeasurementNoise);
    }
    if experiment.noise_calibration == NoiseCalibrationStatus::Uncalibrated {
        blockers.push(ExperimentBlocker::MeasurementNoiseUncalibrated);
    }
    if experiment.measurement_sigma.is_none() {
        blockers.push(ExperimentBlocker::MeasurementSigmaMissing);
    }
    if experiment
        .calibration_reference
        .as_ref()
        .is_none_or(|reference| reference.trim().is_empty())
    {
        blockers.push(ExperimentBlocker::CalibrationReferenceMissing);
    }
    for prediction in &experiment.predictions {
        if prediction.disposition != PredictionDisposition::Quantitative {
            blockers.push(ExperimentBlocker::PredictionNotQuantitative {
                model_id: prediction.model_id.clone(),
                disposition: prediction.disposition,
            });
            continue;
        }
        if prediction.predictive_mean.is_none() {
            blockers.push(ExperimentBlocker::PredictionMissingMean {
                model_id: prediction.model_id.clone(),
            });
        }
        if prediction.predictive_sigma.is_none() {
            blockers.push(ExperimentBlocker::PredictionMissingUncertainty {
                model_id: prediction.model_id.clone(),
            });
        }
        if prediction
            .derivation_reference
            .as_ref()
            .is_none_or(|reference| reference.trim().is_empty())
        {
            blockers.push(ExperimentBlocker::PredictionMissingDerivation {
                model_id: prediction.model_id.clone(),
            });
        }
    }

    let open_world_unresolved_excluded =
        matches!(config.model_universe, ModelUniverseCoverage::OpenWorld);
    if !blockers.is_empty() {
        return Ok(ExperimentDiscriminationScore {
            experiment_id: experiment.experiment_id.clone(),
            observable_id: experiment.observable_id.clone(),
            cost: experiment.cost,
            availability_probability: experiment.availability_probability,
            status: ExperimentDiscriminationStatus::NotComparable { blockers },
            prior_scenario_scores: Vec::new(),
            worst_case_model_identity_eig_nats: None,
            weighted_mean_model_identity_eig_nats: None,
            availability_adjusted_worst_case_model_identity_eig_nats: None,
            availability_adjusted_weighted_mean_model_identity_eig_nats: None,
            worst_case_eig_per_cost: None,
            weighted_mean_eig_per_cost: None,
            open_world_unresolved_excluded,
        });
    }

    let measurement_sigma = experiment.measurement_sigma.expect("validated above");
    let normal_predictions: Vec<(f64, f64)> = experiment
        .predictions
        .iter()
        .map(|prediction| {
            (
                prediction.predictive_mean.expect("validated above"),
                prediction.predictive_sigma.expect("validated above"),
            )
        })
        .collect();

    let mut prior_scenario_scores = Vec::with_capacity(config.prior_scenarios.len());
    let mut worst_case = f64::INFINITY;
    let mut weighted_sum = 0.0;
    let mut availability_adjusted_worst_case = f64::INFINITY;
    let mut availability_adjusted_weighted_sum = 0.0;
    let mut maximum_error = 0.0_f64;
    let max_scenario_weight = config
        .prior_scenarios
        .iter()
        .map(|scenario| scenario.scenario_weight)
        .fold(0.0_f64, f64::max);
    let normalized_scenario_weights: Vec<f64> = config
        .prior_scenarios
        .iter()
        .map(|scenario| scenario.scenario_weight / max_scenario_weight)
        .collect();
    let total_scenario_weight: f64 = normalized_scenario_weights.iter().sum();

    for (scenario, normalized_scenario_weight) in config
        .prior_scenarios
        .iter()
        .zip(normalized_scenario_weights.iter().copied())
    {
        let probabilities = normalize_probabilities(&scenario.probabilities);
        let coarse = gaussian_model_identity_eig(
            &normal_predictions,
            measurement_sigma,
            &probabilities,
            config.quadrature_intervals,
        )
        .ok_or_else(|| {
            ModelDiscriminationError::NumericalOverflow(experiment.experiment_id.clone())
        })?;
        let refined = gaussian_model_identity_eig(
            &normal_predictions,
            measurement_sigma,
            &probabilities,
            config.quadrature_intervals * 2,
        )
        .ok_or_else(|| {
            ModelDiscriminationError::NumericalOverflow(experiment.experiment_id.clone())
        })?;
        let error = (refined - coarse).abs();
        let prior_entropy = discrete_entropy(&probabilities);
        let converged = error <= config.quadrature_tolerance_nats;
        maximum_error = maximum_error.max(error);
        let availability_adjusted_eig = experiment.availability_probability * refined;
        let expected_posterior_entropy = prior_entropy - availability_adjusted_eig;
        worst_case = worst_case.min(refined);
        weighted_sum += normalized_scenario_weight * refined;
        availability_adjusted_worst_case =
            availability_adjusted_worst_case.min(availability_adjusted_eig);
        availability_adjusted_weighted_sum +=
            normalized_scenario_weight * availability_adjusted_eig;
        prior_scenario_scores.push(PriorScenarioEIG {
            scenario_id: scenario.scenario_id.clone(),
            prior_entropy_nats: prior_entropy,
            coarse_eig_nats: coarse,
            refined_eig_nats: refined,
            availability_adjusted_eig_nats: availability_adjusted_eig,
            expected_posterior_entropy_nats: expected_posterior_entropy,
            quadrature_error_nats: error,
            converged,
        });
    }

    if prior_scenario_scores.iter().any(|score| !score.converged) {
        return Ok(ExperimentDiscriminationScore {
            experiment_id: experiment.experiment_id.clone(),
            observable_id: experiment.observable_id.clone(),
            cost: experiment.cost,
            availability_probability: experiment.availability_probability,
            status: ExperimentDiscriminationStatus::QuadratureNotConverged {
                maximum_error_nats: maximum_error,
                tolerance_nats: config.quadrature_tolerance_nats,
            },
            prior_scenario_scores,
            worst_case_model_identity_eig_nats: None,
            weighted_mean_model_identity_eig_nats: None,
            availability_adjusted_worst_case_model_identity_eig_nats: None,
            availability_adjusted_weighted_mean_model_identity_eig_nats: None,
            worst_case_eig_per_cost: None,
            weighted_mean_eig_per_cost: None,
            open_world_unresolved_excluded,
        });
    }

    let weighted_mean = weighted_sum / total_scenario_weight;
    let worst_case = worst_case.max(0.0);
    let weighted_mean = weighted_mean.max(0.0);
    let availability_adjusted_worst_case = availability_adjusted_worst_case.max(0.0);
    let availability_adjusted_weighted_mean =
        (availability_adjusted_weighted_sum / total_scenario_weight).max(0.0);
    Ok(ExperimentDiscriminationScore {
        experiment_id: experiment.experiment_id.clone(),
        observable_id: experiment.observable_id.clone(),
        cost: experiment.cost,
        availability_probability: experiment.availability_probability,
        status: ExperimentDiscriminationStatus::Scored,
        prior_scenario_scores,
        worst_case_model_identity_eig_nats: Some(worst_case),
        weighted_mean_model_identity_eig_nats: Some(weighted_mean),
        availability_adjusted_worst_case_model_identity_eig_nats: Some(
            availability_adjusted_worst_case,
        ),
        availability_adjusted_weighted_mean_model_identity_eig_nats: Some(
            availability_adjusted_weighted_mean,
        ),
        worst_case_eig_per_cost: Some(availability_adjusted_worst_case / experiment.cost),
        weighted_mean_eig_per_cost: Some(availability_adjusted_weighted_mean / experiment.cost),
        open_world_unresolved_excluded,
    })
}

fn normalize_probabilities(probabilities: &[f64]) -> Vec<f64> {
    let maximum = probabilities.iter().copied().fold(0.0_f64, f64::max);
    let scaled_total: f64 = probabilities.iter().map(|value| value / maximum).sum();
    probabilities
        .iter()
        .map(|value| (value / maximum) / scaled_total)
        .collect()
}

fn discrete_entropy(probabilities: &[f64]) -> f64 {
    -probabilities
        .iter()
        .copied()
        .filter(|probability| *probability > 0.0)
        .map(|probability| probability * probability.ln())
        .sum::<f64>()
}

fn gaussian_model_identity_eig(
    means_and_predictive_sigmas: &[(f64, f64)],
    measurement_sigma: f64,
    probabilities: &[f64],
    intervals: usize,
) -> Option<f64> {
    let mut distributions = Vec::new();
    let mut lower = f64::INFINITY;
    let mut upper = f64::NEG_INFINITY;
    for ((mean, predictive_sigma), probability) in means_and_predictive_sigmas
        .iter()
        .copied()
        .zip(probabilities.iter().copied())
    {
        if probability == 0.0 {
            continue;
        }
        let sigma = predictive_sigma.hypot(measurement_sigma);
        if !sigma.is_finite() || sigma <= 0.0 {
            return None;
        }
        lower = lower.min(mean - 10.0 * sigma);
        upper = upper.max(mean + 10.0 * sigma);
        distributions.push((mean, sigma, probability));
    }
    if distributions.len() <= 1 || !lower.is_finite() || !upper.is_finite() || upper <= lower {
        return Some(0.0);
    }

    let step = (upper - lower) / intervals as f64;
    if !step.is_finite() || step <= 0.0 {
        return None;
    }
    let mut sum = 0.0;
    for index in 0..=intervals {
        let y = lower + index as f64 * step;
        let density = mutual_information_density(y, &distributions)?;
        let weight = if index == 0 || index == intervals {
            1.0
        } else if index % 2 == 0 {
            2.0
        } else {
            4.0
        };
        sum += weight * density;
    }
    let integral = sum * step / 3.0;
    if !integral.is_finite() {
        return None;
    }
    if integral < -1.0e-9 {
        return None;
    }
    Some(integral.max(0.0))
}

fn mutual_information_density(observation: f64, distributions: &[(f64, f64, f64)]) -> Option<f64> {
    let log_likelihoods: Vec<f64> = distributions
        .iter()
        .map(|(mean, sigma, _)| normal_log_density(observation, *mean, *sigma))
        .collect();
    let log_joint: Vec<f64> = distributions
        .iter()
        .zip(log_likelihoods.iter())
        .map(|((_, _, probability), log_likelihood)| probability.ln() + log_likelihood)
        .collect();
    let log_mixture = log_sum_exp(&log_joint)?;
    let mut density = 0.0;
    for ((_, _, probability), (log_joint_value, log_likelihood)) in distributions.iter().zip(
        log_joint
            .iter()
            .copied()
            .zip(log_likelihoods.iter().copied()),
    ) {
        let joint = log_joint_value.exp();
        if joint == 0.0 {
            continue;
        }
        // I(M;Y) = sum_m pi_m p(y|m) log[p(y|m) / p(y)].
        // The mixture density includes the prior; the numerator is conditional.
        density += joint * (log_likelihood - log_mixture);
        if !density.is_finite() || !probability.is_finite() {
            return None;
        }
    }
    Some(density)
}

fn normal_log_density(value: f64, mean: f64, sigma: f64) -> f64 {
    let standardized = (value - mean) / sigma;
    -0.5 * std::f64::consts::TAU.ln() - sigma.ln() - 0.5 * standardized * standardized
}

fn log_sum_exp(values: &[f64]) -> Option<f64> {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return None;
    }
    let sum = values
        .iter()
        .map(|value| (*value - maximum).exp())
        .sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return None;
    }
    Some(maximum + sum.ln())
}

fn is_lower_hex_git_sha(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hypothesis(id: &str, byte: u8) -> DiscriminationHypothesis {
        DiscriminationHypothesis {
            model_id: id.to_string(),
            source_revision_sha: (0..40).map(|_| byte as char).collect(),
            specification_reference: format!("spec://{id}"),
        }
    }

    fn prediction(id: &str, mean: Option<f64>, sigma: Option<f64>) -> GaussianModelPrediction {
        GaussianModelPrediction {
            model_id: id.to_string(),
            disposition: if mean.is_some() {
                PredictionDisposition::Quantitative
            } else {
                PredictionDisposition::NotDerived
            },
            predictive_mean: mean,
            predictive_sigma: sigma,
            derivation_reference: mean.map(|_| format!("derivation://{id}")),
        }
    }

    fn experiment(
        id: &str,
        cost: f64,
        measurement_sigma: Option<f64>,
        predictions: Vec<GaussianModelPrediction>,
    ) -> CandidateExperiment {
        CandidateExperiment {
            experiment_id: id.to_string(),
            observable_id: "observable-y".to_string(),
            domain_id: "synthetic-domain".to_string(),
            units: "dimensionless".to_string(),
            measurement_noise_model: MeasurementNoiseModel::Gaussian,
            measurement_sigma,
            noise_calibration: NoiseCalibrationStatus::SyntheticKnownAnswer,
            calibration_reference: Some("synthetic-noise-fixture-v1".to_string()),
            availability_probability: 1.0,
            cost,
            exclusive_group: None,
            predictions,
        }
    }

    fn config(
        prior_scenarios: Vec<ModelPriorScenario>,
        intervals: usize,
    ) -> ModelDiscriminationConfig {
        ModelDiscriminationConfig {
            model_universe: ModelUniverseCoverage::OpenWorld,
            prior_scenarios,
            quadrature_intervals: intervals,
            quadrature_tolerance_nats: 1.0e-6,
        }
    }

    fn equal_prior() -> ModelPriorScenario {
        ModelPriorScenario {
            scenario_id: "equal".to_string(),
            scenario_weight: 1.0,
            probabilities: vec![0.5, 0.5],
        }
    }

    #[test]
    fn gaussian_predictive_kl_matches_equal_variance_closed_form() {
        let left_mean: f64 = -0.25;
        let right_mean: f64 = 0.75;
        let predictive_sigma: f64 = 0.3;
        let measurement_sigma: f64 = 0.4;
        let total_sigma = predictive_sigma.hypot(measurement_sigma);
        let observed = gaussian_predictive_kl_nats(
            left_mean,
            predictive_sigma,
            right_mean,
            predictive_sigma,
            measurement_sigma,
        )
        .unwrap();
        let expected = (left_mean - right_mean).powi(2) / (2.0 * total_sigma.powi(2));

        assert!((observed - expected).abs() < 1.0e-12);
    }

    #[test]
    fn gaussian_predictive_kl_is_directed_and_asymmetric() {
        let forward = gaussian_predictive_kl_nats(0.0, 0.2, 0.5, 0.8, 0.1).unwrap();
        let reverse = gaussian_predictive_kl_nats(0.5, 0.8, 0.0, 0.2, 0.1).unwrap();

        assert!(forward >= 0.0);
        assert!(reverse >= 0.0);
        assert!((forward - reverse).abs() > 1.0e-3);
    }

    #[test]
    fn gaussian_predictive_kl_is_zero_for_identical_distributions() {
        assert_eq!(
            gaussian_predictive_kl_nats(0.5, 0.2, 0.5, 0.2, 0.1).unwrap(),
            0.0
        );
    }

    #[test]
    fn gaussian_predictive_kl_rejects_invalid_or_overflowing_inputs() {
        assert_eq!(
            gaussian_predictive_kl_nats(0.0, 0.2, 1.0, 0.2, 0.0),
            Err(ModelDiscriminationError::InvalidGaussianKlInputs)
        );
        assert_eq!(
            gaussian_predictive_kl_nats(f64::NAN, 0.2, 1.0, 0.2, 0.1),
            Err(ModelDiscriminationError::InvalidGaussianKlInputs)
        );
        assert_eq!(
            gaussian_predictive_kl_nats(1.0e308, 0.2, -1.0e308, 0.2, 0.1),
            Err(ModelDiscriminationError::InvalidGaussianKlInputs)
        );
    }

    #[test]
    fn identical_predictions_have_zero_model_identity_eig() {
        let hypotheses = vec![hypothesis("model-a", b'a'), hypothesis("model-b", b'b')];
        let experiment = experiment(
            "same-outcome",
            1.0,
            Some(0.1),
            vec![
                prediction("model-a", Some(2.0), Some(0.05)),
                prediction("model-b", Some(2.0), Some(0.05)),
            ],
        );
        let report = rank_model_discriminating_experiments(
            &hypotheses,
            &[experiment],
            &config(vec![equal_prior()], 128),
        )
        .unwrap();
        let score = &report.experiments[0];
        assert_eq!(score.status, ExperimentDiscriminationStatus::Scored);
        assert!(score.worst_case_model_identity_eig_nats.unwrap().abs() < 1.0e-10);
    }

    #[test]
    fn separated_models_approach_prior_entropy_and_gain_increases_with_separation() {
        let hypotheses = vec![hypothesis("model-a", b'a'), hypothesis("model-b", b'b')];
        let make = |separation: f64| {
            experiment(
                "candidate",
                1.0,
                Some(0.1),
                vec![
                    prediction("model-a", Some(-separation / 2.0), Some(0.0)),
                    prediction("model-b", Some(separation / 2.0), Some(0.0)),
                ],
            )
        };
        let close = rank_model_discriminating_experiments(
            &hypotheses,
            &[make(0.2)],
            &config(vec![equal_prior()], 256),
        )
        .unwrap();
        let far = rank_model_discriminating_experiments(
            &hypotheses,
            &[make(4.0)],
            &config(vec![equal_prior()], 256),
        )
        .unwrap();
        assert!(
            far.experiments[0]
                .worst_case_model_identity_eig_nats
                .unwrap()
                > close.experiments[0]
                    .worst_case_model_identity_eig_nats
                    .unwrap()
        );
        assert!(
            far.experiments[0]
                .worst_case_model_identity_eig_nats
                .unwrap()
                > std::f64::consts::LN_2 * 0.95
        );
        assert!(
            far.experiments[0]
                .worst_case_model_identity_eig_nats
                .unwrap()
                <= std::f64::consts::LN_2 + 1.0e-6
        );
    }

    #[test]
    fn eig_is_invariant_to_common_affine_outcome_scaling() {
        let hypotheses = vec![hypothesis("model-a", b'a'), hypothesis("model-b", b'b')];
        let base = experiment(
            "base",
            1.0,
            Some(0.2),
            vec![
                prediction("model-a", Some(1.0), Some(0.1)),
                prediction("model-b", Some(2.0), Some(0.15)),
            ],
        );
        let scaled = experiment(
            "scaled",
            1.0,
            Some(20.0),
            vec![
                prediction("model-a", Some(100.0), Some(10.0)),
                prediction("model-b", Some(200.0), Some(15.0)),
            ],
        );
        let report = rank_model_discriminating_experiments(
            &hypotheses,
            &[base, scaled],
            &config(vec![equal_prior()], 256),
        )
        .unwrap();
        let left = report
            .experiments
            .iter()
            .find(|score| score.experiment_id == "base")
            .unwrap();
        let right = report
            .experiments
            .iter()
            .find(|score| score.experiment_id == "scaled")
            .unwrap();
        assert!(
            (left.worst_case_model_identity_eig_nats.unwrap()
                - right.worst_case_model_identity_eig_nats.unwrap())
            .abs()
                < 1.0e-7
        );
    }

    #[test]
    fn prior_sensitivity_is_reported_and_worst_case_is_used_for_ranking() {
        let hypotheses = vec![hypothesis("model-a", b'a'), hypothesis("model-b", b'b')];
        let priors = vec![
            ModelPriorScenario {
                scenario_id: "balanced".to_string(),
                scenario_weight: 1.0,
                probabilities: vec![0.5, 0.5],
            },
            ModelPriorScenario {
                scenario_id: "skewed".to_string(),
                scenario_weight: 1.0,
                probabilities: vec![0.99, 0.01],
            },
        ];
        let report = rank_model_discriminating_experiments(
            &hypotheses,
            &[experiment(
                "measurement",
                2.0,
                Some(0.05),
                vec![
                    prediction("model-a", Some(0.0), Some(0.0)),
                    prediction("model-b", Some(1.0), Some(0.0)),
                ],
            )],
            &config(priors, 256),
        )
        .unwrap();
        let score = &report.experiments[0];
        assert_eq!(score.prior_scenario_scores.len(), 2);
        assert!(
            score.prior_scenario_scores[0].refined_eig_nats
                > score.prior_scenario_scores[1].refined_eig_nats
        );
        assert!(score.worst_case_model_identity_eig_nats.is_some());
    }

    fn detection_prediction(
        model_id: &str,
        probability_detected: f64,
    ) -> ModelDetectionProbability {
        ModelDetectionProbability {
            model_id: model_id.to_string(),
            probability_detected,
            derivation_reference: format!("detection-model://{model_id}"),
        }
    }

    fn detection_candidate(
        id: &str,
        execution_probability: f64,
        predictions: Vec<GaussianModelPrediction>,
        detection: Vec<ModelDetectionProbability>,
    ) -> DetectionAwareCandidateExperiment {
        let mut experiment = experiment(id, 1.0, Some(0.1), predictions);
        experiment.availability_probability = execution_probability;
        DetectionAwareCandidateExperiment {
            experiment,
            detection_probabilities: detection,
        }
    }

    #[test]
    fn equal_model_detection_probability_reduces_to_uninformative_availability() {
        let hypotheses = vec![hypothesis("model-a", b'a'), hypothesis("model-b", b'b')];
        let predictions = vec![
            prediction("model-a", Some(-0.8), Some(0.1)),
            prediction("model-b", Some(0.8), Some(0.1)),
        ];
        let candidate = detection_candidate(
            "equal-detection",
            0.8,
            predictions.clone(),
            vec![
                detection_prediction("model-a", 0.5),
                detection_prediction("model-b", 0.5),
            ],
        );
        let report = rank_detection_aware_model_discriminating_experiments(
            &hypotheses,
            &[candidate],
            &config(vec![equal_prior()], 256),
        )
        .unwrap();
        let score = &report.experiments[0];
        let scenario = &score.prior_scenario_scores[0];
        assert_eq!(score.status, ExperimentDiscriminationStatus::Scored);
        assert!(scenario.detection_indicator_eig_nats.abs() < 1.0e-12);
        assert!((scenario.expected_detection_probability - 0.5).abs() < 1.0e-12);
        let mut legacy = experiment("legacy", 1.0, Some(0.1), predictions);
        legacy.availability_probability = 0.4;
        let legacy_report = rank_model_discriminating_experiments(
            &hypotheses,
            &[legacy],
            &config(vec![equal_prior()], 256),
        )
        .unwrap();
        assert!(
            (scenario.availability_adjusted_total_eig_nats
                - legacy_report.experiments[0]
                    .availability_adjusted_weighted_mean_model_identity_eig_nats
                    .unwrap())
            .abs()
                < 1.0e-8
        );
    }

    #[test]
    fn detection_indicator_can_discriminate_when_value_distributions_are_identical() {
        let hypotheses = vec![hypothesis("model-a", b'a'), hypothesis("model-b", b'b')];
        let candidate = detection_candidate(
            "detection-only",
            1.0,
            vec![
                prediction("model-a", Some(0.0), Some(0.2)),
                prediction("model-b", Some(0.0), Some(0.2)),
            ],
            vec![
                detection_prediction("model-a", 0.9),
                detection_prediction("model-b", 0.1),
            ],
        );
        let report = rank_detection_aware_model_discriminating_experiments(
            &hypotheses,
            &[candidate],
            &config(vec![equal_prior()], 256),
        )
        .unwrap();
        let scenario = &report.experiments[0].prior_scenario_scores[0];
        let expected = binary_entropy(0.5) - 0.5 * binary_entropy(0.9) - 0.5 * binary_entropy(0.1);
        assert!((scenario.detection_indicator_eig_nats - expected).abs() < 1.0e-12);
        assert!(scenario.conditional_detected_value_eig_nats.unwrap().abs() < 1.0e-10);
        assert!((scenario.total_eig_given_execution_nats - expected).abs() < 1.0e-10);
    }

    #[test]
    fn model_conditional_detection_reweights_the_detected_measurement_prior() {
        let hypotheses = vec![hypothesis("model-a", b'a'), hypothesis("model-b", b'b')];
        let candidate = detection_candidate(
            "detection-conditioned-value",
            0.5,
            vec![
                prediction("model-a", Some(-1.0), Some(0.1)),
                prediction("model-b", Some(1.0), Some(0.1)),
            ],
            vec![
                detection_prediction("model-a", 0.9),
                detection_prediction("model-b", 0.1),
            ],
        );
        let report = rank_detection_aware_model_discriminating_experiments(
            &hypotheses,
            &[candidate],
            &config(vec![equal_prior()], 256),
        )
        .unwrap();
        let scenario = &report.experiments[0].prior_scenario_scores[0];
        assert!((scenario.expected_detection_probability - 0.5).abs() < 1.0e-12);
        assert!(scenario.detection_indicator_eig_nats > 0.0);
        assert!(scenario.conditional_detected_value_eig_nats.unwrap() > 0.0);
        assert!(
            (scenario.expected_posterior_entropy_nats
                - (scenario.prior_entropy_nats - 0.5 * scenario.total_eig_given_execution_nats))
                .abs()
                < 1.0e-10
        );
    }

    #[test]
    fn zero_detection_execution_and_zero_prior_models_are_defined() {
        let hypotheses = vec![hypothesis("model-a", b'a'), hypothesis("model-b", b'b')];
        let candidate = detection_candidate(
            "never-detected",
            0.0,
            vec![
                prediction("model-a", Some(-1.0), Some(0.1)),
                prediction("model-b", Some(1.0), Some(0.1)),
            ],
            vec![
                detection_prediction("model-a", 0.0),
                detection_prediction("model-b", 0.8),
            ],
        );
        let mut prior = equal_prior();
        prior.probabilities = vec![1.0, 0.0];
        let report = rank_detection_aware_model_discriminating_experiments(
            &hypotheses,
            &[candidate],
            &config(vec![prior], 256),
        )
        .unwrap();
        let scenario = &report.experiments[0].prior_scenario_scores[0];
        assert_eq!(scenario.expected_detection_probability, 0.0);
        assert_eq!(scenario.detection_indicator_eig_nats, 0.0);
        assert_eq!(scenario.total_eig_given_execution_nats, 0.0);
        assert_eq!(scenario.availability_adjusted_total_eig_nats, 0.0);
        assert_eq!(scenario.conditional_detected_value_eig_nats, None);
        assert!((scenario.expected_posterior_entropy_nats).abs() < 1.0e-12);
    }

    #[test]
    fn detection_aware_eig_matches_independent_joint_outcome_integral() {
        let hypotheses = vec![hypothesis("model-a", b'a'), hypothesis("model-b", b'b')];
        let candidate = detection_candidate(
            "joint-reference",
            0.7,
            vec![
                prediction("model-a", Some(-0.7), Some(0.1)),
                prediction("model-b", Some(0.9), Some(0.2)),
            ],
            vec![
                detection_prediction("model-a", 0.8),
                detection_prediction("model-b", 0.25),
            ],
        );
        let report = rank_detection_aware_model_discriminating_experiments(
            &hypotheses,
            &[candidate],
            &config(vec![equal_prior()], 512),
        )
        .unwrap();
        let observed =
            report.experiments[0].prior_scenario_scores[0].availability_adjusted_total_eig_nats;

        // Independent direct joint likelihood integral over (D=0) and
        // (D=1, Y=y), rather than reusing the conditional-EIG decomposition.
        let prior = [0.5_f64, 0.5];
        let detection = [0.8_f64, 0.25];
        let means = [-0.7, 0.9];
        let sigmas = [0.1_f64.hypot(0.1), 0.2_f64.hypot(0.1)];
        let q_bar = prior[0] * detection[0] + prior[1] * detection[1];
        let mut direct = 0.0;
        for index in 0..2 {
            let p_miss = prior[index] * (1.0 - detection[index]);
            if p_miss > 0.0 && q_bar < 1.0 {
                direct += p_miss * ((1.0 - detection[index]) / (1.0 - q_bar)).ln();
            }
        }
        let lower = means[0] - 10.0 * sigmas[0].max(sigmas[1]);
        let upper = means[1] + 10.0 * sigmas[0].max(sigmas[1]);
        let intervals = 20_000_usize;
        let step = (upper - lower) / intervals as f64;
        let mut integral = 0.0;
        for index in 0..=intervals {
            let y = lower + index as f64 * step;
            let densities = [
                (-0.5 * (y - means[0]).powi(2) / sigmas[0].powi(2)).exp()
                    / (std::f64::consts::TAU.sqrt() * sigmas[0]),
                (-0.5 * (y - means[1]).powi(2) / sigmas[1].powi(2)).exp()
                    / (std::f64::consts::TAU.sqrt() * sigmas[1]),
            ];
            let joints = [
                prior[0] * detection[0] * densities[0],
                prior[1] * detection[1] * densities[1],
            ];
            let mixture = joints[0] + joints[1];
            let mut density = 0.0;
            for model in 0..2 {
                if joints[model] > 0.0 {
                    density += joints[model] * (densities[model] * q_bar / mixture).ln();
                }
            }
            let weight = if index == 0 || index == intervals {
                1.0
            } else if index % 2 == 0 {
                2.0
            } else {
                4.0
            };
            integral += weight * density;
        }
        direct += integral * step / 3.0;
        direct *= 0.7;
        assert!(
            (observed - direct).abs() < 2.0e-6,
            "observed={observed:.12}, direct={direct:.12}, absolute_error={:.6e}",
            (observed - direct).abs()
        );
    }

    #[test]
    fn certain_model_independent_detection_has_zero_indicator_information() {
        let hypotheses = vec![hypothesis("model-a", b'a'), hypothesis("model-b", b'b')];
        for probability in [0.0, 1.0] {
            let candidate = detection_candidate(
                "certain-detection",
                1.0,
                vec![
                    prediction("model-a", Some(-0.5), Some(0.1)),
                    prediction("model-b", Some(0.5), Some(0.1)),
                ],
                vec![
                    detection_prediction("model-a", probability),
                    detection_prediction("model-b", probability),
                ],
            );
            let report = rank_detection_aware_model_discriminating_experiments(
                &hypotheses,
                &[candidate],
                &config(vec![equal_prior()], 256),
            )
            .unwrap();
            let scenario = &report.experiments[0].prior_scenario_scores[0];
            assert!(scenario.detection_indicator_eig_nats.abs() < 1.0e-12);
            assert!((scenario.expected_detection_probability - probability).abs() < 1.0e-12);
            if probability == 0.0 {
                assert_eq!(scenario.conditional_detected_value_eig_nats, None);
            } else {
                assert!(scenario.conditional_detected_value_eig_nats.unwrap() > 0.0);
            }
        }
    }

    #[test]
    fn prediction_and_detection_row_permutations_preserve_detection_aware_eig() {
        let hypotheses = vec![
            hypothesis("model-a", b'a'),
            hypothesis("model-b", b'b'),
            hypothesis("model-c", b'c'),
        ];
        let predictions = vec![
            prediction("model-a", Some(-0.8), Some(0.1)),
            prediction("model-b", Some(0.1), Some(0.2)),
            prediction("model-c", Some(1.3), Some(0.15)),
        ];
        let detection_rows = vec![
            detection_prediction("model-a", 0.85),
            detection_prediction("model-b", 0.45),
            detection_prediction("model-c", 0.10),
        ];
        let first = detection_candidate(
            "first-order",
            0.9,
            predictions.clone(),
            detection_rows.clone(),
        );
        let second = detection_candidate(
            "permuted-order",
            0.9,
            vec![
                predictions[2].clone(),
                predictions[0].clone(),
                predictions[1].clone(),
            ],
            vec![
                detection_rows[1].clone(),
                detection_rows[2].clone(),
                detection_rows[0].clone(),
            ],
        );
        let report = rank_detection_aware_model_discriminating_experiments(
            &hypotheses,
            &[first, second],
            &config(
                vec![
                    ModelPriorScenario {
                        scenario_id: "balanced".to_string(),
                        scenario_weight: 1.0,
                        probabilities: vec![1.0, 1.0, 1.0],
                    },
                    ModelPriorScenario {
                        scenario_id: "skewed".to_string(),
                        scenario_weight: 1.0,
                        probabilities: vec![4.0, 1.0, 1.0],
                    },
                ],
                256,
            ),
        )
        .unwrap();
        let first = report
            .experiments
            .iter()
            .find(|score| score.experiment_id == "first-order")
            .unwrap();
        let second = report
            .experiments
            .iter()
            .find(|score| score.experiment_id == "permuted-order")
            .unwrap();
        assert_eq!(first.status, ExperimentDiscriminationStatus::Scored);
        assert_eq!(second.status, ExperimentDiscriminationStatus::Scored);
        for (left, right) in first
            .prior_scenario_scores
            .iter()
            .zip(&second.prior_scenario_scores)
        {
            assert!(
                (left.detection_indicator_eig_nats - right.detection_indicator_eig_nats).abs()
                    < 1.0e-12
            );
            assert!(
                (left.total_eig_given_execution_nats - right.total_eig_given_execution_nats).abs()
                    < 1.0e-10
            );
            assert!(
                (left.availability_adjusted_total_eig_nats
                    - right.availability_adjusted_total_eig_nats)
                    .abs()
                    < 1.0e-10
            );
        }
    }

    #[test]
    fn non_comparable_prediction_remains_typed_even_when_detection_rows_are_incomplete() {
        let hypotheses = vec![hypothesis("model-a", b'a'), hypothesis("model-b", b'b')];
        let candidate = detection_candidate(
            "partial-model",
            1.0,
            vec![
                prediction("model-a", Some(-1.0), Some(0.1)),
                GaussianModelPrediction {
                    model_id: "model-b".to_string(),
                    disposition: PredictionDisposition::NotDerived,
                    predictive_mean: None,
                    predictive_sigma: None,
                    derivation_reference: None,
                },
            ],
            vec![detection_prediction("model-a", 0.5)],
        );
        let report = rank_detection_aware_model_discriminating_experiments(
            &hypotheses,
            &[candidate],
            &config(vec![equal_prior()], 128),
        )
        .unwrap();
        assert!(matches!(
            &report.experiments[0].status,
            ExperimentDiscriminationStatus::NotComparable { .. }
        ));
    }

    #[test]
    fn detection_probabilities_fail_closed_on_missing_duplicate_or_invalid_rows() {
        let hypotheses = vec![hypothesis("model-a", b'a'), hypothesis("model-b", b'b')];
        let base = vec![
            prediction("model-a", Some(-1.0), Some(0.1)),
            prediction("model-b", Some(1.0), Some(0.1)),
        ];
        let missing = detection_candidate(
            "missing",
            1.0,
            base.clone(),
            vec![detection_prediction("model-a", 0.5)],
        );
        assert!(matches!(
            rank_detection_aware_model_discriminating_experiments(
                &hypotheses,
                &[missing],
                &config(vec![equal_prior()], 128)
            ),
            Err(ModelDiscriminationError::DetectionProbabilityCount { .. })
        ));

        let duplicate = detection_candidate(
            "duplicate",
            1.0,
            base.clone(),
            vec![
                detection_prediction("model-a", 0.5),
                detection_prediction("model-a", 0.4),
            ],
        );
        assert!(matches!(
            rank_detection_aware_model_discriminating_experiments(
                &hypotheses,
                &[duplicate],
                &config(vec![equal_prior()], 128)
            ),
            Err(ModelDiscriminationError::DuplicateDetectionProbability { .. })
        ));

        let invalid = detection_candidate(
            "invalid",
            1.0,
            base,
            vec![
                detection_prediction("model-a", 1.01),
                detection_prediction("model-b", 0.5),
            ],
        );
        assert!(matches!(
            rank_detection_aware_model_discriminating_experiments(
                &hypotheses,
                &[invalid],
                &config(vec![equal_prior()], 128)
            ),
            Err(ModelDiscriminationError::InvalidDetectionProbability { .. })
        ));

        let unexpected = detection_candidate(
            "unexpected",
            1.0,
            vec![
                prediction("model-a", Some(-1.0), Some(0.1)),
                prediction("model-b", Some(1.0), Some(0.1)),
            ],
            vec![
                detection_prediction("model-a", 0.5),
                detection_prediction("model-c", 0.5),
            ],
        );
        assert!(matches!(
            rank_detection_aware_model_discriminating_experiments(
                &hypotheses,
                &[unexpected],
                &config(vec![equal_prior()], 128)
            ),
            Err(ModelDiscriminationError::UnexpectedDetectionProbability { .. })
        ));

        let mut missing_reference = vec![
            detection_prediction("model-a", 0.5),
            detection_prediction("model-b", 0.5),
        ];
        missing_reference[1].derivation_reference = " ".to_string();
        let ungrounded = detection_candidate(
            "missing-derivation",
            1.0,
            vec![
                prediction("model-a", Some(-1.0), Some(0.1)),
                prediction("model-b", Some(1.0), Some(0.1)),
            ],
            missing_reference,
        );
        assert!(matches!(
            rank_detection_aware_model_discriminating_experiments(
                &hypotheses,
                &[ungrounded],
                &config(vec![equal_prior()], 128)
            ),
            Err(ModelDiscriminationError::MissingDetectionDerivation { .. })
        ));
    }

    #[test]
    fn availability_adjusts_expected_eig_without_erasing_conditional_eig() {
        let hypotheses = vec![hypothesis("model-a", b'a'), hypothesis("model-b", b'b')];
        let make = |id: &str, availability_probability: f64| {
            let mut candidate = experiment(
                id,
                2.0,
                Some(0.1),
                vec![
                    prediction("model-a", Some(-1.0), Some(0.0)),
                    prediction("model-b", Some(1.0), Some(0.0)),
                ],
            );
            candidate.availability_probability = availability_probability;
            candidate
        };
        let report = rank_model_discriminating_experiments(
            &hypotheses,
            &[make("always", 1.0), make("half", 0.5), make("never", 0.0)],
            &config(vec![equal_prior()], 256),
        )
        .unwrap();

        let score = |id: &str| {
            report
                .experiments
                .iter()
                .find(|score| score.experiment_id == id)
                .unwrap()
        };
        let always = score("always");
        let half = score("half");
        let never = score("never");
        let conditional = always.worst_case_model_identity_eig_nats.unwrap();

        assert!((half.worst_case_model_identity_eig_nats.unwrap() - conditional).abs() < 1.0e-12);
        assert!(
            (half
                .availability_adjusted_worst_case_model_identity_eig_nats
                .unwrap()
                - 0.5 * conditional)
                .abs()
                < 1.0e-12
        );
        assert!(
            (never
                .availability_adjusted_worst_case_model_identity_eig_nats
                .unwrap())
            .abs()
                < 1.0e-12
        );
        let half_scenario = &half.prior_scenario_scores[0];
        assert!(
            (half_scenario.expected_posterior_entropy_nats
                - (half_scenario.prior_entropy_nats
                    - half_scenario.availability_adjusted_eig_nats))
                .abs()
                < 1.0e-12
        );
        assert!(
            (never.prior_scenario_scores[0].expected_posterior_entropy_nats
                - never.prior_scenario_scores[0].prior_entropy_nats)
                .abs()
                < 1.0e-12
        );
        assert_eq!(
            report.experiments[0].experiment_id, "always",
            "availability-adjusted EIG per cost should rank the fully available candidate first"
        );
    }

    #[test]
    fn invalid_availability_probability_fails_closed() {
        let hypotheses = vec![hypothesis("model-a", b'a'), hypothesis("model-b", b'b')];
        let mut candidate = experiment(
            "invalid-availability",
            1.0,
            Some(0.1),
            vec![
                prediction("model-a", Some(0.0), Some(0.0)),
                prediction("model-b", Some(1.0), Some(0.0)),
            ],
        );
        candidate.availability_probability = f64::NAN;

        assert_eq!(
            rank_model_discriminating_experiments(
                &hypotheses,
                &[candidate],
                &config(vec![equal_prior()], 128),
            ),
            Err(ModelDiscriminationError::InvalidAvailabilityProbability(
                "invalid-availability".to_string()
            ))
        );
    }

    #[test]
    fn not_derived_prediction_and_uncalibrated_noise_block_numeric_score() {
        let hypotheses = vec![hypothesis("model-a", b'a'), hypothesis("model-b", b'b')];
        let mut experiment = experiment(
            "blocked",
            1.0,
            None,
            vec![
                prediction("model-a", Some(0.0), Some(0.0)),
                prediction("model-b", None, None),
            ],
        );
        experiment.noise_calibration = NoiseCalibrationStatus::Uncalibrated;
        let report = rank_model_discriminating_experiments(
            &hypotheses,
            &[experiment],
            &config(vec![equal_prior()], 128),
        )
        .unwrap();
        let score = &report.experiments[0];
        assert!(matches!(
            score.status,
            ExperimentDiscriminationStatus::NotComparable { .. }
        ));
        assert!(score.worst_case_model_identity_eig_nats.is_none());
    }

    #[test]
    fn open_world_unresolved_model_is_not_assigned_a_numeric_distribution() {
        let hypotheses = vec![
            hypothesis("model-a", b'a'),
            hypothesis("model-b", b'b'),
            // OTHER_UNRESOLVED is not a member of the numeric hypothesis set.
        ];
        let result = rank_model_discriminating_experiments(
            &hypotheses,
            &[experiment(
                "candidate",
                1.0,
                Some(0.1),
                vec![
                    prediction("model-a", Some(0.0), Some(0.0)),
                    prediction("model-b", Some(1.0), Some(0.0)),
                ],
            )],
            &config(vec![equal_prior()], 128),
        )
        .unwrap();
        assert!(result.experiments[0].open_world_unresolved_excluded);
        assert_eq!(result.model_universe, ModelUniverseCoverage::OpenWorld);
    }

    #[test]
    fn cost_normalized_worst_case_eig_determines_ranking() {
        let hypotheses = vec![hypothesis("model-a", b'a'), hypothesis("model-b", b'b')];
        let expensive = experiment(
            "expensive",
            10.0,
            Some(0.1),
            vec![
                prediction("model-a", Some(0.0), Some(0.0)),
                prediction("model-b", Some(2.0), Some(0.0)),
            ],
        );
        let cheap = experiment(
            "cheap",
            1.0,
            Some(0.1),
            vec![
                prediction("model-a", Some(0.0), Some(0.0)),
                prediction("model-b", Some(1.0), Some(0.0)),
            ],
        );
        let report = rank_model_discriminating_experiments(
            &hypotheses,
            &[expensive, cheap],
            &config(vec![equal_prior()], 256),
        )
        .unwrap();
        assert_eq!(report.experiments[0].experiment_id, "cheap");
    }

    #[test]
    fn insufficient_quadrature_refinement_is_not_ranked() {
        let hypotheses = vec![hypothesis("model-a", b'a'), hypothesis("model-b", b'b')];
        let experiment = experiment(
            "underresolved",
            1.0,
            Some(0.1),
            vec![
                prediction("model-a", Some(-0.2), Some(0.0)),
                prediction("model-b", Some(0.2), Some(0.0)),
            ],
        );
        let mut config = config(vec![equal_prior()], 64);
        config.quadrature_tolerance_nats = 1.0e-30;
        let report =
            rank_model_discriminating_experiments(&hypotheses, &[experiment], &config).unwrap();
        let score = &report.experiments[0];

        assert!(matches!(
            score.status,
            ExperimentDiscriminationStatus::QuadratureNotConverged { .. }
        ));
        assert!(score.worst_case_model_identity_eig_nats.is_none());
        assert!(score.weighted_mean_eig_per_cost.is_none());
    }

    #[test]
    fn numeric_unresolved_hypothesis_is_rejected() {
        let hypotheses = vec![
            hypothesis("model-a", b'a'),
            DiscriminationHypothesis {
                model_id: OTHER_UNRESOLVED_MODEL_ID.to_string(),
                source_revision_sha: "d".repeat(40),
                specification_reference: "synthetic://unresolved".to_string(),
            },
        ];
        let candidate = experiment(
            "candidate",
            1.0,
            Some(0.1),
            vec![
                prediction("model-a", Some(0.0), Some(0.0)),
                prediction(OTHER_UNRESOLVED_MODEL_ID, Some(1.0), Some(0.0)),
            ],
        );

        assert_eq!(
            rank_model_discriminating_experiments(
                &hypotheses,
                &[candidate],
                &config(
                    vec![ModelPriorScenario {
                        scenario_id: "equal".to_string(),
                        scenario_weight: 1.0,
                        probabilities: vec![0.5, 0.5],
                    }],
                    128,
                ),
            ),
            Err(ModelDiscriminationError::UnresolvedHypothesisMustRemainOpen)
        );
    }

    #[test]
    fn malformed_prediction_coverage_and_prior_fail_closed() {
        let hypotheses = vec![hypothesis("model-a", b'a'), hypothesis("model-b", b'b')];
        let incomplete = experiment(
            "incomplete",
            1.0,
            Some(0.1),
            vec![prediction("model-a", Some(0.0), Some(0.0))],
        );
        assert!(matches!(
            rank_model_discriminating_experiments(
                &hypotheses,
                &[incomplete],
                &config(vec![equal_prior()], 128)
            ),
            Err(ModelDiscriminationError::PredictionCount { .. })
        ));

        let mut prior = equal_prior();
        prior.probabilities[1] = f64::NAN;
        assert!(matches!(
            rank_model_discriminating_experiments(
                &hypotheses,
                &[experiment(
                    "valid",
                    1.0,
                    Some(0.1),
                    vec![
                        prediction("model-a", Some(0.0), Some(0.0)),
                        prediction("model-b", Some(1.0), Some(0.0)),
                    ]
                )],
                &config(vec![prior], 128)
            ),
            Err(ModelDiscriminationError::InvalidPriorProbability { .. })
        ));
    }
}
