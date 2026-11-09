// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic planted-model recovery for scalar Gaussian experiments.
//!
//! This evaluates a frozen MAP/abstention rule on synthetic observations
//! generated from each candidate model. An optional OTHER_UNRESOLVED generator
//! can test the open-world failure mode: its distribution is used only to
//! generate observations and is never added to the inference candidate set.
//!
//! This is an end-to-end model-recovery metric, not model-identity EIG or
//! expected KL divergence. It is bounded to scalar Gaussian predictive models;
//! it is not a general Bayesian inference or posterior predictive engine.

use super::model_discrimination::{
    CandidateExperiment, DiscriminationHypothesis, ExperimentBlocker, ModelDiscriminationError,
    ModelPriorScenario,
};
use super::theory_prediction::{
    MeasurementNoiseModel, NoiseCalibrationStatus, OTHER_UNRESOLVED_MODEL_ID, PredictionDisposition,
};
use std::collections::BTreeSet;
use thiserror::Error;

/// Maximum number of synthetic observations across all generating cases.
pub const MAX_MODEL_RECOVERY_TRIALS: usize = 1_000_000;

/// Maximum number of generating cases (named hypotheses plus optional open-world generator).
pub const MAX_MODEL_RECOVERY_GENERATORS: usize = 17;

/// One planted data-generating model instance.
///
/// For a named generator, `generator_id` must match one candidate hypothesis.
/// The optional `OTHER_UNRESOLVED` generator is deliberately *not* an inference
/// candidate. Its numeric distribution is used only to plant an M-open
/// synthetic test and is never passed to the classifier.
#[derive(Clone, Debug, PartialEq)]
pub struct RecoveryGenerator {
    pub generator_id: String,
    pub predictive_mean: f64,
    /// Standard deviation of the generator's predictive uncertainty.
    pub predictive_sigma: f64,
}

/// One row in the recovery confusion matrix is a generating model; columns are
/// the named inference models followed by an abstention column.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelRecoveryReport {
    pub experiment_id: String,
    pub prior_scenario_id: String,
    pub model_ids: Vec<String>,
    pub generator_ids: Vec<String>,
    /// Rows follow `generator_ids`; columns follow `model_ids`, with the
    /// final column reserved for abstentions.
    pub confusion_counts: Vec<Vec<u64>>,
    pub samples_per_generator: usize,
    pub availability_probability: f64,
    /// Number of trials per generator that produced no usable observation.
    /// These are counted as abstentions, not discarded from the denominator.
    pub unavailable_observations_by_generator: Vec<u64>,
    pub abstention_threshold: f64,
    /// Fraction correctly recovered across named generating models, with
    /// equal sampling per named model.
    pub named_model_accuracy: f64,
    /// Accuracy weighted by the supplied model prior.
    pub prior_weighted_named_model_accuracy: f64,
    /// Fraction of named-model trials assigned to any named model.
    pub named_model_coverage: f64,
    /// Accuracy among non-abstained named-model trials; None when coverage is 0.
    pub named_model_selective_accuracy: Option<f64>,
    /// Fraction of OTHER_UNRESOLVED trials assigned to any named model.
    /// A larger value is a false-confident assignment under this M-open fixture.
    pub open_world_false_confident_assignment_rate: Option<f64>,
    /// Fraction of OTHER_UNRESOLVED trials correctly left unresolved by abstention.
    pub open_world_abstention_rate: Option<f64>,
}

/// Validation or numerical failure in the model-recovery reference.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ModelRecoveryError {
    #[error(transparent)]
    InvalidDiscriminationDesign(#[from] ModelDiscriminationError),
    #[error("model-recovery sample count must be in 1..=1000000")]
    InvalidSampleCount,
    #[error("model-recovery total trial count exceeds {MAX_MODEL_RECOVERY_TRIALS}")]
    TooManyTrials,
    #[error("model-recovery abstention threshold must be finite and in (0, 1]")]
    InvalidAbstentionThreshold,
    #[error("prior scenario has {actual} probabilities; expected {expected}")]
    PriorProbabilityCount { expected: usize, actual: usize },
    #[error("prior scenario ID must not be empty")]
    EmptyPriorScenarioId,
    #[error("prior scenario contains an invalid probability")]
    InvalidPriorProbability,
    #[error("prior scenario has zero total probability")]
    ZeroPriorProbability,
    #[error("generator set is empty")]
    NoGenerators,
    #[error("generator set has {count} rows; maximum is {maximum}")]
    TooManyGenerators { count: usize, maximum: usize },
    #[error("generator ID is empty")]
    EmptyGeneratorId,
    #[error("duplicate generator ID: {0}")]
    DuplicateGeneratorId(String),
    #[error("unknown generator ID {0}; only named hypotheses and OTHER_UNRESOLVED are permitted")]
    UnknownGeneratorId(String),
    #[error("named generating model is missing from the generator set: {0}")]
    MissingNamedGenerator(String),
    #[error("generator {generator_id} has invalid predictive parameters")]
    InvalidGeneratorParameters { generator_id: String },
    #[error("experiment is not comparable for model recovery: {blockers:?}")]
    NotComparable { blockers: Vec<ExperimentBlocker> },
    #[error("synthetic observation or likelihood calculation became non-finite")]
    NonFiniteSyntheticCalculation,
}

#[derive(Clone, Copy, Debug)]
struct GaussianInferenceModel {
    mean: f64,
    sigma: f64,
    prior: f64,
}

/// Simulate a deterministic Monte Carlo model-recovery experiment.
///
/// Each generator receives exactly `samples_per_generator` observations.
/// Candidate-model posteriors are computed from the declared `prior` and the
/// experiment's validated scalar Gaussian predictive distributions. The
/// classifier emits the maximum-posterior model only when its posterior
/// probability reaches `minimum_posterior_probability`; otherwise it abstains.
///
/// The function reports the full generator-by-prediction confusion matrix,
/// equal-weight named-model accuracy, prior-weighted accuracy, coverage,
/// selective accuracy, and M-open false-confident assignment rate. It does not
/// assign any likelihood to OTHER_UNRESOLVED.
pub fn evaluate_model_recovery(
    hypotheses: &[DiscriminationHypothesis],
    experiment: &CandidateExperiment,
    prior: &ModelPriorScenario,
    generators: &[RecoveryGenerator],
    samples_per_generator: usize,
    seed: u64,
    minimum_posterior_probability: f64,
) -> Result<ModelRecoveryReport, ModelRecoveryError> {
    if samples_per_generator == 0 {
        return Err(ModelRecoveryError::InvalidSampleCount);
    }
    if !experiment.availability_probability.is_finite()
        || !(0.0..=1.0).contains(&experiment.availability_probability)
    {
        return Err(ModelDiscriminationError::InvalidAvailabilityProbability(
            experiment.experiment_id.clone(),
        )
        .into());
    }
    if !minimum_posterior_probability.is_finite()
        || minimum_posterior_probability <= 0.0
        || minimum_posterior_probability > 1.0
    {
        return Err(ModelRecoveryError::InvalidAbstentionThreshold);
    }
    if hypotheses.len() < 2 {
        return Err(ModelDiscriminationError::TooFewHypotheses.into());
    }
    if hypotheses.len() > super::model_discrimination::MAX_DISCRIMINATION_HYPOTHESES {
        return Err(ModelDiscriminationError::TooManyHypotheses {
            count: hypotheses.len(),
            maximum: super::model_discrimination::MAX_DISCRIMINATION_HYPOTHESES,
        }
        .into());
    }

    let named_trial_count = hypotheses
        .len()
        .checked_mul(samples_per_generator)
        .ok_or(ModelRecoveryError::TooManyTrials)?;
    let maximum_generator_count = hypotheses.len().saturating_add(1);
    if generators.len() > maximum_generator_count
        || generators.len() > MAX_MODEL_RECOVERY_GENERATORS
    {
        return Err(ModelRecoveryError::TooManyGenerators {
            count: generators.len(),
            maximum: maximum_generator_count.min(MAX_MODEL_RECOVERY_GENERATORS),
        });
    }
    if generators.is_empty() {
        return Err(ModelRecoveryError::NoGenerators);
    }
    let total_trials = generators
        .len()
        .checked_mul(samples_per_generator)
        .ok_or(ModelRecoveryError::TooManyTrials)?;
    if total_trials > MAX_MODEL_RECOVERY_TRIALS || named_trial_count > total_trials {
        return Err(ModelRecoveryError::TooManyTrials);
    }

    if prior.scenario_id.trim().is_empty() {
        return Err(ModelRecoveryError::EmptyPriorScenarioId);
    }
    if prior.probabilities.len() != hypotheses.len() {
        return Err(ModelRecoveryError::PriorProbabilityCount {
            expected: hypotheses.len(),
            actual: prior.probabilities.len(),
        });
    }
    if prior
        .probabilities
        .iter()
        .any(|probability| !probability.is_finite() || *probability < 0.0)
    {
        return Err(ModelRecoveryError::InvalidPriorProbability);
    }
    let prior_sum: f64 = prior.probabilities.iter().sum();
    if !prior_sum.is_finite() || prior_sum <= 0.0 {
        return Err(ModelRecoveryError::ZeroPriorProbability);
    }
    let normalized_prior: Vec<f64> = prior
        .probabilities
        .iter()
        .map(|probability| probability / prior_sum)
        .collect();

    let mut model_ids = Vec::with_capacity(hypotheses.len());
    let mut expected_model_ids = BTreeSet::new();
    for hypothesis in hypotheses {
        if hypothesis.model_id.trim().is_empty() {
            return Err(ModelDiscriminationError::EmptyHypothesisId(model_ids.len()).into());
        }
        if hypothesis.model_id == OTHER_UNRESOLVED_MODEL_ID
            || !expected_model_ids.insert(hypothesis.model_id.as_str())
        {
            return Err(ModelDiscriminationError::DuplicateHypothesisId(
                hypothesis.model_id.clone(),
            )
            .into());
        }
        model_ids.push(hypothesis.model_id.clone());
    }

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
    if experiment.predictions.len() != hypotheses.len() {
        return Err(ModelDiscriminationError::PredictionCount {
            experiment_id: experiment.experiment_id.clone(),
            expected: hypotheses.len(),
            actual: experiment.predictions.len(),
        }
        .into());
    }
    let mut inference_models = Vec::with_capacity(hypotheses.len());
    let mut seen_prediction_ids = BTreeSet::new();
    for prediction in &experiment.predictions {
        if !seen_prediction_ids.insert(prediction.model_id.as_str()) {
            return Err(ModelDiscriminationError::DuplicateModelPrediction {
                experiment_id: experiment.experiment_id.clone(),
                model_id: prediction.model_id.clone(),
            }
            .into());
        }
        if prediction.disposition != PredictionDisposition::Quantitative {
            blockers.push(ExperimentBlocker::PredictionNotQuantitative {
                model_id: prediction.model_id.clone(),
                disposition: prediction.disposition,
            });
            continue;
        }
        let Some(mean) = prediction.predictive_mean else {
            blockers.push(ExperimentBlocker::PredictionMissingMean {
                model_id: prediction.model_id.clone(),
            });
            continue;
        };
        let Some(predictive_sigma) = prediction.predictive_sigma else {
            blockers.push(ExperimentBlocker::PredictionMissingUncertainty {
                model_id: prediction.model_id.clone(),
            });
            continue;
        };
        if prediction
            .derivation_reference
            .as_ref()
            .is_none_or(|reference| reference.trim().is_empty())
        {
            blockers.push(ExperimentBlocker::PredictionMissingDerivation {
                model_id: prediction.model_id.clone(),
            });
        }
        if !mean.is_finite() || !predictive_sigma.is_finite() || predictive_sigma < 0.0 {
            return Err(ModelDiscriminationError::NonFinitePredictiveMean {
                experiment_id: experiment.experiment_id.clone(),
                model_id: prediction.model_id.clone(),
            }
            .into());
        }
        let model_index = model_ids
            .iter()
            .position(|model_id| model_id == &prediction.model_id)
            .ok_or_else(|| ModelDiscriminationError::MissingModelPrediction {
                experiment_id: experiment.experiment_id.clone(),
                model_id: prediction.model_id.clone(),
            })?;
        inference_models.push((
            model_index,
            GaussianInferenceModel {
                mean,
                sigma: predictive_sigma,
                prior: normalized_prior[model_index],
            },
        ));
    }
    if seen_prediction_ids.len() != hypotheses.len() {
        return Err(ModelDiscriminationError::PredictionCount {
            experiment_id: experiment.experiment_id.clone(),
            expected: hypotheses.len(),
            actual: seen_prediction_ids.len(),
        }
        .into());
    }
    if experiment
        .measurement_sigma
        .is_some_and(|sigma| !sigma.is_finite() || sigma <= 0.0)
    {
        return Err(ModelDiscriminationError::InvalidMeasurementSigma {
            experiment_id: experiment.experiment_id.clone(),
        }
        .into());
    }
    if !blockers.is_empty() {
        return Err(ModelRecoveryError::NotComparable { blockers });
    }
    let measurement_sigma = experiment.measurement_sigma.unwrap();

    // Store inference model distributions in canonical hypothesis order.
    inference_models.sort_by_key(|(index, _)| *index);
    let inference_models: Vec<GaussianInferenceModel> = inference_models
        .into_iter()
        .map(|(_, model)| {
            let total_sigma = model.sigma.hypot(measurement_sigma);
            if !total_sigma.is_finite() || total_sigma <= 0.0 {
                return Err(ModelRecoveryError::NonFiniteSyntheticCalculation);
            }
            Ok(GaussianInferenceModel {
                sigma: total_sigma,
                ..model
            })
        })
        .collect::<Result<_, _>>()?;

    let mut generator_ids = Vec::with_capacity(generators.len());
    let mut seen_generator_ids = BTreeSet::new();
    let mut has_unresolved_generator = false;
    for generator in generators {
        if generator.generator_id.trim().is_empty() {
            return Err(ModelRecoveryError::EmptyGeneratorId);
        }
        if !seen_generator_ids.insert(generator.generator_id.as_str()) {
            return Err(ModelRecoveryError::DuplicateGeneratorId(
                generator.generator_id.clone(),
            ));
        }
        if generator.generator_id == OTHER_UNRESOLVED_MODEL_ID {
            has_unresolved_generator = true;
        } else if !expected_model_ids.contains(generator.generator_id.as_str()) {
            return Err(ModelRecoveryError::UnknownGeneratorId(
                generator.generator_id.clone(),
            ));
        }
        if !generator.predictive_mean.is_finite()
            || !generator.predictive_sigma.is_finite()
            || generator.predictive_sigma < 0.0
        {
            return Err(ModelRecoveryError::InvalidGeneratorParameters {
                generator_id: generator.generator_id.clone(),
            });
        }
        generator_ids.push(generator.generator_id.clone());
    }
    for model_id in &model_ids {
        if !seen_generator_ids.contains(model_id.as_str()) {
            return Err(ModelRecoveryError::MissingNamedGenerator(model_id.clone()));
        }
    }

    let mut rng = SplitMix64::new(seed);
    let abstain_column = model_ids.len();
    let mut confusion_counts = vec![vec![0_u64; model_ids.len() + 1]; generators.len()];
    let mut unavailable_observations_by_generator = vec![0_u64; generators.len()];
    let mut named_correct = 0_u64;
    let mut named_assigned = 0_u64;
    let mut named_abstained = 0_u64;
    let mut prior_weighted_accuracy = 0.0;

    for (generator_index, generator) in generators.iter().enumerate() {
        let generator_sigma = generator.predictive_sigma.hypot(measurement_sigma);
        if !generator_sigma.is_finite() || generator_sigma <= 0.0 {
            return Err(ModelRecoveryError::InvalidGeneratorParameters {
                generator_id: generator.generator_id.clone(),
            });
        }
        for _ in 0..samples_per_generator {
            if rng.unit_open() > experiment.availability_probability {
                confusion_counts[generator_index][abstain_column] += 1;
                unavailable_observations_by_generator[generator_index] += 1;
                if generator.generator_id != OTHER_UNRESOLVED_MODEL_ID {
                    named_abstained += 1;
                }
                continue;
            }
            let observation = generator.predictive_mean + generator_sigma * rng.standard_normal();
            if !observation.is_finite() {
                return Err(ModelRecoveryError::NonFiniteSyntheticCalculation);
            }
            let mut log_joint = Vec::with_capacity(inference_models.len());
            for model in &inference_models {
                let value = if model.prior > 0.0 {
                    model.prior.ln() + normal_log_density(observation, model.mean, model.sigma)
                } else {
                    f64::NEG_INFINITY
                };
                log_joint.push(value);
            }
            let normalizer =
                log_sum_exp(&log_joint).ok_or(ModelRecoveryError::NonFiniteSyntheticCalculation)?;
            let (best_index, best_log_joint) = log_joint
                .iter()
                .copied()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(&right.1))
                .ok_or(ModelRecoveryError::NonFiniteSyntheticCalculation)?;
            let posterior = (best_log_joint - normalizer).exp();

            let is_named_generator = generator.generator_id != OTHER_UNRESOLVED_MODEL_ID;
            if posterior >= minimum_posterior_probability {
                confusion_counts[generator_index][best_index] += 1;
                if is_named_generator {
                    named_assigned += 1;
                    if model_ids[best_index] == generator.generator_id {
                        named_correct += 1;
                    }
                }
            } else {
                confusion_counts[generator_index][abstain_column] += 1;
                if is_named_generator {
                    named_abstained += 1;
                }
            }
        }
        if let Some(model_index) = model_ids
            .iter()
            .position(|model_id| model_id == &generator.generator_id)
        {
            let row_accuracy = confusion_counts[generator_index][model_index] as f64
                / samples_per_generator as f64;
            prior_weighted_accuracy += normalized_prior[model_index] * row_accuracy;
        }
    }

    let named_total = named_trial_count as u64;
    let named_model_accuracy = named_correct as f64 / named_total as f64;
    let named_model_coverage = named_assigned as f64 / named_total as f64;
    let named_model_selective_accuracy =
        (named_assigned > 0).then_some(named_correct as f64 / named_assigned as f64);
    let (open_world_false_confident_assignment_rate, open_world_abstention_rate) =
        if has_unresolved_generator {
            let row_index = generator_ids
                .iter()
                .position(|id| id == OTHER_UNRESOLVED_MODEL_ID)
                .ok_or(ModelRecoveryError::NonFiniteSyntheticCalculation)?;
            let abstentions = confusion_counts[row_index][abstain_column];
            let confident = samples_per_generator as u64 - abstentions;
            (
                Some(confident as f64 / samples_per_generator as f64),
                Some(abstentions as f64 / samples_per_generator as f64),
            )
        } else {
            (None, None)
        };
    let _ = named_abstained;

    Ok(ModelRecoveryReport {
        experiment_id: experiment.experiment_id.clone(),
        prior_scenario_id: prior.scenario_id.clone(),
        model_ids,
        generator_ids,
        confusion_counts,
        samples_per_generator,
        availability_probability: experiment.availability_probability,
        unavailable_observations_by_generator,
        abstention_threshold: minimum_posterior_probability,
        named_model_accuracy,
        prior_weighted_named_model_accuracy: prior_weighted_accuracy,
        named_model_coverage,
        named_model_selective_accuracy,
        open_world_false_confident_assignment_rate,
        open_world_abstention_rate,
    })
}

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn unit_open(&mut self) -> f64 {
        let mantissa = (self.next_u64() >> 11) as f64 + 0.5;
        mantissa / 9_007_199_254_740_992.0
    }

    fn standard_normal(&mut self) -> f64 {
        let radius = (-2.0 * self.unit_open().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.unit_open();
        radius * angle.cos()
    }
}

fn normal_log_density(value: f64, mean: f64, sigma: f64) -> f64 {
    let standardized = (value - mean) / sigma;
    -0.5 * standardized * standardized - sigma.ln() - 0.5 * std::f64::consts::TAU.ln()
}

fn log_sum_exp(values: &[f64]) -> Option<f64> {
    let maximum = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .max_by(f64::total_cmp)?;
    let total = values
        .iter()
        .copied()
        .map(|value| (value - maximum).exp())
        .sum::<f64>();
    (total.is_finite() && total > 0.0).then_some(maximum + total.ln())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_discrimination::GaussianModelPrediction;
    use crate::theory_prediction::PredictionDisposition;

    fn hypothesis(id: &str, byte: u8) -> DiscriminationHypothesis {
        DiscriminationHypothesis {
            model_id: id.to_string(),
            source_revision_sha: (0..40).map(|_| byte as char).collect(),
            specification_reference: format!("synthetic://model/{id}"),
        }
    }

    fn prediction(id: &str, mean: f64, sigma: f64) -> GaussianModelPrediction {
        GaussianModelPrediction {
            model_id: id.to_string(),
            disposition: PredictionDisposition::Quantitative,
            predictive_mean: Some(mean),
            predictive_sigma: Some(sigma),
            derivation_reference: Some(format!("synthetic://prediction/{id}")),
        }
    }

    fn experiment(predictions: Vec<GaussianModelPrediction>) -> CandidateExperiment {
        CandidateExperiment {
            experiment_id: "recovery-fixture".to_string(),
            observable_id: "scalar-observable".to_string(),
            domain_id: "synthetic-domain".to_string(),
            units: "unit".to_string(),
            measurement_noise_model: MeasurementNoiseModel::Gaussian,
            measurement_sigma: Some(0.05),
            noise_calibration: NoiseCalibrationStatus::SyntheticKnownAnswer,
            calibration_reference: Some("synthetic://noise".to_string()),
            availability_probability: 1.0,
            cost: 1.0,
            exclusive_group: None,
            predictions,
        }
    }

    fn prior() -> ModelPriorScenario {
        ModelPriorScenario {
            scenario_id: "equal".to_string(),
            scenario_weight: 1.0,
            probabilities: vec![1.0, 1.0, 1.0],
        }
    }

    #[test]
    fn unavailable_trials_remain_in_denominator_as_explicit_abstentions() {
        let hypotheses = vec![hypothesis("a", b'a'), hypothesis("b", b'b')];
        let mut candidate = experiment(vec![prediction("a", 0.0, 0.0), prediction("b", 5.0, 0.0)]);
        candidate.availability_probability = 0.0;
        let generators = vec![
            RecoveryGenerator {
                generator_id: "a".to_string(),
                predictive_mean: 0.0,
                predictive_sigma: 0.0,
            },
            RecoveryGenerator {
                generator_id: "b".to_string(),
                predictive_mean: 5.0,
                predictive_sigma: 0.0,
            },
            RecoveryGenerator {
                generator_id: OTHER_UNRESOLVED_MODEL_ID.to_string(),
                predictive_mean: 20.0,
                predictive_sigma: 0.0,
            },
        ];
        let report =
            evaluate_model_recovery(&hypotheses, &candidate, &prior(), &generators, 64, 123, 0.8)
                .unwrap();

        assert_eq!(report.availability_probability, 0.0);
        assert_eq!(
            report.unavailable_observations_by_generator,
            vec![64, 64, 64]
        );
        assert_eq!(report.named_model_accuracy, 0.0);
        assert_eq!(report.named_model_coverage, 0.0);
        assert_eq!(report.named_model_selective_accuracy, None);
        assert_eq!(report.open_world_false_confident_assignment_rate, Some(0.0));
        assert_eq!(report.open_world_abstention_rate, Some(1.0));
        assert!(report.confusion_counts.iter().all(|row| row[2] == 64));
    }

    #[test]
    fn fully_available_recovery_has_no_unavailable_trials() {
        let hypotheses = vec![hypothesis("a", b'a'), hypothesis("b", b'b')];
        let candidate = experiment(vec![prediction("a", 0.0, 0.0), prediction("b", 5.0, 0.0)]);
        let generators = vec![
            RecoveryGenerator {
                generator_id: "a".to_string(),
                predictive_mean: 0.0,
                predictive_sigma: 0.0,
            },
            RecoveryGenerator {
                generator_id: "b".to_string(),
                predictive_mean: 5.0,
                predictive_sigma: 0.0,
            },
        ];
        let report =
            evaluate_model_recovery(&hypotheses, &candidate, &prior(), &generators, 32, 456, 0.8)
                .unwrap();

        assert_eq!(report.availability_probability, 1.0);
        assert_eq!(report.unavailable_observations_by_generator, vec![0, 0]);
        assert_eq!(report.named_model_coverage, 1.0);
        assert_eq!(report.named_model_accuracy, 1.0);
    }

    #[test]
    fn separated_generators_are_recovered_on_deterministic_trials() {
        let hypotheses = vec![
            hypothesis("a", b'a'),
            hypothesis("b", b'b'),
            hypothesis("c", b'c'),
        ];
        let experiment = experiment(vec![
            prediction("a", -10.0, 0.0),
            prediction("b", 0.0, 0.0),
            prediction("c", 10.0, 0.0),
        ]);
        let generators = vec![
            RecoveryGenerator {
                generator_id: "a".into(),
                predictive_mean: -10.0,
                predictive_sigma: 0.0,
            },
            RecoveryGenerator {
                generator_id: "b".into(),
                predictive_mean: 0.0,
                predictive_sigma: 0.0,
            },
            RecoveryGenerator {
                generator_id: "c".into(),
                predictive_mean: 10.0,
                predictive_sigma: 0.0,
            },
        ];
        let report = evaluate_model_recovery(
            &hypotheses,
            &experiment,
            &prior(),
            &generators,
            256,
            42,
            0.99,
        )
        .unwrap();

        assert_eq!(report.named_model_accuracy, 1.0);
        assert_eq!(report.named_model_coverage, 1.0);
        assert_eq!(report.named_model_selective_accuracy, Some(1.0));
    }

    #[test]
    fn identical_models_are_abstained_when_posterior_threshold_exceeds_prior() {
        let hypotheses = vec![
            hypothesis("a", b'a'),
            hypothesis("b", b'b'),
            hypothesis("c", b'c'),
        ];
        let experiment = experiment(vec![
            prediction("a", 0.0, 0.1),
            prediction("b", 0.0, 0.1),
            prediction("c", 0.0, 0.1),
        ]);
        let generators = vec![
            RecoveryGenerator {
                generator_id: "a".into(),
                predictive_mean: 0.0,
                predictive_sigma: 0.1,
            },
            RecoveryGenerator {
                generator_id: "b".into(),
                predictive_mean: 0.0,
                predictive_sigma: 0.1,
            },
            RecoveryGenerator {
                generator_id: "c".into(),
                predictive_mean: 0.0,
                predictive_sigma: 0.1,
            },
        ];
        let report =
            evaluate_model_recovery(&hypotheses, &experiment, &prior(), &generators, 32, 7, 0.8)
                .unwrap();

        assert_eq!(report.named_model_accuracy, 0.0);
        assert_eq!(report.named_model_coverage, 0.0);
        assert_eq!(report.named_model_selective_accuracy, None);
        for row in report.confusion_counts {
            assert_eq!(row[3], 32);
        }
    }

    #[test]
    fn identical_likelihoods_expose_prior_dominated_recovery() {
        let hypotheses = vec![
            hypothesis("a", b'a'),
            hypothesis("b", b'b'),
            hypothesis("c", b'c'),
        ];
        let experiment = experiment(vec![
            prediction("a", 0.0, 0.1),
            prediction("b", 0.0, 0.1),
            prediction("c", 0.0, 0.1),
        ]);
        let generators = vec![
            RecoveryGenerator {
                generator_id: "a".into(),
                predictive_mean: 0.0,
                predictive_sigma: 0.1,
            },
            RecoveryGenerator {
                generator_id: "b".into(),
                predictive_mean: 0.0,
                predictive_sigma: 0.1,
            },
            RecoveryGenerator {
                generator_id: "c".into(),
                predictive_mean: 0.0,
                predictive_sigma: 0.1,
            },
        ];
        let skewed_prior = ModelPriorScenario {
            scenario_id: "model-a-dominant".to_string(),
            scenario_weight: 1.0,
            probabilities: vec![8.0, 1.0, 1.0],
        };
        let report = evaluate_model_recovery(
            &hypotheses,
            &experiment,
            &skewed_prior,
            &generators,
            64,
            77,
            0.75,
        )
        .unwrap();

        // The observation has zero ability to distinguish the models. A
        // high posterior threshold still accepts the prior winner, so
        // prior-weighted accuracy is high while equal-generator accuracy is
        // only one third.
        assert!((report.named_model_accuracy - 1.0 / 3.0).abs() < 1.0e-12);
        assert!((report.prior_weighted_named_model_accuracy - 0.8).abs() < 1.0e-12);
        assert_eq!(report.named_model_coverage, 1.0);
        assert!(report.confusion_counts.iter().all(|row| row[0] == 64));
    }

    #[test]
    fn open_world_generator_is_not_an_inference_candidate() {
        let hypotheses = vec![
            hypothesis("a", b'a'),
            hypothesis("b", b'b'),
            hypothesis("c", b'c'),
        ];
        let experiment = experiment(vec![
            prediction("a", -1.0, 0.1),
            prediction("b", 0.0, 0.1),
            prediction("c", 1.0, 0.1),
        ]);
        let mut generators = vec![
            RecoveryGenerator {
                generator_id: "a".into(),
                predictive_mean: -1.0,
                predictive_sigma: 0.1,
            },
            RecoveryGenerator {
                generator_id: "b".into(),
                predictive_mean: 0.0,
                predictive_sigma: 0.1,
            },
            RecoveryGenerator {
                generator_id: "c".into(),
                predictive_mean: 1.0,
                predictive_sigma: 0.1,
            },
        ];
        generators.push(RecoveryGenerator {
            generator_id: OTHER_UNRESOLVED_MODEL_ID.to_string(),
            predictive_mean: 0.5,
            predictive_sigma: 0.1,
        });
        let report = evaluate_model_recovery(
            &hypotheses,
            &experiment,
            &prior(),
            &generators,
            512,
            99,
            0.9,
        )
        .unwrap();

        assert_eq!(
            report.generator_ids.last().unwrap(),
            OTHER_UNRESOLVED_MODEL_ID
        );
        assert!(
            report
                .model_ids
                .iter()
                .all(|id| id != OTHER_UNRESOLVED_MODEL_ID)
        );
        assert!(report.open_world_false_confident_assignment_rate.is_some());
        assert!(report.open_world_abstention_rate.is_some());
    }

    #[test]
    fn same_seed_reproduces_the_confusion_matrix() {
        let hypotheses = vec![
            hypothesis("a", b'a'),
            hypothesis("b", b'b'),
            hypothesis("c", b'c'),
        ];
        let experiment = experiment(vec![
            prediction("a", -0.5, 0.1),
            prediction("b", 0.0, 0.1),
            prediction("c", 0.5, 0.1),
        ]);
        let generators = vec![
            RecoveryGenerator {
                generator_id: "a".into(),
                predictive_mean: -0.5,
                predictive_sigma: 0.1,
            },
            RecoveryGenerator {
                generator_id: "b".into(),
                predictive_mean: 0.0,
                predictive_sigma: 0.1,
            },
            RecoveryGenerator {
                generator_id: "c".into(),
                predictive_mean: 0.5,
                predictive_sigma: 0.1,
            },
        ];

        let first = evaluate_model_recovery(
            &hypotheses,
            &experiment,
            &prior(),
            &generators,
            128,
            1234,
            0.8,
        )
        .unwrap();
        let second = evaluate_model_recovery(
            &hypotheses,
            &experiment,
            &prior(),
            &generators,
            128,
            1234,
            0.8,
        )
        .unwrap();
        assert_eq!(first.confusion_counts, second.confusion_counts);
    }

    #[test]
    fn partial_prediction_and_invalid_recovery_inputs_fail_closed() {
        let hypotheses = vec![
            hypothesis("a", b'a'),
            hypothesis("b", b'b'),
            hypothesis("c", b'c'),
        ];
        let mut experiment = experiment(vec![
            prediction("a", -0.5, 0.1),
            prediction("b", 0.0, 0.1),
            prediction("c", 0.5, 0.1),
        ]);
        experiment.predictions[2].disposition = PredictionDisposition::NotDerived;
        experiment.predictions[2].predictive_mean = None;
        experiment.predictions[2].predictive_sigma = None;
        let generators = vec![
            RecoveryGenerator {
                generator_id: "a".into(),
                predictive_mean: -0.5,
                predictive_sigma: 0.1,
            },
            RecoveryGenerator {
                generator_id: "b".into(),
                predictive_mean: 0.0,
                predictive_sigma: 0.1,
            },
            RecoveryGenerator {
                generator_id: "c".into(),
                predictive_mean: 0.5,
                predictive_sigma: 0.1,
            },
        ];

        assert!(matches!(
            evaluate_model_recovery(&hypotheses, &experiment, &prior(), &generators, 16, 1, 0.8),
            Err(ModelRecoveryError::NotComparable { .. })
        ));
        assert_eq!(
            evaluate_model_recovery(&hypotheses, &experiment, &prior(), &generators, 16, 1, 0.0),
            Err(ModelRecoveryError::InvalidAbstentionThreshold)
        );
    }
}
