// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Comparative abiotic baselines across related environments.
//!
//! This module is a small empirical-Bayes random-intercept reference model inspired by
//! systemic biosignature retrievals. It estimates a shared context/signal slope and
//! partially pooled system-specific intercepts from explicitly qualified abiotic controls.
//! A held-out world is evaluated against the baseline of its own system.
//!
//! This is not a full hierarchical Bayesian retrieval, a planetary evolution model, or a
//! life detector. A comparative anomaly is only a model-relative outlier. It is not proof
//! of biology, life, or intelligence. Poor support or context extrapolation must remain
//! an explicit inadequacy/abstention outcome.

use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

/// Whether an observation may contribute to an abiotic reference baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationRole {
    /// Independent control with a separately recorded qualification reference.
    QualifiedAbioticControl,
    /// Candidate to be tested against a fitted baseline.
    Candidate,
    /// Unresolved observation; never silently included in the baseline fit.
    Unresolved,
}

/// One sample-level observation in a related-environment/system comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparativeObservation {
    pub observation_id: String,
    pub system_id: String,
    /// Physical independence unit (e.g. world, site, or independently prepared specimen).
    pub independence_group: String,
    /// One declared environmental/context covariate. It must be comparable within the
    /// fitted benchmark and is not inferred from the target signal.
    pub context: f64,
    /// Observed signal in the benchmark's declared units.
    pub signal: f64,
    /// One-standard-deviation measurement uncertainty in signal units.
    pub measurement_stddev: f64,
    pub role: ObservationRole,
    /// Required for QualifiedAbioticControl; DOI, source artifact, or independent
    /// qualification receipt. This field records provenance, not infallible ground truth.
    pub qualification_evidence_ref: Option<String>,
}

/// Fitted system-specific baseline, including partial-pooling diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SystemBaseline {
    pub qualified_control_count: usize,
    pub context_min: f64,
    pub context_max: f64,
    pub observed_mean_residual: f64,
    pub posterior_intercept: f64,
    pub posterior_intercept_variance: f64,
}

/// Diagnostics for the common context/signal relationship and system-level variation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ComparativeBaselineDiagnostics {
    pub qualified_control_count: usize,
    pub system_count: usize,
    pub common_context_slope: f64,
    pub slope_standard_error: f64,
    pub global_intercept: f64,
    pub within_system_variance: f64,
    pub between_system_variance: f64,
}

/// Empirical-Bayes comparative baseline using a common slope and partially pooled
/// system-specific intercepts.
///
/// The model is fitted only from observations marked QualifiedAbioticControl. Candidate
/// and Unresolved observations are excluded even when supplied in the same input slice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparativeBaselineModel {
    diagnostics: ComparativeBaselineDiagnostics,
    systems: BTreeMap<String, SystemBaseline>,
}

/// Model-relative disposition. Neither value is a biological/life verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparativeDisposition {
    ConsistentWithQualifiedAbioticBaseline,
    ComparativeAnomaly,
}

/// Evaluation of one held-out candidate against its own system's baseline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparativeEvaluation {
    pub observation_id: String,
    pub system_id: String,
    pub predicted_mean: f64,
    pub predictive_stddev: f64,
    pub residual: f64,
    pub standardized_residual: f64,
    pub gaussian_log_predictive_density: f64,
    pub anomaly_threshold: f64,
    pub disposition: ComparativeDisposition,
}

/// Invalid input, insufficient controls, or an unsupported extrapolation.
#[derive(Debug, Clone, PartialEq)]
pub enum ComparativeBaselineError {
    EmptyObservations,
    EmptyIdentifier { field: &'static str },
    DuplicateObservationId(String),
    DuplicateControlIndependenceGroup(String),
    MissingQualificationEvidence { observation_id: String },
    InvalidValue { observation_id: String, field: &'static str, value: f64 },
    TooFewSystems { observed: usize },
    TooFewControls { system_id: String, observed: usize },
    InsufficientContextVariation,
    UncalibratedSystem(String),
    ContextOutsideControlRange {
        system_id: String,
        context: f64,
        minimum: f64,
        maximum: f64,
    },
    InvalidAnomalyThreshold(f64),
    ControlCannotBeEvaluated(String),
    DegeneratePredictiveVariance(String),
    NonFiniteFit,
}

impl fmt::Display for ComparativeBaselineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyObservations => write!(f, "comparative baseline requires observations"),
            Self::EmptyIdentifier { field } => write!(f, "{field} must not be empty"),
            Self::DuplicateObservationId(id) => {
                write!(f, "duplicate observation_id: {id}")
            }
            Self::DuplicateControlIndependenceGroup(id) => {
                write!(f, "qualified controls share independence_group: {id}")
            }
            Self::MissingQualificationEvidence { observation_id } => write!(
                f,
                "qualified abiotic control {observation_id} requires qualification evidence"
            ),
            Self::InvalidValue { observation_id, field, value } => write!(
                f,
                "observation {observation_id} has invalid {field}={value}"
            ),
            Self::TooFewSystems { observed } => write!(
                f,
                "partial-pooling baseline requires at least two systems, got {observed}"
            ),
            Self::TooFewControls { system_id, observed } => write!(
                f,
                "system {system_id} requires at least two independent qualified controls, got {observed}"
            ),
            Self::InsufficientContextVariation => write!(
                f,
                "qualified controls do not span enough context variation to estimate a common slope"
            ),
            Self::UncalibratedSystem(system_id) => {
                write!(f, "no qualified abiotic baseline for system {system_id}")
            }
            Self::ContextOutsideControlRange {
                system_id,
                context,
                minimum,
                maximum,
            } => write!(
                f,
                "candidate context {context} for system {system_id} is outside qualified control range [{minimum}, {maximum}]"
            ),
            Self::InvalidAnomalyThreshold(value) => {
                write!(f, "anomaly threshold must be finite and > 0, got {value}")
            }
            Self::ControlCannotBeEvaluated(id) => write!(
                f,
                "qualified control {id} cannot be evaluated as a held-out candidate"
            ),
            Self::DegeneratePredictiveVariance(id) => write!(
                f,
                "predictive variance is zero or non-finite for candidate {id}"
            ),
            Self::NonFiniteFit => write!(f, "comparative baseline fit produced non-finite values"),
        }
    }
}

impl Error for ComparativeBaselineError {}

impl ComparativeBaselineModel {
    /// Fit a common within-system context slope and empirical-Bayes system intercepts.
    ///
    /// The model uses a normal random-intercept approximation. The system intercept prior
    /// is centered on the pooled intercept; its variance is estimated by the method of
    /// moments after subtracting average sampling variance. This is an interpretable
    /// reference/ablation, not a substitute for full HBM posterior sampling.
    pub fn fit(
        observations: &[ComparativeObservation],
    ) -> Result<Self, ComparativeBaselineError> {
        if observations.is_empty() {
            return Err(ComparativeBaselineError::EmptyObservations);
        }

        let mut seen_observations = HashSet::new();
        let mut seen_control_groups = HashSet::new();
        let mut by_system: BTreeMap<String, Vec<&ComparativeObservation>> = BTreeMap::new();

        for observation in observations {
            for (field, value) in [
                ("observation_id", observation.observation_id.trim()),
                ("system_id", observation.system_id.trim()),
                ("independence_group", observation.independence_group.trim()),
            ] {
                if value.is_empty() {
                    return Err(ComparativeBaselineError::EmptyIdentifier { field });
                }
            }
            if !seen_observations.insert(observation.observation_id.as_str()) {
                return Err(ComparativeBaselineError::DuplicateObservationId(
                    observation.observation_id.clone(),
                ));
            }
            for (field, value) in [
                ("context", observation.context),
                ("signal", observation.signal),
                ("measurement_stddev", observation.measurement_stddev),
            ] {
                if !value.is_finite()
                    || (field == "measurement_stddev" && value < 0.0)
                {
                    return Err(ComparativeBaselineError::InvalidValue {
                        observation_id: observation.observation_id.clone(),
                        field,
                        value,
                    });
                }
            }

            if observation.role == ObservationRole::QualifiedAbioticControl {
                if observation
                    .qualification_evidence_ref
                    .as_ref()
                    .is_none_or(|reference| reference.trim().is_empty())
                {
                    return Err(ComparativeBaselineError::MissingQualificationEvidence {
                        observation_id: observation.observation_id.clone(),
                    });
                }
                if !seen_control_groups.insert(observation.independence_group.as_str()) {
                    return Err(
                        ComparativeBaselineError::DuplicateControlIndependenceGroup(
                            observation.independence_group.clone(),
                        ),
                    );
                }
                by_system
                    .entry(observation.system_id.clone())
                    .or_default()
                    .push(observation);
            }
        }

        if by_system.len() < 2 {
            return Err(ComparativeBaselineError::TooFewSystems {
                observed: by_system.len(),
            });
        }
        for (system_id, controls) in &by_system {
            if controls.len() < 2 {
                return Err(ComparativeBaselineError::TooFewControls {
                    system_id: system_id.clone(),
                    observed: controls.len(),
                });
            }
        }

        // Fixed-effect within-system centering estimates the shared context slope without
        // confounding it with a system-specific abiotic intercept.
        let mut sxx = 0.0;
        let mut sxy = 0.0;
        let mut total_controls = 0usize;
        for controls in by_system.values() {
            let count = controls.len() as f64;
            let mean_x = controls.iter().map(|item| item.context).sum::<f64>() / count;
            let mean_y = controls.iter().map(|item| item.signal).sum::<f64>() / count;
            for item in controls {
                let centered_x = item.context - mean_x;
                let centered_y = item.signal - mean_y;
                sxx += centered_x * centered_x;
                sxy += centered_x * centered_y;
                total_controls += 1;
            }
        }
        if !sxx.is_finite() || sxx <= f64::EPSILON {
            return Err(ComparativeBaselineError::InsufficientContextVariation);
        }
        let common_context_slope = sxy / sxx;

        let mut raw_systems = Vec::with_capacity(by_system.len());
        let mut within_sse = 0.0;
        let mut total_measurement_variance = 0.0;

        for (system_id, controls) in &by_system {
            let count = controls.len();
            let residuals: Vec<f64> = controls
                .iter()
                .map(|item| item.signal - common_context_slope * item.context)
                .collect();
            let mean_residual = residuals.iter().sum::<f64>() / count as f64;

            for (item, residual) in controls.iter().zip(&residuals) {
                let delta = *residual - mean_residual;
                within_sse += delta * delta;
                total_measurement_variance += item.measurement_stddev.powi(2);
            }
            let context_min = controls
                .iter()
                .map(|item| item.context)
                .fold(f64::INFINITY, f64::min);
            let context_max = controls
                .iter()
                .map(|item| item.context)
                .fold(f64::NEG_INFINITY, f64::max);

            raw_systems.push((
                system_id.clone(),
                count,
                mean_residual,
                context_min,
                context_max,
                controls
                    .iter()
                    .map(|item| item.measurement_stddev.powi(2))
                    .sum::<f64>(),
            ));
        }

        // One fitted common slope plus one fitted mean per system consume degrees
        // of freedom; measurement uncertainty is then subtracted from the residual
        // variance estimate below.
        let degrees_of_freedom = total_controls.saturating_sub(by_system.len() + 1);
        let observed_within_variance = if degrees_of_freedom > 0 {
            within_sse / degrees_of_freedom as f64
        } else {
            0.0
        };
        let mean_control_measurement_variance =
            total_measurement_variance / total_controls as f64;
        let within_system_variance =
            (observed_within_variance - mean_control_measurement_variance).max(0.0);
        // Each system contributes one draw from the system-level random-intercept
        // distribution. Use the unweighted mean of system means for the prior center so
        // a system with more sampled worlds does not dominate the between-system estimate.
        let group_means = raw_systems
            .iter()
            .map(|entry| entry.2)
            .collect::<Vec<_>>();
        let global_intercept = group_means.iter().sum::<f64>() / group_means.len() as f64;
        let between_observed_variance = sample_variance(&group_means, global_intercept);
        let average_group_sampling_variance = raw_systems
            .iter()
            .map(|entry| {
                (within_system_variance * entry.1 as f64 + entry.5)
                    / (entry.1 as f64).powi(2)
            })
            .sum::<f64>()
            / raw_systems.len() as f64;
        let between_system_variance =
            (between_observed_variance - average_group_sampling_variance).max(0.0);

        let slope_variance =
            (within_system_variance + mean_control_measurement_variance) / sxx;
        let slope_standard_error = slope_variance.sqrt();

        let mut systems = BTreeMap::new();
        for (
            system_id,
            count,
            observed_mean_residual,
            context_min,
            context_max,
            measurement_variance_sum,
        ) in raw_systems
        {
            let n = count as f64;
            let sampling_variance =
                (within_system_variance * n + measurement_variance_sum) / n.powi(2);
            let denominator = between_system_variance + sampling_variance;
            let pooling_weight = if denominator > 0.0 {
                between_system_variance / denominator
            } else {
                0.0
            };
            let posterior_intercept = global_intercept
                + pooling_weight * (observed_mean_residual - global_intercept);
            let posterior_intercept_variance = if denominator > 0.0 {
                between_system_variance * sampling_variance / denominator
            } else {
                0.0
            };

            systems.insert(
                system_id,
                SystemBaseline {
                    qualified_control_count: count,
                    context_min,
                    context_max,
                    observed_mean_residual,
                    posterior_intercept,
                    posterior_intercept_variance,
                },
            );
        }

        let diagnostics = ComparativeBaselineDiagnostics {
            qualified_control_count: total_controls,
            system_count: systems.len(),
            common_context_slope,
            slope_standard_error,
            global_intercept,
            within_system_variance,
            between_system_variance,
        };
        if ![
            diagnostics.common_context_slope,
            diagnostics.slope_standard_error,
            diagnostics.global_intercept,
            diagnostics.within_system_variance,
            diagnostics.between_system_variance,
        ]
        .iter()
        .all(|value| value.is_finite())
            || systems.values().any(|system| {
                !system.posterior_intercept.is_finite()
                    || !system.posterior_intercept_variance.is_finite()
            })
        {
            return Err(ComparativeBaselineError::NonFiniteFit);
        }

        Ok(Self {
            diagnostics,
            systems,
        })
    }

    pub const fn diagnostics(&self) -> ComparativeBaselineDiagnostics {
        self.diagnostics
    }

    pub fn system_baseline(&self, system_id: &str) -> Option<SystemBaseline> {
        self.systems.get(system_id).copied()
    }

    /// Evaluate a held-out candidate against its system-conditioned predictive baseline.
    ///
    /// Context extrapolation is rejected rather than converted into a confident anomaly.
    pub fn evaluate(
        &self,
        candidate: &ComparativeObservation,
        anomaly_threshold: f64,
    ) -> Result<ComparativeEvaluation, ComparativeBaselineError> {
        if !anomaly_threshold.is_finite() || anomaly_threshold <= 0.0 {
            return Err(ComparativeBaselineError::InvalidAnomalyThreshold(
                anomaly_threshold,
            ));
        }
        if candidate.role == ObservationRole::QualifiedAbioticControl {
            return Err(ComparativeBaselineError::ControlCannotBeEvaluated(
                candidate.observation_id.clone(),
            ));
        }
        for (field, value) in [
            ("context", candidate.context),
            ("signal", candidate.signal),
            ("measurement_stddev", candidate.measurement_stddev),
        ] {
            if !value.is_finite()
                || (field == "measurement_stddev" && value < 0.0)
            {
                return Err(ComparativeBaselineError::InvalidValue {
                    observation_id: candidate.observation_id.clone(),
                    field,
                    value,
                });
            }
        }

        let system = self
            .systems
            .get(&candidate.system_id)
            .ok_or_else(|| {
                ComparativeBaselineError::UncalibratedSystem(candidate.system_id.clone())
            })?;
        if candidate.context < system.context_min || candidate.context > system.context_max {
            return Err(ComparativeBaselineError::ContextOutsideControlRange {
                system_id: candidate.system_id.clone(),
                context: candidate.context,
                minimum: system.context_min,
                maximum: system.context_max,
            });
        }

        let predicted_mean = self.diagnostics.common_context_slope * candidate.context
            + system.posterior_intercept;
        let predictive_variance = self.diagnostics.within_system_variance
            + system.posterior_intercept_variance
            + candidate.measurement_stddev.powi(2)
            + (self.diagnostics.slope_standard_error * candidate.context).powi(2);
        if !predictive_variance.is_finite() || predictive_variance <= 0.0 {
            return Err(ComparativeBaselineError::DegeneratePredictiveVariance(
                candidate.observation_id.clone(),
            ));
        }
        let predictive_stddev = predictive_variance.sqrt();
        let residual = candidate.signal - predicted_mean;
        let standardized_residual = residual / predictive_stddev;
        let gaussian_log_predictive_density = -0.5
            * ((2.0 * std::f64::consts::PI * predictive_variance).ln()
                + residual.powi(2) / predictive_variance);
        if ![
            predicted_mean,
            predictive_stddev,
            residual,
            standardized_residual,
            gaussian_log_predictive_density,
        ]
        .iter()
        .all(|value| value.is_finite())
        {
            return Err(ComparativeBaselineError::NonFiniteFit);
        }

        let disposition = if standardized_residual.abs() > anomaly_threshold {
            ComparativeDisposition::ComparativeAnomaly
        } else {
            ComparativeDisposition::ConsistentWithQualifiedAbioticBaseline
        };

        Ok(ComparativeEvaluation {
            observation_id: candidate.observation_id.clone(),
            system_id: candidate.system_id.clone(),
            predicted_mean,
            predictive_stddev,
            residual,
            standardized_residual,
            gaussian_log_predictive_density,
            anomaly_threshold,
            disposition,
        })
    }
}

fn sample_variance(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn control(
        id: &str,
        system: &str,
        context: f64,
        signal: f64,
        independence_group: &str,
    ) -> ComparativeObservation {
        ComparativeObservation {
            observation_id: id.to_owned(),
            system_id: system.to_owned(),
            independence_group: independence_group.to_owned(),
            context,
            signal,
            measurement_stddev: 0.05,
            role: ObservationRole::QualifiedAbioticControl,
            qualification_evidence_ref: Some(format!("qualification:{id}")),
        }
    }

    fn target(
        id: &str,
        system: &str,
        context: f64,
        signal: f64,
    ) -> ComparativeObservation {
        ComparativeObservation {
            observation_id: id.to_owned(),
            system_id: system.to_owned(),
            independence_group: format!("target:{id}"),
            context,
            signal,
            measurement_stddev: 0.05,
            role: ObservationRole::Candidate,
            qualification_evidence_ref: None,
        }
    }

    fn controls() -> Vec<ComparativeObservation> {
        vec![
            control("a0", "system-a", 0.0, 5.0, "world-a0"),
            control("a1", "system-a", 1.0, 7.0, "world-a1"),
            control("a2", "system-a", 2.0, 9.0, "world-a2"),
            control("b0", "system-b", 0.0, -3.0, "world-b0"),
            control("b1", "system-b", 1.0, -1.0, "world-b1"),
            control("b2", "system-b", 2.0, 1.0, "world-b2"),
        ]
    }

    #[test]
    fn system_conditioning_removes_false_anomalies_from_shared_system_offsets() {
        let model = ComparativeBaselineModel::fit(&controls()).unwrap();
        let held_out = target("b-target", "system-b", 1.5, 0.0);
        let evaluation = model.evaluate(&held_out, 3.0).unwrap();

        assert!((model.diagnostics().common_context_slope - 2.0).abs() < 1.0e-12);
        assert!((evaluation.predicted_mean - 0.0).abs() < 1.0e-3);
        assert!(evaluation.standardized_residual.abs() < 1.0);
        assert_eq!(
            evaluation.disposition,
            ComparativeDisposition::ConsistentWithQualifiedAbioticBaseline
        );

        // A pooled global intercept is +1.0, which would make this system-b target
        // look anomalous despite its exact agreement with its own system's abiotic trend.
        let global_prediction =
            model.diagnostics().common_context_slope * held_out.context
                + model.diagnostics().global_intercept;
        assert!((global_prediction - 4.0).abs() < 1.0e-12);
    }

    #[test]
    fn system_level_prior_center_is_not_dominated_by_larger_control_groups() {
        let observations = vec![
            control("a0", "system-a", 0.0, 5.0, "world-a0"),
            control("a1", "system-a", 1.0, 7.0, "world-a1"),
            control("a2", "system-a", 2.0, 9.0, "world-a2"),
            control("b0", "system-b", 0.0, -3.0, "world-b0"),
            control("b2", "system-b", 2.0, 1.0, "world-b2"),
        ];
        let model = ComparativeBaselineModel::fit(&observations).unwrap();

        assert!((model.diagnostics().global_intercept - 1.0).abs() < 1.0e-12);
        let system_b = target("b-target-unequal", "system-b", 1.0, -1.0);
        let evaluation = model.evaluate(&system_b, 3.0).unwrap();
        assert!(evaluation.standardized_residual.abs() < 1.0);
    }

    #[test]
    fn comparative_anomaly_is_a_model_relative_outlier_not_a_life_verdict() {
        let model = ComparativeBaselineModel::fit(&controls()).unwrap();
        let held_out = target("b-outlier", "system-b", 1.5, 2.0);
        let evaluation = model.evaluate(&held_out, 3.0).unwrap();

        assert!(evaluation.standardized_residual > 3.0);
        assert_eq!(
            evaluation.disposition,
            ComparativeDisposition::ComparativeAnomaly
        );
    }

    #[test]
    fn candidate_observations_do_not_leak_into_the_abiotic_fit() {
        let baseline = controls();
        let model_a = ComparativeBaselineModel::fit(&baseline).unwrap();

        let mut with_candidate = baseline;
        with_candidate.push(target("candidate", "system-b", 1.5, 1000.0));
        let model_b = ComparativeBaselineModel::fit(&with_candidate).unwrap();

        assert_eq!(model_a.diagnostics(), model_b.diagnostics());
        assert_eq!(
            model_a.system_baseline("system-b"),
            model_b.system_baseline("system-b")
        );
    }

    #[test]
    fn context_extrapolation_is_rejected_instead_of_scored_as_anomaly() {
        let model = ComparativeBaselineModel::fit(&controls()).unwrap();
        let held_out = target("out-of-context", "system-b", 3.0, 100.0);
        assert!(matches!(
            model.evaluate(&held_out, 3.0),
            Err(ComparativeBaselineError::ContextOutsideControlRange { .. })
        ));
    }

    #[test]
    fn an_uncalibrated_system_requires_abstention() {
        let model = ComparativeBaselineModel::fit(&controls()).unwrap();
        let held_out = target("unknown-system", "system-c", 1.0, 100.0);
        assert!(matches!(
            model.evaluate(&held_out, 3.0),
            Err(ComparativeBaselineError::UncalibratedSystem(_))
        ));
    }

    #[test]
    fn unresolved_observations_are_not_used_as_baseline_controls() {
        let mut observations = controls();
        observations.push(ComparativeObservation {
            observation_id: "unresolved-1".to_owned(),
            system_id: "system-a".to_owned(),
            independence_group: "unknown-world".to_owned(),
            context: 1.5,
            signal: 1000.0,
            measurement_stddev: 0.05,
            role: ObservationRole::Unresolved,
            qualification_evidence_ref: None,
        });
        let model = ComparativeBaselineModel::fit(&observations).unwrap();
        assert_eq!(model.diagnostics().qualified_control_count, 6);
    }

    #[test]
    fn abiotic_controls_require_qualification_and_independent_units() {
        let mut observations = controls();
        observations[0].qualification_evidence_ref = None;
        assert!(matches!(
            ComparativeBaselineModel::fit(&observations),
            Err(ComparativeBaselineError::MissingQualificationEvidence { .. })
        ));

        let mut observations = controls();
        observations[1].independence_group = observations[0].independence_group.clone();
        assert!(matches!(
            ComparativeBaselineModel::fit(&observations),
            Err(ComparativeBaselineError::DuplicateControlIndependenceGroup(_))
        ));
    }
}
