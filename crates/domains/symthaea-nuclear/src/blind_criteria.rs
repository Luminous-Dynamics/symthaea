// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Preregistered metric bounds for nuclear blind-validation reports.
//!
//! This module deliberately does not embed a universal "good enough" threshold.
//! Scientific programs must choose and register their criteria before examining
//! the candidate blind result. The `registration_evidence_id` is intended to
//! point at an immutable/timestamped evidence record outside this crate.
//!
//! The API can enforce that a report is evaluated against the exact declared
//! protocol/model/bounds. It **cannot by itself prove chronology**—that the
//! external registration really preceded observation—so the resulting outcome
//! is named `MeetsAllDeclaredBounds`, not "scientifically qualified".

use crate::blind_validation::{BlindHoldout, BlindValidationReport};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BlindMetricBounds {
    pub max_rms_mev: f64,
    pub max_mae_mev: f64,
    pub max_abs_bias_mev: f64,
    pub max_max_abs_error_mev: f64,
}

impl BlindMetricBounds {
    fn validate(self) -> Result<(), BlindCriteriaError> {
        for value in [
            self.max_rms_mev,
            self.max_mae_mev,
            self.max_abs_bias_mev,
            self.max_max_abs_error_mev,
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(BlindCriteriaError::InvalidMetricBound);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelativeRmsRequirement {
    pub baseline_model_id: String,
    /// Required fractional RMS reduction relative to the baseline.
    /// 0.10 means candidate RMS must be at least 10% lower.
    pub min_improvement_fraction: f64,
}

impl RelativeRmsRequirement {
    fn validate(&self) -> Result<(), BlindCriteriaError> {
        if self.baseline_model_id.trim().is_empty() {
            return Err(BlindCriteriaError::EmptyBaselineModelId);
        }
        if !self.min_improvement_fraction.is_finite()
            || self.min_improvement_fraction < 0.0
            || self.min_improvement_fraction >= 1.0
        {
            return Err(BlindCriteriaError::InvalidImprovementFraction);
        }
        Ok(())
    }
}

/// Criteria whose identity is expected to be frozen outside this crate before
/// blind adjudication begins.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreregisteredBlindCriteria {
    registration_evidence_id: String,
    protocol: BlindHoldout,
    candidate_model_id: String,
    bounds: BlindMetricBounds,
    relative_rms: Option<RelativeRmsRequirement>,
}

impl PreregisteredBlindCriteria {
    pub fn new(
        registration_evidence_id: impl Into<String>,
        protocol: BlindHoldout,
        candidate_model_id: impl Into<String>,
        bounds: BlindMetricBounds,
        relative_rms: Option<RelativeRmsRequirement>,
    ) -> Result<Self, BlindCriteriaError> {
        let registration_evidence_id = registration_evidence_id.into();
        let candidate_model_id = candidate_model_id.into();
        if registration_evidence_id.trim().is_empty() {
            return Err(BlindCriteriaError::EmptyRegistrationEvidenceId);
        }
        if candidate_model_id.trim().is_empty() {
            return Err(BlindCriteriaError::EmptyCandidateModelId);
        }
        bounds.validate()?;
        if let Some(requirement) = &relative_rms {
            requirement.validate()?;
        }
        Ok(Self {
            registration_evidence_id,
            protocol,
            candidate_model_id,
            bounds,
            relative_rms,
        })
    }

    pub fn registration_evidence_id(&self) -> &str {
        &self.registration_evidence_id
    }

    pub fn protocol(&self) -> BlindHoldout {
        self.protocol
    }

    pub fn candidate_model_id(&self) -> &str {
        &self.candidate_model_id
    }

    pub fn bounds(&self) -> BlindMetricBounds {
        self.bounds
    }

    pub fn relative_rms(&self) -> Option<&RelativeRmsRequirement> {
        self.relative_rms.as_ref()
    }

    pub fn evaluate(
        &self,
        candidate: &BlindValidationReport,
        baseline: Option<&BlindValidationReport>,
    ) -> Result<DeclaredBoundsEvaluation, BlindCriteriaError> {
        if candidate.protocol != self.protocol {
            return Err(BlindCriteriaError::CandidateProtocolMismatch);
        }
        if candidate.method != self.candidate_model_id {
            return Err(BlindCriteriaError::CandidateModelMismatch);
        }
        validate_report_metrics(candidate)?;

        let mut checks = vec![
            MetricBoundCheck::upper(
                "rms_mev",
                candidate.rms_mev,
                self.bounds.max_rms_mev,
            ),
            MetricBoundCheck::upper(
                "mae_mev",
                candidate.mae_mev,
                self.bounds.max_mae_mev,
            ),
            MetricBoundCheck::upper(
                "abs_bias_mev",
                candidate.bias_mev.abs(),
                self.bounds.max_abs_bias_mev,
            ),
            MetricBoundCheck::upper(
                "max_abs_error_mev",
                candidate.max_abs_error_mev,
                self.bounds.max_max_abs_error_mev,
            ),
        ];

        let baseline_report = if let Some(requirement) = &self.relative_rms {
            let baseline = baseline.ok_or(BlindCriteriaError::MissingRequiredBaseline)?;
            if baseline.protocol != self.protocol {
                return Err(BlindCriteriaError::BaselineProtocolMismatch);
            }
            if baseline.method != requirement.baseline_model_id {
                return Err(BlindCriteriaError::BaselineModelMismatch);
            }
            if baseline.n_holdout != candidate.n_holdout {
                return Err(BlindCriteriaError::BaselineHoldoutCountMismatch);
            }
            validate_report_metrics(baseline)?;
            if baseline.rms_mev <= 0.0 {
                return Err(BlindCriteriaError::NonPositiveBaselineRms);
            }
            let observed_improvement = (baseline.rms_mev - candidate.rms_mev) / baseline.rms_mev;
            checks.push(MetricBoundCheck::lower(
                "rms_improvement_fraction",
                observed_improvement,
                requirement.min_improvement_fraction,
            ));
            Some(baseline.clone())
        } else {
            None
        };

        let outcome = if checks.iter().all(|check| check.passed) {
            DeclaredBoundsOutcome::MeetsAllDeclaredBounds
        } else {
            DeclaredBoundsOutcome::ViolatesAtLeastOneDeclaredBound
        };

        Ok(DeclaredBoundsEvaluation {
            registration_evidence_id: self.registration_evidence_id.clone(),
            protocol: self.protocol,
            candidate: candidate.clone(),
            baseline: baseline_report,
            checks,
            outcome,
            chronology_note: "registration_evidence_id must be verified externally as immutable/timestamped before blind observation; this crate does not prove chronology".to_string(),
        })
    }
}

fn validate_report_metrics(report: &BlindValidationReport) -> Result<(), BlindCriteriaError> {
    if report.n_holdout == 0 || report.n_training == 0 {
        return Err(BlindCriteriaError::EmptyReportPartition);
    }
    for value in [
        report.bias_mev,
        report.mae_mev,
        report.rms_mev,
        report.max_abs_error_mev,
    ] {
        if !value.is_finite() {
            return Err(BlindCriteriaError::NonFiniteReportMetric);
        }
    }
    if report.mae_mev < 0.0 || report.rms_mev < 0.0 || report.max_abs_error_mev < 0.0 {
        return Err(BlindCriteriaError::NegativeErrorMetric);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricComparator {
    LessThanOrEqual,
    GreaterThanOrEqual,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricBoundCheck {
    pub metric: String,
    pub observed: f64,
    pub declared_bound: f64,
    pub comparator: MetricComparator,
    pub passed: bool,
}

impl MetricBoundCheck {
    fn upper(metric: impl Into<String>, observed: f64, bound: f64) -> Self {
        Self {
            metric: metric.into(),
            observed,
            declared_bound: bound,
            comparator: MetricComparator::LessThanOrEqual,
            passed: observed <= bound,
        }
    }

    fn lower(metric: impl Into<String>, observed: f64, bound: f64) -> Self {
        Self {
            metric: metric.into(),
            observed,
            declared_bound: bound,
            comparator: MetricComparator::GreaterThanOrEqual,
            passed: observed >= bound,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeclaredBoundsOutcome {
    MeetsAllDeclaredBounds,
    ViolatesAtLeastOneDeclaredBound,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeclaredBoundsEvaluation {
    pub registration_evidence_id: String,
    pub protocol: BlindHoldout,
    pub candidate: BlindValidationReport,
    pub baseline: Option<BlindValidationReport>,
    pub checks: Vec<MetricBoundCheck>,
    pub outcome: DeclaredBoundsOutcome,
    /// Explicitly records that temporal preregistration must be established by
    /// the referenced external evidence, not inferred from this local object.
    pub chronology_note: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindCriteriaError {
    EmptyRegistrationEvidenceId,
    EmptyCandidateModelId,
    EmptyBaselineModelId,
    InvalidMetricBound,
    InvalidImprovementFraction,
    CandidateProtocolMismatch,
    CandidateModelMismatch,
    MissingRequiredBaseline,
    BaselineProtocolMismatch,
    BaselineModelMismatch,
    BaselineHoldoutCountMismatch,
    NonPositiveBaselineRms,
    EmptyReportPartition,
    NonFiniteReportMetric,
    NegativeErrorMetric,
}

impl fmt::Display for BlindCriteriaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyRegistrationEvidenceId => "registration evidence id must not be empty",
            Self::EmptyCandidateModelId => "candidate model id must not be empty",
            Self::EmptyBaselineModelId => "baseline model id must not be empty",
            Self::InvalidMetricBound => "metric bounds must be finite and non-negative",
            Self::InvalidImprovementFraction => {
                "minimum RMS improvement fraction must be finite and in [0,1)"
            }
            Self::CandidateProtocolMismatch => "candidate report protocol differs from registration",
            Self::CandidateModelMismatch => "candidate report model differs from registration",
            Self::MissingRequiredBaseline => "registered relative RMS criterion requires a baseline report",
            Self::BaselineProtocolMismatch => "baseline report protocol differs from registration",
            Self::BaselineModelMismatch => "baseline report model differs from registration",
            Self::BaselineHoldoutCountMismatch => {
                "baseline and candidate were not adjudicated on equal holdout counts"
            }
            Self::NonPositiveBaselineRms => "relative RMS improvement requires a positive baseline RMS",
            Self::EmptyReportPartition => "blind report has an empty training or holdout partition",
            Self::NonFiniteReportMetric => "blind report contains a non-finite metric",
            Self::NegativeErrorMetric => "blind report contains a negative error metric",
        };
        write!(f, "{message}")
    }
}

impl std::error::Error for BlindCriteriaError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(method: &str, rms: f64, mae: f64, bias: f64, max_error: f64) -> BlindValidationReport {
        BlindValidationReport {
            protocol: BlindHoldout::ProtonFrontier { train_z_max: 82 },
            protocol_label: "proton-frontier:z>82".to_string(),
            method: method.to_string(),
            n_training: 100,
            n_holdout: 20,
            bias_mev: bias,
            mae_mev: mae,
            rms_mev: rms,
            max_abs_error_mev: max_error,
        }
    }

    fn criteria() -> PreregisteredBlindCriteria {
        PreregisteredBlindCriteria::new(
            "evidence:pre-registration:example",
            BlindHoldout::ProtonFrontier { train_z_max: 82 },
            "candidate",
            BlindMetricBounds {
                max_rms_mev: 1.0,
                max_mae_mev: 0.8,
                max_abs_bias_mev: 0.25,
                max_max_abs_error_mev: 3.0,
            },
            Some(RelativeRmsRequirement {
                baseline_model_id: "DZ10".to_string(),
                min_improvement_fraction: 0.10,
            }),
        )
        .unwrap()
    }

    #[test]
    fn declared_bounds_can_be_met_without_becoming_global_qualification() {
        let candidate = report("candidate", 0.8, 0.6, 0.1, 2.0);
        let baseline = report("DZ10", 1.0, 0.8, 0.0, 2.5);
        let evaluation = criteria().evaluate(&candidate, Some(&baseline)).unwrap();
        assert_eq!(
            evaluation.outcome,
            DeclaredBoundsOutcome::MeetsAllDeclaredBounds
        );
        assert!(evaluation.checks.iter().all(|check| check.passed));
        assert!(evaluation.chronology_note.contains("does not prove chronology"));
    }

    #[test]
    fn one_failed_bound_fails_the_declared_bundle() {
        let candidate = report("candidate", 0.8, 0.6, 0.30, 2.0);
        let baseline = report("DZ10", 1.0, 0.8, 0.0, 2.5);
        let evaluation = criteria().evaluate(&candidate, Some(&baseline)).unwrap();
        assert_eq!(
            evaluation.outcome,
            DeclaredBoundsOutcome::ViolatesAtLeastOneDeclaredBound
        );
        assert!(evaluation.checks.iter().any(|check| !check.passed));
    }

    #[test]
    fn wrong_model_or_missing_baseline_fails_closed() {
        let wrong = report("other", 0.8, 0.6, 0.1, 2.0);
        assert_eq!(
            criteria().evaluate(&wrong, None).unwrap_err(),
            BlindCriteriaError::CandidateModelMismatch
        );

        let candidate = report("candidate", 0.8, 0.6, 0.1, 2.0);
        assert_eq!(
            criteria().evaluate(&candidate, None).unwrap_err(),
            BlindCriteriaError::MissingRequiredBaseline
        );
    }

    #[test]
    fn invalid_or_unidentified_registration_fails_closed() {
        assert_eq!(
            PreregisteredBlindCriteria::new(
                "",
                BlindHoldout::ProtonFrontier { train_z_max: 82 },
                "candidate",
                BlindMetricBounds {
                    max_rms_mev: 1.0,
                    max_mae_mev: 1.0,
                    max_abs_bias_mev: 1.0,
                    max_max_abs_error_mev: 2.0,
                },
                None,
            )
            .unwrap_err(),
            BlindCriteriaError::EmptyRegistrationEvidenceId
        );
    }
}
