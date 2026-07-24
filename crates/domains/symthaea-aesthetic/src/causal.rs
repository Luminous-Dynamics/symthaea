// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Causal-study evidence for aesthetic metrics and policies.
//!
//! This module deliberately distinguishes randomized, quasi-experimental,
//! observational, and mechanism-only evidence. A correlation is never promoted
//! into a causal claim merely because its effect estimate is large.

use crate::{IntegrityError, digest_json, load_json, save_json_atomic};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::Path;

pub const CAUSAL_STUDY_SCHEMA_VERSION: u32 = 1;
pub const CAUSAL_REPORT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalDesign {
    RandomizedControlled,
    RandomizedBlocked,
    NaturalExperiment,
    MatchedObservational,
    InterruptedTimeSeries,
    MechanismProbe,
}

impl CausalDesign {
    pub const fn supports_causal_language(self) -> bool {
        matches!(
            self,
            Self::RandomizedControlled | Self::RandomizedBlocked | Self::NaturalExperiment
        )
    }

    pub const fn requires_assignment_seed(self) -> bool {
        matches!(self, Self::RandomizedControlled | Self::RandomizedBlocked)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Estimand {
    AverageTreatmentEffect,
    AverageTreatmentEffectOnTreated,
    IntentionToTreat,
    PerProtocol,
    LocalAverageTreatmentEffect,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudyArm {
    pub arm_id: String,
    pub is_control: bool,
    /// Signed intervention intensity. Zero is allowed for a control arm.
    pub intervention_dose: f32,
    pub assigned: u64,
    pub completed: u64,
    pub baseline_mean: Option<f32>,
    pub outcome_mean: f32,
    pub outcome_variance: f32,
}

impl StudyArm {
    fn validate(&self) -> Result<(), CausalError> {
        validate_identifier(&self.arm_id, "arm id")?;
        validate_finite(self.intervention_dose, "intervention dose")?;
        if self.assigned == 0 {
            return Err(CausalError::InvalidArm(format!(
                "arm {} has no assigned units",
                self.arm_id
            )));
        }
        if self.completed > self.assigned {
            return Err(CausalError::InvalidArm(format!(
                "arm {} completed count exceeds assigned count",
                self.arm_id
            )));
        }
        validate_finite(self.outcome_mean, "outcome mean")?;
        validate_nonnegative(self.outcome_variance, "outcome variance")?;
        if let Some(value) = self.baseline_mean {
            validate_finite(value, "baseline mean")?;
        }
        Ok(())
    }

    pub fn attrition_fraction(&self) -> f32 {
        1.0 - self.completed as f32 / self.assigned as f32
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CausalStudy {
    pub schema_version: u32,
    pub study_id: String,
    pub preregistration_id: String,
    pub intervention_metric_id: String,
    pub outcome_metric_id: String,
    pub design: CausalDesign,
    pub estimand: Estimand,
    pub assignment_seed: Option<u64>,
    pub arms: Vec<StudyArm>,
    pub measured_confounders: Vec<String>,
    pub known_unmeasured_confounders: Vec<String>,
    pub collection_started_unix_ms: u64,
    pub collection_ended_unix_ms: u64,
}

impl CausalStudy {
    pub fn validate(&self) -> Result<(), CausalError> {
        if self.schema_version == 0 || self.schema_version > CAUSAL_STUDY_SCHEMA_VERSION {
            return Err(CausalError::UnsupportedStudySchema(self.schema_version));
        }
        validate_identifier(&self.study_id, "study id")?;
        validate_identifier(&self.preregistration_id, "preregistration id")?;
        validate_identifier(&self.intervention_metric_id, "intervention metric id")?;
        validate_identifier(&self.outcome_metric_id, "outcome metric id")?;
        if self.intervention_metric_id == self.outcome_metric_id {
            return Err(CausalError::IdenticalInterventionAndOutcome);
        }
        if self.collection_ended_unix_ms <= self.collection_started_unix_ms {
            return Err(CausalError::InvalidCollectionWindow);
        }
        if self.design.requires_assignment_seed() && self.assignment_seed.is_none() {
            return Err(CausalError::MissingAssignmentSeed);
        }
        if self.arms.len() < 2 {
            return Err(CausalError::InsufficientArms);
        }
        let mut arm_ids = BTreeSet::new();
        let mut controls = 0usize;
        for arm in &self.arms {
            arm.validate()?;
            if !arm_ids.insert(arm.arm_id.as_str()) {
                return Err(CausalError::DuplicateArm(arm.arm_id.clone()));
            }
            controls += usize::from(arm.is_control);
        }
        if controls != 1 {
            return Err(CausalError::InvalidControlCount(controls));
        }
        validate_unique_labels(&self.measured_confounders, "measured confounder")?;
        validate_unique_labels(
            &self.known_unmeasured_confounders,
            "known unmeasured confounder",
        )?;
        let measured = self
            .measured_confounders
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if self
            .known_unmeasured_confounders
            .iter()
            .any(|name| measured.contains(name.as_str()))
        {
            return Err(CausalError::ConfounderClassificationOverlap);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, CausalError> {
        self.validate()?;
        digest_json(self).map_err(CausalError::Integrity)
    }

    pub fn save(&self, path: &Path) -> Result<(), CausalError> {
        self.validate()?;
        save_json_atomic::<_, CausalError>(path, self)
    }

    pub fn load(path: &Path) -> Result<Self, CausalError> {
        let study: Self = load_json::<_, CausalError>(path)?;
        study.validate()?;
        Ok(study)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CausalPolicy {
    pub minimum_completed_per_arm: u64,
    pub maximum_attrition_fraction: f32,
    pub maximum_attrition_imbalance: f32,
    pub maximum_baseline_imbalance: f32,
    pub minimum_standardized_effect: f32,
    pub maximum_standard_error: f32,
    pub allow_quasi_experimental_claims: bool,
}

impl CausalPolicy {
    pub const fn production() -> Self {
        Self {
            minimum_completed_per_arm: 30,
            maximum_attrition_fraction: 0.20,
            maximum_attrition_imbalance: 0.10,
            maximum_baseline_imbalance: 0.20,
            minimum_standardized_effect: 0.10,
            maximum_standard_error: 0.25,
            allow_quasi_experimental_claims: false,
        }
    }

    pub fn validate(&self) -> Result<(), CausalError> {
        if self.minimum_completed_per_arm == 0 {
            return Err(CausalError::InvalidPolicy(
                "minimum completed per arm must be positive".to_owned(),
            ));
        }
        validate_unit(self.maximum_attrition_fraction, "maximum attrition fraction")?;
        validate_unit(self.maximum_attrition_imbalance, "maximum attrition imbalance")?;
        validate_nonnegative(self.maximum_baseline_imbalance, "maximum baseline imbalance")?;
        validate_nonnegative(self.minimum_standardized_effect, "minimum standardized effect")?;
        validate_nonnegative(self.maximum_standard_error, "maximum standard error")?;
        Ok(())
    }
}

impl Default for CausalPolicy {
    fn default() -> Self {
        Self::production()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CausalFindingCode {
    DesignAuthority,
    SampleSupport,
    Attrition,
    BaselineBalance,
    UnmeasuredConfounding,
    EstimatePrecision,
    EffectMagnitude,
}

impl CausalFindingCode {
    pub const ALL: [Self; 7] = [
        Self::DesignAuthority,
        Self::SampleSupport,
        Self::Attrition,
        Self::BaselineBalance,
        Self::UnmeasuredConfounding,
        Self::EstimatePrecision,
        Self::EffectMagnitude,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalFindingStatus {
    Pass,
    Warning,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CausalFinding {
    pub code: CausalFindingCode,
    pub status: CausalFindingStatus,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalAuthority {
    Causal,
    QuasiExperimental,
    Associational,
    MechanismOnly,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArmEffect {
    pub arm_id: String,
    pub raw_effect: f32,
    pub baseline_adjusted_effect: Option<f32>,
    pub pooled_standard_deviation: f32,
    pub standardized_effect: f32,
    pub standard_error: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CausalReport {
    pub schema_version: u32,
    pub study: CausalStudy,
    pub policy: CausalPolicy,
    pub arm_effects: Vec<ArmEffect>,
    pub maximum_attrition: f32,
    pub attrition_imbalance: f32,
    pub maximum_baseline_imbalance: Option<f32>,
    pub authority: CausalAuthority,
    pub findings: Vec<CausalFinding>,
}

impl CausalReport {
    pub fn validate(&self) -> Result<(), CausalError> {
        if self.schema_version == 0 || self.schema_version > CAUSAL_REPORT_SCHEMA_VERSION {
            return Err(CausalError::UnsupportedReportSchema(self.schema_version));
        }
        let expected = evaluate_causal_study(self.study.clone(), self.policy)?;
        if self != &expected {
            return Err(CausalError::DerivedReportMismatch);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, CausalError> {
        self.validate()?;
        digest_json(self).map_err(CausalError::Integrity)
    }

    pub fn supports_causal_claim(&self) -> bool {
        self.authority == CausalAuthority::Causal
    }
}

pub fn evaluate_causal_study(
    study: CausalStudy,
    policy: CausalPolicy,
) -> Result<CausalReport, CausalError> {
    study.validate()?;
    policy.validate()?;
    let control = study
        .arms
        .iter()
        .find(|arm| arm.is_control)
        .ok_or(CausalError::InvalidControlCount(0))?;
    let maximum_attrition = study
        .arms
        .iter()
        .map(StudyArm::attrition_fraction)
        .fold(0.0f32, f32::max);
    let minimum_attrition = study
        .arms
        .iter()
        .map(StudyArm::attrition_fraction)
        .fold(1.0f32, f32::min);
    let attrition_imbalance = maximum_attrition - minimum_attrition;
    let mut maximum_baseline_imbalance = None::<f32>;
    let mut effects = Vec::new();
    for arm in study.arms.iter().filter(|arm| !arm.is_control) {
        let raw_effect = arm.outcome_mean - control.outcome_mean;
        let baseline_adjusted_effect = match (arm.baseline_mean, control.baseline_mean) {
            (Some(treatment), Some(baseline)) => {
                let imbalance = (treatment - baseline).abs();
                maximum_baseline_imbalance = Some(
                    maximum_baseline_imbalance
                        .map_or(imbalance, |current| current.max(imbalance)),
                );
                Some(raw_effect - (treatment - baseline))
            }
            _ => None,
        };
        let pooled_variance = (arm.outcome_variance + control.outcome_variance) / 2.0;
        let pooled_standard_deviation = pooled_variance.sqrt();
        let standardized_effect = if pooled_standard_deviation <= f32::EPSILON {
            if raw_effect.abs() <= f32::EPSILON { 0.0 } else { raw_effect.signum() * f32::MAX }
        } else {
            raw_effect / pooled_standard_deviation
        };
        let standard_error = (
            arm.outcome_variance / arm.completed.max(1) as f32
                + control.outcome_variance / control.completed.max(1) as f32
        )
            .sqrt();
        effects.push(ArmEffect {
            arm_id: arm.arm_id.clone(),
            raw_effect,
            baseline_adjusted_effect,
            pooled_standard_deviation,
            standardized_effect,
            standard_error,
        });
    }

    let design_status = match study.design {
        CausalDesign::RandomizedControlled | CausalDesign::RandomizedBlocked => {
            CausalFindingStatus::Pass
        }
        CausalDesign::NaturalExperiment if policy.allow_quasi_experimental_claims => {
            CausalFindingStatus::Warning
        }
        _ => CausalFindingStatus::Warning,
    };
    let sample_pass = study
        .arms
        .iter()
        .all(|arm| arm.completed >= policy.minimum_completed_per_arm);
    let attrition_pass = maximum_attrition <= policy.maximum_attrition_fraction
        && attrition_imbalance <= policy.maximum_attrition_imbalance;
    let baseline_status = match maximum_baseline_imbalance {
        Some(value) if value > policy.maximum_baseline_imbalance => CausalFindingStatus::Fail,
        Some(_) => CausalFindingStatus::Pass,
        None if matches!(study.design, CausalDesign::RandomizedControlled) => {
            CausalFindingStatus::Warning
        }
        None => CausalFindingStatus::Warning,
    };
    let confounding_status = if study.known_unmeasured_confounders.is_empty() {
        CausalFindingStatus::Pass
    } else if study.design.supports_causal_language() {
        CausalFindingStatus::Warning
    } else {
        CausalFindingStatus::Fail
    };
    let precision_pass = effects
        .iter()
        .all(|effect| effect.standard_error <= policy.maximum_standard_error);
    let magnitude_pass = effects.iter().any(|effect| {
        effect.standardized_effect.is_finite()
            && effect.standardized_effect.abs() >= policy.minimum_standardized_effect
    });
    let findings = vec![
        finding(
            CausalFindingCode::DesignAuthority,
            design_status,
            format!("study design is {:?}", study.design),
        ),
        finding(
            CausalFindingCode::SampleSupport,
            pass_fail(sample_pass),
            format!(
                "minimum completed per arm is {}",
                study.arms.iter().map(|arm| arm.completed).min().unwrap_or(0)
            ),
        ),
        finding(
            CausalFindingCode::Attrition,
            pass_fail(attrition_pass),
            format!(
                "maximum attrition {:.4}; imbalance {:.4}",
                maximum_attrition, attrition_imbalance
            ),
        ),
        finding(
            CausalFindingCode::BaselineBalance,
            baseline_status,
            format!("maximum observed baseline imbalance {maximum_baseline_imbalance:?}"),
        ),
        finding(
            CausalFindingCode::UnmeasuredConfounding,
            confounding_status,
            format!(
                "{} known unmeasured confounders",
                study.known_unmeasured_confounders.len()
            ),
        ),
        finding(
            CausalFindingCode::EstimatePrecision,
            pass_fail(precision_pass),
            format!(
                "maximum standard error {:.4}",
                effects.iter().map(|effect| effect.standard_error).fold(0.0f32, f32::max)
            ),
        ),
        finding(
            CausalFindingCode::EffectMagnitude,
            pass_warn(magnitude_pass),
            format!(
                "maximum absolute standardized effect {:.4}",
                effects
                    .iter()
                    .map(|effect| effect.standardized_effect.abs())
                    .filter(|value| value.is_finite())
                    .fold(0.0f32, f32::max)
            ),
        ),
    ];
    let has_fail = findings
        .iter()
        .any(|finding| finding.status == CausalFindingStatus::Fail);
    let authority = if has_fail || !sample_pass || !attrition_pass {
        CausalAuthority::Unsupported
    } else {
        match study.design {
            CausalDesign::RandomizedControlled | CausalDesign::RandomizedBlocked => {
                CausalAuthority::Causal
            }
            CausalDesign::NaturalExperiment => CausalAuthority::QuasiExperimental,
            CausalDesign::MatchedObservational | CausalDesign::InterruptedTimeSeries => {
                CausalAuthority::Associational
            }
            CausalDesign::MechanismProbe => CausalAuthority::MechanismOnly,
        }
    };
    Ok(CausalReport {
        schema_version: CAUSAL_REPORT_SCHEMA_VERSION,
        study,
        policy,
        arm_effects: effects,
        maximum_attrition,
        attrition_imbalance,
        maximum_baseline_imbalance,
        authority,
        findings,
    })
}

fn finding(code: CausalFindingCode, status: CausalFindingStatus, detail: String) -> CausalFinding {
    CausalFinding { code, status, detail }
}

const fn pass_fail(pass: bool) -> CausalFindingStatus {
    if pass { CausalFindingStatus::Pass } else { CausalFindingStatus::Fail }
}

const fn pass_warn(pass: bool) -> CausalFindingStatus {
    if pass { CausalFindingStatus::Pass } else { CausalFindingStatus::Warning }
}

fn validate_identifier(value: &str, field: &str) -> Result<(), CausalError> {
    if value.trim().is_empty() {
        return Err(CausalError::InvalidIdentifier(field.to_owned()));
    }
    Ok(())
}

fn validate_unique_labels(values: &[String], field: &str) -> Result<(), CausalError> {
    let mut unique = BTreeSet::new();
    for value in values {
        validate_identifier(value, field)?;
        if !unique.insert(value.as_str()) {
            return Err(CausalError::DuplicateLabel(value.clone()));
        }
    }
    Ok(())
}

fn validate_finite(value: f32, field: &str) -> Result<(), CausalError> {
    if !value.is_finite() {
        return Err(CausalError::InvalidNumber(field.to_owned()));
    }
    Ok(())
}

fn validate_nonnegative(value: f32, field: &str) -> Result<(), CausalError> {
    validate_finite(value, field)?;
    if value < 0.0 {
        return Err(CausalError::InvalidNumber(field.to_owned()));
    }
    Ok(())
}

fn validate_unit(value: f32, field: &str) -> Result<(), CausalError> {
    validate_finite(value, field)?;
    if !(0.0..=1.0).contains(&value) {
        return Err(CausalError::InvalidNumber(field.to_owned()));
    }
    Ok(())
}

#[derive(Debug)]
pub enum CausalError {
    UnsupportedStudySchema(u32),
    UnsupportedReportSchema(u32),
    InvalidIdentifier(String),
    InvalidNumber(String),
    InvalidArm(String),
    InvalidPolicy(String),
    InvalidCollectionWindow,
    MissingAssignmentSeed,
    InsufficientArms,
    InvalidControlCount(usize),
    DuplicateArm(String),
    DuplicateLabel(String),
    ConfounderClassificationOverlap,
    IdenticalInterventionAndOutcome,
    DerivedReportMismatch,
    Integrity(IntegrityError),
    Persistence(String),
}

impl std::fmt::Display for CausalError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedStudySchema(version) => write!(formatter, "unsupported causal-study schema {version}"),
            Self::UnsupportedReportSchema(version) => write!(formatter, "unsupported causal-report schema {version}"),
            Self::InvalidIdentifier(field) => write!(formatter, "{field} must not be empty"),
            Self::InvalidNumber(field) => write!(formatter, "{field} must be finite and within its documented range"),
            Self::InvalidArm(detail) | Self::InvalidPolicy(detail) | Self::Persistence(detail) => formatter.write_str(detail),
            Self::InvalidCollectionWindow => formatter.write_str("causal study has an invalid collection window"),
            Self::MissingAssignmentSeed => formatter.write_str("randomized design requires an assignment seed"),
            Self::InsufficientArms => formatter.write_str("causal study requires at least two arms"),
            Self::InvalidControlCount(count) => write!(formatter, "causal study requires exactly one control arm, found {count}"),
            Self::DuplicateArm(id) => write!(formatter, "causal study duplicates arm {id}"),
            Self::DuplicateLabel(label) => write!(formatter, "causal study duplicates label {label}"),
            Self::ConfounderClassificationOverlap => formatter.write_str("a confounder cannot be both measured and unmeasured"),
            Self::IdenticalInterventionAndOutcome => formatter.write_str("intervention and outcome metrics must differ"),
            Self::DerivedReportMismatch => formatter.write_str("causal report does not match recomputed evidence"),
            Self::Integrity(error) => write!(formatter, "causal evidence integrity failed: {error}"),
        }
    }
}

impl std::error::Error for CausalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Integrity(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for CausalError {
    fn from(error: std::io::Error) -> Self { Self::Persistence(error.to_string()) }
}

impl From<serde_json::Error> for CausalError {
    fn from(error: serde_json::Error) -> Self { Self::Persistence(error.to_string()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(design: CausalDesign) -> CausalStudy {
        CausalStudy {
            schema_version: CAUSAL_STUDY_SCHEMA_VERSION,
            study_id: "study-1".to_owned(),
            preregistration_id: "pre-1".to_owned(),
            intervention_metric_id: "metric-order".to_owned(),
            outcome_metric_id: "human-preference".to_owned(),
            design,
            estimand: Estimand::IntentionToTreat,
            assignment_seed: Some(7),
            arms: vec![
                StudyArm {
                    arm_id: "control".to_owned(), is_control: true, intervention_dose: 0.0,
                    assigned: 100, completed: 95, baseline_mean: Some(0.50), outcome_mean: 0.52, outcome_variance: 0.04,
                },
                StudyArm {
                    arm_id: "treatment".to_owned(), is_control: false, intervention_dose: 0.2,
                    assigned: 100, completed: 94, baseline_mean: Some(0.51), outcome_mean: 0.64, outcome_variance: 0.05,
                },
            ],
            measured_confounders: vec!["expertise".to_owned()],
            known_unmeasured_confounders: Vec::new(),
            collection_started_unix_ms: 10,
            collection_ended_unix_ms: 20,
        }
    }

    #[test]
    fn randomized_study_can_support_causal_claim() {
        let report = evaluate_causal_study(fixture(CausalDesign::RandomizedControlled), CausalPolicy::production()).expect("report");
        assert_eq!(report.authority, CausalAuthority::Causal);
        assert!(report.supports_causal_claim());
        report.validate().expect("valid report");
    }

    #[test]
    fn observational_study_is_never_promoted_to_causal() {
        let report = evaluate_causal_study(fixture(CausalDesign::MatchedObservational), CausalPolicy::production()).expect("report");
        assert_eq!(report.authority, CausalAuthority::Associational);
        assert!(!report.supports_causal_claim());
    }

    #[test]
    fn randomized_design_requires_assignment_seed() {
        let mut study = fixture(CausalDesign::RandomizedControlled);
        study.assignment_seed = None;
        assert!(matches!(study.validate(), Err(CausalError::MissingAssignmentSeed)));
    }

    #[test]
    fn forged_report_is_rejected() {
        let mut report = evaluate_causal_study(fixture(CausalDesign::RandomizedControlled), CausalPolicy::production()).expect("report");
        report.authority = CausalAuthority::Unsupported;
        assert!(matches!(report.validate(), Err(CausalError::DerivedReportMismatch)));
    }
}
