// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Preregistration and execution-quality evidence for human aesthetic studies.
//! A study may be useful without being definitive, but its evidential authority
//! must reflect blinding, attrition, multiplicity, protocol deviations, and the
//! temporal relation between registration and data collection.

use crate::{IntegrityError, digest_json, load_json, save_json_atomic};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::Path;

pub const PREREGISTRATION_SCHEMA_VERSION: u32 = 1;
pub const STUDY_QUALITY_REPORT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegistrationTiming {
    Prospective,
    Retrospective,
    Unregistered,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MultiplicityCorrection {
    None,
    Bonferroni,
    Holm,
    FalseDiscoveryRate,
    Hierarchical,
    BayesianMultilevel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisteredHypothesis {
    pub hypothesis_id: String,
    pub statement: String,
    pub primary: bool,
    pub directional: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StudyPreregistration {
    pub schema_version: u32,
    pub preregistration_id: String,
    pub study_id: String,
    pub registered_unix_ms: u64,
    pub planned_collection_start_unix_ms: u64,
    pub hypotheses: Vec<RegisteredHypothesis>,
    pub primary_outcomes: Vec<String>,
    pub exclusion_rules: Vec<String>,
    pub analysis_plan_digest: String,
    pub planned_sample_size: u64,
    pub planned_multiplicity_correction: MultiplicityCorrection,
}

impl StudyPreregistration {
    pub fn validate(&self) -> Result<(), StudyQualityError> {
        if self.schema_version == 0 || self.schema_version > PREREGISTRATION_SCHEMA_VERSION {
            return Err(StudyQualityError::UnsupportedPreregistrationSchema(self.schema_version));
        }
        validate_identifier(&self.preregistration_id, "preregistration id")?;
        validate_identifier(&self.study_id, "study id")?;
        validate_identifier(&self.analysis_plan_digest, "analysis plan digest")?;
        if self.planned_sample_size == 0 {
            return Err(StudyQualityError::InvalidPreregistration(
                "planned sample size must be positive".to_owned(),
            ));
        }
        if self.hypotheses.is_empty() || self.primary_outcomes.is_empty() {
            return Err(StudyQualityError::InvalidPreregistration(
                "hypotheses and primary outcomes must not be empty".to_owned(),
            ));
        }
        let mut ids = BTreeSet::new();
        let mut primary = 0usize;
        for hypothesis in &self.hypotheses {
            validate_identifier(&hypothesis.hypothesis_id, "hypothesis id")?;
            validate_identifier(&hypothesis.statement, "hypothesis statement")?;
            if !ids.insert(hypothesis.hypothesis_id.as_str()) {
                return Err(StudyQualityError::DuplicateHypothesis(
                    hypothesis.hypothesis_id.clone(),
                ));
            }
            primary += usize::from(hypothesis.primary);
        }
        if primary == 0 {
            return Err(StudyQualityError::InvalidPreregistration(
                "at least one primary hypothesis is required".to_owned(),
            ));
        }
        validate_unique(&self.primary_outcomes, "primary outcome")?;
        validate_unique(&self.exclusion_rules, "exclusion rule")?;
        Ok(())
    }

    pub fn timing(&self, actual_collection_start_unix_ms: u64) -> RegistrationTiming {
        if self.registered_unix_ms == 0 {
            RegistrationTiming::Unregistered
        } else if self.registered_unix_ms < actual_collection_start_unix_ms
            && self.registered_unix_ms <= self.planned_collection_start_unix_ms
        {
            RegistrationTiming::Prospective
        } else {
            RegistrationTiming::Retrospective
        }
    }

    pub fn digest(&self) -> Result<String, StudyQualityError> {
        self.validate()?;
        digest_json(self).map_err(StudyQualityError::Integrity)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolDeviation {
    pub deviation_id: String,
    pub material: bool,
    pub disclosed: bool,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudyExecution {
    pub study_id: String,
    pub collection_started_unix_ms: u64,
    pub collection_ended_unix_ms: u64,
    pub enrolled: u64,
    pub completed: u64,
    pub excluded_after_collection: u64,
    pub hypotheses_tested: usize,
    pub reported_primary_outcomes: Vec<String>,
    pub participant_blinded: bool,
    pub assessor_blinded: bool,
    pub analyst_blinded_until_lock: bool,
    pub manipulation_check_pass_rate: Option<f32>,
    pub applied_multiplicity_correction: MultiplicityCorrection,
    pub deviations: Vec<ProtocolDeviation>,
    pub analysis_plan_digest: String,
}

impl StudyExecution {
    pub fn validate(&self) -> Result<(), StudyQualityError> {
        validate_identifier(&self.study_id, "study id")?;
        validate_identifier(&self.analysis_plan_digest, "analysis plan digest")?;
        if self.collection_ended_unix_ms <= self.collection_started_unix_ms {
            return Err(StudyQualityError::InvalidExecution(
                "collection window is invalid".to_owned(),
            ));
        }
        if self.enrolled == 0 || self.completed > self.enrolled {
            return Err(StudyQualityError::InvalidExecution(
                "participant counts are invalid".to_owned(),
            ));
        }
        if self.excluded_after_collection > self.completed {
            return Err(StudyQualityError::InvalidExecution(
                "post-collection exclusions exceed completions".to_owned(),
            ));
        }
        if self.hypotheses_tested == 0 {
            return Err(StudyQualityError::InvalidExecution(
                "at least one hypothesis must be tested".to_owned(),
            ));
        }
        validate_unique(&self.reported_primary_outcomes, "reported primary outcome")?;
        if let Some(rate) = self.manipulation_check_pass_rate {
            validate_unit(rate, "manipulation check pass rate")?;
        }
        let mut ids = BTreeSet::new();
        for deviation in &self.deviations {
            validate_identifier(&deviation.deviation_id, "deviation id")?;
            validate_identifier(&deviation.detail, "deviation detail")?;
            if !ids.insert(deviation.deviation_id.as_str()) {
                return Err(StudyQualityError::DuplicateDeviation(
                    deviation.deviation_id.clone(),
                ));
            }
        }
        Ok(())
    }

    pub fn attrition_fraction(&self) -> f32 {
        1.0 - self.completed as f32 / self.enrolled as f32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StudyQualityPolicy {
    pub minimum_completion_fraction: f32,
    pub minimum_sample_fraction_of_plan: f32,
    pub minimum_manipulation_check_pass_rate: f32,
    pub require_prospective_registration: bool,
    pub require_analysis_plan_match: bool,
    pub require_multiplicity_correction_for_multiple_tests: bool,
    pub maximum_undisclosed_material_deviations: usize,
}

impl StudyQualityPolicy {
    pub const fn production() -> Self {
        Self {
            minimum_completion_fraction: 0.80,
            minimum_sample_fraction_of_plan: 0.90,
            minimum_manipulation_check_pass_rate: 0.70,
            require_prospective_registration: true,
            require_analysis_plan_match: true,
            require_multiplicity_correction_for_multiple_tests: true,
            maximum_undisclosed_material_deviations: 0,
        }
    }

    pub fn validate(&self) -> Result<(), StudyQualityError> {
        validate_unit(self.minimum_completion_fraction, "minimum completion fraction")?;
        validate_unit(self.minimum_sample_fraction_of_plan, "minimum sample fraction")?;
        validate_unit(
            self.minimum_manipulation_check_pass_rate,
            "minimum manipulation check pass rate",
        )?;
        Ok(())
    }
}

impl Default for StudyQualityPolicy {
    fn default() -> Self { Self::production() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum StudyQualityFindingCode {
    RegistrationTiming,
    SampleRealization,
    Attrition,
    Blinding,
    OutcomeCompleteness,
    AnalysisPlanIntegrity,
    MultiplicityControl,
    ProtocolDeviations,
    ManipulationCheck,
}

impl StudyQualityFindingCode {
    pub const ALL: [Self; 9] = [
        Self::RegistrationTiming,
        Self::SampleRealization,
        Self::Attrition,
        Self::Blinding,
        Self::OutcomeCompleteness,
        Self::AnalysisPlanIntegrity,
        Self::MultiplicityControl,
        Self::ProtocolDeviations,
        Self::ManipulationCheck,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StudyQualityFindingStatus { Pass, Warning, Fail }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StudyQualityFinding {
    pub code: StudyQualityFindingCode,
    pub status: StudyQualityFindingStatus,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StudyQualityOutcome { High, Moderate, Low, Invalid }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudyQualityReport {
    pub schema_version: u32,
    pub preregistration: StudyPreregistration,
    pub execution: StudyExecution,
    pub policy: StudyQualityPolicy,
    pub registration_timing: RegistrationTiming,
    pub completion_fraction: f32,
    pub sample_fraction_of_plan: f32,
    pub undisclosed_material_deviations: usize,
    pub outcome: StudyQualityOutcome,
    pub findings: Vec<StudyQualityFinding>,
}

impl StudyQualityReport {
    pub fn validate(&self) -> Result<(), StudyQualityError> {
        if self.schema_version == 0 || self.schema_version > STUDY_QUALITY_REPORT_SCHEMA_VERSION {
            return Err(StudyQualityError::UnsupportedReportSchema(self.schema_version));
        }
        let expected = evaluate_study_quality(
            self.preregistration.clone(),
            self.execution.clone(),
            self.policy,
        )?;
        if self != &expected {
            return Err(StudyQualityError::DerivedReportMismatch);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, StudyQualityError> {
        self.validate()?;
        digest_json(self).map_err(StudyQualityError::Integrity)
    }

    pub fn save(&self, path: &Path) -> Result<(), StudyQualityError> {
        self.validate()?;
        save_json_atomic::<_, StudyQualityError>(path, self)
    }

    pub fn load(path: &Path) -> Result<Self, StudyQualityError> {
        let report: Self = load_json::<_, StudyQualityError>(path)?;
        report.validate()?;
        Ok(report)
    }
}

pub fn evaluate_study_quality(
    preregistration: StudyPreregistration,
    execution: StudyExecution,
    policy: StudyQualityPolicy,
) -> Result<StudyQualityReport, StudyQualityError> {
    preregistration.validate()?;
    execution.validate()?;
    policy.validate()?;
    if preregistration.study_id != execution.study_id {
        return Err(StudyQualityError::StudyBindingMismatch);
    }
    let registration_timing = preregistration.timing(execution.collection_started_unix_ms);
    let completion_fraction = execution.completed as f32 / execution.enrolled as f32;
    let sample_fraction_of_plan = execution.completed as f32 / preregistration.planned_sample_size as f32;
    let undisclosed_material_deviations = execution
        .deviations
        .iter()
        .filter(|deviation| deviation.material && !deviation.disclosed)
        .count();
    let registration_ok = !policy.require_prospective_registration
        || registration_timing == RegistrationTiming::Prospective;
    let sample_ok = sample_fraction_of_plan >= policy.minimum_sample_fraction_of_plan;
    let attrition_ok = completion_fraction >= policy.minimum_completion_fraction;
    let blinding_count = [
        execution.participant_blinded,
        execution.assessor_blinded,
        execution.analyst_blinded_until_lock,
    ]
    .into_iter()
    .filter(|value| *value)
    .count();
    let primary = preregistration
        .primary_outcomes
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let reported = execution
        .reported_primary_outcomes
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let outcomes_ok = primary == reported;
    let plan_ok = preregistration.analysis_plan_digest == execution.analysis_plan_digest;
    let multiplicity_ok = execution.hypotheses_tested <= 1
        || !policy.require_multiplicity_correction_for_multiple_tests
        || (execution.applied_multiplicity_correction != MultiplicityCorrection::None
            && execution.applied_multiplicity_correction
                == preregistration.planned_multiplicity_correction);
    let deviations_ok = undisclosed_material_deviations
        <= policy.maximum_undisclosed_material_deviations;
    let manipulation_status = match execution.manipulation_check_pass_rate {
        Some(rate) if rate >= policy.minimum_manipulation_check_pass_rate => StudyQualityFindingStatus::Pass,
        Some(_) => StudyQualityFindingStatus::Fail,
        None => StudyQualityFindingStatus::Warning,
    };
    let findings = vec![
        finding(StudyQualityFindingCode::RegistrationTiming, pass_fail(registration_ok), format!("registration is {registration_timing:?}")),
        finding(StudyQualityFindingCode::SampleRealization, pass_fail(sample_ok), format!("sample realization {:.4}", sample_fraction_of_plan)),
        finding(StudyQualityFindingCode::Attrition, pass_fail(attrition_ok), format!("completion fraction {:.4}", completion_fraction)),
        finding(StudyQualityFindingCode::Blinding, if blinding_count >= 2 { StudyQualityFindingStatus::Pass } else if blinding_count == 1 { StudyQualityFindingStatus::Warning } else { StudyQualityFindingStatus::Fail }, format!("{blinding_count} of 3 roles blinded")),
        finding(StudyQualityFindingCode::OutcomeCompleteness, pass_fail(outcomes_ok), format!("reported {} of {} registered primary outcomes", reported.len(), primary.len())),
        finding(StudyQualityFindingCode::AnalysisPlanIntegrity, pass_fail(!policy.require_analysis_plan_match || plan_ok), format!("analysis plan digest match: {plan_ok}")),
        finding(StudyQualityFindingCode::MultiplicityControl, pass_fail(multiplicity_ok), format!("tested {} hypotheses with {:?}", execution.hypotheses_tested, execution.applied_multiplicity_correction)),
        finding(StudyQualityFindingCode::ProtocolDeviations, pass_fail(deviations_ok), format!("{undisclosed_material_deviations} undisclosed material deviations")),
        finding(StudyQualityFindingCode::ManipulationCheck, manipulation_status, format!("manipulation check pass rate {:?}", execution.manipulation_check_pass_rate)),
    ];
    let failures = findings.iter().filter(|finding| finding.status == StudyQualityFindingStatus::Fail).count();
    let warnings = findings.iter().filter(|finding| finding.status == StudyQualityFindingStatus::Warning).count();
    let outcome = if failures >= 3 || !registration_ok || !plan_ok && policy.require_analysis_plan_match {
        StudyQualityOutcome::Invalid
    } else if failures > 0 {
        StudyQualityOutcome::Low
    } else if warnings > 1 {
        StudyQualityOutcome::Moderate
    } else {
        StudyQualityOutcome::High
    };
    Ok(StudyQualityReport {
        schema_version: STUDY_QUALITY_REPORT_SCHEMA_VERSION,
        preregistration,
        execution,
        policy,
        registration_timing,
        completion_fraction,
        sample_fraction_of_plan,
        undisclosed_material_deviations,
        outcome,
        findings,
    })
}

fn finding(code: StudyQualityFindingCode, status: StudyQualityFindingStatus, detail: String) -> StudyQualityFinding {
    StudyQualityFinding { code, status, detail }
}

const fn pass_fail(pass: bool) -> StudyQualityFindingStatus {
    if pass { StudyQualityFindingStatus::Pass } else { StudyQualityFindingStatus::Fail }
}

fn validate_identifier(value: &str, field: &str) -> Result<(), StudyQualityError> {
    if value.trim().is_empty() { Err(StudyQualityError::InvalidIdentifier(field.to_owned())) } else { Ok(()) }
}

fn validate_unique(values: &[String], field: &str) -> Result<(), StudyQualityError> {
    let mut unique = BTreeSet::new();
    for value in values {
        validate_identifier(value, field)?;
        if !unique.insert(value.as_str()) {
            return Err(StudyQualityError::DuplicateLabel(value.clone()));
        }
    }
    Ok(())
}

fn validate_unit(value: f32, field: &str) -> Result<(), StudyQualityError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        Err(StudyQualityError::InvalidNumber(field.to_owned()))
    } else { Ok(()) }
}

#[derive(Debug)]
pub enum StudyQualityError {
    UnsupportedPreregistrationSchema(u32),
    UnsupportedReportSchema(u32),
    InvalidIdentifier(String),
    InvalidNumber(String),
    InvalidPreregistration(String),
    InvalidExecution(String),
    DuplicateHypothesis(String),
    DuplicateDeviation(String),
    DuplicateLabel(String),
    StudyBindingMismatch,
    DerivedReportMismatch,
    Integrity(IntegrityError),
    Persistence(String),
}

impl std::fmt::Display for StudyQualityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedPreregistrationSchema(version) => write!(formatter, "unsupported preregistration schema {version}"),
            Self::UnsupportedReportSchema(version) => write!(formatter, "unsupported study-quality schema {version}"),
            Self::InvalidIdentifier(field) => write!(formatter, "{field} must not be empty"),
            Self::InvalidNumber(field) => write!(formatter, "{field} must be finite and in [0, 1]"),
            Self::InvalidPreregistration(detail) | Self::InvalidExecution(detail) | Self::Persistence(detail) => formatter.write_str(detail),
            Self::DuplicateHypothesis(id) => write!(formatter, "duplicate hypothesis {id}"),
            Self::DuplicateDeviation(id) => write!(formatter, "duplicate protocol deviation {id}"),
            Self::DuplicateLabel(label) => write!(formatter, "duplicate label {label}"),
            Self::StudyBindingMismatch => formatter.write_str("preregistration and execution refer to different studies"),
            Self::DerivedReportMismatch => formatter.write_str("study-quality report does not match recomputed evidence"),
            Self::Integrity(error) => write!(formatter, "study-quality integrity failed: {error}"),
        }
    }
}

impl std::error::Error for StudyQualityError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self { Self::Integrity(error) => Some(error), _ => None }
    }
}

impl From<std::io::Error> for StudyQualityError {
    fn from(error: std::io::Error) -> Self { Self::Persistence(error.to_string()) }
}
impl From<serde_json::Error> for StudyQualityError {
    fn from(error: serde_json::Error) -> Self { Self::Persistence(error.to_string()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preregistration() -> StudyPreregistration {
        StudyPreregistration {
            schema_version: PREREGISTRATION_SCHEMA_VERSION,
            preregistration_id: "pre-1".to_owned(),
            study_id: "study-1".to_owned(),
            registered_unix_ms: 10,
            planned_collection_start_unix_ms: 20,
            hypotheses: vec![RegisteredHypothesis { hypothesis_id: "h1".to_owned(), statement: "metric predicts preference".to_owned(), primary: true, directional: true }],
            primary_outcomes: vec!["preference".to_owned()],
            exclusion_rules: vec!["failed attention check".to_owned()],
            analysis_plan_digest: "plan:v1".to_owned(),
            planned_sample_size: 100,
            planned_multiplicity_correction: MultiplicityCorrection::Holm,
        }
    }

    fn execution() -> StudyExecution {
        StudyExecution {
            study_id: "study-1".to_owned(), collection_started_unix_ms: 30, collection_ended_unix_ms: 40,
            enrolled: 110, completed: 100, excluded_after_collection: 2, hypotheses_tested: 2,
            reported_primary_outcomes: vec!["preference".to_owned()], participant_blinded: true,
            assessor_blinded: true, analyst_blinded_until_lock: true,
            manipulation_check_pass_rate: Some(0.90), applied_multiplicity_correction: MultiplicityCorrection::Holm,
            deviations: Vec::new(), analysis_plan_digest: "plan:v1".to_owned(),
        }
    }

    #[test]
    fn strong_execution_receives_high_quality() {
        let report = evaluate_study_quality(preregistration(), execution(), StudyQualityPolicy::production()).expect("report");
        assert_eq!(report.outcome, StudyQualityOutcome::High);
        report.validate().expect("valid report");
    }

    #[test]
    fn outcome_switching_and_plan_drift_fail() {
        let mut execution = execution();
        execution.reported_primary_outcomes = vec!["engagement".to_owned()];
        execution.analysis_plan_digest = "plan:changed".to_owned();
        let report = evaluate_study_quality(preregistration(), execution, StudyQualityPolicy::production()).expect("report");
        assert!(matches!(report.outcome, StudyQualityOutcome::Low | StudyQualityOutcome::Invalid));
    }

    #[test]
    fn forged_quality_outcome_is_rejected() {
        let mut report = evaluate_study_quality(preregistration(), execution(), StudyQualityPolicy::production()).expect("report");
        report.outcome = StudyQualityOutcome::Invalid;
        assert!(matches!(report.validate(), Err(StudyQualityError::DerivedReportMismatch)));
    }
}
