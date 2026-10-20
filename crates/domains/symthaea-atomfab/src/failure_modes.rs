//! Structured failure-mode evaluation for atomistic and process-model claims.
//!
//! Aggregate MAE can hide a small number of catastrophic prediction failures,
//! family-specific transfer failures, or dangerous cases that should have been
//! abstentions. This module records each disposition and summarizes outcomes
//! only within one exact campaign/model/evaluator/material-family/split scope.
//!
//! Failure modes and abstentions are reported separately. An out-of-domain
//! abstention is not counted as a failed prediction; a model that emits an
//! out-of-domain recommendation is a failure. This is a reporting primitive,
//! not a scientific solver or physical-safety qualification.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const FAILURE_MODE_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureMode {
    NonFiniteEnergy,
    NonFiniteForce,
    SolverNonConvergence,
    GeometryCollapse,
    InteratomicOverlap,
    UnsupportedSpecies,
    OutOfDomain,
    ForceSpike,
    EnergyOrderingViolation,
    ObservableViolation,
    UncertaintyUndercoverage,
    InterfaceBreakdown,
    InfeasibleProcess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureSeverity {
    Informational,
    Degraded,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureDisposition {
    Pass,
    Failed,
    Abstained,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureAssessmentBasis {
    Reference,
    Simulation,
    Measurement,
    IndependentReproduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationSplit {
    Validation,
    Test,
    FamilyHoldout,
    Prospective,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FailureRecord {
    pub record_id: String,
    pub campaign_id: String,
    pub model_id: String,
    pub model_digest: String,
    pub evaluator_digest: String,
    pub evaluation_scope_digest: String,
    pub material_family: String,
    /// Frozen stratum such as particle size × support surface × temperature regime.
    /// Summaries must not pool distinct strata into one headline error rate.
    pub evaluation_stratum: String,
    pub split: EvaluationSplit,
    pub parent_system_id: String,
    pub sample_id: String,
    pub assessment_basis: FailureAssessmentBasis,
    pub disposition: FailureDisposition,
    pub failure_mode: Option<FailureMode>,
    pub severity: Option<FailureSeverity>,
    pub evidence_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailureModeError {
    EmptyRecordSet,
    EmptyRequiredField(&'static str),
    InvalidDigest(&'static str),
    DuplicateRecordId,
    MismatchedEvaluationScope,
    InvalidDispositionShape,
    UnsupportedSchemaVersion(u16),
    InvalidSummary,
}

impl fmt::Display for FailureModeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRecordSet => write!(f, "failure-mode summary needs at least one record"),
            Self::EmptyRequiredField(field) => {
                write!(f, "failure record field must not be empty: {field}")
            }
            Self::InvalidDigest(field) => {
                write!(
                    f,
                    "failure record field is not a lowercase SHA-256 digest: {field}"
                )
            }
            Self::DuplicateRecordId => write!(f, "failure record IDs must be unique"),
            Self::MismatchedEvaluationScope => write!(
                f,
                "failure records must share exact campaign, model, evaluator, family, split, scope and evidence basis"
            ),
            Self::InvalidDispositionShape => write!(
                f,
                "pass requires no failure mode or severity; failed/abstained require both"
            ),
            Self::UnsupportedSchemaVersion(version) => {
                write!(
                    f,
                    "unsupported failure-mode summary schema version: {version}"
                )
            }
            Self::InvalidSummary => write!(
                f,
                "failure-mode summary does not match canonical recomputation"
            ),
        }
    }
}

impl Error for FailureModeError {}

impl FailureRecord {
    fn validate(&self) -> Result<(), FailureModeError> {
        for (value, name) in [
            (&self.record_id, "record_id"),
            (&self.campaign_id, "campaign_id"),
            (&self.model_id, "model_id"),
            (&self.material_family, "material_family"),
            (&self.evaluation_stratum, "evaluation_stratum"),
            (&self.parent_system_id, "parent_system_id"),
            (&self.sample_id, "sample_id"),
        ] {
            if value.trim().is_empty() {
                return Err(FailureModeError::EmptyRequiredField(name));
            }
        }
        for (value, name) in [
            (&self.model_digest, "model_digest"),
            (&self.evaluator_digest, "evaluator_digest"),
            (&self.evaluation_scope_digest, "evaluation_scope_digest"),
            (&self.evidence_digest, "evidence_digest"),
        ] {
            validate_digest(value, name)?;
        }
        match (self.disposition, self.failure_mode, self.severity) {
            (FailureDisposition::Pass, None, None) => Ok(()),
            (FailureDisposition::Failed, Some(_), Some(_)) => Ok(()),
            (FailureDisposition::Abstained, Some(_), Some(_)) => Ok(()),
            _ => Err(FailureModeError::InvalidDispositionShape),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FailureModeSummary {
    pub schema_version: u16,
    pub campaign_id: String,
    pub model_id: String,
    pub model_digest: String,
    pub evaluator_digest: String,
    pub evaluation_scope_digest: String,
    pub material_family: String,
    pub evaluation_stratum: String,
    pub split: EvaluationSplit,
    pub assessment_basis: FailureAssessmentBasis,
    pub total_assessed: u64,
    pub passed_count: u64,
    pub failed_count: u64,
    pub abstained_count: u64,
    pub critical_failure_count: u64,
    pub failure_mode_counts: BTreeMap<FailureMode, u64>,
    pub abstention_reason_counts: BTreeMap<FailureMode, u64>,
    pub severity_counts: BTreeMap<FailureSeverity, u64>,
    pub record_ids: Vec<String>,
    pub evidence_digests: Vec<String>,
}

impl FailureModeSummary {
    /// Validate a persisted summary by rebuilding it from the exact records.
    pub fn validate(&self, records: &[FailureRecord]) -> Result<(), FailureModeError> {
        if self.schema_version != FAILURE_MODE_SCHEMA_VERSION {
            return Err(FailureModeError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        let canonical = summarize_failure_modes(records)?;
        if canonical != *self {
            return Err(FailureModeError::InvalidSummary);
        }
        Ok(())
    }
}

/// Summarize one matched model/family/stratum/split campaign scope, never a
/// mixture of unlike models, support surfaces, particle sizes or evaluation
/// partitions. Counts retain their denominators;
/// no aggregate mean may hide a critical failure. Abstentions are not failures,
/// though their reasons remain visible.
pub fn summarize_failure_modes(
    records: &[FailureRecord],
) -> Result<FailureModeSummary, FailureModeError> {
    let first = records.first().ok_or(FailureModeError::EmptyRecordSet)?;
    for record in records {
        record.validate()?;
    }

    let mut ids = BTreeSet::new();
    for record in records {
        if !ids.insert(record.record_id.as_str()) {
            return Err(FailureModeError::DuplicateRecordId);
        }
        if record.campaign_id != first.campaign_id
            || record.model_id != first.model_id
            || record.model_digest != first.model_digest
            || record.evaluator_digest != first.evaluator_digest
            || record.evaluation_scope_digest != first.evaluation_scope_digest
            || record.material_family != first.material_family
            || record.evaluation_stratum != first.evaluation_stratum
            || record.split != first.split
            || record.assessment_basis != first.assessment_basis
        {
            return Err(FailureModeError::MismatchedEvaluationScope);
        }
    }

    let total_assessed = u64::try_from(records.len()).unwrap_or(u64::MAX);
    let passed_count = records
        .iter()
        .filter(|record| record.disposition == FailureDisposition::Pass)
        .count() as u64;
    let failed_count = records
        .iter()
        .filter(|record| record.disposition == FailureDisposition::Failed)
        .count() as u64;
    let abstained_count = records
        .iter()
        .filter(|record| record.disposition == FailureDisposition::Abstained)
        .count() as u64;
    let critical_failure_count = records
        .iter()
        .filter(|record| {
            record.disposition == FailureDisposition::Failed
                && record.severity == Some(FailureSeverity::Critical)
        })
        .count() as u64;

    let mut failure_mode_counts = BTreeMap::new();
    let mut abstention_reason_counts = BTreeMap::new();
    let mut severity_counts = BTreeMap::new();
    for record in records {
        match record.disposition {
            FailureDisposition::Pass => {}
            FailureDisposition::Failed => {
                *failure_mode_counts
                    .entry(record.failure_mode.expect("record validated"))
                    .or_insert(0) += 1;
                *severity_counts
                    .entry(record.severity.expect("record validated"))
                    .or_insert(0) += 1;
            }
            FailureDisposition::Abstained => {
                *abstention_reason_counts
                    .entry(record.failure_mode.expect("record validated"))
                    .or_insert(0) += 1;
            }
        }
    }

    let mut record_ids: Vec<String> = records
        .iter()
        .map(|record| record.record_id.clone())
        .collect();
    record_ids.sort();
    let mut evidence_digests: Vec<String> = records
        .iter()
        .map(|record| record.evidence_digest.clone())
        .collect();
    evidence_digests.sort();

    Ok(FailureModeSummary {
        schema_version: FAILURE_MODE_SCHEMA_VERSION,
        campaign_id: first.campaign_id.clone(),
        model_id: first.model_id.clone(),
        model_digest: first.model_digest.clone(),
        evaluator_digest: first.evaluator_digest.clone(),
        evaluation_scope_digest: first.evaluation_scope_digest.clone(),
        material_family: first.material_family.clone(),
        evaluation_stratum: first.evaluation_stratum.clone(),
        split: first.split,
        assessment_basis: first.assessment_basis,
        total_assessed,
        passed_count,
        failed_count,
        abstained_count,
        critical_failure_count,
        failure_mode_counts,
        abstention_reason_counts,
        severity_counts,
        record_ids,
        evidence_digests,
    })
}

fn validate_digest(value: &str, field: &'static str) -> Result<(), FailureModeError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Err(FailureModeError::InvalidDigest(field))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_keeps_failure_modes_and_abstentions_separate() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../fixtures/failure-mode-records-v1.json"))
                .expect("failure fixture must parse");
        let _record_schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schemas/failure-mode-records-v1.schema.json"
        ))
        .expect("failure record schema must parse");
        let _summary_schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schemas/failure-mode-summary-v1.schema.json"
        ))
        .expect("failure summary schema must parse");
        let records: Vec<FailureRecord> = serde_json::from_value(fixture["records"].clone())
            .expect("failure records should match Rust contract");
        let summary = summarize_failure_modes(&records).expect("fixture scope is matched");
        assert_eq!(summary.total_assessed, 4);
        assert_eq!(summary.passed_count, 1);
        assert_eq!(summary.failed_count, 2);
        assert_eq!(summary.abstained_count, 1);
        assert_eq!(summary.critical_failure_count, 1);
        assert_eq!(
            summary
                .failure_mode_counts
                .get(&FailureMode::NonFiniteForce),
            Some(&1)
        );
        assert_eq!(
            summary
                .failure_mode_counts
                .get(&FailureMode::EnergyOrderingViolation),
            Some(&1)
        );
        assert_eq!(
            summary
                .abstention_reason_counts
                .get(&FailureMode::OutOfDomain),
            Some(&1)
        );
        assert_eq!(
            summary.severity_counts.get(&FailureSeverity::Critical),
            Some(&1)
        );
        summary.validate(&records).expect("summary must recompute");

        let mut forged = summary.clone();
        forged.critical_failure_count = 0;
        assert_eq!(
            forged.validate(&records),
            Err(FailureModeError::InvalidSummary)
        );
    }

    #[test]
    fn summaries_do_not_pool_models_families_or_splits() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../fixtures/failure-mode-records-v1.json")).unwrap();
        let records: Vec<FailureRecord> =
            serde_json::from_value(fixture["records"].clone()).unwrap();
        let mut other_model = records.clone();
        other_model[0].model_id = "different-model".to_owned();
        assert_eq!(
            summarize_failure_modes(&other_model),
            Err(FailureModeError::MismatchedEvaluationScope)
        );

        let mut other_stratum = records.clone();
        other_stratum[0].evaluation_stratum = "cu38_alpha_alumina_0001_fixture".to_owned();
        assert_eq!(
            summarize_failure_modes(&other_stratum),
            Err(FailureModeError::MismatchedEvaluationScope)
        );

        let mut other_split = records;
        other_split[0].split = EvaluationSplit::Test;
        assert_eq!(
            summarize_failure_modes(&other_split),
            Err(FailureModeError::MismatchedEvaluationScope)
        );
    }

    #[test]
    fn malformed_disposition_and_duplicate_id_are_rejected() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../fixtures/failure-mode-records-v1.json")).unwrap();
        let mut records: Vec<FailureRecord> =
            serde_json::from_value(fixture["records"].clone()).unwrap();
        records[0].failure_mode = Some(FailureMode::GeometryCollapse);
        assert_eq!(
            summarize_failure_modes(&records),
            Err(FailureModeError::InvalidDispositionShape)
        );

        records[0].failure_mode = None;
        records[1].record_id = records[0].record_id.clone();
        assert_eq!(
            summarize_failure_modes(&records),
            Err(FailureModeError::DuplicateRecordId)
        );
    }
}
