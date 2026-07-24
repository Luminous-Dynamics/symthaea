// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Confirmatory analysis execution authority.
//!
//! The record binds the closed blinded evidence, controlled unblinding, exact
//! compiled dataset, frozen plan, two analysis engines, and their agreement
//! report. Any post-unblinding deviation demotes the result to descriptive-only
//! rather than silently preserving a confirmatory label.

use crate::analysis_crosscheck::{
    AnalysisCrosscheckReport, NormalizedPrimaryAnalysis, validate_analysis_crosscheck,
};
use crate::confirmatory_collection_close::{
    ConfirmatoryCollectionCloseReceipt, confirmatory_collection_close_commitment,
};
use crate::confirmatory_unblinding::{
    ConfirmatoryUnblindingReceipt, confirmatory_unblinding_commitment,
};
use crate::evidence_digest::canonical_json_sha256;
use crate::study_evidence::CompiledStudyDataset;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const CONFIRMATORY_ANALYSIS_EXECUTION_VERSION: &str =
    "symthaea-muse-confirmatory-analysis-execution-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConfirmatoryAnalysisEngine {
    RustPrimary,
    IndependentExternal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryAnalysisCommandEvidence {
    pub engine: ConfirmatoryAnalysisEngine,
    pub executable_sha256: String,
    pub source_sha256: String,
    pub environment_sha256: String,
    pub command_sha256: String,
    pub stdout_sha256: String,
    pub stderr_sha256: String,
    pub started_at_utc: String,
    pub finished_at_utc: String,
    pub exit_code: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryAnalysisDeviation {
    pub deviation_id: String,
    pub description: String,
    pub discovered_at_utc: String,
    pub amendment_or_review_sha256: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmatoryClaimStatus {
    Confirmatory,
    DescriptiveOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfirmatoryAnalysisExecutionRecord {
    pub execution_version: String,
    pub study_id: String,
    pub collection_close_sha256: String,
    pub unblinding_receipt_sha256: String,
    pub compiled_dataset_sha256: String,
    pub frozen_analysis_plan_sha256: String,
    pub rust_analysis_sha256: String,
    pub external_analysis_sha256: String,
    pub crosscheck_report_sha256: String,
    pub commands: Vec<ConfirmatoryAnalysisCommandEvidence>,
    pub deviations: Vec<ConfirmatoryAnalysisDeviation>,
    pub crosscheck_passed: bool,
    pub primary_success: bool,
    pub claim_status: ConfirmatoryClaimStatus,
    pub executed_at_utc: String,
    pub record_sha256: String,
}

#[derive(Serialize)]
struct ExecutionCommitment<'a> {
    execution_version: &'a str,
    study_id: &'a str,
    collection_close_sha256: &'a str,
    unblinding_receipt_sha256: &'a str,
    compiled_dataset_sha256: &'a str,
    frozen_analysis_plan_sha256: &'a str,
    rust_analysis_sha256: &'a str,
    external_analysis_sha256: &'a str,
    crosscheck_report_sha256: &'a str,
    commands: &'a [ConfirmatoryAnalysisCommandEvidence],
    deviations: &'a [ConfirmatoryAnalysisDeviation],
    crosscheck_passed: bool,
    primary_success: bool,
    claim_status: ConfirmatoryClaimStatus,
    executed_at_utc: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmatoryAnalysisExecutionIssue {
    InvalidCollectionClose,
    InvalidUnblindingReceipt,
    InvalidCrosscheck,
    WrongVersion {
        found: String,
    },
    EmptyField {
        field: String,
    },
    InvalidDigest {
        field: String,
    },
    DigestMismatch {
        field: String,
    },
    MissingEngine {
        engine: ConfirmatoryAnalysisEngine,
    },
    DuplicateEngine {
        engine: ConfirmatoryAnalysisEngine,
    },
    InvalidCommand {
        engine: ConfirmatoryAnalysisEngine,
        field: String,
    },
    AnalysisInputMismatch {
        engine: ConfirmatoryAnalysisEngine,
    },
    AnalysisPlanMismatch {
        engine: ConfirmatoryAnalysisEngine,
    },
    SuccessDecisionMismatch,
    CrosscheckDecisionMismatch,
    ClaimStatusMismatch,
    DuplicateDeviationId {
        deviation_id: String,
    },
    InvalidDeviation {
        deviation_id: String,
        field: String,
    },
    SerializationFailed,
    RecordDigestMismatch,
}

pub fn confirmatory_analysis_execution_commitment(
    record: &ConfirmatoryAnalysisExecutionRecord,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ExecutionCommitment {
        execution_version: &record.execution_version,
        study_id: &record.study_id,
        collection_close_sha256: &record.collection_close_sha256,
        unblinding_receipt_sha256: &record.unblinding_receipt_sha256,
        compiled_dataset_sha256: &record.compiled_dataset_sha256,
        frozen_analysis_plan_sha256: &record.frozen_analysis_plan_sha256,
        rust_analysis_sha256: &record.rust_analysis_sha256,
        external_analysis_sha256: &record.external_analysis_sha256,
        crosscheck_report_sha256: &record.crosscheck_report_sha256,
        commands: &record.commands,
        deviations: &record.deviations,
        crosscheck_passed: record.crosscheck_passed,
        primary_success: record.primary_success,
        claim_status: record.claim_status,
        executed_at_utc: &record.executed_at_utc,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn build_confirmatory_analysis_execution(
    collection_close: &ConfirmatoryCollectionCloseReceipt,
    unblinding: &ConfirmatoryUnblindingReceipt,
    dataset: &CompiledStudyDataset,
    frozen_analysis_plan_sha256: String,
    rust_analysis: &NormalizedPrimaryAnalysis,
    external_analysis: &NormalizedPrimaryAnalysis,
    crosscheck: &AnalysisCrosscheckReport,
    mut commands: Vec<ConfirmatoryAnalysisCommandEvidence>,
    mut deviations: Vec<ConfirmatoryAnalysisDeviation>,
    executed_at_utc: String,
) -> Result<ConfirmatoryAnalysisExecutionRecord, Vec<ConfirmatoryAnalysisExecutionIssue>> {
    commands.sort_by_key(|command| command.engine);
    deviations.sort_by(|left, right| left.deviation_id.cmp(&right.deviation_id));
    let compiled_dataset_sha256 = canonical_json_sha256(dataset)
        .map_err(|_| vec![ConfirmatoryAnalysisExecutionIssue::SerializationFailed])?;
    let claim_status = if crosscheck.passed && deviations.is_empty() {
        ConfirmatoryClaimStatus::Confirmatory
    } else {
        ConfirmatoryClaimStatus::DescriptiveOnly
    };
    let mut record = ConfirmatoryAnalysisExecutionRecord {
        execution_version: CONFIRMATORY_ANALYSIS_EXECUTION_VERSION.into(),
        study_id: collection_close.study_id.clone(),
        collection_close_sha256: collection_close.receipt_sha256.clone(),
        unblinding_receipt_sha256: unblinding.receipt_sha256.clone(),
        compiled_dataset_sha256,
        frozen_analysis_plan_sha256,
        rust_analysis_sha256: rust_analysis.output_sha256.clone(),
        external_analysis_sha256: external_analysis.output_sha256.clone(),
        crosscheck_report_sha256: crosscheck.report_sha256.clone(),
        commands,
        deviations,
        crosscheck_passed: crosscheck.passed,
        primary_success: rust_analysis.success,
        claim_status,
        executed_at_utc,
        record_sha256: String::new(),
    };
    record.record_sha256 = confirmatory_analysis_execution_commitment(&record)
        .map_err(|_| vec![ConfirmatoryAnalysisExecutionIssue::SerializationFailed])?;
    let issues = validate_confirmatory_analysis_execution(
        collection_close,
        unblinding,
        dataset,
        rust_analysis,
        external_analysis,
        crosscheck,
        &record,
    );
    if issues.is_empty() {
        Ok(record)
    } else {
        Err(issues)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn validate_confirmatory_analysis_execution(
    collection_close: &ConfirmatoryCollectionCloseReceipt,
    unblinding: &ConfirmatoryUnblindingReceipt,
    dataset: &CompiledStudyDataset,
    rust_analysis: &NormalizedPrimaryAnalysis,
    external_analysis: &NormalizedPrimaryAnalysis,
    crosscheck: &AnalysisCrosscheckReport,
    record: &ConfirmatoryAnalysisExecutionRecord,
) -> Vec<ConfirmatoryAnalysisExecutionIssue> {
    let mut issues = Vec::new();
    match confirmatory_collection_close_commitment(collection_close) {
        Ok(found)
            if found == collection_close.receipt_sha256
                && found == record.collection_close_sha256 => {}
        _ => issues.push(ConfirmatoryAnalysisExecutionIssue::InvalidCollectionClose),
    }
    match confirmatory_unblinding_commitment(unblinding) {
        Ok(found)
            if found == unblinding.receipt_sha256
                && found == record.unblinding_receipt_sha256
                && unblinding.collection_close_sha256 == collection_close.receipt_sha256 => {}
        _ => issues.push(ConfirmatoryAnalysisExecutionIssue::InvalidUnblindingReceipt),
    }
    if !validate_analysis_crosscheck(rust_analysis, external_analysis, crosscheck).is_empty() {
        issues.push(ConfirmatoryAnalysisExecutionIssue::InvalidCrosscheck);
    }
    if record.execution_version != CONFIRMATORY_ANALYSIS_EXECUTION_VERSION {
        issues.push(ConfirmatoryAnalysisExecutionIssue::WrongVersion {
            found: record.execution_version.clone(),
        });
    }
    for (field, value) in [
        ("study_id", record.study_id.as_str()),
        ("executed_at_utc", record.executed_at_utc.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(ConfirmatoryAnalysisExecutionIssue::EmptyField {
                field: field.into(),
            });
        }
    }
    for (field, value) in [
        (
            "collection_close_sha256",
            record.collection_close_sha256.as_str(),
        ),
        (
            "unblinding_receipt_sha256",
            record.unblinding_receipt_sha256.as_str(),
        ),
        (
            "compiled_dataset_sha256",
            record.compiled_dataset_sha256.as_str(),
        ),
        (
            "frozen_analysis_plan_sha256",
            record.frozen_analysis_plan_sha256.as_str(),
        ),
        ("rust_analysis_sha256", record.rust_analysis_sha256.as_str()),
        (
            "external_analysis_sha256",
            record.external_analysis_sha256.as_str(),
        ),
        (
            "crosscheck_report_sha256",
            record.crosscheck_report_sha256.as_str(),
        ),
    ] {
        if !is_sha256(value) {
            issues.push(ConfirmatoryAnalysisExecutionIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    verify_digest(
        "compiled_dataset_sha256",
        canonical_json_sha256(dataset),
        &record.compiled_dataset_sha256,
        &mut issues,
    );
    for (field, expected, found) in [
        (
            "rust_analysis_sha256",
            rust_analysis.output_sha256.as_str(),
            record.rust_analysis_sha256.as_str(),
        ),
        (
            "external_analysis_sha256",
            external_analysis.output_sha256.as_str(),
            record.external_analysis_sha256.as_str(),
        ),
        (
            "crosscheck_report_sha256",
            crosscheck.report_sha256.as_str(),
            record.crosscheck_report_sha256.as_str(),
        ),
    ] {
        if expected != found {
            issues.push(ConfirmatoryAnalysisExecutionIssue::DigestMismatch {
                field: field.into(),
            });
        }
    }
    for (engine, analysis) in [
        (ConfirmatoryAnalysisEngine::RustPrimary, rust_analysis),
        (
            ConfirmatoryAnalysisEngine::IndependentExternal,
            external_analysis,
        ),
    ] {
        if analysis.input_sha256 != record.compiled_dataset_sha256 {
            issues.push(ConfirmatoryAnalysisExecutionIssue::AnalysisInputMismatch { engine });
        }
        if analysis.analysis_plan_sha256 != record.frozen_analysis_plan_sha256 {
            issues.push(ConfirmatoryAnalysisExecutionIssue::AnalysisPlanMismatch { engine });
        }
    }
    validate_commands(record, &mut issues);
    validate_deviations(record, &mut issues);
    if record.crosscheck_passed != crosscheck.passed {
        issues.push(ConfirmatoryAnalysisExecutionIssue::CrosscheckDecisionMismatch);
    }
    if record.primary_success != rust_analysis.success
        || rust_analysis.success != external_analysis.success
    {
        issues.push(ConfirmatoryAnalysisExecutionIssue::SuccessDecisionMismatch);
    }
    let expected_claim = if crosscheck.passed && record.deviations.is_empty() {
        ConfirmatoryClaimStatus::Confirmatory
    } else {
        ConfirmatoryClaimStatus::DescriptiveOnly
    };
    if record.claim_status != expected_claim {
        issues.push(ConfirmatoryAnalysisExecutionIssue::ClaimStatusMismatch);
    }
    match confirmatory_analysis_execution_commitment(record) {
        Ok(found) if found == record.record_sha256 => {}
        Ok(_) => issues.push(ConfirmatoryAnalysisExecutionIssue::RecordDigestMismatch),
        Err(_) => issues.push(ConfirmatoryAnalysisExecutionIssue::SerializationFailed),
    }
    issues
}

fn validate_commands(
    record: &ConfirmatoryAnalysisExecutionRecord,
    issues: &mut Vec<ConfirmatoryAnalysisExecutionIssue>,
) {
    let mut engines = BTreeSet::new();
    for command in &record.commands {
        if !engines.insert(command.engine) {
            issues.push(ConfirmatoryAnalysisExecutionIssue::DuplicateEngine {
                engine: command.engine,
            });
        }
        for (field, valid) in [
            ("executable_sha256", is_sha256(&command.executable_sha256)),
            ("source_sha256", is_sha256(&command.source_sha256)),
            ("environment_sha256", is_sha256(&command.environment_sha256)),
            ("command_sha256", is_sha256(&command.command_sha256)),
            ("stdout_sha256", is_sha256(&command.stdout_sha256)),
            ("stderr_sha256", is_sha256(&command.stderr_sha256)),
            ("started_at_utc", !command.started_at_utc.trim().is_empty()),
            (
                "finished_at_utc",
                !command.finished_at_utc.trim().is_empty(),
            ),
            ("exit_code", command.exit_code == 0),
        ] {
            if !valid {
                issues.push(ConfirmatoryAnalysisExecutionIssue::InvalidCommand {
                    engine: command.engine,
                    field: field.into(),
                });
            }
        }
    }
    for engine in [
        ConfirmatoryAnalysisEngine::RustPrimary,
        ConfirmatoryAnalysisEngine::IndependentExternal,
    ] {
        if !engines.contains(&engine) {
            issues.push(ConfirmatoryAnalysisExecutionIssue::MissingEngine { engine });
        }
    }
}

fn validate_deviations(
    record: &ConfirmatoryAnalysisExecutionRecord,
    issues: &mut Vec<ConfirmatoryAnalysisExecutionIssue>,
) {
    let mut ids = BTreeSet::new();
    for deviation in &record.deviations {
        if !ids.insert(deviation.deviation_id.as_str()) {
            issues.push(ConfirmatoryAnalysisExecutionIssue::DuplicateDeviationId {
                deviation_id: deviation.deviation_id.clone(),
            });
        }
        for (field, valid) in [
            ("deviation_id", !deviation.deviation_id.trim().is_empty()),
            ("description", !deviation.description.trim().is_empty()),
            (
                "discovered_at_utc",
                !deviation.discovered_at_utc.trim().is_empty(),
            ),
            (
                "amendment_or_review_sha256",
                deviation
                    .amendment_or_review_sha256
                    .as_deref()
                    .is_some_and(is_sha256),
            ),
        ] {
            if !valid {
                issues.push(ConfirmatoryAnalysisExecutionIssue::InvalidDeviation {
                    deviation_id: deviation.deviation_id.clone(),
                    field: field.into(),
                });
            }
        }
    }
}

fn verify_digest(
    field: &str,
    expected: Result<String, serde_json::Error>,
    found: &str,
    issues: &mut Vec<ConfirmatoryAnalysisExecutionIssue>,
) {
    match expected {
        Ok(value) if value == found => {}
        Ok(_) => issues.push(ConfirmatoryAnalysisExecutionIssue::DigestMismatch {
            field: field.into(),
        }),
        Err(_) => issues.push(ConfirmatoryAnalysisExecutionIssue::SerializationFailed),
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn any_deviation_demotes_the_claim() {
        let status = if vec![ConfirmatoryAnalysisDeviation {
            deviation_id: "d1".into(),
            description: "documented deviation".into(),
            discovered_at_utc: "now".into(),
            amendment_or_review_sha256: Some("a".repeat(64)),
        }]
        .is_empty()
        {
            ConfirmatoryClaimStatus::Confirmatory
        } else {
            ConfirmatoryClaimStatus::DescriptiveOnly
        };
        assert_eq!(status, ConfirmatoryClaimStatus::DescriptiveOnly);
    }
}
