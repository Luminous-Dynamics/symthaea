// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Root release commitment for external review and confirmatory readiness.

use crate::confirmatory_amendment_control::ConfirmatoryAmendmentLedger;
use crate::confirmatory_readiness::{
    ConfirmatoryDryRunEvidence, ConfirmatoryReadinessReport, HumanStudyGovernanceEvidence,
    IndependentReproductionReadiness, WorkspaceValidationEvidence,
};
use crate::evidence_digest::canonical_json_sha256;
use crate::external_review_completion::ExternalReviewCompletionEvidence;
use crate::external_review_package::{ExternalReviewPackage, ReviewEvidenceIndex};
use crate::external_review_protocol::FrozenExternalReviewProtocol;
use crate::external_review_resolution::ExternalReviewResolutionLedger;
use crate::external_review_response::ExternalReviewResponse;
use crate::study_operations_release::StudyOperationsReleaseBundle;
use serde::{Deserialize, Serialize};

pub const CONFIRMATORY_READINESS_RELEASE_VERSION: &str =
    "symthaea-muse-confirmatory-readiness-release-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryReadinessReleaseBundle {
    pub release_version: String,
    pub study_operations_release_sha256: String,
    pub external_review_protocol_sha256: String,
    pub review_evidence_index_sha256: String,
    pub review_package_set_sha256: String,
    pub review_response_set_sha256: String,
    pub review_resolution_ledger_sha256: String,
    pub review_completion_sha256: String,
    pub confirmatory_amendment_ledger_sha256: String,
    pub workspace_validation_sha256: String,
    pub human_governance_sha256: String,
    pub dry_run_sha256: String,
    pub independent_reproduction_sha256: String,
    pub readiness_report_sha256: String,
    pub source_archive_sha256: String,
    pub flake_lock_sha256: String,
    pub toolchain_evidence_sha256: String,
    pub external_timestamp_receipt_sha256: String,
    pub release_sha256: String,
}

#[derive(Serialize)]
struct ConfirmatoryReadinessReleaseCommitment<'a> {
    release_version: &'a str,
    study_operations_release_sha256: &'a str,
    external_review_protocol_sha256: &'a str,
    review_evidence_index_sha256: &'a str,
    review_package_set_sha256: &'a str,
    review_response_set_sha256: &'a str,
    review_resolution_ledger_sha256: &'a str,
    review_completion_sha256: &'a str,
    confirmatory_amendment_ledger_sha256: &'a str,
    workspace_validation_sha256: &'a str,
    human_governance_sha256: &'a str,
    dry_run_sha256: &'a str,
    independent_reproduction_sha256: &'a str,
    readiness_report_sha256: &'a str,
    source_archive_sha256: &'a str,
    flake_lock_sha256: &'a str,
    toolchain_evidence_sha256: &'a str,
    external_timestamp_receipt_sha256: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmatoryReadinessReleaseIssue {
    WrongVersion { found: String },
    SerializationFailed { field: String },
    InvalidDigest { field: String },
    DigestMismatch { field: String },
    ReleaseDigestMismatch,
}

#[allow(clippy::too_many_arguments)]
pub fn build_confirmatory_readiness_release(
    operations_release: &StudyOperationsReleaseBundle,
    review_protocol: &FrozenExternalReviewProtocol,
    review_index: &ReviewEvidenceIndex,
    review_packages: &[ExternalReviewPackage],
    review_responses: &[ExternalReviewResponse],
    review_resolution: &ExternalReviewResolutionLedger,
    review_completion: &ExternalReviewCompletionEvidence,
    amendments: &ConfirmatoryAmendmentLedger,
    workspace: &WorkspaceValidationEvidence,
    governance: &HumanStudyGovernanceEvidence,
    dry_run: &ConfirmatoryDryRunEvidence,
    reproduction: &IndependentReproductionReadiness,
    readiness: &ConfirmatoryReadinessReport,
    source_archive_sha256: String,
    flake_lock_sha256: String,
    toolchain_evidence_sha256: String,
    external_timestamp_receipt_sha256: String,
) -> Result<ConfirmatoryReadinessReleaseBundle, serde_json::Error> {
    let mut package_digests = review_packages
        .iter()
        .map(|package| package.package_sha256.clone())
        .collect::<Vec<_>>();
    package_digests.sort();
    let mut response_digests = review_responses
        .iter()
        .map(|response| response.response_sha256.clone())
        .collect::<Vec<_>>();
    response_digests.sort();
    let mut bundle = ConfirmatoryReadinessReleaseBundle {
        release_version: CONFIRMATORY_READINESS_RELEASE_VERSION.into(),
        study_operations_release_sha256: canonical_json_sha256(operations_release)?,
        external_review_protocol_sha256: canonical_json_sha256(review_protocol)?,
        review_evidence_index_sha256: canonical_json_sha256(review_index)?,
        review_package_set_sha256: canonical_json_sha256(&package_digests)?,
        review_response_set_sha256: canonical_json_sha256(&response_digests)?,
        review_resolution_ledger_sha256: canonical_json_sha256(review_resolution)?,
        review_completion_sha256: canonical_json_sha256(review_completion)?,
        confirmatory_amendment_ledger_sha256: canonical_json_sha256(amendments)?,
        workspace_validation_sha256: canonical_json_sha256(workspace)?,
        human_governance_sha256: canonical_json_sha256(governance)?,
        dry_run_sha256: canonical_json_sha256(dry_run)?,
        independent_reproduction_sha256: canonical_json_sha256(reproduction)?,
        readiness_report_sha256: canonical_json_sha256(readiness)?,
        source_archive_sha256,
        flake_lock_sha256,
        toolchain_evidence_sha256,
        external_timestamp_receipt_sha256,
        release_sha256: String::new(),
    };
    bundle.release_sha256 = confirmatory_readiness_release_commitment(&bundle)?;
    Ok(bundle)
}

pub fn confirmatory_readiness_release_commitment(
    bundle: &ConfirmatoryReadinessReleaseBundle,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ConfirmatoryReadinessReleaseCommitment {
        release_version: &bundle.release_version,
        study_operations_release_sha256: &bundle.study_operations_release_sha256,
        external_review_protocol_sha256: &bundle.external_review_protocol_sha256,
        review_evidence_index_sha256: &bundle.review_evidence_index_sha256,
        review_package_set_sha256: &bundle.review_package_set_sha256,
        review_response_set_sha256: &bundle.review_response_set_sha256,
        review_resolution_ledger_sha256: &bundle.review_resolution_ledger_sha256,
        review_completion_sha256: &bundle.review_completion_sha256,
        confirmatory_amendment_ledger_sha256: &bundle.confirmatory_amendment_ledger_sha256,
        workspace_validation_sha256: &bundle.workspace_validation_sha256,
        human_governance_sha256: &bundle.human_governance_sha256,
        dry_run_sha256: &bundle.dry_run_sha256,
        independent_reproduction_sha256: &bundle.independent_reproduction_sha256,
        readiness_report_sha256: &bundle.readiness_report_sha256,
        source_archive_sha256: &bundle.source_archive_sha256,
        flake_lock_sha256: &bundle.flake_lock_sha256,
        toolchain_evidence_sha256: &bundle.toolchain_evidence_sha256,
        external_timestamp_receipt_sha256: &bundle.external_timestamp_receipt_sha256,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn validate_confirmatory_readiness_release(
    operations_release: &StudyOperationsReleaseBundle,
    review_protocol: &FrozenExternalReviewProtocol,
    review_index: &ReviewEvidenceIndex,
    review_packages: &[ExternalReviewPackage],
    review_responses: &[ExternalReviewResponse],
    review_resolution: &ExternalReviewResolutionLedger,
    review_completion: &ExternalReviewCompletionEvidence,
    amendments: &ConfirmatoryAmendmentLedger,
    workspace: &WorkspaceValidationEvidence,
    governance: &HumanStudyGovernanceEvidence,
    dry_run: &ConfirmatoryDryRunEvidence,
    reproduction: &IndependentReproductionReadiness,
    readiness: &ConfirmatoryReadinessReport,
    bundle: &ConfirmatoryReadinessReleaseBundle,
) -> Vec<ConfirmatoryReadinessReleaseIssue> {
    let mut issues = Vec::new();
    if bundle.release_version != CONFIRMATORY_READINESS_RELEASE_VERSION {
        issues.push(ConfirmatoryReadinessReleaseIssue::WrongVersion {
            found: bundle.release_version.clone(),
        });
    }
    verify_digest(
        operations_release,
        &bundle.study_operations_release_sha256,
        "study_operations_release_sha256",
        &mut issues,
    );
    verify_digest(
        review_protocol,
        &bundle.external_review_protocol_sha256,
        "external_review_protocol_sha256",
        &mut issues,
    );
    verify_digest(
        review_index,
        &bundle.review_evidence_index_sha256,
        "review_evidence_index_sha256",
        &mut issues,
    );
    let mut package_digests = review_packages
        .iter()
        .map(|package| package.package_sha256.clone())
        .collect::<Vec<_>>();
    package_digests.sort();
    verify_digest(
        &package_digests,
        &bundle.review_package_set_sha256,
        "review_package_set_sha256",
        &mut issues,
    );
    let mut response_digests = review_responses
        .iter()
        .map(|response| response.response_sha256.clone())
        .collect::<Vec<_>>();
    response_digests.sort();
    verify_digest(
        &response_digests,
        &bundle.review_response_set_sha256,
        "review_response_set_sha256",
        &mut issues,
    );
    verify_digest(
        review_resolution,
        &bundle.review_resolution_ledger_sha256,
        "review_resolution_ledger_sha256",
        &mut issues,
    );
    verify_digest(
        review_completion,
        &bundle.review_completion_sha256,
        "review_completion_sha256",
        &mut issues,
    );
    verify_digest(
        amendments,
        &bundle.confirmatory_amendment_ledger_sha256,
        "confirmatory_amendment_ledger_sha256",
        &mut issues,
    );
    verify_digest(
        workspace,
        &bundle.workspace_validation_sha256,
        "workspace_validation_sha256",
        &mut issues,
    );
    verify_digest(
        governance,
        &bundle.human_governance_sha256,
        "human_governance_sha256",
        &mut issues,
    );
    verify_digest(
        dry_run,
        &bundle.dry_run_sha256,
        "dry_run_sha256",
        &mut issues,
    );
    verify_digest(
        reproduction,
        &bundle.independent_reproduction_sha256,
        "independent_reproduction_sha256",
        &mut issues,
    );
    verify_digest(
        readiness,
        &bundle.readiness_report_sha256,
        "readiness_report_sha256",
        &mut issues,
    );
    for (field, digest) in [
        (
            "source_archive_sha256",
            bundle.source_archive_sha256.as_str(),
        ),
        ("flake_lock_sha256", bundle.flake_lock_sha256.as_str()),
        (
            "toolchain_evidence_sha256",
            bundle.toolchain_evidence_sha256.as_str(),
        ),
        (
            "external_timestamp_receipt_sha256",
            bundle.external_timestamp_receipt_sha256.as_str(),
        ),
        ("release_sha256", bundle.release_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(ConfirmatoryReadinessReleaseIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    match confirmatory_readiness_release_commitment(bundle) {
        Ok(value) if value == bundle.release_sha256 => {}
        Ok(_) => issues.push(ConfirmatoryReadinessReleaseIssue::ReleaseDigestMismatch),
        Err(_) => issues.push(ConfirmatoryReadinessReleaseIssue::SerializationFailed {
            field: "release".into(),
        }),
    }
    issues
}

fn verify_digest<T: Serialize>(
    value: &T,
    expected: &str,
    field: &str,
    issues: &mut Vec<ConfirmatoryReadinessReleaseIssue>,
) {
    match canonical_json_sha256(value) {
        Ok(value) if value == expected => {}
        Ok(_) => issues.push(ConfirmatoryReadinessReleaseIssue::DigestMismatch {
            field: field.into(),
        }),
        Err(_) => issues.push(ConfirmatoryReadinessReleaseIssue::SerializationFailed {
            field: field.into(),
        }),
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_version_is_explicit() {
        assert!(CONFIRMATORY_READINESS_RELEASE_VERSION.ends_with("v1"));
    }
}
