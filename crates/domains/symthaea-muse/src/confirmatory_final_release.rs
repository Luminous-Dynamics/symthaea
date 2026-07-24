// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Root commitment for the complete confirmatory execution and publication.
//!
//! This release root links readiness, collection, unblinding, analysis,
//! publication, and the initial post-publication audit ledger. It cannot make a
//! result true; it makes substitution or omission of an authority detectable.

use crate::confirmatory_analysis_execution::{
    ConfirmatoryAnalysisExecutionRecord, confirmatory_analysis_execution_commitment,
};
use crate::confirmatory_collection_close::{
    ConfirmatoryCollectionCloseReceipt, confirmatory_collection_close_commitment,
};
use crate::confirmatory_collection_protocol::{
    ConfirmatoryCollectionProtocol, confirmatory_collection_protocol_commitment,
};
use crate::confirmatory_publication::{
    ConfirmatoryPublicationRecord, confirmatory_publication_commitment,
};
use crate::confirmatory_readiness_release::{
    ConfirmatoryReadinessReleaseBundle, confirmatory_readiness_release_commitment,
};
use crate::confirmatory_unblinding::{
    ConfirmatoryUnblindingReceipt, confirmatory_unblinding_commitment,
};
use crate::evidence_digest::canonical_json_sha256;
use crate::post_publication_audit::{
    PostPublicationAuditLedger, post_publication_audit_commitment,
};
use crate::study_orchestration::{
    StudyLifecyclePhase, StudyOrchestrationLog, validate_study_orchestration,
};
use crate::study_release::{StudyReleaseBundle, study_release_commitment};
use serde::{Deserialize, Serialize};

pub const CONFIRMATORY_FINAL_RELEASE_VERSION: &str = "symthaea-muse-confirmatory-final-release-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmatoryFinalReleaseBundle {
    pub release_version: String,
    pub study_id: String,
    pub readiness_release_sha256: String,
    pub collection_protocol_sha256: String,
    pub collection_close_sha256: String,
    pub unblinding_receipt_sha256: String,
    pub analysis_execution_sha256: String,
    pub publication_record_sha256: String,
    pub post_publication_audit_sha256: String,
    pub study_release_bundle_sha256: String,
    pub orchestration_log_sha256: String,
    pub source_revision: String,
    pub workspace_tree_sha256: String,
    pub execution_environment_sha256: String,
    pub public_release_uri: String,
    pub released_at_utc: String,
    pub bundle_sha256: String,
}

#[derive(Serialize)]
struct FinalReleaseCommitment<'a> {
    release_version: &'a str,
    study_id: &'a str,
    readiness_release_sha256: &'a str,
    collection_protocol_sha256: &'a str,
    collection_close_sha256: &'a str,
    unblinding_receipt_sha256: &'a str,
    analysis_execution_sha256: &'a str,
    publication_record_sha256: &'a str,
    post_publication_audit_sha256: &'a str,
    study_release_bundle_sha256: &'a str,
    orchestration_log_sha256: &'a str,
    source_revision: &'a str,
    workspace_tree_sha256: &'a str,
    execution_environment_sha256: &'a str,
    public_release_uri: &'a str,
    released_at_utc: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmatoryFinalReleaseIssue {
    InvalidReadinessRelease,
    InvalidCollectionProtocol,
    InvalidCollectionClose,
    InvalidUnblindingReceipt,
    InvalidAnalysisExecution,
    InvalidPublicationRecord,
    InvalidPostPublicationAudit,
    InvalidStudyRelease,
    InvalidOrchestration,
    OrchestrationNotPublished,
    WrongVersion { found: String },
    EmptyField { field: String },
    InvalidDigest { field: String },
    CrossAuthorityMismatch { field: String },
    SerializationFailed,
    BundleDigestMismatch,
}

#[allow(clippy::too_many_arguments)]
pub fn build_confirmatory_final_release(
    readiness: &ConfirmatoryReadinessReleaseBundle,
    protocol: &ConfirmatoryCollectionProtocol,
    close: &ConfirmatoryCollectionCloseReceipt,
    unblinding: &ConfirmatoryUnblindingReceipt,
    analysis: &ConfirmatoryAnalysisExecutionRecord,
    publication: &ConfirmatoryPublicationRecord,
    audit: &PostPublicationAuditLedger,
    study_release: &StudyReleaseBundle,
    orchestration: &StudyOrchestrationLog,
    public_release_uri: String,
    released_at_utc: String,
) -> Result<ConfirmatoryFinalReleaseBundle, Vec<ConfirmatoryFinalReleaseIssue>> {
    let mut bundle = ConfirmatoryFinalReleaseBundle {
        release_version: CONFIRMATORY_FINAL_RELEASE_VERSION.into(),
        study_id: publication.study_id.clone(),
        readiness_release_sha256: readiness.release_sha256.clone(),
        collection_protocol_sha256: protocol.protocol_sha256.clone(),
        collection_close_sha256: close.receipt_sha256.clone(),
        unblinding_receipt_sha256: unblinding.receipt_sha256.clone(),
        analysis_execution_sha256: analysis.record_sha256.clone(),
        publication_record_sha256: publication.record_sha256.clone(),
        post_publication_audit_sha256: audit.ledger_sha256.clone(),
        study_release_bundle_sha256: study_release.bundle_sha256.clone(),
        orchestration_log_sha256: orchestration.log_sha256.clone(),
        source_revision: study_release.source_revision.clone(),
        workspace_tree_sha256: study_release.workspace_tree_sha256.clone(),
        execution_environment_sha256: study_release.execution_environment_sha256.clone(),
        public_release_uri,
        released_at_utc,
        bundle_sha256: String::new(),
    };
    bundle.bundle_sha256 = confirmatory_final_release_commitment(&bundle)
        .map_err(|_| vec![ConfirmatoryFinalReleaseIssue::SerializationFailed])?;
    let issues = validate_confirmatory_final_release(
        readiness,
        protocol,
        close,
        unblinding,
        analysis,
        publication,
        audit,
        study_release,
        orchestration,
        &bundle,
    );
    if issues.is_empty() {
        Ok(bundle)
    } else {
        Err(issues)
    }
}

pub fn confirmatory_final_release_commitment(
    bundle: &ConfirmatoryFinalReleaseBundle,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&FinalReleaseCommitment {
        release_version: &bundle.release_version,
        study_id: &bundle.study_id,
        readiness_release_sha256: &bundle.readiness_release_sha256,
        collection_protocol_sha256: &bundle.collection_protocol_sha256,
        collection_close_sha256: &bundle.collection_close_sha256,
        unblinding_receipt_sha256: &bundle.unblinding_receipt_sha256,
        analysis_execution_sha256: &bundle.analysis_execution_sha256,
        publication_record_sha256: &bundle.publication_record_sha256,
        post_publication_audit_sha256: &bundle.post_publication_audit_sha256,
        study_release_bundle_sha256: &bundle.study_release_bundle_sha256,
        orchestration_log_sha256: &bundle.orchestration_log_sha256,
        source_revision: &bundle.source_revision,
        workspace_tree_sha256: &bundle.workspace_tree_sha256,
        execution_environment_sha256: &bundle.execution_environment_sha256,
        public_release_uri: &bundle.public_release_uri,
        released_at_utc: &bundle.released_at_utc,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn validate_confirmatory_final_release(
    readiness: &ConfirmatoryReadinessReleaseBundle,
    protocol: &ConfirmatoryCollectionProtocol,
    close: &ConfirmatoryCollectionCloseReceipt,
    unblinding: &ConfirmatoryUnblindingReceipt,
    analysis: &ConfirmatoryAnalysisExecutionRecord,
    publication: &ConfirmatoryPublicationRecord,
    audit: &PostPublicationAuditLedger,
    study_release: &StudyReleaseBundle,
    orchestration: &StudyOrchestrationLog,
    bundle: &ConfirmatoryFinalReleaseBundle,
) -> Vec<ConfirmatoryFinalReleaseIssue> {
    let mut issues = Vec::new();
    verify_authority(
        confirmatory_readiness_release_commitment(readiness),
        &readiness.release_sha256,
        &bundle.readiness_release_sha256,
        ConfirmatoryFinalReleaseIssue::InvalidReadinessRelease,
        &mut issues,
    );
    verify_authority(
        confirmatory_collection_protocol_commitment(protocol),
        &protocol.protocol_sha256,
        &bundle.collection_protocol_sha256,
        ConfirmatoryFinalReleaseIssue::InvalidCollectionProtocol,
        &mut issues,
    );
    verify_authority(
        confirmatory_collection_close_commitment(close),
        &close.receipt_sha256,
        &bundle.collection_close_sha256,
        ConfirmatoryFinalReleaseIssue::InvalidCollectionClose,
        &mut issues,
    );
    verify_authority(
        confirmatory_unblinding_commitment(unblinding),
        &unblinding.receipt_sha256,
        &bundle.unblinding_receipt_sha256,
        ConfirmatoryFinalReleaseIssue::InvalidUnblindingReceipt,
        &mut issues,
    );
    verify_authority(
        confirmatory_analysis_execution_commitment(analysis),
        &analysis.record_sha256,
        &bundle.analysis_execution_sha256,
        ConfirmatoryFinalReleaseIssue::InvalidAnalysisExecution,
        &mut issues,
    );
    verify_authority(
        confirmatory_publication_commitment(publication),
        &publication.record_sha256,
        &bundle.publication_record_sha256,
        ConfirmatoryFinalReleaseIssue::InvalidPublicationRecord,
        &mut issues,
    );
    verify_authority(
        post_publication_audit_commitment(audit),
        &audit.ledger_sha256,
        &bundle.post_publication_audit_sha256,
        ConfirmatoryFinalReleaseIssue::InvalidPostPublicationAudit,
        &mut issues,
    );
    verify_authority(
        study_release_commitment(study_release),
        &study_release.bundle_sha256,
        &bundle.study_release_bundle_sha256,
        ConfirmatoryFinalReleaseIssue::InvalidStudyRelease,
        &mut issues,
    );
    if !validate_study_orchestration(orchestration).is_empty()
        || orchestration.log_sha256 != bundle.orchestration_log_sha256
    {
        issues.push(ConfirmatoryFinalReleaseIssue::InvalidOrchestration);
    }
    if orchestration.current_phase != StudyLifecyclePhase::Published {
        issues.push(ConfirmatoryFinalReleaseIssue::OrchestrationNotPublished);
    }
    if bundle.release_version != CONFIRMATORY_FINAL_RELEASE_VERSION {
        issues.push(ConfirmatoryFinalReleaseIssue::WrongVersion {
            found: bundle.release_version.clone(),
        });
    }
    for (field, value) in [
        ("study_id", bundle.study_id.as_str()),
        ("source_revision", bundle.source_revision.as_str()),
        ("public_release_uri", bundle.public_release_uri.as_str()),
        ("released_at_utc", bundle.released_at_utc.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(ConfirmatoryFinalReleaseIssue::EmptyField {
                field: field.into(),
            });
        }
    }
    for (field, value) in [
        (
            "workspace_tree_sha256",
            bundle.workspace_tree_sha256.as_str(),
        ),
        (
            "execution_environment_sha256",
            bundle.execution_environment_sha256.as_str(),
        ),
    ] {
        if !is_sha256(value) {
            issues.push(ConfirmatoryFinalReleaseIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    for (field, consistent) in [
        ("study_id.protocol", bundle.study_id == protocol.study_id),
        ("study_id.close", bundle.study_id == close.study_id),
        (
            "study_id.unblinding",
            bundle.study_id == unblinding.study_id,
        ),
        ("study_id.analysis", bundle.study_id == analysis.study_id),
        ("study_id.audit", bundle.study_id == audit.study_id),
        (
            "readiness.protocol",
            protocol.readiness_release_sha256 == readiness.release_sha256,
        ),
        (
            "close.protocol",
            close.protocol_sha256 == protocol.protocol_sha256,
        ),
        (
            "unblinding.close",
            unblinding.collection_close_sha256 == close.receipt_sha256,
        ),
        (
            "analysis.unblinding",
            analysis.unblinding_receipt_sha256 == unblinding.receipt_sha256,
        ),
        (
            "publication.analysis",
            publication.analysis_execution_sha256 == analysis.record_sha256,
        ),
        (
            "audit.publication",
            audit.publication_sha256 == publication.record_sha256,
        ),
    ] {
        if !consistent {
            issues.push(ConfirmatoryFinalReleaseIssue::CrossAuthorityMismatch {
                field: field.into(),
            });
        }
    }
    match confirmatory_final_release_commitment(bundle) {
        Ok(found) if found == bundle.bundle_sha256 => {}
        Ok(_) => issues.push(ConfirmatoryFinalReleaseIssue::BundleDigestMismatch),
        Err(_) => issues.push(ConfirmatoryFinalReleaseIssue::SerializationFailed),
    }
    issues
}

fn verify_authority(
    computed: Result<String, serde_json::Error>,
    authority_digest: &str,
    bound_digest: &str,
    issue: ConfirmatoryFinalReleaseIssue,
    issues: &mut Vec<ConfirmatoryFinalReleaseIssue>,
) {
    match computed {
        Ok(found) if found == authority_digest && found == bound_digest => {}
        _ => issues.push(issue),
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
        assert!(CONFIRMATORY_FINAL_RELEASE_VERSION.ends_with("-v1"));
    }
}
