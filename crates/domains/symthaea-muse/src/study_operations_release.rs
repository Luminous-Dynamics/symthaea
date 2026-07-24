// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Root commitment for V10 pilot operations and independent reproduction.

use crate::analysis_crosscheck::AnalysisCrosscheckReport;
use crate::cohort_registry::PilotCohortRegistry;
use crate::evidence_digest::canonical_json_sha256;
use crate::pilot_collection::PilotCollectionEnvelope;
use crate::pilot_monitoring::PilotOperationalSnapshot;
use crate::pilot_protocol::{FrozenPilotProtocol, PilotAmendmentLedger};
use crate::pilot_report::PilotReviewReport;
use crate::pilot_schedule::PilotParticipantScheduleBook;
use crate::reproducibility_attestation::IndependentReproductionAttestation;
use crate::study_orchestration::StudyOrchestrationLog;
use crate::study_release::StudyReleaseBundle;
use serde::{Deserialize, Serialize};

pub const STUDY_OPERATIONS_RELEASE_VERSION: &str = "symthaea-muse-study-operations-release-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StudyOperationsReleaseBundle {
    pub release_version: String,
    pub base_study_release_sha256: String,
    pub pilot_protocol_sha256: String,
    pub pilot_schedule_sha256: String,
    pub pilot_cohort_registry_sha256: String,
    pub pilot_collection_sha256: String,
    pub pilot_operational_snapshot_sha256: String,
    pub pilot_amendment_ledger_sha256: String,
    pub pilot_review_report_sha256: String,
    pub orchestration_log_sha256: String,
    pub analysis_crosscheck_sha256: String,
    pub reproduction_attestation_sha256: String,
    pub source_archive_sha256: String,
    pub nix_flake_lock_sha256: String,
    pub toolchain_evidence_sha256: String,
    pub release_sha256: String,
}

#[derive(Serialize)]
struct OperationsReleaseCommitment<'a> {
    release_version: &'a str,
    base_study_release_sha256: &'a str,
    pilot_protocol_sha256: &'a str,
    pilot_schedule_sha256: &'a str,
    pilot_cohort_registry_sha256: &'a str,
    pilot_collection_sha256: &'a str,
    pilot_operational_snapshot_sha256: &'a str,
    pilot_amendment_ledger_sha256: &'a str,
    pilot_review_report_sha256: &'a str,
    orchestration_log_sha256: &'a str,
    analysis_crosscheck_sha256: &'a str,
    reproduction_attestation_sha256: &'a str,
    source_archive_sha256: &'a str,
    nix_flake_lock_sha256: &'a str,
    toolchain_evidence_sha256: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StudyOperationsReleaseIssue {
    WrongVersion { found: String },
    SerializationFailed { field: String },
    DigestMismatch { field: String },
    InvalidDigest { field: String },
    ReleaseDigestMismatch,
}

#[allow(clippy::too_many_arguments)]
pub fn build_study_operations_release(
    base_release: &StudyReleaseBundle,
    pilot_protocol: &FrozenPilotProtocol,
    pilot_schedule: &PilotParticipantScheduleBook,
    cohort_registry: &PilotCohortRegistry,
    pilot_collection: &PilotCollectionEnvelope,
    operational_snapshot: &PilotOperationalSnapshot,
    amendment_ledger: &PilotAmendmentLedger,
    pilot_report: &PilotReviewReport,
    orchestration: &StudyOrchestrationLog,
    crosscheck: &AnalysisCrosscheckReport,
    attestation: &IndependentReproductionAttestation,
    source_archive_sha256: String,
    nix_flake_lock_sha256: String,
    toolchain_evidence_sha256: String,
) -> Result<StudyOperationsReleaseBundle, serde_json::Error> {
    let mut bundle = StudyOperationsReleaseBundle {
        release_version: STUDY_OPERATIONS_RELEASE_VERSION.into(),
        base_study_release_sha256: canonical_json_sha256(base_release)?,
        pilot_protocol_sha256: canonical_json_sha256(pilot_protocol)?,
        pilot_schedule_sha256: canonical_json_sha256(pilot_schedule)?,
        pilot_cohort_registry_sha256: canonical_json_sha256(cohort_registry)?,
        pilot_collection_sha256: canonical_json_sha256(pilot_collection)?,
        pilot_operational_snapshot_sha256: canonical_json_sha256(operational_snapshot)?,
        pilot_amendment_ledger_sha256: canonical_json_sha256(amendment_ledger)?,
        pilot_review_report_sha256: canonical_json_sha256(pilot_report)?,
        orchestration_log_sha256: canonical_json_sha256(orchestration)?,
        analysis_crosscheck_sha256: canonical_json_sha256(crosscheck)?,
        reproduction_attestation_sha256: canonical_json_sha256(attestation)?,
        source_archive_sha256,
        nix_flake_lock_sha256,
        toolchain_evidence_sha256,
        release_sha256: String::new(),
    };
    bundle.release_sha256 = study_operations_release_commitment(&bundle)?;
    Ok(bundle)
}

pub fn study_operations_release_commitment(
    bundle: &StudyOperationsReleaseBundle,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&OperationsReleaseCommitment {
        release_version: &bundle.release_version,
        base_study_release_sha256: &bundle.base_study_release_sha256,
        pilot_protocol_sha256: &bundle.pilot_protocol_sha256,
        pilot_schedule_sha256: &bundle.pilot_schedule_sha256,
        pilot_cohort_registry_sha256: &bundle.pilot_cohort_registry_sha256,
        pilot_collection_sha256: &bundle.pilot_collection_sha256,
        pilot_operational_snapshot_sha256: &bundle.pilot_operational_snapshot_sha256,
        pilot_amendment_ledger_sha256: &bundle.pilot_amendment_ledger_sha256,
        pilot_review_report_sha256: &bundle.pilot_review_report_sha256,
        orchestration_log_sha256: &bundle.orchestration_log_sha256,
        analysis_crosscheck_sha256: &bundle.analysis_crosscheck_sha256,
        reproduction_attestation_sha256: &bundle.reproduction_attestation_sha256,
        source_archive_sha256: &bundle.source_archive_sha256,
        nix_flake_lock_sha256: &bundle.nix_flake_lock_sha256,
        toolchain_evidence_sha256: &bundle.toolchain_evidence_sha256,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn validate_study_operations_release(
    base_release: &StudyReleaseBundle,
    pilot_protocol: &FrozenPilotProtocol,
    pilot_schedule: &PilotParticipantScheduleBook,
    cohort_registry: &PilotCohortRegistry,
    pilot_collection: &PilotCollectionEnvelope,
    operational_snapshot: &PilotOperationalSnapshot,
    amendment_ledger: &PilotAmendmentLedger,
    pilot_report: &PilotReviewReport,
    orchestration: &StudyOrchestrationLog,
    crosscheck: &AnalysisCrosscheckReport,
    attestation: &IndependentReproductionAttestation,
    bundle: &StudyOperationsReleaseBundle,
) -> Vec<StudyOperationsReleaseIssue> {
    let mut issues = Vec::new();
    if bundle.release_version != STUDY_OPERATIONS_RELEASE_VERSION {
        issues.push(StudyOperationsReleaseIssue::WrongVersion {
            found: bundle.release_version.clone(),
        });
    }
    for (field, digest) in [
        (
            "source_archive_sha256",
            bundle.source_archive_sha256.as_str(),
        ),
        (
            "nix_flake_lock_sha256",
            bundle.nix_flake_lock_sha256.as_str(),
        ),
        (
            "toolchain_evidence_sha256",
            bundle.toolchain_evidence_sha256.as_str(),
        ),
        ("release_sha256", bundle.release_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(StudyOperationsReleaseIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    verify_digest(
        base_release,
        &bundle.base_study_release_sha256,
        "base_study_release_sha256",
        &mut issues,
    );
    verify_digest(
        pilot_protocol,
        &bundle.pilot_protocol_sha256,
        "pilot_protocol_sha256",
        &mut issues,
    );
    verify_digest(
        pilot_schedule,
        &bundle.pilot_schedule_sha256,
        "pilot_schedule_sha256",
        &mut issues,
    );
    verify_digest(
        cohort_registry,
        &bundle.pilot_cohort_registry_sha256,
        "pilot_cohort_registry_sha256",
        &mut issues,
    );
    verify_digest(
        pilot_collection,
        &bundle.pilot_collection_sha256,
        "pilot_collection_sha256",
        &mut issues,
    );
    verify_digest(
        operational_snapshot,
        &bundle.pilot_operational_snapshot_sha256,
        "pilot_operational_snapshot_sha256",
        &mut issues,
    );
    verify_digest(
        amendment_ledger,
        &bundle.pilot_amendment_ledger_sha256,
        "pilot_amendment_ledger_sha256",
        &mut issues,
    );
    verify_digest(
        pilot_report,
        &bundle.pilot_review_report_sha256,
        "pilot_review_report_sha256",
        &mut issues,
    );
    verify_digest(
        orchestration,
        &bundle.orchestration_log_sha256,
        "orchestration_log_sha256",
        &mut issues,
    );
    verify_digest(
        crosscheck,
        &bundle.analysis_crosscheck_sha256,
        "analysis_crosscheck_sha256",
        &mut issues,
    );
    verify_digest(
        attestation,
        &bundle.reproduction_attestation_sha256,
        "reproduction_attestation_sha256",
        &mut issues,
    );
    match study_operations_release_commitment(bundle) {
        Ok(value) if value == bundle.release_sha256 => {}
        Ok(_) => issues.push(StudyOperationsReleaseIssue::ReleaseDigestMismatch),
        Err(_) => issues.push(StudyOperationsReleaseIssue::SerializationFailed {
            field: "operations_release".into(),
        }),
    }
    issues
}

fn verify_digest<T: Serialize>(
    value: &T,
    expected: &str,
    field: &str,
    issues: &mut Vec<StudyOperationsReleaseIssue>,
) {
    match canonical_json_sha256(value) {
        Ok(value) if value == expected => {}
        Ok(_) => issues.push(StudyOperationsReleaseIssue::DigestMismatch {
            field: field.into(),
        }),
        Err(_) => issues.push(StudyOperationsReleaseIssue::SerializationFailed {
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
    fn version_is_explicit() {
        assert!(STUDY_OPERATIONS_RELEASE_VERSION.ends_with("v1"));
    }
}
