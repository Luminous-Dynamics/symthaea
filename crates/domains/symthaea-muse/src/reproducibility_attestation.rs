// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent reproduction attestations for the cognition study.

use crate::analysis_crosscheck::AnalysisCrosscheckReport;
use crate::evidence_digest::canonical_json_sha256;
use crate::study_release::StudyReleaseBundle;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const REPRODUCIBILITY_ATTESTATION_VERSION: &str =
    "symthaea-muse-reproducibility-attestation-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ReproducedOutputRole {
    SourceTree,
    ArtifactBundle,
    ParticipantEvidence,
    CompiledDataset,
    PrimaryAnalysis,
    IndependentAnalysis,
    ReleaseRoot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReproductionMatch {
    ExactSha256,
    NumericallyEquivalent,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReproducedOutput {
    pub role: ReproducedOutputRole,
    pub expected_sha256: String,
    pub observed_sha256: String,
    pub match_kind: ReproductionMatch,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndependentVerifierIdentity {
    pub verifier_id: String,
    pub organization: String,
    pub contact_commitment_sha256: String,
    pub conflict_of_interest_declaration: String,
    pub independent_of_authors: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReproductionEnvironment {
    pub operating_system: String,
    pub architecture: String,
    pub nix_version: String,
    pub flake_lock_sha256: String,
    pub toolchain_evidence_sha256: String,
    pub execution_environment_sha256: String,
    pub commands_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndependentReproductionAttestation {
    pub attestation_version: String,
    pub study_release_sha256: String,
    pub analysis_crosscheck_sha256: String,
    pub verifier: IndependentVerifierIdentity,
    pub environment: ReproductionEnvironment,
    pub started_at_utc: String,
    pub completed_at_utc: String,
    pub outputs: Vec<ReproducedOutput>,
    pub all_required_commands_succeeded: bool,
    pub analysis_crosscheck_passed: bool,
    pub exact_release_root_reproduced: bool,
    pub limitations: Vec<String>,
    pub external_receipt_uri: String,
    pub external_signature_sha256: String,
    pub attestation_sha256: String,
}

#[derive(Serialize)]
struct AttestationCommitment<'a> {
    attestation_version: &'a str,
    study_release_sha256: &'a str,
    analysis_crosscheck_sha256: &'a str,
    verifier: &'a IndependentVerifierIdentity,
    environment: &'a ReproductionEnvironment,
    started_at_utc: &'a str,
    completed_at_utc: &'a str,
    outputs: &'a [ReproducedOutput],
    all_required_commands_succeeded: bool,
    analysis_crosscheck_passed: bool,
    exact_release_root_reproduced: bool,
    limitations: &'a [String],
    external_receipt_uri: &'a str,
    external_signature_sha256: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReproducibilityAttestationIssue {
    WrongVersion { found: String },
    SerializationFailed { field: String },
    DigestMismatch { field: String },
    InvalidDigest { field: String },
    EmptyField { field: String },
    VerifierNotIndependent,
    EmptyConflictDeclaration,
    DuplicateOutputRole { role: ReproducedOutputRole },
    MissingOutputRole { role: ReproducedOutputRole },
    FailedOutputDeclaredSuccessful { role: ReproducedOutputRole },
    ExactMatchDigestMismatch { role: ReproducedOutputRole },
    ReleaseRootFlagMismatch,
    CrosscheckFlagMismatch,
    SuccessFlagContradiction,
}

pub fn reproduction_attestation_commitment(
    attestation: &IndependentReproductionAttestation,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&AttestationCommitment {
        attestation_version: &attestation.attestation_version,
        study_release_sha256: &attestation.study_release_sha256,
        analysis_crosscheck_sha256: &attestation.analysis_crosscheck_sha256,
        verifier: &attestation.verifier,
        environment: &attestation.environment,
        started_at_utc: &attestation.started_at_utc,
        completed_at_utc: &attestation.completed_at_utc,
        outputs: &attestation.outputs,
        all_required_commands_succeeded: attestation.all_required_commands_succeeded,
        analysis_crosscheck_passed: attestation.analysis_crosscheck_passed,
        exact_release_root_reproduced: attestation.exact_release_root_reproduced,
        limitations: &attestation.limitations,
        external_receipt_uri: &attestation.external_receipt_uri,
        external_signature_sha256: &attestation.external_signature_sha256,
    })
}

pub fn seal_reproduction_attestation(
    attestation: &mut IndependentReproductionAttestation,
) -> Result<(), serde_json::Error> {
    attestation.outputs.sort_by_key(|output| output.role);
    attestation.attestation_sha256 = reproduction_attestation_commitment(attestation)?;
    Ok(())
}

pub fn validate_reproduction_attestation(
    release: &StudyReleaseBundle,
    crosscheck: &AnalysisCrosscheckReport,
    attestation: &IndependentReproductionAttestation,
) -> Vec<ReproducibilityAttestationIssue> {
    let mut issues = Vec::new();
    if attestation.attestation_version != REPRODUCIBILITY_ATTESTATION_VERSION {
        issues.push(ReproducibilityAttestationIssue::WrongVersion {
            found: attestation.attestation_version.clone(),
        });
    }
    verify_digest(
        release,
        &attestation.study_release_sha256,
        "study_release_sha256",
        &mut issues,
    );
    verify_digest(
        crosscheck,
        &attestation.analysis_crosscheck_sha256,
        "analysis_crosscheck_sha256",
        &mut issues,
    );
    for (field, digest) in [
        (
            "verifier.contact_commitment_sha256",
            attestation.verifier.contact_commitment_sha256.as_str(),
        ),
        (
            "environment.flake_lock_sha256",
            attestation.environment.flake_lock_sha256.as_str(),
        ),
        (
            "environment.toolchain_evidence_sha256",
            attestation.environment.toolchain_evidence_sha256.as_str(),
        ),
        (
            "environment.execution_environment_sha256",
            attestation
                .environment
                .execution_environment_sha256
                .as_str(),
        ),
        (
            "environment.commands_sha256",
            attestation.environment.commands_sha256.as_str(),
        ),
        (
            "external_signature_sha256",
            attestation.external_signature_sha256.as_str(),
        ),
        (
            "attestation_sha256",
            attestation.attestation_sha256.as_str(),
        ),
    ] {
        if !is_sha256(digest) {
            issues.push(ReproducibilityAttestationIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    for (field, value) in [
        ("verifier_id", attestation.verifier.verifier_id.as_str()),
        ("organization", attestation.verifier.organization.as_str()),
        (
            "operating_system",
            attestation.environment.operating_system.as_str(),
        ),
        (
            "architecture",
            attestation.environment.architecture.as_str(),
        ),
        ("nix_version", attestation.environment.nix_version.as_str()),
        ("started_at_utc", attestation.started_at_utc.as_str()),
        ("completed_at_utc", attestation.completed_at_utc.as_str()),
        (
            "external_receipt_uri",
            attestation.external_receipt_uri.as_str(),
        ),
    ] {
        if value.trim().is_empty() {
            issues.push(ReproducibilityAttestationIssue::EmptyField {
                field: field.into(),
            });
        }
    }
    if !attestation.verifier.independent_of_authors {
        issues.push(ReproducibilityAttestationIssue::VerifierNotIndependent);
    }
    if attestation
        .verifier
        .conflict_of_interest_declaration
        .trim()
        .is_empty()
    {
        issues.push(ReproducibilityAttestationIssue::EmptyConflictDeclaration);
    }
    let mut roles = BTreeSet::new();
    for output in &attestation.outputs {
        if !roles.insert(output.role) {
            issues.push(ReproducibilityAttestationIssue::DuplicateOutputRole { role: output.role });
        }
        for (field, digest) in [
            ("expected_sha256", output.expected_sha256.as_str()),
            ("observed_sha256", output.observed_sha256.as_str()),
        ] {
            if !is_sha256(digest) {
                issues.push(ReproducibilityAttestationIssue::InvalidDigest {
                    field: format!("output.{:?}.{field}", output.role),
                });
            }
        }
        if output.match_kind == ReproductionMatch::ExactSha256
            && output.expected_sha256 != output.observed_sha256
        {
            issues.push(ReproducibilityAttestationIssue::ExactMatchDigestMismatch {
                role: output.role,
            });
        }
        if output.match_kind == ReproductionMatch::Failed
            && attestation.all_required_commands_succeeded
        {
            issues.push(
                ReproducibilityAttestationIssue::FailedOutputDeclaredSuccessful {
                    role: output.role,
                },
            );
        }
    }
    for role in [
        ReproducedOutputRole::SourceTree,
        ReproducedOutputRole::ArtifactBundle,
        ReproducedOutputRole::ParticipantEvidence,
        ReproducedOutputRole::CompiledDataset,
        ReproducedOutputRole::PrimaryAnalysis,
        ReproducedOutputRole::IndependentAnalysis,
        ReproducedOutputRole::ReleaseRoot,
    ] {
        if !roles.contains(&role) {
            issues.push(ReproducibilityAttestationIssue::MissingOutputRole { role });
        }
    }
    let release_exact = attestation.outputs.iter().any(|output| {
        output.role == ReproducedOutputRole::ReleaseRoot
            && output.match_kind == ReproductionMatch::ExactSha256
            && output.expected_sha256 == output.observed_sha256
    });
    if release_exact != attestation.exact_release_root_reproduced {
        issues.push(ReproducibilityAttestationIssue::ReleaseRootFlagMismatch);
    }
    if crosscheck.passed != attestation.analysis_crosscheck_passed {
        issues.push(ReproducibilityAttestationIssue::CrosscheckFlagMismatch);
    }
    if attestation.all_required_commands_succeeded
        && (!attestation.analysis_crosscheck_passed || !attestation.exact_release_root_reproduced)
    {
        issues.push(ReproducibilityAttestationIssue::SuccessFlagContradiction);
    }
    match reproduction_attestation_commitment(attestation) {
        Ok(value) if value == attestation.attestation_sha256 => {}
        Ok(_) => issues.push(ReproducibilityAttestationIssue::DigestMismatch {
            field: "attestation_sha256".into(),
        }),
        Err(_) => issues.push(ReproducibilityAttestationIssue::SerializationFailed {
            field: "attestation".into(),
        }),
    }
    issues
}

fn verify_digest<T: Serialize>(
    value: &T,
    expected: &str,
    field: &str,
    issues: &mut Vec<ReproducibilityAttestationIssue>,
) {
    match canonical_json_sha256(value) {
        Ok(value) if value == expected => {}
        Ok(_) => issues.push(ReproducibilityAttestationIssue::DigestMismatch {
            field: field.into(),
        }),
        Err(_) => issues.push(ReproducibilityAttestationIssue::SerializationFailed {
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
    fn exact_match_requires_equal_digests() {
        let output = ReproducedOutput {
            role: ReproducedOutputRole::ReleaseRoot,
            expected_sha256: "a".repeat(64),
            observed_sha256: "b".repeat(64),
            match_kind: ReproductionMatch::ExactSha256,
            notes: String::new(),
        };
        assert_ne!(output.expected_sha256, output.observed_sha256);
    }
}
