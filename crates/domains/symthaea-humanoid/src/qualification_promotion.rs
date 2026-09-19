// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed, non-capability qualification prerequisites for contact-authority promotion.
//!
//! QUAL-PROMO-001 makes the exact D4 -> D5B5 focused qualification chain
//! machine-checkable without turning CI metadata into runtime plant authority.
//! The records in this module are descriptive assertions. This module performs
//! no GitHub/network lookup and therefore cannot establish that a supplied
//! `ExecutablePassed` assertion is externally truthful. It also never constructs
//! D4 Established evidence, mutates the contact-authority state machine, or
//! invokes inverse dynamics / a QP.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

pub const AUTHORITY_PROMOTION_PREREQUISITE_SCHEMA_V1: &str =
    "symthaea.humanoid.authority-promotion-prerequisite-manifest.v1";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
pub enum QualificationExecutionStateV1 {
    QueuedInfrastructure,
    Executing,
    ExecutablePassed,
    ExecutableFailed,
    CancelledInfrastructure,
    Superseded,
}

impl QualificationExecutionStateV1 {
    pub const fn id(self) -> &'static str {
        match self {
            Self::QueuedInfrastructure => "queued-infrastructure",
            Self::Executing => "executing",
            Self::ExecutablePassed => "executable-passed",
            Self::ExecutableFailed => "executable-failed",
            Self::CancelledInfrastructure => "cancelled-infrastructure",
            Self::Superseded => "superseded",
        }
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
pub enum QualificationPredecessorRoleV1 {
    ContactAuthorityD4,
    VerifiedSurfaceSupportD5A,
    MujocoContactBiasDyn002C,
    CompleteContactEvidenceD5B,
    IndependentCompleteVerifierD5B2,
    PreparedAuthorityMappingD5B3,
    IndependentPreparedVerifierD5B4,
    SealedPreAuthorityCandidateD5B5,
}

impl QualificationPredecessorRoleV1 {
    pub const fn id(self) -> &'static str {
        match self {
            Self::ContactAuthorityD4 => "hum-wrench-001d4",
            Self::VerifiedSurfaceSupportD5A => "hum-wrench-001d5a",
            Self::MujocoContactBiasDyn002C => "hum-dyn-002c",
            Self::CompleteContactEvidenceD5B => "hum-wrench-001d5b",
            Self::IndependentCompleteVerifierD5B2 => "hum-wrench-001d5b2",
            Self::PreparedAuthorityMappingD5B3 => "hum-wrench-001d5b3",
            Self::IndependentPreparedVerifierD5B4 => "hum-wrench-001d5b4",
            Self::SealedPreAuthorityCandidateD5B5 => "hum-wrench-001d5b5",
        }
    }
}

pub const REQUIRED_QUALIFICATION_PREDECESSOR_ROLES_V1: [QualificationPredecessorRoleV1; 8] = [
    QualificationPredecessorRoleV1::ContactAuthorityD4,
    QualificationPredecessorRoleV1::VerifiedSurfaceSupportD5A,
    QualificationPredecessorRoleV1::MujocoContactBiasDyn002C,
    QualificationPredecessorRoleV1::CompleteContactEvidenceD5B,
    QualificationPredecessorRoleV1::IndependentCompleteVerifierD5B2,
    QualificationPredecessorRoleV1::PreparedAuthorityMappingD5B3,
    QualificationPredecessorRoleV1::IndependentPreparedVerifierD5B4,
    QualificationPredecessorRoleV1::SealedPreAuthorityCandidateD5B5,
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationExecutionRecordV1 {
    pub role: QualificationPredecessorRoleV1,
    pub source_sha: String,
    pub qualifier_sha: String,
    pub workflow_name: String,
    pub workflow_id: u64,
    pub run_id: u64,
    pub run_attempt: u32,
    pub execution_state: QualificationExecutionStateV1,
    pub evidence_artifact_id: u64,
    pub evidence_artifact_name: String,
    pub evidence_artifact_sha256: String,
}

/// Descriptive set of exact qualification assertions required before D5C work
/// may be promoted. This type is intentionally serializable/deserializable: it
/// is audit data, not a capability. Deserialization never establishes authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityPromotionPrerequisiteManifestV1 {
    schema_id: String,
    records: Vec<QualificationExecutionRecordV1>,
    manifest_lineage_id: String,
}

impl AuthorityPromotionPrerequisiteManifestV1 {
    pub fn from_records(records: Vec<QualificationExecutionRecordV1>) -> Self {
        let manifest_lineage_id = manifest_lineage_id(&records);
        Self {
            schema_id: AUTHORITY_PROMOTION_PREREQUISITE_SCHEMA_V1.to_string(),
            records,
            manifest_lineage_id,
        }
    }

    pub fn schema_id(&self) -> &str {
        &self.schema_id
    }

    pub fn records(&self) -> &[QualificationExecutionRecordV1] {
        &self.records
    }

    pub fn manifest_lineage_id(&self) -> &str {
        &self.manifest_lineage_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthorityPromotionPrerequisiteErrorV1 {
    SchemaMismatch,
    DuplicateRole(QualificationPredecessorRoleV1),
    MissingRole(QualificationPredecessorRoleV1),
    MalformedSourceSha(QualificationPredecessorRoleV1),
    MalformedQualifierSha(QualificationPredecessorRoleV1),
    SourceShaMismatch(QualificationPredecessorRoleV1),
    QualifierShaMismatch(QualificationPredecessorRoleV1),
    WorkflowNameMismatch(QualificationPredecessorRoleV1),
    WorkflowIdMismatch(QualificationPredecessorRoleV1),
    RunIdMismatch(QualificationPredecessorRoleV1),
    InvalidRunAttempt(QualificationPredecessorRoleV1),
    NonPassingExecutionState {
        role: QualificationPredecessorRoleV1,
        state: QualificationExecutionStateV1,
    },
    MissingEvidenceArtifactId(QualificationPredecessorRoleV1),
    MissingEvidenceArtifactName(QualificationPredecessorRoleV1),
    EvidenceArtifactNameMismatch(QualificationPredecessorRoleV1),
    MalformedEvidenceArtifactSha256(QualificationPredecessorRoleV1),
    ManifestLineageMismatch,
}

/// Structural verification report only.
///
/// `structurally_satisfied()` means the supplied assertions exactly match the
/// frozen v1 prerequisite contract. It does not prove that GitHub actually
/// executed those jobs and cannot establish runtime contact authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityPromotionPrerequisiteVerificationV1 {
    pub manifest_lineage_id: String,
    pub checked_record_count: usize,
    pub exact_contract_match: bool,
    pub all_executions_asserted_passed: bool,
    pub evidence_artifacts_bound: bool,
}

impl AuthorityPromotionPrerequisiteVerificationV1 {
    pub const fn structurally_satisfied(&self) -> bool {
        self.exact_contract_match
            && self.all_executions_asserted_passed
            && self.evidence_artifacts_bound
    }

    /// This pure checker has no external GitHub attestation channel.
    pub const fn verifies_external_execution_truth(&self) -> bool {
        false
    }

    /// CI qualification metadata is never a runtime plant-authority token.
    pub const fn establishes_runtime_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy)]
struct ExpectedQualificationV1 {
    source_sha: &'static str,
    qualifier_sha: &'static str,
    workflow_name: &'static str,
    workflow_id: u64,
    run_id: u64,
    artifact_name: &'static str,
}

pub fn verify_authority_promotion_prerequisites_v1(
    manifest: &AuthorityPromotionPrerequisiteManifestV1,
) -> Result<AuthorityPromotionPrerequisiteVerificationV1, AuthorityPromotionPrerequisiteErrorV1> {
    if manifest.schema_id != AUTHORITY_PROMOTION_PREREQUISITE_SCHEMA_V1 {
        return Err(AuthorityPromotionPrerequisiteErrorV1::SchemaMismatch);
    }

    let mut seen = BTreeSet::new();
    for record in &manifest.records {
        if !seen.insert(record.role) {
            return Err(AuthorityPromotionPrerequisiteErrorV1::DuplicateRole(
                record.role,
            ));
        }

        let expected = expected_qualification(record.role);
        if !is_lower_hex(&record.source_sha, 40) {
            return Err(AuthorityPromotionPrerequisiteErrorV1::MalformedSourceSha(
                record.role,
            ));
        }
        if !is_lower_hex(&record.qualifier_sha, 40) {
            return Err(
                AuthorityPromotionPrerequisiteErrorV1::MalformedQualifierSha(record.role),
            );
        }
        if record.source_sha != expected.source_sha {
            return Err(AuthorityPromotionPrerequisiteErrorV1::SourceShaMismatch(
                record.role,
            ));
        }
        if record.qualifier_sha != expected.qualifier_sha {
            return Err(
                AuthorityPromotionPrerequisiteErrorV1::QualifierShaMismatch(record.role),
            );
        }
        if record.workflow_name != expected.workflow_name {
            return Err(
                AuthorityPromotionPrerequisiteErrorV1::WorkflowNameMismatch(record.role),
            );
        }
        if record.workflow_id != expected.workflow_id {
            return Err(AuthorityPromotionPrerequisiteErrorV1::WorkflowIdMismatch(
                record.role,
            ));
        }
        if record.run_id != expected.run_id {
            return Err(AuthorityPromotionPrerequisiteErrorV1::RunIdMismatch(
                record.role,
            ));
        }
        if record.run_attempt != 1 {
            return Err(AuthorityPromotionPrerequisiteErrorV1::InvalidRunAttempt(
                record.role,
            ));
        }
        if record.execution_state != QualificationExecutionStateV1::ExecutablePassed {
            return Err(
                AuthorityPromotionPrerequisiteErrorV1::NonPassingExecutionState {
                    role: record.role,
                    state: record.execution_state,
                },
            );
        }
        if record.evidence_artifact_id == 0 {
            return Err(
                AuthorityPromotionPrerequisiteErrorV1::MissingEvidenceArtifactId(record.role),
            );
        }
        if record.evidence_artifact_name.trim().is_empty() {
            return Err(
                AuthorityPromotionPrerequisiteErrorV1::MissingEvidenceArtifactName(record.role),
            );
        }
        if record.evidence_artifact_name != expected.artifact_name {
            return Err(
                AuthorityPromotionPrerequisiteErrorV1::EvidenceArtifactNameMismatch(record.role),
            );
        }
        if !is_lower_hex(&record.evidence_artifact_sha256, 64)
            || record.evidence_artifact_sha256.bytes().all(|byte| byte == b'0')
        {
            return Err(
                AuthorityPromotionPrerequisiteErrorV1::MalformedEvidenceArtifactSha256(
                    record.role,
                ),
            );
        }
    }

    for role in REQUIRED_QUALIFICATION_PREDECESSOR_ROLES_V1 {
        if !seen.contains(&role) {
            return Err(AuthorityPromotionPrerequisiteErrorV1::MissingRole(role));
        }
    }

    let recomputed_manifest_lineage_id = manifest_lineage_id(&manifest.records);
    if recomputed_manifest_lineage_id != manifest.manifest_lineage_id {
        return Err(AuthorityPromotionPrerequisiteErrorV1::ManifestLineageMismatch);
    }

    Ok(AuthorityPromotionPrerequisiteVerificationV1 {
        manifest_lineage_id: recomputed_manifest_lineage_id,
        checked_record_count: manifest.records.len(),
        exact_contract_match: true,
        all_executions_asserted_passed: true,
        evidence_artifacts_bound: true,
    })
}

fn expected_qualification(role: QualificationPredecessorRoleV1) -> ExpectedQualificationV1 {
    match role {
        QualificationPredecessorRoleV1::ContactAuthorityD4 => ExpectedQualificationV1 {
            source_sha: "c00fedffffd9d0240f5a5be44cac293bac6cef44",
            qualifier_sha: "4210a5e51127f6c8a6c663766dc6d130bf4224df",
            workflow_name: "HUM-WRENCH-001D4 Contact Authority",
            workflow_id: 361_892_536,
            run_id: 35_426_272_913,
            artifact_name: "hum-wrench-001d4-4210a5e51127f6c8a6c663766dc6d130bf4224df",
        },
        QualificationPredecessorRoleV1::VerifiedSurfaceSupportD5A => ExpectedQualificationV1 {
            source_sha: "b64e62287cfb49797d280369545672cff87b48dd",
            qualifier_sha: "32b49ed10987e7b3a48ed54dd8dd7b551d1731e8",
            workflow_name: "HUM-WRENCH-001D5A Verified Surface Support",
            workflow_id: 361_896_937,
            run_id: 35_431_016_942,
            artifact_name: "hum-wrench-001d5a-32b49ed10987e7b3a48ed54dd8dd7b551d1731e8",
        },
        QualificationPredecessorRoleV1::MujocoContactBiasDyn002C => ExpectedQualificationV1 {
            source_sha: "67107f337ee454d4d9da859b7fd4dea8628cedf2",
            qualifier_sha: "0b222328a6123a87716c8a535cb21179dc8e6774",
            workflow_name: "HUM-DYN-002C MuJoCo Contact Bias",
            workflow_id: 361_532_488,
            run_id: 35_431_329_409,
            artifact_name: "hum-dyn-002c-0b222328a6123a87716c8a535cb21179dc8e6774",
        },
        QualificationPredecessorRoleV1::CompleteContactEvidenceD5B => ExpectedQualificationV1 {
            source_sha: "156ed01855b2c6fd89118b5ac4063282added0cb",
            qualifier_sha: "3d69c5f7006fc300ae538effde87ebec9b43486e",
            workflow_name: "HUM-WRENCH-001D5B Complete Contact Evidence",
            workflow_id: 361_945_663,
            run_id: 35_431_653_533,
            artifact_name: "hum-wrench-001d5b-3d69c5f7006fc300ae538effde87ebec9b43486e",
        },
        QualificationPredecessorRoleV1::IndependentCompleteVerifierD5B2 => {
            ExpectedQualificationV1 {
                source_sha: "47f6e0339a36ddfcef025e6277adbab1b91b6163",
                qualifier_sha: "75e91066db405daf49d1b78f9630d68e4ff72eeb",
                workflow_name: "HUM-WRENCH-001D5B2 Independent Complete Contact Verifier",
                workflow_id: 361_971_294,
                run_id: 35_434_249_956,
                artifact_name: "hum-wrench-001d5b2-75e91066db405daf49d1b78f9630d68e4ff72eeb",
            }
        }
        QualificationPredecessorRoleV1::PreparedAuthorityMappingD5B3 => {
            ExpectedQualificationV1 {
                source_sha: "44a0cdf8b3c939d02b8dad89a531d18b50d4979a",
                qualifier_sha: "66343eb1f9d57fb9d22458bb22252c2059cbe469",
                workflow_name: "HUM-WRENCH-001D5B3 Prepared Contact Authority Evidence",
                workflow_id: 361_973_645,
                run_id: 35_434_645_590,
                artifact_name: "hum-wrench-001d5b3-66343eb1f9d57fb9d22458bb22252c2059cbe469",
            }
        }
        QualificationPredecessorRoleV1::IndependentPreparedVerifierD5B4 => {
            ExpectedQualificationV1 {
                source_sha: "9e8248fbdc92af1179ff3a29fed9f84d8d52fa99",
                qualifier_sha: "bb4f2029e13308067660e4f0f0b59bdb6c77a984",
                workflow_name: "HUM-WRENCH-001D5B4 Independent Prepared Authority Verifier",
                workflow_id: 361_999_685,
                run_id: 35_437_190_217,
                artifact_name: "hum-wrench-001d5b4-bb4f2029e13308067660e4f0f0b59bdb6c77a984",
            }
        }
        QualificationPredecessorRoleV1::SealedPreAuthorityCandidateD5B5 => {
            ExpectedQualificationV1 {
                source_sha: "792ead8e08565995b0bfc5e3f6ab5ebbde6aa8df",
                qualifier_sha: "015f82ff0df8746a0c0c06f14a00ba6b25e70ed6",
                workflow_name: "HUM-WRENCH-001D5B5 Sealed Pre-Authority Surface Candidate",
                workflow_id: 362_002_235,
                run_id: 35_437_475_927,
                artifact_name: "hum-wrench-001d5b5-015f82ff0df8746a0c0c06f14a00ba6b25e70ed6",
            }
        }
    }
}

fn manifest_lineage_id(records: &[QualificationExecutionRecordV1]) -> String {
    let mut ordered = records.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|record| record.role);

    let mut lineage = format!(
        "authority-promotion-prerequisite-manifest-v1:schema:{}:count:{}",
        component(AUTHORITY_PROMOTION_PREREQUISITE_SCHEMA_V1),
        ordered.len(),
    );
    for record in ordered {
        lineage.push_str(":record:");
        lineage.push_str(&record_lineage_component(record));
    }
    lineage
}

fn record_lineage_component(record: &QualificationExecutionRecordV1) -> String {
    format!(
        "role:{}:source:{}:qualifier:{}:workflow-name:{}:workflow-id:{}:run-id:{}:run-attempt:{}:state:{}:artifact-id:{}:artifact-name:{}:artifact-sha256:{}",
        record.role.id(),
        component(&record.source_sha),
        component(&record.qualifier_sha),
        component(&record.workflow_name),
        record.workflow_id,
        record.run_id,
        record.run_attempt,
        record.execution_state.id(),
        record.evidence_artifact_id,
        component(&record.evidence_artifact_name),
        component(&record.evidence_artifact_sha256),
    )
}

fn component(value: &str) -> String {
    format!("{}:{}", value.len(), value)
}

fn is_lower_hex(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARTIFACT_DIGEST: &str =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn passed_records() -> Vec<QualificationExecutionRecordV1> {
        REQUIRED_QUALIFICATION_PREDECESSOR_ROLES_V1
            .iter()
            .copied()
            .enumerate()
            .map(|(index, role)| {
                let expected = expected_qualification(role);
                QualificationExecutionRecordV1 {
                    role,
                    source_sha: expected.source_sha.to_string(),
                    qualifier_sha: expected.qualifier_sha.to_string(),
                    workflow_name: expected.workflow_name.to_string(),
                    workflow_id: expected.workflow_id,
                    run_id: expected.run_id,
                    run_attempt: 1,
                    execution_state: QualificationExecutionStateV1::ExecutablePassed,
                    evidence_artifact_id: 1_000 + index as u64,
                    evidence_artifact_name: expected.artifact_name.to_string(),
                    evidence_artifact_sha256: ARTIFACT_DIGEST.to_string(),
                }
            })
            .collect()
    }

    #[test]
    fn exact_pass_set_is_structurally_satisfied_but_never_authority() {
        let manifest = AuthorityPromotionPrerequisiteManifestV1::from_records(passed_records());
        let verification = verify_authority_promotion_prerequisites_v1(&manifest).unwrap();

        assert!(verification.structurally_satisfied());
        assert_eq!(verification.checked_record_count, 8);
        assert!(!verification.verifies_external_execution_truth());
        assert!(!verification.establishes_runtime_authority());
    }

    #[test]
    fn queued_infrastructure_cannot_be_promoted() {
        let mut records = passed_records();
        let d5b5 = records
            .iter_mut()
            .find(|record| {
                record.role == QualificationPredecessorRoleV1::SealedPreAuthorityCandidateD5B5
            })
            .unwrap();
        d5b5.execution_state = QualificationExecutionStateV1::QueuedInfrastructure;
        d5b5.evidence_artifact_id = 0;
        d5b5.evidence_artifact_name.clear();
        d5b5.evidence_artifact_sha256.clear();
        let manifest = AuthorityPromotionPrerequisiteManifestV1::from_records(records);

        assert_eq!(
            verify_authority_promotion_prerequisites_v1(&manifest),
            Err(
                AuthorityPromotionPrerequisiteErrorV1::NonPassingExecutionState {
                    role: QualificationPredecessorRoleV1::SealedPreAuthorityCandidateD5B5,
                    state: QualificationExecutionStateV1::QueuedInfrastructure,
                }
            )
        );
    }

    #[test]
    fn duplicate_role_fails_closed() {
        let mut records = passed_records();
        records.push(records[0].clone());
        let manifest = AuthorityPromotionPrerequisiteManifestV1::from_records(records);

        assert_eq!(
            verify_authority_promotion_prerequisites_v1(&manifest),
            Err(AuthorityPromotionPrerequisiteErrorV1::DuplicateRole(
                QualificationPredecessorRoleV1::ContactAuthorityD4,
            ))
        );
    }

    #[test]
    fn missing_role_fails_closed() {
        let mut records = passed_records();
        records.retain(|record| {
            record.role != QualificationPredecessorRoleV1::IndependentPreparedVerifierD5B4
        });
        let manifest = AuthorityPromotionPrerequisiteManifestV1::from_records(records);

        assert_eq!(
            verify_authority_promotion_prerequisites_v1(&manifest),
            Err(AuthorityPromotionPrerequisiteErrorV1::MissingRole(
                QualificationPredecessorRoleV1::IndependentPreparedVerifierD5B4,
            ))
        );
    }

    #[test]
    fn qualifier_head_substitution_fails_closed() {
        let mut records = passed_records();
        records[0].qualifier_sha = "1111111111111111111111111111111111111111".to_string();
        let manifest = AuthorityPromotionPrerequisiteManifestV1::from_records(records);

        assert_eq!(
            verify_authority_promotion_prerequisites_v1(&manifest),
            Err(AuthorityPromotionPrerequisiteErrorV1::QualifierShaMismatch(
                QualificationPredecessorRoleV1::ContactAuthorityD4,
            ))
        );
    }

    #[test]
    fn workflow_or_run_substitution_fails_closed() {
        let mut workflow_records = passed_records();
        workflow_records[1].workflow_id += 1;
        let workflow_manifest =
            AuthorityPromotionPrerequisiteManifestV1::from_records(workflow_records);
        assert_eq!(
            verify_authority_promotion_prerequisites_v1(&workflow_manifest),
            Err(AuthorityPromotionPrerequisiteErrorV1::WorkflowIdMismatch(
                QualificationPredecessorRoleV1::VerifiedSurfaceSupportD5A,
            ))
        );

        let mut run_records = passed_records();
        run_records[2].run_id += 1;
        let run_manifest = AuthorityPromotionPrerequisiteManifestV1::from_records(run_records);
        assert_eq!(
            verify_authority_promotion_prerequisites_v1(&run_manifest),
            Err(AuthorityPromotionPrerequisiteErrorV1::RunIdMismatch(
                QualificationPredecessorRoleV1::MujocoContactBiasDyn002C,
            ))
        );
    }

    #[test]
    fn non_first_run_attempt_fails_closed() {
        for invalid_attempt in [0_u32, 2_u32] {
            let mut records = passed_records();
            records[4].run_attempt = invalid_attempt;
            let manifest = AuthorityPromotionPrerequisiteManifestV1::from_records(records);

            assert_eq!(
                verify_authority_promotion_prerequisites_v1(&manifest),
                Err(AuthorityPromotionPrerequisiteErrorV1::InvalidRunAttempt(
                    QualificationPredecessorRoleV1::IndependentCompleteVerifierD5B2,
                ))
            );
        }
    }

    #[test]
    fn passed_record_requires_exact_artifact_identity_and_digest() {
        let mut id_records = passed_records();
        id_records[3].evidence_artifact_id = 0;
        let id_manifest = AuthorityPromotionPrerequisiteManifestV1::from_records(id_records);
        assert_eq!(
            verify_authority_promotion_prerequisites_v1(&id_manifest),
            Err(AuthorityPromotionPrerequisiteErrorV1::MissingEvidenceArtifactId(
                QualificationPredecessorRoleV1::CompleteContactEvidenceD5B,
            ))
        );

        let mut name_records = passed_records();
        name_records[3].evidence_artifact_name.push_str("-substituted");
        let name_manifest = AuthorityPromotionPrerequisiteManifestV1::from_records(name_records);
        assert_eq!(
            verify_authority_promotion_prerequisites_v1(&name_manifest),
            Err(AuthorityPromotionPrerequisiteErrorV1::EvidenceArtifactNameMismatch(
                QualificationPredecessorRoleV1::CompleteContactEvidenceD5B,
            ))
        );

        let mut digest_records = passed_records();
        digest_records[3].evidence_artifact_sha256.clear();
        let digest_manifest = AuthorityPromotionPrerequisiteManifestV1::from_records(digest_records);
        assert_eq!(
            verify_authority_promotion_prerequisites_v1(&digest_manifest),
            Err(
                AuthorityPromotionPrerequisiteErrorV1::MalformedEvidenceArtifactSha256(
                    QualificationPredecessorRoleV1::CompleteContactEvidenceD5B,
                )
            )
        );

        let mut zero_digest_records = passed_records();
        zero_digest_records[3].evidence_artifact_sha256 = "0".repeat(64);
        let zero_digest_manifest =
            AuthorityPromotionPrerequisiteManifestV1::from_records(zero_digest_records);
        assert_eq!(
            verify_authority_promotion_prerequisites_v1(&zero_digest_manifest),
            Err(
                AuthorityPromotionPrerequisiteErrorV1::MalformedEvidenceArtifactSha256(
                    QualificationPredecessorRoleV1::CompleteContactEvidenceD5B,
                )
            )
        );
    }

    #[test]
    fn manifest_lineage_is_order_independent() {
        let records = passed_records();
        let forward = AuthorityPromotionPrerequisiteManifestV1::from_records(records.clone());
        let mut reversed_records = records;
        reversed_records.reverse();
        let reversed = AuthorityPromotionPrerequisiteManifestV1::from_records(reversed_records);

        assert_eq!(forward.manifest_lineage_id(), reversed.manifest_lineage_id());
        assert!(
            verify_authority_promotion_prerequisites_v1(&reversed)
                .unwrap()
                .structurally_satisfied()
        );
    }

    #[test]
    fn manifest_lineage_tampering_fails_closed() {
        let mut manifest = AuthorityPromotionPrerequisiteManifestV1::from_records(passed_records());
        manifest.manifest_lineage_id.push_str(":tampered");

        assert_eq!(
            verify_authority_promotion_prerequisites_v1(&manifest),
            Err(AuthorityPromotionPrerequisiteErrorV1::ManifestLineageMismatch)
        );
    }
}
