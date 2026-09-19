// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Frozen release-time provenance contract for future D5C contact-authority promotion.
//!
//! QUAL-PROMO-003 defines the shape of the exact promotion pin that a future
//! D5C source subject must compile in after QUAL-PROMO-001/002 have executable
//! qualification and a real external attestation artifact exists. This module
//! intentionally contains **no instantiated promotion pin** today.
//!
//! The pin is repository qualification provenance only. Even a structurally
//! valid frozen pin cannot establish live D4 contact state, freshness, an
//! establishment epoch, QP authority, hardware contact truth, or physical safety.

use crate::AUTHORITY_PROMOTION_PREREQUISITE_SCHEMA_V1;

pub const AUTHORITY_PROMOTION_PIN_SCHEMA_V1: &str =
    "symthaea.humanoid.d5c-authority-promotion-pin.v1";
pub const QUAL_PROMO_002_EVIDENCE_SCHEMA_V1: &str =
    "symthaea.qual-promo-002.github-actions-evidence.v1";

pub const QUAL_PROMO_001_SOURCE_V1: &str =
    "148a4d6088657d941a07ee32e04d8de3a67de548";
pub const QUAL_PROMO_001_QUALIFIER_V1: &str =
    "f847d33dc8873b44077f3c4acc61eae8f39e758d";
pub const QUAL_PROMO_001_WORKFLOW_ID_V1: u64 = 362_021_347;
pub const QUAL_PROMO_001_RUN_ID_V1: u64 = 35_439_636_330;

pub const QUAL_PROMO_002_SOURCE_V1: &str =
    "246df6db0c71e3075ee4e2bfb060e223c3d4087a";
pub const QUAL_PROMO_002_QUALIFIER_V1: &str =
    "729cdc6ca6bd3851267bacfb8e4f0a56dada74a5";
pub const QUAL_PROMO_002_WORKFLOW_ID_V1: u64 = 362_050_996;
pub const QUAL_PROMO_002_RUN_ID_V1: u64 = 35_445_177_427;

/// Compile-time repository provenance for one future D5C source lineage.
///
/// Fields are private, this type has no `Deserialize`, no `Default`, and this
/// source tranche intentionally provides no constructor and no global instance.
/// A later, explicitly reviewed QUAL-PROMO-003 instantiation change must add the
/// one frozen constant after the exact external attestation exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrozenAuthorityPromotionPinV1 {
    schema_id: &'static str,

    promo001_source_sha: &'static str,
    promo001_qualifier_sha: &'static str,
    promo001_workflow_id: u64,
    promo001_run_id: u64,
    promo001_run_attempt: u32,

    promo002_source_sha: &'static str,
    promo002_qualifier_sha: &'static str,
    promo002_workflow_id: u64,
    promo002_qualifier_run_id: u64,
    promo002_qualifier_run_attempt: u32,

    attestor_workflow_id: u64,
    attestor_run_id: u64,
    attestor_run_attempt: u32,
    collector_head_sha: &'static str,

    attestation_artifact_id: u64,
    attestation_artifact_name: &'static str,
    attestation_artifact_sha256: &'static str,

    prerequisite_manifest_schema_id: &'static str,
    external_evidence_schema_id: &'static str,
    manifest_lineage_id: &'static str,

    manifest_sha256: &'static str,
    records_sha256: &'static str,
    github_api_evidence_sha256: &'static str,
    receipt_sha256: &'static str,
    checksums_sha256: &'static str,
}

impl FrozenAuthorityPromotionPinV1 {
    pub const fn schema_id(&self) -> &'static str {
        self.schema_id
    }

    pub const fn promo001_source_sha(&self) -> &'static str {
        self.promo001_source_sha
    }

    pub const fn promo001_qualifier_sha(&self) -> &'static str {
        self.promo001_qualifier_sha
    }

    pub const fn promo001_run_id(&self) -> u64 {
        self.promo001_run_id
    }

    pub const fn promo002_source_sha(&self) -> &'static str {
        self.promo002_source_sha
    }

    pub const fn promo002_qualifier_sha(&self) -> &'static str {
        self.promo002_qualifier_sha
    }

    pub const fn promo002_qualifier_run_id(&self) -> u64 {
        self.promo002_qualifier_run_id
    }

    pub const fn attestor_run_id(&self) -> u64 {
        self.attestor_run_id
    }

    pub const fn attestor_run_attempt(&self) -> u32 {
        self.attestor_run_attempt
    }

    pub const fn collector_head_sha(&self) -> &'static str {
        self.collector_head_sha
    }

    pub const fn attestation_artifact_id(&self) -> u64 {
        self.attestation_artifact_id
    }

    pub const fn attestation_artifact_name(&self) -> &'static str {
        self.attestation_artifact_name
    }

    pub const fn attestation_artifact_sha256(&self) -> &'static str {
        self.attestation_artifact_sha256
    }

    pub const fn manifest_lineage_id(&self) -> &'static str {
        self.manifest_lineage_id
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthorityPromotionPinErrorV1 {
    SchemaMismatch,
    Promo001SourceMismatch,
    Promo001QualifierMismatch,
    Promo001WorkflowMismatch,
    Promo001RunMismatch,
    Promo001AttemptMismatch,
    Promo002SourceMismatch,
    Promo002QualifierMismatch,
    Promo002WorkflowMismatch,
    Promo002QualifierRunMismatch,
    Promo002QualifierAttemptMismatch,
    MissingAttestorWorkflowId,
    MissingAttestorRunId,
    AttestorAttemptMismatch,
    MalformedCollectorHeadSha,
    MissingAttestationArtifactId,
    AttestationArtifactNameMismatch,
    MalformedAttestationArtifactSha256,
    PrerequisiteManifestSchemaMismatch,
    ExternalEvidenceSchemaMismatch,
    MissingManifestLineage,
    MalformedManifestSha256,
    MalformedRecordsSha256,
    MalformedGithubApiEvidenceSha256,
    MalformedReceiptSha256,
    MalformedChecksumsSha256,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthorityPromotionPinVerificationV1 {
    pub qualified_source_identities_bound: bool,
    pub exact_first_attempt_qualifiers_bound: bool,
    pub external_attestation_subject_bound: bool,
    pub retained_payload_digests_bound: bool,
}

impl AuthorityPromotionPinVerificationV1 {
    pub const fn structurally_satisfied(self) -> bool {
        self.qualified_source_identities_bound
            && self.exact_first_attempt_qualifiers_bound
            && self.external_attestation_subject_bound
            && self.retained_payload_digests_bound
    }

    /// A source pin is release provenance, never live plant authority.
    pub const fn establishes_runtime_authority(self) -> bool {
        false
    }

    /// Live D4 mode / epoch / freshness must still be checked at control time.
    pub const fn verifies_live_d4_state(self) -> bool {
        false
    }

    /// GitHub/artifact truth must be reverified by the future D5C qualifier.
    pub const fn performs_external_reverification(self) -> bool {
        false
    }
}

pub fn verify_frozen_authority_promotion_pin_v1(
    pin: &FrozenAuthorityPromotionPinV1,
) -> Result<AuthorityPromotionPinVerificationV1, AuthorityPromotionPinErrorV1> {
    if pin.schema_id != AUTHORITY_PROMOTION_PIN_SCHEMA_V1 {
        return Err(AuthorityPromotionPinErrorV1::SchemaMismatch);
    }

    if pin.promo001_source_sha != QUAL_PROMO_001_SOURCE_V1 {
        return Err(AuthorityPromotionPinErrorV1::Promo001SourceMismatch);
    }
    if pin.promo001_qualifier_sha != QUAL_PROMO_001_QUALIFIER_V1 {
        return Err(AuthorityPromotionPinErrorV1::Promo001QualifierMismatch);
    }
    if pin.promo001_workflow_id != QUAL_PROMO_001_WORKFLOW_ID_V1 {
        return Err(AuthorityPromotionPinErrorV1::Promo001WorkflowMismatch);
    }
    if pin.promo001_run_id != QUAL_PROMO_001_RUN_ID_V1 {
        return Err(AuthorityPromotionPinErrorV1::Promo001RunMismatch);
    }
    if pin.promo001_run_attempt != 1 {
        return Err(AuthorityPromotionPinErrorV1::Promo001AttemptMismatch);
    }

    if pin.promo002_source_sha != QUAL_PROMO_002_SOURCE_V1 {
        return Err(AuthorityPromotionPinErrorV1::Promo002SourceMismatch);
    }
    if pin.promo002_qualifier_sha != QUAL_PROMO_002_QUALIFIER_V1 {
        return Err(AuthorityPromotionPinErrorV1::Promo002QualifierMismatch);
    }
    if pin.promo002_workflow_id != QUAL_PROMO_002_WORKFLOW_ID_V1 {
        return Err(AuthorityPromotionPinErrorV1::Promo002WorkflowMismatch);
    }
    if pin.promo002_qualifier_run_id != QUAL_PROMO_002_RUN_ID_V1 {
        return Err(AuthorityPromotionPinErrorV1::Promo002QualifierRunMismatch);
    }
    if pin.promo002_qualifier_run_attempt != 1 {
        return Err(AuthorityPromotionPinErrorV1::Promo002QualifierAttemptMismatch);
    }

    if pin.attestor_workflow_id == 0 {
        return Err(AuthorityPromotionPinErrorV1::MissingAttestorWorkflowId);
    }
    if pin.attestor_run_id == 0 {
        return Err(AuthorityPromotionPinErrorV1::MissingAttestorRunId);
    }
    if pin.attestor_run_attempt != 1 {
        return Err(AuthorityPromotionPinErrorV1::AttestorAttemptMismatch);
    }
    if !is_lower_hex_nonzero(pin.collector_head_sha, 40) {
        return Err(AuthorityPromotionPinErrorV1::MalformedCollectorHeadSha);
    }

    if pin.attestation_artifact_id == 0 {
        return Err(AuthorityPromotionPinErrorV1::MissingAttestationArtifactId);
    }
    let expected_artifact_name = format!("qual-promo-002-{}", pin.collector_head_sha);
    if pin.attestation_artifact_name != expected_artifact_name {
        return Err(AuthorityPromotionPinErrorV1::AttestationArtifactNameMismatch);
    }
    if !is_lower_hex_nonzero(pin.attestation_artifact_sha256, 64) {
        return Err(AuthorityPromotionPinErrorV1::MalformedAttestationArtifactSha256);
    }

    if pin.prerequisite_manifest_schema_id != AUTHORITY_PROMOTION_PREREQUISITE_SCHEMA_V1 {
        return Err(AuthorityPromotionPinErrorV1::PrerequisiteManifestSchemaMismatch);
    }
    if pin.external_evidence_schema_id != QUAL_PROMO_002_EVIDENCE_SCHEMA_V1 {
        return Err(AuthorityPromotionPinErrorV1::ExternalEvidenceSchemaMismatch);
    }
    if is_placeholder(pin.manifest_lineage_id) {
        return Err(AuthorityPromotionPinErrorV1::MissingManifestLineage);
    }

    for (value, error) in [
        (pin.manifest_sha256, AuthorityPromotionPinErrorV1::MalformedManifestSha256),
        (pin.records_sha256, AuthorityPromotionPinErrorV1::MalformedRecordsSha256),
        (
            pin.github_api_evidence_sha256,
            AuthorityPromotionPinErrorV1::MalformedGithubApiEvidenceSha256,
        ),
        (pin.receipt_sha256, AuthorityPromotionPinErrorV1::MalformedReceiptSha256),
        (
            pin.checksums_sha256,
            AuthorityPromotionPinErrorV1::MalformedChecksumsSha256,
        ),
    ] {
        if !is_lower_hex_nonzero(value, 64) {
            return Err(error);
        }
    }

    Ok(AuthorityPromotionPinVerificationV1 {
        qualified_source_identities_bound: true,
        exact_first_attempt_qualifiers_bound: true,
        external_attestation_subject_bound: true,
        retained_payload_digests_bound: true,
    })
}

fn is_lower_hex_nonzero(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len
        && value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        && value.bytes().any(|byte| byte != b'0')
}

fn is_placeholder(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed.is_empty()
        || matches!(
            trimmed.to_ascii_lowercase().as_str(),
            "none" | "unset" | "pending" | "placeholder" | "todo"
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str = "1111111111111111111111111111111111111111";
    const DIGEST_A: &str =
        "1111111111111111111111111111111111111111111111111111111111111111";
    const DIGEST_B: &str =
        "2222222222222222222222222222222222222222222222222222222222222222";
    const DIGEST_C: &str =
        "3333333333333333333333333333333333333333333333333333333333333333";
    const DIGEST_D: &str =
        "4444444444444444444444444444444444444444444444444444444444444444";
    const DIGEST_E: &str =
        "5555555555555555555555555555555555555555555555555555555555555555";
    const DIGEST_F: &str =
        "6666666666666666666666666666666666666666666666666666666666666666";

    fn synthetic_pin() -> FrozenAuthorityPromotionPinV1 {
        FrozenAuthorityPromotionPinV1 {
            schema_id: AUTHORITY_PROMOTION_PIN_SCHEMA_V1,
            promo001_source_sha: QUAL_PROMO_001_SOURCE_V1,
            promo001_qualifier_sha: QUAL_PROMO_001_QUALIFIER_V1,
            promo001_workflow_id: QUAL_PROMO_001_WORKFLOW_ID_V1,
            promo001_run_id: QUAL_PROMO_001_RUN_ID_V1,
            promo001_run_attempt: 1,
            promo002_source_sha: QUAL_PROMO_002_SOURCE_V1,
            promo002_qualifier_sha: QUAL_PROMO_002_QUALIFIER_V1,
            promo002_workflow_id: QUAL_PROMO_002_WORKFLOW_ID_V1,
            promo002_qualifier_run_id: QUAL_PROMO_002_RUN_ID_V1,
            promo002_qualifier_run_attempt: 1,
            attestor_workflow_id: 123,
            attestor_run_id: 456,
            attestor_run_attempt: 1,
            collector_head_sha: SHA_A,
            attestation_artifact_id: 789,
            attestation_artifact_name:
                "qual-promo-002-1111111111111111111111111111111111111111",
            attestation_artifact_sha256: DIGEST_A,
            prerequisite_manifest_schema_id: AUTHORITY_PROMOTION_PREREQUISITE_SCHEMA_V1,
            external_evidence_schema_id: QUAL_PROMO_002_EVIDENCE_SCHEMA_V1,
            manifest_lineage_id: "synthetic-non-authority-test-lineage",
            manifest_sha256: DIGEST_B,
            records_sha256: DIGEST_C,
            github_api_evidence_sha256: DIGEST_D,
            receipt_sha256: DIGEST_E,
            checksums_sha256: DIGEST_F,
        }
    }

    #[test]
    fn structurally_complete_synthetic_pin_is_still_not_runtime_authority() {
        let verification = verify_frozen_authority_promotion_pin_v1(&synthetic_pin()).unwrap();
        assert!(verification.structurally_satisfied());
        assert!(!verification.establishes_runtime_authority());
        assert!(!verification.verifies_live_d4_state());
        assert!(!verification.performs_external_reverification());
    }

    #[test]
    fn promo002_qualifier_substitution_fails_closed() {
        let mut pin = synthetic_pin();
        pin.promo002_qualifier_sha = SHA_A;
        assert_eq!(
            verify_frozen_authority_promotion_pin_v1(&pin),
            Err(AuthorityPromotionPinErrorV1::Promo002QualifierMismatch)
        );
    }

    #[test]
    fn any_attestor_rerun_requires_a_new_versioned_lineage() {
        let mut pin = synthetic_pin();
        pin.attestor_run_attempt = 2;
        assert_eq!(
            verify_frozen_authority_promotion_pin_v1(&pin),
            Err(AuthorityPromotionPinErrorV1::AttestorAttemptMismatch)
        );
    }

    #[test]
    fn artifact_name_must_bind_the_collector_head() {
        let mut pin = synthetic_pin();
        pin.attestation_artifact_name = "qual-promo-002-wrong";
        assert_eq!(
            verify_frozen_authority_promotion_pin_v1(&pin),
            Err(AuthorityPromotionPinErrorV1::AttestationArtifactNameMismatch)
        );
    }

    #[test]
    fn placeholder_manifest_lineage_is_rejected() {
        let mut pin = synthetic_pin();
        pin.manifest_lineage_id = "PENDING";
        assert_eq!(
            verify_frozen_authority_promotion_pin_v1(&pin),
            Err(AuthorityPromotionPinErrorV1::MissingManifestLineage)
        );
    }

    #[test]
    fn all_zero_payload_digest_is_rejected() {
        let mut pin = synthetic_pin();
        pin.records_sha256 =
            "0000000000000000000000000000000000000000000000000000000000000000";
        assert_eq!(
            verify_frozen_authority_promotion_pin_v1(&pin),
            Err(AuthorityPromotionPinErrorV1::MalformedRecordsSha256)
        );
    }
}
