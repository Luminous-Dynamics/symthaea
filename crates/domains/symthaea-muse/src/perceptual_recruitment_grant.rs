// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1RIR-A: restricted recruitment admission evidence.
//!
//! Real-world recruitment linkage remains outside the scientific evidence plane.
//! This module exposes only a frozen duplicate-control policy, signed opaque
//! enrollment grants, a frozen issuance register, terminal grant dispositions,
//! and signed aggregate accounting.
//!
//! The strongest supported duplicate claim is deliberately scoped to the frozen
//! recruitment mechanism. None of these artifacts establish universal human
//! identity, legal de-identification, or a participant bearer credential.

use crate::evidence_digest::{canonical_json_bytes, canonical_json_sha256, decode_hex_32};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::{rngs::OsRng, RngCore};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const PERCEPTUAL_RECRUITMENT_POLICY_VERSION: &str =
    "mel003-perceptual-recruitment-policy-v1";
pub const PERCEPTUAL_RECRUITMENT_GRANT_VERSION: &str =
    "mel003-perceptual-recruitment-enrollment-grant-v1";
pub const PERCEPTUAL_RECRUITMENT_ISSUANCE_REGISTER_VERSION: &str =
    "mel003-perceptual-recruitment-issuance-register-v1";
pub const PERCEPTUAL_RECRUITMENT_GRANT_LEDGER_VERSION: &str =
    "mel003-perceptual-recruitment-grant-disposition-ledger-v1";
pub const PERCEPTUAL_RECRUITMENT_ACCOUNTING_VERSION: &str =
    "mel003-perceptual-recruitment-accounting-v1";
pub const ZERO_SHA256: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

const GRANT_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea.mel003.p1.restricted-recruitment.v1/enrollment-grant";
const ACCOUNTING_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea.mel003.p1.restricted-recruitment.v1/accounting";
const GRANT_ID_BYTES: usize = 16;
const ED25519_PUBLIC_KEY_BYTES: usize = 32;
const ED25519_SIGNATURE_BYTES: usize = 64;
const MAX_SIGNED_MESSAGE_BYTES: usize = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RestrictedRecruitmentLinkageMechanismV1 {
    ExistingRecruitmentAccount,
    InstitutionalParticipantId,
    VerifiedContactChannel,
    ReviewedOpaqueSubjectHandle,
    UnverifiableAnonymousRecruitment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DuplicateEnrollmentClaimCeilingV1 {
    /// Bounded claim only: no duplicate enrollment was accepted under the exact
    /// frozen recruitment/linkage mechanism.
    NoDuplicateAcceptedUnderFrozenMechanism,
    /// The recruitment design does not establish duplicate-subject control.
    DuplicatePreventionNotEstablished,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecruitmentVerifierIdentityV1 {
    pub signer_id: String,
    pub key_epoch: u64,
    pub verifying_key_bytes: Vec<u8>,
}

pub struct RecruitmentSigningKeyV1 {
    signer_id: String,
    key_epoch: u64,
    inner: SigningKey,
}

impl RecruitmentSigningKeyV1 {
    pub fn from_seed(
        signer_id: impl Into<String>,
        key_epoch: u64,
        seed: [u8; 32],
    ) -> Result<Self, PerceptualRecruitmentEvidenceIssueV1> {
        let signer_id = signer_id.into();
        if signer_id.trim().is_empty() || key_epoch == 0 || seed == [0u8; 32] {
            return Err(PerceptualRecruitmentEvidenceIssueV1::InvalidAuthorityIdentity);
        }
        Ok(Self {
            signer_id,
            key_epoch,
            inner: SigningKey::from_bytes(&seed),
        })
    }

    pub fn verifier_identity(&self) -> RecruitmentVerifierIdentityV1 {
        RecruitmentVerifierIdentityV1 {
            signer_id: self.signer_id.clone(),
            key_epoch: self.key_epoch,
            verifying_key_bytes: self.inner.verifying_key().to_bytes().to_vec(),
        }
    }

    fn sign(
        &self,
        domain: &[u8],
        message: &[u8],
    ) -> Result<Vec<u8>, PerceptualRecruitmentEvidenceIssueV1> {
        let transcript = signature_transcript(domain, message)?;
        Ok(self.inner.sign(&transcript).to_bytes().to_vec())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPerceptualRecruitmentPolicyV1 {
    pub policy_version: String,
    pub protocol_sha256: String,
    pub recruitment_material_sha256: String,
    pub participant_information_sha256: String,
    pub privacy_notice_sha256: String,
    pub linkage_retention_deletion_policy_sha256: String,
    pub linkage_mechanism: RestrictedRecruitmentLinkageMechanismV1,
    pub duplicate_enrollment_claim_ceiling: DuplicateEnrollmentClaimCeilingV1,
    pub recruitment_authority: RecruitmentVerifierIdentityV1,
    pub direct_identity_export_to_study_evidence_prohibited: bool,
    pub private_arm_mapping_access_prohibited: bool,
    pub scored_response_access_prohibited: bool,
    pub correctness_or_significance_access_prohibited: bool,
    pub recruitment_service_schedule_choice_prohibited: bool,
    pub enrollment_grant_reuse_prohibited: bool,
    pub post_enrollment_grant_reissue_prohibited: bool,
    pub policy_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenRecruitmentEnrollmentGrantV1 {
    pub grant_version: String,
    pub recruitment_policy_sha256: String,
    pub protocol_sha256: String,
    /// Random opaque 128-bit grant identifier. It is neither recruitment
    /// identity nor a long-lived participant bearer credential.
    pub grant_id: String,
    /// Opaque commitment to the restricted screening/eligibility transaction.
    pub eligibility_attempt_sha256: String,
    /// Opaque chronology-event commitment for later P1ER reconciliation.
    pub issued_chronology_event_sha256: String,
    pub authority_signer_id: String,
    pub authority_key_epoch: u64,
    pub authority_signature: Vec<u8>,
    pub grant_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecruitmentGrantIssuanceRecordV1 {
    pub sequence: u32,
    pub recruitment_policy_sha256: String,
    pub grant: FrozenRecruitmentEnrollmentGrantV1,
    pub previous_record_head_sha256: String,
    pub record_head_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenRecruitmentGrantIssuanceRegisterV1 {
    pub register_version: String,
    pub recruitment_policy_sha256: String,
    /// Append order is evidence. Do not sort after issuance.
    pub entries: Vec<RecruitmentGrantIssuanceRecordV1>,
    pub final_record_head_sha256: String,
    pub register_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecruitmentGrantDispositionV1 {
    ConsumedForEnrollment,
    WithdrawnBeforeEnrollment,
    RevokedBeforeEnrollment,
    ExpiredBeforeEnrollment,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecruitmentGrantDispositionRecordV1 {
    pub sequence: u32,
    pub recruitment_policy_sha256: String,
    pub grant_sha256: String,
    pub disposition: RecruitmentGrantDispositionV1,
    /// Present only for `ConsumedForEnrollment`, binding the one-time grant to
    /// the exact P1ENR eligibility gate later joined to allocation.
    pub eligibility_gate_sha256: Option<String>,
    pub disposition_chronology_event_sha256: String,
    pub previous_record_head_sha256: String,
    pub record_head_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenRecruitmentGrantDispositionLedgerV1 {
    pub ledger_version: String,
    pub recruitment_policy_sha256: String,
    pub entries: Vec<RecruitmentGrantDispositionRecordV1>,
    pub final_record_head_sha256: String,
    pub ledger_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenRecruitmentAccountingReceiptV1 {
    pub receipt_version: String,
    pub recruitment_policy_sha256: String,
    pub issuance_register_sha256: String,
    pub disposition_ledger_sha256: String,
    pub grants_issued_count: usize,
    pub grants_consumed_for_enrollment_count: usize,
    pub pre_enrollment_withdrawal_count: usize,
    pub pre_enrollment_revoked_count: usize,
    pub pre_enrollment_expired_count: usize,
    pub active_unused_grant_count: usize,
    /// Screening failures occur before grant issuance, so this is an explicit
    /// signed aggregate from the restricted recruitment authority rather than
    /// an inference from absent grants.
    pub screening_failure_count: usize,
    pub accounting_chronology_event_sha256: String,
    pub authority_signer_id: String,
    pub authority_key_epoch: u64,
    pub authority_signature: Vec<u8>,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualRecruitmentEvidenceIssueV1 {
    WrongPolicyVersion,
    InvalidDigest { field: String },
    InvalidAuthorityIdentity,
    MissingPolicyProtection { field: String },
    AnonymousRecruitmentOverclaimsDuplicatePrevention,
    PolicyDigestMismatch,
    WrongGrantVersion,
    RecruitmentPolicyMismatch,
    ProtocolMismatch,
    InvalidGrantId,
    GrantAuthorityIdentityMismatch,
    InvalidGrantSignature,
    GrantDigestMismatch,
    WrongIssuanceRegisterVersion,
    IssuanceRegisterPolicyMismatch,
    DuplicateIssuedGrant { grant_sha256: String },
    DuplicateIssuedGrantId { grant_id: String },
    InvalidIssuedGrant { index: usize },
    IssuanceSequenceMismatch { index: usize },
    IssuancePreviousHeadMismatch { index: usize },
    IssuanceRecordHeadMismatch { index: usize },
    IssuanceFinalHeadMismatch,
    IssuanceRegisterDigestMismatch,
    GrantAlreadyIssued { grant_sha256: String },
    WrongLedgerVersion,
    LedgerPolicyMismatch,
    UnknownIssuedGrant { index: usize },
    DuplicateTerminalGrant { grant_sha256: String },
    SequenceMismatch { index: usize },
    PreviousHeadMismatch { index: usize },
    InvalidDispositionBinding { index: usize },
    RecordHeadMismatch { index: usize },
    FinalHeadMismatch,
    LedgerDigestMismatch,
    GrantAlreadyTerminal { grant_sha256: String },
    WrongAccountingVersion,
    AccountingEvidenceMismatch,
    AccountingCountMismatch { field: String },
    AccountingAuthorityIdentityMismatch,
    InvalidAccountingSignature,
    AccountingDigestMismatch,
    SerializationFailed,
    EntropyUnavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct RecruitmentPolicyCommitmentV1<'a> {
    policy_version: &'a str,
    protocol_sha256: &'a str,
    recruitment_material_sha256: &'a str,
    participant_information_sha256: &'a str,
    privacy_notice_sha256: &'a str,
    linkage_retention_deletion_policy_sha256: &'a str,
    linkage_mechanism: RestrictedRecruitmentLinkageMechanismV1,
    duplicate_enrollment_claim_ceiling: DuplicateEnrollmentClaimCeilingV1,
    recruitment_authority: &'a RecruitmentVerifierIdentityV1,
    direct_identity_export_to_study_evidence_prohibited: bool,
    private_arm_mapping_access_prohibited: bool,
    scored_response_access_prohibited: bool,
    correctness_or_significance_access_prohibited: bool,
    recruitment_service_schedule_choice_prohibited: bool,
    enrollment_grant_reuse_prohibited: bool,
    post_enrollment_grant_reissue_prohibited: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct RecruitmentGrantStatementV1<'a> {
    recruitment_policy_sha256: &'a str,
    protocol_sha256: &'a str,
    grant_id: &'a str,
    eligibility_attempt_sha256: &'a str,
    issued_chronology_event_sha256: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct RecruitmentGrantCommitmentV1<'a> {
    grant_version: &'a str,
    statement: RecruitmentGrantStatementV1<'a>,
    authority_signer_id: &'a str,
    authority_key_epoch: u64,
    authority_signature: &'a [u8],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct RecruitmentIssuanceRecordCommitmentV1<'a> {
    sequence: u32,
    recruitment_policy_sha256: &'a str,
    grant: &'a FrozenRecruitmentEnrollmentGrantV1,
    previous_record_head_sha256: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct RecruitmentDispositionRecordCommitmentV1<'a> {
    sequence: u32,
    recruitment_policy_sha256: &'a str,
    grant_sha256: &'a str,
    disposition: RecruitmentGrantDispositionV1,
    eligibility_gate_sha256: &'a Option<String>,
    disposition_chronology_event_sha256: &'a str,
    previous_record_head_sha256: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct RecruitmentAccountingStatementV1<'a> {
    recruitment_policy_sha256: &'a str,
    issuance_register_sha256: &'a str,
    disposition_ledger_sha256: &'a str,
    grants_issued_count: usize,
    grants_consumed_for_enrollment_count: usize,
    pre_enrollment_withdrawal_count: usize,
    pre_enrollment_revoked_count: usize,
    pre_enrollment_expired_count: usize,
    active_unused_grant_count: usize,
    screening_failure_count: usize,
    accounting_chronology_event_sha256: &'a str,
}

pub fn recruitment_policy_commitment(
    policy: &FrozenPerceptualRecruitmentPolicyV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&RecruitmentPolicyCommitmentV1 {
        policy_version: &policy.policy_version,
        protocol_sha256: &policy.protocol_sha256,
        recruitment_material_sha256: &policy.recruitment_material_sha256,
        participant_information_sha256: &policy.participant_information_sha256,
        privacy_notice_sha256: &policy.privacy_notice_sha256,
        linkage_retention_deletion_policy_sha256: &policy.linkage_retention_deletion_policy_sha256,
        linkage_mechanism: policy.linkage_mechanism,
        duplicate_enrollment_claim_ceiling: policy.duplicate_enrollment_claim_ceiling,
        recruitment_authority: &policy.recruitment_authority,
        direct_identity_export_to_study_evidence_prohibited: policy
            .direct_identity_export_to_study_evidence_prohibited,
        private_arm_mapping_access_prohibited: policy.private_arm_mapping_access_prohibited,
        scored_response_access_prohibited: policy.scored_response_access_prohibited,
        correctness_or_significance_access_prohibited: policy
            .correctness_or_significance_access_prohibited,
        recruitment_service_schedule_choice_prohibited: policy
            .recruitment_service_schedule_choice_prohibited,
        enrollment_grant_reuse_prohibited: policy.enrollment_grant_reuse_prohibited,
        post_enrollment_grant_reissue_prohibited: policy.post_enrollment_grant_reissue_prohibited,
    })
}

pub fn seal_recruitment_policy(
    policy: &mut FrozenPerceptualRecruitmentPolicyV1,
) -> Result<(), serde_json::Error> {
    policy.policy_sha256 = recruitment_policy_commitment(policy)?;
    Ok(())
}

pub fn validate_recruitment_policy(
    policy: &FrozenPerceptualRecruitmentPolicyV1,
) -> Vec<PerceptualRecruitmentEvidenceIssueV1> {
    let mut issues = Vec::new();
    if policy.policy_version != PERCEPTUAL_RECRUITMENT_POLICY_VERSION {
        issues.push(PerceptualRecruitmentEvidenceIssueV1::WrongPolicyVersion);
    }
    for (field, value) in [
        ("protocol_sha256", policy.protocol_sha256.as_str()),
        (
            "recruitment_material_sha256",
            policy.recruitment_material_sha256.as_str(),
        ),
        (
            "participant_information_sha256",
            policy.participant_information_sha256.as_str(),
        ),
        (
            "privacy_notice_sha256",
            policy.privacy_notice_sha256.as_str(),
        ),
        (
            "linkage_retention_deletion_policy_sha256",
            policy.linkage_retention_deletion_policy_sha256.as_str(),
        ),
    ] {
        if decode_hex_32(value).is_none() {
            issues.push(PerceptualRecruitmentEvidenceIssueV1::InvalidDigest {
                field: field.into(),
            });
        }
    }
    if !valid_verifier_identity(&policy.recruitment_authority) {
        issues.push(PerceptualRecruitmentEvidenceIssueV1::InvalidAuthorityIdentity);
    }
    for (field, enabled) in [
        (
            "direct_identity_export_to_study_evidence_prohibited",
            policy.direct_identity_export_to_study_evidence_prohibited,
        ),
        (
            "private_arm_mapping_access_prohibited",
            policy.private_arm_mapping_access_prohibited,
        ),
        (
            "scored_response_access_prohibited",
            policy.scored_response_access_prohibited,
        ),
        (
            "correctness_or_significance_access_prohibited",
            policy.correctness_or_significance_access_prohibited,
        ),
        (
            "recruitment_service_schedule_choice_prohibited",
            policy.recruitment_service_schedule_choice_prohibited,
        ),
        (
            "enrollment_grant_reuse_prohibited",
            policy.enrollment_grant_reuse_prohibited,
        ),
        (
            "post_enrollment_grant_reissue_prohibited",
            policy.post_enrollment_grant_reissue_prohibited,
        ),
    ] {
        if !enabled {
            issues.push(PerceptualRecruitmentEvidenceIssueV1::MissingPolicyProtection {
                field: field.into(),
            });
        }
    }
    if policy.linkage_mechanism
        == RestrictedRecruitmentLinkageMechanismV1::UnverifiableAnonymousRecruitment
        && policy.duplicate_enrollment_claim_ceiling
            == DuplicateEnrollmentClaimCeilingV1::NoDuplicateAcceptedUnderFrozenMechanism
    {
        issues.push(
            PerceptualRecruitmentEvidenceIssueV1::AnonymousRecruitmentOverclaimsDuplicatePrevention,
        );
    }
    match recruitment_policy_commitment(policy) {
        Ok(found) if found == policy.policy_sha256 => {}
        _ => issues.push(PerceptualRecruitmentEvidenceIssueV1::PolicyDigestMismatch),
    }
    issues
}

pub fn issue_recruitment_enrollment_grant_os_rng(
    policy: &FrozenPerceptualRecruitmentPolicyV1,
    signing_key: &RecruitmentSigningKeyV1,
    eligibility_attempt_sha256: impl Into<String>,
    issued_chronology_event_sha256: impl Into<String>,
) -> Result<FrozenRecruitmentEnrollmentGrantV1, Vec<PerceptualRecruitmentEvidenceIssueV1>> {
    let policy_issues = validate_recruitment_policy(policy);
    if !policy_issues.is_empty() {
        return Err(policy_issues);
    }
    if signing_key.verifier_identity() != policy.recruitment_authority {
        return Err(vec![
            PerceptualRecruitmentEvidenceIssueV1::GrantAuthorityIdentityMismatch,
        ]);
    }
    let eligibility_attempt_sha256 = eligibility_attempt_sha256.into();
    let issued_chronology_event_sha256 = issued_chronology_event_sha256.into();
    for (field, value) in [
        (
            "eligibility_attempt_sha256",
            eligibility_attempt_sha256.as_str(),
        ),
        (
            "issued_chronology_event_sha256",
            issued_chronology_event_sha256.as_str(),
        ),
    ] {
        if decode_hex_32(value).is_none() {
            return Err(vec![PerceptualRecruitmentEvidenceIssueV1::InvalidDigest {
                field: field.into(),
            }]);
        }
    }
    let mut random = [0u8; GRANT_ID_BYTES];
    OsRng
        .try_fill_bytes(&mut random)
        .map_err(|_| vec![PerceptualRecruitmentEvidenceIssueV1::EntropyUnavailable])?;
    let grant_id = random.iter().map(|byte| format!("{byte:02x}")).collect();
    issue_recruitment_enrollment_grant_with_id(
        policy,
        signing_key,
        grant_id,
        eligibility_attempt_sha256,
        issued_chronology_event_sha256,
    )
}

fn issue_recruitment_enrollment_grant_with_id(
    policy: &FrozenPerceptualRecruitmentPolicyV1,
    signing_key: &RecruitmentSigningKeyV1,
    grant_id: String,
    eligibility_attempt_sha256: String,
    issued_chronology_event_sha256: String,
) -> Result<FrozenRecruitmentEnrollmentGrantV1, Vec<PerceptualRecruitmentEvidenceIssueV1>> {
    if !is_lower_hex_128(&grant_id) {
        return Err(vec![PerceptualRecruitmentEvidenceIssueV1::InvalidGrantId]);
    }
    let statement = RecruitmentGrantStatementV1 {
        recruitment_policy_sha256: &policy.policy_sha256,
        protocol_sha256: &policy.protocol_sha256,
        grant_id: &grant_id,
        eligibility_attempt_sha256: &eligibility_attempt_sha256,
        issued_chronology_event_sha256: &issued_chronology_event_sha256,
    };
    let message = canonical_json_bytes(&statement)
        .map_err(|_| vec![PerceptualRecruitmentEvidenceIssueV1::SerializationFailed])?;
    let authority_signature = signing_key
        .sign(GRANT_SIGNATURE_DOMAIN, &message)
        .map_err(|issue| vec![issue])?;
    let mut grant = FrozenRecruitmentEnrollmentGrantV1 {
        grant_version: PERCEPTUAL_RECRUITMENT_GRANT_VERSION.into(),
        recruitment_policy_sha256: policy.policy_sha256.clone(),
        protocol_sha256: policy.protocol_sha256.clone(),
        grant_id,
        eligibility_attempt_sha256,
        issued_chronology_event_sha256,
        authority_signer_id: policy.recruitment_authority.signer_id.clone(),
        authority_key_epoch: policy.recruitment_authority.key_epoch,
        authority_signature,
        grant_sha256: String::new(),
    };
    grant.grant_sha256 = recruitment_grant_commitment(&grant)
        .map_err(|_| vec![PerceptualRecruitmentEvidenceIssueV1::SerializationFailed])?;
    let issues = validate_recruitment_enrollment_grant(policy, &grant);
    if issues.is_empty() {
        Ok(grant)
    } else {
        Err(issues)
    }
}

pub fn recruitment_grant_commitment(
    grant: &FrozenRecruitmentEnrollmentGrantV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&RecruitmentGrantCommitmentV1 {
        grant_version: &grant.grant_version,
        statement: grant_statement(grant),
        authority_signer_id: &grant.authority_signer_id,
        authority_key_epoch: grant.authority_key_epoch,
        authority_signature: &grant.authority_signature,
    })
}

pub fn validate_recruitment_enrollment_grant(
    policy: &FrozenPerceptualRecruitmentPolicyV1,
    grant: &FrozenRecruitmentEnrollmentGrantV1,
) -> Vec<PerceptualRecruitmentEvidenceIssueV1> {
    let mut issues = validate_recruitment_policy(policy);
    if grant.grant_version != PERCEPTUAL_RECRUITMENT_GRANT_VERSION {
        issues.push(PerceptualRecruitmentEvidenceIssueV1::WrongGrantVersion);
    }
    if grant.recruitment_policy_sha256 != policy.policy_sha256 {
        issues.push(PerceptualRecruitmentEvidenceIssueV1::RecruitmentPolicyMismatch);
    }
    if grant.protocol_sha256 != policy.protocol_sha256 {
        issues.push(PerceptualRecruitmentEvidenceIssueV1::ProtocolMismatch);
    }
    if !is_lower_hex_128(&grant.grant_id) {
        issues.push(PerceptualRecruitmentEvidenceIssueV1::InvalidGrantId);
    }
    for (field, value) in [
        (
            "eligibility_attempt_sha256",
            grant.eligibility_attempt_sha256.as_str(),
        ),
        (
            "issued_chronology_event_sha256",
            grant.issued_chronology_event_sha256.as_str(),
        ),
    ] {
        if decode_hex_32(value).is_none() {
            issues.push(PerceptualRecruitmentEvidenceIssueV1::InvalidDigest {
                field: field.into(),
            });
        }
    }
    if grant.authority_signer_id != policy.recruitment_authority.signer_id
        || grant.authority_key_epoch != policy.recruitment_authority.key_epoch
    {
        issues.push(PerceptualRecruitmentEvidenceIssueV1::GrantAuthorityIdentityMismatch);
    }
    match canonical_json_bytes(&grant_statement(grant)) {
        Ok(message) => {
            if verify_signature(
                &policy.recruitment_authority,
                GRANT_SIGNATURE_DOMAIN,
                &message,
                &grant.authority_signature,
            )
            .is_err()
            {
                issues.push(PerceptualRecruitmentEvidenceIssueV1::InvalidGrantSignature);
            }
        }
        Err(_) => issues.push(PerceptualRecruitmentEvidenceIssueV1::SerializationFailed),
    }
    match recruitment_grant_commitment(grant) {
        Ok(found) if found == grant.grant_sha256 => {}
        _ => issues.push(PerceptualRecruitmentEvidenceIssueV1::GrantDigestMismatch),
    }
    issues
}

pub fn new_recruitment_grant_issuance_register(
    policy: &FrozenPerceptualRecruitmentPolicyV1,
) -> Result<FrozenRecruitmentGrantIssuanceRegisterV1, serde_json::Error> {
    let mut register = FrozenRecruitmentGrantIssuanceRegisterV1 {
        register_version: PERCEPTUAL_RECRUITMENT_ISSUANCE_REGISTER_VERSION.into(),
        recruitment_policy_sha256: policy.policy_sha256.clone(),
        entries: Vec::new(),
        final_record_head_sha256: ZERO_SHA256.into(),
        register_sha256: String::new(),
    };
    register.register_sha256 = recruitment_issuance_register_commitment(&register)?;
    Ok(register)
}

pub fn append_recruitment_grant_issuance(
    policy: &FrozenPerceptualRecruitmentPolicyV1,
    register: &mut FrozenRecruitmentGrantIssuanceRegisterV1,
    grant: FrozenRecruitmentEnrollmentGrantV1,
) -> Result<(), Vec<PerceptualRecruitmentEvidenceIssueV1>> {
    let issues = validate_recruitment_grant_issuance_register(policy, register);
    if !issues.is_empty() {
        return Err(issues);
    }
    let grant_issues = validate_recruitment_enrollment_grant(policy, &grant);
    if !grant_issues.is_empty() {
        return Err(grant_issues);
    }
    if register
        .entries
        .iter()
        .any(|entry| entry.grant.grant_sha256 == grant.grant_sha256)
    {
        return Err(vec![
            PerceptualRecruitmentEvidenceIssueV1::GrantAlreadyIssued {
                grant_sha256: grant.grant_sha256.clone(),
            },
        ]);
    }
    if register
        .entries
        .iter()
        .any(|entry| entry.grant.grant_id == grant.grant_id)
    {
        return Err(vec![
            PerceptualRecruitmentEvidenceIssueV1::DuplicateIssuedGrantId {
                grant_id: grant.grant_id.clone(),
            },
        ]);
    }
    let mut record = RecruitmentGrantIssuanceRecordV1 {
        sequence: register.entries.len() as u32,
        recruitment_policy_sha256: policy.policy_sha256.clone(),
        grant,
        previous_record_head_sha256: register.final_record_head_sha256.clone(),
        record_head_sha256: String::new(),
    };
    record.record_head_sha256 = recruitment_issuance_record_commitment(&record)
        .map_err(|_| vec![PerceptualRecruitmentEvidenceIssueV1::SerializationFailed])?;
    register.entries.push(record);
    register.final_record_head_sha256 = register
        .entries
        .last()
        .map(|entry| entry.record_head_sha256.clone())
        .unwrap_or_else(|| ZERO_SHA256.into());
    register.register_sha256 = recruitment_issuance_register_commitment(register)
        .map_err(|_| vec![PerceptualRecruitmentEvidenceIssueV1::SerializationFailed])?;
    let issues = validate_recruitment_grant_issuance_register(policy, register);
    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

pub fn recruitment_issuance_record_commitment(
    record: &RecruitmentGrantIssuanceRecordV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&RecruitmentIssuanceRecordCommitmentV1 {
        sequence: record.sequence,
        recruitment_policy_sha256: &record.recruitment_policy_sha256,
        grant: &record.grant,
        previous_record_head_sha256: &record.previous_record_head_sha256,
    })
}

pub fn recruitment_issuance_register_commitment(
    register: &FrozenRecruitmentGrantIssuanceRegisterV1,
) -> Result<String, serde_json::Error> {
    let mut unsigned = register.clone();
    unsigned.register_sha256.clear();
    canonical_json_sha256(&unsigned)
}

pub fn validate_recruitment_grant_issuance_register(
    policy: &FrozenPerceptualRecruitmentPolicyV1,
    register: &FrozenRecruitmentGrantIssuanceRegisterV1,
) -> Vec<PerceptualRecruitmentEvidenceIssueV1> {
    let mut issues = validate_recruitment_policy(policy);
    if register.register_version != PERCEPTUAL_RECRUITMENT_ISSUANCE_REGISTER_VERSION {
        issues.push(PerceptualRecruitmentEvidenceIssueV1::WrongIssuanceRegisterVersion);
    }
    if register.recruitment_policy_sha256 != policy.policy_sha256 {
        issues.push(PerceptualRecruitmentEvidenceIssueV1::IssuanceRegisterPolicyMismatch);
    }
    let mut grant_digests = BTreeSet::new();
    let mut grant_ids = BTreeSet::new();
    let mut previous = ZERO_SHA256.to_string();
    for (index, entry) in register.entries.iter().enumerate() {
        if entry.sequence != index as u32 {
            issues.push(PerceptualRecruitmentEvidenceIssueV1::IssuanceSequenceMismatch { index });
        }
        if entry.recruitment_policy_sha256 != policy.policy_sha256
            || !validate_recruitment_enrollment_grant(policy, &entry.grant).is_empty()
        {
            issues.push(PerceptualRecruitmentEvidenceIssueV1::InvalidIssuedGrant { index });
        }
        if !grant_digests.insert(entry.grant.grant_sha256.as_str()) {
            issues.push(PerceptualRecruitmentEvidenceIssueV1::DuplicateIssuedGrant {
                grant_sha256: entry.grant.grant_sha256.clone(),
            });
        }
        if !grant_ids.insert(entry.grant.grant_id.as_str()) {
            issues.push(PerceptualRecruitmentEvidenceIssueV1::DuplicateIssuedGrantId {
                grant_id: entry.grant.grant_id.clone(),
            });
        }
        if entry.previous_record_head_sha256 != previous {
            issues.push(PerceptualRecruitmentEvidenceIssueV1::IssuancePreviousHeadMismatch {
                index,
            });
        }
        match recruitment_issuance_record_commitment(entry) {
            Ok(found) if found == entry.record_head_sha256 => {}
            _ => issues.push(PerceptualRecruitmentEvidenceIssueV1::IssuanceRecordHeadMismatch {
                index,
            }),
        }
        previous = entry.record_head_sha256.clone();
    }
    if register.final_record_head_sha256 != previous {
        issues.push(PerceptualRecruitmentEvidenceIssueV1::IssuanceFinalHeadMismatch);
    }
    match recruitment_issuance_register_commitment(register) {
        Ok(found) if found == register.register_sha256 => {}
        _ => issues.push(PerceptualRecruitmentEvidenceIssueV1::IssuanceRegisterDigestMismatch),
    }
    issues
}

pub fn new_recruitment_grant_disposition_ledger(
    policy: &FrozenPerceptualRecruitmentPolicyV1,
    _register: &FrozenRecruitmentGrantIssuanceRegisterV1,
) -> Result<FrozenRecruitmentGrantDispositionLedgerV1, serde_json::Error> {
    let mut ledger = FrozenRecruitmentGrantDispositionLedgerV1 {
        ledger_version: PERCEPTUAL_RECRUITMENT_GRANT_LEDGER_VERSION.into(),
        recruitment_policy_sha256: policy.policy_sha256.clone(),
        entries: Vec::new(),
        final_record_head_sha256: ZERO_SHA256.into(),
        ledger_sha256: String::new(),
    };
    ledger.ledger_sha256 = recruitment_grant_ledger_commitment(&ledger)?;
    Ok(ledger)
}

#[allow(clippy::too_many_arguments)]
pub fn append_recruitment_grant_disposition(
    policy: &FrozenPerceptualRecruitmentPolicyV1,
    register: &FrozenRecruitmentGrantIssuanceRegisterV1,
    ledger: &mut FrozenRecruitmentGrantDispositionLedgerV1,
    grant: &FrozenRecruitmentEnrollmentGrantV1,
    disposition: RecruitmentGrantDispositionV1,
    eligibility_gate_sha256: Option<String>,
    disposition_chronology_event_sha256: impl Into<String>,
) -> Result<(), Vec<PerceptualRecruitmentEvidenceIssueV1>> {
    let issues = validate_recruitment_grant_disposition_ledger(policy, register, ledger);
    if !issues.is_empty() {
        return Err(issues);
    }
    let grant_issues = validate_recruitment_enrollment_grant(policy, grant);
    if !grant_issues.is_empty() {
        return Err(grant_issues);
    }
    if !register.entries.iter().any(|entry| entry.grant == *grant) {
        return Err(vec![PerceptualRecruitmentEvidenceIssueV1::UnknownIssuedGrant {
            index: ledger.entries.len(),
        }]);
    }
    if ledger
        .entries
        .iter()
        .any(|entry| entry.grant_sha256 == grant.grant_sha256)
    {
        return Err(vec![
            PerceptualRecruitmentEvidenceIssueV1::GrantAlreadyTerminal {
                grant_sha256: grant.grant_sha256.clone(),
            },
        ]);
    }
    if !valid_disposition_binding(disposition, eligibility_gate_sha256.as_deref()) {
        return Err(vec![
            PerceptualRecruitmentEvidenceIssueV1::InvalidDispositionBinding {
                index: ledger.entries.len(),
            },
        ]);
    }
    let disposition_chronology_event_sha256 = disposition_chronology_event_sha256.into();
    if decode_hex_32(&disposition_chronology_event_sha256).is_none() {
        return Err(vec![PerceptualRecruitmentEvidenceIssueV1::InvalidDigest {
            field: "disposition_chronology_event_sha256".into(),
        }]);
    }
    let mut record = RecruitmentGrantDispositionRecordV1 {
        sequence: ledger.entries.len() as u32,
        recruitment_policy_sha256: policy.policy_sha256.clone(),
        grant_sha256: grant.grant_sha256.clone(),
        disposition,
        eligibility_gate_sha256,
        disposition_chronology_event_sha256,
        previous_record_head_sha256: ledger.final_record_head_sha256.clone(),
        record_head_sha256: String::new(),
    };
    record.record_head_sha256 = recruitment_disposition_record_commitment(&record)
        .map_err(|_| vec![PerceptualRecruitmentEvidenceIssueV1::SerializationFailed])?;
    ledger.entries.push(record);
    ledger.final_record_head_sha256 = ledger
        .entries
        .last()
        .map(|entry| entry.record_head_sha256.clone())
        .unwrap_or_else(|| ZERO_SHA256.into());
    ledger.ledger_sha256 = recruitment_grant_ledger_commitment(ledger)
        .map_err(|_| vec![PerceptualRecruitmentEvidenceIssueV1::SerializationFailed])?;
    let issues = validate_recruitment_grant_disposition_ledger(policy, register, ledger);
    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

pub fn recruitment_disposition_record_commitment(
    record: &RecruitmentGrantDispositionRecordV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&RecruitmentDispositionRecordCommitmentV1 {
        sequence: record.sequence,
        recruitment_policy_sha256: &record.recruitment_policy_sha256,
        grant_sha256: &record.grant_sha256,
        disposition: record.disposition,
        eligibility_gate_sha256: &record.eligibility_gate_sha256,
        disposition_chronology_event_sha256: &record.disposition_chronology_event_sha256,
        previous_record_head_sha256: &record.previous_record_head_sha256,
    })
}

pub fn recruitment_grant_ledger_commitment(
    ledger: &FrozenRecruitmentGrantDispositionLedgerV1,
) -> Result<String, serde_json::Error> {
    let mut unsigned = ledger.clone();
    unsigned.ledger_sha256.clear();
    canonical_json_sha256(&unsigned)
}

pub fn validate_recruitment_grant_disposition_ledger(
    policy: &FrozenPerceptualRecruitmentPolicyV1,
    register: &FrozenRecruitmentGrantIssuanceRegisterV1,
    ledger: &FrozenRecruitmentGrantDispositionLedgerV1,
) -> Vec<PerceptualRecruitmentEvidenceIssueV1> {
    let mut issues = validate_recruitment_grant_issuance_register(policy, register);
    if ledger.ledger_version != PERCEPTUAL_RECRUITMENT_GRANT_LEDGER_VERSION {
        issues.push(PerceptualRecruitmentEvidenceIssueV1::WrongLedgerVersion);
    }
    if ledger.recruitment_policy_sha256 != policy.policy_sha256 {
        issues.push(PerceptualRecruitmentEvidenceIssueV1::LedgerPolicyMismatch);
    }
    let issued: BTreeSet<_> = register
        .entries
        .iter()
        .map(|entry| entry.grant.grant_sha256.as_str())
        .collect();
    let mut terminal = BTreeSet::new();
    let mut previous = ZERO_SHA256.to_string();
    for (index, entry) in ledger.entries.iter().enumerate() {
        if entry.sequence != index as u32 {
            issues.push(PerceptualRecruitmentEvidenceIssueV1::SequenceMismatch { index });
        }
        if entry.recruitment_policy_sha256 != policy.policy_sha256
            || !issued.contains(entry.grant_sha256.as_str())
        {
            issues.push(PerceptualRecruitmentEvidenceIssueV1::UnknownIssuedGrant { index });
        }
        if !terminal.insert(entry.grant_sha256.as_str()) {
            issues.push(PerceptualRecruitmentEvidenceIssueV1::DuplicateTerminalGrant {
                grant_sha256: entry.grant_sha256.clone(),
            });
        }
        if entry.previous_record_head_sha256 != previous {
            issues.push(PerceptualRecruitmentEvidenceIssueV1::PreviousHeadMismatch { index });
        }
        if !valid_disposition_binding(entry.disposition, entry.eligibility_gate_sha256.as_deref()) {
            issues.push(PerceptualRecruitmentEvidenceIssueV1::InvalidDispositionBinding { index });
        }
        if decode_hex_32(&entry.disposition_chronology_event_sha256).is_none() {
            issues.push(PerceptualRecruitmentEvidenceIssueV1::InvalidDigest {
                field: format!("entries[{index}].disposition_chronology_event_sha256"),
            });
        }
        match recruitment_disposition_record_commitment(entry) {
            Ok(found) if found == entry.record_head_sha256 => {}
            _ => issues.push(PerceptualRecruitmentEvidenceIssueV1::RecordHeadMismatch { index }),
        }
        previous = entry.record_head_sha256.clone();
    }
    if ledger.final_record_head_sha256 != previous {
        issues.push(PerceptualRecruitmentEvidenceIssueV1::FinalHeadMismatch);
    }
    match recruitment_grant_ledger_commitment(ledger) {
        Ok(found) if found == ledger.ledger_sha256 => {}
        _ => issues.push(PerceptualRecruitmentEvidenceIssueV1::LedgerDigestMismatch),
    }
    issues
}

pub fn sign_recruitment_accounting_receipt(
    policy: &FrozenPerceptualRecruitmentPolicyV1,
    signing_key: &RecruitmentSigningKeyV1,
    register: &FrozenRecruitmentGrantIssuanceRegisterV1,
    ledger: &FrozenRecruitmentGrantDispositionLedgerV1,
    screening_failure_count: usize,
    accounting_chronology_event_sha256: impl Into<String>,
) -> Result<FrozenRecruitmentAccountingReceiptV1, Vec<PerceptualRecruitmentEvidenceIssueV1>> {
    let issues = validate_recruitment_grant_disposition_ledger(policy, register, ledger);
    if !issues.is_empty() {
        return Err(issues);
    }
    if signing_key.verifier_identity() != policy.recruitment_authority {
        return Err(vec![
            PerceptualRecruitmentEvidenceIssueV1::AccountingAuthorityIdentityMismatch,
        ]);
    }
    let accounting_chronology_event_sha256 = accounting_chronology_event_sha256.into();
    if decode_hex_32(&accounting_chronology_event_sha256).is_none() {
        return Err(vec![PerceptualRecruitmentEvidenceIssueV1::InvalidDigest {
            field: "accounting_chronology_event_sha256".into(),
        }]);
    }
    let counts = disposition_counts(ledger);
    let active_unused_grant_count = register.entries.len().saturating_sub(ledger.entries.len());
    let mut receipt = FrozenRecruitmentAccountingReceiptV1 {
        receipt_version: PERCEPTUAL_RECRUITMENT_ACCOUNTING_VERSION.into(),
        recruitment_policy_sha256: policy.policy_sha256.clone(),
        issuance_register_sha256: register.register_sha256.clone(),
        disposition_ledger_sha256: ledger.ledger_sha256.clone(),
        grants_issued_count: register.entries.len(),
        grants_consumed_for_enrollment_count: counts.consumed,
        pre_enrollment_withdrawal_count: counts.withdrawn,
        pre_enrollment_revoked_count: counts.revoked,
        pre_enrollment_expired_count: counts.expired,
        active_unused_grant_count,
        screening_failure_count,
        accounting_chronology_event_sha256,
        authority_signer_id: policy.recruitment_authority.signer_id.clone(),
        authority_key_epoch: policy.recruitment_authority.key_epoch,
        authority_signature: Vec::new(),
        receipt_sha256: String::new(),
    };
    let message = canonical_json_bytes(&accounting_statement(&receipt))
        .map_err(|_| vec![PerceptualRecruitmentEvidenceIssueV1::SerializationFailed])?;
    receipt.authority_signature = signing_key
        .sign(ACCOUNTING_SIGNATURE_DOMAIN, &message)
        .map_err(|issue| vec![issue])?;
    receipt.receipt_sha256 = recruitment_accounting_receipt_commitment(&receipt)
        .map_err(|_| vec![PerceptualRecruitmentEvidenceIssueV1::SerializationFailed])?;
    let issues = validate_recruitment_accounting_receipt(policy, register, ledger, &receipt);
    if issues.is_empty() {
        Ok(receipt)
    } else {
        Err(issues)
    }
}

pub fn recruitment_accounting_receipt_commitment(
    receipt: &FrozenRecruitmentAccountingReceiptV1,
) -> Result<String, serde_json::Error> {
    let mut unsigned = receipt.clone();
    unsigned.receipt_sha256.clear();
    canonical_json_sha256(&unsigned)
}

pub fn validate_recruitment_accounting_receipt(
    policy: &FrozenPerceptualRecruitmentPolicyV1,
    register: &FrozenRecruitmentGrantIssuanceRegisterV1,
    ledger: &FrozenRecruitmentGrantDispositionLedgerV1,
    receipt: &FrozenRecruitmentAccountingReceiptV1,
) -> Vec<PerceptualRecruitmentEvidenceIssueV1> {
    let mut issues = validate_recruitment_grant_disposition_ledger(policy, register, ledger);
    if receipt.receipt_version != PERCEPTUAL_RECRUITMENT_ACCOUNTING_VERSION {
        issues.push(PerceptualRecruitmentEvidenceIssueV1::WrongAccountingVersion);
    }
    if receipt.recruitment_policy_sha256 != policy.policy_sha256
        || receipt.issuance_register_sha256 != register.register_sha256
        || receipt.disposition_ledger_sha256 != ledger.ledger_sha256
    {
        issues.push(PerceptualRecruitmentEvidenceIssueV1::AccountingEvidenceMismatch);
    }
    let counts = disposition_counts(ledger);
    let expected = [
        (
            "grants_issued_count",
            receipt.grants_issued_count,
            register.entries.len(),
        ),
        (
            "grants_consumed_for_enrollment_count",
            receipt.grants_consumed_for_enrollment_count,
            counts.consumed,
        ),
        (
            "pre_enrollment_withdrawal_count",
            receipt.pre_enrollment_withdrawal_count,
            counts.withdrawn,
        ),
        (
            "pre_enrollment_revoked_count",
            receipt.pre_enrollment_revoked_count,
            counts.revoked,
        ),
        (
            "pre_enrollment_expired_count",
            receipt.pre_enrollment_expired_count,
            counts.expired,
        ),
        (
            "active_unused_grant_count",
            receipt.active_unused_grant_count,
            register.entries.len().saturating_sub(ledger.entries.len()),
        ),
    ];
    for (field, found, expected) in expected {
        if found != expected {
            issues.push(PerceptualRecruitmentEvidenceIssueV1::AccountingCountMismatch {
                field: field.into(),
            });
        }
    }
    if decode_hex_32(&receipt.accounting_chronology_event_sha256).is_none() {
        issues.push(PerceptualRecruitmentEvidenceIssueV1::InvalidDigest {
            field: "accounting_chronology_event_sha256".into(),
        });
    }
    if receipt.authority_signer_id != policy.recruitment_authority.signer_id
        || receipt.authority_key_epoch != policy.recruitment_authority.key_epoch
    {
        issues.push(PerceptualRecruitmentEvidenceIssueV1::AccountingAuthorityIdentityMismatch);
    }
    match canonical_json_bytes(&accounting_statement(receipt)) {
        Ok(message) => {
            if verify_signature(
                &policy.recruitment_authority,
                ACCOUNTING_SIGNATURE_DOMAIN,
                &message,
                &receipt.authority_signature,
            )
            .is_err()
            {
                issues.push(PerceptualRecruitmentEvidenceIssueV1::InvalidAccountingSignature);
            }
        }
        Err(_) => issues.push(PerceptualRecruitmentEvidenceIssueV1::SerializationFailed),
    }
    match recruitment_accounting_receipt_commitment(receipt) {
        Ok(found) if found == receipt.receipt_sha256 => {}
        _ => issues.push(PerceptualRecruitmentEvidenceIssueV1::AccountingDigestMismatch),
    }
    issues
}

fn grant_statement(grant: &FrozenRecruitmentEnrollmentGrantV1) -> RecruitmentGrantStatementV1<'_> {
    RecruitmentGrantStatementV1 {
        recruitment_policy_sha256: &grant.recruitment_policy_sha256,
        protocol_sha256: &grant.protocol_sha256,
        grant_id: &grant.grant_id,
        eligibility_attempt_sha256: &grant.eligibility_attempt_sha256,
        issued_chronology_event_sha256: &grant.issued_chronology_event_sha256,
    }
}

fn accounting_statement(
    receipt: &FrozenRecruitmentAccountingReceiptV1,
) -> RecruitmentAccountingStatementV1<'_> {
    RecruitmentAccountingStatementV1 {
        recruitment_policy_sha256: &receipt.recruitment_policy_sha256,
        issuance_register_sha256: &receipt.issuance_register_sha256,
        disposition_ledger_sha256: &receipt.disposition_ledger_sha256,
        grants_issued_count: receipt.grants_issued_count,
        grants_consumed_for_enrollment_count: receipt.grants_consumed_for_enrollment_count,
        pre_enrollment_withdrawal_count: receipt.pre_enrollment_withdrawal_count,
        pre_enrollment_revoked_count: receipt.pre_enrollment_revoked_count,
        pre_enrollment_expired_count: receipt.pre_enrollment_expired_count,
        active_unused_grant_count: receipt.active_unused_grant_count,
        screening_failure_count: receipt.screening_failure_count,
        accounting_chronology_event_sha256: &receipt.accounting_chronology_event_sha256,
    }
}

#[derive(Default)]
struct DispositionCountsV1 {
    consumed: usize,
    withdrawn: usize,
    revoked: usize,
    expired: usize,
}

fn disposition_counts(ledger: &FrozenRecruitmentGrantDispositionLedgerV1) -> DispositionCountsV1 {
    let mut counts = DispositionCountsV1::default();
    for entry in &ledger.entries {
        match entry.disposition {
            RecruitmentGrantDispositionV1::ConsumedForEnrollment => counts.consumed += 1,
            RecruitmentGrantDispositionV1::WithdrawnBeforeEnrollment => counts.withdrawn += 1,
            RecruitmentGrantDispositionV1::RevokedBeforeEnrollment => counts.revoked += 1,
            RecruitmentGrantDispositionV1::ExpiredBeforeEnrollment => counts.expired += 1,
        }
    }
    counts
}

fn valid_disposition_binding(
    disposition: RecruitmentGrantDispositionV1,
    eligibility_gate_sha256: Option<&str>,
) -> bool {
    match disposition {
        RecruitmentGrantDispositionV1::ConsumedForEnrollment => {
            eligibility_gate_sha256.and_then(decode_hex_32).is_some()
        }
        RecruitmentGrantDispositionV1::WithdrawnBeforeEnrollment
        | RecruitmentGrantDispositionV1::RevokedBeforeEnrollment
        | RecruitmentGrantDispositionV1::ExpiredBeforeEnrollment => {
            eligibility_gate_sha256.is_none()
        }
    }
}

fn is_lower_hex_128(value: &str) -> bool {
    value.len() == GRANT_ID_BYTES * 2
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_verifier_identity(identity: &RecruitmentVerifierIdentityV1) -> bool {
    if identity.signer_id.trim().is_empty()
        || identity.key_epoch == 0
        || identity.verifying_key_bytes.len() != ED25519_PUBLIC_KEY_BYTES
    {
        return false;
    }
    let Ok(bytes): Result<[u8; ED25519_PUBLIC_KEY_BYTES], _> =
        identity.verifying_key_bytes.as_slice().try_into()
    else {
        return false;
    };
    VerifyingKey::from_bytes(&bytes).is_ok()
}

fn signature_transcript(
    domain: &[u8],
    message: &[u8],
) -> Result<Vec<u8>, PerceptualRecruitmentEvidenceIssueV1> {
    if domain.is_empty() || message.is_empty() || message.len() > MAX_SIGNED_MESSAGE_BYTES {
        return Err(PerceptualRecruitmentEvidenceIssueV1::SerializationFailed);
    }
    let mut transcript = Vec::with_capacity(8 + domain.len() + message.len());
    transcript.extend_from_slice(&(domain.len() as u64).to_le_bytes());
    transcript.extend_from_slice(domain);
    transcript.extend_from_slice(message);
    Ok(transcript)
}

fn verify_signature(
    identity: &RecruitmentVerifierIdentityV1,
    domain: &[u8],
    message: &[u8],
    signature: &[u8],
) -> Result<(), PerceptualRecruitmentEvidenceIssueV1> {
    if !valid_verifier_identity(identity) || signature.len() != ED25519_SIGNATURE_BYTES {
        return Err(PerceptualRecruitmentEvidenceIssueV1::InvalidAuthorityIdentity);
    }
    let key_bytes: [u8; ED25519_PUBLIC_KEY_BYTES] = identity
        .verifying_key_bytes
        .as_slice()
        .try_into()
        .map_err(|_| PerceptualRecruitmentEvidenceIssueV1::InvalidAuthorityIdentity)?;
    let key = VerifyingKey::from_bytes(&key_bytes)
        .map_err(|_| PerceptualRecruitmentEvidenceIssueV1::InvalidAuthorityIdentity)?;
    let signature = Signature::from_slice(signature)
        .map_err(|_| PerceptualRecruitmentEvidenceIssueV1::InvalidAuthorityIdentity)?;
    let transcript = signature_transcript(domain, message)?;
    key.verify(&transcript, &signature)
        .map_err(|_| PerceptualRecruitmentEvidenceIssueV1::InvalidAuthorityIdentity)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(marker: u8) -> String {
        format!("{marker:02x}").repeat(32)
    }

    fn policy_and_key() -> (FrozenPerceptualRecruitmentPolicyV1, RecruitmentSigningKeyV1) {
        let key =
            RecruitmentSigningKeyV1::from_seed("recruitment-fixture", 1, [0x44; 32]).unwrap();
        let mut policy = FrozenPerceptualRecruitmentPolicyV1 {
            policy_version: PERCEPTUAL_RECRUITMENT_POLICY_VERSION.into(),
            protocol_sha256: digest(1),
            recruitment_material_sha256: digest(2),
            participant_information_sha256: digest(3),
            privacy_notice_sha256: digest(4),
            linkage_retention_deletion_policy_sha256: digest(5),
            linkage_mechanism: RestrictedRecruitmentLinkageMechanismV1::ExistingRecruitmentAccount,
            duplicate_enrollment_claim_ceiling:
                DuplicateEnrollmentClaimCeilingV1::NoDuplicateAcceptedUnderFrozenMechanism,
            recruitment_authority: key.verifier_identity(),
            direct_identity_export_to_study_evidence_prohibited: true,
            private_arm_mapping_access_prohibited: true,
            scored_response_access_prohibited: true,
            correctness_or_significance_access_prohibited: true,
            recruitment_service_schedule_choice_prohibited: true,
            enrollment_grant_reuse_prohibited: true,
            post_enrollment_grant_reissue_prohibited: true,
            policy_sha256: String::new(),
        };
        seal_recruitment_policy(&mut policy).unwrap();
        (policy, key)
    }

    fn grant_with_id(
        policy: &FrozenPerceptualRecruitmentPolicyV1,
        key: &RecruitmentSigningKeyV1,
        id_byte: u8,
        attempt: u8,
        chronology: u8,
    ) -> FrozenRecruitmentEnrollmentGrantV1 {
        issue_recruitment_enrollment_grant_with_id(
            policy,
            key,
            format!("{id_byte:02x}").repeat(GRANT_ID_BYTES),
            digest(attempt),
            digest(chronology),
        )
        .unwrap()
    }

    #[test]
    fn issuance_register_drives_terminal_accounting() {
        let (policy, key) = policy_and_key();
        assert!(validate_recruitment_policy(&policy).is_empty());
        let consumed = grant_with_id(&policy, &key, 0x11, 6, 7);
        let withdrawn = grant_with_id(&policy, &key, 0x22, 8, 9);
        let active = grant_with_id(&policy, &key, 0x33, 10, 11);
        let mut register = new_recruitment_grant_issuance_register(&policy).unwrap();
        for grant in [consumed.clone(), withdrawn.clone(), active] {
            append_recruitment_grant_issuance(&policy, &mut register, grant).unwrap();
        }
        let mut ledger = new_recruitment_grant_disposition_ledger(&policy, &register).unwrap();
        append_recruitment_grant_disposition(
            &policy,
            &register,
            &mut ledger,
            &consumed,
            RecruitmentGrantDispositionV1::ConsumedForEnrollment,
            Some(digest(12)),
            digest(13),
        )
        .unwrap();
        append_recruitment_grant_disposition(
            &policy,
            &register,
            &mut ledger,
            &withdrawn,
            RecruitmentGrantDispositionV1::WithdrawnBeforeEnrollment,
            None,
            digest(14),
        )
        .unwrap();
        let accounting = sign_recruitment_accounting_receipt(
            &policy,
            &key,
            &register,
            &ledger,
            3,
            digest(15),
        )
        .unwrap();
        assert_eq!(accounting.grants_issued_count, 3);
        assert_eq!(accounting.grants_consumed_for_enrollment_count, 1);
        assert_eq!(accounting.pre_enrollment_withdrawal_count, 1);
        assert_eq!(accounting.active_unused_grant_count, 1);
        assert_eq!(accounting.screening_failure_count, 3);
        assert!(validate_recruitment_accounting_receipt(
            &policy,
            &register,
            &ledger,
            &accounting
        )
        .is_empty());
    }

    #[test]
    fn accounting_is_bound_to_exact_issuance_register_not_caller_subset() {
        let (policy, key) = policy_and_key();
        let first = grant_with_id(&policy, &key, 0x11, 6, 7);
        let second = grant_with_id(&policy, &key, 0x22, 8, 9);
        let mut full = new_recruitment_grant_issuance_register(&policy).unwrap();
        append_recruitment_grant_issuance(&policy, &mut full, first.clone()).unwrap();
        append_recruitment_grant_issuance(&policy, &mut full, second).unwrap();
        let ledger = new_recruitment_grant_disposition_ledger(&policy, &full).unwrap();
        let accounting =
            sign_recruitment_accounting_receipt(&policy, &key, &full, &ledger, 0, digest(10))
                .unwrap();

        let mut truncated = new_recruitment_grant_issuance_register(&policy).unwrap();
        append_recruitment_grant_issuance(&policy, &mut truncated, first).unwrap();
        assert!(validate_recruitment_accounting_receipt(
            &policy,
            &truncated,
            &ledger,
            &accounting
        )
        .iter()
        .any(|issue| matches!(
            issue,
            PerceptualRecruitmentEvidenceIssueV1::AccountingEvidenceMismatch
                | PerceptualRecruitmentEvidenceIssueV1::AccountingCountMismatch { .. }
        )));
    }

    #[test]
    fn one_grant_cannot_be_issued_twice_or_receive_two_terminal_dispositions() {
        let (policy, key) = policy_and_key();
        let grant = grant_with_id(&policy, &key, 0x11, 6, 7);
        let mut register = new_recruitment_grant_issuance_register(&policy).unwrap();
        append_recruitment_grant_issuance(&policy, &mut register, grant.clone()).unwrap();
        let replay = append_recruitment_grant_issuance(&policy, &mut register, grant.clone());
        assert!(matches!(
            replay,
            Err(ref issues) if issues.iter().any(|issue| matches!(
                issue,
                PerceptualRecruitmentEvidenceIssueV1::GrantAlreadyIssued { .. }
            ))
        ));

        let mut ledger = new_recruitment_grant_disposition_ledger(&policy, &register).unwrap();
        append_recruitment_grant_disposition(
            &policy,
            &register,
            &mut ledger,
            &grant,
            RecruitmentGrantDispositionV1::ConsumedForEnrollment,
            Some(digest(10)),
            digest(11),
        )
        .unwrap();
        let terminal_replay = append_recruitment_grant_disposition(
            &policy,
            &register,
            &mut ledger,
            &grant,
            RecruitmentGrantDispositionV1::WithdrawnBeforeEnrollment,
            None,
            digest(12),
        );
        assert!(matches!(
            terminal_replay,
            Err(ref issues) if issues.iter().any(|issue| matches!(
                issue,
                PerceptualRecruitmentEvidenceIssueV1::GrantAlreadyTerminal { .. }
            ))
        ));
    }

    #[test]
    fn disposition_of_unissued_grant_is_rejected() {
        let (policy, key) = policy_and_key();
        let issued = grant_with_id(&policy, &key, 0x11, 6, 7);
        let unissued = grant_with_id(&policy, &key, 0x22, 8, 9);
        let mut register = new_recruitment_grant_issuance_register(&policy).unwrap();
        append_recruitment_grant_issuance(&policy, &mut register, issued).unwrap();
        let mut ledger = new_recruitment_grant_disposition_ledger(&policy, &register).unwrap();
        let result = append_recruitment_grant_disposition(
            &policy,
            &register,
            &mut ledger,
            &unissued,
            RecruitmentGrantDispositionV1::WithdrawnBeforeEnrollment,
            None,
            digest(10),
        );
        assert!(matches!(
            result,
            Err(ref issues) if issues.iter().any(|issue| matches!(
                issue,
                PerceptualRecruitmentEvidenceIssueV1::UnknownIssuedGrant { .. }
            ))
        ));
    }

    #[test]
    fn anonymous_recruitment_cannot_claim_duplicate_prevention() {
        let (mut policy, _) = policy_and_key();
        policy.linkage_mechanism =
            RestrictedRecruitmentLinkageMechanismV1::UnverifiableAnonymousRecruitment;
        seal_recruitment_policy(&mut policy).unwrap();
        assert!(validate_recruitment_policy(&policy).iter().any(|issue| matches!(
            issue,
            PerceptualRecruitmentEvidenceIssueV1::AnonymousRecruitmentOverclaimsDuplicatePrevention
        )));
    }

    #[test]
    fn consumed_grant_requires_exact_gate_digest() {
        let (policy, key) = policy_and_key();
        let grant = grant_with_id(&policy, &key, 0x11, 6, 7);
        let mut register = new_recruitment_grant_issuance_register(&policy).unwrap();
        append_recruitment_grant_issuance(&policy, &mut register, grant.clone()).unwrap();
        let mut ledger = new_recruitment_grant_disposition_ledger(&policy, &register).unwrap();
        let result = append_recruitment_grant_disposition(
            &policy,
            &register,
            &mut ledger,
            &grant,
            RecruitmentGrantDispositionV1::ConsumedForEnrollment,
            None,
            digest(11),
        );
        assert!(matches!(
            result,
            Err(ref issues) if issues.iter().any(|issue| matches!(
                issue,
                PerceptualRecruitmentEvidenceIssueV1::InvalidDispositionBinding { .. }
            ))
        ));
    }

    #[test]
    fn changed_grant_signature_fails_closed() {
        let (policy, key) = policy_and_key();
        let mut grant = grant_with_id(&policy, &key, 0x11, 6, 7);
        grant.authority_signature[0] ^= 0x01;
        assert!(validate_recruitment_enrollment_grant(&policy, &grant).iter().any(|issue| matches!(
            issue,
            PerceptualRecruitmentEvidenceIssueV1::InvalidGrantSignature
                | PerceptualRecruitmentEvidenceIssueV1::GrantDigestMismatch
        )));
    }
}
