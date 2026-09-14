// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Current-head-qualified verifier independence composition.
//!
//! This layer consumes opaque capabilities from ASSURE-015D and ASSURE-015E.
//! A persisted separation report or currentness report cannot recreate the final
//! capability: all exact bindings must still agree at one use-time and one
//! shared composition challenge.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_evidence_active_independence::ActiveIndependenceQualification;
use symthaea_evidence_lifecycle_head_currentness::{
    CurrentActiveVerifierProfile, LifecycleHeadAuthorityPolicy,
    LifecycleHeadCurrentnessAttestation,
};

pub const CURRENT_INDEPENDENCE_SCHEMA_V1: &str =
    "symthaea.assurance.current-head-qualified-independence.v1";

const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.current-head-qualified-independence.digest.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurrentIndependenceDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurrentIndependenceIssue {
    InvalidLeftAuthorityPolicy,
    InvalidRightAuthorityPolicy,
    InvalidLeftCurrentnessAttestation,
    InvalidRightCurrentnessAttestation,
    LeftCurrentnessDigestMismatch,
    RightCurrentnessDigestMismatch,
    LeftAuthorityPolicyDigestMismatch,
    RightAuthorityPolicyDigestMismatch,
    LeftHeadStatementDigestMismatch,
    RightHeadStatementDigestMismatch,
    LeftHeadSequenceMismatch,
    RightHeadSequenceMismatch,
    LeftProfileIdMismatch,
    RightProfileIdMismatch,
    LeftRecordDigestMismatch,
    RightRecordDigestMismatch,
    LeftGraphDigestMismatch,
    RightGraphDigestMismatch,
    LeftCompletenessDigestMismatch,
    RightCompletenessDigestMismatch,
    LeftProvenanceDigestMismatch,
    RightProvenanceDigestMismatch,
    LeftLifecycleDigestMismatch,
    RightLifecycleDigestMismatch,
    LeftVerificationTimeMismatch,
    RightVerificationTimeMismatch,
    LeftUseTimeMismatch { observed: u64, required: u64 },
    RightUseTimeMismatch { observed: u64, required: u64 },
    LeftCurrentnessUseTimeMismatch { observed: u64, required: u64 },
    RightCurrentnessUseTimeMismatch { observed: u64, required: u64 },
    LeftChallengeMismatch,
    RightChallengeMismatch,
}

impl CurrentIndependenceIssue {
    fn is_blocking(&self) -> bool {
        matches!(
            self,
            Self::LeftUseTimeMismatch { .. }
                | Self::RightUseTimeMismatch { .. }
                | Self::LeftCurrentnessUseTimeMismatch { .. }
                | Self::RightCurrentnessUseTimeMismatch { .. }
                | Self::LeftChallengeMismatch
                | Self::RightChallengeMismatch
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CurrentIndependenceReport {
    pub schema_version: String,
    pub disposition: CurrentIndependenceDisposition,
    pub left_profile_id: String,
    pub right_profile_id: String,
    pub left_verifier_ref: String,
    pub right_verifier_ref: String,
    pub active_independence_qualification_digest: String,
    pub separation_policy_digest: String,
    pub graph_digest: String,
    pub relation_completeness_digest: String,
    pub left_record_digest: String,
    pub right_record_digest: String,
    pub left_lifecycle_digest: String,
    pub right_lifecycle_digest: String,
    pub left_head_sequence: u64,
    pub right_head_sequence: u64,
    pub left_head_statement_digest: String,
    pub right_head_statement_digest: String,
    pub left_currentness_attestation_digest: String,
    pub right_currentness_attestation_digest: String,
    pub left_head_authority_policy_digest: Option<String>,
    pub right_head_authority_policy_digest: Option<String>,
    pub composition_challenge_nonce_blake3_hex: String,
    pub use_at_ms: u64,
    pub issues: Vec<CurrentIndependenceIssue>,
    pub qualification_digest: Option<String>,
}

impl CurrentIndependenceReport {
    pub const fn grants_physical_authority(&self) -> bool { false }
    pub const fn universal_independence_established(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentIndependenceQualification {
    qualification_digest: String,
    active_independence_qualification_digest: String,
    left_profile_id: String,
    right_profile_id: String,
    left_verifier_ref: String,
    right_verifier_ref: String,
    left_record_digest: String,
    right_record_digest: String,
    left_lifecycle_digest: String,
    right_lifecycle_digest: String,
    graph_digest: String,
    relation_completeness_digest: String,
    separation_policy_digest: String,
    left_head_authority_policy_digest: String,
    right_head_authority_policy_digest: String,
    left_head_statement_digest: String,
    right_head_statement_digest: String,
    left_currentness_attestation_digest: String,
    right_currentness_attestation_digest: String,
    left_head_sequence: u64,
    right_head_sequence: u64,
    composition_challenge_nonce_blake3_hex: String,
    use_at_ms: u64,
}

impl CurrentIndependenceQualification {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn active_independence_qualification_digest(&self) -> &str { &self.active_independence_qualification_digest }
    pub fn left_profile_id(&self) -> &str { &self.left_profile_id }
    pub fn right_profile_id(&self) -> &str { &self.right_profile_id }
    pub fn left_verifier_ref(&self) -> &str { &self.left_verifier_ref }
    pub fn right_verifier_ref(&self) -> &str { &self.right_verifier_ref }
    pub fn left_record_digest(&self) -> &str { &self.left_record_digest }
    pub fn right_record_digest(&self) -> &str { &self.right_record_digest }
    pub fn left_lifecycle_digest(&self) -> &str { &self.left_lifecycle_digest }
    pub fn right_lifecycle_digest(&self) -> &str { &self.right_lifecycle_digest }
    pub fn graph_digest(&self) -> &str { &self.graph_digest }
    pub fn relation_completeness_digest(&self) -> &str { &self.relation_completeness_digest }
    pub fn separation_policy_digest(&self) -> &str { &self.separation_policy_digest }
    pub fn left_head_authority_policy_digest(&self) -> &str { &self.left_head_authority_policy_digest }
    pub fn right_head_authority_policy_digest(&self) -> &str { &self.right_head_authority_policy_digest }
    pub fn left_head_statement_digest(&self) -> &str { &self.left_head_statement_digest }
    pub fn right_head_statement_digest(&self) -> &str { &self.right_head_statement_digest }
    pub fn left_currentness_attestation_digest(&self) -> &str { &self.left_currentness_attestation_digest }
    pub fn right_currentness_attestation_digest(&self) -> &str { &self.right_currentness_attestation_digest }
    pub const fn left_head_sequence(&self) -> u64 { self.left_head_sequence }
    pub const fn right_head_sequence(&self) -> u64 { self.right_head_sequence }
    pub fn composition_challenge_nonce_blake3_hex(&self) -> &str { &self.composition_challenge_nonce_blake3_hex }
    pub const fn use_at_ms(&self) -> u64 { self.use_at_ms }
    pub const fn grants_physical_authority(&self) -> bool { false }
    pub const fn universal_independence_established(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentIndependenceAssessment {
    pub report: CurrentIndependenceReport,
    current: Option<CurrentIndependenceQualification>,
}

impl CurrentIndependenceAssessment {
    pub fn current(&self) -> Option<&CurrentIndependenceQualification> { self.current.as_ref() }
    pub fn into_current(self) -> Option<CurrentIndependenceQualification> { self.current }
}

#[allow(clippy::too_many_arguments)]
pub fn compose_current_independence(
    active_independence: &ActiveIndependenceQualification,
    left_current: &CurrentActiveVerifierProfile,
    right_current: &CurrentActiveVerifierProfile,
    left_currentness: &LifecycleHeadCurrentnessAttestation,
    right_currentness: &LifecycleHeadCurrentnessAttestation,
    left_authority_policy: &LifecycleHeadAuthorityPolicy,
    right_authority_policy: &LifecycleHeadAuthorityPolicy,
    expected_composition_challenge_nonce_blake3_hex: &str,
    use_at_ms: u64,
) -> CurrentIndependenceAssessment {
    let left_policy_digest = left_authority_policy.canonical_digest();
    let right_policy_digest = right_authority_policy.canonical_digest();
    let left_currentness_digest = left_currentness.canonical_digest();
    let right_currentness_digest = right_currentness.canonical_digest();
    let mut issues = Vec::new();

    if !left_authority_policy.validate() { issues.push(CurrentIndependenceIssue::InvalidLeftAuthorityPolicy); }
    if !right_authority_policy.validate() { issues.push(CurrentIndependenceIssue::InvalidRightAuthorityPolicy); }
    if !left_currentness.validate() { issues.push(CurrentIndependenceIssue::InvalidLeftCurrentnessAttestation); }
    if !right_currentness.validate() { issues.push(CurrentIndependenceIssue::InvalidRightCurrentnessAttestation); }

    if left_currentness_digest.as_deref() != Some(left_current.currentness_attestation_digest()) {
        issues.push(CurrentIndependenceIssue::LeftCurrentnessDigestMismatch);
    }
    if right_currentness_digest.as_deref() != Some(right_current.currentness_attestation_digest()) {
        issues.push(CurrentIndependenceIssue::RightCurrentnessDigestMismatch);
    }
    if left_policy_digest.as_deref() != Some(left_currentness.authority_policy_digest.as_str()) {
        issues.push(CurrentIndependenceIssue::LeftAuthorityPolicyDigestMismatch);
    }
    if right_policy_digest.as_deref() != Some(right_currentness.authority_policy_digest.as_str()) {
        issues.push(CurrentIndependenceIssue::RightAuthorityPolicyDigestMismatch);
    }
    if left_current.lifecycle_head_statement_digest() != left_currentness.head_statement_digest {
        issues.push(CurrentIndependenceIssue::LeftHeadStatementDigestMismatch);
    }
    if right_current.lifecycle_head_statement_digest() != right_currentness.head_statement_digest {
        issues.push(CurrentIndependenceIssue::RightHeadStatementDigestMismatch);
    }
    if left_current.head_sequence() != left_currentness.head_sequence {
        issues.push(CurrentIndependenceIssue::LeftHeadSequenceMismatch);
    }
    if right_current.head_sequence() != right_currentness.head_sequence {
        issues.push(CurrentIndependenceIssue::RightHeadSequenceMismatch);
    }

    if active_independence.left_profile_id() != left_current.profile_id() {
        issues.push(CurrentIndependenceIssue::LeftProfileIdMismatch);
    }
    if active_independence.right_profile_id() != right_current.profile_id() {
        issues.push(CurrentIndependenceIssue::RightProfileIdMismatch);
    }
    if active_independence.left_record_digest() != left_current.record_digest() {
        issues.push(CurrentIndependenceIssue::LeftRecordDigestMismatch);
    }
    if active_independence.right_record_digest() != right_current.record_digest() {
        issues.push(CurrentIndependenceIssue::RightRecordDigestMismatch);
    }
    if active_independence.graph_digest() != left_current.graph_digest() {
        issues.push(CurrentIndependenceIssue::LeftGraphDigestMismatch);
    }
    if active_independence.graph_digest() != right_current.graph_digest() {
        issues.push(CurrentIndependenceIssue::RightGraphDigestMismatch);
    }
    if active_independence.relation_completeness_digest() != left_current.relation_completeness_digest() {
        issues.push(CurrentIndependenceIssue::LeftCompletenessDigestMismatch);
    }
    if active_independence.relation_completeness_digest() != right_current.relation_completeness_digest() {
        issues.push(CurrentIndependenceIssue::RightCompletenessDigestMismatch);
    }
    if active_independence.left_provenance_attestation_digest() != left_current.provenance_attestation_digest() {
        issues.push(CurrentIndependenceIssue::LeftProvenanceDigestMismatch);
    }
    if active_independence.right_provenance_attestation_digest() != right_current.provenance_attestation_digest() {
        issues.push(CurrentIndependenceIssue::RightProvenanceDigestMismatch);
    }
    if active_independence.left_lifecycle_digest() != left_current.lifecycle_digest() {
        issues.push(CurrentIndependenceIssue::LeftLifecycleDigestMismatch);
    }
    if active_independence.right_lifecycle_digest() != right_current.lifecycle_digest() {
        issues.push(CurrentIndependenceIssue::RightLifecycleDigestMismatch);
    }
    if active_independence.left_verification_at_ms() != left_current.verification_at_ms() {
        issues.push(CurrentIndependenceIssue::LeftVerificationTimeMismatch);
    }
    if active_independence.right_verification_at_ms() != right_current.verification_at_ms() {
        issues.push(CurrentIndependenceIssue::RightVerificationTimeMismatch);
    }

    if left_current.use_at_ms() != use_at_ms {
        issues.push(CurrentIndependenceIssue::LeftUseTimeMismatch { observed: left_current.use_at_ms(), required: use_at_ms });
    }
    if right_current.use_at_ms() != use_at_ms {
        issues.push(CurrentIndependenceIssue::RightUseTimeMismatch { observed: right_current.use_at_ms(), required: use_at_ms });
    }
    if active_independence.use_at_ms() != use_at_ms {
        // Both sides are role-bound to this same #2781 qualification use-time.
        issues.push(CurrentIndependenceIssue::LeftUseTimeMismatch { observed: active_independence.use_at_ms(), required: use_at_ms });
        issues.push(CurrentIndependenceIssue::RightUseTimeMismatch { observed: active_independence.use_at_ms(), required: use_at_ms });
    }
    if left_currentness.asserted_current_at_ms != use_at_ms {
        issues.push(CurrentIndependenceIssue::LeftCurrentnessUseTimeMismatch { observed: left_currentness.asserted_current_at_ms, required: use_at_ms });
    }
    if right_currentness.asserted_current_at_ms != use_at_ms {
        issues.push(CurrentIndependenceIssue::RightCurrentnessUseTimeMismatch { observed: right_currentness.asserted_current_at_ms, required: use_at_ms });
    }
    if left_currentness.challenge_nonce_blake3_hex != expected_composition_challenge_nonce_blake3_hex {
        issues.push(CurrentIndependenceIssue::LeftChallengeMismatch);
    }
    if right_currentness.challenge_nonce_blake3_hex != expected_composition_challenge_nonce_blake3_hex {
        issues.push(CurrentIndependenceIssue::RightChallengeMismatch);
    }

    let invalid = issues.iter().any(|issue| !issue.is_blocking());
    let blocked = issues.iter().any(CurrentIndependenceIssue::is_blocking);
    let disposition = if invalid {
        CurrentIndependenceDisposition::Invalid
    } else if blocked {
        CurrentIndependenceDisposition::Blocked
    } else {
        CurrentIndependenceDisposition::Qualified
    };

    let current = if disposition == CurrentIndependenceDisposition::Qualified {
        let left_policy_digest = left_policy_digest.clone().expect("validated left policy");
        let right_policy_digest = right_policy_digest.clone().expect("validated right policy");
        let left_currentness_digest = left_currentness_digest.clone().expect("validated left currentness");
        let right_currentness_digest = right_currentness_digest.clone().expect("validated right currentness");
        let qualification_digest = qualification_digest(
            active_independence,
            left_current,
            right_current,
            &left_policy_digest,
            &right_policy_digest,
            &left_currentness_digest,
            &right_currentness_digest,
            expected_composition_challenge_nonce_blake3_hex,
            use_at_ms,
        );
        Some(CurrentIndependenceQualification {
            qualification_digest,
            active_independence_qualification_digest: active_independence.qualification_digest().to_string(),
            left_profile_id: active_independence.left_profile_id().to_string(),
            right_profile_id: active_independence.right_profile_id().to_string(),
            left_verifier_ref: active_independence.left_verifier_ref().to_string(),
            right_verifier_ref: active_independence.right_verifier_ref().to_string(),
            left_record_digest: active_independence.left_record_digest().to_string(),
            right_record_digest: active_independence.right_record_digest().to_string(),
            left_lifecycle_digest: active_independence.left_lifecycle_digest().to_string(),
            right_lifecycle_digest: active_independence.right_lifecycle_digest().to_string(),
            graph_digest: active_independence.graph_digest().to_string(),
            relation_completeness_digest: active_independence.relation_completeness_digest().to_string(),
            separation_policy_digest: active_independence.policy_digest().to_string(),
            left_head_authority_policy_digest: left_policy_digest,
            right_head_authority_policy_digest: right_policy_digest,
            left_head_statement_digest: left_current.lifecycle_head_statement_digest().to_string(),
            right_head_statement_digest: right_current.lifecycle_head_statement_digest().to_string(),
            left_currentness_attestation_digest: left_currentness_digest,
            right_currentness_attestation_digest: right_currentness_digest,
            left_head_sequence: left_current.head_sequence(),
            right_head_sequence: right_current.head_sequence(),
            composition_challenge_nonce_blake3_hex: expected_composition_challenge_nonce_blake3_hex.to_string(),
            use_at_ms,
        })
    } else {
        None
    };

    let qualification_digest = current.as_ref().map(|value| value.qualification_digest.clone());
    CurrentIndependenceAssessment {
        report: CurrentIndependenceReport {
            schema_version: CURRENT_INDEPENDENCE_SCHEMA_V1.into(),
            disposition,
            left_profile_id: active_independence.left_profile_id().to_string(),
            right_profile_id: active_independence.right_profile_id().to_string(),
            left_verifier_ref: active_independence.left_verifier_ref().to_string(),
            right_verifier_ref: active_independence.right_verifier_ref().to_string(),
            active_independence_qualification_digest: active_independence.qualification_digest().to_string(),
            separation_policy_digest: active_independence.policy_digest().to_string(),
            graph_digest: active_independence.graph_digest().to_string(),
            relation_completeness_digest: active_independence.relation_completeness_digest().to_string(),
            left_record_digest: active_independence.left_record_digest().to_string(),
            right_record_digest: active_independence.right_record_digest().to_string(),
            left_lifecycle_digest: active_independence.left_lifecycle_digest().to_string(),
            right_lifecycle_digest: active_independence.right_lifecycle_digest().to_string(),
            left_head_sequence: left_current.head_sequence(),
            right_head_sequence: right_current.head_sequence(),
            left_head_statement_digest: left_current.lifecycle_head_statement_digest().to_string(),
            right_head_statement_digest: right_current.lifecycle_head_statement_digest().to_string(),
            left_currentness_attestation_digest: left_current.currentness_attestation_digest().to_string(),
            right_currentness_attestation_digest: right_current.currentness_attestation_digest().to_string(),
            left_head_authority_policy_digest: left_policy_digest,
            right_head_authority_policy_digest: right_policy_digest,
            composition_challenge_nonce_blake3_hex: expected_composition_challenge_nonce_blake3_hex.to_string(),
            use_at_ms,
            issues,
            qualification_digest,
        },
        current,
    }
}

#[allow(clippy::too_many_arguments)]
fn qualification_digest(
    active_independence: &ActiveIndependenceQualification,
    left_current: &CurrentActiveVerifierProfile,
    right_current: &CurrentActiveVerifierProfile,
    left_policy_digest: &str,
    right_policy_digest: &str,
    left_currentness_digest: &str,
    right_currentness_digest: &str,
    challenge: &str,
    use_at_ms: u64,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    for value in [
        CURRENT_INDEPENDENCE_SCHEMA_V1,
        active_independence.qualification_digest(),
        active_independence.left_profile_id(),
        active_independence.right_profile_id(),
        active_independence.left_verifier_ref(),
        active_independence.right_verifier_ref(),
        active_independence.left_record_digest(),
        active_independence.right_record_digest(),
        active_independence.graph_digest(),
        active_independence.relation_completeness_digest(),
        active_independence.policy_digest(),
        left_current.lifecycle_head_statement_digest(),
        right_current.lifecycle_head_statement_digest(),
        left_currentness_digest,
        right_currentness_digest,
        left_policy_digest,
        right_policy_digest,
        challenge,
    ] {
        push_field(&mut hasher, value);
    }
    push_u64(&mut hasher, left_current.head_sequence());
    push_u64(&mut hasher, right_current.head_sequence());
    push_u64(&mut hasher, use_at_ms);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn push_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_be_bytes());
}
