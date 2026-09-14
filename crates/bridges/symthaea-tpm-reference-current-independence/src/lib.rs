// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! TPM/reference role binding for current-head-qualified verifier independence.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_domain_awareness_evidence::IndependentVerification;
use symthaea_evidence_active_independence::ActiveIndependenceQualification;
use symthaea_evidence_current_independence::CurrentIndependenceQualification;
use symthaea_formal_safety::SafetyEvidenceReceipt;
use symthaea_tpm_reference_evidence::TpmReferenceCandidateEvidence;

pub const TPM_CURRENT_INDEPENDENCE_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.tpm-reference-current-independence-policy.v1";
pub const TPM_CURRENT_INDEPENDENCE_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.tpm-reference-current-independence-report.v1";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm-reference-current-independence-policy.digest.v1\0";
const REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm-reference-current-independence-report.digest.v1\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm-reference-current-independence-qualification.digest.v1\0";
const MAX_TEXT_BYTES: usize = 512;
const MAX_EVIDENCE_REFS: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TpmCurrentIndependencePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub obligation_key: String,
    pub expected_graph_digest: String,
    pub expected_relation_completeness_digest: String,
    pub expected_separation_policy_digest: String,
    pub expected_campaign_head_authority_policy_digest: String,
    pub expected_obligation_head_authority_policy_digest: String,
    pub max_obligation_review_lag_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl TpmCurrentIndependencePolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == TPM_CURRENT_INDEPENDENCE_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && digest_text(&self.obligation_key)
            && digest_text(&self.expected_graph_digest)
            && digest_text(&self.expected_relation_completeness_digest)
            && digest_text(&self.expected_separation_policy_digest)
            && digest_text(&self.expected_campaign_head_authority_policy_digest)
            && digest_text(&self.expected_obligation_head_authority_policy_digest)
            && self.max_obligation_review_lag_ms > 0
            && valid_refs(&self.evidence_refs)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() { return None; }
        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DIGEST_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.obligation_key.as_str(),
            self.expected_graph_digest.as_str(),
            self.expected_relation_completeness_digest.as_str(),
            self.expected_separation_policy_digest.as_str(),
            self.expected_campaign_head_authority_policy_digest.as_str(),
            self.expected_obligation_head_authority_policy_digest.as_str(),
        ] { push_field(&mut hasher, value); }
        push_u64(&mut hasher, self.max_obligation_review_lag_ms);
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TpmCurrentIndependenceDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TpmCurrentIndependenceIssue {
    InvalidPolicy,
    InvalidCandidate,
    InvalidVerification,
    ObligationMismatch,
    CurrentQualificationParentMismatch,
    CampaignVerifierRoleMismatch,
    ObligationVerifierRoleMismatch,
    CampaignVerificationTimeMismatch,
    ObligationVerificationTimeMismatch,
    VerificationPredatesCampaign,
    VerificationTooLate,
    CurrentUsePredatesObligationVerification,
    GraphDigestMismatch,
    RelationCompletenessDigestMismatch,
    SeparationPolicyDigestMismatch,
    CampaignHeadAuthorityPolicyDigestMismatch,
    ObligationHeadAuthorityPolicyDigestMismatch,
}

impl TpmCurrentIndependenceIssue {
    fn code(&self) -> &'static str {
        match self {
            Self::InvalidPolicy => "invalid-policy",
            Self::InvalidCandidate => "invalid-candidate",
            Self::InvalidVerification => "invalid-verification",
            Self::ObligationMismatch => "obligation-mismatch",
            Self::CurrentQualificationParentMismatch => "current-parent-mismatch",
            Self::CampaignVerifierRoleMismatch => "campaign-verifier-role-mismatch",
            Self::ObligationVerifierRoleMismatch => "obligation-verifier-role-mismatch",
            Self::CampaignVerificationTimeMismatch => "campaign-verification-time-mismatch",
            Self::ObligationVerificationTimeMismatch => "obligation-verification-time-mismatch",
            Self::VerificationPredatesCampaign => "verification-predates-campaign",
            Self::VerificationTooLate => "verification-too-late",
            Self::CurrentUsePredatesObligationVerification => "current-use-predates-obligation-verification",
            Self::GraphDigestMismatch => "graph-digest-mismatch",
            Self::RelationCompletenessDigestMismatch => "relation-completeness-digest-mismatch",
            Self::SeparationPolicyDigestMismatch => "separation-policy-digest-mismatch",
            Self::CampaignHeadAuthorityPolicyDigestMismatch => "campaign-head-authority-policy-digest-mismatch",
            Self::ObligationHeadAuthorityPolicyDigestMismatch => "obligation-head-authority-policy-digest-mismatch",
        }
    }

    fn is_invalid(&self) -> bool {
        matches!(
            self,
            Self::InvalidPolicy
                | Self::InvalidCandidate
                | Self::InvalidVerification
                | Self::ObligationMismatch
                | Self::CurrentQualificationParentMismatch
                | Self::CampaignVerifierRoleMismatch
                | Self::ObligationVerifierRoleMismatch
                | Self::CampaignVerificationTimeMismatch
                | Self::ObligationVerificationTimeMismatch
                | Self::VerificationPredatesCampaign
                | Self::CurrentUsePredatesObligationVerification
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TpmCurrentIndependenceReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub candidate_digest: String,
    pub obligation_key: String,
    pub campaign_verifier_ref: String,
    pub obligation_verifier_ref: String,
    pub campaign_verified_at_ms: u64,
    pub obligation_verified_at_ms: u64,
    pub current_use_at_ms: u64,
    pub active_independence_qualification_digest: String,
    pub current_independence_qualification_digest: String,
    pub graph_digest: String,
    pub relation_completeness_digest: String,
    pub separation_policy_digest: String,
    pub campaign_head_authority_policy_digest: String,
    pub obligation_head_authority_policy_digest: String,
    pub disposition: TpmCurrentIndependenceDisposition,
    pub issues: Vec<TpmCurrentIndependenceIssue>,
}

impl TpmCurrentIndependenceReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DIGEST_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.candidate_digest.as_str(),
            self.obligation_key.as_str(),
            self.campaign_verifier_ref.as_str(),
            self.obligation_verifier_ref.as_str(),
            self.active_independence_qualification_digest.as_str(),
            self.current_independence_qualification_digest.as_str(),
            self.graph_digest.as_str(),
            self.relation_completeness_digest.as_str(),
            self.separation_policy_digest.as_str(),
            self.campaign_head_authority_policy_digest.as_str(),
            self.obligation_head_authority_policy_digest.as_str(),
        ] { push_field(&mut hasher, value); }
        push_u64(&mut hasher, self.campaign_verified_at_ms);
        push_u64(&mut hasher, self.obligation_verified_at_ms);
        push_u64(&mut hasher, self.current_use_at_ms);
        push_field(&mut hasher, match self.disposition {
            TpmCurrentIndependenceDisposition::Invalid => "invalid",
            TpmCurrentIndependenceDisposition::Blocked => "blocked",
            TpmCurrentIndependenceDisposition::Qualified => "qualified",
        });
        for issue in &self.issues { push_field(&mut hasher, issue.code()); }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpmCurrentCrossStageQualification {
    qualification_digest: String,
    report_digest: String,
    current_independence_qualification_digest: String,
    campaign_verifier_ref: String,
    obligation_verifier_ref: String,
    campaign_verified_at_ms: u64,
    obligation_verified_at_ms: u64,
    current_use_at_ms: u64,
}

impl TpmCurrentCrossStageQualification {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn current_independence_qualification_digest(&self) -> &str { &self.current_independence_qualification_digest }
    pub fn campaign_verifier_ref(&self) -> &str { &self.campaign_verifier_ref }
    pub fn obligation_verifier_ref(&self) -> &str { &self.obligation_verifier_ref }
    pub const fn campaign_verified_at_ms(&self) -> u64 { self.campaign_verified_at_ms }
    pub const fn obligation_verified_at_ms(&self) -> u64 { self.obligation_verified_at_ms }
    pub const fn current_use_at_ms(&self) -> u64 { self.current_use_at_ms }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpmCurrentQualifiedEvidence {
    pub report: TpmCurrentIndependenceReport,
    pub receipt: SafetyEvidenceReceipt,
    qualification: TpmCurrentCrossStageQualification,
}

impl TpmCurrentQualifiedEvidence {
    pub fn qualification(&self) -> &TpmCurrentCrossStageQualification { &self.qualification }
    pub fn into_qualification(self) -> TpmCurrentCrossStageQualification { self.qualification }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TpmCurrentIndependenceError {
    NotQualified(TpmCurrentIndependenceReport),
    BaseVerificationFailed,
}

pub fn assess_tpm_current_independence(
    candidate: &TpmReferenceCandidateEvidence,
    verification: &IndependentVerification,
    active_independence: &ActiveIndependenceQualification,
    current_independence: &CurrentIndependenceQualification,
    policy: &TpmCurrentIndependencePolicy,
) -> TpmCurrentIndependenceReport {
    let policy_digest = policy.canonical_digest();
    let mut issues = Vec::new();
    if !policy.validate() { issues.push(TpmCurrentIndependenceIssue::InvalidPolicy); }
    if !candidate.validate() { issues.push(TpmCurrentIndependenceIssue::InvalidCandidate); }
    if !verification.validate() { issues.push(TpmCurrentIndependenceIssue::InvalidVerification); }
    if policy.obligation_key != candidate.obligation_key() {
        issues.push(TpmCurrentIndependenceIssue::ObligationMismatch);
    }
    if current_independence.active_independence_qualification_digest() != active_independence.qualification_digest() {
        issues.push(TpmCurrentIndependenceIssue::CurrentQualificationParentMismatch);
    }
    if current_independence.left_verifier_ref() != candidate.campaign_verifier_ref {
        issues.push(TpmCurrentIndependenceIssue::CampaignVerifierRoleMismatch);
    }
    if current_independence.right_verifier_ref() != verification.verifier_ref {
        issues.push(TpmCurrentIndependenceIssue::ObligationVerifierRoleMismatch);
    }
    if active_independence.left_verification_at_ms() != candidate.campaign_verified_at_ms {
        issues.push(TpmCurrentIndependenceIssue::CampaignVerificationTimeMismatch);
    }
    if active_independence.right_verification_at_ms() != verification.verified_at_ms {
        issues.push(TpmCurrentIndependenceIssue::ObligationVerificationTimeMismatch);
    }
    if verification.verified_at_ms < candidate.campaign_verified_at_ms {
        issues.push(TpmCurrentIndependenceIssue::VerificationPredatesCampaign);
    } else if verification.verified_at_ms - candidate.campaign_verified_at_ms > policy.max_obligation_review_lag_ms {
        issues.push(TpmCurrentIndependenceIssue::VerificationTooLate);
    }
    if current_independence.use_at_ms() < verification.verified_at_ms {
        issues.push(TpmCurrentIndependenceIssue::CurrentUsePredatesObligationVerification);
    }
    if current_independence.graph_digest() != policy.expected_graph_digest {
        issues.push(TpmCurrentIndependenceIssue::GraphDigestMismatch);
    }
    if current_independence.relation_completeness_digest() != policy.expected_relation_completeness_digest {
        issues.push(TpmCurrentIndependenceIssue::RelationCompletenessDigestMismatch);
    }
    if current_independence.separation_policy_digest() != policy.expected_separation_policy_digest {
        issues.push(TpmCurrentIndependenceIssue::SeparationPolicyDigestMismatch);
    }
    if current_independence.left_head_authority_policy_digest() != policy.expected_campaign_head_authority_policy_digest {
        issues.push(TpmCurrentIndependenceIssue::CampaignHeadAuthorityPolicyDigestMismatch);
    }
    if current_independence.right_head_authority_policy_digest() != policy.expected_obligation_head_authority_policy_digest {
        issues.push(TpmCurrentIndependenceIssue::ObligationHeadAuthorityPolicyDigestMismatch);
    }

    let disposition = if issues.iter().any(TpmCurrentIndependenceIssue::is_invalid) {
        TpmCurrentIndependenceDisposition::Invalid
    } else if issues.is_empty() {
        TpmCurrentIndependenceDisposition::Qualified
    } else {
        TpmCurrentIndependenceDisposition::Blocked
    };

    TpmCurrentIndependenceReport {
        schema_version: TPM_CURRENT_INDEPENDENCE_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest,
        candidate_digest: candidate.evidence_digest.clone(),
        obligation_key: candidate.obligation_key(),
        campaign_verifier_ref: candidate.campaign_verifier_ref.clone(),
        obligation_verifier_ref: verification.verifier_ref.clone(),
        campaign_verified_at_ms: candidate.campaign_verified_at_ms,
        obligation_verified_at_ms: verification.verified_at_ms,
        current_use_at_ms: current_independence.use_at_ms(),
        active_independence_qualification_digest: active_independence.qualification_digest().to_string(),
        current_independence_qualification_digest: current_independence.qualification_digest().to_string(),
        graph_digest: current_independence.graph_digest().to_string(),
        relation_completeness_digest: current_independence.relation_completeness_digest().to_string(),
        separation_policy_digest: current_independence.separation_policy_digest().to_string(),
        campaign_head_authority_policy_digest: current_independence.left_head_authority_policy_digest().to_string(),
        obligation_head_authority_policy_digest: current_independence.right_head_authority_policy_digest().to_string(),
        disposition,
        issues,
    }
}

pub fn verify_with_tpm_current_independence(
    candidate: &TpmReferenceCandidateEvidence,
    verification: &IndependentVerification,
    active_independence: &ActiveIndependenceQualification,
    current_independence: &CurrentIndependenceQualification,
    policy: &TpmCurrentIndependencePolicy,
) -> Result<TpmCurrentQualifiedEvidence, TpmCurrentIndependenceError> {
    let report = assess_tpm_current_independence(
        candidate,
        verification,
        active_independence,
        current_independence,
        policy,
    );
    if report.disposition != TpmCurrentIndependenceDisposition::Qualified {
        return Err(TpmCurrentIndependenceError::NotQualified(report));
    }
    let mut receipt = candidate
        .verify(verification)
        .map_err(|_| TpmCurrentIndependenceError::BaseVerificationFailed)?;
    let report_digest = report.canonical_digest();
    receipt.evidence_digest = report_digest.clone();
    let qualification_digest = qualification_digest(&report, &report_digest);
    let qualification = TpmCurrentCrossStageQualification {
        qualification_digest,
        report_digest,
        current_independence_qualification_digest: current_independence.qualification_digest().to_string(),
        campaign_verifier_ref: candidate.campaign_verifier_ref.clone(),
        obligation_verifier_ref: verification.verifier_ref.clone(),
        campaign_verified_at_ms: candidate.campaign_verified_at_ms,
        obligation_verified_at_ms: verification.verified_at_ms,
        current_use_at_ms: current_independence.use_at_ms(),
    };
    Ok(TpmCurrentQualifiedEvidence { report, receipt, qualification })
}

fn qualification_digest(report: &TpmCurrentIndependenceReport, report_digest: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    push_field(&mut hasher, report_digest);
    push_field(&mut hasher, &report.current_independence_qualification_digest);
    push_field(&mut hasher, &report.campaign_verifier_ref);
    push_field(&mut hasher, &report.obligation_verifier_ref);
    push_u64(&mut hasher, report.campaign_verified_at_ms);
    push_u64(&mut hasher, report.obligation_verified_at_ms);
    push_u64(&mut hasher, report.current_use_at_ms);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty() && value.len() <= MAX_TEXT_BYTES && value.trim() == value
}

fn digest_text(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|digest| {
        digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn valid_refs(values: &[String]) -> bool {
    !values.is_empty()
        && values.len() <= MAX_EVIDENCE_REFS
        && values.iter().all(|value| canonical_text(value))
        && {
            let mut sorted = values.to_vec();
            sorted.sort();
            sorted.dedup();
            sorted.len() == values.len()
        }
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn push_u64(hasher: &mut blake3::Hasher, value: u64) { hasher.update(&value.to_be_bytes()); }

fn push_sorted_refs(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut refs = refs.to_vec();
    refs.sort();
    for reference in refs { push_field(hasher, &reference); }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(label: &str) -> String { format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex()) }

    fn policy() -> TpmCurrentIndependencePolicy {
        TpmCurrentIndependencePolicy {
            schema_version: TPM_CURRENT_INDEPENDENCE_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:tpm-current".into(),
            obligation_key: d("obligation"),
            expected_graph_digest: d("graph"),
            expected_relation_completeness_digest: d("completeness"),
            expected_separation_policy_digest: d("separation-policy"),
            expected_campaign_head_authority_policy_digest: d("campaign-head-policy"),
            expected_obligation_head_authority_policy_digest: d("obligation-head-policy"),
            max_obligation_review_lag_ms: 1_000,
            evidence_refs: vec!["review:tpm-current-policy".into()],
        }
    }

    #[test]
    fn policy_digest_changes_when_any_trust_root_changes() {
        let left = policy();
        let mut right = left.clone();
        right.expected_obligation_head_authority_policy_digest = d("different-obligation-head-policy");
        assert_ne!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn zero_review_lag_is_not_a_valid_policy() {
        let mut policy = policy();
        policy.max_obligation_review_lag_ms = 0;
        assert!(!policy.validate());
    }

    #[test]
    fn duplicate_policy_evidence_refs_are_rejected() {
        let mut policy = policy();
        policy.evidence_refs = vec!["review:x".into(), "review:x".into()];
        assert!(!policy.validate());
    }
}
