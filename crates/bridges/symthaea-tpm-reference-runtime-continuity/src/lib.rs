// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! TPM/reference binding for runtime-backed current verifier independence.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_domain_awareness_evidence::IndependentVerification;
use symthaea_evidence_verifier_runtime_continuity::ContinuousVerifierExecution;
use symthaea_formal_safety::{EvidenceKind, SafetyEvidenceReceipt};
use symthaea_tpm_reference_current_independence::TpmCurrentCrossStageQualification;
use symthaea_tpm_reference_evidence::TpmReferenceCandidateEvidence;

pub const TPM_RUNTIME_CONTINUITY_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.tpm-reference-runtime-continuity-policy.v1";
pub const TPM_RUNTIME_CONTINUITY_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.tpm-reference-runtime-continuity-report.v1";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm-reference-runtime-continuity-policy.digest.v1\0";
const OBLIGATION_INPUT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm-reference-obligation-runtime-input.digest.v1\0";
const STRICT_RECEIPT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.strict-safety-evidence-receipt.digest.v1\0";
const REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm-reference-runtime-continuity-report.digest.v1\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm-reference-runtime-backed-qualification.digest.v1\0";
const MAX_TEXT_BYTES: usize = 512;
const MAX_EVIDENCE_REFS: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TpmRuntimeContinuityPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_campaign_runtime_policy_digest: String,
    pub expected_obligation_runtime_policy_digest: String,
    /// Exact reviewed input subject for the campaign verifier computation.
    pub expected_campaign_runtime_input_digest: String,
    pub evidence_refs: Vec<String>,
}

impl TpmRuntimeContinuityPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == TPM_RUNTIME_CONTINUITY_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && digest_text(&self.expected_campaign_runtime_policy_digest)
            && digest_text(&self.expected_obligation_runtime_policy_digest)
            && digest_text(&self.expected_campaign_runtime_input_digest)
            && valid_refs(&self.evidence_refs)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DIGEST_DOMAIN);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.policy_id);
        push_field(
            &mut hasher,
            &self.expected_campaign_runtime_policy_digest,
        );
        push_field(
            &mut hasher,
            &self.expected_obligation_runtime_policy_digest,
        );
        push_field(
            &mut hasher,
            &self.expected_campaign_runtime_input_digest,
        );
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TpmRuntimeContinuityDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TpmRuntimeContinuityIssue {
    InvalidPolicy,
    InvalidCandidate,
    InvalidVerification,
    BaseReceiptFailed,
    ParentCampaignRoleMismatch,
    ParentObligationRoleMismatch,
    ParentCampaignTimeMismatch,
    ParentObligationTimeMismatch,
    CampaignRuntimeVerifierMismatch,
    ObligationRuntimeVerifierMismatch,
    CampaignRuntimeCompletionMismatch,
    ObligationRuntimeCompletionMismatch,
    CampaignRuntimeAssessedAfterCurrentUse,
    ObligationRuntimeAssessedAfterCurrentUse,
    CampaignRuntimePolicyMismatch,
    ObligationRuntimePolicyMismatch,
    CampaignRuntimeInputMismatch,
    CampaignRuntimeOutputMismatch,
    ObligationRuntimeInputMismatch,
    ObligationRuntimeOutputMismatch,
    SameRuntimeProcessInstance,
}

impl TpmRuntimeContinuityIssue {
    fn code(&self) -> &'static str {
        match self {
            Self::InvalidPolicy => "invalid-policy",
            Self::InvalidCandidate => "invalid-candidate",
            Self::InvalidVerification => "invalid-verification",
            Self::BaseReceiptFailed => "base-receipt-failed",
            Self::ParentCampaignRoleMismatch => "parent-campaign-role-mismatch",
            Self::ParentObligationRoleMismatch => "parent-obligation-role-mismatch",
            Self::ParentCampaignTimeMismatch => "parent-campaign-time-mismatch",
            Self::ParentObligationTimeMismatch => "parent-obligation-time-mismatch",
            Self::CampaignRuntimeVerifierMismatch => "campaign-runtime-verifier-mismatch",
            Self::ObligationRuntimeVerifierMismatch => "obligation-runtime-verifier-mismatch",
            Self::CampaignRuntimeCompletionMismatch => "campaign-runtime-completion-mismatch",
            Self::ObligationRuntimeCompletionMismatch => "obligation-runtime-completion-mismatch",
            Self::CampaignRuntimeAssessedAfterCurrentUse => "campaign-runtime-assessed-after-current-use",
            Self::ObligationRuntimeAssessedAfterCurrentUse => "obligation-runtime-assessed-after-current-use",
            Self::CampaignRuntimePolicyMismatch => "campaign-runtime-policy-mismatch",
            Self::ObligationRuntimePolicyMismatch => "obligation-runtime-policy-mismatch",
            Self::CampaignRuntimeInputMismatch => "campaign-runtime-input-mismatch",
            Self::CampaignRuntimeOutputMismatch => "campaign-runtime-output-mismatch",
            Self::ObligationRuntimeInputMismatch => "obligation-runtime-input-mismatch",
            Self::ObligationRuntimeOutputMismatch => "obligation-runtime-output-mismatch",
            Self::SameRuntimeProcessInstance => "same-runtime-process-instance",
        }
    }

    fn is_invalid(&self) -> bool {
        matches!(
            self,
            Self::InvalidPolicy
                | Self::InvalidCandidate
                | Self::InvalidVerification
                | Self::BaseReceiptFailed
                | Self::ParentCampaignRoleMismatch
                | Self::ParentObligationRoleMismatch
                | Self::ParentCampaignTimeMismatch
                | Self::ParentObligationTimeMismatch
                | Self::CampaignRuntimeVerifierMismatch
                | Self::ObligationRuntimeVerifierMismatch
                | Self::CampaignRuntimeCompletionMismatch
                | Self::ObligationRuntimeCompletionMismatch
                | Self::CampaignRuntimeAssessedAfterCurrentUse
                | Self::ObligationRuntimeAssessedAfterCurrentUse
                | Self::CampaignRuntimeOutputMismatch
                | Self::ObligationRuntimeInputMismatch
                | Self::ObligationRuntimeOutputMismatch
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TpmRuntimeContinuityReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub parent_tpm_qualification_digest: String,
    pub campaign_verifier_ref: String,
    pub obligation_verifier_ref: String,
    pub campaign_verified_at_ms: u64,
    pub obligation_verified_at_ms: u64,
    pub current_use_at_ms: u64,
    pub campaign_runtime_continuity_digest: String,
    pub obligation_runtime_continuity_digest: String,
    pub campaign_runtime_policy_digest: String,
    pub obligation_runtime_policy_digest: String,
    pub campaign_process_instance_id: String,
    pub obligation_process_instance_id: String,
    pub expected_campaign_input_digest: String,
    pub campaign_output_digest: String,
    pub expected_obligation_input_digest: Option<String>,
    pub obligation_output_digest: String,
    pub strict_receipt_digest: Option<String>,
    pub disposition: TpmRuntimeContinuityDisposition,
    pub issues: Vec<TpmRuntimeContinuityIssue>,
}

impl TpmRuntimeContinuityReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DIGEST_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.parent_tpm_qualification_digest.as_str(),
            self.campaign_verifier_ref.as_str(),
            self.obligation_verifier_ref.as_str(),
            self.campaign_runtime_continuity_digest.as_str(),
            self.obligation_runtime_continuity_digest.as_str(),
            self.campaign_runtime_policy_digest.as_str(),
            self.obligation_runtime_policy_digest.as_str(),
            self.campaign_process_instance_id.as_str(),
            self.obligation_process_instance_id.as_str(),
            self.expected_campaign_input_digest.as_str(),
            self.campaign_output_digest.as_str(),
            self.expected_obligation_input_digest.as_deref().unwrap_or("-"),
            self.obligation_output_digest.as_str(),
            self.strict_receipt_digest.as_deref().unwrap_or("-"),
        ] {
            push_field(&mut hasher, value);
        }
        push_u64(&mut hasher, self.campaign_verified_at_ms);
        push_u64(&mut hasher, self.obligation_verified_at_ms);
        push_u64(&mut hasher, self.current_use_at_ms);
        push_field(
            &mut hasher,
            match self.disposition {
                TpmRuntimeContinuityDisposition::Invalid => "invalid",
                TpmRuntimeContinuityDisposition::Blocked => "blocked",
                TpmRuntimeContinuityDisposition::Qualified => "qualified",
            },
        );
        for issue in &self.issues {
            push_field(&mut hasher, issue.code());
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeBackedTpmQualification {
    qualification_digest: String,
    report_digest: String,
    parent_tpm_qualification_digest: String,
    campaign_runtime_continuity_digest: String,
    obligation_runtime_continuity_digest: String,
    strict_receipt_digest: String,
    campaign_verifier_ref: String,
    obligation_verifier_ref: String,
    campaign_verified_at_ms: u64,
    obligation_verified_at_ms: u64,
    current_use_at_ms: u64,
}

impl RuntimeBackedTpmQualification {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn parent_tpm_qualification_digest(&self) -> &str { &self.parent_tpm_qualification_digest }
    pub fn campaign_runtime_continuity_digest(&self) -> &str { &self.campaign_runtime_continuity_digest }
    pub fn obligation_runtime_continuity_digest(&self) -> &str { &self.obligation_runtime_continuity_digest }
    pub fn strict_receipt_digest(&self) -> &str { &self.strict_receipt_digest }
    pub fn campaign_verifier_ref(&self) -> &str { &self.campaign_verifier_ref }
    pub fn obligation_verifier_ref(&self) -> &str { &self.obligation_verifier_ref }
    pub const fn campaign_verified_at_ms(&self) -> u64 { self.campaign_verified_at_ms }
    pub const fn obligation_verified_at_ms(&self) -> u64 { self.obligation_verified_at_ms }
    pub const fn current_use_at_ms(&self) -> u64 { self.current_use_at_ms }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpmRuntimeQualifiedEvidence {
    pub report: TpmRuntimeContinuityReport,
    pub receipt: SafetyEvidenceReceipt,
    qualification: RuntimeBackedTpmQualification,
}

impl TpmRuntimeQualifiedEvidence {
    pub fn qualification(&self) -> &RuntimeBackedTpmQualification { &self.qualification }
    pub fn into_qualification(self) -> RuntimeBackedTpmQualification { self.qualification }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TpmRuntimeContinuityError {
    NotQualified(TpmRuntimeContinuityReport),
    BaseReceiptFailed,
}

pub fn strict_receipt_digest(receipt: &SafetyEvidenceReceipt) -> Option<String> {
    if !receipt.validate() {
        return None;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(STRICT_RECEIPT_DIGEST_DOMAIN);
    push_field(&mut hasher, &receipt.receipt_id);
    push_field(&mut hasher, &receipt.obligation_key);
    push_field(&mut hasher, evidence_kind_code(receipt.evidence_kind));
    push_field(&mut hasher, &receipt.evidence_ref);
    push_field(&mut hasher, &receipt.evidence_digest);
    push_field(&mut hasher, &receipt.verifier_ref);
    push_u64(&mut hasher, receipt.verified_at_ms);
    Some(format!("blake3:{}", hasher.finalize().to_hex()))
}

pub fn obligation_runtime_input_digest(
    candidate: &TpmReferenceCandidateEvidence,
) -> Option<String> {
    if !candidate.validate() {
        return None;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(OBLIGATION_INPUT_DIGEST_DOMAIN);
    push_field(&mut hasher, &candidate.evidence_digest);
    push_field(&mut hasher, &candidate.obligation_key());
    push_field(&mut hasher, &candidate.campaign_verifier_ref);
    push_u64(&mut hasher, candidate.campaign_verified_at_ms);
    Some(format!("blake3:{}", hasher.finalize().to_hex()))
}

#[allow(clippy::too_many_arguments)]
pub fn assess_tpm_runtime_continuity(
    candidate: &TpmReferenceCandidateEvidence,
    verification: &IndependentVerification,
    parent: &TpmCurrentCrossStageQualification,
    campaign_runtime: &ContinuousVerifierExecution,
    obligation_runtime: &ContinuousVerifierExecution,
    policy: &TpmRuntimeContinuityPolicy,
) -> TpmRuntimeContinuityReport {
    let policy_digest = policy.canonical_digest();
    let expected_obligation_input_digest = obligation_runtime_input_digest(candidate);
    let base_receipt = candidate.verify(verification).ok();
    let strict_receipt_digest = base_receipt.as_ref().and_then(strict_receipt_digest);
    let mut issues = Vec::new();

    if !policy.validate() {
        issues.push(TpmRuntimeContinuityIssue::InvalidPolicy);
    }
    if !candidate.validate() {
        issues.push(TpmRuntimeContinuityIssue::InvalidCandidate);
    }
    if !verification.validate() {
        issues.push(TpmRuntimeContinuityIssue::InvalidVerification);
    }
    if base_receipt.is_none() || strict_receipt_digest.is_none() {
        issues.push(TpmRuntimeContinuityIssue::BaseReceiptFailed);
    }

    if parent.campaign_verifier_ref() != candidate.campaign_verifier_ref {
        issues.push(TpmRuntimeContinuityIssue::ParentCampaignRoleMismatch);
    }
    if parent.obligation_verifier_ref() != verification.verifier_ref {
        issues.push(TpmRuntimeContinuityIssue::ParentObligationRoleMismatch);
    }
    if parent.campaign_verified_at_ms() != candidate.campaign_verified_at_ms {
        issues.push(TpmRuntimeContinuityIssue::ParentCampaignTimeMismatch);
    }
    if parent.obligation_verified_at_ms() != verification.verified_at_ms {
        issues.push(TpmRuntimeContinuityIssue::ParentObligationTimeMismatch);
    }

    if campaign_runtime.verifier_ref() != candidate.campaign_verifier_ref {
        issues.push(TpmRuntimeContinuityIssue::CampaignRuntimeVerifierMismatch);
    }
    if obligation_runtime.verifier_ref() != verification.verifier_ref {
        issues.push(TpmRuntimeContinuityIssue::ObligationRuntimeVerifierMismatch);
    }
    if campaign_runtime.completed_at_ms() != candidate.campaign_verified_at_ms {
        issues.push(TpmRuntimeContinuityIssue::CampaignRuntimeCompletionMismatch);
    }
    if obligation_runtime.completed_at_ms() != verification.verified_at_ms {
        issues.push(TpmRuntimeContinuityIssue::ObligationRuntimeCompletionMismatch);
    }
    if campaign_runtime.assessed_at_ms() > parent.current_use_at_ms() {
        issues.push(TpmRuntimeContinuityIssue::CampaignRuntimeAssessedAfterCurrentUse);
    }
    if obligation_runtime.assessed_at_ms() > parent.current_use_at_ms() {
        issues.push(TpmRuntimeContinuityIssue::ObligationRuntimeAssessedAfterCurrentUse);
    }

    if campaign_runtime.policy_digest() != policy.expected_campaign_runtime_policy_digest {
        issues.push(TpmRuntimeContinuityIssue::CampaignRuntimePolicyMismatch);
    }
    if obligation_runtime.policy_digest() != policy.expected_obligation_runtime_policy_digest {
        issues.push(TpmRuntimeContinuityIssue::ObligationRuntimePolicyMismatch);
    }
    if campaign_runtime.input_digest() != policy.expected_campaign_runtime_input_digest {
        issues.push(TpmRuntimeContinuityIssue::CampaignRuntimeInputMismatch);
    }
    if campaign_runtime.output_digest() != candidate.evidence_digest {
        issues.push(TpmRuntimeContinuityIssue::CampaignRuntimeOutputMismatch);
    }
    if expected_obligation_input_digest.as_deref() != Some(obligation_runtime.input_digest()) {
        issues.push(TpmRuntimeContinuityIssue::ObligationRuntimeInputMismatch);
    }
    if strict_receipt_digest.as_deref() != Some(obligation_runtime.output_digest()) {
        issues.push(TpmRuntimeContinuityIssue::ObligationRuntimeOutputMismatch);
    }
    if campaign_runtime.process_instance_id() == obligation_runtime.process_instance_id() {
        issues.push(TpmRuntimeContinuityIssue::SameRuntimeProcessInstance);
    }

    let disposition = if issues.iter().any(TpmRuntimeContinuityIssue::is_invalid) {
        TpmRuntimeContinuityDisposition::Invalid
    } else if issues.is_empty() {
        TpmRuntimeContinuityDisposition::Qualified
    } else {
        TpmRuntimeContinuityDisposition::Blocked
    };

    TpmRuntimeContinuityReport {
        schema_version: TPM_RUNTIME_CONTINUITY_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest,
        parent_tpm_qualification_digest: parent.qualification_digest().to_string(),
        campaign_verifier_ref: candidate.campaign_verifier_ref.clone(),
        obligation_verifier_ref: verification.verifier_ref.clone(),
        campaign_verified_at_ms: candidate.campaign_verified_at_ms,
        obligation_verified_at_ms: verification.verified_at_ms,
        current_use_at_ms: parent.current_use_at_ms(),
        campaign_runtime_continuity_digest: campaign_runtime.continuity_digest().to_string(),
        obligation_runtime_continuity_digest: obligation_runtime.continuity_digest().to_string(),
        campaign_runtime_policy_digest: campaign_runtime.policy_digest().to_string(),
        obligation_runtime_policy_digest: obligation_runtime.policy_digest().to_string(),
        campaign_process_instance_id: campaign_runtime.process_instance_id().to_string(),
        obligation_process_instance_id: obligation_runtime.process_instance_id().to_string(),
        expected_campaign_input_digest: policy.expected_campaign_runtime_input_digest.clone(),
        campaign_output_digest: campaign_runtime.output_digest().to_string(),
        expected_obligation_input_digest,
        obligation_output_digest: obligation_runtime.output_digest().to_string(),
        strict_receipt_digest,
        disposition,
        issues,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn verify_with_tpm_runtime_continuity(
    candidate: &TpmReferenceCandidateEvidence,
    verification: &IndependentVerification,
    parent: &TpmCurrentCrossStageQualification,
    campaign_runtime: &ContinuousVerifierExecution,
    obligation_runtime: &ContinuousVerifierExecution,
    policy: &TpmRuntimeContinuityPolicy,
) -> Result<TpmRuntimeQualifiedEvidence, TpmRuntimeContinuityError> {
    let report = assess_tpm_runtime_continuity(
        candidate,
        verification,
        parent,
        campaign_runtime,
        obligation_runtime,
        policy,
    );
    if report.disposition != TpmRuntimeContinuityDisposition::Qualified {
        return Err(TpmRuntimeContinuityError::NotQualified(report));
    }

    let mut receipt = candidate
        .verify(verification)
        .map_err(|_| TpmRuntimeContinuityError::BaseReceiptFailed)?;
    let strict_receipt_digest = strict_receipt_digest(&receipt)
        .ok_or(TpmRuntimeContinuityError::BaseReceiptFailed)?;
    let report_digest = report.canonical_digest();
    receipt.evidence_digest = report_digest.clone();
    let qualification_digest = qualification_digest(
        &report,
        &report_digest,
        &strict_receipt_digest,
    );
    let qualification = RuntimeBackedTpmQualification {
        qualification_digest,
        report_digest,
        parent_tpm_qualification_digest: parent.qualification_digest().to_string(),
        campaign_runtime_continuity_digest: campaign_runtime.continuity_digest().to_string(),
        obligation_runtime_continuity_digest: obligation_runtime.continuity_digest().to_string(),
        strict_receipt_digest,
        campaign_verifier_ref: candidate.campaign_verifier_ref.clone(),
        obligation_verifier_ref: verification.verifier_ref.clone(),
        campaign_verified_at_ms: candidate.campaign_verified_at_ms,
        obligation_verified_at_ms: verification.verified_at_ms,
        current_use_at_ms: parent.current_use_at_ms(),
    };

    Ok(TpmRuntimeQualifiedEvidence {
        report,
        receipt,
        qualification,
    })
}

fn qualification_digest(
    report: &TpmRuntimeContinuityReport,
    report_digest: &str,
    strict_receipt_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    for value in [
        report_digest,
        strict_receipt_digest,
        report.parent_tpm_qualification_digest.as_str(),
        report.campaign_runtime_continuity_digest.as_str(),
        report.obligation_runtime_continuity_digest.as_str(),
        report.campaign_runtime_policy_digest.as_str(),
        report.obligation_runtime_policy_digest.as_str(),
        report.campaign_verifier_ref.as_str(),
        report.obligation_verifier_ref.as_str(),
    ] {
        push_field(&mut hasher, value);
    }
    push_u64(&mut hasher, report.campaign_verified_at_ms);
    push_u64(&mut hasher, report.obligation_verified_at_ms);
    push_u64(&mut hasher, report.current_use_at_ms);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn evidence_kind_code(kind: EvidenceKind) -> &'static str {
    match kind {
        EvidenceKind::FormalProof => "formal-proof",
        EvidenceKind::Simulation => "simulation",
        EvidenceKind::Test => "test",
        EvidenceKind::Telemetry => "telemetry",
        EvidenceKind::Standard => "standard",
    }
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT_BYTES
        && !value.chars().any(char::is_control)
}

fn digest_text(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|digest| {
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn valid_refs(refs: &[String]) -> bool {
    if refs.len() > MAX_EVIDENCE_REFS || refs.iter().any(|value| !canonical_text(value)) {
        return false;
    }
    let mut sorted = refs.to_vec();
    sorted.sort();
    sorted.dedup();
    sorted.len() == refs.len()
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn push_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_be_bytes());
}

fn push_sorted_refs(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut refs = refs.to_vec();
    refs.sort();
    for reference in refs {
        push_field(hasher, &reference);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn receipt() -> SafetyEvidenceReceipt {
        SafetyEvidenceReceipt {
            receipt_id: "receipt:1".into(),
            obligation_key: d("obligation"),
            evidence_kind: EvidenceKind::Test,
            evidence_ref: "evidence:campaign".into(),
            evidence_digest: d("campaign-output"),
            verifier_ref: "verifier:obligation".into(),
            verified_at_ms: 4_000,
        }
    }

    #[test]
    fn strict_receipt_digest_binds_verifier_and_time() {
        let first = strict_receipt_digest(&receipt()).unwrap();
        let mut changed = receipt();
        changed.verifier_ref = "verifier:other".into();
        let second = strict_receipt_digest(&changed).unwrap();
        assert_ne!(first, second);

        let mut changed_time = receipt();
        changed_time.verified_at_ms += 1;
        assert_ne!(first, strict_receipt_digest(&changed_time).unwrap());
    }

    #[test]
    fn policy_identity_binds_campaign_input_and_both_runtime_policies() {
        let base = TpmRuntimeContinuityPolicy {
            schema_version: TPM_RUNTIME_CONTINUITY_POLICY_SCHEMA_V1.into(),
            policy_id: "tpm-runtime:1".into(),
            expected_campaign_runtime_policy_digest: d("campaign-runtime-policy"),
            expected_obligation_runtime_policy_digest: d("obligation-runtime-policy"),
            expected_campaign_runtime_input_digest: d("campaign-input"),
            evidence_refs: vec!["review:tpm-runtime".into()],
        };
        let first = base.canonical_digest().unwrap();
        let mut changed = base.clone();
        changed.expected_campaign_runtime_input_digest = d("other-input");
        assert_ne!(first, changed.canonical_digest().unwrap());
        let mut changed_policy = base;
        changed_policy.expected_obligation_runtime_policy_digest = d("other-policy");
        assert_ne!(first, changed_policy.canonical_digest().unwrap());
    }
}
