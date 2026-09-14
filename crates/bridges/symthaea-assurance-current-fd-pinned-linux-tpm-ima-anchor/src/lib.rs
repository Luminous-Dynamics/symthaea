// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Current Linux TPM+IMA provider assurance bound to FD-pinned quote execution.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_assurance_linux_tpm_ima_current_anchor::CurrentLinuxTpmImaRuntimeAnchor;
use symthaea_assurance_tpm2_fd_pinned_checkquote::VerifiedFdPinnedTpm2Quote;

pub const CURRENT_FD_PINNED_ANCHOR_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.current-fd-pinned-linux-tpm-ima-anchor-policy.v1";
pub const CURRENT_FD_PINNED_ANCHOR_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.current-fd-pinned-linux-tpm-ima-anchor-report.v1";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.current-fd-pinned-linux-tpm-ima-anchor-policy.digest.v1\0";
const REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.current-fd-pinned-linux-tpm-ima-anchor-report.digest.v1\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.current-fd-pinned-linux-tpm-ima-anchor-qualification.digest.v1\0";
const MAX_TEXT_BYTES: usize = 512;
const MAX_EVIDENCE_REFS: usize = 128;
const SHA256_LEN: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CurrentFdPinnedAnchorPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_current_anchor_policy_digest: String,
    pub expected_fd_pinned_policy_digest: String,
    pub evidence_refs: Vec<String>,
}

impl CurrentFdPinnedAnchorPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == CURRENT_FD_PINNED_ANCHOR_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3_digest(&self.expected_current_anchor_policy_digest)
            && valid_blake3_digest(&self.expected_fd_pinned_policy_digest)
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
        push_field(&mut hasher, &self.expected_current_anchor_policy_digest);
        push_field(&mut hasher, &self.expected_fd_pinned_policy_digest);
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurrentFdPinnedAnchorDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurrentFdPinnedAnchorIssue {
    InvalidPolicy,
    CurrentAnchorPolicyMismatch,
    FdPinnedPolicyMismatch,
    QuoteQualificationMismatch,
    ChallengeMismatch,
    AkBindingMismatch,
    QuoteArtifactMismatch,
    VerificationReceiptMismatch,
    PcrSelectionMismatch,
}

impl CurrentFdPinnedAnchorIssue {
    fn code(&self) -> &'static str {
        match self {
            Self::InvalidPolicy => "invalid-policy",
            Self::CurrentAnchorPolicyMismatch => "current-anchor-policy-mismatch",
            Self::FdPinnedPolicyMismatch => "fd-pinned-policy-mismatch",
            Self::QuoteQualificationMismatch => "quote-qualification-mismatch",
            Self::ChallengeMismatch => "challenge-mismatch",
            Self::AkBindingMismatch => "ak-binding-mismatch",
            Self::QuoteArtifactMismatch => "quote-artifact-mismatch",
            Self::VerificationReceiptMismatch => "verification-receipt-mismatch",
            Self::PcrSelectionMismatch => "pcr-selection-mismatch",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CurrentFdPinnedAnchorReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub current_anchor_policy_digest: String,
    pub current_anchor_qualification_digest: String,
    pub fd_pinned_policy_digest: String,
    pub fd_pinned_qualification_digest: String,
    pub quote_qualification_digest: String,
    pub challenge_digest: String,
    pub ak_binding_digest: String,
    pub quote_artifact_digest: String,
    pub verification_receipt_digest: String,
    pub pcr_selection_digest: String,
    pub executable_digest: String,
    pub execution_record_digest: String,
    pub canonical_executable_path: String,
    pub nix_store_root: String,
    pub current_ledger_revision: u64,
    pub current_ledger_head_digest: String,
    pub current_ledger_state_digest: String,
    pub use_at_ms: u64,
    pub disposition: CurrentFdPinnedAnchorDisposition,
    pub issues: Vec<CurrentFdPinnedAnchorIssue>,
}

impl CurrentFdPinnedAnchorReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DIGEST_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.current_anchor_policy_digest.as_str(),
            self.current_anchor_qualification_digest.as_str(),
            self.fd_pinned_policy_digest.as_str(),
            self.fd_pinned_qualification_digest.as_str(),
            self.quote_qualification_digest.as_str(),
            self.challenge_digest.as_str(),
            self.ak_binding_digest.as_str(),
            self.quote_artifact_digest.as_str(),
            self.verification_receipt_digest.as_str(),
            self.pcr_selection_digest.as_str(),
            self.executable_digest.as_str(),
            self.execution_record_digest.as_str(),
            self.canonical_executable_path.as_str(),
            self.nix_store_root.as_str(),
            self.current_ledger_head_digest.as_str(),
            self.current_ledger_state_digest.as_str(),
        ] {
            push_field(&mut hasher, value);
        }
        hasher.update(&self.current_ledger_revision.to_le_bytes());
        hasher.update(&self.use_at_ms.to_le_bytes());
        push_field(
            &mut hasher,
            match self.disposition {
                CurrentFdPinnedAnchorDisposition::Invalid => "invalid",
                CurrentFdPinnedAnchorDisposition::Blocked => "blocked",
                CurrentFdPinnedAnchorDisposition::Qualified => "qualified",
            },
        );
        for issue in &self.issues {
            push_field(&mut hasher, issue.code());
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentFdPinnedLinuxTpmImaRuntimeAnchor {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    current_anchor_policy_digest: String,
    current_anchor_qualification_digest: String,
    fd_pinned_policy_digest: String,
    fd_pinned_qualification_digest: String,
    quote_qualification_digest: String,
    challenge_digest: String,
    ak_binding_digest: String,
    quote_artifact_digest: String,
    verification_receipt_digest: String,
    pcr_selection_digest: String,
    pcr_binding_policy_digest: String,
    pcr_binding_qualification_digest: String,
    measurement_list_digest: String,
    matched_required_measurements: Vec<String>,
    pcr: u8,
    pcr_value: [u8; SHA256_LEN],
    executable_digest: String,
    execution_record_digest: String,
    canonical_executable_path: String,
    nix_store_root: String,
    executable_device: u64,
    executable_inode: u64,
    executable_file_size: u64,
    ledger_id: String,
    current_ledger_revision: u64,
    current_ledger_head_digest: String,
    current_ledger_state_digest: String,
    head_statement_digest: String,
    currentness_attestation_digest: String,
    head_authority_policy_digest: String,
    authority_sequence: u64,
    use_at_ms: u64,
}

impl CurrentFdPinnedLinuxTpmImaRuntimeAnchor {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn current_anchor_policy_digest(&self) -> &str { &self.current_anchor_policy_digest }
    pub fn current_anchor_qualification_digest(&self) -> &str { &self.current_anchor_qualification_digest }
    pub fn fd_pinned_policy_digest(&self) -> &str { &self.fd_pinned_policy_digest }
    pub fn fd_pinned_qualification_digest(&self) -> &str { &self.fd_pinned_qualification_digest }
    pub fn quote_qualification_digest(&self) -> &str { &self.quote_qualification_digest }
    pub fn challenge_digest(&self) -> &str { &self.challenge_digest }
    pub fn ak_binding_digest(&self) -> &str { &self.ak_binding_digest }
    pub fn quote_artifact_digest(&self) -> &str { &self.quote_artifact_digest }
    pub fn verification_receipt_digest(&self) -> &str { &self.verification_receipt_digest }
    pub fn pcr_selection_digest(&self) -> &str { &self.pcr_selection_digest }
    pub fn pcr_binding_policy_digest(&self) -> &str { &self.pcr_binding_policy_digest }
    pub fn pcr_binding_qualification_digest(&self) -> &str { &self.pcr_binding_qualification_digest }
    pub fn measurement_list_digest(&self) -> &str { &self.measurement_list_digest }
    pub fn matched_required_measurements(&self) -> &[String] { &self.matched_required_measurements }
    pub const fn pcr(&self) -> u8 { self.pcr }
    pub const fn pcr_value(&self) -> [u8; SHA256_LEN] { self.pcr_value }
    pub fn executable_digest(&self) -> &str { &self.executable_digest }
    pub fn execution_record_digest(&self) -> &str { &self.execution_record_digest }
    pub fn canonical_executable_path(&self) -> &str { &self.canonical_executable_path }
    pub fn nix_store_root(&self) -> &str { &self.nix_store_root }
    pub const fn executable_device(&self) -> u64 { self.executable_device }
    pub const fn executable_inode(&self) -> u64 { self.executable_inode }
    pub const fn executable_file_size(&self) -> u64 { self.executable_file_size }
    pub fn ledger_id(&self) -> &str { &self.ledger_id }
    pub const fn current_ledger_revision(&self) -> u64 { self.current_ledger_revision }
    pub fn current_ledger_head_digest(&self) -> &str { &self.current_ledger_head_digest }
    pub fn current_ledger_state_digest(&self) -> &str { &self.current_ledger_state_digest }
    pub fn head_statement_digest(&self) -> &str { &self.head_statement_digest }
    pub fn currentness_attestation_digest(&self) -> &str { &self.currentness_attestation_digest }
    pub fn head_authority_policy_digest(&self) -> &str { &self.head_authority_policy_digest }
    pub const fn authority_sequence(&self) -> u64 { self.authority_sequence }
    pub const fn use_at_ms(&self) -> u64 { self.use_at_ms }
    pub const fn executable_path_toctou_closed(&self) -> bool { true }
    pub const fn authority_current_ledger_membership_established(&self) -> bool { true }
    pub const fn dependency_closure_atomically_pinned(&self) -> bool { false }
    pub const fn root_resistant_immutability_established(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentFdPinnedAnchorQualification {
    pub report: CurrentFdPinnedAnchorReport,
    verified: CurrentFdPinnedLinuxTpmImaRuntimeAnchor,
}

impl CurrentFdPinnedAnchorQualification {
    pub fn verified(&self) -> &CurrentFdPinnedLinuxTpmImaRuntimeAnchor { &self.verified }
    pub fn into_verified(self) -> CurrentFdPinnedLinuxTpmImaRuntimeAnchor { self.verified }
}

pub fn bind_current_anchor_to_fd_pinned_quote(
    policy: &CurrentFdPinnedAnchorPolicy,
    current: &CurrentLinuxTpmImaRuntimeAnchor,
    fd_quote: &VerifiedFdPinnedTpm2Quote,
) -> Result<CurrentFdPinnedAnchorQualification, CurrentFdPinnedAnchorReport> {
    let policy_digest = policy.canonical_digest();
    let mut report = CurrentFdPinnedAnchorReport {
        schema_version: CURRENT_FD_PINNED_ANCHOR_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest: policy_digest.clone(),
        current_anchor_policy_digest: current.policy_digest().into(),
        current_anchor_qualification_digest: current.qualification_digest().into(),
        fd_pinned_policy_digest: fd_quote.policy_digest().into(),
        fd_pinned_qualification_digest: fd_quote.qualification_digest().into(),
        quote_qualification_digest: current.quote_qualification_digest().into(),
        challenge_digest: current.challenge_digest().into(),
        ak_binding_digest: current.ak_binding_digest().into(),
        quote_artifact_digest: current.quote_artifact_digest().into(),
        verification_receipt_digest: current.verification_receipt_digest().into(),
        pcr_selection_digest: current.pcr_selection_digest().into(),
        executable_digest: fd_quote.executable_digest().into(),
        execution_record_digest: fd_quote.execution_record_digest().into(),
        canonical_executable_path: fd_quote.canonical_path().into(),
        nix_store_root: fd_quote.nix_store_root().into(),
        current_ledger_revision: current.current_ledger_revision(),
        current_ledger_head_digest: current.current_ledger_head_digest().into(),
        current_ledger_state_digest: current.current_ledger_state_digest().into(),
        use_at_ms: current.use_at_ms(),
        disposition: CurrentFdPinnedAnchorDisposition::Invalid,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report.issues.push(CurrentFdPinnedAnchorIssue::InvalidPolicy);
        return Err(finalize_report(report));
    }
    if current.policy_digest() != policy.expected_current_anchor_policy_digest {
        report.issues.push(CurrentFdPinnedAnchorIssue::CurrentAnchorPolicyMismatch);
        return Err(finalize_report(report));
    }
    if fd_quote.policy_digest() != policy.expected_fd_pinned_policy_digest {
        report.issues.push(CurrentFdPinnedAnchorIssue::FdPinnedPolicyMismatch);
        return Err(finalize_report(report));
    }
    if current.quote_qualification_digest() != fd_quote.inner_quote_qualification_digest() {
        report.issues.push(CurrentFdPinnedAnchorIssue::QuoteQualificationMismatch);
        return Err(finalize_report(report));
    }
    if current.challenge_digest() != fd_quote.challenge_digest() {
        report.issues.push(CurrentFdPinnedAnchorIssue::ChallengeMismatch);
        return Err(finalize_report(report));
    }
    if current.ak_binding_digest() != fd_quote.ak_binding_digest() {
        report.issues.push(CurrentFdPinnedAnchorIssue::AkBindingMismatch);
        return Err(finalize_report(report));
    }
    if current.quote_artifact_digest() != fd_quote.quote_artifact_digest() {
        report.issues.push(CurrentFdPinnedAnchorIssue::QuoteArtifactMismatch);
        return Err(finalize_report(report));
    }
    if current.verification_receipt_digest() != fd_quote.verification_receipt_digest() {
        report.issues.push(CurrentFdPinnedAnchorIssue::VerificationReceiptMismatch);
        return Err(finalize_report(report));
    }
    if current.pcr_selection_digest() != fd_quote.pcr_selection_digest() {
        report.issues.push(CurrentFdPinnedAnchorIssue::PcrSelectionMismatch);
        return Err(finalize_report(report));
    }

    report.disposition = CurrentFdPinnedAnchorDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let qualification_digest = qualification_digest(
        &report_digest,
        &policy_digest,
        current.qualification_digest(),
        fd_quote.qualification_digest(),
        fd_quote.execution_record_digest(),
    );
    let verified = CurrentFdPinnedLinuxTpmImaRuntimeAnchor {
        qualification_digest,
        report_digest,
        policy_digest,
        current_anchor_policy_digest: current.policy_digest().into(),
        current_anchor_qualification_digest: current.qualification_digest().into(),
        fd_pinned_policy_digest: fd_quote.policy_digest().into(),
        fd_pinned_qualification_digest: fd_quote.qualification_digest().into(),
        quote_qualification_digest: current.quote_qualification_digest().into(),
        challenge_digest: current.challenge_digest().into(),
        ak_binding_digest: current.ak_binding_digest().into(),
        quote_artifact_digest: current.quote_artifact_digest().into(),
        verification_receipt_digest: current.verification_receipt_digest().into(),
        pcr_selection_digest: current.pcr_selection_digest().into(),
        pcr_binding_policy_digest: current.pcr_binding_policy_digest().into(),
        pcr_binding_qualification_digest: current.pcr_binding_qualification_digest().into(),
        measurement_list_digest: current.measurement_list_digest().into(),
        matched_required_measurements: current.matched_required_measurements().to_vec(),
        pcr: current.pcr(),
        pcr_value: current.pcr_value(),
        executable_digest: fd_quote.executable_digest().into(),
        execution_record_digest: fd_quote.execution_record_digest().into(),
        canonical_executable_path: fd_quote.canonical_path().into(),
        nix_store_root: fd_quote.nix_store_root().into(),
        executable_device: fd_quote.device(),
        executable_inode: fd_quote.inode(),
        executable_file_size: fd_quote.file_size(),
        ledger_id: current.ledger_id().into(),
        current_ledger_revision: current.current_ledger_revision(),
        current_ledger_head_digest: current.current_ledger_head_digest().into(),
        current_ledger_state_digest: current.current_ledger_state_digest().into(),
        head_statement_digest: current.head_statement_digest().into(),
        currentness_attestation_digest: current.currentness_attestation_digest().into(),
        head_authority_policy_digest: current.head_authority_policy_digest().into(),
        authority_sequence: current.authority_sequence(),
        use_at_ms: current.use_at_ms(),
    };
    Ok(CurrentFdPinnedAnchorQualification { report, verified })
}

fn qualification_digest(
    report_digest: &str,
    policy_digest: &str,
    current_anchor_qualification_digest: &str,
    fd_pinned_qualification_digest: &str,
    execution_record_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    for value in [
        report_digest,
        policy_digest,
        current_anchor_qualification_digest,
        fd_pinned_qualification_digest,
        execution_record_digest,
    ] {
        push_field(&mut hasher, value);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn finalize_report(mut report: CurrentFdPinnedAnchorReport) -> CurrentFdPinnedAnchorReport {
    report.disposition = if report.issues.contains(&CurrentFdPinnedAnchorIssue::InvalidPolicy) {
        CurrentFdPinnedAnchorDisposition::Invalid
    } else {
        CurrentFdPinnedAnchorDisposition::Blocked
    };
    report
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT_BYTES
        && !value.chars().any(char::is_control)
}

fn valid_refs(values: &[String]) -> bool {
    !values.is_empty()
        && values.len() <= MAX_EVIDENCE_REFS
        && values.iter().all(|value| canonical_text(value))
        && values.iter().collect::<BTreeSet<_>>().len() == values.len()
}

fn valid_blake3_digest(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|digest| {
        digest.len() == 64
            && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            && !digest.bytes().any(|byte| byte.is_ascii_uppercase())
    })
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn push_sorted_refs(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut refs = refs.to_vec();
    refs.sort();
    hasher.update(&(refs.len() as u64).to_le_bytes());
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

    fn policy() -> CurrentFdPinnedAnchorPolicy {
        CurrentFdPinnedAnchorPolicy {
            schema_version: CURRENT_FD_PINNED_ANCHOR_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:current-fd-pinned".into(),
            expected_current_anchor_policy_digest: d("current-anchor-policy"),
            expected_fd_pinned_policy_digest: d("fd-pinned-policy"),
            evidence_refs: vec!["review:terminal-composition".into()],
        }
    }

    #[test]
    fn policy_digest_binds_both_parent_policies() {
        let base = policy();
        let digest = base.canonical_digest().unwrap();

        let mut changed = base.clone();
        changed.expected_current_anchor_policy_digest = d("other-current-policy");
        assert_ne!(digest, changed.canonical_digest().unwrap());

        let mut changed = base;
        changed.expected_fd_pinned_policy_digest = d("other-fd-policy");
        assert_ne!(digest, changed.canonical_digest().unwrap());
    }

    #[test]
    fn evidence_ref_order_is_nonsemantic() {
        let mut left = policy();
        left.evidence_refs = vec!["review:a".into(), "review:b".into()];
        let mut right = left.clone();
        right.evidence_refs.reverse();
        assert_eq!(left.canonical_digest(), right.canonical_digest());
    }
}
