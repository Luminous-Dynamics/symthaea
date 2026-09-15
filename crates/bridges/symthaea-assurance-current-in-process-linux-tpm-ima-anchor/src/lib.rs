// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authority-current Linux TPM+IMA assurance bound to in-process quote verification.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_assurance_linux_tpm_ima_current_anchor::CurrentLinuxTpmImaRuntimeAnchor;
use symthaea_assurance_tpm2_in_process_ecdsa::{
    VerifiedInProcessTpm2Quote, IN_PROCESS_BACKEND_ID_V1,
};

pub const CURRENT_IN_PROCESS_ANCHOR_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.current-in-process-linux-tpm-ima-anchor-policy.v1";
pub const CURRENT_IN_PROCESS_ANCHOR_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.current-in-process-linux-tpm-ima-anchor-report.v1";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.current-in-process-linux-tpm-ima-anchor-policy.digest.v1\0";
const REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.current-in-process-linux-tpm-ima-anchor-report.digest.v1\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.current-in-process-linux-tpm-ima-anchor-qualification.digest.v1\0";
const MAX_TEXT_BYTES: usize = 1024;
const MAX_EVIDENCE_REFS: usize = 128;
const SHA256_LEN: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CurrentInProcessAnchorPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_current_anchor_policy_digest: String,
    pub expected_in_process_policy_digest: String,
    pub expected_backend_id: String,
    pub evidence_refs: Vec<String>,
}

impl CurrentInProcessAnchorPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == CURRENT_IN_PROCESS_ANCHOR_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3_digest(&self.expected_current_anchor_policy_digest)
            && valid_blake3_digest(&self.expected_in_process_policy_digest)
            && self.expected_backend_id == IN_PROCESS_BACKEND_ID_V1
            && valid_refs(&self.evidence_refs)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DIGEST_DOMAIN);
        for field in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.expected_current_anchor_policy_digest.as_str(),
            self.expected_in_process_policy_digest.as_str(),
            self.expected_backend_id.as_str(),
        ] {
            push_field(&mut hasher, field);
        }
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurrentInProcessAnchorDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurrentInProcessAnchorIssue {
    InvalidPolicy,
    CurrentAnchorPolicyMismatch,
    InProcessPolicyMismatch,
    BackendMismatch,
    QuoteQualificationMismatch,
    ChallengeMismatch,
    AkBindingMismatch,
    QuoteArtifactMismatch,
    VerificationReceiptMismatch,
    PcrSelectionMismatch,
}

impl CurrentInProcessAnchorIssue {
    fn code(&self) -> &'static str {
        match self {
            Self::InvalidPolicy => "invalid-policy",
            Self::CurrentAnchorPolicyMismatch => "current-anchor-policy-mismatch",
            Self::InProcessPolicyMismatch => "in-process-policy-mismatch",
            Self::BackendMismatch => "backend-mismatch",
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
pub struct CurrentInProcessAnchorReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub current_anchor_policy_digest: String,
    pub current_anchor_qualification_digest: String,
    pub in_process_policy_digest: String,
    pub in_process_qualification_digest: String,
    pub raw_quote_qualification_digest: String,
    pub challenge_digest: String,
    pub ak_binding_digest: String,
    pub quote_artifact_digest: String,
    pub verification_receipt_digest: String,
    pub pcr_selection_digest: String,
    pub host_identity_digest: String,
    pub host_executable_path: String,
    pub host_nix_store_root: String,
    pub host_executable_blake3: String,
    pub backend_id: String,
    pub current_ledger_revision: u64,
    pub current_ledger_head_digest: String,
    pub current_ledger_state_digest: String,
    pub use_at_ms: u64,
    pub disposition: CurrentInProcessAnchorDisposition,
    pub issues: Vec<CurrentInProcessAnchorIssue>,
}

impl CurrentInProcessAnchorReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DIGEST_DOMAIN);
        for field in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.current_anchor_policy_digest.as_str(),
            self.current_anchor_qualification_digest.as_str(),
            self.in_process_policy_digest.as_str(),
            self.in_process_qualification_digest.as_str(),
            self.raw_quote_qualification_digest.as_str(),
            self.challenge_digest.as_str(),
            self.ak_binding_digest.as_str(),
            self.quote_artifact_digest.as_str(),
            self.verification_receipt_digest.as_str(),
            self.pcr_selection_digest.as_str(),
            self.host_identity_digest.as_str(),
            self.host_executable_path.as_str(),
            self.host_nix_store_root.as_str(),
            self.host_executable_blake3.as_str(),
            self.backend_id.as_str(),
            self.current_ledger_head_digest.as_str(),
            self.current_ledger_state_digest.as_str(),
        ] {
            push_field(&mut hasher, field);
        }
        hasher.update(&self.current_ledger_revision.to_le_bytes());
        hasher.update(&self.use_at_ms.to_le_bytes());
        push_field(
            &mut hasher,
            match self.disposition {
                CurrentInProcessAnchorDisposition::Invalid => "invalid",
                CurrentInProcessAnchorDisposition::Blocked => "blocked",
                CurrentInProcessAnchorDisposition::Qualified => "qualified",
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
pub struct CurrentInProcessLinuxTpmImaRuntimeAnchor {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    current_anchor_policy_digest: String,
    current_anchor_qualification_digest: String,
    in_process_policy_digest: String,
    in_process_qualification_digest: String,
    raw_quote_qualification_digest: String,
    possession_policy_digest: String,
    pcr_binding_policy_digest: String,
    pcr_binding_qualification_digest: String,
    platform_qualification_digest: String,
    challenge_digest: String,
    ak_binding_digest: String,
    quote_artifact_digest: String,
    verification_receipt_digest: String,
    pcr_selection_digest: String,
    measurement_list_digest: String,
    matched_required_measurements: Vec<String>,
    pcr: u8,
    pcr_value: [u8; SHA256_LEN],
    host_identity_digest: String,
    host_executable_path: String,
    host_nix_store_root: String,
    host_executable_blake3: String,
    host_device: u64,
    host_inode: u64,
    host_file_size: u64,
    backend_id: String,
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

impl CurrentInProcessLinuxTpmImaRuntimeAnchor {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn current_anchor_policy_digest(&self) -> &str { &self.current_anchor_policy_digest }
    pub fn current_anchor_qualification_digest(&self) -> &str { &self.current_anchor_qualification_digest }
    pub fn in_process_policy_digest(&self) -> &str { &self.in_process_policy_digest }
    pub fn in_process_qualification_digest(&self) -> &str { &self.in_process_qualification_digest }
    pub fn raw_quote_qualification_digest(&self) -> &str { &self.raw_quote_qualification_digest }
    pub fn possession_policy_digest(&self) -> &str { &self.possession_policy_digest }
    pub fn pcr_binding_policy_digest(&self) -> &str { &self.pcr_binding_policy_digest }
    pub fn pcr_binding_qualification_digest(&self) -> &str { &self.pcr_binding_qualification_digest }
    pub fn platform_qualification_digest(&self) -> &str { &self.platform_qualification_digest }
    pub fn challenge_digest(&self) -> &str { &self.challenge_digest }
    pub fn ak_binding_digest(&self) -> &str { &self.ak_binding_digest }
    pub fn quote_artifact_digest(&self) -> &str { &self.quote_artifact_digest }
    pub fn verification_receipt_digest(&self) -> &str { &self.verification_receipt_digest }
    pub fn pcr_selection_digest(&self) -> &str { &self.pcr_selection_digest }
    pub fn measurement_list_digest(&self) -> &str { &self.measurement_list_digest }
    pub fn matched_required_measurements(&self) -> &[String] { &self.matched_required_measurements }
    pub const fn pcr(&self) -> u8 { self.pcr }
    pub const fn pcr_value(&self) -> [u8; SHA256_LEN] { self.pcr_value }
    pub fn host_identity_digest(&self) -> &str { &self.host_identity_digest }
    pub fn host_executable_path(&self) -> &str { &self.host_executable_path }
    pub fn host_nix_store_root(&self) -> &str { &self.host_nix_store_root }
    pub fn host_executable_blake3(&self) -> &str { &self.host_executable_blake3 }
    pub const fn host_device(&self) -> u64 { self.host_device }
    pub const fn host_inode(&self) -> u64 { self.host_inode }
    pub const fn host_file_size(&self) -> u64 { self.host_file_size }
    pub fn backend_id(&self) -> &str { &self.backend_id }
    pub fn ledger_id(&self) -> &str { &self.ledger_id }
    pub const fn current_ledger_revision(&self) -> u64 { self.current_ledger_revision }
    pub fn current_ledger_head_digest(&self) -> &str { &self.current_ledger_head_digest }
    pub fn current_ledger_state_digest(&self) -> &str { &self.current_ledger_state_digest }
    pub fn head_statement_digest(&self) -> &str { &self.head_statement_digest }
    pub fn currentness_attestation_digest(&self) -> &str { &self.currentness_attestation_digest }
    pub fn head_authority_policy_digest(&self) -> &str { &self.head_authority_policy_digest }
    pub const fn authority_sequence(&self) -> u64 { self.authority_sequence }
    pub const fn use_at_ms(&self) -> u64 { self.use_at_ms }
    pub const fn signature_verification_in_process(&self) -> bool { true }
    pub const fn external_checkquote_process_required(&self) -> bool { false }
    pub const fn running_host_backing_file_identity_observed(&self) -> bool { true }
    pub const fn authority_current_ledger_membership_established(&self) -> bool { true }
    pub const fn mapped_memory_identity_established(&self) -> bool { false }
    pub const fn dependency_closure_atomically_pinned(&self) -> bool { false }
    pub const fn privileged_in_place_mutation_excluded(&self) -> bool { false }
    pub const fn root_resistant_immutability_established(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentInProcessAnchorQualification {
    pub report: CurrentInProcessAnchorReport,
    verified: CurrentInProcessLinuxTpmImaRuntimeAnchor,
}

impl CurrentInProcessAnchorQualification {
    pub fn verified(&self) -> &CurrentInProcessLinuxTpmImaRuntimeAnchor { &self.verified }
    pub fn into_verified(self) -> CurrentInProcessLinuxTpmImaRuntimeAnchor { self.verified }
}

pub fn bind_current_anchor_to_in_process_quote(
    policy: &CurrentInProcessAnchorPolicy,
    current: &CurrentLinuxTpmImaRuntimeAnchor,
    in_process: &VerifiedInProcessTpm2Quote,
) -> Result<CurrentInProcessAnchorQualification, CurrentInProcessAnchorReport> {
    let policy_digest = policy.canonical_digest();
    let mut report = CurrentInProcessAnchorReport {
        schema_version: CURRENT_IN_PROCESS_ANCHOR_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest: policy_digest.clone(),
        current_anchor_policy_digest: current.policy_digest().into(),
        current_anchor_qualification_digest: current.qualification_digest().into(),
        in_process_policy_digest: in_process.policy_digest().into(),
        in_process_qualification_digest: in_process.qualification_digest().into(),
        raw_quote_qualification_digest: in_process.parent_qualification_digest().into(),
        challenge_digest: in_process.challenge_digest().into(),
        ak_binding_digest: in_process.ak_binding_digest().into(),
        quote_artifact_digest: in_process.quote_artifact_digest().into(),
        verification_receipt_digest: in_process.verification_receipt_digest().into(),
        pcr_selection_digest: in_process.pcr_selection_digest().into(),
        host_identity_digest: in_process.host_identity_digest().into(),
        host_executable_path: in_process.host_executable_path().into(),
        host_nix_store_root: in_process.host_nix_store_root().into(),
        host_executable_blake3: in_process.host_executable_blake3().into(),
        backend_id: in_process.backend_id().into(),
        current_ledger_revision: current.current_ledger_revision(),
        current_ledger_head_digest: current.current_ledger_head_digest().into(),
        current_ledger_state_digest: current.current_ledger_state_digest().into(),
        use_at_ms: current.use_at_ms(),
        disposition: CurrentInProcessAnchorDisposition::Invalid,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report.issues.push(CurrentInProcessAnchorIssue::InvalidPolicy);
        return Err(finalize(report));
    }
    if current.policy_digest() != policy.expected_current_anchor_policy_digest {
        report.issues.push(CurrentInProcessAnchorIssue::CurrentAnchorPolicyMismatch);
        return Err(finalize(report));
    }
    if in_process.policy_digest() != policy.expected_in_process_policy_digest {
        report.issues.push(CurrentInProcessAnchorIssue::InProcessPolicyMismatch);
        return Err(finalize(report));
    }
    if in_process.backend_id() != policy.expected_backend_id {
        report.issues.push(CurrentInProcessAnchorIssue::BackendMismatch);
        return Err(finalize(report));
    }
    if current.quote_qualification_digest() != in_process.parent_qualification_digest() {
        report.issues.push(CurrentInProcessAnchorIssue::QuoteQualificationMismatch);
        return Err(finalize(report));
    }
    if current.challenge_digest() != in_process.challenge_digest() {
        report.issues.push(CurrentInProcessAnchorIssue::ChallengeMismatch);
        return Err(finalize(report));
    }
    if current.ak_binding_digest() != in_process.ak_binding_digest() {
        report.issues.push(CurrentInProcessAnchorIssue::AkBindingMismatch);
        return Err(finalize(report));
    }
    if current.quote_artifact_digest() != in_process.quote_artifact_digest() {
        report.issues.push(CurrentInProcessAnchorIssue::QuoteArtifactMismatch);
        return Err(finalize(report));
    }
    if current.verification_receipt_digest() != in_process.verification_receipt_digest() {
        report.issues.push(CurrentInProcessAnchorIssue::VerificationReceiptMismatch);
        return Err(finalize(report));
    }
    if current.pcr_selection_digest() != in_process.pcr_selection_digest() {
        report.issues.push(CurrentInProcessAnchorIssue::PcrSelectionMismatch);
        return Err(finalize(report));
    }

    report.disposition = CurrentInProcessAnchorDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let qualification_digest = qualification_digest(
        &policy_digest,
        current.qualification_digest(),
        in_process.qualification_digest(),
        in_process.parent_qualification_digest(),
        in_process.host_identity_digest(),
        in_process.backend_id(),
        &report_digest,
    );
    let verified = CurrentInProcessLinuxTpmImaRuntimeAnchor {
        qualification_digest,
        report_digest,
        policy_digest,
        current_anchor_policy_digest: current.policy_digest().into(),
        current_anchor_qualification_digest: current.qualification_digest().into(),
        in_process_policy_digest: in_process.policy_digest().into(),
        in_process_qualification_digest: in_process.qualification_digest().into(),
        raw_quote_qualification_digest: in_process.parent_qualification_digest().into(),
        possession_policy_digest: current.possession_policy_digest().into(),
        pcr_binding_policy_digest: current.pcr_binding_policy_digest().into(),
        pcr_binding_qualification_digest: current.pcr_binding_qualification_digest().into(),
        platform_qualification_digest: current.platform_qualification_digest().into(),
        challenge_digest: current.challenge_digest().into(),
        ak_binding_digest: current.ak_binding_digest().into(),
        quote_artifact_digest: current.quote_artifact_digest().into(),
        verification_receipt_digest: current.verification_receipt_digest().into(),
        pcr_selection_digest: current.pcr_selection_digest().into(),
        measurement_list_digest: current.measurement_list_digest().into(),
        matched_required_measurements: current.matched_required_measurements().to_vec(),
        pcr: current.pcr(),
        pcr_value: current.pcr_value(),
        host_identity_digest: in_process.host_identity_digest().into(),
        host_executable_path: in_process.host_executable_path().into(),
        host_nix_store_root: in_process.host_nix_store_root().into(),
        host_executable_blake3: in_process.host_executable_blake3().into(),
        host_device: in_process.host_device(),
        host_inode: in_process.host_inode(),
        host_file_size: in_process.host_file_size(),
        backend_id: in_process.backend_id().into(),
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

    Ok(CurrentInProcessAnchorQualification { report, verified })
}

fn finalize(mut report: CurrentInProcessAnchorReport) -> CurrentInProcessAnchorReport {
    report.disposition = if report.issues.contains(&CurrentInProcessAnchorIssue::InvalidPolicy) {
        CurrentInProcessAnchorDisposition::Invalid
    } else {
        CurrentInProcessAnchorDisposition::Blocked
    };
    report
}

fn qualification_digest(
    policy_digest: &str,
    current_qualification_digest: &str,
    in_process_qualification_digest: &str,
    raw_quote_qualification_digest: &str,
    host_identity_digest: &str,
    backend_id: &str,
    report_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    for field in [
        policy_digest,
        current_qualification_digest,
        in_process_qualification_digest,
        raw_quote_qualification_digest,
        host_identity_digest,
        backend_id,
        report_digest,
    ] {
        push_field(&mut hasher, field);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn valid_blake3_digest(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("blake3:") else { return false; };
    hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT_BYTES
        && !value.chars().any(char::is_control)
}

fn valid_refs(refs: &[String]) -> bool {
    if refs.len() > MAX_EVIDENCE_REFS || !refs.iter().all(|value| canonical_text(value)) {
        return false;
    }
    let mut seen = BTreeSet::new();
    refs.iter().all(|value| seen.insert(value.as_str()))
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn push_sorted_refs(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut refs = refs.to_vec();
    refs.sort();
    hasher.update(&(refs.len() as u64).to_le_bytes());
    for reference in refs { push_field(hasher, &reference); }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seed: &str) -> String {
        format!("blake3:{}", blake3::hash(seed.as_bytes()).to_hex())
    }

    fn policy() -> CurrentInProcessAnchorPolicy {
        CurrentInProcessAnchorPolicy {
            schema_version: CURRENT_IN_PROCESS_ANCHOR_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:current-in-process-anchor:1".into(),
            expected_current_anchor_policy_digest: digest("current"),
            expected_in_process_policy_digest: digest("in-process"),
            expected_backend_id: IN_PROCESS_BACKEND_ID_V1.into(),
            evidence_refs: vec!["review:current".into(), "review:in-process".into()],
        }
    }

    #[test]
    fn evidence_reference_order_is_nonsemantic() {
        let left = policy();
        let mut right = left.clone();
        right.evidence_refs.reverse();
        assert_eq!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn both_parent_policies_are_semantic() {
        let base = policy();
        let base_digest = base.canonical_digest().unwrap();
        let mut changed = base.clone();
        changed.expected_current_anchor_policy_digest = digest("other-current");
        assert_ne!(base_digest, changed.canonical_digest().unwrap());
        changed = base.clone();
        changed.expected_in_process_policy_digest = digest("other-in-process");
        assert_ne!(base_digest, changed.canonical_digest().unwrap());
    }

    #[test]
    fn backend_is_fixed_to_reviewed_profile() {
        let mut changed = policy();
        changed.expected_backend_id = "alternate-backend".into();
        assert!(!changed.validate());
    }

    #[test]
    fn qualification_identity_commits_both_parents_and_host() {
        let p = digest("policy");
        let current = digest("current-qualification");
        let in_process = digest("in-process-qualification");
        let raw = digest("raw-quote");
        let host = digest("host");
        let report = digest("report");
        let base = qualification_digest(
            &p, &current, &in_process, &raw, &host, IN_PROCESS_BACKEND_ID_V1, &report,
        );
        assert_ne!(
            base,
            qualification_digest(
                &p, &current, &digest("other-in-process"), &raw, &host,
                IN_PROCESS_BACKEND_ID_V1, &report,
            )
        );
        assert_ne!(
            base,
            qualification_digest(
                &p, &current, &in_process, &raw, &digest("other-host"),
                IN_PROCESS_BACKEND_ID_V1, &report,
            )
        );
    }
}
