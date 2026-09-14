// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Current-head composition for ledger-relative Linux TPM2 + IMA assurance.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_assurance_acceptance_ledger_head_currentness::{
    snapshot_acceptance_ledger, CurrentAcceptanceLedgerHead,
};
use symthaea_assurance_linux_tpm_ima_ledger_anchor::LedgerRelativeLinuxTpmImaAnchor;
use symthaea_assurance_tpm2_attestation_possession::AttestationAcceptanceLedger;

pub const CURRENT_LINUX_TPM_IMA_ANCHOR_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.current-linux-tpm-ima-anchor-policy.v1";
pub const CURRENT_LINUX_TPM_IMA_ANCHOR_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.current-linux-tpm-ima-anchor-report.v1";
pub const CURRENT_LEDGER_ANTI_REPLAY_SCOPE_V1: &str =
    "authority_current_acceptance_ledger_membership_at_exact_challenged_use_time_v1";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.current-linux-tpm-ima-anchor-policy.digest.v1\0";
const REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.current-linux-tpm-ima-anchor-report.digest.v1\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.current-linux-tpm-ima-anchor-qualification.digest.v1\0";
const MAX_TEXT_BYTES: usize = 512;
const MAX_EVIDENCE_REFS: usize = 128;
const MAX_LEDGER_RECORDS: u64 = 1_000_000;
const SHA256_LEN: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CurrentLinuxTpmImaAnchorPolicy {
    pub schema_version: String,
    pub provider_id: String,
    pub expected_ledger_relative_anchor_policy_digest: String,
    pub expected_head_authority_policy_digest: String,
    pub expected_ledger_id: String,
    pub max_ledger_records: u64,
    pub evidence_refs: Vec<String>,
}

impl CurrentLinuxTpmImaAnchorPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == CURRENT_LINUX_TPM_IMA_ANCHOR_POLICY_SCHEMA_V1
            && canonical_text(&self.provider_id)
            && valid_blake3_digest(&self.expected_ledger_relative_anchor_policy_digest)
            && valid_blake3_digest(&self.expected_head_authority_policy_digest)
            && canonical_text(&self.expected_ledger_id)
            && self.max_ledger_records > 0
            && self.max_ledger_records <= MAX_LEDGER_RECORDS
            && valid_refs(&self.evidence_refs)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DIGEST_DOMAIN);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.provider_id);
        push_field(
            &mut hasher,
            &self.expected_ledger_relative_anchor_policy_digest,
        );
        push_field(&mut hasher, &self.expected_head_authority_policy_digest);
        push_field(&mut hasher, &self.expected_ledger_id);
        hasher.update(&self.max_ledger_records.to_le_bytes());
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurrentLinuxTpmImaAnchorDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurrentLinuxTpmImaAnchorIssue {
    InvalidPolicy,
    InvalidCurrentLedger { reason: String },
    AnchorPolicyMismatch,
    HeadAuthorityPolicyMismatch,
    LedgerIdMismatch,
    CurrentLedgerRevisionMismatch { expected: u64, observed: u64 },
    CurrentLedgerHeadMismatch,
    CurrentLedgerStateMismatch,
    AnchorAcceptanceRevisionInvalid,
    AnchorAcceptanceNotInCurrentLedger,
    AnchorAcceptanceDigestMismatch,
    AnchorPredecessorMismatch,
    AnchorAcceptanceTimeMismatch,
    AnchorAfterCurrentnessUseTime,
}

impl CurrentLinuxTpmImaAnchorIssue {
    fn is_invalid(&self) -> bool {
        matches!(
            self,
            Self::InvalidPolicy | Self::InvalidCurrentLedger { .. }
        )
    }

    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::InvalidCurrentLedger { reason } => format!("invalid-current-ledger:{reason}"),
            Self::AnchorPolicyMismatch => "anchor-policy-mismatch".into(),
            Self::HeadAuthorityPolicyMismatch => "head-authority-policy-mismatch".into(),
            Self::LedgerIdMismatch => "ledger-id-mismatch".into(),
            Self::CurrentLedgerRevisionMismatch { expected, observed } => {
                format!("current-ledger-revision-mismatch:{expected}:{observed}")
            }
            Self::CurrentLedgerHeadMismatch => "current-ledger-head-mismatch".into(),
            Self::CurrentLedgerStateMismatch => "current-ledger-state-mismatch".into(),
            Self::AnchorAcceptanceRevisionInvalid => "anchor-acceptance-revision-invalid".into(),
            Self::AnchorAcceptanceNotInCurrentLedger => "anchor-acceptance-not-in-current-ledger".into(),
            Self::AnchorAcceptanceDigestMismatch => "anchor-acceptance-digest-mismatch".into(),
            Self::AnchorPredecessorMismatch => "anchor-predecessor-mismatch".into(),
            Self::AnchorAcceptanceTimeMismatch => "anchor-acceptance-time-mismatch".into(),
            Self::AnchorAfterCurrentnessUseTime => "anchor-after-currentness-use-time".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CurrentLinuxTpmImaAnchorReport {
    pub schema_version: String,
    pub provider_id: String,
    pub policy_digest: Option<String>,
    pub ledger_relative_anchor_policy_digest: String,
    pub ledger_relative_anchor_qualification_digest: String,
    pub head_authority_policy_digest: String,
    pub ledger_id: String,
    pub current_ledger_revision: u64,
    pub current_ledger_head_digest: String,
    pub current_ledger_state_digest: String,
    pub head_statement_digest: String,
    pub currentness_attestation_digest: String,
    pub authority_sequence: u64,
    pub acceptance_revision: u64,
    pub acceptance_digest: String,
    pub accepted_at_ms: u64,
    pub use_at_ms: u64,
    pub disposition: CurrentLinuxTpmImaAnchorDisposition,
    pub issues: Vec<CurrentLinuxTpmImaAnchorIssue>,
}

impl CurrentLinuxTpmImaAnchorReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DIGEST_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.provider_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.ledger_relative_anchor_policy_digest.as_str(),
            self.ledger_relative_anchor_qualification_digest.as_str(),
            self.head_authority_policy_digest.as_str(),
            self.ledger_id.as_str(),
            self.current_ledger_head_digest.as_str(),
            self.current_ledger_state_digest.as_str(),
            self.head_statement_digest.as_str(),
            self.currentness_attestation_digest.as_str(),
            self.acceptance_digest.as_str(),
        ] {
            push_field(&mut hasher, value);
        }
        hasher.update(&self.current_ledger_revision.to_le_bytes());
        hasher.update(&self.authority_sequence.to_le_bytes());
        hasher.update(&self.acceptance_revision.to_le_bytes());
        hasher.update(&self.accepted_at_ms.to_le_bytes());
        hasher.update(&self.use_at_ms.to_le_bytes());
        push_field(
            &mut hasher,
            match self.disposition {
                CurrentLinuxTpmImaAnchorDisposition::Invalid => "invalid",
                CurrentLinuxTpmImaAnchorDisposition::Blocked => "blocked",
                CurrentLinuxTpmImaAnchorDisposition::Qualified => "qualified",
            },
        );
        for issue in &self.issues {
            push_field(&mut hasher, &issue.code());
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentLinuxTpmImaRuntimeAnchor {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    ledger_relative_anchor_policy_digest: String,
    ledger_relative_anchor_qualification_digest: String,
    possession_policy_digest: String,
    pcr_binding_policy_digest: String,
    pcr_binding_qualification_digest: String,
    platform_qualification_digest: String,
    challenge_digest: String,
    ak_binding_digest: String,
    quote_qualification_digest: String,
    quote_artifact_digest: String,
    verification_receipt_digest: String,
    pcr_selection_digest: String,
    measurement_list_digest: String,
    matched_required_measurements: Vec<String>,
    pcr: u8,
    pcr_value: [u8; SHA256_LEN],
    acceptance_revision: u64,
    acceptance_digest: String,
    accepted_at_ms: u64,
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

impl CurrentLinuxTpmImaRuntimeAnchor {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn ledger_relative_anchor_policy_digest(&self) -> &str { &self.ledger_relative_anchor_policy_digest }
    pub fn ledger_relative_anchor_qualification_digest(&self) -> &str { &self.ledger_relative_anchor_qualification_digest }
    pub fn possession_policy_digest(&self) -> &str { &self.possession_policy_digest }
    pub fn pcr_binding_policy_digest(&self) -> &str { &self.pcr_binding_policy_digest }
    pub fn pcr_binding_qualification_digest(&self) -> &str { &self.pcr_binding_qualification_digest }
    pub fn platform_qualification_digest(&self) -> &str { &self.platform_qualification_digest }
    pub fn challenge_digest(&self) -> &str { &self.challenge_digest }
    pub fn ak_binding_digest(&self) -> &str { &self.ak_binding_digest }
    pub fn quote_qualification_digest(&self) -> &str { &self.quote_qualification_digest }
    pub fn quote_artifact_digest(&self) -> &str { &self.quote_artifact_digest }
    pub fn verification_receipt_digest(&self) -> &str { &self.verification_receipt_digest }
    pub fn pcr_selection_digest(&self) -> &str { &self.pcr_selection_digest }
    pub fn measurement_list_digest(&self) -> &str { &self.measurement_list_digest }
    pub fn matched_required_measurements(&self) -> &[String] { &self.matched_required_measurements }
    pub const fn pcr(&self) -> u8 { self.pcr }
    pub const fn pcr_value(&self) -> [u8; SHA256_LEN] { self.pcr_value }
    pub const fn acceptance_revision(&self) -> u64 { self.acceptance_revision }
    pub fn acceptance_digest(&self) -> &str { &self.acceptance_digest }
    pub const fn accepted_at_ms(&self) -> u64 { self.accepted_at_ms }
    pub fn ledger_id(&self) -> &str { &self.ledger_id }
    pub const fn current_ledger_revision(&self) -> u64 { self.current_ledger_revision }
    pub fn current_ledger_head_digest(&self) -> &str { &self.current_ledger_head_digest }
    pub fn current_ledger_state_digest(&self) -> &str { &self.current_ledger_state_digest }
    pub fn head_statement_digest(&self) -> &str { &self.head_statement_digest }
    pub fn currentness_attestation_digest(&self) -> &str { &self.currentness_attestation_digest }
    pub fn head_authority_policy_digest(&self) -> &str { &self.head_authority_policy_digest }
    pub const fn authority_sequence(&self) -> u64 { self.authority_sequence }
    pub const fn use_at_ms(&self) -> u64 { self.use_at_ms }
    pub const fn anti_replay_scope(&self) -> &'static str { CURRENT_LEDGER_ANTI_REPLAY_SCOPE_V1 }
    pub const fn establishes_authority_current_ledger_membership_at_use_time(&self) -> bool { true }
    pub const fn establishes_trusted_time_provenance(&self) -> bool { false }
    pub const fn establishes_authenticated_tracker_persistence(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentLinuxTpmImaAnchorQualification {
    pub report: CurrentLinuxTpmImaAnchorReport,
    verified: CurrentLinuxTpmImaRuntimeAnchor,
}

impl CurrentLinuxTpmImaAnchorQualification {
    pub fn verified(&self) -> &CurrentLinuxTpmImaRuntimeAnchor { &self.verified }
    pub fn into_verified(self) -> CurrentLinuxTpmImaRuntimeAnchor { self.verified }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

pub fn qualify_current_linux_tpm_ima_anchor(
    policy: &CurrentLinuxTpmImaAnchorPolicy,
    anchor: &LedgerRelativeLinuxTpmImaAnchor,
    current_head: &CurrentAcceptanceLedgerHead,
    ledger: &AttestationAcceptanceLedger,
) -> Result<CurrentLinuxTpmImaAnchorQualification, CurrentLinuxTpmImaAnchorReport> {
    let policy_digest = policy.canonical_digest();
    let mut report = CurrentLinuxTpmImaAnchorReport {
        schema_version: CURRENT_LINUX_TPM_IMA_ANCHOR_REPORT_SCHEMA_V1.into(),
        provider_id: policy.provider_id.clone(),
        policy_digest: policy_digest.clone(),
        ledger_relative_anchor_policy_digest: anchor.anchor_policy_digest().into(),
        ledger_relative_anchor_qualification_digest: anchor.qualification_digest().into(),
        head_authority_policy_digest: current_head.authority_policy_digest().into(),
        ledger_id: current_head.ledger_id().into(),
        current_ledger_revision: current_head.ledger_revision(),
        current_ledger_head_digest: current_head.ledger_head_digest().into(),
        current_ledger_state_digest: current_head.ledger_state_digest().into(),
        head_statement_digest: current_head.head_statement_digest().into(),
        currentness_attestation_digest: current_head.currentness_attestation_digest().into(),
        authority_sequence: current_head.authority_sequence(),
        acceptance_revision: anchor.acceptance_revision(),
        acceptance_digest: anchor.acceptance_digest().into(),
        accepted_at_ms: anchor.accepted_at_ms(),
        use_at_ms: current_head.use_at_ms(),
        disposition: CurrentLinuxTpmImaAnchorDisposition::Invalid,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report.issues.push(CurrentLinuxTpmImaAnchorIssue::InvalidPolicy);
        return Err(finalize_report(report));
    }
    if anchor.anchor_policy_digest() != policy.expected_ledger_relative_anchor_policy_digest {
        report.issues.push(CurrentLinuxTpmImaAnchorIssue::AnchorPolicyMismatch);
        return Err(finalize_report(report));
    }
    if current_head.authority_policy_digest() != policy.expected_head_authority_policy_digest {
        report.issues.push(CurrentLinuxTpmImaAnchorIssue::HeadAuthorityPolicyMismatch);
        return Err(finalize_report(report));
    }
    if current_head.ledger_id() != policy.expected_ledger_id {
        report.issues.push(CurrentLinuxTpmImaAnchorIssue::LedgerIdMismatch);
        return Err(finalize_report(report));
    }

    let snapshot = match snapshot_acceptance_ledger(ledger, policy.max_ledger_records) {
        Ok(snapshot) => snapshot,
        Err(reason) => {
            report.issues.push(CurrentLinuxTpmImaAnchorIssue::InvalidCurrentLedger { reason });
            return Err(finalize_report(report));
        }
    };
    if snapshot.revision() != current_head.ledger_revision() {
        report.issues.push(CurrentLinuxTpmImaAnchorIssue::CurrentLedgerRevisionMismatch {
            expected: current_head.ledger_revision(),
            observed: snapshot.revision(),
        });
        return Err(finalize_report(report));
    }
    if snapshot.head_digest() != current_head.ledger_head_digest() {
        report.issues.push(CurrentLinuxTpmImaAnchorIssue::CurrentLedgerHeadMismatch);
        return Err(finalize_report(report));
    }
    if snapshot.state_digest() != current_head.ledger_state_digest() {
        report.issues.push(CurrentLinuxTpmImaAnchorIssue::CurrentLedgerStateMismatch);
        return Err(finalize_report(report));
    }

    let revision = anchor.acceptance_revision();
    if revision == 0
        || anchor.ledger_records_before().saturating_add(1) != revision
        || revision > current_head.ledger_revision()
    {
        report.issues.push(CurrentLinuxTpmImaAnchorIssue::AnchorAcceptanceRevisionInvalid);
        return Err(finalize_report(report));
    }
    let Some(record) = ledger.records.get((revision - 1) as usize) else {
        report.issues.push(CurrentLinuxTpmImaAnchorIssue::AnchorAcceptanceNotInCurrentLedger);
        return Err(finalize_report(report));
    };
    if record.acceptance_digest() != anchor.acceptance_digest() {
        report.issues.push(CurrentLinuxTpmImaAnchorIssue::AnchorAcceptanceDigestMismatch);
        return Err(finalize_report(report));
    }
    if record.predecessor_acceptance_digest.as_deref() != anchor.ledger_head_before() {
        report.issues.push(CurrentLinuxTpmImaAnchorIssue::AnchorPredecessorMismatch);
        return Err(finalize_report(report));
    }
    if revision == 1 {
        if anchor.ledger_head_before().is_some() {
            report.issues.push(CurrentLinuxTpmImaAnchorIssue::AnchorPredecessorMismatch);
            return Err(finalize_report(report));
        }
    } else {
        let expected_predecessor = ledger.records[(revision - 2) as usize].acceptance_digest();
        if anchor.ledger_head_before() != Some(expected_predecessor.as_str()) {
            report.issues.push(CurrentLinuxTpmImaAnchorIssue::AnchorPredecessorMismatch);
            return Err(finalize_report(report));
        }
    }
    if record.accepted_at_ms != anchor.accepted_at_ms() {
        report.issues.push(CurrentLinuxTpmImaAnchorIssue::AnchorAcceptanceTimeMismatch);
        return Err(finalize_report(report));
    }
    if anchor.accepted_at_ms() > current_head.use_at_ms() {
        report.issues.push(CurrentLinuxTpmImaAnchorIssue::AnchorAfterCurrentnessUseTime);
        return Err(finalize_report(report));
    }

    report.disposition = CurrentLinuxTpmImaAnchorDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let qualification_digest = qualification_digest(&report_digest, &policy_digest, anchor, current_head);
    let verified = CurrentLinuxTpmImaRuntimeAnchor {
        qualification_digest,
        report_digest,
        policy_digest,
        ledger_relative_anchor_policy_digest: anchor.anchor_policy_digest().into(),
        ledger_relative_anchor_qualification_digest: anchor.qualification_digest().into(),
        possession_policy_digest: anchor.possession_policy_digest().into(),
        pcr_binding_policy_digest: anchor.pcr_binding_policy_digest().into(),
        pcr_binding_qualification_digest: anchor.pcr_binding_qualification_digest().into(),
        platform_qualification_digest: anchor.platform_qualification_digest().into(),
        challenge_digest: anchor.challenge_digest().into(),
        ak_binding_digest: anchor.ak_binding_digest().into(),
        quote_qualification_digest: anchor.quote_qualification_digest().into(),
        quote_artifact_digest: anchor.quote_artifact_digest().into(),
        verification_receipt_digest: anchor.verification_receipt_digest().into(),
        pcr_selection_digest: anchor.pcr_selection_digest().into(),
        measurement_list_digest: anchor.measurement_list_digest().into(),
        matched_required_measurements: anchor.matched_required_measurements().to_vec(),
        pcr: anchor.pcr(),
        pcr_value: anchor.pcr_value(),
        acceptance_revision: anchor.acceptance_revision(),
        acceptance_digest: anchor.acceptance_digest().into(),
        accepted_at_ms: anchor.accepted_at_ms(),
        ledger_id: current_head.ledger_id().into(),
        current_ledger_revision: current_head.ledger_revision(),
        current_ledger_head_digest: current_head.ledger_head_digest().into(),
        current_ledger_state_digest: current_head.ledger_state_digest().into(),
        head_statement_digest: current_head.head_statement_digest().into(),
        currentness_attestation_digest: current_head.currentness_attestation_digest().into(),
        head_authority_policy_digest: current_head.authority_policy_digest().into(),
        authority_sequence: current_head.authority_sequence(),
        use_at_ms: current_head.use_at_ms(),
    };
    Ok(CurrentLinuxTpmImaAnchorQualification { report, verified })
}

fn qualification_digest(
    report_digest: &str,
    policy_digest: &str,
    anchor: &LedgerRelativeLinuxTpmImaAnchor,
    current_head: &CurrentAcceptanceLedgerHead,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    for value in [
        report_digest,
        policy_digest,
        anchor.anchor_policy_digest(),
        anchor.qualification_digest(),
        anchor.possession_policy_digest(),
        anchor.pcr_binding_policy_digest(),
        anchor.pcr_binding_qualification_digest(),
        anchor.platform_qualification_digest(),
        anchor.challenge_digest(),
        anchor.ak_binding_digest(),
        anchor.quote_qualification_digest(),
        anchor.quote_artifact_digest(),
        anchor.verification_receipt_digest(),
        anchor.pcr_selection_digest(),
        anchor.measurement_list_digest(),
        anchor.acceptance_digest(),
        current_head.ledger_id(),
        current_head.ledger_head_digest(),
        current_head.ledger_state_digest(),
        current_head.head_statement_digest(),
        current_head.currentness_attestation_digest(),
        current_head.authority_policy_digest(),
    ] {
        push_field(&mut hasher, value);
    }
    hasher.update(&(anchor.matched_required_measurements().len() as u64).to_le_bytes());
    for measurement_id in anchor.matched_required_measurements() {
        push_field(&mut hasher, measurement_id);
    }
    hasher.update(&[anchor.pcr()]);
    hasher.update(&anchor.pcr_value());
    hasher.update(&anchor.acceptance_revision().to_le_bytes());
    hasher.update(&anchor.accepted_at_ms().to_le_bytes());
    hasher.update(&current_head.ledger_revision().to_le_bytes());
    hasher.update(&current_head.authority_sequence().to_le_bytes());
    hasher.update(&current_head.use_at_ms().to_le_bytes());
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn finalize_report(mut report: CurrentLinuxTpmImaAnchorReport) -> CurrentLinuxTpmImaAnchorReport {
    report.disposition = if report.issues.iter().any(CurrentLinuxTpmImaAnchorIssue::is_invalid) {
        CurrentLinuxTpmImaAnchorDisposition::Invalid
    } else {
        CurrentLinuxTpmImaAnchorDisposition::Blocked
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
    use ed25519_dalek::{Signer, SigningKey};
    use sha2::{Digest, Sha256};
    use symthaea_assurance_acceptance_ledger_head_currentness::{
        verify_acceptance_ledger_head_statement, verify_current_acceptance_ledger_head,
        AcceptanceLedgerHeadAuthorityKey, AcceptanceLedgerHeadAuthorityPolicy,
        AcceptanceLedgerHeadClaimScope, AcceptanceLedgerHeadCurrentnessAttestation,
        AcceptanceLedgerHeadStatement, AcceptanceLedgerHeadTracker,
        ACCEPTANCE_HEAD_AUTHORITY_POLICY_SCHEMA_V1, ACCEPTANCE_HEAD_CURRENTNESS_SCHEMA_V1,
        ACCEPTANCE_HEAD_STATEMENT_SCHEMA_V1,
    };
    use symthaea_assurance_linux_tpm_ima_ledger_anchor::{
        accept_linux_tpm_ima_anchor, LinuxTpmImaLedgerAnchorPolicy,
        LINUX_TPM_IMA_LEDGER_ANCHOR_POLICY_SCHEMA_V1,
    };
    use symthaea_assurance_tpm_ima_pcr_binding::{
        bind_verified_tpm_ima_pcr, TpmImaPcrBindingPolicy,
        TPM_IMA_PCR_BINDING_POLICY_SCHEMA_V1,
    };
    use symthaea_assurance_tpm2_attestation_possession::{
        pcr_selection_digest, AttestationAcceptanceRecord, AttestationChallenge,
        AttestationKeyBinding, AttestationPossessionPolicy, PcrBankSelection,
    };
    use symthaea_assurance_tpm2_checkquote_adapter::{
        verify_tpm2_quote, CheckquoteExecutionRequest, CheckquoteExecutor, RawTpm2QuoteBundle,
        Sha256PcrValue, ToolExecution, Tpm2CheckquotePolicy, CHECKQUOTE_POLICY_SCHEMA_V1,
    };
    use symthaea_assurance_tpm2_platform_qualification::{
        Tpm2FixedProperties, Tpm2PlatformQualificationRecord,
    };
    use symthaea_assurance_tpm2_possession_policy_binding::canonical_possession_policy_digest;
    use symthaea_linux_ima_replay::{
        verify_canonical_ima_sha256, ImaReplayPolicy, RequiredImaMeasurement,
        SupportedImaTemplate, IMA_REPLAY_POLICY_SCHEMA_V1,
    };

    const NONCE: [u8; 32] = [0xab; 32];
    const QUALIFIED_SIGNER: [u8; 34] = [0x5a; 34];
    const TPM2_GENERATED_VALUE: u32 = 0xff54_4347;
    const TPM2_ST_ATTEST_QUOTE: u16 = 0x8018;
    const TPM2_ALG_SHA256: u16 = 0x000b;

    #[derive(Clone)]
    struct FakeExecutor { digest: String }

    impl CheckquoteExecutor for FakeExecutor {
        fn executable_blake3(&self, _executable: &str) -> Result<String, String> {
            Ok(self.digest.clone())
        }
        fn execute_checkquote(
            &self,
            _executable: &str,
            _request: &CheckquoteExecutionRequest,
        ) -> Result<ToolExecution, String> {
            Ok(ToolExecution { exit_code: Some(0), stdout: Vec::new(), stderr: Vec::new() })
        }
    }

    fn d(label: &str) -> String { format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex()) }
    fn hex_lower(bytes: &[u8]) -> String { bytes.iter().map(|byte| format!("{byte:02x}")).collect() }
    fn challenge_nonce(label: &str) -> String { blake3::hash(label.as_bytes()).to_hex().to_string() }

    fn platform() -> Tpm2PlatformQualificationRecord {
        Tpm2PlatformQualificationRecord {
            schema_version: "1".into(), qualification_id: "platform:q1".into(),
            adapter_id: "adapter:tpm2:1".into(), trust_store_ref: "trust-store:tpm2:1".into(),
            tcti: "device:/dev/tpmrm0".into(), nv_index: 0x0150_0016,
            nv_name: "000bdeadbeef".into(), before_observation_digest: d("before"),
            after_observation_digest: d("after"), before_counter_value: 42,
            after_counter_value: 42,
            fixed_properties: Tpm2FixedProperties {
                family_indicator: 0x322e3000, specification_revision: 185,
                manufacturer: 0x49465800, firmware_version_1: 1, firmware_version_2: 2,
                raw_output_digest: d("getcap"),
            },
            getcap_blake3: d("getcap-bin"), runtime_subject_digest: d("runtime"),
            qualified_at_ms: 9_000, evidence_refs: vec!["audit:platform".into()],
        }
    }

    fn challenge() -> AttestationChallenge {
        AttestationChallenge {
            schema_version: "1".into(), challenge_id: "challenge:provider:1".into(),
            nonce_hex: hex_lower(&NONCE), issued_at_ms: 10_000, expires_at_ms: 11_000,
            verifier_ref: "verifier:remote-1".into(),
            pcr_selection: vec![PcrBankSelection { hash_alg: "sha256".into(), pcrs: vec![10] }],
            evidence_refs: vec!["request:provider:1".into()],
        }
    }

    fn ak(public: &[u8]) -> AttestationKeyBinding {
        AttestationKeyBinding {
            schema_version: "1".into(), ak_id: "ak:node-1".into(),
            ak_name_hex: "000b01020304".into(), ak_qualified_name_hex: hex_lower(&QUALIFIED_SIGNER),
            public_key_digest: d(std::str::from_utf8(public).unwrap()), name_alg: "sha256".into(),
            signing_scheme: "ecdsa-sha256".into(), fixed_tpm: true, restricted_signing: true,
            reviewed_at_ms: 9_500, evidence_refs: vec!["review:ak".into()],
        }
    }

    fn quote_policy(tool_digest: String) -> Tpm2CheckquotePolicy {
        Tpm2CheckquotePolicy {
            schema_version: CHECKQUOTE_POLICY_SCHEMA_V1.into(), adapter_id: "adapter:checkquote:provider".into(),
            verifier_ref: "verifier:remote-1".into(), checkquote_path: "/nix/store/example/bin/tpm2_checkquote".into(),
            expected_checkquote_blake3: tool_digest, hash_algorithm: "sha256".into(),
            max_public_bytes: 1024 * 1024, max_message_bytes: 1024 * 1024,
            max_signature_bytes: 1024 * 1024, max_tool_output_bytes: 1024 * 1024,
            evidence_refs: vec!["review:checkquote".into()],
        }
    }

    fn quote_message(pcr_value: [u8; 32]) -> Vec<u8> {
        let pcr_digest: [u8; 32] = Sha256::digest(pcr_value).into();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&TPM2_GENERATED_VALUE.to_be_bytes());
        bytes.extend_from_slice(&TPM2_ST_ATTEST_QUOTE.to_be_bytes());
        put_tpm2b(&mut bytes, &QUALIFIED_SIGNER); put_tpm2b(&mut bytes, &NONCE);
        bytes.extend_from_slice(&123u64.to_be_bytes()); bytes.extend_from_slice(&1u32.to_be_bytes());
        bytes.extend_from_slice(&2u32.to_be_bytes()); bytes.push(1); bytes.extend_from_slice(&7u64.to_be_bytes());
        bytes.extend_from_slice(&1u32.to_be_bytes()); bytes.extend_from_slice(&TPM2_ALG_SHA256.to_be_bytes());
        bytes.push(3); let mut bitmap = [0u8; 3]; bitmap[1] = 1u8 << 2; bytes.extend_from_slice(&bitmap);
        put_tpm2b(&mut bytes, &pcr_digest); bytes
    }

    fn put_tpm2b(target: &mut Vec<u8>, value: &[u8]) {
        target.extend_from_slice(&(value.len() as u16).to_be_bytes()); target.extend_from_slice(value);
    }

    fn ima_fixture() -> (ImaReplayPolicy, Vec<u8>, [u8; 32]) {
        let event: [u8; 32] = Sha256::digest(b"verifier-executable").into();
        let mut d_ng = b"sha256:\0".to_vec(); d_ng.extend_from_slice(&event);
        let n_ng = b"/nix/store/example/bin/verifier\0".to_vec();
        let mut data = Vec::new();
        data.extend_from_slice(&(d_ng.len() as u32).to_le_bytes()); data.extend_from_slice(&d_ng);
        data.extend_from_slice(&(n_ng.len() as u32).to_le_bytes()); data.extend_from_slice(&n_ng);
        let template_digest: [u8; 32] = Sha256::digest(&data).into();
        let mut record = Vec::new();
        record.extend_from_slice(&10u32.to_le_bytes()); record.extend_from_slice(&template_digest);
        record.extend_from_slice(&6u32.to_le_bytes()); record.extend_from_slice(b"ima-ng");
        record.extend_from_slice(&(data.len() as u32).to_le_bytes()); record.extend_from_slice(&data);
        let mut extend = Sha256::new(); extend.update([0u8; 32]); extend.update(template_digest);
        let final_pcr: [u8; 32] = extend.finalize().into();
        let policy = ImaReplayPolicy {
            schema_version: IMA_REPLAY_POLICY_SCHEMA_V1.into(), policy_id: "policy:ima:provider".into(),
            expected_pcr: 10, expected_initial_pcr: [0u8; 32], allowed_templates: vec![SupportedImaTemplate::ImaNg],
            required_measurements: vec![RequiredImaMeasurement { measurement_id: "verifier-executable".into(), event_sha256: event, event_name_sha256: None }],
            max_measurement_list_bytes: 1024 * 1024, max_records: 16,
            max_template_name_bytes: 64, max_template_data_bytes: 4096, max_field_bytes: 2048,
            evidence_refs: vec!["review:ima".into()],
        };
        (policy, record, final_pcr)
    }

    fn fixture() -> (
        LedgerRelativeLinuxTpmImaAnchor,
        AttestationAcceptanceLedger,
        CurrentAcceptanceLedgerHead,
        CurrentLinuxTpmImaAnchorPolicy,
    ) {
        let (ima_policy, bytes, final_pcr) = ima_fixture();
        let ima = verify_canonical_ima_sha256(&bytes, final_pcr, &ima_policy).unwrap();
        let platform = platform(); let challenge = challenge(); let public = b"ak-public"; let ak = ak(public);
        let tool_digest = d("tpm2-checkquote"); let quote_policy = quote_policy(tool_digest.clone());
        let bundle = RawTpm2QuoteBundle {
            quote_id: "quote:provider:1".into(), platform_qualification_digest: platform.qualification_digest(),
            ak_public: public.to_vec(), quote_message: quote_message(final_pcr), signature: b"fake-signature".to_vec(),
            pcr_values: vec![Sha256PcrValue { pcr: 10, value: final_pcr }], collected_at_ms: 10_200,
            evidence_refs: vec!["artifact:quote".into()],
        };
        let quote = verify_tpm2_quote(&quote_policy, &challenge, &ak, &bundle, 10_300, &FakeExecutor { digest: tool_digest.clone() }).unwrap();
        let binding_policy = TpmImaPcrBindingPolicy {
            schema_version: TPM_IMA_PCR_BINDING_POLICY_SCHEMA_V1.into(), binding_id: "binding:provider:1".into(), required_pcr: 10,
            expected_quote_policy_digest: quote_policy.canonical_digest().unwrap(), expected_ima_policy_digest: ima_policy.canonical_digest().unwrap(),
            evidence_refs: vec!["review:binding".into()],
        };
        let binding = bind_verified_tpm_ima_pcr(&binding_policy, quote.verified(), ima.verified()).unwrap();
        let possession = AttestationPossessionPolicy {
            schema_version: "1".into(), policy_id: "policy:possession:provider".into(),
            expected_platform_qualification_digest: platform.qualification_digest(), expected_ak_binding_digest: ak.binding_digest(),
            expected_pcr_selection_digest: pcr_selection_digest(&challenge.pcr_selection).unwrap(), expected_verifier_ref: challenge.verifier_ref.clone(),
            expected_verification_tool_digest: tool_digest, max_challenge_lifetime_ms: 2_000, max_quote_to_verification_ms: 500,
            require_fixed_tpm: true, require_restricted_signing: true, evidence_refs: vec!["review:possession".into()],
        };
        let anchor_policy = LinuxTpmImaLedgerAnchorPolicy {
            schema_version: LINUX_TPM_IMA_LEDGER_ANCHOR_POLICY_SCHEMA_V1.into(), anchor_id: "anchor:provider:1".into(),
            expected_pcr_binding_policy_digest: binding_policy.canonical_digest().unwrap(),
            expected_possession_policy_digest: canonical_possession_policy_digest(&possession).unwrap(), max_ledger_records: 100,
            evidence_refs: vec!["review:anchor".into()],
        };
        let mut ledger = AttestationAcceptanceLedger::default();
        let accepted = accept_linux_tpm_ima_anchor(
            &anchor_policy, &possession, &platform, &challenge, &ak, &quote, binding.verified(),
            10_400, 10_500, &mut ledger,
        ).unwrap();
        let anchor = accepted.into_verified();

        let predecessor = ledger.records.last().unwrap().acceptance_digest();
        let second = AttestationAcceptanceRecord {
            revision: 2, challenge_id: "challenge:later".into(), nonce_digest: d("nonce:later"),
            possession_record_digest: d("possession:later"), quote_artifact_digest: d("quote:later"),
            verification_receipt_id: "receipt:later".into(), accepted_at_ms: 10_600,
            predecessor_acceptance_digest: Some(predecessor),
        };
        ledger.records.push(second);

        let signing = SigningKey::from_bytes(&[44u8; 32]);
        let head_policy = AcceptanceLedgerHeadAuthorityPolicy {
            schema_version: ACCEPTANCE_HEAD_AUTHORITY_POLICY_SCHEMA_V1.into(), policy_id: "policy:head:provider".into(),
            ledger_id: "ledger:provider:1".into(), sequence: 1, issued_at_ms: 1, expires_at_ms: 100_000,
            max_ledger_records: 100,
            trusted_keys: vec![AcceptanceLedgerHeadAuthorityKey {
                key_id: "head-key:1".into(), public_key_ed25519_hex: hex::encode(signing.verifying_key().as_bytes()),
                valid_from_ms: 1, valid_until_ms: Some(100_000), revoked_at_ms: None,
                allowed_scopes: vec![AcceptanceLedgerHeadClaimScope::HeadStatement, AcceptanceLedgerHeadClaimScope::CurrentnessAttestation],
                evidence_refs: vec!["review:head-key".into()],
            }],
            evidence_refs: vec!["review:head-policy".into()],
        };
        let snapshot = snapshot_acceptance_ledger(&ledger, 100).unwrap();
        let mut statement = AcceptanceLedgerHeadStatement {
            schema_version: ACCEPTANCE_HEAD_STATEMENT_SCHEMA_V1.into(), ledger_id: head_policy.ledger_id.clone(),
            ledger_revision: snapshot.revision(), ledger_head_digest: snapshot.head_digest().into(), ledger_state_digest: snapshot.state_digest().into(),
            authority_sequence: 1, previous_head_statement_digest: None, issued_at_ms: 10_700,
            authority_policy_digest: head_policy.canonical_digest().unwrap(), signer_key_id: "head-key:1".into(),
            signer_public_key_ed25519_hex: hex::encode(signing.verifying_key().as_bytes()), evidence_refs: vec!["head:1".into()],
            signature_ed25519_hex: "00".repeat(64),
        };
        let bytes_to_sign = statement.canonical_unsigned_bytes().unwrap(); statement.signature_ed25519_hex = hex::encode(signing.sign(&bytes_to_sign).to_bytes());
        let verified_head = verify_acceptance_ledger_head_statement(&head_policy, &statement).into_verified().unwrap();
        let mut tracker = AcceptanceLedgerHeadTracker::default(); let tracked = tracker.observe(&verified_head, &ledger, &head_policy).unwrap();
        let nonce = challenge_nonce("provider-currentness");
        let mut currentness = AcceptanceLedgerHeadCurrentnessAttestation {
            schema_version: ACCEPTANCE_HEAD_CURRENTNESS_SCHEMA_V1.into(), head_statement_digest: tracked.statement_digest().into(),
            ledger_id: tracked.ledger_id().into(), ledger_revision: tracked.ledger_revision(), ledger_head_digest: tracked.ledger_head_digest().into(),
            ledger_state_digest: tracked.ledger_state_digest().into(), authority_sequence: tracked.authority_sequence(),
            authority_policy_digest: head_policy.canonical_digest().unwrap(), asserted_current_at_ms: 10_800,
            challenge_nonce_blake3_hex: nonce.clone(), signer_key_id: "head-key:1".into(),
            signer_public_key_ed25519_hex: hex::encode(signing.verifying_key().as_bytes()), evidence_refs: vec!["currentness:1".into()],
            signature_ed25519_hex: "00".repeat(64),
        };
        let current_bytes = currentness.canonical_unsigned_bytes().unwrap(); currentness.signature_ed25519_hex = hex::encode(signing.sign(&current_bytes).to_bytes());
        let current_head = verify_current_acceptance_ledger_head(&tracked, &head_policy, &currentness, &nonce, 10_800).into_current().unwrap();
        let policy = CurrentLinuxTpmImaAnchorPolicy {
            schema_version: CURRENT_LINUX_TPM_IMA_ANCHOR_POLICY_SCHEMA_V1.into(), provider_id: "provider:linux-tpm-ima:1".into(),
            expected_ledger_relative_anchor_policy_digest: anchor.anchor_policy_digest().into(),
            expected_head_authority_policy_digest: head_policy.canonical_digest().unwrap(), expected_ledger_id: head_policy.ledger_id.clone(),
            max_ledger_records: 100, evidence_refs: vec!["review:provider".into()],
        };
        (anchor, ledger, current_head, policy)
    }

    #[test]
    fn historical_acceptance_inside_current_authoritative_ledger_mints_anchor() {
        let (anchor, ledger, current_head, policy) = fixture();
        let qualified = qualify_current_linux_tpm_ima_anchor(&policy, &anchor, &current_head, &ledger).unwrap();
        assert_eq!(qualified.report.disposition, CurrentLinuxTpmImaAnchorDisposition::Qualified);
        assert_eq!(qualified.verified().acceptance_revision(), 1);
        assert_eq!(qualified.verified().current_ledger_revision(), 2);
        assert!(qualified.verified().establishes_authority_current_ledger_membership_at_use_time());
        assert!(!qualified.verified().establishes_authenticated_tracker_persistence());
        assert!(!qualified.verified().grants_physical_authority());
    }

    #[test]
    fn stale_shorter_ledger_cannot_satisfy_current_head() {
        let (anchor, mut ledger, current_head, policy) = fixture();
        ledger.records.pop();
        let report = qualify_current_linux_tpm_ima_anchor(&policy, &anchor, &current_head, &ledger).unwrap_err();
        assert!(report.issues.iter().any(|issue| matches!(issue, CurrentLinuxTpmImaAnchorIssue::CurrentLedgerRevisionMismatch { .. })));
    }

    #[test]
    fn head_authority_policy_substitution_is_blocked() {
        let (anchor, ledger, current_head, mut policy) = fixture();
        policy.expected_head_authority_policy_digest = d("other-head-policy");
        let report = qualify_current_linux_tpm_ima_anchor(&policy, &anchor, &current_head, &ledger).unwrap_err();
        assert!(report.issues.contains(&CurrentLinuxTpmImaAnchorIssue::HeadAuthorityPolicyMismatch));
    }
}
