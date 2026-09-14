// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ledger-relative acceptance of verified Linux TPM2 + IMA evidence.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_assurance_tpm_ima_pcr_binding::VerifiedTpmImaPcrBinding;
use symthaea_assurance_tpm2_attestation_possession::{
    pcr_selection_digest, AttestationAcceptanceLedger, AttestationAcceptanceRecord,
    AttestationChallenge, AttestationKeyBinding, AttestationPossessionPolicy,
};
use symthaea_assurance_tpm2_checkquote_adapter::Tpm2QuoteQualification;
use symthaea_assurance_tpm2_platform_qualification::Tpm2PlatformQualificationRecord;
use symthaea_assurance_tpm2_possession_policy_binding::canonical_possession_policy_digest;

pub const LINUX_TPM_IMA_LEDGER_ANCHOR_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.linux-tpm-ima-ledger-anchor-policy.v1";
pub const LINUX_TPM_IMA_LEDGER_ANCHOR_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.linux-tpm-ima-ledger-anchor-report.v1";
pub const LEDGER_RELATIVE_ANTI_REPLAY_SCOPE_V1: &str =
    "structurally_valid_caller_supplied_acceptance_ledger_only_not_global_currentness_v1";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.linux-tpm-ima-ledger-anchor-policy.digest.v1\0";
const REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.linux-tpm-ima-ledger-anchor-report.digest.v1\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.linux-tpm-ima-ledger-anchor-qualification.digest.v1\0";
const MAX_TEXT_BYTES: usize = 512;
const MAX_EVIDENCE_REFS: usize = 128;
const MAX_POLICY_LEDGER_RECORDS: u64 = 1_000_000;
const SHA256_LEN: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinuxTpmImaLedgerAnchorPolicy {
    pub schema_version: String,
    pub anchor_id: String,
    pub expected_pcr_binding_policy_digest: String,
    pub expected_possession_policy_digest: String,
    pub max_ledger_records: u64,
    pub evidence_refs: Vec<String>,
}

impl LinuxTpmImaLedgerAnchorPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == LINUX_TPM_IMA_LEDGER_ANCHOR_POLICY_SCHEMA_V1
            && canonical_text(&self.anchor_id)
            && valid_blake3_digest(&self.expected_pcr_binding_policy_digest)
            && valid_blake3_digest(&self.expected_possession_policy_digest)
            && self.max_ledger_records > 0
            && self.max_ledger_records <= MAX_POLICY_LEDGER_RECORDS
            && valid_refs(&self.evidence_refs)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DIGEST_DOMAIN);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.anchor_id);
        push_field(&mut hasher, &self.expected_pcr_binding_policy_digest);
        push_field(&mut hasher, &self.expected_possession_policy_digest);
        hasher.update(&self.max_ledger_records.to_le_bytes());
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinuxTpmImaLedgerAnchorDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinuxTpmImaLedgerAnchorIssue {
    InvalidAnchorPolicy,
    InvalidPossessionPolicy,
    InvalidChallenge,
    InvalidAkBinding,
    InvalidLedgerState { reason: String },
    LedgerCapacityExceeded,
    PcrBindingPolicyMismatch,
    PossessionPolicyMismatch,
    ChallengeBindingMismatch,
    AkBindingMismatch,
    QuotePolicyMismatch,
    QuoteQualificationMismatch,
    QuoteArtifactMismatch,
    VerificationReceiptMismatch,
    PcrSelectionMismatch,
    PlatformBindingMismatch,
    PcrValueMismatch,
    AttestationAcceptanceRejected { reason: String },
}

impl LinuxTpmImaLedgerAnchorIssue {
    fn is_invalid(&self) -> bool {
        matches!(
            self,
            Self::InvalidAnchorPolicy
                | Self::InvalidPossessionPolicy
                | Self::InvalidChallenge
                | Self::InvalidAkBinding
                | Self::InvalidLedgerState { .. }
        )
    }

    fn code(&self) -> String {
        match self {
            Self::InvalidAnchorPolicy => "invalid-anchor-policy".into(),
            Self::InvalidPossessionPolicy => "invalid-possession-policy".into(),
            Self::InvalidChallenge => "invalid-challenge".into(),
            Self::InvalidAkBinding => "invalid-ak-binding".into(),
            Self::InvalidLedgerState { reason } => format!("invalid-ledger:{reason}"),
            Self::LedgerCapacityExceeded => "ledger-capacity-exceeded".into(),
            Self::PcrBindingPolicyMismatch => "pcr-binding-policy-mismatch".into(),
            Self::PossessionPolicyMismatch => "possession-policy-mismatch".into(),
            Self::ChallengeBindingMismatch => "challenge-binding-mismatch".into(),
            Self::AkBindingMismatch => "ak-binding-mismatch".into(),
            Self::QuotePolicyMismatch => "quote-policy-mismatch".into(),
            Self::QuoteQualificationMismatch => "quote-qualification-mismatch".into(),
            Self::QuoteArtifactMismatch => "quote-artifact-mismatch".into(),
            Self::VerificationReceiptMismatch => "verification-receipt-mismatch".into(),
            Self::PcrSelectionMismatch => "pcr-selection-mismatch".into(),
            Self::PlatformBindingMismatch => "platform-binding-mismatch".into(),
            Self::PcrValueMismatch => "pcr-value-mismatch".into(),
            Self::AttestationAcceptanceRejected { reason } => {
                format!("attestation-acceptance-rejected:{reason}")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinuxTpmImaLedgerAnchorReport {
    pub schema_version: String,
    pub anchor_id: String,
    pub anchor_policy_digest: Option<String>,
    pub possession_policy_digest: Option<String>,
    pub pcr_binding_policy_digest: String,
    pub pcr_binding_qualification_digest: String,
    pub platform_qualification_digest: String,
    pub challenge_digest: String,
    pub ak_binding_digest: String,
    pub quote_policy_digest: String,
    pub quote_qualification_digest: String,
    pub quote_artifact_digest: String,
    pub verification_receipt_digest: String,
    pub pcr_selection_digest: String,
    pub measurement_list_digest: String,
    pub matched_required_measurements: Vec<String>,
    pub pcr: u8,
    pub pcr_value: [u8; SHA256_LEN],
    pub ledger_records_before: u64,
    pub ledger_head_before: Option<String>,
    pub qualified_at_ms: u64,
    pub accepted_at_ms: u64,
    pub acceptance_revision: Option<u64>,
    pub acceptance_digest: Option<String>,
    pub disposition: LinuxTpmImaLedgerAnchorDisposition,
    pub issues: Vec<LinuxTpmImaLedgerAnchorIssue>,
}

impl LinuxTpmImaLedgerAnchorReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DIGEST_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.anchor_id.as_str(),
            self.anchor_policy_digest.as_deref().unwrap_or("-"),
            self.possession_policy_digest.as_deref().unwrap_or("-"),
            self.pcr_binding_policy_digest.as_str(),
            self.pcr_binding_qualification_digest.as_str(),
            self.platform_qualification_digest.as_str(),
            self.challenge_digest.as_str(),
            self.ak_binding_digest.as_str(),
            self.quote_policy_digest.as_str(),
            self.quote_qualification_digest.as_str(),
            self.quote_artifact_digest.as_str(),
            self.verification_receipt_digest.as_str(),
            self.pcr_selection_digest.as_str(),
            self.measurement_list_digest.as_str(),
            self.ledger_head_before.as_deref().unwrap_or("-"),
            self.acceptance_digest.as_deref().unwrap_or("-"),
        ] {
            push_field(&mut hasher, value);
        }
        hasher.update(&(self.matched_required_measurements.len() as u64).to_le_bytes());
        for measurement_id in &self.matched_required_measurements {
            push_field(&mut hasher, measurement_id);
        }
        hasher.update(&[self.pcr]);
        hasher.update(&self.pcr_value);
        hasher.update(&self.ledger_records_before.to_le_bytes());
        hasher.update(&self.qualified_at_ms.to_le_bytes());
        hasher.update(&self.accepted_at_ms.to_le_bytes());
        match self.acceptance_revision {
            Some(revision) => {
                hasher.update(&[1]);
                hasher.update(&revision.to_le_bytes());
            }
            None => {
                hasher.update(&[0]);
            }
        }
        push_field(
            &mut hasher,
            match self.disposition {
                LinuxTpmImaLedgerAnchorDisposition::Invalid => "invalid",
                LinuxTpmImaLedgerAnchorDisposition::Blocked => "blocked",
                LinuxTpmImaLedgerAnchorDisposition::Qualified => "qualified",
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

    pub const fn establishes_persistent_ledger_currentness(&self) -> bool {
        false
    }
}

/// Opaque acceptance capability. Replay protection is relative to the exact
/// structurally valid ledger state presented to `accept_linux_tpm_ima_anchor`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerRelativeLinuxTpmImaAnchor {
    qualification_digest: String,
    report_digest: String,
    anchor_policy_digest: String,
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
    ledger_records_before: u64,
    ledger_head_before: Option<String>,
    acceptance_revision: u64,
    acceptance_digest: String,
    qualified_at_ms: u64,
    accepted_at_ms: u64,
}

impl LedgerRelativeLinuxTpmImaAnchor {
    pub fn qualification_digest(&self) -> &str {
        &self.qualification_digest
    }
    pub fn report_digest(&self) -> &str {
        &self.report_digest
    }
    pub fn anchor_policy_digest(&self) -> &str {
        &self.anchor_policy_digest
    }
    pub fn possession_policy_digest(&self) -> &str {
        &self.possession_policy_digest
    }
    pub fn pcr_binding_policy_digest(&self) -> &str {
        &self.pcr_binding_policy_digest
    }
    pub fn pcr_binding_qualification_digest(&self) -> &str {
        &self.pcr_binding_qualification_digest
    }
    pub fn platform_qualification_digest(&self) -> &str {
        &self.platform_qualification_digest
    }
    pub fn challenge_digest(&self) -> &str {
        &self.challenge_digest
    }
    pub fn ak_binding_digest(&self) -> &str {
        &self.ak_binding_digest
    }
    pub fn quote_qualification_digest(&self) -> &str {
        &self.quote_qualification_digest
    }
    pub fn quote_artifact_digest(&self) -> &str {
        &self.quote_artifact_digest
    }
    pub fn verification_receipt_digest(&self) -> &str {
        &self.verification_receipt_digest
    }
    pub fn pcr_selection_digest(&self) -> &str {
        &self.pcr_selection_digest
    }
    pub fn measurement_list_digest(&self) -> &str {
        &self.measurement_list_digest
    }
    pub fn matched_required_measurements(&self) -> &[String] {
        &self.matched_required_measurements
    }
    pub const fn pcr(&self) -> u8 {
        self.pcr
    }
    pub const fn pcr_value(&self) -> [u8; SHA256_LEN] {
        self.pcr_value
    }
    pub const fn ledger_records_before(&self) -> u64 {
        self.ledger_records_before
    }
    pub fn ledger_head_before(&self) -> Option<&str> {
        self.ledger_head_before.as_deref()
    }
    pub const fn acceptance_revision(&self) -> u64 {
        self.acceptance_revision
    }
    pub fn acceptance_digest(&self) -> &str {
        &self.acceptance_digest
    }
    pub const fn qualified_at_ms(&self) -> u64 {
        self.qualified_at_ms
    }
    pub const fn accepted_at_ms(&self) -> u64 {
        self.accepted_at_ms
    }
    pub const fn anti_replay_scope(&self) -> &'static str {
        LEDGER_RELATIVE_ANTI_REPLAY_SCOPE_V1
    }
    pub const fn establishes_persistent_ledger_currentness(&self) -> bool {
        false
    }
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinuxTpmImaLedgerAnchorQualification {
    pub report: LinuxTpmImaLedgerAnchorReport,
    pub acceptance_record: AttestationAcceptanceRecord,
    verified: LedgerRelativeLinuxTpmImaAnchor,
}

impl LinuxTpmImaLedgerAnchorQualification {
    pub fn verified(&self) -> &LedgerRelativeLinuxTpmImaAnchor {
        &self.verified
    }
    pub fn into_verified(self) -> LedgerRelativeLinuxTpmImaAnchor {
        self.verified
    }
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[allow(clippy::too_many_arguments)]
pub fn accept_linux_tpm_ima_anchor(
    anchor_policy: &LinuxTpmImaLedgerAnchorPolicy,
    possession_policy: &AttestationPossessionPolicy,
    platform: &Tpm2PlatformQualificationRecord,
    challenge: &AttestationChallenge,
    ak: &AttestationKeyBinding,
    quote: &Tpm2QuoteQualification,
    pcr_binding: &VerifiedTpmImaPcrBinding,
    qualified_at_ms: u64,
    accepted_at_ms: u64,
    ledger: &mut AttestationAcceptanceLedger,
) -> Result<LinuxTpmImaLedgerAnchorQualification, LinuxTpmImaLedgerAnchorReport> {
    let anchor_policy_digest = anchor_policy.canonical_digest();
    let possession_policy_digest = canonical_possession_policy_digest(possession_policy);
    let platform_qualification_digest = platform.qualification_digest();
    let challenge_digest = challenge.challenge_digest();
    let ak_binding_digest = ak.binding_digest();
    let ledger_records_before = ledger.records.len() as u64;
    let ledger_head_before = ledger
        .records
        .last()
        .map(AttestationAcceptanceRecord::acceptance_digest);

    let mut report = LinuxTpmImaLedgerAnchorReport {
        schema_version: LINUX_TPM_IMA_LEDGER_ANCHOR_REPORT_SCHEMA_V1.into(),
        anchor_id: anchor_policy.anchor_id.clone(),
        anchor_policy_digest: anchor_policy_digest.clone(),
        possession_policy_digest: possession_policy_digest.clone(),
        pcr_binding_policy_digest: pcr_binding.policy_digest().to_string(),
        pcr_binding_qualification_digest: pcr_binding.qualification_digest().to_string(),
        platform_qualification_digest: platform_qualification_digest.clone(),
        challenge_digest: challenge_digest.clone(),
        ak_binding_digest: ak_binding_digest.clone(),
        quote_policy_digest: quote.verified().policy_digest().to_string(),
        quote_qualification_digest: quote.verified().qualification_digest().to_string(),
        quote_artifact_digest: quote.quote_artifacts.artifact_digest(),
        verification_receipt_digest: quote.verification_receipt.receipt_digest(),
        pcr_selection_digest: pcr_selection_digest(&challenge.pcr_selection).unwrap_or_default(),
        measurement_list_digest: pcr_binding.measurement_list_digest().to_string(),
        matched_required_measurements: pcr_binding.matched_required_measurements().to_vec(),
        pcr: pcr_binding.pcr(),
        pcr_value: pcr_binding.pcr_value(),
        ledger_records_before,
        ledger_head_before: ledger_head_before.clone(),
        qualified_at_ms,
        accepted_at_ms,
        acceptance_revision: None,
        acceptance_digest: None,
        disposition: LinuxTpmImaLedgerAnchorDisposition::Invalid,
        issues: Vec::new(),
    };

    if !anchor_policy.validate() {
        report
            .issues
            .push(LinuxTpmImaLedgerAnchorIssue::InvalidAnchorPolicy);
        return Err(finalize_report(report));
    }
    let Some(possession_policy_digest) = possession_policy_digest else {
        report
            .issues
            .push(LinuxTpmImaLedgerAnchorIssue::InvalidPossessionPolicy);
        return Err(finalize_report(report));
    };
    if !challenge.validate() {
        report
            .issues
            .push(LinuxTpmImaLedgerAnchorIssue::InvalidChallenge);
        return Err(finalize_report(report));
    }
    if !ak.validate() {
        report
            .issues
            .push(LinuxTpmImaLedgerAnchorIssue::InvalidAkBinding);
        return Err(finalize_report(report));
    }
    if let Err(reason) = validate_ledger_structure(ledger, anchor_policy.max_ledger_records) {
        report
            .issues
            .push(LinuxTpmImaLedgerAnchorIssue::InvalidLedgerState { reason });
        return Err(finalize_report(report));
    }
    if ledger_records_before >= anchor_policy.max_ledger_records {
        report
            .issues
            .push(LinuxTpmImaLedgerAnchorIssue::LedgerCapacityExceeded);
        return Err(finalize_report(report));
    }
    if pcr_binding.policy_digest() != anchor_policy.expected_pcr_binding_policy_digest {
        report
            .issues
            .push(LinuxTpmImaLedgerAnchorIssue::PcrBindingPolicyMismatch);
        return Err(finalize_report(report));
    }
    if possession_policy_digest != anchor_policy.expected_possession_policy_digest {
        report
            .issues
            .push(LinuxTpmImaLedgerAnchorIssue::PossessionPolicyMismatch);
        return Err(finalize_report(report));
    }
    if challenge_digest != pcr_binding.challenge_digest() {
        report
            .issues
            .push(LinuxTpmImaLedgerAnchorIssue::ChallengeBindingMismatch);
        return Err(finalize_report(report));
    }
    if ak_binding_digest != pcr_binding.ak_binding_digest() {
        report
            .issues
            .push(LinuxTpmImaLedgerAnchorIssue::AkBindingMismatch);
        return Err(finalize_report(report));
    }
    if quote.verified().policy_digest() != pcr_binding.quote_policy_digest() {
        report
            .issues
            .push(LinuxTpmImaLedgerAnchorIssue::QuotePolicyMismatch);
        return Err(finalize_report(report));
    }
    if quote.verified().qualification_digest() != pcr_binding.quote_qualification_digest() {
        report
            .issues
            .push(LinuxTpmImaLedgerAnchorIssue::QuoteQualificationMismatch);
        return Err(finalize_report(report));
    }
    if quote.verified().quote_artifact_digest() != pcr_binding.quote_artifact_digest()
        || quote.quote_artifacts.artifact_digest() != pcr_binding.quote_artifact_digest()
    {
        report
            .issues
            .push(LinuxTpmImaLedgerAnchorIssue::QuoteArtifactMismatch);
        return Err(finalize_report(report));
    }
    if quote.verified().verification_receipt_digest()
        != pcr_binding.verification_receipt_digest()
        || quote.verification_receipt.receipt_digest() != pcr_binding.verification_receipt_digest()
    {
        report
            .issues
            .push(LinuxTpmImaLedgerAnchorIssue::VerificationReceiptMismatch);
        return Err(finalize_report(report));
    }
    let Some(selection_digest) = pcr_selection_digest(&challenge.pcr_selection) else {
        report
            .issues
            .push(LinuxTpmImaLedgerAnchorIssue::PcrSelectionMismatch);
        return Err(finalize_report(report));
    };
    if selection_digest != pcr_binding.pcr_selection_digest()
        || quote.verified().pcr_selection_digest() != pcr_binding.pcr_selection_digest()
    {
        report
            .issues
            .push(LinuxTpmImaLedgerAnchorIssue::PcrSelectionMismatch);
        return Err(finalize_report(report));
    }
    if quote.quote_artifacts.platform_qualification_digest != platform_qualification_digest {
        report
            .issues
            .push(LinuxTpmImaLedgerAnchorIssue::PlatformBindingMismatch);
        return Err(finalize_report(report));
    }
    if quote.verified().pcr_value(pcr_binding.pcr()) != Some(pcr_binding.pcr_value()) {
        report
            .issues
            .push(LinuxTpmImaLedgerAnchorIssue::PcrValueMismatch);
        return Err(finalize_report(report));
    }

    // This is the only mutation point. Every independent cross-object binding
    // check above has already passed, so a later semantic failure cannot leave
    // a consumed challenge without an anchor capability.
    let acceptance_record = match ledger.accept(
        possession_policy,
        platform,
        challenge,
        ak,
        &quote.quote_artifacts,
        &quote.verification_receipt,
        qualified_at_ms,
        accepted_at_ms,
    ) {
        Ok(record) => record,
        Err(error) => {
            report
                .issues
                .push(LinuxTpmImaLedgerAnchorIssue::AttestationAcceptanceRejected {
                    reason: format!("{error:?}"),
                });
            return Err(finalize_report(report));
        }
    };

    let acceptance_digest = acceptance_record.acceptance_digest();
    report.acceptance_revision = Some(acceptance_record.revision);
    report.acceptance_digest = Some(acceptance_digest.clone());
    report.disposition = LinuxTpmImaLedgerAnchorDisposition::Qualified;

    let report_digest = report.canonical_digest();
    let anchor_policy_digest = anchor_policy_digest.expect("validated policy has digest");
    let qualification_digest = qualification_digest(
        &report_digest,
        &anchor_policy_digest,
        &possession_policy_digest,
        pcr_binding,
        &platform_qualification_digest,
        &acceptance_record,
        &acceptance_digest,
        qualified_at_ms,
        accepted_at_ms,
    );
    let verified = LedgerRelativeLinuxTpmImaAnchor {
        qualification_digest,
        report_digest,
        anchor_policy_digest,
        possession_policy_digest,
        pcr_binding_policy_digest: pcr_binding.policy_digest().to_string(),
        pcr_binding_qualification_digest: pcr_binding.qualification_digest().to_string(),
        platform_qualification_digest,
        challenge_digest,
        ak_binding_digest,
        quote_qualification_digest: quote.verified().qualification_digest().to_string(),
        quote_artifact_digest: pcr_binding.quote_artifact_digest().to_string(),
        verification_receipt_digest: pcr_binding.verification_receipt_digest().to_string(),
        pcr_selection_digest: pcr_binding.pcr_selection_digest().to_string(),
        measurement_list_digest: pcr_binding.measurement_list_digest().to_string(),
        matched_required_measurements: pcr_binding.matched_required_measurements().to_vec(),
        pcr: pcr_binding.pcr(),
        pcr_value: pcr_binding.pcr_value(),
        ledger_records_before,
        ledger_head_before,
        acceptance_revision: acceptance_record.revision,
        acceptance_digest,
        qualified_at_ms,
        accepted_at_ms,
    };

    Ok(LinuxTpmImaLedgerAnchorQualification {
        report,
        acceptance_record,
        verified,
    })
}

fn validate_ledger_structure(
    ledger: &AttestationAcceptanceLedger,
    max_records: u64,
) -> Result<(), String> {
    if ledger.records.len() as u64 > max_records {
        return Err("record-count-exceeds-policy".into());
    }

    let mut challenge_ids = BTreeSet::new();
    let mut nonce_digests = BTreeSet::new();
    let mut quote_digests = BTreeSet::new();
    let mut receipt_ids = BTreeSet::new();
    let mut previous_digest: Option<String> = None;
    let mut previous_accepted_at: Option<u64> = None;

    for (index, record) in ledger.records.iter().enumerate() {
        let expected_revision = index as u64 + 1;
        if record.revision != expected_revision {
            return Err(format!("revision:{expected_revision}:{}", record.revision));
        }
        if !canonical_text(&record.challenge_id) || !canonical_text(&record.verification_receipt_id)
        {
            return Err(format!("invalid-text:{expected_revision}"));
        }
        for digest in [
            record.nonce_digest.as_str(),
            record.possession_record_digest.as_str(),
            record.quote_artifact_digest.as_str(),
        ] {
            if !valid_blake3_digest(digest) {
                return Err(format!("invalid-digest:{expected_revision}"));
            }
        }
        match (&previous_digest, &record.predecessor_acceptance_digest) {
            (None, None) => {}
            (Some(expected), Some(observed)) if expected == observed => {}
            _ => return Err(format!("predecessor:{expected_revision}")),
        }
        if record
            .predecessor_acceptance_digest
            .as_deref()
            .is_some_and(|digest| !valid_blake3_digest(digest))
        {
            return Err(format!("invalid-predecessor-digest:{expected_revision}"));
        }
        if previous_accepted_at.is_some_and(|time| record.accepted_at_ms < time) {
            return Err(format!("time-regression:{expected_revision}"));
        }
        if !challenge_ids.insert(record.challenge_id.clone()) {
            return Err(format!("duplicate-challenge:{expected_revision}"));
        }
        if !nonce_digests.insert(record.nonce_digest.clone()) {
            return Err(format!("duplicate-nonce:{expected_revision}"));
        }
        if !quote_digests.insert(record.quote_artifact_digest.clone()) {
            return Err(format!("duplicate-quote:{expected_revision}"));
        }
        if !receipt_ids.insert(record.verification_receipt_id.clone()) {
            return Err(format!("duplicate-receipt:{expected_revision}"));
        }

        previous_digest = Some(record.acceptance_digest());
        previous_accepted_at = Some(record.accepted_at_ms);
    }

    Ok(())
}

fn qualification_digest(
    report_digest: &str,
    anchor_policy_digest: &str,
    possession_policy_digest: &str,
    pcr_binding: &VerifiedTpmImaPcrBinding,
    platform_qualification_digest: &str,
    acceptance_record: &AttestationAcceptanceRecord,
    acceptance_digest: &str,
    qualified_at_ms: u64,
    accepted_at_ms: u64,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    for value in [
        report_digest,
        anchor_policy_digest,
        possession_policy_digest,
        pcr_binding.policy_digest(),
        pcr_binding.qualification_digest(),
        platform_qualification_digest,
        pcr_binding.challenge_digest(),
        pcr_binding.ak_binding_digest(),
        pcr_binding.quote_qualification_digest(),
        pcr_binding.quote_artifact_digest(),
        pcr_binding.verification_receipt_digest(),
        pcr_binding.pcr_selection_digest(),
        pcr_binding.measurement_list_digest(),
        acceptance_digest,
    ] {
        push_field(&mut hasher, value);
    }
    hasher.update(&(pcr_binding.matched_required_measurements().len() as u64).to_le_bytes());
    for measurement_id in pcr_binding.matched_required_measurements() {
        push_field(&mut hasher, measurement_id);
    }
    hasher.update(&[pcr_binding.pcr()]);
    hasher.update(&pcr_binding.pcr_value());
    hasher.update(&acceptance_record.revision.to_le_bytes());
    push_optional_field(
        &mut hasher,
        acceptance_record.predecessor_acceptance_digest.as_deref(),
    );
    hasher.update(&qualified_at_ms.to_le_bytes());
    hasher.update(&accepted_at_ms.to_le_bytes());
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn finalize_report(mut report: LinuxTpmImaLedgerAnchorReport) -> LinuxTpmImaLedgerAnchorReport {
    report.disposition = if report.issues.iter().any(LinuxTpmImaLedgerAnchorIssue::is_invalid) {
        LinuxTpmImaLedgerAnchorDisposition::Invalid
    } else {
        LinuxTpmImaLedgerAnchorDisposition::Blocked
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

fn push_optional_field(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            push_field(hasher, value);
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn push_sorted_refs(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut refs = refs.to_vec();
    refs.sort();
    hasher.update(&(refs.len() as u64).to_le_bytes());
    for reference in refs {
        push_field(hasher, &reference);
    }
}
