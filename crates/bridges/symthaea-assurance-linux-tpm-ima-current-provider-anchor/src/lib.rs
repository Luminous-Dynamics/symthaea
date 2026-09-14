// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Current, freshness-bounded Linux TPM2 + IMA provider anchor.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_assurance_attestation_ledger_current_head::CurrentAcceptanceLedgerHead;
use symthaea_assurance_linux_tpm_ima_ledger_anchor::LedgerRelativeLinuxTpmImaAnchor;
use symthaea_assurance_tpm_ima_pcr_binding::VerifiedTpmImaPcrBinding;

pub const CURRENT_PROVIDER_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.linux-tpm-ima-current-provider-policy.v1";
pub const CURRENT_PROVIDER_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.linux-tpm-ima-current-provider-report.v1";
pub const CURRENT_PROVIDER_SCOPE_V1: &str =
    "current_authoritative_acceptance_head_plus_fresh_tpm_ima_measurement_not_continuous_runtime_v1";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.linux-tpm-ima-current-provider-policy.digest.v1\0";
const REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.linux-tpm-ima-current-provider-report.digest.v1\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.linux-tpm-ima-current-provider-qualification.digest.v1\0";
const MAX_TEXT_BYTES: usize = 512;
const MAX_EVIDENCE_REFS: usize = 128;
const MAX_FRESHNESS_WINDOW_MS: u64 = 86_400_000;
const SHA256_LEN: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CurrentLinuxTpmImaProviderPolicy {
    pub schema_version: String,
    pub provider_id: String,
    pub expected_ledger_id: String,
    pub expected_ledger_anchor_policy_digest: String,
    pub expected_pcr_binding_policy_digest: String,
    pub expected_ledger_head_authority_policy_digest: String,
    pub max_quote_age_at_use_ms: u64,
    pub max_acceptance_age_at_use_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl CurrentLinuxTpmImaProviderPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == CURRENT_PROVIDER_POLICY_SCHEMA_V1
            && canonical_text(&self.provider_id)
            && canonical_text(&self.expected_ledger_id)
            && valid_blake3_digest(&self.expected_ledger_anchor_policy_digest)
            && valid_blake3_digest(&self.expected_pcr_binding_policy_digest)
            && valid_blake3_digest(&self.expected_ledger_head_authority_policy_digest)
            && self.max_quote_age_at_use_ms > 0
            && self.max_quote_age_at_use_ms <= MAX_FRESHNESS_WINDOW_MS
            && self.max_acceptance_age_at_use_ms > 0
            && self.max_acceptance_age_at_use_ms <= MAX_FRESHNESS_WINDOW_MS
            && valid_refs(&self.evidence_refs)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DIGEST_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.provider_id.as_str(),
            self.expected_ledger_id.as_str(),
            self.expected_ledger_anchor_policy_digest.as_str(),
            self.expected_pcr_binding_policy_digest.as_str(),
            self.expected_ledger_head_authority_policy_digest.as_str(),
        ] {
            push_field(&mut hasher, value);
        }
        hasher.update(&self.max_quote_age_at_use_ms.to_le_bytes());
        hasher.update(&self.max_acceptance_age_at_use_ms.to_le_bytes());
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurrentProviderDisposition {
    Invalid,
    Blocked,
    Current,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurrentProviderIssue {
    InvalidPolicy,
    LedgerIdMismatch,
    LedgerAnchorPolicyMismatch,
    PcrBindingPolicyMismatch,
    LedgerHeadAuthorityPolicyMismatch,
    AcceptanceRevisionMismatch,
    AcceptanceDigestMismatch,
    PcrBindingQualificationMismatch,
    ChallengeMismatch,
    AkBindingMismatch,
    QuoteQualificationMismatch,
    QuoteArtifactMismatch,
    VerificationReceiptMismatch,
    PcrSelectionMismatch,
    MeasurementListMismatch,
    RequiredMeasurementsMismatch,
    PcrMismatch,
    UseBeforeQuoteCollection,
    UseBeforeQuoteVerification,
    UseBeforeAcceptance,
    QuoteTooOld { observed_ms: u64, maximum_ms: u64 },
    AcceptanceTooOld { observed_ms: u64, maximum_ms: u64 },
}

impl CurrentProviderIssue {
    fn is_invalid(&self) -> bool {
        matches!(self, Self::InvalidPolicy)
    }

    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::LedgerIdMismatch => "ledger-id-mismatch".into(),
            Self::LedgerAnchorPolicyMismatch => "ledger-anchor-policy-mismatch".into(),
            Self::PcrBindingPolicyMismatch => "pcr-binding-policy-mismatch".into(),
            Self::LedgerHeadAuthorityPolicyMismatch => "ledger-head-authority-policy-mismatch".into(),
            Self::AcceptanceRevisionMismatch => "acceptance-revision-mismatch".into(),
            Self::AcceptanceDigestMismatch => "acceptance-digest-mismatch".into(),
            Self::PcrBindingQualificationMismatch => "pcr-binding-qualification-mismatch".into(),
            Self::ChallengeMismatch => "challenge-mismatch".into(),
            Self::AkBindingMismatch => "ak-binding-mismatch".into(),
            Self::QuoteQualificationMismatch => "quote-qualification-mismatch".into(),
            Self::QuoteArtifactMismatch => "quote-artifact-mismatch".into(),
            Self::VerificationReceiptMismatch => "verification-receipt-mismatch".into(),
            Self::PcrSelectionMismatch => "pcr-selection-mismatch".into(),
            Self::MeasurementListMismatch => "measurement-list-mismatch".into(),
            Self::RequiredMeasurementsMismatch => "required-measurements-mismatch".into(),
            Self::PcrMismatch => "pcr-mismatch".into(),
            Self::UseBeforeQuoteCollection => "use-before-quote-collection".into(),
            Self::UseBeforeQuoteVerification => "use-before-quote-verification".into(),
            Self::UseBeforeAcceptance => "use-before-acceptance".into(),
            Self::QuoteTooOld { observed_ms, maximum_ms } => {
                format!("quote-too-old:{observed_ms}:{maximum_ms}")
            }
            Self::AcceptanceTooOld { observed_ms, maximum_ms } => {
                format!("acceptance-too-old:{observed_ms}:{maximum_ms}")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CurrentLinuxTpmImaProviderReport {
    pub schema_version: String,
    pub provider_id: String,
    pub policy_digest: Option<String>,
    pub ledger_id: String,
    pub ledger_anchor_qualification_digest: String,
    pub ledger_anchor_policy_digest: String,
    pub pcr_binding_qualification_digest: String,
    pub pcr_binding_policy_digest: String,
    pub ledger_head_statement_digest: String,
    pub ledger_head_currentness_digest: String,
    pub ledger_head_authority_policy_digest: String,
    pub acceptance_revision: u64,
    pub acceptance_digest: String,
    pub challenge_digest: String,
    pub ak_binding_digest: String,
    pub quote_artifact_digest: String,
    pub verification_receipt_digest: String,
    pub pcr_selection_digest: String,
    pub measurement_list_digest: String,
    pub pcr: u8,
    pub pcr_value: [u8; SHA256_LEN],
    pub quote_collected_at_ms: u64,
    pub quote_verified_at_ms: u64,
    pub accepted_at_ms: u64,
    pub use_at_ms: u64,
    pub disposition: CurrentProviderDisposition,
    pub issues: Vec<CurrentProviderIssue>,
}

impl CurrentLinuxTpmImaProviderReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DIGEST_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.provider_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.ledger_id.as_str(),
            self.ledger_anchor_qualification_digest.as_str(),
            self.ledger_anchor_policy_digest.as_str(),
            self.pcr_binding_qualification_digest.as_str(),
            self.pcr_binding_policy_digest.as_str(),
            self.ledger_head_statement_digest.as_str(),
            self.ledger_head_currentness_digest.as_str(),
            self.ledger_head_authority_policy_digest.as_str(),
            self.acceptance_digest.as_str(),
            self.challenge_digest.as_str(),
            self.ak_binding_digest.as_str(),
            self.quote_artifact_digest.as_str(),
            self.verification_receipt_digest.as_str(),
            self.pcr_selection_digest.as_str(),
            self.measurement_list_digest.as_str(),
        ] {
            push_field(&mut hasher, value);
        }
        hasher.update(&self.acceptance_revision.to_le_bytes());
        hasher.update(&[self.pcr]);
        hasher.update(&self.pcr_value);
        for time in [
            self.quote_collected_at_ms,
            self.quote_verified_at_ms,
            self.accepted_at_ms,
            self.use_at_ms,
        ] {
            hasher.update(&time.to_le_bytes());
        }
        push_field(
            &mut hasher,
            match self.disposition {
                CurrentProviderDisposition::Invalid => "invalid",
                CurrentProviderDisposition::Blocked => "blocked",
                CurrentProviderDisposition::Current => "current",
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

    pub const fn establishes_continuous_runtime_integrity(&self) -> bool {
        false
    }

    pub const fn establishes_trusted_time(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentLinuxTpmImaProviderAnchor {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    ledger_id: String,
    ledger_anchor_qualification_digest: String,
    pcr_binding_qualification_digest: String,
    ledger_head_statement_digest: String,
    ledger_head_currentness_digest: String,
    ledger_head_authority_policy_digest: String,
    acceptance_revision: u64,
    acceptance_digest: String,
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
    quote_collected_at_ms: u64,
    quote_verified_at_ms: u64,
    accepted_at_ms: u64,
    use_at_ms: u64,
}

impl CurrentLinuxTpmImaProviderAnchor {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn ledger_id(&self) -> &str { &self.ledger_id }
    pub fn ledger_anchor_qualification_digest(&self) -> &str { &self.ledger_anchor_qualification_digest }
    pub fn pcr_binding_qualification_digest(&self) -> &str { &self.pcr_binding_qualification_digest }
    pub fn ledger_head_statement_digest(&self) -> &str { &self.ledger_head_statement_digest }
    pub fn ledger_head_currentness_digest(&self) -> &str { &self.ledger_head_currentness_digest }
    pub fn ledger_head_authority_policy_digest(&self) -> &str { &self.ledger_head_authority_policy_digest }
    pub const fn acceptance_revision(&self) -> u64 { self.acceptance_revision }
    pub fn acceptance_digest(&self) -> &str { &self.acceptance_digest }
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
    pub const fn quote_collected_at_ms(&self) -> u64 { self.quote_collected_at_ms }
    pub const fn quote_verified_at_ms(&self) -> u64 { self.quote_verified_at_ms }
    pub const fn accepted_at_ms(&self) -> u64 { self.accepted_at_ms }
    pub const fn use_at_ms(&self) -> u64 { self.use_at_ms }
    pub const fn provider_scope(&self) -> &'static str { CURRENT_PROVIDER_SCOPE_V1 }
    pub const fn establishes_continuous_runtime_integrity(&self) -> bool { false }
    pub const fn establishes_trusted_time(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentLinuxTpmImaProviderQualification {
    pub report: CurrentLinuxTpmImaProviderReport,
    verified: CurrentLinuxTpmImaProviderAnchor,
}

impl CurrentLinuxTpmImaProviderQualification {
    pub fn verified(&self) -> &CurrentLinuxTpmImaProviderAnchor { &self.verified }
    pub fn into_verified(self) -> CurrentLinuxTpmImaProviderAnchor { self.verified }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

pub fn promote_current_linux_tpm_ima_provider(
    policy: &CurrentLinuxTpmImaProviderPolicy,
    ledger_anchor: &LedgerRelativeLinuxTpmImaAnchor,
    pcr_binding: &VerifiedTpmImaPcrBinding,
    current_head: &CurrentAcceptanceLedgerHead,
) -> Result<CurrentLinuxTpmImaProviderQualification, CurrentLinuxTpmImaProviderReport> {
    let policy_digest = policy.canonical_digest();
    let use_at_ms = current_head.use_at_ms();
    let mut report = CurrentLinuxTpmImaProviderReport {
        schema_version: CURRENT_PROVIDER_REPORT_SCHEMA_V1.into(),
        provider_id: policy.provider_id.clone(),
        policy_digest: policy_digest.clone(),
        ledger_id: current_head.ledger_id().to_string(),
        ledger_anchor_qualification_digest: ledger_anchor.qualification_digest().to_string(),
        ledger_anchor_policy_digest: ledger_anchor.anchor_policy_digest().to_string(),
        pcr_binding_qualification_digest: pcr_binding.qualification_digest().to_string(),
        pcr_binding_policy_digest: pcr_binding.policy_digest().to_string(),
        ledger_head_statement_digest: current_head.head_statement_digest().to_string(),
        ledger_head_currentness_digest: current_head.currentness_attestation_digest().to_string(),
        ledger_head_authority_policy_digest: current_head.authority_policy_digest().to_string(),
        acceptance_revision: current_head.acceptance_revision(),
        acceptance_digest: current_head.acceptance_digest().to_string(),
        challenge_digest: ledger_anchor.challenge_digest().to_string(),
        ak_binding_digest: ledger_anchor.ak_binding_digest().to_string(),
        quote_artifact_digest: ledger_anchor.quote_artifact_digest().to_string(),
        verification_receipt_digest: ledger_anchor.verification_receipt_digest().to_string(),
        pcr_selection_digest: ledger_anchor.pcr_selection_digest().to_string(),
        measurement_list_digest: ledger_anchor.measurement_list_digest().to_string(),
        pcr: ledger_anchor.pcr(),
        pcr_value: ledger_anchor.pcr_value(),
        quote_collected_at_ms: pcr_binding.quote_collected_at_ms(),
        quote_verified_at_ms: pcr_binding.quote_verified_at_ms(),
        accepted_at_ms: ledger_anchor.accepted_at_ms(),
        use_at_ms,
        disposition: CurrentProviderDisposition::Invalid,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report.issues.push(CurrentProviderIssue::InvalidPolicy);
        return Err(finalize_report(report));
    }
    if current_head.ledger_id() != policy.expected_ledger_id {
        report.issues.push(CurrentProviderIssue::LedgerIdMismatch);
    }
    if ledger_anchor.anchor_policy_digest() != policy.expected_ledger_anchor_policy_digest {
        report.issues.push(CurrentProviderIssue::LedgerAnchorPolicyMismatch);
    }
    if pcr_binding.policy_digest() != policy.expected_pcr_binding_policy_digest
        || ledger_anchor.pcr_binding_policy_digest() != policy.expected_pcr_binding_policy_digest
    {
        report.issues.push(CurrentProviderIssue::PcrBindingPolicyMismatch);
    }
    if current_head.authority_policy_digest() != policy.expected_ledger_head_authority_policy_digest {
        report
            .issues
            .push(CurrentProviderIssue::LedgerHeadAuthorityPolicyMismatch);
    }
    if ledger_anchor.acceptance_revision() != current_head.acceptance_revision() {
        report.issues.push(CurrentProviderIssue::AcceptanceRevisionMismatch);
    }
    if ledger_anchor.acceptance_digest() != current_head.acceptance_digest() {
        report.issues.push(CurrentProviderIssue::AcceptanceDigestMismatch);
    }
    if ledger_anchor.pcr_binding_qualification_digest() != pcr_binding.qualification_digest() {
        report
            .issues
            .push(CurrentProviderIssue::PcrBindingQualificationMismatch);
    }
    if ledger_anchor.challenge_digest() != pcr_binding.challenge_digest() {
        report.issues.push(CurrentProviderIssue::ChallengeMismatch);
    }
    if ledger_anchor.ak_binding_digest() != pcr_binding.ak_binding_digest() {
        report.issues.push(CurrentProviderIssue::AkBindingMismatch);
    }
    if ledger_anchor.quote_qualification_digest() != pcr_binding.quote_qualification_digest() {
        report.issues.push(CurrentProviderIssue::QuoteQualificationMismatch);
    }
    if ledger_anchor.quote_artifact_digest() != pcr_binding.quote_artifact_digest() {
        report.issues.push(CurrentProviderIssue::QuoteArtifactMismatch);
    }
    if ledger_anchor.verification_receipt_digest() != pcr_binding.verification_receipt_digest() {
        report
            .issues
            .push(CurrentProviderIssue::VerificationReceiptMismatch);
    }
    if ledger_anchor.pcr_selection_digest() != pcr_binding.pcr_selection_digest() {
        report.issues.push(CurrentProviderIssue::PcrSelectionMismatch);
    }
    if ledger_anchor.measurement_list_digest() != pcr_binding.measurement_list_digest() {
        report.issues.push(CurrentProviderIssue::MeasurementListMismatch);
    }
    if ledger_anchor.matched_required_measurements() != pcr_binding.matched_required_measurements() {
        report
            .issues
            .push(CurrentProviderIssue::RequiredMeasurementsMismatch);
    }
    if ledger_anchor.pcr() != pcr_binding.pcr()
        || ledger_anchor.pcr_value() != pcr_binding.pcr_value()
    {
        report.issues.push(CurrentProviderIssue::PcrMismatch);
    }

    if use_at_ms < pcr_binding.quote_collected_at_ms() {
        report.issues.push(CurrentProviderIssue::UseBeforeQuoteCollection);
    }
    if use_at_ms < pcr_binding.quote_verified_at_ms() {
        report.issues.push(CurrentProviderIssue::UseBeforeQuoteVerification);
    }
    if use_at_ms < ledger_anchor.accepted_at_ms() {
        report.issues.push(CurrentProviderIssue::UseBeforeAcceptance);
    }
    if use_at_ms >= pcr_binding.quote_collected_at_ms() {
        let observed = use_at_ms - pcr_binding.quote_collected_at_ms();
        if observed > policy.max_quote_age_at_use_ms {
            report.issues.push(CurrentProviderIssue::QuoteTooOld {
                observed_ms: observed,
                maximum_ms: policy.max_quote_age_at_use_ms,
            });
        }
    }
    if use_at_ms >= ledger_anchor.accepted_at_ms() {
        let observed = use_at_ms - ledger_anchor.accepted_at_ms();
        if observed > policy.max_acceptance_age_at_use_ms {
            report.issues.push(CurrentProviderIssue::AcceptanceTooOld {
                observed_ms: observed,
                maximum_ms: policy.max_acceptance_age_at_use_ms,
            });
        }
    }

    if !report.issues.is_empty() {
        return Err(finalize_report(report));
    }

    report.disposition = CurrentProviderDisposition::Current;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let qualification_digest = qualification_digest(
        &report_digest,
        &policy_digest,
        ledger_anchor,
        pcr_binding,
        current_head,
    );
    let verified = CurrentLinuxTpmImaProviderAnchor {
        qualification_digest,
        report_digest,
        policy_digest,
        ledger_id: current_head.ledger_id().to_string(),
        ledger_anchor_qualification_digest: ledger_anchor.qualification_digest().to_string(),
        pcr_binding_qualification_digest: pcr_binding.qualification_digest().to_string(),
        ledger_head_statement_digest: current_head.head_statement_digest().to_string(),
        ledger_head_currentness_digest: current_head.currentness_attestation_digest().to_string(),
        ledger_head_authority_policy_digest: current_head.authority_policy_digest().to_string(),
        acceptance_revision: current_head.acceptance_revision(),
        acceptance_digest: current_head.acceptance_digest().to_string(),
        platform_qualification_digest: ledger_anchor.platform_qualification_digest().to_string(),
        challenge_digest: ledger_anchor.challenge_digest().to_string(),
        ak_binding_digest: ledger_anchor.ak_binding_digest().to_string(),
        quote_artifact_digest: ledger_anchor.quote_artifact_digest().to_string(),
        verification_receipt_digest: ledger_anchor.verification_receipt_digest().to_string(),
        pcr_selection_digest: ledger_anchor.pcr_selection_digest().to_string(),
        measurement_list_digest: ledger_anchor.measurement_list_digest().to_string(),
        matched_required_measurements: ledger_anchor.matched_required_measurements().to_vec(),
        pcr: ledger_anchor.pcr(),
        pcr_value: ledger_anchor.pcr_value(),
        quote_collected_at_ms: pcr_binding.quote_collected_at_ms(),
        quote_verified_at_ms: pcr_binding.quote_verified_at_ms(),
        accepted_at_ms: ledger_anchor.accepted_at_ms(),
        use_at_ms,
    };
    Ok(CurrentLinuxTpmImaProviderQualification { report, verified })
}

fn qualification_digest(
    report_digest: &str,
    policy_digest: &str,
    ledger_anchor: &LedgerRelativeLinuxTpmImaAnchor,
    pcr_binding: &VerifiedTpmImaPcrBinding,
    current_head: &CurrentAcceptanceLedgerHead,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    for value in [
        report_digest,
        policy_digest,
        ledger_anchor.qualification_digest(),
        ledger_anchor.anchor_policy_digest(),
        pcr_binding.qualification_digest(),
        pcr_binding.policy_digest(),
        current_head.ledger_id(),
        current_head.head_statement_digest(),
        current_head.currentness_attestation_digest(),
        current_head.authority_policy_digest(),
        current_head.acceptance_digest(),
        ledger_anchor.platform_qualification_digest(),
        ledger_anchor.challenge_digest(),
        ledger_anchor.ak_binding_digest(),
        ledger_anchor.quote_artifact_digest(),
        ledger_anchor.verification_receipt_digest(),
        ledger_anchor.pcr_selection_digest(),
        ledger_anchor.measurement_list_digest(),
    ] {
        push_field(&mut hasher, value);
    }
    hasher.update(&(ledger_anchor.matched_required_measurements().len() as u64).to_le_bytes());
    for measurement_id in ledger_anchor.matched_required_measurements() {
        push_field(&mut hasher, measurement_id);
    }
    hasher.update(&current_head.acceptance_revision().to_le_bytes());
    hasher.update(&[ledger_anchor.pcr()]);
    hasher.update(&ledger_anchor.pcr_value());
    for time in [
        pcr_binding.quote_collected_at_ms(),
        pcr_binding.quote_verified_at_ms(),
        ledger_anchor.accepted_at_ms(),
        current_head.use_at_ms(),
    ] {
        hasher.update(&time.to_le_bytes());
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn finalize_report(mut report: CurrentLinuxTpmImaProviderReport) -> CurrentLinuxTpmImaProviderReport {
    report.disposition = if report.issues.iter().any(CurrentProviderIssue::is_invalid) {
        CurrentProviderDisposition::Invalid
    } else {
        CurrentProviderDisposition::Blocked
    };
    report
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT_BYTES
        && !value.chars().any(char::is_control)
}

fn valid_blake3_digest(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|digest| {
        digest.len() == 64
            && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            && !digest.bytes().any(|byte| byte.is_ascii_uppercase())
    })
}

fn valid_refs(values: &[String]) -> bool {
    !values.is_empty()
        && values.len() <= MAX_EVIDENCE_REFS
        && values.iter().all(|value| canonical_text(value))
        && values.iter().collect::<BTreeSet<_>>().len() == values.len()
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
