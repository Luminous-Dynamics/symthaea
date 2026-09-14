// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pure capability theorem binding a verified TPM2 PCR to Linux IMA replay.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_assurance_tpm2_checkquote_adapter::VerifiedTpm2Quote;
use symthaea_linux_ima_replay::VerifiedImaReplay;

pub const TPM_IMA_PCR_BINDING_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.tpm-ima-pcr-binding-policy.v1";
pub const TPM_IMA_PCR_BINDING_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.tpm-ima-pcr-binding-report.v1";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm-ima-pcr-binding-policy.digest.v1\0";
const REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm-ima-pcr-binding-report.digest.v1\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm-ima-pcr-binding-qualification.digest.v1\0";
const MAX_TEXT_BYTES: usize = 512;
const MAX_EVIDENCE_REFS: usize = 128;
const SHA256_LEN: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TpmImaPcrBindingPolicy {
    pub schema_version: String,
    pub binding_id: String,
    pub required_pcr: u8,
    pub expected_quote_policy_digest: String,
    pub expected_ima_policy_digest: String,
    pub evidence_refs: Vec<String>,
}

impl TpmImaPcrBindingPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == TPM_IMA_PCR_BINDING_POLICY_SCHEMA_V1
            && canonical_text(&self.binding_id)
            && self.required_pcr <= 23
            && valid_blake3_digest(&self.expected_quote_policy_digest)
            && valid_blake3_digest(&self.expected_ima_policy_digest)
            && valid_refs(&self.evidence_refs)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DIGEST_DOMAIN);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.binding_id);
        hasher.update(&[self.required_pcr]);
        push_field(&mut hasher, &self.expected_quote_policy_digest);
        push_field(&mut hasher, &self.expected_ima_policy_digest);
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TpmImaPcrBindingDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TpmImaPcrBindingIssue {
    InvalidPolicy,
    QuotePolicyMismatch,
    ImaPolicyMismatch,
    ImaPcrMismatch { expected: u8, observed: u32 },
    QuoteMissingRequiredPcr { pcr: u8 },
    PcrValueMismatch,
}

impl TpmImaPcrBindingIssue {
    fn is_invalid(&self) -> bool {
        matches!(self, Self::InvalidPolicy)
    }

    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::QuotePolicyMismatch => "quote-policy-mismatch".into(),
            Self::ImaPolicyMismatch => "ima-policy-mismatch".into(),
            Self::ImaPcrMismatch { expected, observed } => {
                format!("ima-pcr-mismatch:{expected}:{observed}")
            }
            Self::QuoteMissingRequiredPcr { pcr } => format!("quote-missing-pcr:{pcr}"),
            Self::PcrValueMismatch => "pcr-value-mismatch".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TpmImaPcrBindingReport {
    pub schema_version: String,
    pub binding_id: String,
    pub policy_digest: Option<String>,
    pub quote_policy_digest: String,
    pub ima_policy_digest: String,
    pub quote_qualification_digest: String,
    pub ima_qualification_digest: String,
    pub challenge_digest: String,
    pub ak_binding_digest: String,
    pub quote_artifact_digest: String,
    pub verification_receipt_digest: String,
    pub pcr_selection_digest: String,
    pub measurement_list_digest: String,
    pub matched_required_measurements: Vec<String>,
    pub ima_record_count: u32,
    pub ima_selected_pcr_record_count: u32,
    pub required_pcr: u8,
    pub quote_pcr_value: Option<[u8; SHA256_LEN]>,
    pub ima_pcr_value: [u8; SHA256_LEN],
    pub quote_collected_at_ms: u64,
    pub quote_verified_at_ms: u64,
    pub disposition: TpmImaPcrBindingDisposition,
    pub issues: Vec<TpmImaPcrBindingIssue>,
}

impl TpmImaPcrBindingReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DIGEST_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.binding_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.quote_policy_digest.as_str(),
            self.ima_policy_digest.as_str(),
            self.quote_qualification_digest.as_str(),
            self.ima_qualification_digest.as_str(),
            self.challenge_digest.as_str(),
            self.ak_binding_digest.as_str(),
            self.quote_artifact_digest.as_str(),
            self.verification_receipt_digest.as_str(),
            self.pcr_selection_digest.as_str(),
            self.measurement_list_digest.as_str(),
        ] {
            push_field(&mut hasher, value);
        }
        hasher.update(&(self.matched_required_measurements.len() as u64).to_le_bytes());
        for measurement_id in &self.matched_required_measurements {
            push_field(&mut hasher, measurement_id);
        }
        hasher.update(&self.ima_record_count.to_le_bytes());
        hasher.update(&self.ima_selected_pcr_record_count.to_le_bytes());
        hasher.update(&[self.required_pcr]);
        match self.quote_pcr_value {
            Some(value) => {
                hasher.update(&[1]);
                hasher.update(&value);
            }
            None => {
                hasher.update(&[0]);
            }
        }
        hasher.update(&self.ima_pcr_value);
        hasher.update(&self.quote_collected_at_ms.to_le_bytes());
        hasher.update(&self.quote_verified_at_ms.to_le_bytes());
        push_field(
            &mut hasher,
            match self.disposition {
                TpmImaPcrBindingDisposition::Invalid => "invalid",
                TpmImaPcrBindingDisposition::Blocked => "blocked",
                TpmImaPcrBindingDisposition::Qualified => "qualified",
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
pub struct VerifiedTpmImaPcrBinding {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    quote_policy_digest: String,
    ima_policy_digest: String,
    quote_qualification_digest: String,
    ima_qualification_digest: String,
    challenge_digest: String,
    ak_binding_digest: String,
    quote_artifact_digest: String,
    verification_receipt_digest: String,
    pcr_selection_digest: String,
    measurement_list_digest: String,
    matched_required_measurements: Vec<String>,
    ima_record_count: u32,
    ima_selected_pcr_record_count: u32,
    pcr: u8,
    pcr_value: [u8; SHA256_LEN],
    quote_collected_at_ms: u64,
    quote_verified_at_ms: u64,
}

impl VerifiedTpmImaPcrBinding {
    pub fn qualification_digest(&self) -> &str {
        &self.qualification_digest
    }

    pub fn report_digest(&self) -> &str {
        &self.report_digest
    }

    pub fn policy_digest(&self) -> &str {
        &self.policy_digest
    }

    pub fn quote_policy_digest(&self) -> &str {
        &self.quote_policy_digest
    }

    pub fn ima_policy_digest(&self) -> &str {
        &self.ima_policy_digest
    }

    pub fn quote_qualification_digest(&self) -> &str {
        &self.quote_qualification_digest
    }

    pub fn ima_qualification_digest(&self) -> &str {
        &self.ima_qualification_digest
    }

    pub fn challenge_digest(&self) -> &str {
        &self.challenge_digest
    }

    pub fn ak_binding_digest(&self) -> &str {
        &self.ak_binding_digest
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

    pub const fn ima_record_count(&self) -> u32 {
        self.ima_record_count
    }

    pub const fn ima_selected_pcr_record_count(&self) -> u32 {
        self.ima_selected_pcr_record_count
    }

    pub const fn pcr(&self) -> u8 {
        self.pcr
    }

    pub const fn pcr_value(&self) -> [u8; SHA256_LEN] {
        self.pcr_value
    }

    pub const fn quote_collected_at_ms(&self) -> u64 {
        self.quote_collected_at_ms
    }

    pub const fn quote_verified_at_ms(&self) -> u64 {
        self.quote_verified_at_ms
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpmImaPcrBindingQualification {
    pub report: TpmImaPcrBindingReport,
    verified: VerifiedTpmImaPcrBinding,
}

impl TpmImaPcrBindingQualification {
    pub fn verified(&self) -> &VerifiedTpmImaPcrBinding {
        &self.verified
    }

    pub fn into_verified(self) -> VerifiedTpmImaPcrBinding {
        self.verified
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

pub fn bind_verified_tpm_ima_pcr(
    policy: &TpmImaPcrBindingPolicy,
    quote: &VerifiedTpm2Quote,
    ima: &VerifiedImaReplay,
) -> Result<TpmImaPcrBindingQualification, TpmImaPcrBindingReport> {
    let policy_digest = policy.canonical_digest();
    let mut report = TpmImaPcrBindingReport {
        schema_version: TPM_IMA_PCR_BINDING_REPORT_SCHEMA_V1.into(),
        binding_id: policy.binding_id.clone(),
        policy_digest: policy_digest.clone(),
        quote_policy_digest: quote.policy_digest().to_string(),
        ima_policy_digest: ima.policy_digest().to_string(),
        quote_qualification_digest: quote.qualification_digest().to_string(),
        ima_qualification_digest: ima.qualification_digest().to_string(),
        challenge_digest: quote.challenge_digest().to_string(),
        ak_binding_digest: quote.ak_binding_digest().to_string(),
        quote_artifact_digest: quote.quote_artifact_digest().to_string(),
        verification_receipt_digest: quote.verification_receipt_digest().to_string(),
        pcr_selection_digest: quote.pcr_selection_digest().to_string(),
        measurement_list_digest: ima.measurement_list_digest().to_string(),
        matched_required_measurements: ima.matched_required_measurements().to_vec(),
        ima_record_count: ima.record_count(),
        ima_selected_pcr_record_count: ima.selected_pcr_record_count(),
        required_pcr: policy.required_pcr,
        quote_pcr_value: None,
        ima_pcr_value: ima.final_pcr(),
        quote_collected_at_ms: quote.collected_at_ms(),
        quote_verified_at_ms: quote.verified_at_ms(),
        disposition: TpmImaPcrBindingDisposition::Invalid,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report.issues.push(TpmImaPcrBindingIssue::InvalidPolicy);
        return Err(finalize_report(report));
    }
    if quote.policy_digest() != policy.expected_quote_policy_digest {
        report
            .issues
            .push(TpmImaPcrBindingIssue::QuotePolicyMismatch);
        return Err(finalize_report(report));
    }
    if ima.policy_digest() != policy.expected_ima_policy_digest {
        report
            .issues
            .push(TpmImaPcrBindingIssue::ImaPolicyMismatch);
        return Err(finalize_report(report));
    }
    if ima.expected_pcr() != u32::from(policy.required_pcr) {
        report.issues.push(TpmImaPcrBindingIssue::ImaPcrMismatch {
            expected: policy.required_pcr,
            observed: ima.expected_pcr(),
        });
        return Err(finalize_report(report));
    }

    let Some(quote_pcr_value) = quote.pcr_value(policy.required_pcr) else {
        report
            .issues
            .push(TpmImaPcrBindingIssue::QuoteMissingRequiredPcr {
                pcr: policy.required_pcr,
            });
        return Err(finalize_report(report));
    };
    report.quote_pcr_value = Some(quote_pcr_value);

    if quote_pcr_value != ima.final_pcr() {
        report
            .issues
            .push(TpmImaPcrBindingIssue::PcrValueMismatch);
        return Err(finalize_report(report));
    }

    report.disposition = TpmImaPcrBindingDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let qualification_digest = qualification_digest(&report_digest, &policy_digest, &report);
    let verified = VerifiedTpmImaPcrBinding {
        qualification_digest,
        report_digest,
        policy_digest,
        quote_policy_digest: report.quote_policy_digest.clone(),
        ima_policy_digest: report.ima_policy_digest.clone(),
        quote_qualification_digest: report.quote_qualification_digest.clone(),
        ima_qualification_digest: report.ima_qualification_digest.clone(),
        challenge_digest: report.challenge_digest.clone(),
        ak_binding_digest: report.ak_binding_digest.clone(),
        quote_artifact_digest: report.quote_artifact_digest.clone(),
        verification_receipt_digest: report.verification_receipt_digest.clone(),
        pcr_selection_digest: report.pcr_selection_digest.clone(),
        measurement_list_digest: report.measurement_list_digest.clone(),
        matched_required_measurements: report.matched_required_measurements.clone(),
        ima_record_count: report.ima_record_count,
        ima_selected_pcr_record_count: report.ima_selected_pcr_record_count,
        pcr: policy.required_pcr,
        pcr_value: quote_pcr_value,
        quote_collected_at_ms: quote.collected_at_ms(),
        quote_verified_at_ms: quote.verified_at_ms(),
    };

    Ok(TpmImaPcrBindingQualification { report, verified })
}

fn qualification_digest(
    report_digest: &str,
    policy_digest: &str,
    report: &TpmImaPcrBindingReport,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    push_field(&mut hasher, report_digest);
    push_field(&mut hasher, policy_digest);
    push_field(&mut hasher, &report.quote_qualification_digest);
    push_field(&mut hasher, &report.ima_qualification_digest);
    push_field(&mut hasher, &report.challenge_digest);
    push_field(&mut hasher, &report.ak_binding_digest);
    push_field(&mut hasher, &report.quote_artifact_digest);
    push_field(&mut hasher, &report.verification_receipt_digest);
    push_field(&mut hasher, &report.pcr_selection_digest);
    push_field(&mut hasher, &report.measurement_list_digest);
    hasher.update(&(report.matched_required_measurements.len() as u64).to_le_bytes());
    for measurement_id in &report.matched_required_measurements {
        push_field(&mut hasher, measurement_id);
    }
    hasher.update(&report.ima_record_count.to_le_bytes());
    hasher.update(&report.ima_selected_pcr_record_count.to_le_bytes());
    hasher.update(&[report.required_pcr]);
    hasher.update(&report.ima_pcr_value);
    hasher.update(&report.quote_collected_at_ms.to_le_bytes());
    hasher.update(&report.quote_verified_at_ms.to_le_bytes());
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn finalize_report(mut report: TpmImaPcrBindingReport) -> TpmImaPcrBindingReport {
    report.disposition = if report
        .issues
        .iter()
        .any(TpmImaPcrBindingIssue::is_invalid)
    {
        TpmImaPcrBindingDisposition::Invalid
    } else {
        TpmImaPcrBindingDisposition::Blocked
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
