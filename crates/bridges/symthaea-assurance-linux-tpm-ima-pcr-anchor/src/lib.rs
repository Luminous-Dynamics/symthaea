// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed composition of verified TPM2 quote PCRs with canonical Linux IMA replay.
//!
//! This crate deliberately proves only a PCR-anchor theorem. It does not claim that the
//! IMA measurement policy already gives every measured artifact its final runtime role.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_assurance_tpm2_attestation_possession::{
    qualify_attestation_possession, AttestationChallenge, AttestationKeyBinding,
    AttestationPossessionPolicy, AttestationPossessionRecord,
};
use symthaea_assurance_tpm2_checkquote_adapter::{
    Tpm2CheckquotePolicy, Tpm2QuoteQualification,
};
use symthaea_assurance_tpm2_platform_qualification::Tpm2PlatformQualificationRecord;
use symthaea_assurance_tpm2_possession_policy_binding::canonical_possession_policy_digest;
use symthaea_linux_ima_replay::{ImaReplayPolicy, ImaReplayQualification};

pub const LINUX_TPM_IMA_PCR_ANCHOR_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.linux-tpm-ima-pcr-anchor-policy.v1";
pub const LINUX_TPM_IMA_PCR_ANCHOR_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.linux-tpm-ima-pcr-anchor-report.v1";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.linux-tpm-ima-pcr-anchor-policy.digest.v1\0";
const REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.linux-tpm-ima-pcr-anchor-report.digest.v1\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.linux-tpm-ima-pcr-anchor-qualification.digest.v1\0";
const SHA256_LEN: usize = 32;
const MAX_TEXT_BYTES: usize = 512;
const MAX_EVIDENCE_REFS: usize = 128;

/// Pins every lower-layer policy whose semantics are admitted into the PCR anchor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinuxTpmImaPcrAnchorPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_checkquote_policy_digest: String,
    pub expected_possession_policy_digest: String,
    pub expected_ima_replay_policy_digest: String,
    /// Exact SHA-256 PCR jointly authenticated by the quote and reproduced by IMA replay.
    pub expected_ima_pcr: u8,
    pub evidence_refs: Vec<String>,
}

impl LinuxTpmImaPcrAnchorPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == LINUX_TPM_IMA_PCR_ANCHOR_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3_digest(&self.expected_checkquote_policy_digest)
            && valid_blake3_digest(&self.expected_possession_policy_digest)
            && valid_blake3_digest(&self.expected_ima_replay_policy_digest)
            && self.expected_ima_pcr <= 23
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
            self.policy_id.as_str(),
            self.expected_checkquote_policy_digest.as_str(),
            self.expected_possession_policy_digest.as_str(),
            self.expected_ima_replay_policy_digest.as_str(),
        ] {
            push_field(&mut hasher, value);
        }
        hasher.update(&[self.expected_ima_pcr]);
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinuxTpmImaPcrAnchorDisposition {
    Invalid,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinuxTpmImaPcrAnchorIssue {
    InvalidAnchorPolicy,
    InvalidCheckquotePolicy,
    CheckquotePolicyDigestMismatch,
    QuoteCapabilityPolicyMismatch,
    InvalidPossessionPolicy,
    PossessionPolicyDigestMismatch,
    InvalidImaReplayPolicy,
    ImaReplayPolicyDigestMismatch,
    ImaCapabilityPolicyMismatch,
    ImaPcrPolicyMismatch { expected: u8, observed: u32 },
    CompositionBeforeQuoteVerification,
    PossessionRequalificationFailed(String),
    QuoteCapabilityBindingMismatch(String),
    ImaPcrNotAuthenticatedByQuote(u8),
    QuoteImaPcrMismatch,
}

impl LinuxTpmImaPcrAnchorIssue {
    fn code(&self) -> String {
        match self {
            Self::InvalidAnchorPolicy => "invalid-anchor-policy".into(),
            Self::InvalidCheckquotePolicy => "invalid-checkquote-policy".into(),
            Self::CheckquotePolicyDigestMismatch => "checkquote-policy-digest-mismatch".into(),
            Self::QuoteCapabilityPolicyMismatch => "quote-capability-policy-mismatch".into(),
            Self::InvalidPossessionPolicy => "invalid-possession-policy".into(),
            Self::PossessionPolicyDigestMismatch => "possession-policy-digest-mismatch".into(),
            Self::InvalidImaReplayPolicy => "invalid-ima-replay-policy".into(),
            Self::ImaReplayPolicyDigestMismatch => "ima-replay-policy-digest-mismatch".into(),
            Self::ImaCapabilityPolicyMismatch => "ima-capability-policy-mismatch".into(),
            Self::ImaPcrPolicyMismatch { expected, observed } => {
                format!("ima-pcr-policy-mismatch:{expected}:{observed}")
            }
            Self::CompositionBeforeQuoteVerification => {
                "composition-before-quote-verification".into()
            }
            Self::PossessionRequalificationFailed(error) => {
                format!("possession-requalification-failed:{error}")
            }
            Self::QuoteCapabilityBindingMismatch(field) => {
                format!("quote-capability-binding-mismatch:{field}")
            }
            Self::ImaPcrNotAuthenticatedByQuote(pcr) => {
                format!("ima-pcr-not-authenticated-by-quote:{pcr}")
            }
            Self::QuoteImaPcrMismatch => "quote-ima-pcr-mismatch".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinuxTpmImaPcrAnchorReport {
    pub schema_version: String,
    pub policy_id: String,
    pub anchor_policy_digest: Option<String>,
    pub checkquote_policy_digest: Option<String>,
    pub possession_policy_digest: Option<String>,
    pub ima_replay_policy_digest: Option<String>,
    pub quote_qualification_digest: String,
    pub possession_record_digest: Option<String>,
    pub ima_qualification_digest: String,
    pub ima_measurement_list_digest: String,
    pub ima_pcr: u8,
    pub quote_pcr_hex: Option<String>,
    pub replayed_ima_pcr_hex: String,
    pub quote_collected_at_ms: u64,
    pub quote_verified_at_ms: u64,
    pub qualified_at_ms: u64,
    pub disposition: LinuxTpmImaPcrAnchorDisposition,
    pub issues: Vec<LinuxTpmImaPcrAnchorIssue>,
}

impl LinuxTpmImaPcrAnchorReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DIGEST_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.anchor_policy_digest.as_deref().unwrap_or("-"),
            self.checkquote_policy_digest.as_deref().unwrap_or("-"),
            self.possession_policy_digest.as_deref().unwrap_or("-"),
            self.ima_replay_policy_digest.as_deref().unwrap_or("-"),
            self.quote_qualification_digest.as_str(),
            self.possession_record_digest.as_deref().unwrap_or("-"),
            self.ima_qualification_digest.as_str(),
            self.ima_measurement_list_digest.as_str(),
            self.quote_pcr_hex.as_deref().unwrap_or("-"),
            self.replayed_ima_pcr_hex.as_str(),
        ] {
            push_field(&mut hasher, value);
        }
        hasher.update(&[self.ima_pcr]);
        for value in [
            self.quote_collected_at_ms,
            self.quote_verified_at_ms,
            self.qualified_at_ms,
        ] {
            hasher.update(&value.to_le_bytes());
        }
        push_field(
            &mut hasher,
            match self.disposition {
                LinuxTpmImaPcrAnchorDisposition::Invalid => "invalid",
                LinuxTpmImaPcrAnchorDisposition::Qualified => "qualified",
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

/// Capability-bearing TPM↔IMA PCR composition result. Intentionally non-serializable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedLinuxTpmImaPcrAnchor {
    qualification_digest: String,
    report_digest: String,
    anchor_policy_digest: String,
    checkquote_policy_digest: String,
    possession_policy_digest: String,
    ima_replay_policy_digest: String,
    quote_qualification_digest: String,
    possession_record_digest: String,
    ima_qualification_digest: String,
    ima_measurement_list_digest: String,
    ima_pcr: u8,
    pcr_value: [u8; SHA256_LEN],
    quote_collected_at_ms: u64,
    quote_verified_at_ms: u64,
    qualified_at_ms: u64,
}

impl VerifiedLinuxTpmImaPcrAnchor {
    pub fn qualification_digest(&self) -> &str {
        &self.qualification_digest
    }
    pub fn report_digest(&self) -> &str {
        &self.report_digest
    }
    pub fn anchor_policy_digest(&self) -> &str {
        &self.anchor_policy_digest
    }
    pub fn checkquote_policy_digest(&self) -> &str {
        &self.checkquote_policy_digest
    }
    pub fn possession_policy_digest(&self) -> &str {
        &self.possession_policy_digest
    }
    pub fn ima_replay_policy_digest(&self) -> &str {
        &self.ima_replay_policy_digest
    }
    pub fn quote_qualification_digest(&self) -> &str {
        &self.quote_qualification_digest
    }
    pub fn possession_record_digest(&self) -> &str {
        &self.possession_record_digest
    }
    pub fn ima_qualification_digest(&self) -> &str {
        &self.ima_qualification_digest
    }
    pub fn ima_measurement_list_digest(&self) -> &str {
        &self.ima_measurement_list_digest
    }
    pub const fn ima_pcr(&self) -> u8 {
        self.ima_pcr
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
    pub const fn qualified_at_ms(&self) -> u64 {
        self.qualified_at_ms
    }
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinuxTpmImaPcrAnchorQualification {
    pub report: LinuxTpmImaPcrAnchorReport,
    pub possession: AttestationPossessionRecord,
    verified: VerifiedLinuxTpmImaPcrAnchor,
}

impl LinuxTpmImaPcrAnchorQualification {
    pub fn verified(&self) -> &VerifiedLinuxTpmImaPcrAnchor {
        &self.verified
    }
    pub fn into_verified(self) -> VerifiedLinuxTpmImaPcrAnchor {
        self.verified
    }
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[allow(clippy::too_many_arguments)]
pub fn qualify_linux_tpm_ima_pcr_anchor(
    anchor_policy: &LinuxTpmImaPcrAnchorPolicy,
    checkquote_policy: &Tpm2CheckquotePolicy,
    possession_policy: &AttestationPossessionPolicy,
    platform: &Tpm2PlatformQualificationRecord,
    challenge: &AttestationChallenge,
    ak: &AttestationKeyBinding,
    quote: &Tpm2QuoteQualification,
    ima_policy: &ImaReplayPolicy,
    ima: &ImaReplayQualification,
    qualified_at_ms: u64,
) -> Result<LinuxTpmImaPcrAnchorQualification, LinuxTpmImaPcrAnchorReport> {
    let quote_verified = quote.verified();
    let ima_verified = ima.verified();
    let anchor_policy_digest = anchor_policy.canonical_digest();
    let checkquote_policy_digest = checkquote_policy.canonical_digest();
    let possession_policy_digest = canonical_possession_policy_digest(possession_policy);
    let ima_replay_policy_digest = ima_policy.canonical_digest();

    let mut report = LinuxTpmImaPcrAnchorReport {
        schema_version: LINUX_TPM_IMA_PCR_ANCHOR_REPORT_SCHEMA_V1.into(),
        policy_id: anchor_policy.policy_id.clone(),
        anchor_policy_digest: anchor_policy_digest.clone(),
        checkquote_policy_digest: checkquote_policy_digest.clone(),
        possession_policy_digest: possession_policy_digest.clone(),
        ima_replay_policy_digest: ima_replay_policy_digest.clone(),
        quote_qualification_digest: quote_verified.qualification_digest().into(),
        possession_record_digest: None,
        ima_qualification_digest: ima_verified.qualification_digest().into(),
        ima_measurement_list_digest: ima_verified.measurement_list_digest().into(),
        ima_pcr: anchor_policy.expected_ima_pcr,
        quote_pcr_hex: None,
        replayed_ima_pcr_hex: hex_lower(&ima_verified.final_pcr()),
        quote_collected_at_ms: quote_verified.collected_at_ms(),
        quote_verified_at_ms: quote_verified.verified_at_ms(),
        qualified_at_ms,
        disposition: LinuxTpmImaPcrAnchorDisposition::Invalid,
        issues: Vec::new(),
    };

    macro_rules! reject {
        ($issue:expr) => {{
            report.issues.push($issue);
            return Err(finalize_report(report));
        }};
    }

    let anchor_policy_digest = match anchor_policy_digest {
        Some(value) => value,
        None => reject!(LinuxTpmImaPcrAnchorIssue::InvalidAnchorPolicy),
    };
    let checkquote_policy_digest = match checkquote_policy_digest {
        Some(value) => value,
        None => reject!(LinuxTpmImaPcrAnchorIssue::InvalidCheckquotePolicy),
    };
    if checkquote_policy_digest != anchor_policy.expected_checkquote_policy_digest {
        reject!(LinuxTpmImaPcrAnchorIssue::CheckquotePolicyDigestMismatch);
    }
    if quote_verified.policy_digest() != checkquote_policy_digest {
        reject!(LinuxTpmImaPcrAnchorIssue::QuoteCapabilityPolicyMismatch);
    }

    let possession_policy_digest = match possession_policy_digest {
        Some(value) => value,
        None => reject!(LinuxTpmImaPcrAnchorIssue::InvalidPossessionPolicy),
    };
    if possession_policy_digest != anchor_policy.expected_possession_policy_digest {
        reject!(LinuxTpmImaPcrAnchorIssue::PossessionPolicyDigestMismatch);
    }

    let ima_replay_policy_digest = match ima_replay_policy_digest {
        Some(value) => value,
        None => reject!(LinuxTpmImaPcrAnchorIssue::InvalidImaReplayPolicy),
    };
    if ima_replay_policy_digest != anchor_policy.expected_ima_replay_policy_digest {
        reject!(LinuxTpmImaPcrAnchorIssue::ImaReplayPolicyDigestMismatch);
    }
    if ima_verified.policy_digest() != ima_replay_policy_digest {
        reject!(LinuxTpmImaPcrAnchorIssue::ImaCapabilityPolicyMismatch);
    }
    if ima_policy.expected_pcr != u32::from(anchor_policy.expected_ima_pcr) {
        reject!(LinuxTpmImaPcrAnchorIssue::ImaPcrPolicyMismatch {
            expected: anchor_policy.expected_ima_pcr,
            observed: ima_policy.expected_pcr,
        });
    }
    if ima_verified.expected_pcr() != ima_policy.expected_pcr {
        reject!(LinuxTpmImaPcrAnchorIssue::ImaPcrPolicyMismatch {
            expected: anchor_policy.expected_ima_pcr,
            observed: ima_verified.expected_pcr(),
        });
    }
    if qualified_at_ms < quote_verified.verified_at_ms() {
        reject!(LinuxTpmImaPcrAnchorIssue::CompositionBeforeQuoteVerification);
    }

    // Re-run the semantic possession theorem rather than trusting a caller-supplied record.
    // This revalidates platform qualification, challenge freshness/nonce, AK properties,
    // verifier independence, selected PCRs, quote artifacts, tool identity and timing.
    let possession = match qualify_attestation_possession(
        possession_policy,
        platform,
        challenge,
        ak,
        &quote.quote_artifacts,
        &quote.verification_receipt,
        qualified_at_ms,
    ) {
        Ok(value) => value,
        Err(error) => reject!(LinuxTpmImaPcrAnchorIssue::PossessionRequalificationFailed(
            format!("{error:?}")
        )),
    };
    let possession_record_digest = possession.possession_digest();
    report.possession_record_digest = Some(possession_record_digest.clone());

    for (field, actual, expected) in [
        (
            "challenge-digest",
            possession.challenge_digest.as_str(),
            quote_verified.challenge_digest(),
        ),
        (
            "ak-binding-digest",
            possession.ak_binding_digest.as_str(),
            quote_verified.ak_binding_digest(),
        ),
        (
            "quote-artifact-digest",
            possession.quote_artifact_digest.as_str(),
            quote_verified.quote_artifact_digest(),
        ),
        (
            "verification-receipt-digest",
            possession.verification_receipt_digest.as_str(),
            quote_verified.verification_receipt_digest(),
        ),
        (
            "pcr-selection-digest",
            possession.pcr_selection_digest.as_str(),
            quote_verified.pcr_selection_digest(),
        ),
    ] {
        if actual != expected {
            reject!(LinuxTpmImaPcrAnchorIssue::QuoteCapabilityBindingMismatch(
                field.into()
            ));
        }
    }

    // The public quote artifacts/receipt are mutable fields of Tpm2QuoteQualification;
    // bind them back to the opaque capability so caller mutation cannot be re-qualified.
    if quote.quote_artifacts.artifact_digest() != quote_verified.quote_artifact_digest() {
        reject!(LinuxTpmImaPcrAnchorIssue::QuoteCapabilityBindingMismatch(
            "public-quote-artifacts".into()
        ));
    }
    if quote.verification_receipt.receipt_digest()
        != quote_verified.verification_receipt_digest()
    {
        reject!(LinuxTpmImaPcrAnchorIssue::QuoteCapabilityBindingMismatch(
            "public-verification-receipt".into()
        ));
    }

    let quoted_pcr = match quote_verified.pcr_value(anchor_policy.expected_ima_pcr) {
        Some(value) => value,
        None => reject!(LinuxTpmImaPcrAnchorIssue::ImaPcrNotAuthenticatedByQuote(
            anchor_policy.expected_ima_pcr
        )),
    };
    report.quote_pcr_hex = Some(hex_lower(&quoted_pcr));
    if quoted_pcr != ima_verified.final_pcr() {
        reject!(LinuxTpmImaPcrAnchorIssue::QuoteImaPcrMismatch);
    }

    report.disposition = LinuxTpmImaPcrAnchorDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let qualification_digest = qualification_digest(
        &report_digest,
        &anchor_policy_digest,
        &checkquote_policy_digest,
        &possession_policy_digest,
        &ima_replay_policy_digest,
        quote_verified.qualification_digest(),
        &possession_record_digest,
        ima_verified.qualification_digest(),
        ima_verified.measurement_list_digest(),
        anchor_policy.expected_ima_pcr,
        quoted_pcr,
        qualified_at_ms,
    );

    let verified = VerifiedLinuxTpmImaPcrAnchor {
        qualification_digest,
        report_digest,
        anchor_policy_digest,
        checkquote_policy_digest,
        possession_policy_digest,
        ima_replay_policy_digest,
        quote_qualification_digest: quote_verified.qualification_digest().into(),
        possession_record_digest,
        ima_qualification_digest: ima_verified.qualification_digest().into(),
        ima_measurement_list_digest: ima_verified.measurement_list_digest().into(),
        ima_pcr: anchor_policy.expected_ima_pcr,
        pcr_value: quoted_pcr,
        quote_collected_at_ms: quote_verified.collected_at_ms(),
        quote_verified_at_ms: quote_verified.verified_at_ms(),
        qualified_at_ms,
    };

    Ok(LinuxTpmImaPcrAnchorQualification {
        report,
        possession,
        verified,
    })
}

#[allow(clippy::too_many_arguments)]
fn qualification_digest(
    report_digest: &str,
    anchor_policy_digest: &str,
    checkquote_policy_digest: &str,
    possession_policy_digest: &str,
    ima_replay_policy_digest: &str,
    quote_qualification_digest: &str,
    possession_record_digest: &str,
    ima_qualification_digest: &str,
    measurement_list_digest: &str,
    ima_pcr: u8,
    pcr_value: [u8; SHA256_LEN],
    qualified_at_ms: u64,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    for value in [
        report_digest,
        anchor_policy_digest,
        checkquote_policy_digest,
        possession_policy_digest,
        ima_replay_policy_digest,
        quote_qualification_digest,
        possession_record_digest,
        ima_qualification_digest,
        measurement_list_digest,
    ] {
        push_field(&mut hasher, value);
    }
    hasher.update(&[ima_pcr]);
    hasher.update(&pcr_value);
    hasher.update(&qualified_at_ms.to_le_bytes());
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn finalize_report(mut report: LinuxTpmImaPcrAnchorReport) -> LinuxTpmImaPcrAnchorReport {
    report.disposition = LinuxTpmImaPcrAnchorDisposition::Invalid;
    report
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_TEXT_BYTES
        && value == value.trim()
        && !value.chars().any(char::is_control)
}

fn valid_blake3_digest(value: &str) -> bool {
    value
        .strip_prefix("blake3:")
        .is_some_and(|hex| hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

fn valid_refs(refs: &[String]) -> bool {
    !refs.is_empty()
        && refs.len() <= MAX_EVIDENCE_REFS
        && refs.iter().all(|value| canonical_text(value))
        && {
            let mut sorted = refs.to_vec();
            sorted.sort();
            sorted.dedup();
            sorted.len() == refs.len()
        }
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

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn policy() -> LinuxTpmImaPcrAnchorPolicy {
        LinuxTpmImaPcrAnchorPolicy {
            schema_version: LINUX_TPM_IMA_PCR_ANCHOR_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:linux-tpm-ima-pcr-anchor:1".into(),
            expected_checkquote_policy_digest: digest("checkquote-policy"),
            expected_possession_policy_digest: digest("possession-policy"),
            expected_ima_replay_policy_digest: digest("ima-replay-policy"),
            expected_ima_pcr: 10,
            evidence_refs: vec!["review:anchor-policy".into(), "issue:2875".into()],
        }
    }

    #[test]
    fn policy_commitment_is_order_independent_for_evidence_refs() {
        let left = policy();
        let mut right = left.clone();
        right.evidence_refs.reverse();
        assert_eq!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn same_id_policy_semantic_drift_changes_commitment() {
        let base = policy();
        let baseline = base.canonical_digest().unwrap();
        let mut variants = Vec::new();

        let mut changed = base.clone();
        changed.expected_checkquote_policy_digest = digest("other-checkquote");
        variants.push(changed);

        let mut changed = base.clone();
        changed.expected_possession_policy_digest = digest("other-possession");
        variants.push(changed);

        let mut changed = base.clone();
        changed.expected_ima_replay_policy_digest = digest("other-ima");
        variants.push(changed);

        let mut changed = base.clone();
        changed.expected_ima_pcr = 11;
        variants.push(changed);

        for changed in variants {
            assert_eq!(changed.policy_id, base.policy_id);
            assert_ne!(changed.canonical_digest().unwrap(), baseline);
        }
    }

    #[test]
    fn invalid_policy_has_no_commitment() {
        let mut invalid = policy();
        invalid.evidence_refs.clear();
        assert!(!invalid.validate());
        assert_eq!(invalid.canonical_digest(), None);
    }

    #[test]
    fn pcr_outside_raw_quote_supported_range_is_rejected() {
        let mut invalid = policy();
        invalid.expected_ima_pcr = 24;
        assert!(!invalid.validate());
    }

    #[test]
    fn evidence_refs_are_unique() {
        let mut invalid = policy();
        invalid.evidence_refs = vec!["same".into(), "same".into()];
        assert!(!invalid.validate());
    }

    #[test]
    fn capability_and_report_never_grant_physical_authority() {
        assert!(!LinuxTpmImaPcrAnchorReport {
            schema_version: LINUX_TPM_IMA_PCR_ANCHOR_REPORT_SCHEMA_V1.into(),
            policy_id: "policy".into(),
            anchor_policy_digest: None,
            checkquote_policy_digest: None,
            possession_policy_digest: None,
            ima_replay_policy_digest: None,
            quote_qualification_digest: "q".into(),
            possession_record_digest: None,
            ima_qualification_digest: "i".into(),
            ima_measurement_list_digest: "m".into(),
            ima_pcr: 10,
            quote_pcr_hex: None,
            replayed_ima_pcr_hex: String::new(),
            quote_collected_at_ms: 1,
            quote_verified_at_ms: 2,
            qualified_at_ms: 3,
            disposition: LinuxTpmImaPcrAnchorDisposition::Invalid,
            issues: vec![LinuxTpmImaPcrAnchorIssue::InvalidAnchorPolicy],
        }
        .grants_physical_authority());
    }
}
