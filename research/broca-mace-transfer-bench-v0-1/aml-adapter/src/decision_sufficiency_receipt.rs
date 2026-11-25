// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Strict, decision-scoped verification for MACE Decision Sufficiency Receipt v1.
//!
//! This module verifies an assessor's signed claim against consumer-owned
//! decision bindings and a separately pinned MACE Ed25519 verifier. It does
//! not decide whether evidence is sufficient; it checks that an independent
//! assessor's receipt is internally coherent, current, correctly bound and
//! signed by a trusted key.

use std::collections::BTreeSet;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::evidence_qualification_gate::{
    EvidenceQualificationRejectV1, EvidenceQualificationSignatureVerifierV1,
};

pub const DECISION_SUFFICIENCY_RECEIPT_SCHEMA_V1: &str =
    "mace_decision_sufficiency_receipt_v1";
pub const DECISION_SUFFICIENCY_PAYLOAD_SCHEMA_V1: &str =
    "mace_decision_sufficiency_payload_v1";
pub const MAX_DECISION_SUFFICIENCY_RECEIPT_BYTES_V1: usize = 1_048_576;
pub const MAX_DECISION_SUFFICIENCY_RECEIPT_TTL_SECONDS_V1: u64 = 900;
pub const MAX_DECISION_SUFFICIENCY_IDS_V1: usize = 10_000;
pub const MAX_DECISION_SUFFICIENCY_IDENTIFIER_BYTES_V1: usize = 256;
pub const MAX_DECISION_SUFFICIENCY_UNIX_SECONDS_V1: u64 = 253_402_300_799;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionSufficiencyReceiptV1 {
    pub schema_version: String,
    pub payload: DecisionSufficiencyPayloadV1,
    pub attestation: DecisionSufficiencyAttestationV1,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionSufficiencyPayloadV1 {
    pub schema_version: String,
    pub decision_episode_id: String,
    pub decision_invocation_id: String,
    pub binding: DecisionSufficiencyBindingV1,
    pub assessment: DecisionSufficiencyAssessmentV1,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionSufficiencyBindingV1 {
    pub subject_scope_sha256: String,
    pub purpose_sha256: String,
    pub decision_need_sha256: String,
    pub option_set_sha256: String,
    pub constraint_set_sha256: String,
    pub environment_revision_sha256: String,
    pub lifecycle_snapshot_sha256: String,
    pub qualification_receipt_sha256: String,
    pub qualified_evidence_set_sha256: String,
    pub reader_input_sha256: String,
    pub reader_profile_sha256: String,
    pub decision_policy_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionSufficiencyDispositionV1 {
    Sufficient,
    Insufficient,
    Indeterminate,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionSufficiencyAssessmentV1 {
    pub disposition: DecisionSufficiencyDispositionV1,
    pub admissible_evidence_ids: Vec<String>,
    pub applicable_requirement_ids: Vec<String>,
    pub satisfied_requirement_ids: Vec<String>,
    pub missing_requirement_ids: Vec<String>,
    pub unresolved_contradiction_ids: Vec<String>,
    pub coverage_report_sha256: String,
    pub contradiction_report_sha256: String,
    pub uncertainty_report_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionSufficiencyAttestationV1 {
    pub assessor_profile_sha256: String,
    pub verifier_profile_sha256: String,
    pub key_id: String,
    pub issued_at_unix: u64,
    pub expires_at_unix: u64,
    pub payload_sha256: String,
    pub attestation_sha256: String,
    pub signature_base64: String,
}

/// Consumer-owned expected values. None of these are copied from the receipt.
pub struct DecisionSufficiencyExpectedBindingV1<'a> {
    pub decision_episode_id: &'a str,
    pub decision_invocation_id: &'a str,
    pub binding: &'a DecisionSufficiencyBindingV1,
    /// Exact complete set returned by the already verified evidence qualifier.
    pub qualified_evidence_ids: &'a [String],
    pub assessor_profile_sha256: &'a str,
    /// Digest of the verifier profile pinned out of band by the consumer.
    pub verifier_profile_sha256: &'a str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionSufficiencyReceiptRejectV1 {
    ReceiptTooLarge,
    InvalidJson,
    UnsupportedSchemaVersion,
    InvalidIdentifier,
    InvalidDigest,
    NonCanonicalIdentifierSet,
    BindingMismatch,
    QualifiedEvidenceSetMismatch,
    RequirementPartitionMismatch,
    SufficientWithUnresolvedRequirements,
    InsufficientWithoutKnownGap,
    InvalidReceiptTtl,
    ReceiptNotYetValid,
    ReceiptExpired,
    InvalidSignatureEncoding,
    PayloadDigestMismatch,
    AttestationDigestMismatch,
    CanonicalizationFailed,
    SignatureRejected(EvidenceQualificationRejectV1),
}

/// Capability object returned only after strict parsing, cross-field
/// validation, exact consumer binding, expiry checks and pinned signature
/// verification. Its fields are private so it cannot be fabricated from JSON.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedDecisionSufficiencyReceiptV1 {
    receipt_sha256: String,
    payload_sha256: String,
    decision_episode_id: String,
    decision_invocation_id: String,
    disposition: DecisionSufficiencyDispositionV1,
    admissible_evidence_ids: Vec<String>,
    expires_at_unix: u64,
}

impl VerifiedDecisionSufficiencyReceiptV1 {
    pub fn receipt_sha256(&self) -> &str {
        &self.receipt_sha256
    }

    pub fn payload_sha256(&self) -> &str {
        &self.payload_sha256
    }

    pub fn decision_episode_id(&self) -> &str {
        &self.decision_episode_id
    }

    pub fn decision_invocation_id(&self) -> &str {
        &self.decision_invocation_id
    }

    pub fn disposition(&self) -> DecisionSufficiencyDispositionV1 {
        self.disposition
    }

    pub fn admissible_evidence_ids(&self) -> &[String] {
        &self.admissible_evidence_ids
    }

    pub fn expires_at_unix(&self) -> u64 {
        self.expires_at_unix
    }
}

/// Call for each consequential use with fresh trusted time and freshly sampled
/// consumer bindings; do not treat the returned capability as a permanent grant.
pub fn parse_and_verify_decision_sufficiency_receipt_v1(
    wire: &[u8],
    expected: &DecisionSufficiencyExpectedBindingV1<'_>,
    trusted_now_unix: u64,
    verifier: &impl EvidenceQualificationSignatureVerifierV1,
) -> Result<VerifiedDecisionSufficiencyReceiptV1, DecisionSufficiencyReceiptRejectV1> {
    if wire.len() > MAX_DECISION_SUFFICIENCY_RECEIPT_BYTES_V1 {
        return Err(DecisionSufficiencyReceiptRejectV1::ReceiptTooLarge);
    }

    // Derive-based deserialization with deny_unknown_fields rejects duplicate
    // known fields, unknown fields and missing fields at every object level.
    let receipt: DecisionSufficiencyReceiptV1 =
        serde_json::from_slice(wire).map_err(|_| DecisionSufficiencyReceiptRejectV1::InvalidJson)?;

    validate_decision_sufficiency_receipt_v1(
        &receipt,
        expected,
        trusted_now_unix,
        verifier,
    )
}

pub fn validate_decision_sufficiency_receipt_v1(
    receipt: &DecisionSufficiencyReceiptV1,
    expected: &DecisionSufficiencyExpectedBindingV1<'_>,
    trusted_now_unix: u64,
    verifier: &impl EvidenceQualificationSignatureVerifierV1,
) -> Result<VerifiedDecisionSufficiencyReceiptV1, DecisionSufficiencyReceiptRejectV1> {
    if receipt.schema_version != DECISION_SUFFICIENCY_RECEIPT_SCHEMA_V1
        || receipt.payload.schema_version != DECISION_SUFFICIENCY_PAYLOAD_SCHEMA_V1
    {
        return Err(DecisionSufficiencyReceiptRejectV1::UnsupportedSchemaVersion);
    }

    validate_atom(&receipt.payload.decision_episode_id)?;
    validate_atom(&receipt.payload.decision_invocation_id)?;
    validate_atom(expected.decision_episode_id)?;
    validate_atom(expected.decision_invocation_id)?;

    if receipt.payload.decision_episode_id != expected.decision_episode_id
        || receipt.payload.decision_invocation_id != expected.decision_invocation_id
        || &receipt.payload.binding != expected.binding
    {
        return Err(DecisionSufficiencyReceiptRejectV1::BindingMismatch);
    }

    validate_binding_digests(&receipt.payload.binding)?;
    validate_binding_digests(expected.binding)?;
    validate_hash(expected.assessor_profile_sha256)?;
    validate_hash(expected.verifier_profile_sha256)?;

    if receipt.attestation.assessor_profile_sha256 != expected.assessor_profile_sha256
        || receipt.attestation.verifier_profile_sha256 != expected.verifier_profile_sha256
    {
        return Err(DecisionSufficiencyReceiptRejectV1::BindingMismatch);
    }
    validate_atom(&receipt.attestation.key_id)?;
    validate_hash(&receipt.attestation.payload_sha256)?;
    validate_hash(&receipt.attestation.attestation_sha256)?;

    validate_sorted_ids(expected.qualified_evidence_ids)?;
    validate_assessment(
        &receipt.payload.assessment,
        expected.qualified_evidence_ids,
    )?;

    if receipt.attestation.issued_at_unix > MAX_DECISION_SUFFICIENCY_UNIX_SECONDS_V1
        || receipt.attestation.expires_at_unix > MAX_DECISION_SUFFICIENCY_UNIX_SECONDS_V1
    {
        return Err(DecisionSufficiencyReceiptRejectV1::InvalidReceiptTtl);
    }

    let lifetime = receipt
        .attestation
        .expires_at_unix
        .checked_sub(receipt.attestation.issued_at_unix)
        .ok_or(DecisionSufficiencyReceiptRejectV1::InvalidReceiptTtl)?;
    if lifetime == 0 || lifetime > MAX_DECISION_SUFFICIENCY_RECEIPT_TTL_SECONDS_V1 {
        return Err(DecisionSufficiencyReceiptRejectV1::InvalidReceiptTtl);
    }
    if trusted_now_unix < receipt.attestation.issued_at_unix {
        return Err(DecisionSufficiencyReceiptRejectV1::ReceiptNotYetValid);
    }
    if trusted_now_unix >= receipt.attestation.expires_at_unix {
        return Err(DecisionSufficiencyReceiptRejectV1::ReceiptExpired);
    }

    let calculated_payload_sha256 = decision_sufficiency_payload_sha256_v1(&receipt.payload)?;
    if calculated_payload_sha256 != receipt.attestation.payload_sha256 {
        return Err(DecisionSufficiencyReceiptRejectV1::PayloadDigestMismatch);
    }

    let calculated_attestation_sha256 = decision_sufficiency_attestation_sha256_v1(
        &calculated_payload_sha256,
        &receipt.attestation,
    )?;
    if calculated_attestation_sha256 != receipt.attestation.attestation_sha256 {
        return Err(DecisionSufficiencyReceiptRejectV1::AttestationDigestMismatch);
    }

    if receipt.attestation.signature_base64.len() != 88 {
        return Err(DecisionSufficiencyReceiptRejectV1::InvalidSignatureEncoding);
    }
    let signature = STANDARD
        .decode(&receipt.attestation.signature_base64)
        .map_err(|_| DecisionSufficiencyReceiptRejectV1::InvalidSignatureEncoding)?;
    if signature.len() != 64 || STANDARD.encode(&signature) != receipt.attestation.signature_base64 {
        return Err(DecisionSufficiencyReceiptRejectV1::InvalidSignatureEncoding);
    }

    // The existing pinned MACE verifier intentionally verifies the lowercase
    // 64-byte hex digest representation. This is the shared MACE v1 signing
    // profile, not a second signature convention or a DSSE envelope.
    verifier
        .verify_signature(
            &receipt.attestation.verifier_profile_sha256,
            &receipt.attestation.key_id,
            &calculated_attestation_sha256,
            &signature,
            receipt.attestation.issued_at_unix,
            trusted_now_unix,
        )
        .map_err(DecisionSufficiencyReceiptRejectV1::SignatureRejected)?;

    let canonical_receipt = serde_json_canonicalizer::to_vec(receipt)
        .map_err(|_| DecisionSufficiencyReceiptRejectV1::CanonicalizationFailed)?;
    let mut receipt_hasher = Sha256::new();
    receipt_hasher.update(b"luminous:mace:decision-sufficiency-receipt:v1\0");
    receipt_hasher.update(&canonical_receipt);
    let receipt_sha256 = hex_digest(&receipt_hasher.finalize());

    Ok(VerifiedDecisionSufficiencyReceiptV1 {
        receipt_sha256,
        payload_sha256: calculated_payload_sha256,
        decision_episode_id: receipt.payload.decision_episode_id.clone(),
        decision_invocation_id: receipt.payload.decision_invocation_id.clone(),
        disposition: receipt.payload.assessment.disposition,
        admissible_evidence_ids: receipt.payload.assessment.admissible_evidence_ids.clone(),
        expires_at_unix: receipt.attestation.expires_at_unix,
    })
}

/// Digest the payload using RFC 8785 JCS and the MACE v1 payload domain.
/// The signature covers decision_sufficiency_attestation_sha256_v1 instead.
pub fn decision_sufficiency_payload_sha256_v1(
    payload: &DecisionSufficiencyPayloadV1,
) -> Result<String, DecisionSufficiencyReceiptRejectV1> {
    let canonical = serde_json_canonicalizer::to_vec(payload)
        .map_err(|_| DecisionSufficiencyReceiptRejectV1::CanonicalizationFailed)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:decision-sufficiency-payload:v1\0");
    hasher.update(&canonical);
    Ok(hex_digest(&hasher.finalize()))
}


/// Hash the payload digest together with every validity/signer field that is
/// otherwise outside the payload. The signature covers this digest, preventing
/// an intermediary from extending expiry or rebinding the signer metadata.
pub fn decision_sufficiency_attestation_sha256_v1(
    payload_sha256: &str,
    attestation: &DecisionSufficiencyAttestationV1,
) -> Result<String, DecisionSufficiencyReceiptRejectV1> {
    validate_hash(payload_sha256)?;
    let preimage = json!({
        "schema_version": "mace_decision_sufficiency_attestation_preimage_v1",
        "payload_sha256": payload_sha256,
        "assessor_profile_sha256": attestation.assessor_profile_sha256,
        "verifier_profile_sha256": attestation.verifier_profile_sha256,
        "key_id": attestation.key_id,
        "issued_at_unix": attestation.issued_at_unix,
        "expires_at_unix": attestation.expires_at_unix
    });
    let canonical = serde_json_canonicalizer::to_vec(&preimage)
        .map_err(|_| DecisionSufficiencyReceiptRejectV1::CanonicalizationFailed)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:decision-sufficiency-attestation:v1\0");
    hasher.update(&canonical);
    Ok(hex_digest(&hasher.finalize()))
}

fn validate_assessment(
    assessment: &DecisionSufficiencyAssessmentV1,
    qualified_evidence_ids: &[String],
) -> Result<(), DecisionSufficiencyReceiptRejectV1> {
    validate_sorted_ids(&assessment.admissible_evidence_ids)?;
    validate_sorted_ids(&assessment.applicable_requirement_ids)?;
    validate_sorted_ids(&assessment.satisfied_requirement_ids)?;
    validate_sorted_ids(&assessment.missing_requirement_ids)?;
    validate_sorted_ids(&assessment.unresolved_contradiction_ids)?;

    for digest in [
        assessment.coverage_report_sha256.as_str(),
        assessment.contradiction_report_sha256.as_str(),
        assessment.uncertainty_report_sha256.as_str(),
    ] {
        validate_hash(digest)?;
    }

    let qualified: BTreeSet<&str> = qualified_evidence_ids.iter().map(String::as_str).collect();
    if assessment
        .admissible_evidence_ids
        .iter()
        .any(|id| !qualified.contains(id.as_str()))
    {
        return Err(DecisionSufficiencyReceiptRejectV1::QualifiedEvidenceSetMismatch);
    }

    let applicable: BTreeSet<&str> = assessment
        .applicable_requirement_ids
        .iter()
        .map(String::as_str)
        .collect();
    let satisfied: BTreeSet<&str> = assessment
        .satisfied_requirement_ids
        .iter()
        .map(String::as_str)
        .collect();
    let missing: BTreeSet<&str> = assessment
        .missing_requirement_ids
        .iter()
        .map(String::as_str)
        .collect();

    if !satisfied.is_subset(&applicable)
        || !missing.is_subset(&applicable)
        || !satisfied.is_disjoint(&missing)
        || satisfied.union(&missing).copied().collect::<BTreeSet<_>>() != applicable
    {
        return Err(DecisionSufficiencyReceiptRejectV1::RequirementPartitionMismatch);
    }

    match assessment.disposition {
        DecisionSufficiencyDispositionV1::Sufficient => {
            if !applicable.is_empty()
                && missing.is_empty()
                && assessment.unresolved_contradiction_ids.is_empty()
                && satisfied == applicable
            {
                Ok(())
            } else {
                Err(DecisionSufficiencyReceiptRejectV1::SufficientWithUnresolvedRequirements)
            }
        }
        DecisionSufficiencyDispositionV1::Insufficient => {
            if missing.is_empty() && assessment.unresolved_contradiction_ids.is_empty() {
                Err(DecisionSufficiencyReceiptRejectV1::InsufficientWithoutKnownGap)
            } else {
                Ok(())
            }
        }
        DecisionSufficiencyDispositionV1::Indeterminate => Ok(()),
    }
}

fn validate_binding_digests(
    binding: &DecisionSufficiencyBindingV1,
) -> Result<(), DecisionSufficiencyReceiptRejectV1> {
    for digest in [
        binding.subject_scope_sha256.as_str(),
        binding.purpose_sha256.as_str(),
        binding.decision_need_sha256.as_str(),
        binding.option_set_sha256.as_str(),
        binding.constraint_set_sha256.as_str(),
        binding.environment_revision_sha256.as_str(),
        binding.lifecycle_snapshot_sha256.as_str(),
        binding.qualification_receipt_sha256.as_str(),
        binding.qualified_evidence_set_sha256.as_str(),
        binding.reader_input_sha256.as_str(),
        binding.reader_profile_sha256.as_str(),
        binding.decision_policy_sha256.as_str(),
    ] {
        validate_hash(digest)?;
    }
    Ok(())
}

fn validate_sorted_ids(ids: &[String]) -> Result<(), DecisionSufficiencyReceiptRejectV1> {
    if ids.len() > MAX_DECISION_SUFFICIENCY_IDS_V1
        || ids.iter().any(|id| validate_atom(id).is_err())
        || ids.windows(2).any(|pair| pair[0].as_bytes() >= pair[1].as_bytes())
    {
        return Err(DecisionSufficiencyReceiptRejectV1::NonCanonicalIdentifierSet);
    }
    Ok(())
}

fn validate_atom(value: &str) -> Result<(), DecisionSufficiencyReceiptRejectV1> {
    if value.is_empty()
        || value.len() > MAX_DECISION_SUFFICIENCY_IDENTIFIER_BYTES_V1
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(DecisionSufficiencyReceiptRejectV1::InvalidIdentifier);
    }
    Ok(())
}

fn validate_hash(value: &str) -> Result<(), DecisionSufficiencyReceiptRejectV1> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(DecisionSufficiencyReceiptRejectV1::InvalidDigest);
    }
    Ok(())
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qualification_trust::{
        PinnedEd25519QualificationVerifierV1, QualificationKeyStatusV1,
        QualificationTrustKeyV1, QualificationTrustProfileV1, qualification_key_id_v1,
    };
    use ed25519_dalek::{Signer, SigningKey};

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn fixture() -> (
        DecisionSufficiencyReceiptV1,
        DecisionSufficiencyBindingV1,
        Vec<String>,
        String,
        String,
        PinnedEd25519QualificationVerifierV1,
    ) {
        let signer = SigningKey::from_bytes(&[7; 32]);
        let public_key = signer.verifying_key().to_bytes();
        let key_id = qualification_key_id_v1(&public_key);
        let profile = QualificationTrustProfileV1 {
            schema_version: 1,
            verifier_id: "mace-decision-test-verifier".to_owned(),
            profile_version: 1,
            keys: vec![QualificationTrustKeyV1 {
                key_id: key_id.clone(),
                public_key,
                status: QualificationKeyStatusV1::Active,
                not_before_unix: 10,
                not_after_unix: 1_000,
            }],
        };
        let verifier_profile_sha256 = profile.profile_sha256().unwrap();
        let verifier = PinnedEd25519QualificationVerifierV1::from_pinned_profile(
            &profile,
            &verifier_profile_sha256,
            1,
        )
        .unwrap();

        let binding = DecisionSufficiencyBindingV1 {
            subject_scope_sha256: digest('a'),
            purpose_sha256: digest('b'),
            decision_need_sha256: digest('c'),
            option_set_sha256: digest('d'),
            constraint_set_sha256: digest('e'),
            environment_revision_sha256: digest('f'),
            lifecycle_snapshot_sha256: digest('1'),
            qualification_receipt_sha256: digest('2'),
            qualified_evidence_set_sha256: digest('3'),
            reader_input_sha256: digest('4'),
            reader_profile_sha256: digest('5'),
            decision_policy_sha256: digest('6'),
        };
        let qualified_evidence_ids = vec!["evidence-1".to_owned(), "evidence-2".to_owned()];
        let assessor_profile_sha256 = digest('7');
        let payload = DecisionSufficiencyPayloadV1 {
            schema_version: DECISION_SUFFICIENCY_PAYLOAD_SCHEMA_V1.to_owned(),
            decision_episode_id: "episode-1".to_owned(),
            decision_invocation_id: "decision-1".to_owned(),
            binding: binding.clone(),
            assessment: DecisionSufficiencyAssessmentV1 {
                disposition: DecisionSufficiencyDispositionV1::Sufficient,
                admissible_evidence_ids: qualified_evidence_ids.clone(),
                applicable_requirement_ids: vec!["requirement-1".to_owned(), "requirement-2".to_owned()],
                satisfied_requirement_ids: vec!["requirement-1".to_owned(), "requirement-2".to_owned()],
                missing_requirement_ids: vec![],
                unresolved_contradiction_ids: vec![],
                coverage_report_sha256: digest('8'),
                contradiction_report_sha256: digest('9'),
                uncertainty_report_sha256: digest('0'),
            },
        };
        let payload_sha256 = decision_sufficiency_payload_sha256_v1(&payload).unwrap();
        let mut attestation = DecisionSufficiencyAttestationV1 {
            assessor_profile_sha256: assessor_profile_sha256.clone(),
            verifier_profile_sha256: verifier_profile_sha256.clone(),
            key_id,
            issued_at_unix: 100,
            expires_at_unix: 300,
            payload_sha256,
            attestation_sha256: String::new(),
            signature_base64: String::new(),
        };
        let attestation_sha256 =
            decision_sufficiency_attestation_sha256_v1(&attestation.payload_sha256, &attestation)
                .unwrap();
        attestation.attestation_sha256 = attestation_sha256.clone();
        attestation.signature_base64 = STANDARD.encode(signer.sign(attestation_sha256.as_bytes()).to_bytes());
        let receipt = DecisionSufficiencyReceiptV1 {
            schema_version: DECISION_SUFFICIENCY_RECEIPT_SCHEMA_V1.to_owned(),
            payload,
            attestation,
        };
        (
            receipt,
            binding,
            qualified_evidence_ids,
            assessor_profile_sha256,
            verifier_profile_sha256,
            verifier,
        )
    }

    fn expected<'a>(
        binding: &'a DecisionSufficiencyBindingV1,
        qualified_evidence_ids: &'a [String],
        assessor_profile_sha256: &'a str,
        verifier_profile_sha256: &'a str,
    ) -> DecisionSufficiencyExpectedBindingV1<'a> {
        DecisionSufficiencyExpectedBindingV1 {
            decision_episode_id: "episode-1",
            decision_invocation_id: "decision-1",
            binding,
            qualified_evidence_ids,
            assessor_profile_sha256,
            verifier_profile_sha256,
        }
    }

    #[test]
    fn valid_sufficient_receipt_returns_verified_capability() {
        let (receipt, binding, qualified, assessor, verifier_profile, verifier) = fixture();
        let expected = expected(&binding, &qualified, &assessor, &verifier_profile);
        let wire = serde_json::to_vec(&receipt).unwrap();
        let verified =
            parse_and_verify_decision_sufficiency_receipt_v1(&wire, &expected, 200, &verifier)
                .unwrap();
        assert_eq!(verified.disposition(), DecisionSufficiencyDispositionV1::Sufficient);
        assert_eq!(verified.admissible_evidence_ids(), qualified);
        assert_eq!(verified.decision_episode_id(), "episode-1");
        assert_eq!(verified.expires_at_unix(), 300);
        assert_eq!(verified.receipt_sha256().len(), 64);
    }

    #[test]
    fn expiry_is_checked_at_consumer_use_time() {
        let (mut receipt, binding, qualified, assessor, verifier_profile, verifier) = fixture();
        receipt.attestation.expires_at_unix = 200;
        let expected = expected(&binding, &qualified, &assessor, &verifier_profile);
        let result = validate_decision_sufficiency_receipt_v1(&receipt, &expected, 200, &verifier);
        assert_eq!(result, Err(DecisionSufficiencyReceiptRejectV1::ReceiptExpired));
    }

    #[test]
    fn expiry_metadata_cannot_be_rebound_without_resigning() {
        let (mut receipt, binding, qualified, assessor, verifier_profile, verifier) = fixture();
        receipt.attestation.expires_at_unix = 301;
        let expected = expected(&binding, &qualified, &assessor, &verifier_profile);
        let result = validate_decision_sufficiency_receipt_v1(&receipt, &expected, 200, &verifier);
        assert_eq!(
            result,
            Err(DecisionSufficiencyReceiptRejectV1::AttestationDigestMismatch)
        );
    }

    #[test]
    fn caller_cannot_rebind_decision_need() {
        let (receipt, mut binding, qualified, assessor, verifier_profile, verifier) = fixture();
        binding.decision_need_sha256 = digest('z');
        let expected = expected(&binding, &qualified, &assessor, &verifier_profile);
        let result = validate_decision_sufficiency_receipt_v1(&receipt, &expected, 200, &verifier);
        assert_eq!(result, Err(DecisionSufficiencyReceiptRejectV1::BindingMismatch));
    }

    #[test]
    fn sufficient_receipt_with_missing_requirement_is_rejected() {
        let (mut receipt, binding, qualified, assessor, verifier_profile, verifier) = fixture();
        receipt.payload.assessment.missing_requirement_ids = vec!["requirement-2".to_owned()];
        let expected = expected(&binding, &qualified, &assessor, &verifier_profile);
        let result = validate_decision_sufficiency_receipt_v1(&receipt, &expected, 200, &verifier);
        assert_eq!(
            result,
            Err(DecisionSufficiencyReceiptRejectV1::RequirementPartitionMismatch)
        );
    }

    #[test]
    fn receipt_cannot_expose_an_unqualified_evidence_id() {
        let (mut receipt, binding, qualified, assessor, verifier_profile, verifier) = fixture();
        receipt.payload.assessment.admissible_evidence_ids.push("rogue".to_owned());
        receipt.payload.assessment.admissible_evidence_ids.sort();
        let expected = expected(&binding, &qualified, &assessor, &verifier_profile);
        let result = validate_decision_sufficiency_receipt_v1(&receipt, &expected, 200, &verifier);
        assert_eq!(
            result,
            Err(DecisionSufficiencyReceiptRejectV1::QualifiedEvidenceSetMismatch)
        );
    }

    #[test]
    fn identifier_sets_must_be_strictly_sorted() {
        let (mut receipt, binding, qualified, assessor, verifier_profile, verifier) = fixture();
        receipt.payload.assessment.admissible_evidence_ids.reverse();
        let expected = expected(&binding, &qualified, &assessor, &verifier_profile);
        let result = validate_decision_sufficiency_receipt_v1(&receipt, &expected, 200, &verifier);
        assert_eq!(
            result,
            Err(DecisionSufficiencyReceiptRejectV1::NonCanonicalIdentifierSet)
        );
    }

    #[test]
    fn bad_signature_is_rejected_by_pinned_verifier() {
        let (mut receipt, binding, qualified, assessor, verifier_profile, verifier) = fixture();
        let mut signature = STANDARD.decode(&receipt.attestation.signature_base64).unwrap();
        signature[0] ^= 0x01;
        receipt.attestation.signature_base64 = STANDARD.encode(signature);
        let expected = expected(&binding, &qualified, &assessor, &verifier_profile);
        let result = validate_decision_sufficiency_receipt_v1(&receipt, &expected, 200, &verifier);
        assert_eq!(
            result,
            Err(DecisionSufficiencyReceiptRejectV1::SignatureRejected(
                EvidenceQualificationRejectV1::SignatureRejected
            ))
        );
    }

    #[test]
    fn jcs_canonicalization_uses_sorted_object_keys() {
        let value = serde_json::json!({"z": 0, "a": 1});
        assert_eq!(
            serde_json_canonicalizer::to_vec(&value).unwrap(),
            br#"{"a":1,"z":0}"#.to_vec()
        );
    }

    #[test]
    fn duplicate_json_fields_are_rejected() {
        let (receipt, _binding, _qualified, _assessor, _verifier_profile, verifier) = fixture();
        let wire = serde_json::to_string(&receipt).unwrap();
        let duplicated = wire.replacen(
            "{\"schema_version\":\"mace_decision_sufficiency_receipt_v1\",",
            "{\"schema_version\":\"mace_decision_sufficiency_receipt_v1\",\"schema_version\":\"mace_decision_sufficiency_receipt_v1\",",
            1,
        );
        let result = parse_and_verify_decision_sufficiency_receipt_v1(
            duplicated.as_bytes(),
            &expected(
                &_binding,
                &_qualified,
                &_assessor,
                &_verifier_profile,
            ),
            200,
            &verifier,
        );
        assert_eq!(result, Err(DecisionSufficiencyReceiptRejectV1::InvalidJson));
    }
}
