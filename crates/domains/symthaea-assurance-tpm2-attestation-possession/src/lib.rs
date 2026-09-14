// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fresh TPM2 quote-possession assurance.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_assurance_tpm2_platform_qualification::Tpm2PlatformQualificationRecord;

const CHALLENGE_DIGEST_SCHEMA: &[u8] = b"symthaea-tpm2-attestation-challenge-v1\0";
const AK_BINDING_DIGEST_SCHEMA: &[u8] = b"symthaea-tpm2-ak-binding-v1\0";
const PCR_SELECTION_DIGEST_SCHEMA: &[u8] = b"symthaea-tpm2-pcr-selection-v1\0";
const QUOTE_ARTIFACT_DIGEST_SCHEMA: &[u8] = b"symthaea-tpm2-quote-artifacts-v1\0";
const VERIFICATION_DIGEST_SCHEMA: &[u8] = b"symthaea-tpm2-quote-verification-v1\0";
const POSSESSION_DIGEST_SCHEMA: &[u8] = b"symthaea-tpm2-attestation-possession-v1\0";
const ACCEPTANCE_DIGEST_SCHEMA: &[u8] = b"symthaea-tpm2-attestation-acceptance-v1\0";

pub const POSSESSION_SCOPE: &str =
    "fresh_ak_possession_over_selected_pcrs_not_ek_chain_or_pcr_goodness_v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PcrBankSelection {
    pub hash_alg: String,
    pub pcrs: Vec<u8>,
}

impl PcrBankSelection {
    pub fn validate(&self) -> bool {
        !self.hash_alg.trim().is_empty()
            && self.hash_alg == self.hash_alg.trim()
            && self.hash_alg == self.hash_alg.to_ascii_lowercase()
            && !self.pcrs.is_empty()
            && self.pcrs.len() <= 24
            && self.pcrs.iter().all(|pcr| *pcr <= 23)
            && strictly_increasing(&self.pcrs)
    }
}

pub fn pcr_selection_digest(selection: &[PcrBankSelection]) -> Option<String> {
    if selection.is_empty()
        || !selection.iter().all(PcrBankSelection::validate)
        || !selection
            .windows(2)
            .all(|pair| pair[0].hash_alg < pair[1].hash_alg)
    {
        return None;
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(PCR_SELECTION_DIGEST_SCHEMA);
    for bank in selection {
        push_field(&mut hasher, &bank.hash_alg);
        hasher.update(&(bank.pcrs.len() as u64).to_le_bytes());
        hasher.update(&bank.pcrs);
    }
    Some(format!("blake3:{}", hasher.finalize().to_hex()))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttestationChallenge {
    pub schema_version: String,
    pub challenge_id: String,
    /// Exact 32-byte challenge nonce encoded as 64 hexadecimal characters.
    pub nonce_hex: String,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub verifier_ref: String,
    pub pcr_selection: Vec<PcrBankSelection>,
    pub evidence_refs: Vec<String>,
}

impl AttestationChallenge {
    pub fn validate(&self) -> bool {
        !self.schema_version.trim().is_empty()
            && !self.challenge_id.trim().is_empty()
            && valid_hex_exact(&self.nonce_hex, 64)
            && self.expires_at_ms > self.issued_at_ms
            && !self.verifier_ref.trim().is_empty()
            && pcr_selection_digest(&self.pcr_selection).is_some()
            && nonempty_refs(&self.evidence_refs)
    }

    pub fn challenge_digest(&self) -> String {
        let mut refs = self.evidence_refs.clone();
        refs.sort();
        let mut hasher = blake3::Hasher::new();
        hasher.update(CHALLENGE_DIGEST_SCHEMA);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.challenge_id);
        push_field(&mut hasher, &self.nonce_hex.to_ascii_lowercase());
        push_field(&mut hasher, &self.issued_at_ms.to_string());
        push_field(&mut hasher, &self.expires_at_ms.to_string());
        push_field(&mut hasher, &self.verifier_ref);
        push_field(
            &mut hasher,
            &pcr_selection_digest(&self.pcr_selection).unwrap_or_default(),
        );
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttestationKeyBinding {
    pub schema_version: String,
    pub ak_id: String,
    pub ak_name_hex: String,
    pub ak_qualified_name_hex: String,
    pub public_key_digest: String,
    pub name_alg: String,
    pub signing_scheme: String,
    pub fixed_tpm: bool,
    pub restricted_signing: bool,
    pub reviewed_at_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl AttestationKeyBinding {
    pub fn validate(&self) -> bool {
        !self.schema_version.trim().is_empty()
            && !self.ak_id.trim().is_empty()
            && valid_hex_even(&self.ak_name_hex)
            && valid_hex_even(&self.ak_qualified_name_hex)
            && valid_digest(&self.public_key_digest)
            && !self.name_alg.trim().is_empty()
            && !self.signing_scheme.trim().is_empty()
            && self.reviewed_at_ms > 0
            && nonempty_refs(&self.evidence_refs)
    }

    pub fn binding_digest(&self) -> String {
        let mut refs = self.evidence_refs.clone();
        refs.sort();
        let mut hasher = blake3::Hasher::new();
        hasher.update(AK_BINDING_DIGEST_SCHEMA);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.ak_id);
        push_field(&mut hasher, &self.ak_name_hex.to_ascii_lowercase());
        push_field(
            &mut hasher,
            &self.ak_qualified_name_hex.to_ascii_lowercase(),
        );
        push_field(&mut hasher, &self.public_key_digest);
        push_field(&mut hasher, &self.name_alg);
        push_field(&mut hasher, &self.signing_scheme);
        push_field(&mut hasher, if self.fixed_tpm { "1" } else { "0" });
        push_field(
            &mut hasher,
            if self.restricted_signing { "1" } else { "0" },
        );
        push_field(&mut hasher, &self.reviewed_at_ms.to_string());
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tpm2QuoteArtifacts {
    pub schema_version: String,
    pub quote_id: String,
    pub challenge_id: String,
    pub platform_qualification_digest: String,
    pub qualifying_data_hex: String,
    pub quote_message_digest: String,
    pub signature_digest: String,
    pub pcr_values_digest: String,
    pub ak_public_key_digest: String,
    pub pcr_selection_digest: String,
    pub collected_at_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl Tpm2QuoteArtifacts {
    pub fn validate(&self) -> bool {
        !self.schema_version.trim().is_empty()
            && !self.quote_id.trim().is_empty()
            && !self.challenge_id.trim().is_empty()
            && valid_digest(&self.platform_qualification_digest)
            && valid_hex_exact(&self.qualifying_data_hex, 64)
            && valid_digest(&self.quote_message_digest)
            && valid_digest(&self.signature_digest)
            && valid_digest(&self.pcr_values_digest)
            && valid_digest(&self.ak_public_key_digest)
            && valid_digest(&self.pcr_selection_digest)
            && nonempty_refs(&self.evidence_refs)
    }

    pub fn artifact_digest(&self) -> String {
        let mut refs = self.evidence_refs.clone();
        refs.sort();
        let mut hasher = blake3::Hasher::new();
        hasher.update(QUOTE_ARTIFACT_DIGEST_SCHEMA);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.quote_id);
        push_field(&mut hasher, &self.challenge_id);
        push_field(&mut hasher, &self.platform_qualification_digest);
        push_field(&mut hasher, &self.qualifying_data_hex.to_ascii_lowercase());
        push_field(&mut hasher, &self.quote_message_digest);
        push_field(&mut hasher, &self.signature_digest);
        push_field(&mut hasher, &self.pcr_values_digest);
        push_field(&mut hasher, &self.ak_public_key_digest);
        push_field(&mut hasher, &self.pcr_selection_digest);
        push_field(&mut hasher, &self.collected_at_ms.to_string());
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuoteVerificationReceipt {
    pub schema_version: String,
    pub receipt_id: String,
    pub verifier_ref: String,
    pub verification_tool_ref: String,
    pub verification_tool_digest: String,
    pub verified_at_ms: u64,
    pub challenge_digest: String,
    pub ak_binding_digest: String,
    pub quote_artifact_digest: String,
    pub signature_valid: bool,
    pub qualifying_data_matches: bool,
    pub pcr_digest_matches: bool,
    pub pcr_selection_matches: bool,
    pub attestation_magic_valid: bool,
    pub attestation_type_quote: bool,
    pub evidence_refs: Vec<String>,
}

impl QuoteVerificationReceipt {
    pub fn validate(&self) -> bool {
        !self.schema_version.trim().is_empty()
            && !self.receipt_id.trim().is_empty()
            && !self.verifier_ref.trim().is_empty()
            && !self.verification_tool_ref.trim().is_empty()
            && valid_digest(&self.verification_tool_digest)
            && valid_digest(&self.challenge_digest)
            && valid_digest(&self.ak_binding_digest)
            && valid_digest(&self.quote_artifact_digest)
            && self.signature_valid
            && self.qualifying_data_matches
            && self.pcr_digest_matches
            && self.pcr_selection_matches
            && self.attestation_magic_valid
            && self.attestation_type_quote
            && nonempty_refs(&self.evidence_refs)
    }

    pub fn receipt_digest(&self) -> String {
        let mut refs = self.evidence_refs.clone();
        refs.sort();
        let mut hasher = blake3::Hasher::new();
        hasher.update(VERIFICATION_DIGEST_SCHEMA);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.receipt_id);
        push_field(&mut hasher, &self.verifier_ref);
        push_field(&mut hasher, &self.verification_tool_ref);
        push_field(&mut hasher, &self.verification_tool_digest);
        push_field(&mut hasher, &self.verified_at_ms.to_string());
        push_field(&mut hasher, &self.challenge_digest);
        push_field(&mut hasher, &self.ak_binding_digest);
        push_field(&mut hasher, &self.quote_artifact_digest);
        for value in [
            self.signature_valid,
            self.qualifying_data_matches,
            self.pcr_digest_matches,
            self.pcr_selection_matches,
            self.attestation_magic_valid,
            self.attestation_type_quote,
        ] {
            push_field(&mut hasher, if value { "1" } else { "0" });
        }
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttestationPossessionPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_platform_qualification_digest: String,
    pub expected_ak_binding_digest: String,
    pub expected_pcr_selection_digest: String,
    pub expected_verifier_ref: String,
    pub expected_verification_tool_digest: String,
    pub max_challenge_lifetime_ms: u64,
    pub max_quote_to_verification_ms: u64,
    pub require_fixed_tpm: bool,
    pub require_restricted_signing: bool,
    pub evidence_refs: Vec<String>,
}

impl AttestationPossessionPolicy {
    pub fn validate(&self) -> bool {
        !self.schema_version.trim().is_empty()
            && !self.policy_id.trim().is_empty()
            && valid_digest(&self.expected_platform_qualification_digest)
            && valid_digest(&self.expected_ak_binding_digest)
            && valid_digest(&self.expected_pcr_selection_digest)
            && !self.expected_verifier_ref.trim().is_empty()
            && valid_digest(&self.expected_verification_tool_digest)
            && self.max_challenge_lifetime_ms > 0
            && self.max_quote_to_verification_ms > 0
            && nonempty_refs(&self.evidence_refs)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttestationPossessionRecord {
    pub schema_version: String,
    pub policy_id: String,
    pub platform_qualification_digest: String,
    pub challenge_id: String,
    pub challenge_digest: String,
    pub ak_binding_digest: String,
    pub quote_artifact_digest: String,
    pub verification_receipt_digest: String,
    pub pcr_selection_digest: String,
    pub qualified_at_ms: u64,
    pub scope: String,
    pub evidence_refs: Vec<String>,
}

impl AttestationPossessionRecord {
    pub fn possession_digest(&self) -> String {
        let mut refs = self.evidence_refs.clone();
        refs.sort();
        let mut hasher = blake3::Hasher::new();
        hasher.update(POSSESSION_DIGEST_SCHEMA);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.policy_id);
        push_field(&mut hasher, &self.platform_qualification_digest);
        push_field(&mut hasher, &self.challenge_id);
        push_field(&mut hasher, &self.challenge_digest);
        push_field(&mut hasher, &self.ak_binding_digest);
        push_field(&mut hasher, &self.quote_artifact_digest);
        push_field(&mut hasher, &self.verification_receipt_digest);
        push_field(&mut hasher, &self.pcr_selection_digest);
        push_field(&mut hasher, &self.qualified_at_ms.to_string());
        push_field(&mut hasher, &self.scope);
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttestationPossessionError {
    InvalidPolicy,
    InvalidChallenge,
    ChallengeLifetimeExceeded,
    ChallengeVerifierMismatch,
    QualificationOutsideChallengeWindow,
    PlatformQualificationMismatch,
    PlatformQualifiedAfterChallenge,
    InvalidAkBinding,
    AkBindingMismatch,
    AkReviewedAfterChallenge,
    AkNotFixedToTpm,
    AkNotRestrictedSigning,
    VerifierNotIndependent,
    PcrSelectionMismatch,
    InvalidQuoteArtifacts,
    QuoteChallengeMismatch,
    QuoteNonceMismatch,
    QuotePlatformMismatch,
    QuoteAkMismatch,
    QuoteOutsideChallengeWindow,
    InvalidVerificationReceipt,
    VerificationVerifierMismatch,
    VerificationToolMismatch,
    VerificationBindingMismatch,
    VerificationBeforeQuote,
    VerificationLagExceeded,
    VerificationAfterQualification,
}

pub fn qualify_attestation_possession(
    policy: &AttestationPossessionPolicy,
    platform: &Tpm2PlatformQualificationRecord,
    challenge: &AttestationChallenge,
    ak: &AttestationKeyBinding,
    quote: &Tpm2QuoteArtifacts,
    verification: &QuoteVerificationReceipt,
    qualified_at_ms: u64,
) -> Result<AttestationPossessionRecord, AttestationPossessionError> {
    if !policy.validate() {
        return Err(AttestationPossessionError::InvalidPolicy);
    }
    if !challenge.validate() {
        return Err(AttestationPossessionError::InvalidChallenge);
    }
    if challenge.expires_at_ms - challenge.issued_at_ms > policy.max_challenge_lifetime_ms {
        return Err(AttestationPossessionError::ChallengeLifetimeExceeded);
    }
    if challenge.verifier_ref != policy.expected_verifier_ref {
        return Err(AttestationPossessionError::ChallengeVerifierMismatch);
    }
    if qualified_at_ms < challenge.issued_at_ms || qualified_at_ms > challenge.expires_at_ms {
        return Err(AttestationPossessionError::QualificationOutsideChallengeWindow);
    }

    let platform_digest = platform.qualification_digest();
    if platform_digest != policy.expected_platform_qualification_digest {
        return Err(AttestationPossessionError::PlatformQualificationMismatch);
    }
    if platform.qualified_at_ms > challenge.issued_at_ms {
        return Err(AttestationPossessionError::PlatformQualifiedAfterChallenge);
    }

    if !ak.validate() {
        return Err(AttestationPossessionError::InvalidAkBinding);
    }
    let ak_digest = ak.binding_digest();
    if ak_digest != policy.expected_ak_binding_digest {
        return Err(AttestationPossessionError::AkBindingMismatch);
    }
    if ak.reviewed_at_ms > challenge.issued_at_ms {
        return Err(AttestationPossessionError::AkReviewedAfterChallenge);
    }
    if policy.require_fixed_tpm && !ak.fixed_tpm {
        return Err(AttestationPossessionError::AkNotFixedToTpm);
    }
    if policy.require_restricted_signing && !ak.restricted_signing {
        return Err(AttestationPossessionError::AkNotRestrictedSigning);
    }
    if challenge.verifier_ref == platform.trust_store_ref
        || challenge.verifier_ref == platform.adapter_id
        || challenge.verifier_ref == ak.ak_id
    {
        return Err(AttestationPossessionError::VerifierNotIndependent);
    }

    let selection_digest = pcr_selection_digest(&challenge.pcr_selection)
        .ok_or(AttestationPossessionError::InvalidChallenge)?;
    if selection_digest != policy.expected_pcr_selection_digest {
        return Err(AttestationPossessionError::PcrSelectionMismatch);
    }

    if !quote.validate() {
        return Err(AttestationPossessionError::InvalidQuoteArtifacts);
    }
    if quote.challenge_id != challenge.challenge_id {
        return Err(AttestationPossessionError::QuoteChallengeMismatch);
    }
    if !quote.qualifying_data_hex.eq_ignore_ascii_case(&challenge.nonce_hex) {
        return Err(AttestationPossessionError::QuoteNonceMismatch);
    }
    if quote.platform_qualification_digest != platform_digest {
        return Err(AttestationPossessionError::QuotePlatformMismatch);
    }
    if quote.ak_public_key_digest != ak.public_key_digest {
        return Err(AttestationPossessionError::QuoteAkMismatch);
    }
    if quote.pcr_selection_digest != selection_digest {
        return Err(AttestationPossessionError::PcrSelectionMismatch);
    }
    if quote.collected_at_ms < challenge.issued_at_ms
        || quote.collected_at_ms > challenge.expires_at_ms
    {
        return Err(AttestationPossessionError::QuoteOutsideChallengeWindow);
    }

    if !verification.validate() {
        return Err(AttestationPossessionError::InvalidVerificationReceipt);
    }
    if verification.verifier_ref != challenge.verifier_ref {
        return Err(AttestationPossessionError::VerificationVerifierMismatch);
    }
    if verification.verification_tool_digest != policy.expected_verification_tool_digest {
        return Err(AttestationPossessionError::VerificationToolMismatch);
    }
    if verification.challenge_digest != challenge.challenge_digest()
        || verification.ak_binding_digest != ak_digest
        || verification.quote_artifact_digest != quote.artifact_digest()
    {
        return Err(AttestationPossessionError::VerificationBindingMismatch);
    }
    if verification.verified_at_ms < quote.collected_at_ms {
        return Err(AttestationPossessionError::VerificationBeforeQuote);
    }
    if verification.verified_at_ms - quote.collected_at_ms > policy.max_quote_to_verification_ms {
        return Err(AttestationPossessionError::VerificationLagExceeded);
    }
    if verification.verified_at_ms > qualified_at_ms {
        return Err(AttestationPossessionError::VerificationAfterQualification);
    }

    Ok(AttestationPossessionRecord {
        schema_version: "1".into(),
        policy_id: policy.policy_id.clone(),
        platform_qualification_digest: platform_digest,
        challenge_id: challenge.challenge_id.clone(),
        challenge_digest: challenge.challenge_digest(),
        ak_binding_digest: ak_digest,
        quote_artifact_digest: quote.artifact_digest(),
        verification_receipt_digest: verification.receipt_digest(),
        pcr_selection_digest: selection_digest,
        qualified_at_ms,
        scope: POSSESSION_SCOPE.into(),
        evidence_refs: vec![
            format!("challenge:{}", challenge.challenge_digest()),
            format!("ak:{}", ak.binding_digest()),
            format!("quote:{}", quote.artifact_digest()),
            format!("verification:{}", verification.receipt_digest()),
        ],
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttestationAcceptanceRecord {
    pub revision: u64,
    pub challenge_id: String,
    pub nonce_digest: String,
    pub possession_record_digest: String,
    pub quote_artifact_digest: String,
    pub verification_receipt_id: String,
    pub accepted_at_ms: u64,
    pub predecessor_acceptance_digest: Option<String>,
}

impl AttestationAcceptanceRecord {
    pub fn acceptance_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(ACCEPTANCE_DIGEST_SCHEMA);
        push_field(&mut hasher, &self.revision.to_string());
        push_field(&mut hasher, &self.challenge_id);
        push_field(&mut hasher, &self.nonce_digest);
        push_field(&mut hasher, &self.possession_record_digest);
        push_field(&mut hasher, &self.quote_artifact_digest);
        push_field(&mut hasher, &self.verification_receipt_id);
        push_field(&mut hasher, &self.accepted_at_ms.to_string());
        push_field(
            &mut hasher,
            self.predecessor_acceptance_digest.as_deref().unwrap_or(""),
        );
        format!("blake3:{}", hasher.finalize().to_hex())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttestationAcceptanceLedger {
    pub records: Vec<AttestationAcceptanceRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttestationAcceptanceError {
    Qualification(AttestationPossessionError),
    AcceptanceBeforeQualification,
    AcceptanceTimeRegressed,
    ChallengeReplay,
    NonceReplay,
    QuoteReplay,
    VerificationReceiptReplay,
}

impl AttestationAcceptanceLedger {
    #[allow(clippy::too_many_arguments)]
    pub fn accept(
        &mut self,
        policy: &AttestationPossessionPolicy,
        platform: &Tpm2PlatformQualificationRecord,
        challenge: &AttestationChallenge,
        ak: &AttestationKeyBinding,
        quote: &Tpm2QuoteArtifacts,
        verification: &QuoteVerificationReceipt,
        qualified_at_ms: u64,
        accepted_at_ms: u64,
    ) -> Result<AttestationAcceptanceRecord, AttestationAcceptanceError> {
        let possession = qualify_attestation_possession(
            policy,
            platform,
            challenge,
            ak,
            quote,
            verification,
            qualified_at_ms,
        )
        .map_err(AttestationAcceptanceError::Qualification)?;

        if accepted_at_ms < qualified_at_ms {
            return Err(AttestationAcceptanceError::AcceptanceBeforeQualification);
        }
        if self
            .records
            .last()
            .is_some_and(|last| accepted_at_ms < last.accepted_at_ms)
        {
            return Err(AttestationAcceptanceError::AcceptanceTimeRegressed);
        }

        let nonce_bytes = hex_to_bytes(&challenge.nonce_hex).ok_or(
            AttestationAcceptanceError::Qualification(AttestationPossessionError::InvalidChallenge),
        )?;
        let nonce_digest = digest_bytes(&nonce_bytes);
        let quote_digest = quote.artifact_digest();

        if self
            .records
            .iter()
            .any(|record| record.challenge_id == challenge.challenge_id)
        {
            return Err(AttestationAcceptanceError::ChallengeReplay);
        }
        if self
            .records
            .iter()
            .any(|record| record.nonce_digest == nonce_digest)
        {
            return Err(AttestationAcceptanceError::NonceReplay);
        }
        if self
            .records
            .iter()
            .any(|record| record.quote_artifact_digest == quote_digest)
        {
            return Err(AttestationAcceptanceError::QuoteReplay);
        }
        if self
            .records
            .iter()
            .any(|record| record.verification_receipt_id == verification.receipt_id)
        {
            return Err(AttestationAcceptanceError::VerificationReceiptReplay);
        }

        let record = AttestationAcceptanceRecord {
            revision: self.records.len() as u64 + 1,
            challenge_id: challenge.challenge_id.clone(),
            nonce_digest,
            possession_record_digest: possession.possession_digest(),
            quote_artifact_digest: quote_digest,
            verification_receipt_id: verification.receipt_id.clone(),
            accepted_at_ms,
            predecessor_acceptance_digest: self
                .records
                .last()
                .map(AttestationAcceptanceRecord::acceptance_digest),
        };
        self.records.push(record.clone());
        Ok(record)
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

fn strictly_increasing(values: &[u8]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn nonempty_refs(refs: &[String]) -> bool {
    !refs.is_empty() && refs.iter().all(|value| !value.trim().is_empty())
}

fn valid_digest(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|digest| {
        digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
    })
}

fn valid_hex_exact(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_hex_even(value: &str) -> bool {
    !value.is_empty()
        && value.len() % 2 == 0
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn hex_to_bytes(value: &str) -> Option<Vec<u8>> {
    if !valid_hex_even(value) {
        return None;
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair).ok()?;
            u8::from_str_radix(text, 16).ok()
        })
        .collect()
}

fn digest_bytes(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes).to_hex())
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_assurance_tpm2_platform_qualification::Tpm2FixedProperties;

    fn digest(label: &str) -> String {
        digest_bytes(label.as_bytes())
    }

    fn platform() -> Tpm2PlatformQualificationRecord {
        Tpm2PlatformQualificationRecord {
            schema_version: "1".into(),
            qualification_id: "platform:q1".into(),
            adapter_id: "adapter:tpm2:1".into(),
            trust_store_ref: "trust-store:tpm2:1".into(),
            tcti: "device:/dev/tpmrm0".into(),
            nv_index: 0x0150_0016,
            nv_name: "000bdeadbeef".into(),
            before_observation_digest: digest("before"),
            after_observation_digest: digest("after"),
            before_counter_value: 42,
            after_counter_value: 42,
            fixed_properties: Tpm2FixedProperties {
                family_indicator: 0x322e3000,
                specification_revision: 185,
                manufacturer: 0x49465800,
                firmware_version_1: 1,
                firmware_version_2: 2,
                raw_output_digest: digest("getcap"),
            },
            getcap_blake3: digest("getcap-bin"),
            runtime_subject_digest: digest("runtime"),
            qualified_at_ms: 9_000,
            evidence_refs: vec!["audit:platform".into()],
        }
    }

    fn selection() -> Vec<PcrBankSelection> {
        vec![PcrBankSelection {
            hash_alg: "sha256".into(),
            pcrs: vec![0, 2, 7],
        }]
    }

    fn challenge(id: &str, nonce: &str) -> AttestationChallenge {
        AttestationChallenge {
            schema_version: "1".into(),
            challenge_id: id.into(),
            nonce_hex: nonce.into(),
            issued_at_ms: 10_000,
            expires_at_ms: 11_000,
            verifier_ref: "verifier:remote-1".into(),
            pcr_selection: selection(),
            evidence_refs: vec!["request:attestation".into()],
        }
    }

    fn ak() -> AttestationKeyBinding {
        AttestationKeyBinding {
            schema_version: "1".into(),
            ak_id: "ak:node-1".into(),
            ak_name_hex: "000b01020304".into(),
            ak_qualified_name_hex: "000b05060708".into(),
            public_key_digest: digest("ak-pub"),
            name_alg: "sha256".into(),
            signing_scheme: "ecdsa-sha256".into(),
            fixed_tpm: true,
            restricted_signing: true,
            reviewed_at_ms: 9_500,
            evidence_refs: vec!["review:ak-public".into()],
        }
    }

    fn quote(challenge: &AttestationChallenge) -> Tpm2QuoteArtifacts {
        Tpm2QuoteArtifacts {
            schema_version: "1".into(),
            quote_id: format!("quote:{}", challenge.challenge_id),
            challenge_id: challenge.challenge_id.clone(),
            platform_qualification_digest: platform().qualification_digest(),
            qualifying_data_hex: challenge.nonce_hex.clone(),
            quote_message_digest: digest("quote-message"),
            signature_digest: digest("quote-signature"),
            pcr_values_digest: digest("pcr-values"),
            ak_public_key_digest: ak().public_key_digest,
            pcr_selection_digest: pcr_selection_digest(&challenge.pcr_selection).unwrap(),
            collected_at_ms: 10_200,
            evidence_refs: vec!["artifact:quote-bundle".into()],
        }
    }

    fn verification(
        challenge: &AttestationChallenge,
        quote: &Tpm2QuoteArtifacts,
    ) -> QuoteVerificationReceipt {
        QuoteVerificationReceipt {
            schema_version: "1".into(),
            receipt_id: format!("verification:{}", challenge.challenge_id),
            verifier_ref: challenge.verifier_ref.clone(),
            verification_tool_ref: "tool:tpm2-checkquote".into(),
            verification_tool_digest: digest("tpm2-checkquote"),
            verified_at_ms: 10_300,
            challenge_digest: challenge.challenge_digest(),
            ak_binding_digest: ak().binding_digest(),
            quote_artifact_digest: quote.artifact_digest(),
            signature_valid: true,
            qualifying_data_matches: true,
            pcr_digest_matches: true,
            pcr_selection_matches: true,
            attestation_magic_valid: true,
            attestation_type_quote: true,
            evidence_refs: vec!["audit:quote-verification".into()],
        }
    }

    fn policy(challenge: &AttestationChallenge) -> AttestationPossessionPolicy {
        AttestationPossessionPolicy {
            schema_version: "1".into(),
            policy_id: "policy:attestation:1".into(),
            expected_platform_qualification_digest: platform().qualification_digest(),
            expected_ak_binding_digest: ak().binding_digest(),
            expected_pcr_selection_digest: pcr_selection_digest(&challenge.pcr_selection).unwrap(),
            expected_verifier_ref: challenge.verifier_ref.clone(),
            expected_verification_tool_digest: digest("tpm2-checkquote"),
            max_challenge_lifetime_ms: 2_000,
            max_quote_to_verification_ms: 500,
            require_fixed_tpm: true,
            require_restricted_signing: true,
            evidence_refs: vec!["review:attestation-policy".into()],
        }
    }

    #[test]
    fn fresh_exact_quote_establishes_narrow_ak_possession() {
        let challenge = challenge("challenge:1", &"ab".repeat(32));
        let quote = quote(&challenge);
        let verification = verification(&challenge, &quote);
        let record = qualify_attestation_possession(
            &policy(&challenge),
            &platform(),
            &challenge,
            &ak(),
            &quote,
            &verification,
            10_400,
        )
        .unwrap();
        assert_eq!(record.scope, POSSESSION_SCOPE);
        assert!(!record.grants_physical_authority());
    }

    #[test]
    fn nonce_substitution_is_rejected() {
        let challenge = challenge("challenge:1", &"ab".repeat(32));
        let mut quote = quote(&challenge);
        quote.qualifying_data_hex = "cd".repeat(32);
        let verification = verification(&challenge, &quote);
        assert_eq!(
            qualify_attestation_possession(
                &policy(&challenge),
                &platform(),
                &challenge,
                &ak(),
                &quote,
                &verification,
                10_400,
            ),
            Err(AttestationPossessionError::QuoteNonceMismatch)
        );
    }

    #[test]
    fn pcr_selection_substitution_is_rejected() {
        let challenge = challenge("challenge:1", &"ab".repeat(32));
        let mut quote = quote(&challenge);
        quote.pcr_selection_digest = digest("other-selection");
        let verification = verification(&challenge, &quote);
        assert_eq!(
            qualify_attestation_possession(
                &policy(&challenge),
                &platform(),
                &challenge,
                &ak(),
                &quote,
                &verification,
                10_400,
            ),
            Err(AttestationPossessionError::PcrSelectionMismatch)
        );
    }

    #[test]
    fn unrestricted_or_nonfixed_ak_is_rejected_when_policy_requires_it() {
        let challenge = challenge("challenge:1", &"ab".repeat(32));
        let quote = quote(&challenge);
        let verification = verification(&challenge, &quote);
        let mut bad_ak = ak();
        bad_ak.fixed_tpm = false;
        let mut policy = policy(&challenge);
        policy.expected_ak_binding_digest = bad_ak.binding_digest();
        assert_eq!(
            qualify_attestation_possession(
                &policy,
                &platform(),
                &challenge,
                &bad_ak,
                &quote,
                &verification,
                10_400,
            ),
            Err(AttestationPossessionError::AkNotFixedToTpm)
        );
    }

    #[test]
    fn failed_independent_quote_check_is_rejected() {
        let challenge = challenge("challenge:1", &"ab".repeat(32));
        let quote = quote(&challenge);
        let mut verification = verification(&challenge, &quote);
        verification.pcr_digest_matches = false;
        assert_eq!(
            qualify_attestation_possession(
                &policy(&challenge),
                &platform(),
                &challenge,
                &ak(),
                &quote,
                &verification,
                10_400,
            ),
            Err(AttestationPossessionError::InvalidVerificationReceipt)
        );
    }

    #[test]
    fn verifier_and_tool_are_policy_bound() {
        let challenge = challenge("challenge:1", &"ab".repeat(32));
        let quote = quote(&challenge);
        let mut verification = verification(&challenge, &quote);
        verification.verification_tool_digest = digest("other-tool");
        assert_eq!(
            qualify_attestation_possession(
                &policy(&challenge),
                &platform(),
                &challenge,
                &ak(),
                &quote,
                &verification,
                10_400,
            ),
            Err(AttestationPossessionError::VerificationToolMismatch)
        );
    }

    #[test]
    fn quote_outside_freshness_window_is_rejected() {
        let challenge = challenge("challenge:1", &"ab".repeat(32));
        let mut quote = quote(&challenge);
        quote.collected_at_ms = 11_001;
        let verification = verification(&challenge, &quote);
        assert_eq!(
            qualify_attestation_possession(
                &policy(&challenge),
                &platform(),
                &challenge,
                &ak(),
                &quote,
                &verification,
                10_400,
            ),
            Err(AttestationPossessionError::QuoteOutsideChallengeWindow)
        );
    }

    #[test]
    fn accepted_challenge_and_nonce_are_one_shot() {
        let challenge = challenge("challenge:1", &"ab".repeat(32));
        let quote = quote(&challenge);
        let verification = verification(&challenge, &quote);
        let mut ledger = AttestationAcceptanceLedger::default();
        ledger
            .accept(
                &policy(&challenge),
                &platform(),
                &challenge,
                &ak(),
                &quote,
                &verification,
                10_400,
                10_500,
            )
            .unwrap();
        assert_eq!(
            ledger.accept(
                &policy(&challenge),
                &platform(),
                &challenge,
                &ak(),
                &quote,
                &verification,
                10_400,
                10_600,
            ),
            Err(AttestationAcceptanceError::ChallengeReplay)
        );

        let second = challenge("challenge:2", &"ab".repeat(32));
        let second_quote = quote(&second);
        let second_verification = verification(&second, &second_quote);
        assert_eq!(
            ledger.accept(
                &policy(&second),
                &platform(),
                &second,
                &ak(),
                &second_quote,
                &second_verification,
                10_400,
                10_700,
            ),
            Err(AttestationAcceptanceError::NonceReplay)
        );
    }
}
