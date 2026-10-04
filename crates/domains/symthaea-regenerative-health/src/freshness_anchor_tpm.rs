// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deployment-neutral contract for TPM NV-counter freshness evidence.
//!
//! This module intentionally does not speak TPM wire protocol. A concrete
//! platform adapter must supply cryptographically verified TPM evidence.
//! The contract makes the security-relevant identity and freshness bindings
//! explicit before that evidence can participate in the freshness-anchor path.
//!
//! TCG TPM 2.0 defines an NV counter as an 8-octet value modified with
//! TPM2_NV_Increment(). The NV Index Name is derived from the NV public area,
//! so the numeric counter value alone is not a sufficient identity for the
//! protected state.

use serde::{Deserialize, Serialize};

use crate::freshness_anchor_assurance::{
    FreshnessAnchorBacking, FreshnessAnchorEvidenceKind, FreshnessAnchorProfile,
    FreshnessAnchorVerificationReceipt,
};

pub const SCHEMA_VERSION: &str = "0.1";
pub const TPM_QUOTE_NONCE_BYTES: usize = 32;

/// Fresh verifier-generated nonce used as TPM2_Quote qualifying data.
///
/// The nonce is deliberately 256 bits so it exceeds the freshness entropy
/// floor used by current RATS interaction guidance. It is never serialized
/// into the freshness receipt itself; only its domain-separated digest is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TpmQuoteChallenge {
    pub nonce: [u8; TPM_QUOTE_NONCE_BYTES],
}

impl TpmQuoteChallenge {
    pub fn new(nonce: [u8; TPM_QUOTE_NONCE_BYTES]) -> Option<Self> {
        if nonce.iter().all(|byte| *byte == 0) {
            None
        } else {
            Some(Self { nonce })
        }
    }

    pub fn generate() -> Result<Self, getrandom::Error> {
        loop {
            let mut nonce = [0u8; TPM_QUOTE_NONCE_BYTES];
            getrandom::getrandom(&mut nonce)?;
            if let Some(challenge) = Self::new(nonce) {
                return Ok(challenge);
            }
        }
    }

    pub fn digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea:freshness-tpm-quote-challenge:v1\\0");
        hasher.update(&self.nonce);
        hasher.finalize().to_hex().to_string()
    }
}



#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TpmNvCounterEvidence {
    /// Stable TPM/attesting-environment identity commitment.
    pub tpm_identity_digest: String,
    /// Digest of the TPM NV Index Name, not merely the numeric handle.
    pub nv_index_name_digest: String,
    /// Digest of the exact TPMS_NV_PUBLIC used to derive/verify the Index Name.
    pub nv_public_digest: String,
    /// Digest of the authorization policy required for the counter.
    pub auth_policy_digest: String,
    /// Identifier for the attestation key used to authenticate the evidence.
    pub attestation_key_id_digest: String,
    /// Digest of the verifier challenge/freshness handle bound into the evidence.
    pub quote_nonce_digest: String,
    /// Digest of the TPM quote or equivalent attestation statement.
    pub quote_digest: String,
    /// Digest binding the quote to the measured platform state required by policy.
    pub pcr_binding_digest: String,
    /// Counter value read from the exact NV counter described above.
    pub counter_value: u64,
}

impl TpmNvCounterEvidence {
    pub fn validate_structure(&self) -> bool {
        [
            &self.tpm_identity_digest,
            &self.nv_index_name_digest,
            &self.nv_public_digest,
            &self.auth_policy_digest,
            &self.attestation_key_id_digest,
            &self.quote_nonce_digest,
            &self.quote_digest,
            &self.pcr_binding_digest,
        ]
        .iter()
        .all(|value| !value.trim().is_empty())
    }

    /// Validate the evidence envelope against the exact freshness receipt.
    ///
    /// This is structural only. It does not verify the TPM quote signature,
    /// PCR values, authorization policy, or device identity; those remain the
    /// concrete platform verifier's responsibility.
    pub fn validate_challenge(
        &self,
        challenge: &TpmQuoteChallenge,
        receipt: &FreshnessAnchorVerificationReceipt,
    ) -> bool {
        self.quote_nonce_digest == challenge.digest()
            && receipt.freshness_handle_digest == challenge.digest()
    }

    pub fn validate_against_receipt(
        &self,
        profile: &FreshnessAnchorProfile,
        receipt: &FreshnessAnchorVerificationReceipt,
    ) -> bool {
        self.validate_structure()
            && matches!(profile.backing, FreshnessAnchorBacking::HardwareProtected)
            && matches!(
                &receipt.evidence_kind,
                FreshnessAnchorEvidenceKind::HardwareMonotonicCounter {
                    observed_counter,
                    ..
                } if *observed_counter == self.counter_value
            )
            && receipt.generation == self.counter_value
            && receipt.freshness_handle_digest == self.quote_nonce_digest
            && receipt.evidence_digest == self.binding_digest()
    }

    pub fn as_evidence_kind(&self) -> FreshnessAnchorEvidenceKind {
        FreshnessAnchorEvidenceKind::HardwareMonotonicCounter {
            backend_identity_digest: self.tpm_identity_digest.clone(),
            counter_namespace_digest: self.nv_index_name_digest.clone(),
            observed_counter: self.counter_value,
        }
    }

    /// Deterministic commitment used when the external verifier signs an
    /// attestation result over this evidence statement.
    pub fn binding_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea:freshness-tpm-nv-counter-evidence:v1\\0");
        fn write_string(hasher: &mut blake3::Hasher, value: &str) {
            hasher.update(&(value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        write_string(&mut hasher, &self.tpm_identity_digest);
        write_string(&mut hasher, &self.nv_index_name_digest);
        write_string(&mut hasher, &self.nv_public_digest);
        write_string(&mut hasher, &self.auth_policy_digest);
        write_string(&mut hasher, &self.attestation_key_id_digest);
        write_string(&mut hasher, &self.quote_nonce_digest);
        write_string(&mut hasher, &self.quote_digest);
        write_string(&mut hasher, &self.pcr_binding_digest);
        hasher.update(&self.counter_value.to_le_bytes());
        hasher.finalize().to_hex().to_string()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TpmNvCounterVerificationError {
    InvalidEvidence,
    BackingMismatch,
    CounterGenerationMismatch,
    HandleMismatch,
    EvidenceIdentityMismatch,
    EvidenceDigestMismatch,
    QuoteVerificationFailed,
    ChallengeDigestMismatch,
}

pub trait TpmNvCounterEvidenceVerifier {
    fn verify(
        &self,
        evidence: &TpmNvCounterEvidence,
        profile: &FreshnessAnchorProfile,
        receipt: &FreshnessAnchorVerificationReceipt,
    ) -> Result<(), TpmNvCounterVerificationError>;
}

/// Structural gate followed by the deployment-specific cryptographic verifier.
pub fn verify_tpm_nv_counter<V: TpmNvCounterEvidenceVerifier>(
    evidence: &TpmNvCounterEvidence,
    profile: &FreshnessAnchorProfile,
    receipt: &FreshnessAnchorVerificationReceipt,
    verifier: &V,
) -> Result<FreshnessAnchorEvidenceKind, TpmNvCounterVerificationError> {
    if !evidence.validate_structure() {
        return Err(TpmNvCounterVerificationError::InvalidEvidence);
    }
    if !matches!(profile.backing, FreshnessAnchorBacking::HardwareProtected) {
        return Err(TpmNvCounterVerificationError::BackingMismatch);
    }
    if receipt.generation != evidence.counter_value
        || receipt.evidence_kind.observed_sequence() != evidence.counter_value
    {
        return Err(TpmNvCounterVerificationError::CounterGenerationMismatch);
    }
    if receipt.freshness_handle_digest != evidence.quote_nonce_digest {
        return Err(TpmNvCounterVerificationError::HandleMismatch);
    }
    if receipt.evidence_digest != evidence.binding_digest() {
        return Err(TpmNvCounterVerificationError::EvidenceDigestMismatch);
    }
    let FreshnessAnchorEvidenceKind::HardwareMonotonicCounter {
        backend_identity_digest,
        counter_namespace_digest,
        observed_counter,
    } = &receipt.evidence_kind
    else {
        return Err(TpmNvCounterVerificationError::BackingMismatch);
    };
    if backend_identity_digest != &evidence.tpm_identity_digest
        || counter_namespace_digest != &evidence.nv_index_name_digest
        || *observed_counter != evidence.counter_value
    {
        return Err(TpmNvCounterVerificationError::EvidenceIdentityMismatch);
    }

    verifier.verify(evidence, profile, receipt)?;
    Ok(evidence.as_evidence_kind())
}

pub fn verify_tpm_nv_counter_with_challenge<V: TpmNvCounterEvidenceVerifier>(
    evidence: &TpmNvCounterEvidence,
    challenge: &TpmQuoteChallenge,
    profile: &FreshnessAnchorProfile,
    receipt: &FreshnessAnchorVerificationReceipt,
    verifier: &V,
) -> Result<FreshnessAnchorEvidenceKind, TpmNvCounterVerificationError> {
    if !evidence.validate_challenge(challenge, receipt) {
        return Err(TpmNvCounterVerificationError::ChallengeDigestMismatch);
    }
    verify_tpm_nv_counter(evidence, profile, receipt, verifier)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::freshness_anchor_assurance::FreshnessAnchorCapabilities;

    fn profile() -> FreshnessAnchorProfile {
        FreshnessAnchorProfile::new(
            FreshnessAnchorBacking::HardwareProtected,
            FreshnessAnchorCapabilities::authoritative(),
            "tpm:nv:0x1500016",
        )
        .unwrap()
    }

    fn evidence(counter_value: u64) -> TpmNvCounterEvidence {
        TpmNvCounterEvidence {
            tpm_identity_digest: "tpm-identity".into(),
            nv_index_name_digest: "nv-name".into(),
            nv_public_digest: "nv-public".into(),
            auth_policy_digest: "auth-policy".into(),
            attestation_key_id_digest: "ak-id".into(),
            quote_nonce_digest: "quote-handle".into(),
            quote_digest: "quote".into(),
            pcr_binding_digest: "pcr-binding".into(),
            counter_value,
        }
    }

    fn receipt(counter_value: u64) -> FreshnessAnchorVerificationReceipt {
        let profile = profile();
        FreshnessAnchorVerificationReceipt {
            schema_version: SCHEMA_VERSION.into(),
            profile_fingerprint: profile.fingerprint(),
            receiver_id: "receiver-1".into(),
            generation: counter_value,
            state_fingerprint: "state-1".into(),
            recovery_policy_fingerprint: "policy".into(),
            authority_reference: "authority".into(),
            authority_statement_digest: "authority-statement".into(),
            authentication_binding: "auth-binding".into(),
            verifier_reference: "verifier-1".into(),
            verifier_policy_digest: "verifier-policy".into(),
            reference_values_digest: "reference-values".into(),
            trust_anchor_set_digest: "trust-anchors".into(),
            freshness_handle_digest: "quote-handle".into(),
            evidence_reference: "evidence".into(),
            evidence_digest: evidence(counter_value).binding_digest(),
            evidence_kind: FreshnessAnchorEvidenceKind::HardwareMonotonicCounter {
                backend_identity_digest: "tpm-identity".into(),
                counter_namespace_digest: "nv-name".into(),
                observed_counter: counter_value,
            },
        }
    }

    struct Accept;
    impl TpmNvCounterEvidenceVerifier for Accept {
        fn verify(
            &self,
            _: &TpmNvCounterEvidence,
            _: &FreshnessAnchorProfile,
            _: &FreshnessAnchorVerificationReceipt,
        ) -> Result<(), TpmNvCounterVerificationError> {
            Ok(())
        }
    }

    #[test]
    fn nonzero_challenge_has_stable_digest() {
        let challenge = TpmQuoteChallenge::new([7u8; TPM_QUOTE_NONCE_BYTES]).unwrap();
        assert_eq!(challenge.digest().len(), 64);
        assert_eq!(challenge.digest(), challenge.digest());
    }

    #[test]
    fn zero_challenge_is_rejected() {
        assert!(TpmQuoteChallenge::new([0u8; TPM_QUOTE_NONCE_BYTES]).is_none());
    }

    #[test]
    fn challenge_must_bind_quote_and_receipt_handles() {
        let evidence = evidence(7);
        let receipt = receipt(7);
        let challenge = TpmQuoteChallenge::new([9u8; TPM_QUOTE_NONCE_BYTES]).unwrap();
        assert!(!evidence.validate_challenge(&challenge, &receipt));
        assert_eq!(
            verify_tpm_nv_counter_with_challenge(
                &evidence,
                &challenge,
                &profile(),
                &receipt,
                &Accept
            )
            .unwrap_err(),
            TpmNvCounterVerificationError::ChallengeDigestMismatch
        );
    }

    #[test]
    fn challenge_mismatch_is_rejected_before_external_verifier() {
        let challenge = TpmQuoteChallenge::new([9u8; TPM_QUOTE_NONCE_BYTES]).unwrap();
        let evidence = evidence(7);
        let receipt = receipt(7);

        struct MustNotRun;
        impl TpmNvCounterEvidenceVerifier for MustNotRun {
            fn verify(
                &self,
                _: &TpmNvCounterEvidence,
                _: &FreshnessAnchorProfile,
                _: &FreshnessAnchorVerificationReceipt,
            ) -> Result<(), TpmNvCounterVerificationError> {
                panic!("external verifier must not run before challenge validation");
            }
        }

        assert_eq!(
            verify_tpm_nv_counter_with_challenge(
                &evidence,
                &challenge,
                &profile(),
                &receipt,
                &MustNotRun,
            )
            .unwrap_err(),
            TpmNvCounterVerificationError::ChallengeDigestMismatch
        );
    }

    #[test]
    fn matching_challenge_allows_tpm_evidence_verification() {
        let challenge = TpmQuoteChallenge::new([7u8; TPM_QUOTE_NONCE_BYTES]).unwrap();
        let mut evidence = evidence(7);
        let digest = challenge.digest();
        evidence.quote_nonce_digest = digest.clone();
        let mut receipt = receipt(7);
        receipt.freshness_handle_digest = digest;
        receipt.evidence_digest = evidence.binding_digest();

        assert!(verify_tpm_nv_counter_with_challenge(
            &evidence,
            &challenge,
            &profile(),
            &receipt,
            &Accept,
        ).is_ok());
    }

    #[test]
    fn exact_counter_identity_round_trips() {
        let evidence = evidence(7);
        let receipt = receipt(7);
        assert!(evidence.validate_structure());
        assert!(evidence.validate_against_receipt(&profile(), &receipt));
        assert!(verify_tpm_nv_counter(&evidence, &profile(), &receipt, &Accept).is_ok());
    }

    #[test]
    fn lower_counter_cannot_back_generation() {
        let evidence = evidence(6);
        let receipt = receipt(7);
        assert_eq!(
            verify_tpm_nv_counter(&evidence, &profile(), &receipt, &Accept).unwrap_err(),
            TpmNvCounterVerificationError::CounterGenerationMismatch
        );
    }

    #[test]
    fn typed_evidence_identity_must_match_tpm_identity() {
        let evidence = evidence(7);
        let mut receipt = receipt(7);
        receipt.evidence_kind = FreshnessAnchorEvidenceKind::HardwareMonotonicCounter {
            backend_identity_digest: "different-tpm".into(),
            counter_namespace_digest: "nv-name".into(),
            observed_counter: 7,
        };

        assert_eq!(
            verify_tpm_nv_counter(&evidence, &profile(), &receipt, &Accept).unwrap_err(),
            TpmNvCounterVerificationError::EvidenceIdentityMismatch
        );
    }

    #[test]
    fn evidence_digest_must_match_canonical_evidence() {
        let evidence = evidence(7);
        let mut receipt = receipt(7);
        receipt.evidence_digest = "different-evidence".into();

        assert_eq!(
            verify_tpm_nv_counter(&evidence, &profile(), &receipt, &Accept).unwrap_err(),
            TpmNvCounterVerificationError::EvidenceDigestMismatch
        );
    }

    #[test]
    fn quote_handle_must_match_receipt_handle() {
        let evidence = evidence(7);
        let mut receipt = receipt(7);
        receipt.freshness_handle_digest = "different-handle".into();

        assert_eq!(
            verify_tpm_nv_counter(&evidence, &profile(), &receipt, &Accept).unwrap_err(),
            TpmNvCounterVerificationError::HandleMismatch
        );
    }

    #[test]
    fn empty_identity_is_rejected_before_verifier() {
        let mut evidence = evidence(7);
        evidence.nv_index_name_digest.clear();
        assert_eq!(
            verify_tpm_nv_counter(&evidence, &profile(), &receipt(7), &Accept).unwrap_err(),
            TpmNvCounterVerificationError::InvalidEvidence
        );
    }

    #[test]
    fn binding_digest_changes_when_quote_changes() {
        let mut evidence = evidence(7);
        let original = evidence.binding_digest();
        evidence.quote_digest = "different-quote".into();
        assert_ne!(original, evidence.binding_digest());
    }
}
