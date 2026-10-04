// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Concrete cryptographic protection for freshness-anchor verifier results.
//!
//! This module does not replace deployment-specific evidence appraisal. It
//! provides one concrete way for a verifier to authenticate the exact
//! freshness verification statement produced by that appraisal.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::freshness_anchor_assurance::{
    FreshnessAnchorAssuranceError, FreshnessAnchorEvidenceVerifier, FreshnessAnchorProfile,
    FreshnessAnchorVerificationReceipt, VerifiedFreshnessAnchor,
};

pub const SCHEMA_VERSION: &str = "0.1";
pub const CRYPTOSUITE: &str = "symthaea-freshness-anchor-ed25519-v1";

const DOMAIN_SEPARATOR: &[u8] = b"symthaea:freshness-anchor-attestation:v1\\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshnessAnchorVerifierAttestation {
    pub schema_version: String,
    pub cryptosuite: String,
    pub verifier_reference: String,
    pub statement_digest: String,
    pub verifying_key: String,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreshnessAnchorVerifierTrustPolicy {
    pub expected_verifier_reference: String,
    pub expected_key_fingerprint: String,
    pub expected_trust_anchor_set_digest: String,
}

impl FreshnessAnchorVerifierTrustPolicy {
    pub fn new(
        expected_verifier_reference: impl Into<String>,
        expected_key_fingerprint: impl Into<String>,
        expected_trust_anchor_set_digest: impl Into<String>,
    ) -> Self {
        Self {
            expected_verifier_reference: expected_verifier_reference.into(),
            expected_key_fingerprint: expected_key_fingerprint.into(),
            expected_trust_anchor_set_digest: expected_trust_anchor_set_digest.into(),
        }
    }

    fn validate(&self, receipt: &FreshnessAnchorVerificationReceipt, attestation: &FreshnessAnchorVerifierAttestation) -> bool {
        !self.expected_verifier_reference.trim().is_empty()
            && !self.expected_key_fingerprint.trim().is_empty()
            && !self.expected_trust_anchor_set_digest.trim().is_empty()
            && receipt.verifier_reference == self.expected_verifier_reference
            && attestation.verifier_reference == self.expected_verifier_reference
            && receipt.trust_anchor_set_digest == self.expected_trust_anchor_set_digest
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FreshnessAnchorAttestationVerificationError {
    InvalidEnvelope,
    VerifierReferenceMismatch,
    StatementBindingMismatch,
    InvalidVerificationKey,
    InvalidSignature,
    SignatureVerificationFailed,
    AnchorVerificationFailed(FreshnessAnchorAssuranceError),
    VerifierTrustPolicyMismatch,
}

fn signed_message(statement_digest: &str) -> Vec<u8> {
    let mut message = Vec::with_capacity(DOMAIN_SEPARATOR.len() + 8 + statement_digest.len());
    message.extend_from_slice(DOMAIN_SEPARATOR);
    message.extend_from_slice(&(statement_digest.len() as u64).to_le_bytes());
    message.extend_from_slice(statement_digest.as_bytes());
    message
}

impl FreshnessAnchorVerifierAttestation {
    pub fn sign(
        verifier_reference: impl Into<String>,
        receipt: &FreshnessAnchorVerificationReceipt,
        signing_key: &SigningKey,
    ) -> Self {
        let verifier_reference = verifier_reference.into();
        let statement_digest = receipt.binding_digest();
        let signature = signing_key.sign(&signed_message(&statement_digest));

        Self {
            schema_version: SCHEMA_VERSION.into(),
            cryptosuite: CRYPTOSUITE.into(),
            verifier_reference,
            statement_digest,
            verifying_key: hex::encode(signing_key.verifying_key().as_bytes()),
            signature: signature.to_bytes().to_vec(),
        }
    }

    pub fn verify(
        &self,
        receipt: &FreshnessAnchorVerificationReceipt,
    ) -> Result<(), FreshnessAnchorAttestationVerificationError> {
        if self.schema_version != SCHEMA_VERSION
            || self.cryptosuite != CRYPTOSUITE
            || self.verifier_reference.trim().is_empty()
            || self.statement_digest.trim().is_empty()
        {
            return Err(FreshnessAnchorAttestationVerificationError::InvalidEnvelope);
        }

        if self.verifier_reference != receipt.verifier_reference {
            return Err(FreshnessAnchorAttestationVerificationError::VerifierReferenceMismatch);
        }

        if self.statement_digest != receipt.binding_digest() {
            return Err(FreshnessAnchorAttestationVerificationError::StatementBindingMismatch);
        }

        let key_bytes = hex::decode(&self.verifying_key)
            .map_err(|_| FreshnessAnchorAttestationVerificationError::InvalidVerificationKey)?;
        let key_bytes: [u8; 32] = key_bytes
            .try_into()
            .map_err(|_| FreshnessAnchorAttestationVerificationError::InvalidVerificationKey)?;
        let verifying_key = VerifyingKey::from_bytes(&key_bytes)
            .map_err(|_| FreshnessAnchorAttestationVerificationError::InvalidVerificationKey)?;

        let signature_bytes: [u8; 64] = self.signature.as_slice().try_into().map_err(|_| {
            FreshnessAnchorAttestationVerificationError::InvalidSignature
        })?;
        let signature = Signature::from_bytes(&signature_bytes);

        verifying_key
            .verify(&signed_message(&self.statement_digest), &signature)
            .map_err(|_| {
                FreshnessAnchorAttestationVerificationError::SignatureVerificationFailed
            })
    }

    pub fn verifying_key_fingerprint(
        &self,
    ) -> Result<String, FreshnessAnchorAttestationVerificationError> {
        let key_bytes = hex::decode(&self.verifying_key)
            .map_err(|_| FreshnessAnchorAttestationVerificationError::InvalidVerificationKey)?;
        if key_bytes.len() != 32 {
            return Err(FreshnessAnchorAttestationVerificationError::InvalidVerificationKey);
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea:freshness-anchor-verifier-key:v1\\0");
        hasher.update(&key_bytes);
        Ok(hasher.finalize().to_hex().to_string())
    }
}

struct AttestationVerifier<'a> {
    attestation: &'a FreshnessAnchorVerifierAttestation,
}

impl FreshnessAnchorEvidenceVerifier for AttestationVerifier<'_> {
    fn verify(
        &self,
        _profile: &FreshnessAnchorProfile,
        receipt: &FreshnessAnchorVerificationReceipt,
    ) -> bool {
        self.attestation.verify(receipt).is_ok()
    }
}

pub fn verify_attested_anchor(
    profile: FreshnessAnchorProfile,
    receipt: FreshnessAnchorVerificationReceipt,
    attestation: &FreshnessAnchorVerifierAttestation,
) -> Result<VerifiedFreshnessAnchor, FreshnessAnchorAttestationVerificationError> {
    attestation.verify(&receipt)?;
    VerifiedFreshnessAnchor::verify(profile, receipt, &AttestationVerifier { attestation })
        .map_err(FreshnessAnchorAttestationVerificationError::AnchorVerificationFailed)
}

pub fn verify_attested_anchor_with_trust_policy(
    profile: FreshnessAnchorProfile,
    receipt: FreshnessAnchorVerificationReceipt,
    attestation: &FreshnessAnchorVerifierAttestation,
    trust_policy: &FreshnessAnchorVerifierTrustPolicy,
) -> Result<VerifiedFreshnessAnchor, FreshnessAnchorAttestationVerificationError> {
    attestation.verify(&receipt)?;
    if !trust_policy.validate(&receipt, attestation)
        || attestation.verifying_key_fingerprint()? != trust_policy.expected_key_fingerprint
    {
        return Err(FreshnessAnchorAttestationVerificationError::VerifierTrustPolicyMismatch);
    }

    VerifiedFreshnessAnchor::verify(profile, receipt, &AttestationVerifier { attestation })
        .map_err(FreshnessAnchorAttestationVerificationError::AnchorVerificationFailed)
}

    #[test]
    fn trust_policy_rejects_unexpected_verifier_key() {
        let signing_key = key(7);
        let r = receipt();
        let attestation =
            FreshnessAnchorVerifierAttestation::sign(r.verifier_reference.clone(), &r, &signing_key);
        let policy = FreshnessAnchorVerifierTrustPolicy::new(
            "verifier-1",
            FreshnessAnchorVerifierAttestation::sign(
                "verifier-1",
                &r,
                &key(8),
            )
            .verifying_key_fingerprint()
            .unwrap(),
            "trust-anchors-1",
        );

        assert_eq!(
            verify_attested_anchor_with_trust_policy(profile(), r, &attestation, &policy)
                .unwrap_err(),
            FreshnessAnchorAttestationVerificationError::VerifierTrustPolicyMismatch
        );
    }

    #[test]
    fn trust_policy_binds_trust_anchor_set() {
        let signing_key = key(7);
        let r = receipt();
        let attestation =
            FreshnessAnchorVerifierAttestation::sign(r.verifier_reference.clone(), &r, &signing_key);
        let policy = FreshnessAnchorVerifierTrustPolicy::new(
            "verifier-1",
            attestation.verifying_key_fingerprint().unwrap(),
            "different-trust-anchor-set",
        );

        assert_eq!(
            verify_attested_anchor_with_trust_policy(profile(), r, &attestation, &policy)
                .unwrap_err(),
            FreshnessAnchorAttestationVerificationError::VerifierTrustPolicyMismatch
        );
    }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::freshness_anchor_assurance::{
        FreshnessAnchorBacking, FreshnessAnchorCapabilities, FreshnessAnchorEvidenceKind,
    };

    fn profile() -> FreshnessAnchorProfile {
        FreshnessAnchorProfile::new(
            FreshnessAnchorBacking::RemoteAuthority,
            FreshnessAnchorCapabilities::authoritative(),
            "remote://authority-a",
        )
        .unwrap()
    }

    fn receipt() -> FreshnessAnchorVerificationReceipt {
        let profile = profile();
        FreshnessAnchorVerificationReceipt {
            schema_version: SCHEMA_VERSION.into(),
            profile_fingerprint: profile.fingerprint(),
            receiver_id: "receiver-1".into(),
            generation: 7,
            state_fingerprint: "state-1".into(),
            recovery_policy_fingerprint: "recovery-policy-1".into(),
            authority_reference: "authority-1".into(),
            authority_statement_digest: "authority-statement-1".into(),
            authentication_binding: "authentication-1".into(),
            verifier_reference: "verifier-1".into(),
            verifier_policy_digest: "verifier-policy-1".into(),
            reference_values_digest: "reference-values-1".into(),
            trust_anchor_set_digest: "trust-anchors-1".into(),
            freshness_handle_digest: "freshness-handle-1".into(),
            evidence_reference: "evidence-1".into(),
            evidence_digest: "digest-1".into(),
            evidence_kind: FreshnessAnchorEvidenceKind::RemoteMonotonicSequence {
                authority_identity_digest: "authority-id-1".into(),
                authority_namespace_digest: "authority-namespace-1".into(),
                observed_sequence: 7,
            },
        }
    }

    fn key(byte: u8) -> SigningKey {
        SigningKey::from_bytes(&[byte; 32])
    }

    #[test]
    fn signed_result_round_trips_into_verified_anchor() {
        let signing_key = key(7);
        let r = receipt();
        let attestation =
            FreshnessAnchorVerifierAttestation::sign(r.verifier_reference.clone(), &r, &signing_key);

        assert!(attestation.verify(&r).is_ok());
        assert!(verify_attested_anchor(profile(), r, &attestation).is_ok());
    }

    #[test]
    fn matching_trust_policy_allows_verified_anchor() {
        let signing_key = key(7);
        let r = receipt();
        let attestation =
            FreshnessAnchorVerifierAttestation::sign(r.verifier_reference.clone(), &r, &signing_key);
        let policy = FreshnessAnchorVerifierTrustPolicy::new(
            "verifier-1",
            attestation.verifying_key_fingerprint().unwrap(),
            "trust-anchors-1",
        );

        assert!(verify_attested_anchor_with_trust_policy(profile(), r, &attestation, &policy).is_ok());
    }

    #[test]
    fn receipt_mutation_invalidates_signed_result() {
        let signing_key = key(7);
        let r = receipt();
        let attestation =
            FreshnessAnchorVerifierAttestation::sign(r.verifier_reference.clone(), &r, &signing_key);
        let mut modified = r;
        modified.generation = 8;

        assert_eq!(
            attestation.verify(&modified).unwrap_err(),
            FreshnessAnchorAttestationVerificationError::StatementBindingMismatch
        );
    }

    #[test]
    fn verifier_reference_cannot_be_spliced() {
        let signing_key = key(7);
        let mut r = receipt();
        let attestation =
            FreshnessAnchorVerifierAttestation::sign(r.verifier_reference.clone(), &r, &signing_key);
        r.verifier_reference = "verifier-2".into();

        assert_eq!(
            attestation.verify(&r).unwrap_err(),
            FreshnessAnchorAttestationVerificationError::VerifierReferenceMismatch
        );
    }

    #[test]
    fn wrong_key_cannot_verify() {
        let signing_key = key(7);
        let wrong_key = key(8);
        let r = receipt();
        let mut attestation =
            FreshnessAnchorVerifierAttestation::sign(r.verifier_reference.clone(), &r, &signing_key);
        attestation.verifying_key = hex::encode(wrong_key.verifying_key().as_bytes());

        assert_eq!(
            attestation.verify(&r).unwrap_err(),
            FreshnessAnchorAttestationVerificationError::SignatureVerificationFailed
        );
    }

    #[test]
    fn signature_length_is_checked() {
        let signing_key = key(7);
        let r = receipt();
        let mut attestation =
            FreshnessAnchorVerifierAttestation::sign(r.verifier_reference.clone(), &r, &signing_key);
        attestation.signature.pop();

        assert_eq!(
            attestation.verify(&r).unwrap_err(),
            FreshnessAnchorAttestationVerificationError::InvalidSignature
        );
    }

    #[test]
    fn public_key_fingerprint_is_stable_and_nonempty() {
        let signing_key = key(7);
        let r = receipt();
        let attestation =
            FreshnessAnchorVerifierAttestation::sign(r.verifier_reference.clone(), &r, &signing_key);

        let fingerprint = attestation.verifying_key_fingerprint().unwrap();
        assert_eq!(fingerprint.len(), 64);
    }
}
