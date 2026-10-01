// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! External cryptographic attestation adapter for observation-fabric receipts.
//!
//! This crate deliberately lives outside symthaea-core. The core defines the
//! receipt/envelope semantics and canonical payload; this adapter adds one concrete
//! cryptographic proof suite (detached Ed25519) without making cryptographic key
//! resolution, authorization, revocation, or substantive truth part of the core.
//!
//! The adapter is Symthaea-native. It is not a claim of W3C Data Integrity
//! conformance: W3C Data Integrity 1.1 is currently a Working Draft, and its
//! document-processing and cryptosuite rules are distinct from this envelope's
//! explicit canonical byte contract.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use symthaea_core::observation_fabric::{
    IndependenceVerificationReceipt, ReceiptAttestationEnvelope,
    ReceiptAttestationTemporalStatus,
};

pub const CRYPTOSUITE: &str = "symthaea-ed25519-detached-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReceiptAttestationVerificationOutcome {
    Verified,
    InvalidEnvelope,
    ReceiptCommitmentMismatch,
    NotYetValid,
    Expired,
    MissingProof,
    InvalidProofEncoding,
    CryptosuiteMismatch,
    VerificationMethodMismatch,
    ProofPurposeMismatch,
    DomainMismatch,
    ChallengeMismatch,
    InvalidSignature,
}

#[derive(Debug, Clone)]
pub struct Ed25519ReceiptVerifier {
    verification_method: String,
    verifying_key: VerifyingKey,
    expected_proof_purpose: Option<String>,
    expected_domain: Option<String>,
    expected_challenge: Option<String>,
    now_unix_ns: i128,
}

impl Ed25519ReceiptVerifier {
    pub fn new(
        verification_method: impl Into<String>,
        verifying_key: VerifyingKey,
        now_unix_ns: i128,
    ) -> Self {
        Self {
            verification_method: verification_method.into(),
            verifying_key,
            expected_proof_purpose: None,
            expected_domain: None,
            expected_challenge: None,
            now_unix_ns,
        }
    }

    pub fn with_expected_proof_purpose(mut self, purpose: impl Into<String>) -> Self {
        self.expected_proof_purpose = Some(purpose.into());
        self
    }

    pub fn with_expected_domain(mut self, domain: impl Into<String>) -> Self {
        self.expected_domain = Some(domain.into());
        self
    }

    pub fn with_expected_challenge(mut self, challenge: impl Into<String>) -> Self {
        self.expected_challenge = Some(challenge.into());
        self
    }

    /// Verify structure, receipt commitment, temporal status, policy bindings,
    /// and the detached Ed25519 proof. This does not perform issuer
    /// authorization, credential validation, revocation/status resolution,
    /// or substantive observation validation.
    pub fn verify(
        &self,
        envelope: &ReceiptAttestationEnvelope,
        receipt: &IndependenceVerificationReceipt,
    ) -> ReceiptAttestationVerificationOutcome {
        if envelope.validate().is_err() {
            return ReceiptAttestationVerificationOutcome::InvalidEnvelope;
        }
        if !envelope.verify_against_receipt(receipt) {
            return ReceiptAttestationVerificationOutcome::ReceiptCommitmentMismatch;
        }

        match envelope.temporal_status_at(self.now_unix_ns) {
            ReceiptAttestationTemporalStatus::NotYetValid =>
                return ReceiptAttestationVerificationOutcome::NotYetValid,
            ReceiptAttestationTemporalStatus::Expired =>
                return ReceiptAttestationVerificationOutcome::Expired,
            ReceiptAttestationTemporalStatus::Valid => {}
        }

        if envelope.cryptosuite.as_deref() != Some(CRYPTOSUITE) {
            return ReceiptAttestationVerificationOutcome::CryptosuiteMismatch;
        }
        if envelope.verification_method.as_deref() != Some(self.verification_method.as_str()) {
            return ReceiptAttestationVerificationOutcome::VerificationMethodMismatch;
        }
        if let Some(expected) = &self.expected_proof_purpose {
            if envelope.proof_purpose != *expected {
                return ReceiptAttestationVerificationOutcome::ProofPurposeMismatch;
            }
        }
        if let Some(expected) = &self.expected_domain {
            if envelope.domain.as_deref() != Some(expected.as_str()) {
                return ReceiptAttestationVerificationOutcome::DomainMismatch;
            }
        }
        if let Some(expected) = &self.expected_challenge {
            if envelope.challenge.as_deref() != Some(expected.as_str()) {
                return ReceiptAttestationVerificationOutcome::ChallengeMismatch;
            }
        }

        let Some(proof) = envelope.proof.as_deref() else {
            return ReceiptAttestationVerificationOutcome::MissingProof;
        };
        let Ok(proof_bytes) = <[u8; 64]>::try_from(proof) else {
            return ReceiptAttestationVerificationOutcome::InvalidProofEncoding;
        };
        let signature = Signature::from_bytes(&proof_bytes);

        match self.verifying_key.verify(&envelope.canonical_payload_bytes(), &signature) {
            Ok(()) => ReceiptAttestationVerificationOutcome::Verified,
            Err(_) => ReceiptAttestationVerificationOutcome::InvalidSignature,
        }
    }
}

/// Attach a detached Ed25519 proof. Key publication, authorization, rotation,
/// and revocation remain outside this adapter.
pub fn sign_envelope(
    envelope: &mut ReceiptAttestationEnvelope,
    signing_key: &SigningKey,
    verification_method: impl Into<String>,
) -> Result<(), SignEnvelopeError> {
    envelope.cryptosuite = Some(CRYPTOSUITE.to_string());
    envelope.verification_method = Some(verification_method.into());
    envelope.validate().map_err(|_| SignEnvelopeError::InvalidEnvelope)?;
    let signature = signing_key.sign(&envelope.canonical_payload_bytes());
    envelope.proof = Some(signature.to_bytes().to_vec());
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SignEnvelopeError {
    #[error("receipt attestation envelope is structurally invalid")]
    InvalidEnvelope,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::observation_fabric::{
        AssetRef, DisclosurePolicy, Observation, ObservationGraph, ObservationModality,
        ObservationProvenance, ObservationQuality, ObservationTime, ProvenanceCoverage,
        ProvenanceVerification, SensorIdentity,
    };

    fn receipt() -> IndependenceVerificationReceipt {
        let base = Observation {
            id: "obs-a".into(),
            modality: ObservationModality::Optical,
            time: ObservationTime { observed_at_unix_ns: 100, time_uncertainty_ns: 1 },
            location: None,
            feature_of_interest_id: None,
            quality: ObservationQuality {
                confidence: 0.9,
                measurement_uncertainty: None,
                calibrated: true,
            },
            provenance: ObservationProvenance {
                source: SensorIdentity::new("sensor-a"),
                verification: ProvenanceVerification::Unverified,
                coverage: ProvenanceCoverage::Complete,
                attestation_id: None,
                acquired_by: None,
                parent_observation_ids: Vec::new(),
                processing_fingerprint: None,
                processing_activity: None,
            },
            asset: Some(AssetRef::blake3(b"a")),
            disclosure: DisclosurePolicy::restricted(),
        };
        let mut other = base.clone();
        other.id = "obs-b".into();
        other.provenance.source = SensorIdentity::new("sensor-b");
        let graph = ObservationGraph { observations: vec![base, other], relations: vec![] };
        let assessment = graph.assess_independence_detailed("obs-a", "obs-b").expect("assessment");
        IndependenceVerificationReceipt::from_assessment(&assessment)
    }

    fn envelope_and_key() -> (
        ReceiptAttestationEnvelope,
        SigningKey,
        IndependenceVerificationReceipt,
    ) {
        let receipt = receipt();
        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let mut envelope = ReceiptAttestationEnvelope::from_receipt(
            &receipt, "attester-a", "observation-independence", 100,
        );
        envelope.expires_at_unix_ns = Some(200);
        sign_envelope(&mut envelope, &signing_key, "did:example:attester-a#key-1")
            .expect("sign");
        (envelope, signing_key, receipt)
    }

    #[test]
    fn signs_and_verifies_detached_receipt() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        assert_eq!(
            verifier.verify(&envelope, &receipt),
            ReceiptAttestationVerificationOutcome::Verified
        );
    }

    #[test]
    fn payload_mutation_invalidates_signature() {
        let (mut envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        assert_eq!(
            verifier.verify(&envelope, &receipt),
            ReceiptAttestationVerificationOutcome::Verified
        );
        envelope.challenge = Some("challenge".into());
        assert_eq!(
            verifier.verify(&envelope, &receipt),
            ReceiptAttestationVerificationOutcome::InvalidSignature
        );
    }

    #[test]
    fn temporal_and_key_policy_failures_are_distinct() {
        let (mut envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            200,
        );
        assert_eq!(
            verifier.verify(&envelope, &receipt),
            ReceiptAttestationVerificationOutcome::Expired
        );

        envelope.expires_at_unix_ns = Some(300);
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:wrong#key-1",
            signing_key.verifying_key(),
            150,
        );
        assert_eq!(
            verifier.verify(&envelope, &receipt),
            ReceiptAttestationVerificationOutcome::VerificationMethodMismatch
        );
    }

    #[test]
    fn proof_purpose_domain_and_challenge_are_policy_checks() {
        let (mut envelope, signing_key, receipt) = envelope_and_key();
        envelope.domain = Some("example.org".into());
        envelope.challenge = Some("c1".into());
        sign_envelope(&mut envelope, &signing_key, "did:example:attester-a#key-1")
            .expect("resign");
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .with_expected_proof_purpose("observation-independence")
        .with_expected_domain("example.org")
        .with_expected_challenge("c1");
        assert_eq!(
            verifier.verify(&envelope, &receipt),
            ReceiptAttestationVerificationOutcome::Verified
        );
    }
}
