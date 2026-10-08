//! RFC 9942 post-quantum hybrid assurance seam.
//!
//! This module deliberately does not implement ML-DSA itself. It freezes the
//! semantic boundary between an already-verified classical RFC 9942 capability
//! and a separately authenticated ML-DSA-65 attestation.
//!
//! The attestation is an application-level profile. It is not a provisional
//! COSE composite algorithm.

use sha2::{Digest, Sha256};

use crate::semantic_evidence_vds::{
    Rfc9942VerifiedReceipt, MAX_RFC9942_RECEIPT_ENCODED_BYTES,
};

pub const HYBRID_POLICY_ID: &str =
    "symthaea-swarm/rfc9942-pq-bound-mldsa65-v1";
pub const HYBRID_POLICY_VERSION: u16 = 1;
pub const ML_DSA_65_COSE_ALGORITHM_ID: i64 = -49;
pub const ML_DSA_65_PUBLIC_KEY_BYTES: usize = 1952;
pub const ML_DSA_65_SIGNATURE_BYTES: usize = 3309;
pub const MAX_HYBRID_RECEIPT_WIRE_BYTES: usize =
    MAX_RFC9942_RECEIPT_ENCODED_BYTES;

pub const HYBRID_TRANSCRIPT_DOMAIN: &[u8] =
    b"symthaea-swarm/rfc9942-pq-bound-mldsa65-transcript-v1";

/// Provider boundary for the actual ML-DSA-65 implementation.
///
/// A provider may be a software implementation, HSM, KMS, remote signer, or
/// independent verification oracle. The semantic module never assumes which.
pub trait MlDsa65Verifier {
    fn verify(
        &self,
        verifying_key: &[u8],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), MlDsa65VerifyError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlDsa65VerifyError {
    InvalidPublicKey,
    InvalidSignature,
    VerificationFailed,
    ResourceLimitExceeded,
    ProviderFailure,
}

/// Canonical fixed-width transcript metadata for the application-level
/// post-quantum overlay.
///
/// The transcript binds:
/// - the exact SHA-256 identity of the serialized RFC 9942 Receipt; and
/// - the exact SHA-256 identity of the already verified classical capability.
///
/// Parsed Rust structures are not serialized into this transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rfc9942HybridTranscript {
    receipt_sha256: [u8; 32],
    classical_capability_sha256: [u8; 32],
    transcript_sha256: [u8; 32],
}

impl Rfc9942HybridTranscript {
    pub fn new(
        verified_classical: &Rfc9942VerifiedReceipt,
        exact_receipt_wire: &[u8],
    ) -> Result<Self, Rfc9942HybridError> {
        if exact_receipt_wire.is_empty()
            || exact_receipt_wire.len() > MAX_HYBRID_RECEIPT_WIRE_BYTES
        {
            return Err(Rfc9942HybridError::ReceiptWireTooLarge);
        }

        let receipt_sha256 = sha256(exact_receipt_wire);
        if receipt_sha256 != verified_classical.receipt_sha256() {
            return Err(Rfc9942HybridError::ReceiptWireIdentityMismatch);
        }

        let classical_capability_sha256 = verified_classical.capability_sha256();
        let transcript_sha256 =
            digest_transcript(receipt_sha256, classical_capability_sha256);

        Ok(Self {
            receipt_sha256,
            classical_capability_sha256,
            transcript_sha256,
        })
    }

    pub const fn receipt_sha256(&self) -> [u8; 32] {
        self.receipt_sha256
    }

    pub const fn classical_capability_sha256(&self) -> [u8; 32] {
        self.classical_capability_sha256
    }

    pub const fn transcript_sha256(&self) -> [u8; 32] {
        self.transcript_sha256
    }

    /// Exact fixed-width bytes authenticated by the ML-DSA-65 attestation.
    ///
    /// Domain separation is included in transcript_sha256. These bytes carry
    /// the explicit policy/version and the three fixed identities needed by
    /// the verifier.
    pub fn signing_bytes(&self) -> [u8; 106] {
        let mut out = [0u8; 106];
        out[0..2].copy_from_slice(&HYBRID_POLICY_VERSION.to_be_bytes());
        out[2..10].copy_from_slice(&ML_DSA_65_COSE_ALGORITHM_ID.to_be_bytes());
        out[10..42].copy_from_slice(&self.receipt_sha256);
        out[42..74].copy_from_slice(&self.classical_capability_sha256);
        out[74..106].copy_from_slice(&self.transcript_sha256);
        out
    }
}

/// PQ attestation material bound to one exact, already-verified RFC 9942
/// Receipt. Its construction is private to the hybrid verification path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rfc9942PqAttestation {
    verifying_key_sha256: [u8; 32],
    signature_sha256: [u8; 32],
    transcript: Rfc9942HybridTranscript,
}

impl Rfc9942PqAttestation {
    pub const fn verifying_key_sha256(&self) -> [u8; 32] {
        self.verifying_key_sha256
    }

    pub const fn signature_sha256(&self) -> [u8; 32] {
        self.signature_sha256
    }

    pub const fn transcript(&self) -> Rfc9942HybridTranscript {
        self.transcript
    }
}

/// Stronger application-level assurance:
///
/// verified classical RFC 9942 capability
/// + verified ML-DSA-65 attestation over the exact receipt identity.
///
/// This does not assert truth or authorization.
#[must_use = "retain the PQ-bound capability when crossing a trust boundary"]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rfc9942HybridVerifiedReceipt {
    classical_capability_sha256: [u8; 32],
    pq_attestation: Rfc9942PqAttestation,
    hybrid_capability_sha256: [u8; 32],
}

impl Rfc9942HybridVerifiedReceipt {
    pub fn verify(
        verified_classical: &Rfc9942VerifiedReceipt,
        exact_receipt_wire: &[u8],
        verifying_key: &[u8],
        signature: &[u8],
        verifier: &impl MlDsa65Verifier,
    ) -> Result<Self, Rfc9942HybridError> {
        if verifying_key.len() != ML_DSA_65_PUBLIC_KEY_BYTES {
            return Err(Rfc9942HybridError::PqPublicKeyWrongLength);
        }
        if signature.len() != ML_DSA_65_SIGNATURE_BYTES {
            return Err(Rfc9942HybridError::PqSignatureWrongLength);
        }
        if verifying_key.iter().all(|byte| *byte == 0) {
            return Err(Rfc9942HybridError::PqPublicKeyAllZero);
        }
        if signature.iter().all(|byte| *byte == 0) {
            return Err(Rfc9942HybridError::PqSignatureAllZero);
        }

        let transcript =
            Rfc9942HybridTranscript::new(verified_classical, exact_receipt_wire)?;

        verifier
            .verify(verifying_key, &transcript.signing_bytes(), signature)
            .map_err(Rfc9942HybridError::from_provider)?;

        let pq_attestation = Rfc9942PqAttestation {
            verifying_key_sha256: sha256(verifying_key),
            signature_sha256: sha256(signature),
            transcript,
        };

        let classical_capability_sha256 =
            verified_classical.capability_sha256();
        let hybrid_capability_sha256 = digest_hybrid_capability(
            classical_capability_sha256,
            &pq_attestation,
        );

        Ok(Self {
            classical_capability_sha256,
            pq_attestation,
            hybrid_capability_sha256,
        })
    }

    pub const fn classical_capability_sha256(&self) -> [u8; 32] {
        self.classical_capability_sha256
    }

    pub const fn pq_attestation(&self) -> Rfc9942PqAttestation {
        self.pq_attestation
    }

    /// Identity of this application-level hybrid assurance result.
    pub const fn hybrid_capability_sha256(&self) -> [u8; 32] {
        self.hybrid_capability_sha256
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rfc9942HybridError {
    ReceiptWireTooLarge,
    ReceiptWireIdentityMismatch,
    PqPublicKeyWrongLength,
    PqPublicKeyAllZero,
    PqSignatureWrongLength,
    PqSignatureAllZero,
    PqPublicKeyInvalid,
    PqSignatureInvalid,
    PqSignatureVerificationFailed,
    PqResourceLimitExceeded,
    PqProviderFailure,
}

impl Rfc9942HybridError {
    const fn from_provider(error: MlDsa65VerifyError) -> Self {
        match error {
            MlDsa65VerifyError::InvalidPublicKey => Self::PqPublicKeyInvalid,
            MlDsa65VerifyError::InvalidSignature => Self::PqSignatureInvalid,
            MlDsa65VerifyError::VerificationFailed => {
                Self::PqSignatureVerificationFailed
            }
            MlDsa65VerifyError::ResourceLimitExceeded => {
                Self::PqResourceLimitExceeded
            }
            MlDsa65VerifyError::ProviderFailure => Self::PqProviderFailure,
        }
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn digest_transcript(
    receipt_sha256: [u8; 32],
    classical_capability_sha256: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(HYBRID_TRANSCRIPT_DOMAIN);
    hasher.update(HYBRID_POLICY_VERSION.to_be_bytes());
    hasher.update(ML_DSA_65_COSE_ALGORITHM_ID.to_be_bytes());
    hasher.update((receipt_sha256.len() as u32).to_be_bytes());
    hasher.update(receipt_sha256);
    hasher.update((classical_capability_sha256.len() as u32).to_be_bytes());
    hasher.update(classical_capability_sha256);
    hasher.finalize().into()
}

fn digest_hybrid_capability(
    classical_capability_sha256: [u8; 32],
    pq_attestation: &Rfc9942PqAttestation,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(
        b"symthaea-swarm/rfc9942-hybrid-verified-receipt-capability-v1",
    );
    hasher.update(HYBRID_POLICY_VERSION.to_be_bytes());
    hasher.update(classical_capability_sha256);
    hasher.update(pq_attestation.verifying_key_sha256);
    hasher.update(pq_attestation.signature_sha256);
    hasher.update(pq_attestation.transcript.receipt_sha256);
    hasher.update(
        pq_attestation
            .transcript
            .classical_capability_sha256,
    );
    hasher.update(pq_attestation.transcript.transcript_sha256);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcript(receipt: u8, classical: u8) -> Rfc9942HybridTranscript {
        let receipt_sha256 = [receipt; 32];
        let classical_capability_sha256 = [classical; 32];
        Rfc9942HybridTranscript {
            receipt_sha256,
            classical_capability_sha256,
            transcript_sha256:
                digest_transcript(receipt_sha256, classical_capability_sha256),
        }
    }

    #[test]
    fn transcript_is_fixed_width_and_domain_separated() {
        let t = transcript(1, 2);
        assert_eq!(t.signing_bytes().len(), 106);

        let mut second_domain = HYBRID_TRANSCRIPT_DOMAIN.to_vec();
        second_domain.push(0);
        let mut hasher = Sha256::new();
        hasher.update(&second_domain);
        hasher.update(HYBRID_POLICY_VERSION.to_be_bytes());
        hasher.update(ML_DSA_65_COSE_ALGORITHM_ID.to_be_bytes());
        hasher.update((32u32).to_be_bytes());
        hasher.update([1u8; 32]);
        hasher.update((32u32).to_be_bytes());
        hasher.update([2u8; 32]);

        assert_ne!(t.transcript_sha256(), hasher.finalize().into());
    }

    #[test]
    fn transcript_changes_when_receipt_identity_changes() {
        assert_ne!(
            transcript(1, 2).signing_bytes(),
            transcript(3, 2).signing_bytes()
        );
    }

    #[test]
    fn transcript_changes_when_classical_capability_changes() {
        assert_ne!(
            transcript(1, 2).signing_bytes(),
            transcript(1, 3).signing_bytes()
        );
    }

    #[test]
    fn provider_resource_failure_has_distinct_semantic_class() {
        assert_eq!(
            Rfc9942HybridError::from_provider(
                MlDsa65VerifyError::ResourceLimitExceeded,
            ),
            Rfc9942HybridError::PqResourceLimitExceeded
        );
    }

    #[test]
    fn exact_sizes_match_fips_204_ml_dsa_65_profile() {
        assert_eq!(ML_DSA_65_PUBLIC_KEY_BYTES, 1952);
        assert_eq!(ML_DSA_65_SIGNATURE_BYTES, 3309);
    }
}
