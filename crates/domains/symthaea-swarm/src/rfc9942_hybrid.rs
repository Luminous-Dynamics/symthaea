//! RFC 9942 post-quantum hybrid assurance seam.
//!
//! This module deliberately does not implement ML-DSA itself. It freezes the
//! semantic boundary between an already-verified classical RFC 9942 capability
//! and a separately authenticated ML-DSA-65 attestation.
//!
//! The attestation is an application-level profile. It is not a provisional
//! COSE composite algorithm.

use sha2::{Digest, Sha256};

use crate::rfc9942_selection::Rfc9942VerifiedReceiptSelection;
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

/// Trust-policy boundary for the ML-DSA verification key.
///
/// Cryptographic validity and authorization are deliberately separate. The
/// policy must validate the exact key bytes, key identifier, and evaluation
/// time against an independently trusted registry/snapshot.
pub trait MlDsa65KeyPolicy {
    /// Authorize a key and return the exact snapshot-bound authorization
    /// receipt used for this decision. Returning the digest and authorization
    /// as separate calls would permit a mutable registry to change between
    /// them, creating a policy time-of-check/time-of-use gap.
    fn authorize(
        &self,
        key_id: &Rfc9942PqKeyId,
        verifying_key_sha256: [u8; 32],
        evaluation_time_unix_seconds: u64,
    ) -> Result<MlDsa65KeyAuthorization, MlDsa65KeyAuthorizationError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rfc9942PqKeyId([u8; 16]);

impl Rfc9942PqKeyId {
    pub fn new(bytes: [u8; 16]) -> Result<Self, Rfc9942HybridError> {
        if bytes == [0u8; 16] {
            return Err(Rfc9942HybridError::PqKeyIdInvalid);
        }
        Ok(Self(bytes))
    }

    pub const fn bytes(&self) -> [u8; 16] {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlDsa65KeyAuthorizationError {
    UnknownKey,
    NotYetValid,
    Expired,
    Revoked,
    UsageNotPermitted,
    InvalidPolicyDigest,
    PolicyUnavailable,
}

/// Snapshot-bound result of key authorization.
///
/// The digest and the key/time tuple travel together so downstream verification
/// cannot accidentally combine an authorization result with a different trust
/// snapshot, key, or evaluation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlDsa65KeyAuthorization {
    policy_digest_sha256: [u8; 32],
    key_id: Rfc9942PqKeyId,
    verifying_key_sha256: [u8; 32],
    evaluation_time_unix_seconds: u64,
}

impl MlDsa65KeyAuthorization {
    pub fn new(
        policy_digest_sha256: [u8; 32],
        key_id: Rfc9942PqKeyId,
        verifying_key_sha256: [u8; 32],
        evaluation_time_unix_seconds: u64,
    ) -> Result<Self, MlDsa65KeyAuthorizationError> {
        if policy_digest_sha256 == [0; 32] {
            return Err(MlDsa65KeyAuthorizationError::InvalidPolicyDigest);
        }
        Ok(Self {
            policy_digest_sha256,
            key_id,
            verifying_key_sha256,
            evaluation_time_unix_seconds,
        })
    }

    pub const fn policy_digest_sha256(&self) -> [u8; 32] {
        self.policy_digest_sha256
    }

    pub const fn key_id(&self) -> Rfc9942PqKeyId {
        self.key_id
    }

    pub const fn verifying_key_sha256(&self) -> [u8; 32] {
        self.verifying_key_sha256
    }

    pub const fn evaluation_time_unix_seconds(&self) -> u64 {
        self.evaluation_time_unix_seconds
    }
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
    policy_digest_sha256: [u8; 32],
    key_id: Rfc9942PqKeyId,
    verifying_key_sha256: [u8; 32],
    receipt_sha256: [u8; 32],
    classical_capability_sha256: [u8; 32],
    transcript_sha256: [u8; 32],
}

impl Rfc9942HybridTranscript {
    pub fn new(
        verified_classical: &Rfc9942VerifiedReceipt,
        exact_receipt_wire: &[u8],
        key_id: Rfc9942PqKeyId,
        verifying_key_sha256: [u8; 32],
        policy_digest_sha256: [u8; 32],
    ) -> Result<Self, Rfc9942HybridError> {
        if exact_receipt_wire.is_empty()
            || exact_receipt_wire.len() > MAX_HYBRID_RECEIPT_WIRE_BYTES
        {
            return Err(Rfc9942HybridError::ReceiptWireTooLarge);
        }
        if policy_digest_sha256 == [0; 32] {
            return Err(Rfc9942HybridError::PqKeyPolicyInvalid);
        }

        let receipt_sha256 = sha256(exact_receipt_wire);
        if receipt_sha256 != verified_classical.receipt_sha256() {
            return Err(Rfc9942HybridError::ReceiptWireIdentityMismatch);
        }

        let classical_capability_sha256 = verified_classical.capability_sha256();
        let transcript_sha256 = digest_transcript(
            policy_digest_sha256,
            key_id,
            verifying_key_sha256,
            receipt_sha256,
            classical_capability_sha256,
        );

        Ok(Self {
            policy_digest_sha256,
            key_id,
            verifying_key_sha256,
            receipt_sha256,
            classical_capability_sha256,
            transcript_sha256,
        })
    }

    pub const fn policy_digest_sha256(&self) -> [u8; 32] {
        self.policy_digest_sha256
    }

    pub const fn key_id(&self) -> Rfc9942PqKeyId {
        self.key_id
    }

    pub const fn verifying_key_sha256(&self) -> [u8; 32] {
        self.verifying_key_sha256
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
    pub fn signing_bytes(&self) -> [u8; 186] {
        let mut out = [0u8; 186];
        out[0..2].copy_from_slice(&HYBRID_POLICY_VERSION.to_be_bytes());
        out[2..10].copy_from_slice(&ML_DSA_65_COSE_ALGORITHM_ID.to_be_bytes());
        out[10..42].copy_from_slice(&self.policy_digest_sha256);
        out[42..58].copy_from_slice(&self.key_id.0);
        out[58..90].copy_from_slice(&self.verifying_key_sha256);
        out[90..122].copy_from_slice(&self.receipt_sha256);
        out[122..154].copy_from_slice(&self.classical_capability_sha256);
        out[154..186].copy_from_slice(&self.transcript_sha256);
        out
    }
}

/// PQ attestation material bound to one exact, already-verified RFC 9942
/// Receipt. Its construction is private to the hybrid verification path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rfc9942PqAttestation {
    key_id: Rfc9942PqKeyId,
    verifying_key_sha256: [u8; 32],
    signature_sha256: [u8; 32],
    transcript: Rfc9942HybridTranscript,
}

impl Rfc9942PqAttestation {
    pub const fn key_id(&self) -> Rfc9942PqKeyId {
        self.key_id
    }

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
    key_policy_digest_sha256: [u8; 32],
    key_authorization_evaluation_time_unix_seconds: u64,
    pq_attestation: Rfc9942PqAttestation,
    hybrid_capability_sha256: [u8; 32],
}

impl Rfc9942HybridVerifiedReceipt {
    pub fn verify_with_policy(
        verified_classical: &Rfc9942VerifiedReceipt,
        exact_receipt_wire: &[u8],
        key_id: Rfc9942PqKeyId,
        verifying_key: &[u8],
        signature: &[u8],
        evaluation_time_unix_seconds: u64,
        key_policy: &impl MlDsa65KeyPolicy,
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

        let verifying_key_sha256 = sha256(verifying_key);
        let authorization = key_policy
            .authorize(
                &key_id,
                verifying_key_sha256,
                evaluation_time_unix_seconds,
            )
            .map_err(Rfc9942HybridError::from_key_policy)?;

        if authorization.key_id() != key_id
            || authorization.verifying_key_sha256() != verifying_key_sha256
            || authorization.evaluation_time_unix_seconds()
                != evaluation_time_unix_seconds
        {
            return Err(Rfc9942HybridError::PqKeyAuthorizationBindingMismatch);
        }
        let policy_digest_sha256 = authorization.policy_digest_sha256();

        let transcript = Rfc9942HybridTranscript::new(
            verified_classical,
            exact_receipt_wire,
            key_id,
            verifying_key_sha256,
            policy_digest_sha256,
        )?;

        verifier
            .verify(verifying_key, &transcript.signing_bytes(), signature)
            .map_err(Rfc9942HybridError::from_provider)?;

        let pq_attestation = Rfc9942PqAttestation {
            key_id,
            verifying_key_sha256,
            signature_sha256: sha256(signature),
            transcript,
        };

        let classical_capability_sha256 =
            verified_classical.capability_sha256();
        let hybrid_capability_sha256 = digest_hybrid_capability(
            classical_capability_sha256,
            policy_digest_sha256,
            evaluation_time_unix_seconds,
            &pq_attestation,
        );

        Ok(Self {
            classical_capability_sha256,
            key_policy_digest_sha256: policy_digest_sha256,
            key_authorization_evaluation_time_unix_seconds:
                evaluation_time_unix_seconds,
            pq_attestation,
            hybrid_capability_sha256,
        })
    }

    pub const fn classical_capability_sha256(&self) -> [u8; 32] {
        self.classical_capability_sha256
    }

    pub const fn key_policy_digest_sha256(&self) -> [u8; 32] {
        self.key_policy_digest_sha256
    }

    pub const fn key_authorization_evaluation_time_unix_seconds(&self) -> u64 {
        self.key_authorization_evaluation_time_unix_seconds
    }

    pub const fn pq_attestation(&self) -> Rfc9942PqAttestation {
        self.pq_attestation
    }

    /// Identity of this application-level hybrid assurance result.
    pub const fn hybrid_capability_sha256(&self) -> [u8; 32] {
        self.hybrid_capability_sha256
    }
}

/// Requirement applied after receipt verification but before durable
/// publication. Classical validity is preserved as a distinct assurance level;
/// it never silently satisfies a hybrid-required request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rfc9942HybridRequirement {
    ClassicalAllowed,
    HybridRequired,
}

/// Immutable, fail-closed admission of a verified selection under an explicit
/// assurance requirement. The constructor binds the selected receipt digest,
/// classical capability, and PQ capability together. Durable projection should
/// accept this wrapper when the caller's policy requires hybrid assurance.
#[must_use = "retain the admitted assurance when crossing a durable evidence boundary"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rfc9942SelectionAssuranceAdmission {
    requirement: Rfc9942HybridRequirement,
    selection: Rfc9942VerifiedReceiptSelection,
    hybrid: Option<Rfc9942HybridVerifiedReceipt>,
}

impl Rfc9942SelectionAssuranceAdmission {
    pub fn admit(
        requirement: Rfc9942HybridRequirement,
        selection: Rfc9942VerifiedReceiptSelection,
        hybrid: Option<Rfc9942HybridVerifiedReceipt>,
    ) -> Result<Self, Rfc9942HybridError> {
        if requirement == Rfc9942HybridRequirement::HybridRequired && hybrid.is_none() {
            return Err(Rfc9942HybridError::HybridRequiredButMissing);
        }

        let decision = selection.decision();
        decision
            .validated_digest()
            .map_err(|_| Rfc9942HybridError::InvalidSelectionWitness)?;
        let selected_receipt_sha256 = decision
            .selected_receipt_sha256
            .ok_or(Rfc9942HybridError::InvalidSelectionWitness)?;
        let classical_capability_sha256 = selection.verified_capability_sha256();
        if selected_receipt_sha256 == [0; 32] || classical_capability_sha256 == [0; 32] {
            return Err(Rfc9942HybridError::InvalidSelectionWitness);
        }

        if let Some(hybrid) = hybrid {
            let attestation = hybrid.pq_attestation();
            let transcript = attestation.transcript();
            if hybrid.classical_capability_sha256() != classical_capability_sha256
                || transcript.classical_capability_sha256()
                    != classical_capability_sha256
                || transcript.receipt_sha256() != selected_receipt_sha256
                || transcript.policy_digest_sha256()
                    != hybrid.key_policy_digest_sha256()
                || transcript.key_id() != attestation.key_id()
                || transcript.verifying_key_sha256()
                    != attestation.verifying_key_sha256()
                || hybrid.hybrid_capability_sha256() == [0; 32]
            {
                return Err(Rfc9942HybridError::HybridSelectionBindingMismatch);
            }
        }

        Ok(Self {
            requirement,
            selection,
            hybrid,
        })
    }

    pub const fn requirement(&self) -> Rfc9942HybridRequirement {
        self.requirement
    }

    pub const fn selection(&self) -> &Rfc9942VerifiedReceiptSelection {
        &self.selection
    }

    pub const fn hybrid(&self) -> Option<&Rfc9942HybridVerifiedReceipt> {
        self.hybrid.as_ref()
    }

    pub const fn is_hybrid_verified(&self) -> bool {
        self.hybrid.is_some()
    }

    pub const fn hybrid_capability_sha256(&self) -> Option<[u8; 32]> {
        match self.hybrid {
            Some(hybrid) => Some(hybrid.hybrid_capability_sha256()),
            None => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rfc9942HybridError {
    ReceiptWireTooLarge,
    ReceiptWireIdentityMismatch,
    PqKeyIdInvalid,
    PqKeyPolicyInvalid,
    PqKeyAuthorizationBindingMismatch,
    PqKeyUnknown,
    PqKeyNotYetValid,
    PqKeyExpired,
    PqKeyRevoked,
    PqKeyUsageNotPermitted,
    PqKeyPolicyUnavailable,
    PqPublicKeyWrongLength,
    PqPublicKeyAllZero,
    PqSignatureWrongLength,
    PqSignatureAllZero,
    PqPublicKeyInvalid,
    PqSignatureInvalid,
    PqSignatureVerificationFailed,
    PqResourceLimitExceeded,
    PqProviderFailure,
    HybridRequiredButMissing,
    InvalidSelectionWitness,
    HybridSelectionBindingMismatch,
}

impl Rfc9942HybridError {
    const fn from_key_policy(error: MlDsa65KeyAuthorizationError) -> Self {
        match error {
            MlDsa65KeyAuthorizationError::UnknownKey => Self::PqKeyUnknown,
            MlDsa65KeyAuthorizationError::NotYetValid => Self::PqKeyNotYetValid,
            MlDsa65KeyAuthorizationError::Expired => Self::PqKeyExpired,
            MlDsa65KeyAuthorizationError::Revoked => Self::PqKeyRevoked,
            MlDsa65KeyAuthorizationError::UsageNotPermitted => {
                Self::PqKeyUsageNotPermitted
            }
            MlDsa65KeyAuthorizationError::InvalidPolicyDigest => {
                Self::PqKeyPolicyInvalid
            }
            MlDsa65KeyAuthorizationError::PolicyUnavailable => {
                Self::PqKeyPolicyUnavailable
            }
        }
    }

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
    policy_digest_sha256: [u8; 32],
    key_id: Rfc9942PqKeyId,
    verifying_key_sha256: [u8; 32],
    receipt_sha256: [u8; 32],
    classical_capability_sha256: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(HYBRID_TRANSCRIPT_DOMAIN);
    hasher.update(HYBRID_POLICY_VERSION.to_be_bytes());
    hasher.update(ML_DSA_65_COSE_ALGORITHM_ID.to_be_bytes());
    hasher.update(policy_digest_sha256);
    hasher.update(key_id.0);
    hasher.update(verifying_key_sha256);
    hasher.update((receipt_sha256.len() as u32).to_be_bytes());
    hasher.update(receipt_sha256);
    hasher.update((classical_capability_sha256.len() as u32).to_be_bytes());
    hasher.update(classical_capability_sha256);
    hasher.finalize().into()
}

fn digest_hybrid_capability(
    classical_capability_sha256: [u8; 32],
    policy_digest_sha256: [u8; 32],
    evaluation_time_unix_seconds: u64,
    pq_attestation: &Rfc9942PqAttestation,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(
        b"symthaea-swarm/rfc9942-hybrid-verified-receipt-capability-v1",
    );
    hasher.update(HYBRID_POLICY_VERSION.to_be_bytes());
    hasher.update(classical_capability_sha256);
    hasher.update(policy_digest_sha256);
    hasher.update(evaluation_time_unix_seconds.to_be_bytes());
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
        let key_id = Rfc9942PqKeyId([9; 16]);
        let verifying_key_sha256 = [8; 32];
        let policy_digest_sha256 = [7; 32];
        Rfc9942HybridTranscript {
            policy_digest_sha256,
            key_id,
            verifying_key_sha256,
            receipt_sha256,
            classical_capability_sha256,
            transcript_sha256: digest_transcript(
                policy_digest_sha256,
                key_id,
                verifying_key_sha256,
                receipt_sha256,
                classical_capability_sha256,
            ),
        }
    }

    #[test]
    fn transcript_is_fixed_width_and_domain_separated() {
        let t = transcript(1, 2);
        assert_eq!(t.signing_bytes().len(), 186);

        let mut second_domain = HYBRID_TRANSCRIPT_DOMAIN.to_vec();
        second_domain.push(0);
        let mut hasher = Sha256::new();
        hasher.update(&second_domain);
        hasher.update(HYBRID_POLICY_VERSION.to_be_bytes());
        hasher.update(ML_DSA_65_COSE_ALGORITHM_ID.to_be_bytes());
        hasher.update([7u8; 32]);
        hasher.update([9u8; 16]);
        hasher.update([8u8; 32]);
        hasher.update((32u32).to_be_bytes());
        hasher.update([1u8; 32]);
        hasher.update((32u32).to_be_bytes());
        hasher.update([2u8; 32]);

        assert_ne!(t.transcript_sha256(), hasher.finalize().into());
    }

    #[test]
    fn signing_bytes_bind_policy_digest() {
        let mut a = transcript(1, 2);
        let mut b = a;
        b.policy_digest_sha256 = [6; 32];
        b.transcript_sha256 = digest_transcript(
            b.policy_digest_sha256,
            b.key_id,
            b.verifying_key_sha256,
            b.receipt_sha256,
            b.classical_capability_sha256,
        );
        assert_ne!(a.signing_bytes(), b.signing_bytes());
        assert_ne!(a.transcript_sha256(), b.transcript_sha256());
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
    fn transcript_changes_when_key_identity_changes() {
        let a = transcript(1, 2);
        let mut b = transcript(1, 2);
        b.key_id = Rfc9942PqKeyId([10; 16]);
        b.transcript_sha256 = digest_transcript(
            b.policy_digest_sha256,
            b.key_id,
            b.verifying_key_sha256,
            b.receipt_sha256,
            b.classical_capability_sha256,
        );
        assert_ne!(a.signing_bytes(), b.signing_bytes());
        assert_ne!(a.transcript_sha256(), b.transcript_sha256());
        assert_ne!(a.key_id(), b.key_id());
    }

    #[test]
    fn authorization_receipt_binds_policy_key_and_evaluation_time() {
        let key_id = Rfc9942PqKeyId::new([4; 16]).unwrap();
        let auth = MlDsa65KeyAuthorization::new([3; 32], key_id, [2; 32], 1234)
            .unwrap();
        assert_eq!(auth.policy_digest_sha256(), [3; 32]);
        assert_eq!(auth.key_id(), key_id);
        assert_eq!(auth.verifying_key_sha256(), [2; 32]);
        assert_eq!(auth.evaluation_time_unix_seconds(), 1234);
        assert_eq!(
            MlDsa65KeyAuthorization::new([0; 32], key_id, [2; 32], 1234),
            Err(MlDsa65KeyAuthorizationError::InvalidPolicyDigest)
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
