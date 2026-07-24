// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hybrid classical and post-quantum public verification.
//!
//! Series 20 established a secret-free Ed25519 public-verification bundle. This
//! module adds an ML-DSA-65 overlay without weakening or silently replacing that
//! established path. The same bounded, domain-separated transcript is signed by
//! both algorithms whenever hybrid mode is required.

use std::collections::HashSet;

use fips204::{
    ml_dsa_65,
    traits::{SerDes, Signer, Verifier},
};
use serde::{Deserialize, Serialize};

use crate::{
    CheckpointPublicKeyId, CheckpointPublicSignature, CheckpointPublicSigningKey,
    CheckpointPublicVerificationBundle, CheckpointPublicVerificationError,
    CheckpointPublicVerifyingKey, MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES,
    MAX_CHECKPOINT_PUBLIC_SIGNATURE_DOMAIN_BYTES,
};

pub const CHECKPOINT_ML_DSA_65_VERIFYING_KEY_SCHEMA: &str =
    "symthaea.checkpoint-ml-dsa-65-verifying-key.v1";
pub const CHECKPOINT_ML_DSA_65_SIGNATURE_SCHEMA: &str =
    "symthaea.checkpoint-ml-dsa-65-signature.v1";
pub const CHECKPOINT_HYBRID_SIGNER_SCHEMA: &str =
    "symthaea.checkpoint-hybrid-public-signer.v1";
pub const CHECKPOINT_HYBRID_POLICY_SCHEMA: &str =
    "symthaea.checkpoint-hybrid-public-policy.v1";
pub const CHECKPOINT_HYBRID_ENDORSEMENT_SCHEMA: &str =
    "symthaea.checkpoint-hybrid-public-endorsement.v1";
pub const CHECKPOINT_HYBRID_VERIFICATION_BUNDLE_SCHEMA: &str =
    "symthaea.checkpoint-hybrid-verification-bundle.v1";
pub const CHECKPOINT_HYBRID_VERIFICATION_SUMMARY_SCHEMA: &str =
    "symthaea.checkpoint-hybrid-verification-summary.v1";

pub const MAX_CHECKPOINT_HYBRID_SIGNERS: usize = 128;
pub const MAX_CHECKPOINT_HYBRID_ENDORSEMENTS: usize = 256;
pub const ML_DSA_65_PUBLIC_KEY_BYTES: usize = ml_dsa_65::PK_LEN;
pub const ML_DSA_65_SIGNATURE_BYTES: usize = ml_dsa_65::SIG_LEN;

const HYBRID_PUBLICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea-checkpoint-hybrid-publication-digest-v1\0";
const HYBRID_POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea-checkpoint-hybrid-policy-digest-v1\0";
const HYBRID_ENDORSEMENT_BODY_DOMAIN: &[u8] =
    b"symthaea-checkpoint-hybrid-endorsement-body-v1\0";
const HYBRID_ENDORSEMENT_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea-checkpoint-hybrid-endorsement-signature-v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CheckpointMlDsa65KeyId(pub [u8; 16]);

impl CheckpointMlDsa65KeyId {
    pub fn new(bytes: [u8; 16]) -> Result<Self, CheckpointHybridVerificationError> {
        if bytes == [0u8; 16] {
            return Err(CheckpointHybridVerificationError::InvalidPostQuantumKeyId);
        }
        Ok(Self(bytes))
    }

    pub fn from_verifying_key_bytes(bytes: &[u8]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-checkpoint-ml-dsa-65-key-id-v1\0");
        hasher.update(bytes);
        let digest = hasher.finalize();
        let mut id = [0u8; 16];
        id.copy_from_slice(&digest.as_bytes()[..16]);
        if id == [0u8; 16] {
            id[15] = 1;
        }
        Self(id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointMlDsa65VerifyingKey {
    pub schema: String,
    pub key_id: CheckpointMlDsa65KeyId,
    pub verifying_key_bytes: Vec<u8>,
}

impl CheckpointMlDsa65VerifyingKey {
    pub fn new(
        key_id: CheckpointMlDsa65KeyId,
        verifying_key_bytes: Vec<u8>,
    ) -> Result<Self, CheckpointHybridVerificationError> {
        let key = Self {
            schema: CHECKPOINT_ML_DSA_65_VERIFYING_KEY_SCHEMA.to_owned(),
            key_id,
            verifying_key_bytes,
        };
        key.validate()?;
        Ok(key)
    }

    pub fn validate(&self) -> Result<(), CheckpointHybridVerificationError> {
        if self.schema != CHECKPOINT_ML_DSA_65_VERIFYING_KEY_SCHEMA
            || self.key_id.0 == [0u8; 16]
            || self.verifying_key_bytes.len() != ML_DSA_65_PUBLIC_KEY_BYTES
            || self.verifying_key_bytes.iter().all(|byte| *byte == 0)
        {
            return Err(CheckpointHybridVerificationError::InvalidPostQuantumVerifyingKey);
        }
        let bytes: [u8; ml_dsa_65::PK_LEN] = self
            .verifying_key_bytes
            .as_slice()
            .try_into()
            .map_err(|_| CheckpointHybridVerificationError::InvalidPostQuantumVerifyingKey)?;
        ml_dsa_65::PublicKey::try_from_bytes(bytes)
            .map_err(|_| CheckpointHybridVerificationError::InvalidPostQuantumVerifyingKey)?;
        Ok(())
    }

    pub fn verify(
        &self,
        domain: &[u8],
        message: &[u8],
        signature: &CheckpointMlDsa65Signature,
    ) -> Result<(), CheckpointHybridVerificationError> {
        self.validate()?;
        validate_hybrid_message(domain, message)?;
        signature.validate()?;
        if signature.key_id != self.key_id {
            return Err(CheckpointHybridVerificationError::WrongPostQuantumKey);
        }
        let public_key_bytes: [u8; ml_dsa_65::PK_LEN] = self
            .verifying_key_bytes
            .as_slice()
            .try_into()
            .map_err(|_| CheckpointHybridVerificationError::InvalidPostQuantumVerifyingKey)?;
        let signature_bytes: [u8; ml_dsa_65::SIG_LEN] = signature
            .signature_bytes
            .as_slice()
            .try_into()
            .map_err(|_| CheckpointHybridVerificationError::InvalidPostQuantumSignature)?;
        let public_key = ml_dsa_65::PublicKey::try_from_bytes(public_key_bytes)
            .map_err(|_| CheckpointHybridVerificationError::InvalidPostQuantumVerifyingKey)?;
        let transcript = hybrid_domain_separated_message(domain, message);
        if public_key.verify(&transcript, &signature_bytes, &[]) {
            Ok(())
        } else {
            Err(CheckpointHybridVerificationError::PostQuantumSignatureFailed)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointMlDsa65Signature {
    pub schema: String,
    pub key_id: CheckpointMlDsa65KeyId,
    pub signature_bytes: Vec<u8>,
}

impl CheckpointMlDsa65Signature {
    pub fn validate(&self) -> Result<(), CheckpointHybridVerificationError> {
        if self.schema != CHECKPOINT_ML_DSA_65_SIGNATURE_SCHEMA
            || self.key_id.0 == [0u8; 16]
            || self.signature_bytes.len() != ML_DSA_65_SIGNATURE_BYTES
            || self.signature_bytes.iter().all(|byte| *byte == 0)
        {
            return Err(CheckpointHybridVerificationError::InvalidPostQuantumSignature);
        }
        Ok(())
    }
}

pub trait CheckpointMlDsa65SigningProvider {
    fn key_id(&self) -> CheckpointMlDsa65KeyId;

    fn verifying_key(
        &self,
    ) -> Result<CheckpointMlDsa65VerifyingKey, CheckpointHybridVerificationError>;

    fn sign(
        &self,
        domain: &[u8],
        message: &[u8],
    ) -> Result<CheckpointMlDsa65Signature, CheckpointHybridVerificationError>;
}

pub struct CheckpointSoftwareMlDsa65SigningKey {
    key_id: CheckpointMlDsa65KeyId,
    private_key: ml_dsa_65::PrivateKey,
    verifying_key_bytes: Vec<u8>,
}

impl CheckpointSoftwareMlDsa65SigningKey {
    pub fn generate(
        key_id: CheckpointMlDsa65KeyId,
    ) -> Result<Self, CheckpointHybridVerificationError> {
        let (public_key, private_key) = ml_dsa_65::try_keygen()
            .map_err(|_| CheckpointHybridVerificationError::PostQuantumKeyGenerationFailed)?;
        let verifying_key_bytes = public_key.into_bytes().to_vec();
        let key = Self {
            key_id,
            private_key,
            verifying_key_bytes,
        };
        key.verifying_key()?.validate()?;
        Ok(key)
    }
}

impl CheckpointMlDsa65SigningProvider for CheckpointSoftwareMlDsa65SigningKey {
    fn key_id(&self) -> CheckpointMlDsa65KeyId {
        self.key_id
    }

    fn verifying_key(
        &self,
    ) -> Result<CheckpointMlDsa65VerifyingKey, CheckpointHybridVerificationError> {
        CheckpointMlDsa65VerifyingKey::new(
            self.key_id,
            self.verifying_key_bytes.clone(),
        )
    }

    fn sign(
        &self,
        domain: &[u8],
        message: &[u8],
    ) -> Result<CheckpointMlDsa65Signature, CheckpointHybridVerificationError> {
        validate_hybrid_message(domain, message)?;
        let transcript = hybrid_domain_separated_message(domain, message);
        let signature = self
            .private_key
            .try_sign(&transcript, &[])
            .map_err(|_| CheckpointHybridVerificationError::PostQuantumSigningFailed)?;
        Ok(CheckpointMlDsa65Signature {
            schema: CHECKPOINT_ML_DSA_65_SIGNATURE_SCHEMA.to_owned(),
            key_id: self.key_id,
            signature_bytes: signature.to_vec(),
        })
    }
}

fn validate_hybrid_message(
    domain: &[u8],
    message: &[u8],
) -> Result<(), CheckpointHybridVerificationError> {
    if domain.is_empty()
        || domain.len() > MAX_CHECKPOINT_PUBLIC_SIGNATURE_DOMAIN_BYTES
        || message.is_empty()
        || message.len() > MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES
    {
        return Err(CheckpointHybridVerificationError::InvalidMessage);
    }
    Ok(())
}

fn hybrid_domain_separated_message(domain: &[u8], message: &[u8]) -> Vec<u8> {
    let mut transcript = Vec::with_capacity(8 + domain.len() + message.len());
    transcript.extend_from_slice(&(domain.len() as u64).to_le_bytes());
    transcript.extend_from_slice(domain);
    transcript.extend_from_slice(message);
    transcript
}

fn hybrid_digest<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<[u8; 32], CheckpointHybridVerificationError> {
    let encoded = postcard::to_stdvec(value)
        .map_err(|_| CheckpointHybridVerificationError::Encoding)?;
    if encoded.is_empty() || encoded.len() > MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES {
        return Err(CheckpointHybridVerificationError::TooLarge);
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok(*hasher.finalize().as_bytes())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointHybridVerificationError {
    InvalidPostQuantumKeyId,
    InvalidPostQuantumVerifyingKey,
    InvalidPostQuantumSignature,
    WrongPostQuantumKey,
    PostQuantumKeyGenerationFailed,
    PostQuantumSigningFailed,
    PostQuantumSignatureFailed,
    InvalidMessage,
    Encoding,
    TooLarge,
    InvalidHybridSigner,
    InvalidHybridPolicy,
    InvalidHybridEndorsement,
    DuplicateHybridSigner,
    UnknownHybridSigner,
    InsufficientHybridSignatures,
    HybridDowngrade,
    InvalidHybridBundle,
    ClassicalVerificationFailed,
}

impl std::fmt::Display for CheckpointHybridVerificationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidPostQuantumKeyId => "invalid ML-DSA-65 key identifier",
            Self::InvalidPostQuantumVerifyingKey => "invalid ML-DSA-65 verifying key",
            Self::InvalidPostQuantumSignature => "invalid ML-DSA-65 signature",
            Self::WrongPostQuantumKey => "ML-DSA-65 signature uses the wrong key",
            Self::PostQuantumKeyGenerationFailed => "ML-DSA-65 key generation failed",
            Self::PostQuantumSigningFailed => "ML-DSA-65 signing failed",
            Self::PostQuantumSignatureFailed => "ML-DSA-65 verification failed",
            Self::InvalidMessage => "invalid hybrid signature message",
            Self::Encoding => "hybrid artifact encoding failed",
            Self::TooLarge => "hybrid artifact exceeds its bound",
            Self::InvalidHybridSigner => "invalid hybrid signer",
            Self::InvalidHybridPolicy => "invalid hybrid signature policy",
            Self::InvalidHybridEndorsement => "invalid hybrid endorsement",
            Self::DuplicateHybridSigner => "duplicate hybrid signer",
            Self::UnknownHybridSigner => "unknown hybrid signer",
            Self::InsufficientHybridSignatures => "insufficient hybrid signatures",
            Self::HybridDowngrade => "post-quantum signature downgrade rejected",
            Self::InvalidHybridBundle => "invalid hybrid verification bundle",
            Self::ClassicalVerificationFailed => "classical public verification failed",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for CheckpointHybridVerificationError {}

impl From<CheckpointPublicVerificationError> for CheckpointHybridVerificationError {
    fn from(_: CheckpointPublicVerificationError) -> Self {
        Self::ClassicalVerificationFailed
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckpointHybridSignatureRequirement {
    ClassicalOnlyAllowed,
    HybridRequired,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointHybridPublicSigner {
    pub schema: String,
    pub classical_verifying_key: CheckpointPublicVerifyingKey,
    pub post_quantum_verifying_key: CheckpointMlDsa65VerifyingKey,
    pub organization_binding: [u8; 32],
    pub valid_from_unix_seconds: u64,
    pub valid_until_unix_seconds: u64,
}

impl CheckpointHybridPublicSigner {
    pub fn validate(&self) -> Result<(), CheckpointHybridVerificationError> {
        self.classical_verifying_key.validate()?;
        self.post_quantum_verifying_key.validate()?;
        if self.schema != CHECKPOINT_HYBRID_SIGNER_SCHEMA
            || self.organization_binding == [0u8; 32]
            || self.valid_from_unix_seconds == 0
            || self.valid_until_unix_seconds <= self.valid_from_unix_seconds
        {
            return Err(CheckpointHybridVerificationError::InvalidHybridSigner);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointHybridSignaturePolicy {
    pub schema: String,
    pub policy_id: [u8; 16],
    pub signers: Vec<CheckpointHybridPublicSigner>,
    pub threshold: u16,
    pub valid_from_unix_seconds: u64,
    pub hybrid_required_from_unix_seconds: u64,
    pub valid_until_unix_seconds: u64,
}

impl CheckpointHybridSignaturePolicy {
    pub fn validate(&self) -> Result<(), CheckpointHybridVerificationError> {
        if self.schema != CHECKPOINT_HYBRID_POLICY_SCHEMA
            || self.policy_id == [0u8; 16]
            || self.signers.len() < 2
            || self.signers.len() > MAX_CHECKPOINT_HYBRID_SIGNERS
            || self.threshold < 2
            || usize::from(self.threshold) > self.signers.len()
            || self.valid_from_unix_seconds == 0
            || self.hybrid_required_from_unix_seconds < self.valid_from_unix_seconds
            || self.hybrid_required_from_unix_seconds > self.valid_until_unix_seconds
            || self.valid_until_unix_seconds <= self.valid_from_unix_seconds
        {
            return Err(CheckpointHybridVerificationError::InvalidHybridPolicy);
        }
        let mut classical_ids = HashSet::with_capacity(self.signers.len());
        let mut classical_keys = HashSet::with_capacity(self.signers.len());
        let mut pq_ids = HashSet::with_capacity(self.signers.len());
        let mut pq_keys = HashSet::with_capacity(self.signers.len());
        let mut organizations = HashSet::with_capacity(self.signers.len());
        for signer in &self.signers {
            signer.validate()?;
            if signer.valid_from_unix_seconds > self.valid_from_unix_seconds
                || signer.valid_until_unix_seconds < self.valid_until_unix_seconds
                || !classical_ids.insert(signer.classical_verifying_key.key_id)
                || !classical_keys.insert(signer.classical_verifying_key.verifying_key_bytes)
                || !pq_ids.insert(signer.post_quantum_verifying_key.key_id)
                || !pq_keys.insert(signer.post_quantum_verifying_key.verifying_key_bytes.clone())
                || !organizations.insert(signer.organization_binding)
            {
                return Err(CheckpointHybridVerificationError::DuplicateHybridSigner);
            }
        }
        Ok(())
    }

    pub fn requirement_at(
        &self,
        verification_time_unix_seconds: u64,
    ) -> Result<CheckpointHybridSignatureRequirement, CheckpointHybridVerificationError> {
        self.validate()?;
        if verification_time_unix_seconds < self.valid_from_unix_seconds
            || verification_time_unix_seconds > self.valid_until_unix_seconds
        {
            return Err(CheckpointHybridVerificationError::InvalidHybridPolicy);
        }
        if verification_time_unix_seconds >= self.hybrid_required_from_unix_seconds {
            Ok(CheckpointHybridSignatureRequirement::HybridRequired)
        } else {
            Ok(CheckpointHybridSignatureRequirement::ClassicalOnlyAllowed)
        }
    }

    pub fn digest(&self) -> Result<[u8; 32], CheckpointHybridVerificationError> {
        self.validate()?;
        hybrid_digest(HYBRID_POLICY_DIGEST_DOMAIN, self)
    }

    pub fn signer_by_classical_id(
        &self,
        key_id: CheckpointPublicKeyId,
    ) -> Option<&CheckpointHybridPublicSigner> {
        self.signers
            .iter()
            .find(|signer| signer.classical_verifying_key.key_id == key_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CheckpointHybridEndorsementBody {
    policy_digest: [u8; 32],
    publication_digest: [u8; 32],
    classical_key_id: CheckpointPublicKeyId,
    post_quantum_key_id: CheckpointMlDsa65KeyId,
    signed_at_unix_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointHybridPublicationEndorsement {
    pub schema: String,
    pub policy_digest: [u8; 32],
    pub publication_digest: [u8; 32],
    pub classical_key_id: CheckpointPublicKeyId,
    pub post_quantum_key_id: CheckpointMlDsa65KeyId,
    pub signed_at_unix_seconds: u64,
    pub classical_signature: CheckpointPublicSignature,
    pub post_quantum_signature: Option<CheckpointMlDsa65Signature>,
}

impl CheckpointHybridPublicationEndorsement {
    pub fn sign_hybrid(
        classical_signing_key: &CheckpointPublicSigningKey,
        post_quantum_provider: &impl CheckpointMlDsa65SigningProvider,
        policy: &CheckpointHybridSignaturePolicy,
        publication_digest: [u8; 32],
        signed_at_unix_seconds: u64,
    ) -> Result<Self, CheckpointHybridVerificationError> {
        let signer = policy
            .signer_by_classical_id(classical_signing_key.key_id())
            .ok_or(CheckpointHybridVerificationError::UnknownHybridSigner)?;
        if signer.post_quantum_verifying_key.key_id != post_quantum_provider.key_id()
            || publication_digest == [0u8; 32]
            || signed_at_unix_seconds < signer.valid_from_unix_seconds
            || signed_at_unix_seconds > signer.valid_until_unix_seconds
        {
            return Err(CheckpointHybridVerificationError::InvalidHybridEndorsement);
        }
        let policy_digest = policy.digest()?;
        let body = CheckpointHybridEndorsementBody {
            policy_digest,
            publication_digest,
            classical_key_id: classical_signing_key.key_id(),
            post_quantum_key_id: post_quantum_provider.key_id(),
            signed_at_unix_seconds,
        };
        let body_digest = hybrid_digest(HYBRID_ENDORSEMENT_BODY_DOMAIN, &body)?;
        Ok(Self {
            schema: CHECKPOINT_HYBRID_ENDORSEMENT_SCHEMA.to_owned(),
            policy_digest,
            publication_digest,
            classical_key_id: body.classical_key_id,
            post_quantum_key_id: body.post_quantum_key_id,
            signed_at_unix_seconds,
            classical_signature: classical_signing_key.sign(
                HYBRID_ENDORSEMENT_SIGNATURE_DOMAIN,
                &body_digest,
            )?,
            post_quantum_signature: Some(post_quantum_provider.sign(
                HYBRID_ENDORSEMENT_SIGNATURE_DOMAIN,
                &body_digest,
            )?),
        })
    }

    pub fn sign_classical_only(
        classical_signing_key: &CheckpointPublicSigningKey,
        policy: &CheckpointHybridSignaturePolicy,
        publication_digest: [u8; 32],
        signed_at_unix_seconds: u64,
    ) -> Result<Self, CheckpointHybridVerificationError> {
        let signer = policy
            .signer_by_classical_id(classical_signing_key.key_id())
            .ok_or(CheckpointHybridVerificationError::UnknownHybridSigner)?;
        if publication_digest == [0u8; 32]
            || signed_at_unix_seconds < signer.valid_from_unix_seconds
            || signed_at_unix_seconds > signer.valid_until_unix_seconds
            || signed_at_unix_seconds >= policy.hybrid_required_from_unix_seconds
        {
            return Err(CheckpointHybridVerificationError::HybridDowngrade);
        }
        let policy_digest = policy.digest()?;
        let body = CheckpointHybridEndorsementBody {
            policy_digest,
            publication_digest,
            classical_key_id: classical_signing_key.key_id(),
            post_quantum_key_id: signer.post_quantum_verifying_key.key_id,
            signed_at_unix_seconds,
        };
        let body_digest = hybrid_digest(HYBRID_ENDORSEMENT_BODY_DOMAIN, &body)?;
        Ok(Self {
            schema: CHECKPOINT_HYBRID_ENDORSEMENT_SCHEMA.to_owned(),
            policy_digest,
            publication_digest,
            classical_key_id: body.classical_key_id,
            post_quantum_key_id: body.post_quantum_key_id,
            signed_at_unix_seconds,
            classical_signature: classical_signing_key.sign(
                HYBRID_ENDORSEMENT_SIGNATURE_DOMAIN,
                &body_digest,
            )?,
            post_quantum_signature: None,
        })
    }

    fn body_digest(&self) -> Result<[u8; 32], CheckpointHybridVerificationError> {
        if self.schema != CHECKPOINT_HYBRID_ENDORSEMENT_SCHEMA
            || self.policy_digest == [0u8; 32]
            || self.publication_digest == [0u8; 32]
            || self.classical_key_id.0 == [0u8; 16]
            || self.post_quantum_key_id.0 == [0u8; 16]
            || self.signed_at_unix_seconds == 0
        {
            return Err(CheckpointHybridVerificationError::InvalidHybridEndorsement);
        }
        hybrid_digest(
            HYBRID_ENDORSEMENT_BODY_DOMAIN,
            &CheckpointHybridEndorsementBody {
                policy_digest: self.policy_digest,
                publication_digest: self.publication_digest,
                classical_key_id: self.classical_key_id,
                post_quantum_key_id: self.post_quantum_key_id,
                signed_at_unix_seconds: self.signed_at_unix_seconds,
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointHybridVerificationBundle {
    pub schema: String,
    pub classical_bundle: CheckpointPublicVerificationBundle,
    pub policy: CheckpointHybridSignaturePolicy,
    pub endorsements: Vec<CheckpointHybridPublicationEndorsement>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointHybridVerificationSummary {
    pub schema: String,
    pub publication_digest: [u8; 32],
    pub policy_digest: [u8; 32],
    pub requirement: CheckpointHybridSignatureRequirement,
    pub valid_classical_signatures: usize,
    pub valid_post_quantum_signatures: usize,
    pub unique_organizations: usize,
}

impl CheckpointHybridVerificationSummary {
    pub fn validate(&self) -> Result<(), CheckpointHybridVerificationError> {
        if self.schema != CHECKPOINT_HYBRID_VERIFICATION_SUMMARY_SCHEMA
            || self.publication_digest == [0u8; 32]
            || self.policy_digest == [0u8; 32]
            || self.valid_classical_signatures < 2
            || self.unique_organizations < 2
            || (self.requirement == CheckpointHybridSignatureRequirement::HybridRequired
                && self.valid_post_quantum_signatures < 2)
        {
            return Err(CheckpointHybridVerificationError::InvalidHybridBundle);
        }
        Ok(())
    }
}

impl CheckpointHybridVerificationBundle {
    pub fn verify(
        &self,
        verification_time_unix_seconds: u64,
    ) -> Result<CheckpointHybridVerificationSummary, CheckpointHybridVerificationError> {
        if self.schema != CHECKPOINT_HYBRID_VERIFICATION_BUNDLE_SCHEMA
            || self.endorsements.is_empty()
            || self.endorsements.len() > MAX_CHECKPOINT_HYBRID_ENDORSEMENTS
        {
            return Err(CheckpointHybridVerificationError::InvalidHybridBundle);
        }
        self.classical_bundle.verify(verification_time_unix_seconds)?;
        self.policy.validate()?;
        let requirement = self.policy.requirement_at(verification_time_unix_seconds)?;
        let publication_digest = hybrid_digest(
            HYBRID_PUBLICATION_DIGEST_DOMAIN,
            &self.classical_bundle,
        )?;
        let policy_digest = self.policy.digest()?;
        let mut classical_signers = HashSet::new();
        let mut pq_signers = HashSet::new();
        let mut organizations = HashSet::new();
        for endorsement in &self.endorsements {
            let signer = self
                .policy
                .signer_by_classical_id(endorsement.classical_key_id)
                .ok_or(CheckpointHybridVerificationError::UnknownHybridSigner)?;
            if endorsement.policy_digest != policy_digest
                || endorsement.publication_digest != publication_digest
                || endorsement.post_quantum_key_id != signer.post_quantum_verifying_key.key_id
                || endorsement.signed_at_unix_seconds < signer.valid_from_unix_seconds
                || endorsement.signed_at_unix_seconds > signer.valid_until_unix_seconds
                || endorsement.signed_at_unix_seconds > verification_time_unix_seconds
                || !classical_signers.insert(endorsement.classical_key_id)
                || !organizations.insert(signer.organization_binding)
            {
                return Err(CheckpointHybridVerificationError::InvalidHybridEndorsement);
            }
            let body_digest = endorsement.body_digest()?;
            signer.classical_verifying_key.verify(
                HYBRID_ENDORSEMENT_SIGNATURE_DOMAIN,
                &body_digest,
                &endorsement.classical_signature,
            )?;
            match (&requirement, &endorsement.post_quantum_signature) {
                (
                    CheckpointHybridSignatureRequirement::HybridRequired,
                    Some(signature),
                ) => {
                    signer.post_quantum_verifying_key.verify(
                        HYBRID_ENDORSEMENT_SIGNATURE_DOMAIN,
                        &body_digest,
                        signature,
                    )?;
                    if !pq_signers.insert(endorsement.post_quantum_key_id) {
                        return Err(CheckpointHybridVerificationError::DuplicateHybridSigner);
                    }
                }
                (CheckpointHybridSignatureRequirement::HybridRequired, None) => {
                    return Err(CheckpointHybridVerificationError::HybridDowngrade);
                }
                (
                    CheckpointHybridSignatureRequirement::ClassicalOnlyAllowed,
                    Some(signature),
                ) => {
                    signer.post_quantum_verifying_key.verify(
                        HYBRID_ENDORSEMENT_SIGNATURE_DOMAIN,
                        &body_digest,
                        signature,
                    )?;
                    if !pq_signers.insert(endorsement.post_quantum_key_id) {
                        return Err(CheckpointHybridVerificationError::DuplicateHybridSigner);
                    }
                }
                (CheckpointHybridSignatureRequirement::ClassicalOnlyAllowed, None) => {}
            }
        }
        if classical_signers.len() < usize::from(self.policy.threshold)
            || organizations.len() < usize::from(self.policy.threshold)
            || (requirement == CheckpointHybridSignatureRequirement::HybridRequired
                && pq_signers.len() < usize::from(self.policy.threshold))
        {
            return Err(CheckpointHybridVerificationError::InsufficientHybridSignatures);
        }
        let summary = CheckpointHybridVerificationSummary {
            schema: CHECKPOINT_HYBRID_VERIFICATION_SUMMARY_SCHEMA.to_owned(),
            publication_digest,
            policy_digest,
            requirement,
            valid_classical_signatures: classical_signers.len(),
            valid_post_quantum_signatures: pq_signers.len(),
            unique_organizations: organizations.len(),
        };
        summary.validate()?;
        Ok(summary)
    }
}

pub fn encode_checkpoint_hybrid_verification_bundle(
    bundle: &CheckpointHybridVerificationBundle,
) -> Result<Vec<u8>, CheckpointHybridVerificationError> {
    let encoded = postcard::to_stdvec(bundle)
        .map_err(|_| CheckpointHybridVerificationError::Encoding)?;
    if encoded.is_empty() || encoded.len() > MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES {
        return Err(CheckpointHybridVerificationError::TooLarge);
    }
    Ok(encoded)
}

pub fn decode_checkpoint_hybrid_verification_bundle(
    encoded: &[u8],
) -> Result<CheckpointHybridVerificationBundle, CheckpointHybridVerificationError> {
    if encoded.is_empty() || encoded.len() > MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES {
        return Err(CheckpointHybridVerificationError::TooLarge);
    }
    let bundle: CheckpointHybridVerificationBundle = postcard::from_bytes(encoded)
        .map_err(|_| CheckpointHybridVerificationError::Encoding)?;
    if bundle.schema != CHECKPOINT_HYBRID_VERIFICATION_BUNDLE_SCHEMA {
        return Err(CheckpointHybridVerificationError::InvalidHybridBundle);
    }
    Ok(bundle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classical_key(id: u8, seed: u8) -> CheckpointPublicSigningKey {
        CheckpointPublicSigningKey::from_seed(
            CheckpointPublicKeyId::new([id; 16]).unwrap(),
            [seed; 32],
        )
        .unwrap()
    }

    fn hybrid_policy(
        first: &CheckpointPublicSigningKey,
        first_pq: &CheckpointSoftwareMlDsa65SigningKey,
        second: &CheckpointPublicSigningKey,
        second_pq: &CheckpointSoftwareMlDsa65SigningKey,
    ) -> CheckpointHybridSignaturePolicy {
        CheckpointHybridSignaturePolicy {
            schema: CHECKPOINT_HYBRID_POLICY_SCHEMA.to_owned(),
            policy_id: [0x44; 16],
            signers: vec![
                CheckpointHybridPublicSigner {
                    schema: CHECKPOINT_HYBRID_SIGNER_SCHEMA.to_owned(),
                    classical_verifying_key: first.verifying_key(),
                    post_quantum_verifying_key: first_pq.verifying_key().unwrap(),
                    organization_binding: [0x51; 32],
                    valid_from_unix_seconds: 100,
                    valid_until_unix_seconds: 500,
                },
                CheckpointHybridPublicSigner {
                    schema: CHECKPOINT_HYBRID_SIGNER_SCHEMA.to_owned(),
                    classical_verifying_key: second.verifying_key(),
                    post_quantum_verifying_key: second_pq.verifying_key().unwrap(),
                    organization_binding: [0x52; 32],
                    valid_from_unix_seconds: 100,
                    valid_until_unix_seconds: 500,
                },
            ],
            threshold: 2,
            valid_from_unix_seconds: 100,
            hybrid_required_from_unix_seconds: 300,
            valid_until_unix_seconds: 500,
        }
    }

    #[test]
    fn ml_dsa_65_round_trip_verifies() {
        let key = CheckpointSoftwareMlDsa65SigningKey::generate(
            CheckpointMlDsa65KeyId::new([0x31; 16]).unwrap(),
        )
        .unwrap();
        let signature = key.sign(b"test-domain", b"public evidence").unwrap();
        key.verifying_key()
            .unwrap()
            .verify(b"test-domain", b"public evidence", &signature)
            .unwrap();
        assert!(key
            .verifying_key()
            .unwrap()
            .verify(b"test-domain", b"altered", &signature)
            .is_err());
    }

    #[test]
    fn migration_cutoff_rejects_classical_only_signing() {
        let first = classical_key(1, 11);
        let second = classical_key(2, 12);
        let first_pq = CheckpointSoftwareMlDsa65SigningKey::generate(
            CheckpointMlDsa65KeyId::new([0x41; 16]).unwrap(),
        )
        .unwrap();
        let second_pq = CheckpointSoftwareMlDsa65SigningKey::generate(
            CheckpointMlDsa65KeyId::new([0x42; 16]).unwrap(),
        )
        .unwrap();
        let policy = hybrid_policy(&first, &first_pq, &second, &second_pq);
        assert_eq!(
            policy.requirement_at(299).unwrap(),
            CheckpointHybridSignatureRequirement::ClassicalOnlyAllowed
        );
        assert_eq!(
            policy.requirement_at(300).unwrap(),
            CheckpointHybridSignatureRequirement::HybridRequired
        );
        CheckpointHybridPublicationEndorsement::sign_classical_only(
            &first,
            &policy,
            [0x61; 32],
            299,
        )
        .unwrap();
        assert!(matches!(
            CheckpointHybridPublicationEndorsement::sign_classical_only(
                &first,
                &policy,
                [0x61; 32],
                300,
            ),
            Err(CheckpointHybridVerificationError::HybridDowngrade)
        ));
    }
}
