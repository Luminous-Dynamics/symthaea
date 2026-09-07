// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generic lifecycle-aware detached attestation semantics for control-plane use.
//!
//! This module separates four concepts that must not collapse into one another:
//!
//! ```text
//! signed envelope
//!     != cryptographically valid signature
//!     != currently trusted signer
//!     != domain/effect authority
//! ```
//!
//! Cryptographic key storage and concrete signature implementations stay behind
//! [`AttestationTrustVerifierV1`]. A successful verification returns a private,
//! non-deserializable [`VerifiedAttestationV1`] witness bound to the exact trust
//! generation and evaluation time used for the decision.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const CONTROL_PLANE_ATTESTATION_VERSION_V1: u16 = 1;
pub const MAX_ATTESTATION_DOMAIN_BYTES_V1: usize = 128;
pub const MAX_ATTESTATION_KEY_ID_BYTES_V1: usize = 256;
pub const MAX_ATTESTATION_SIGNATURE_BYTES_V1: usize = 64 * 1024;
pub const MAX_ATTESTATION_SIGNATURES_V1: usize = 16;
pub const CONTROL_PLANE_ATTESTATION_BODY_DOMAIN_V1: &[u8] =
    b"symthaea.control-plane.detached-attestation-body.v1\0";
pub const CONTROL_PLANE_ATTESTATION_SIGNER_DOMAIN_V1: &[u8] =
    b"symthaea.control-plane.detached-attestation-signer.v1\0";
pub const CONTROL_PLANE_ATTESTATION_ENVELOPE_DOMAIN_V1: &[u8] =
    b"symthaea.control-plane.detached-attestation-envelope.v1\0";

/// Signature algorithms with stable control-plane v1 identities.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum AttestationSignatureAlgorithmV1 {
    Ed25519,
    MlDsa65,
    MlDsa87,
}

impl AttestationSignatureAlgorithmV1 {
    pub const fn canonical_tag(self) -> u8 {
        match self {
            Self::Ed25519 => 1,
            Self::MlDsa65 => 2,
            Self::MlDsa87 => 3,
        }
    }
}

/// Domain-separated commitment to the object being attested.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttestationSubjectV1 {
    pub domain: String,
    pub commitment: [u8; 32],
}

impl AttestationSubjectV1 {
    pub fn new(
        domain: impl Into<String>,
        commitment: [u8; 32],
    ) -> Result<Self, AttestationErrorV1> {
        let subject = Self {
            domain: domain.into(),
            commitment,
        };
        subject.validate()?;
        Ok(subject)
    }

    pub fn validate(&self) -> Result<(), AttestationErrorV1> {
        validate_bounded_nonempty(
            "subject.domain",
            &self.domain,
            MAX_ATTESTATION_DOMAIN_BYTES_V1,
        )?;
        if self.commitment == [0; 32] {
            return Err(AttestationErrorV1::ZeroCommitment("subject.commitment"));
        }
        Ok(())
    }
}

/// Detached signature over signer-bound attestation bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DetachedAttestationSignatureV1 {
    pub algorithm: AttestationSignatureAlgorithmV1,
    pub key_id: String,
    pub signature: Vec<u8>,
}

impl DetachedAttestationSignatureV1 {
    pub fn validate_protocol_bounds(&self) -> Result<(), AttestationErrorV1> {
        validate_bounded_nonempty(
            "signature.key_id",
            &self.key_id,
            MAX_ATTESTATION_KEY_ID_BYTES_V1,
        )?;
        if self.signature.is_empty() {
            return Err(AttestationErrorV1::EmptySignature);
        }
        if self.signature.len() > MAX_ATTESTATION_SIGNATURE_BYTES_V1 {
            return Err(AttestationErrorV1::SignatureTooLarge {
                actual_bytes: self.signature.len(),
                maximum_bytes: MAX_ATTESTATION_SIGNATURE_BYTES_V1,
            });
        }
        Ok(())
    }
}

/// Serializable detached-signature envelope.
///
/// This is evidence-shaped input only. Deserializing or structurally validating
/// it grants no positive authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DetachedAttestationEnvelopeV1 {
    pub protocol_version: u16,
    pub subject: AttestationSubjectV1,
    pub issued_at_unix_s: u64,
    pub valid_until_unix_s: u64,
    /// Exact trust/authority generation the signer asserts this envelope belongs to.
    pub authority_generation_commitment: [u8; 32],
    pub signatures: Vec<DetachedAttestationSignatureV1>,
}

impl DetachedAttestationEnvelopeV1 {
    /// Validate body fields covered by every detached signature.
    pub fn validate_signing_body(&self) -> Result<(), AttestationErrorV1> {
        if self.protocol_version != CONTROL_PLANE_ATTESTATION_VERSION_V1 {
            return Err(AttestationErrorV1::UnsupportedProtocolVersion(
                self.protocol_version,
            ));
        }
        self.subject.validate()?;
        if self.issued_at_unix_s >= self.valid_until_unix_s {
            return Err(AttestationErrorV1::InvalidValidityWindow {
                issued_at_unix_s: self.issued_at_unix_s,
                valid_until_unix_s: self.valid_until_unix_s,
            });
        }
        if self.authority_generation_commitment == [0; 32] {
            return Err(AttestationErrorV1::ZeroCommitment(
                "authority_generation_commitment",
            ));
        }
        Ok(())
    }

    /// Validate the complete envelope as verification input.
    pub fn validate(&self) -> Result<(), AttestationErrorV1> {
        self.validate_signing_body()?;
        if self.signatures.is_empty() {
            return Err(AttestationErrorV1::NoSignatures);
        }
        if self.signatures.len() > MAX_ATTESTATION_SIGNATURES_V1 {
            return Err(AttestationErrorV1::TooManySignatures {
                actual: self.signatures.len(),
                maximum: MAX_ATTESTATION_SIGNATURES_V1,
            });
        }
        let mut identities = BTreeSet::new();
        for signature in &self.signatures {
            signature.validate_protocol_bounds()?;
            let identity = (signature.algorithm, signature.key_id.clone());
            if !identities.insert(identity.clone()) {
                return Err(AttestationErrorV1::DuplicateSigner {
                    algorithm: identity.0,
                    key_id: identity.1,
                });
            }
        }
        Ok(())
    }

    /// Frozen representation-independent bytes for the common attestation body.
    pub fn signing_body_bytes_v1(&self) -> Result<Vec<u8>, AttestationErrorV1> {
        self.validate_signing_body()?;
        let mut out = Vec::with_capacity(
            CONTROL_PLANE_ATTESTATION_BODY_DOMAIN_V1.len()
                + 2
                + 2
                + MAX_ATTESTATION_DOMAIN_BYTES_V1
                + 32
                + 8
                + 8
                + 32,
        );
        out.extend_from_slice(CONTROL_PLANE_ATTESTATION_BODY_DOMAIN_V1);
        out.extend_from_slice(&self.protocol_version.to_be_bytes());
        append_bounded_utf8(
            &mut out,
            "subject.domain",
            &self.subject.domain,
            MAX_ATTESTATION_DOMAIN_BYTES_V1,
        )?;
        out.extend_from_slice(&self.subject.commitment);
        out.extend_from_slice(&self.issued_at_unix_s.to_be_bytes());
        out.extend_from_slice(&self.valid_until_unix_s.to_be_bytes());
        out.extend_from_slice(&self.authority_generation_commitment);
        Ok(out)
    }

    /// Bytes signed by one exact signer identity.
    ///
    /// Binding `(algorithm, key_id)` into the signed message prevents a valid
    /// signature from being relabeled under a different trust-store identity.
    pub fn signer_message_v1(
        &self,
        algorithm: AttestationSignatureAlgorithmV1,
        key_id: &str,
    ) -> Result<Vec<u8>, AttestationErrorV1> {
        validate_bounded_nonempty(
            "signature.key_id",
            key_id,
            MAX_ATTESTATION_KEY_ID_BYTES_V1,
        )?;
        let body = self.signing_body_bytes_v1()?;
        let mut out = Vec::with_capacity(
            CONTROL_PLANE_ATTESTATION_SIGNER_DOMAIN_V1.len()
                + 32
                + 1
                + 2
                + MAX_ATTESTATION_KEY_ID_BYTES_V1,
        );
        out.extend_from_slice(CONTROL_PLANE_ATTESTATION_SIGNER_DOMAIN_V1);
        out.extend_from_slice(blake3::hash(&body).as_bytes());
        out.push(algorithm.canonical_tag());
        append_bounded_utf8(
            &mut out,
            "signature.key_id",
            key_id,
            MAX_ATTESTATION_KEY_ID_BYTES_V1,
        )?;
        Ok(out)
    }

    /// Stable commitment to the complete envelope including detached signatures.
    /// Signature ordering is canonicalized by `(algorithm tag, key_id)`.
    pub fn canonical_commitment_v1(&self) -> Result<[u8; 32], AttestationErrorV1> {
        self.validate()?;
        let body = self.signing_body_bytes_v1()?;
        let mut signatures: Vec<&DetachedAttestationSignatureV1> =
            self.signatures.iter().collect();
        signatures.sort_by(|left, right| {
            left.algorithm
                .canonical_tag()
                .cmp(&right.algorithm.canonical_tag())
                .then_with(|| left.key_id.cmp(&right.key_id))
        });

        let mut out = Vec::with_capacity(
            CONTROL_PLANE_ATTESTATION_ENVELOPE_DOMAIN_V1.len()
                + 32
                + 2
                + signatures.len()
                    * (1 + 2 + MAX_ATTESTATION_KEY_ID_BYTES_V1 + 4),
        );
        out.extend_from_slice(CONTROL_PLANE_ATTESTATION_ENVELOPE_DOMAIN_V1);
        out.extend_from_slice(blake3::hash(&body).as_bytes());
        let count = u16::try_from(signatures.len()).map_err(|_| {
            AttestationErrorV1::TooManySignatures {
                actual: signatures.len(),
                maximum: u16::MAX as usize,
            }
        })?;
        out.extend_from_slice(&count.to_be_bytes());
        for signature in signatures {
            out.push(signature.algorithm.canonical_tag());
            append_bounded_utf8(
                &mut out,
                "signature.key_id",
                &signature.key_id,
                MAX_ATTESTATION_KEY_ID_BYTES_V1,
            )?;
            let signature_len = u32::try_from(signature.signature.len()).map_err(|_| {
                AttestationErrorV1::SignatureTooLarge {
                    actual_bytes: signature.signature.len(),
                    maximum_bytes: u32::MAX as usize,
                }
            })?;
            out.extend_from_slice(&signature_len.to_be_bytes());
            out.extend_from_slice(&signature.signature);
        }
        Ok(*blake3::hash(&out).as_bytes())
    }
}

/// Explicit verification policy. No default is provided: callers must choose a
/// policy rather than inheriting positive authority from library defaults.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestationPolicyV1 {
    pub minimum_valid_signatures: usize,
    pub maximum_signatures: usize,
    pub maximum_signature_bytes: usize,
    pub maximum_key_id_bytes: usize,
    pub required_algorithms: BTreeSet<AttestationSignatureAlgorithmV1>,
    pub allowed_key_ids: Option<BTreeSet<String>>,
}

impl AttestationPolicyV1 {
    pub fn validate(&self) -> Result<(), AttestationErrorV1> {
        if self.minimum_valid_signatures == 0
            || self.maximum_signatures == 0
            || self.minimum_valid_signatures > self.maximum_signatures
            || self.maximum_signatures > MAX_ATTESTATION_SIGNATURES_V1
            || self.maximum_signature_bytes == 0
            || self.maximum_signature_bytes > MAX_ATTESTATION_SIGNATURE_BYTES_V1
            || self.maximum_key_id_bytes == 0
            || self.maximum_key_id_bytes > MAX_ATTESTATION_KEY_ID_BYTES_V1
        {
            return Err(AttestationErrorV1::InvalidPolicy);
        }
        if let Some(allowed) = &self.allowed_key_ids {
            if allowed.is_empty() {
                return Err(AttestationErrorV1::InvalidPolicy);
            }
            for key_id in allowed {
                validate_bounded_nonempty(
                    "policy.allowed_key_id",
                    key_id,
                    self.maximum_key_id_bytes,
                )?;
            }
        }
        Ok(())
    }
}

/// Lifecycle state of one key in an exact trust generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttestationKeyLifecycleV1 {
    Active,
    Retired,
    Revoked,
}

/// Provider-returned key metadata bound to one exact trust generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestationTrustedKeyV1 {
    pub algorithm: AttestationSignatureAlgorithmV1,
    pub key_id: String,
    pub valid_from_unix_s: u64,
    pub valid_until_unix_s: u64,
    pub lifecycle: AttestationKeyLifecycleV1,
    /// Exact subject domains this key may attest under this trust generation.
    pub allowed_subject_domains: BTreeSet<String>,
}

impl AttestationTrustedKeyV1 {
    pub fn validate(&self) -> Result<(), AttestationErrorV1> {
        validate_bounded_nonempty(
            "trusted_key.key_id",
            &self.key_id,
            MAX_ATTESTATION_KEY_ID_BYTES_V1,
        )?;
        if self.valid_from_unix_s >= self.valid_until_unix_s {
            return Err(AttestationErrorV1::InvalidKeyValidityWindow {
                key_id: self.key_id.clone(),
            });
        }
        if self.allowed_subject_domains.is_empty() {
            return Err(AttestationErrorV1::KeyHasNoAllowedDomains(
                self.key_id.clone(),
            ));
        }
        for domain in &self.allowed_subject_domains {
            validate_bounded_nonempty(
                "trusted_key.allowed_subject_domain",
                domain,
                MAX_ATTESTATION_DOMAIN_BYTES_V1,
            )?;
        }
        Ok(())
    }
}

/// Exact trust generation selected by a trust provider for verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestationTrustSnapshotV1 {
    pub generation_commitment: [u8; 32],
    pub valid_from_unix_s: u64,
    pub valid_until_unix_s: u64,
}

impl AttestationTrustSnapshotV1 {
    pub fn validate(&self) -> Result<(), AttestationErrorV1> {
        if self.generation_commitment == [0; 32] {
            return Err(AttestationErrorV1::ZeroCommitment(
                "trust_snapshot.generation_commitment",
            ));
        }
        if self.valid_from_unix_s >= self.valid_until_unix_s {
            return Err(AttestationErrorV1::InvalidTrustSnapshotWindow);
        }
        Ok(())
    }
}

/// Provider boundary for exact-generation trust lookup and cryptographic
/// verification. Implementations own public-key material and lifecycle storage.
pub trait AttestationTrustVerifierV1 {
    /// Return the exact trust generation considered current for this decision.
    fn current_snapshot(&self) -> Result<AttestationTrustSnapshotV1, String>;

    /// Resolve one key strictly inside the requested trust generation.
    fn key_record(
        &self,
        trust_generation_commitment: [u8; 32],
        algorithm: AttestationSignatureAlgorithmV1,
        key_id: &str,
    ) -> Result<Option<AttestationTrustedKeyV1>, String>;

    /// Verify a detached signature using key material from the exact generation.
    fn verify_signature(
        &self,
        trust_generation_commitment: [u8; 32],
        algorithm: AttestationSignatureAlgorithmV1,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

/// Stable signer identity retained by a positive verification witness.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct VerifiedAttestationSignerV1 {
    pub algorithm: AttestationSignatureAlgorithmV1,
    pub key_id: String,
}

/// Positive in-process witness for one exact envelope/trust-generation decision.
///
/// This type intentionally has private fields, no public constructor, and no
/// serde derives. It is not durable positive authority by itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedAttestationV1 {
    subject: AttestationSubjectV1,
    envelope_commitment: [u8; 32],
    trust_generation_commitment: [u8; 32],
    evaluation_time_unix_s: u64,
    natural_valid_until_unix_s: u64,
    valid_signers: Vec<VerifiedAttestationSignerV1>,
}

impl VerifiedAttestationV1 {
    pub fn subject(&self) -> &AttestationSubjectV1 {
        &self.subject
    }

    pub fn envelope_commitment(&self) -> [u8; 32] {
        self.envelope_commitment
    }

    pub fn trust_generation_commitment(&self) -> [u8; 32] {
        self.trust_generation_commitment
    }

    pub fn evaluation_time_unix_s(&self) -> u64 {
        self.evaluation_time_unix_s
    }

    /// Earliest natural time expiry among the envelope, selected trust snapshot,
    /// and every signer retained in this conservative witness.
    pub fn natural_valid_until_unix_s(&self) -> u64 {
        self.natural_valid_until_unix_s
    }

    pub fn valid_signers(&self) -> &[VerifiedAttestationSignerV1] {
        &self.valid_signers
    }

    /// Check time bounds only. This does NOT establish that the same trust
    /// generation is still current or that no revocation occurred.
    pub fn remains_within_time_bounds(&self, at_unix_s: u64) -> bool {
        self.evaluation_time_unix_s <= at_unix_s
            && at_unix_s < self.natural_valid_until_unix_s
    }
}

/// Verify one detached attestation against the provider's exact current trust
/// generation and return a non-serializable positive witness only on full success.
///
/// V1 is intentionally conservative: every signature present in the envelope
/// must be current, policy-admissible, domain-authorized, and cryptographically
/// valid. Invalid extra signatures are not ignored after quorum is satisfied.
pub fn verify_current_attestation_v1(
    envelope: &DetachedAttestationEnvelopeV1,
    policy: &AttestationPolicyV1,
    provider: &dyn AttestationTrustVerifierV1,
    evaluation_time_unix_s: u64,
) -> Result<VerifiedAttestationV1, AttestationErrorV1> {
    policy.validate()?;
    envelope.validate()?;

    if envelope.signatures.len() > policy.maximum_signatures {
        return Err(AttestationErrorV1::TooManySignatures {
            actual: envelope.signatures.len(),
            maximum: policy.maximum_signatures,
        });
    }
    if evaluation_time_unix_s < envelope.issued_at_unix_s {
        return Err(AttestationErrorV1::EnvelopeNotYetValid);
    }
    if evaluation_time_unix_s >= envelope.valid_until_unix_s {
        return Err(AttestationErrorV1::EnvelopeExpired);
    }

    let snapshot = provider
        .current_snapshot()
        .map_err(AttestationErrorV1::TrustProvider)?;
    snapshot.validate()?;
    if envelope.authority_generation_commitment != snapshot.generation_commitment {
        return Err(AttestationErrorV1::TrustGenerationMismatch);
    }
    if evaluation_time_unix_s < snapshot.valid_from_unix_s {
        return Err(AttestationErrorV1::TrustSnapshotNotYetValid);
    }
    if evaluation_time_unix_s >= snapshot.valid_until_unix_s {
        return Err(AttestationErrorV1::TrustSnapshotExpired);
    }

    let mut valid_signers = Vec::with_capacity(envelope.signatures.len());
    let mut valid_algorithms = BTreeSet::new();
    let mut natural_valid_until_unix_s =
        envelope.valid_until_unix_s.min(snapshot.valid_until_unix_s);

    for signature in &envelope.signatures {
        if signature.key_id.len() > policy.maximum_key_id_bytes {
            return Err(AttestationErrorV1::KeyIdTooLong {
                key_id: signature.key_id.clone(),
                actual_bytes: signature.key_id.len(),
                maximum_bytes: policy.maximum_key_id_bytes,
            });
        }
        if signature.signature.len() > policy.maximum_signature_bytes {
            return Err(AttestationErrorV1::SignatureTooLarge {
                actual_bytes: signature.signature.len(),
                maximum_bytes: policy.maximum_signature_bytes,
            });
        }
        if let Some(allowed) = &policy.allowed_key_ids {
            if !allowed.contains(&signature.key_id) {
                return Err(AttestationErrorV1::KeyNotAllowed(signature.key_id.clone()));
            }
        }

        let key = provider
            .key_record(
                snapshot.generation_commitment,
                signature.algorithm,
                &signature.key_id,
            )
            .map_err(AttestationErrorV1::TrustProvider)?
            .ok_or_else(|| AttestationErrorV1::UnknownSigner(signature.key_id.clone()))?;
        key.validate()?;
        if key.algorithm != signature.algorithm || key.key_id != signature.key_id {
            return Err(AttestationErrorV1::TrustProviderKeyIdentityMismatch {
                requested_key_id: signature.key_id.clone(),
                returned_key_id: key.key_id,
            });
        }
        if !key.allowed_subject_domains.contains(&envelope.subject.domain) {
            return Err(AttestationErrorV1::SignerDomainNotAllowed {
                key_id: signature.key_id.clone(),
                domain: envelope.subject.domain.clone(),
            });
        }
        match key.lifecycle {
            AttestationKeyLifecycleV1::Active => {}
            AttestationKeyLifecycleV1::Retired => {
                return Err(AttestationErrorV1::SignerRetired(signature.key_id.clone()));
            }
            AttestationKeyLifecycleV1::Revoked => {
                return Err(AttestationErrorV1::SignerRevoked(signature.key_id.clone()));
            }
        }
        if evaluation_time_unix_s < key.valid_from_unix_s {
            return Err(AttestationErrorV1::SignerNotYetValid(signature.key_id.clone()));
        }
        if evaluation_time_unix_s >= key.valid_until_unix_s {
            return Err(AttestationErrorV1::SignerExpired(signature.key_id.clone()));
        }

        let message = envelope.signer_message_v1(signature.algorithm, &signature.key_id)?;
        let valid = provider
            .verify_signature(
                snapshot.generation_commitment,
                signature.algorithm,
                &signature.key_id,
                &message,
                &signature.signature,
            )
            .map_err(AttestationErrorV1::VerificationProvider)?;
        if !valid {
            return Err(AttestationErrorV1::InvalidSignature {
                algorithm: signature.algorithm,
                key_id: signature.key_id.clone(),
            });
        }

        natural_valid_until_unix_s = natural_valid_until_unix_s.min(key.valid_until_unix_s);
        valid_algorithms.insert(signature.algorithm);
        valid_signers.push(VerifiedAttestationSignerV1 {
            algorithm: signature.algorithm,
            key_id: signature.key_id.clone(),
        });
    }

    if valid_signers.len() < policy.minimum_valid_signatures {
        return Err(AttestationErrorV1::InsufficientValidSignatures {
            actual: valid_signers.len(),
            required: policy.minimum_valid_signatures,
        });
    }
    for required in &policy.required_algorithms {
        if !valid_algorithms.contains(required) {
            return Err(AttestationErrorV1::MissingRequiredAlgorithm(*required));
        }
    }

    valid_signers.sort();
    let envelope_commitment = envelope.canonical_commitment_v1()?;
    Ok(VerifiedAttestationV1 {
        subject: envelope.subject.clone(),
        envelope_commitment,
        trust_generation_commitment: snapshot.generation_commitment,
        evaluation_time_unix_s,
        natural_valid_until_unix_s,
        valid_signers,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttestationErrorV1 {
    UnsupportedProtocolVersion(u16),
    EmptyField(&'static str),
    FieldTooLong {
        field: &'static str,
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    ZeroCommitment(&'static str),
    InvalidValidityWindow {
        issued_at_unix_s: u64,
        valid_until_unix_s: u64,
    },
    NoSignatures,
    TooManySignatures {
        actual: usize,
        maximum: usize,
    },
    EmptySignature,
    SignatureTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    DuplicateSigner {
        algorithm: AttestationSignatureAlgorithmV1,
        key_id: String,
    },
    InvalidPolicy,
    InvalidKeyValidityWindow {
        key_id: String,
    },
    KeyHasNoAllowedDomains(String),
    InvalidTrustSnapshotWindow,
    EnvelopeNotYetValid,
    EnvelopeExpired,
    TrustProvider(String),
    VerificationProvider(String),
    TrustGenerationMismatch,
    TrustSnapshotNotYetValid,
    TrustSnapshotExpired,
    KeyIdTooLong {
        key_id: String,
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    KeyNotAllowed(String),
    UnknownSigner(String),
    TrustProviderKeyIdentityMismatch {
        requested_key_id: String,
        returned_key_id: String,
    },
    SignerDomainNotAllowed {
        key_id: String,
        domain: String,
    },
    SignerNotYetValid(String),
    SignerExpired(String),
    SignerRetired(String),
    SignerRevoked(String),
    InvalidSignature {
        algorithm: AttestationSignatureAlgorithmV1,
        key_id: String,
    },
    InsufficientValidSignatures {
        actual: usize,
        required: usize,
    },
    MissingRequiredAlgorithm(AttestationSignatureAlgorithmV1),
}

impl fmt::Display for AttestationErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedProtocolVersion(version) => {
                write!(f, "unsupported control-plane attestation version {version}")
            }
            Self::EmptyField(field) => write!(f, "attestation field {field} must be non-empty"),
            Self::FieldTooLong {
                field,
                actual_bytes,
                maximum_bytes,
            } => write!(
                f,
                "attestation field {field} is {actual_bytes} bytes; maximum is {maximum_bytes}"
            ),
            Self::ZeroCommitment(field) => {
                write!(f, "attestation commitment field {field} may not be all-zero")
            }
            Self::InvalidValidityWindow {
                issued_at_unix_s,
                valid_until_unix_s,
            } => write!(
                f,
                "invalid attestation validity window [{issued_at_unix_s}, {valid_until_unix_s})"
            ),
            Self::NoSignatures => write!(f, "attestation envelope contains no signatures"),
            Self::TooManySignatures { actual, maximum } => write!(
                f,
                "attestation has {actual} signatures; maximum is {maximum}"
            ),
            Self::EmptySignature => write!(f, "attestation contains an empty signature"),
            Self::SignatureTooLarge {
                actual_bytes,
                maximum_bytes,
            } => write!(
                f,
                "attestation signature is {actual_bytes} bytes; maximum is {maximum_bytes}"
            ),
            Self::DuplicateSigner { algorithm, key_id } => {
                write!(f, "duplicate attestation signer {algorithm:?}:{key_id}")
            }
            Self::InvalidPolicy => write!(f, "invalid attestation verification policy"),
            Self::InvalidKeyValidityWindow { key_id } => {
                write!(f, "invalid validity window for attestation key {key_id}")
            }
            Self::KeyHasNoAllowedDomains(key_id) => {
                write!(f, "attestation key {key_id} has no allowed subject domains")
            }
            Self::InvalidTrustSnapshotWindow => write!(f, "invalid attestation trust snapshot window"),
            Self::EnvelopeNotYetValid => write!(f, "attestation envelope is not yet valid"),
            Self::EnvelopeExpired => write!(f, "attestation envelope has expired"),
            Self::TrustProvider(error) => write!(f, "attestation trust provider failed: {error}"),
            Self::VerificationProvider(error) => {
                write!(f, "attestation signature verifier failed: {error}")
            }
            Self::TrustGenerationMismatch => write!(
                f,
                "attestation authority generation does not match current trust generation"
            ),
            Self::TrustSnapshotNotYetValid => write!(f, "attestation trust snapshot is not yet valid"),
            Self::TrustSnapshotExpired => write!(f, "attestation trust snapshot has expired"),
            Self::KeyIdTooLong {
                key_id,
                actual_bytes,
                maximum_bytes,
            } => write!(
                f,
                "attestation key id {key_id} is {actual_bytes} bytes; policy maximum is {maximum_bytes}"
            ),
            Self::KeyNotAllowed(key_id) => write!(f, "attestation key {key_id} is not allowed by policy"),
            Self::UnknownSigner(key_id) => write!(f, "unknown attestation signer {key_id}"),
            Self::TrustProviderKeyIdentityMismatch {
                requested_key_id,
                returned_key_id,
            } => write!(
                f,
                "trust provider returned key {returned_key_id} for requested key {requested_key_id}"
            ),
            Self::SignerDomainNotAllowed { key_id, domain } => write!(
                f,
                "attestation signer {key_id} is not allowed for subject domain {domain}"
            ),
            Self::SignerNotYetValid(key_id) => write!(f, "attestation signer {key_id} is not yet valid"),
            Self::SignerExpired(key_id) => write!(f, "attestation signer {key_id} has expired"),
            Self::SignerRetired(key_id) => write!(f, "attestation signer {key_id} is retired"),
            Self::SignerRevoked(key_id) => write!(f, "attestation signer {key_id} is revoked"),
            Self::InvalidSignature { algorithm, key_id } => {
                write!(f, "invalid attestation signature {algorithm:?}:{key_id}")
            }
            Self::InsufficientValidSignatures { actual, required } => write!(
                f,
                "attestation has {actual} valid signatures; policy requires {required}"
            ),
            Self::MissingRequiredAlgorithm(algorithm) => {
                write!(f, "attestation is missing required algorithm {algorithm:?}")
            }
        }
    }
}

impl Error for AttestationErrorV1 {}

fn validate_bounded_nonempty(
    field: &'static str,
    value: &str,
    maximum_bytes: usize,
) -> Result<(), AttestationErrorV1> {
    if value.trim().is_empty() {
        return Err(AttestationErrorV1::EmptyField(field));
    }
    if value.len() > maximum_bytes {
        return Err(AttestationErrorV1::FieldTooLong {
            field,
            actual_bytes: value.len(),
            maximum_bytes,
        });
    }
    Ok(())
}

fn append_bounded_utf8(
    out: &mut Vec<u8>,
    field: &'static str,
    value: &str,
    maximum_bytes: usize,
) -> Result<(), AttestationErrorV1> {
    validate_bounded_nonempty(field, value, maximum_bytes)?;
    let len = u16::try_from(value.len()).map_err(|_| AttestationErrorV1::FieldTooLong {
        field,
        actual_bytes: value.len(),
        maximum_bytes: maximum_bytes.min(u16::MAX as usize),
    })?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    const TRUST_GENERATION: [u8; 32] = [0xA0; 32];
    const SUBJECT_COMMITMENT: [u8; 32] = [0xB0; 32];
    const DOMAIN: &str = "test.control-plane.subject.v1";

    #[derive(Clone)]
    struct MockTrustProvider {
        snapshot: AttestationTrustSnapshotV1,
        keys: BTreeMap<(AttestationSignatureAlgorithmV1, String), AttestationTrustedKeyV1>,
    }

    impl MockTrustProvider {
        fn standard() -> Self {
            let mut keys = BTreeMap::new();
            keys.insert(
                (AttestationSignatureAlgorithmV1::Ed25519, "worker-a".into()),
                AttestationTrustedKeyV1 {
                    algorithm: AttestationSignatureAlgorithmV1::Ed25519,
                    key_id: "worker-a".into(),
                    valid_from_unix_s: 50,
                    valid_until_unix_s: 700,
                    lifecycle: AttestationKeyLifecycleV1::Active,
                    allowed_subject_domains: BTreeSet::from([DOMAIN.to_string()]),
                },
            );
            Self {
                snapshot: AttestationTrustSnapshotV1 {
                    generation_commitment: TRUST_GENERATION,
                    valid_from_unix_s: 20,
                    valid_until_unix_s: 800,
                },
                keys,
            }
        }
    }

    impl AttestationTrustVerifierV1 for MockTrustProvider {
        fn current_snapshot(&self) -> Result<AttestationTrustSnapshotV1, String> {
            Ok(self.snapshot.clone())
        }

        fn key_record(
            &self,
            trust_generation_commitment: [u8; 32],
            algorithm: AttestationSignatureAlgorithmV1,
            key_id: &str,
        ) -> Result<Option<AttestationTrustedKeyV1>, String> {
            if trust_generation_commitment != self.snapshot.generation_commitment {
                return Ok(None);
            }
            Ok(self.keys.get(&(algorithm, key_id.to_string())).cloned())
        }

        fn verify_signature(
            &self,
            trust_generation_commitment: [u8; 32],
            algorithm: AttestationSignatureAlgorithmV1,
            key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            let expected = mock_signature(
                trust_generation_commitment,
                algorithm,
                key_id,
                message,
            );
            Ok(signature == expected.as_slice())
        }
    }

    fn mock_signature(
        generation: [u8; 32],
        algorithm: AttestationSignatureAlgorithmV1,
        key_id: &str,
        message: &[u8],
    ) -> Vec<u8> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"test-only-attestation-signature\0");
        hasher.update(&generation);
        hasher.update(&[algorithm.canonical_tag()]);
        hasher.update(key_id.as_bytes());
        hasher.update(message);
        hasher.finalize().as_bytes().to_vec()
    }

    fn policy() -> AttestationPolicyV1 {
        AttestationPolicyV1 {
            minimum_valid_signatures: 1,
            maximum_signatures: 4,
            maximum_signature_bytes: 256,
            maximum_key_id_bytes: 128,
            required_algorithms: BTreeSet::from([
                AttestationSignatureAlgorithmV1::Ed25519,
            ]),
            allowed_key_ids: Some(BTreeSet::from(["worker-a".to_string()])),
        }
    }

    fn unsigned_envelope() -> DetachedAttestationEnvelopeV1 {
        DetachedAttestationEnvelopeV1 {
            protocol_version: CONTROL_PLANE_ATTESTATION_VERSION_V1,
            subject: AttestationSubjectV1::new(DOMAIN, SUBJECT_COMMITMENT).unwrap(),
            issued_at_unix_s: 100,
            valid_until_unix_s: 750,
            authority_generation_commitment: TRUST_GENERATION,
            signatures: Vec::new(),
        }
    }

    fn signed_envelope() -> DetachedAttestationEnvelopeV1 {
        let mut envelope = unsigned_envelope();
        let message = envelope
            .signer_message_v1(AttestationSignatureAlgorithmV1::Ed25519, "worker-a")
            .unwrap();
        envelope.signatures.push(DetachedAttestationSignatureV1 {
            algorithm: AttestationSignatureAlgorithmV1::Ed25519,
            key_id: "worker-a".into(),
            signature: mock_signature(
                TRUST_GENERATION,
                AttestationSignatureAlgorithmV1::Ed25519,
                "worker-a",
                &message,
            ),
        });
        envelope
    }

    #[test]
    fn exact_current_attestation_yields_private_positive_witness() {
        let envelope = signed_envelope();
        let verified = verify_current_attestation_v1(
            &envelope,
            &policy(),
            &MockTrustProvider::standard(),
            200,
        )
        .unwrap();

        assert_eq!(verified.subject(), &envelope.subject);
        assert_eq!(
            verified.envelope_commitment(),
            envelope.canonical_commitment_v1().unwrap()
        );
        assert_eq!(verified.trust_generation_commitment(), TRUST_GENERATION);
        assert_eq!(verified.evaluation_time_unix_s(), 200);
        assert_eq!(verified.natural_valid_until_unix_s(), 700);
        assert!(verified.remains_within_time_bounds(699));
        assert!(!verified.remains_within_time_bounds(700));
        assert_eq!(verified.valid_signers().len(), 1);
    }

    #[test]
    fn subject_domain_and_signer_identity_substitution_fail_closed() {
        let provider = MockTrustProvider::standard();

        let mut subject_changed = signed_envelope();
        subject_changed.subject.commitment[0] ^= 0x01;
        assert!(matches!(
            verify_current_attestation_v1(&subject_changed, &policy(), &provider, 200),
            Err(AttestationErrorV1::InvalidSignature { .. })
        ));

        let mut domain_changed = signed_envelope();
        domain_changed.subject.domain = "other.domain.v1".into();
        assert!(matches!(
            verify_current_attestation_v1(&domain_changed, &policy(), &provider, 200),
            Err(AttestationErrorV1::SignerDomainNotAllowed { .. })
                | Err(AttestationErrorV1::InvalidSignature { .. })
        ));

        let mut relabeled = signed_envelope();
        relabeled.signatures[0].key_id = "worker-b".into();
        let mut alias_provider = provider.clone();
        let mut alias = alias_provider
            .keys
            .get(&(AttestationSignatureAlgorithmV1::Ed25519, "worker-a".into()))
            .unwrap()
            .clone();
        alias.key_id = "worker-b".into();
        alias_provider.keys.insert(
            (AttestationSignatureAlgorithmV1::Ed25519, "worker-b".into()),
            alias,
        );
        let mut alias_policy = policy();
        alias_policy.allowed_key_ids = Some(BTreeSet::from(["worker-b".to_string()]));
        assert!(matches!(
            verify_current_attestation_v1(&relabeled, &alias_policy, &alias_provider, 200),
            Err(AttestationErrorV1::InvalidSignature { .. })
        ));
    }

    #[test]
    fn trust_generation_substitution_fails_before_signature_authority() {
        let mut envelope = signed_envelope();
        envelope.authority_generation_commitment[0] ^= 0x01;
        assert!(matches!(
            verify_current_attestation_v1(
                &envelope,
                &policy(),
                &MockTrustProvider::standard(),
                200,
            ),
            Err(AttestationErrorV1::TrustGenerationMismatch)
        ));
    }

    #[test]
    fn unknown_not_yet_expired_retired_and_revoked_keys_fail_closed() {
        let envelope = signed_envelope();

        let mut unknown = MockTrustProvider::standard();
        unknown.keys.clear();
        assert!(matches!(
            verify_current_attestation_v1(&envelope, &policy(), &unknown, 200),
            Err(AttestationErrorV1::UnknownSigner(_))
        ));

        let mut not_yet = MockTrustProvider::standard();
        not_yet
            .keys
            .get_mut(&(AttestationSignatureAlgorithmV1::Ed25519, "worker-a".into()))
            .unwrap()
            .valid_from_unix_s = 300;
        assert!(matches!(
            verify_current_attestation_v1(&envelope, &policy(), &not_yet, 200),
            Err(AttestationErrorV1::SignerNotYetValid(_))
        ));

        let mut expired = MockTrustProvider::standard();
        expired
            .keys
            .get_mut(&(AttestationSignatureAlgorithmV1::Ed25519, "worker-a".into()))
            .unwrap()
            .valid_until_unix_s = 200;
        assert!(matches!(
            verify_current_attestation_v1(&envelope, &policy(), &expired, 200),
            Err(AttestationErrorV1::SignerExpired(_))
        ));

        let mut retired = MockTrustProvider::standard();
        retired
            .keys
            .get_mut(&(AttestationSignatureAlgorithmV1::Ed25519, "worker-a".into()))
            .unwrap()
            .lifecycle = AttestationKeyLifecycleV1::Retired;
        assert!(matches!(
            verify_current_attestation_v1(&envelope, &policy(), &retired, 200),
            Err(AttestationErrorV1::SignerRetired(_))
        ));

        let mut revoked = MockTrustProvider::standard();
        revoked
            .keys
            .get_mut(&(AttestationSignatureAlgorithmV1::Ed25519, "worker-a".into()))
            .unwrap()
            .lifecycle = AttestationKeyLifecycleV1::Revoked;
        assert!(matches!(
            verify_current_attestation_v1(&envelope, &policy(), &revoked, 200),
            Err(AttestationErrorV1::SignerRevoked(_))
        ));
    }

    #[test]
    fn envelope_and_trust_windows_are_half_open_and_fail_closed() {
        let provider = MockTrustProvider::standard();
        let envelope = signed_envelope();

        assert!(matches!(
            verify_current_attestation_v1(&envelope, &policy(), &provider, 99),
            Err(AttestationErrorV1::EnvelopeNotYetValid)
        ));
        assert!(matches!(
            verify_current_attestation_v1(&envelope, &policy(), &provider, 750),
            Err(AttestationErrorV1::EnvelopeExpired)
        ));

        let mut stale = provider.clone();
        stale.snapshot.valid_until_unix_s = 200;
        assert!(matches!(
            verify_current_attestation_v1(&envelope, &policy(), &stale, 200),
            Err(AttestationErrorV1::TrustSnapshotExpired)
        ));
    }

    #[test]
    fn policy_algorithm_and_identity_requirements_are_authority_bearing() {
        let provider = MockTrustProvider::standard();
        let envelope = signed_envelope();

        let mut wrong_algorithm = policy();
        wrong_algorithm.required_algorithms = BTreeSet::from([
            AttestationSignatureAlgorithmV1::MlDsa65,
        ]);
        assert!(matches!(
            verify_current_attestation_v1(&envelope, &wrong_algorithm, &provider, 200),
            Err(AttestationErrorV1::MissingRequiredAlgorithm(
                AttestationSignatureAlgorithmV1::MlDsa65
            ))
        ));

        let mut wrong_key = policy();
        wrong_key.allowed_key_ids = Some(BTreeSet::from(["worker-b".to_string()]));
        assert!(matches!(
            verify_current_attestation_v1(&envelope, &wrong_key, &provider, 200),
            Err(AttestationErrorV1::KeyNotAllowed(_))
        ));
    }

    #[test]
    fn duplicate_signer_identity_is_rejected_before_trust_evaluation() {
        let mut envelope = signed_envelope();
        envelope.signatures.push(envelope.signatures[0].clone());
        assert!(matches!(
            envelope.validate(),
            Err(AttestationErrorV1::DuplicateSigner { .. })
        ));
    }

    #[test]
    fn envelope_commitment_is_signature_order_independent() {
        let mut first = unsigned_envelope();
        let ed_message = first
            .signer_message_v1(AttestationSignatureAlgorithmV1::Ed25519, "worker-a")
            .unwrap();
        let pq_message = first
            .signer_message_v1(AttestationSignatureAlgorithmV1::MlDsa65, "worker-b")
            .unwrap();
        first.signatures = vec![
            DetachedAttestationSignatureV1 {
                algorithm: AttestationSignatureAlgorithmV1::Ed25519,
                key_id: "worker-a".into(),
                signature: mock_signature(
                    TRUST_GENERATION,
                    AttestationSignatureAlgorithmV1::Ed25519,
                    "worker-a",
                    &ed_message,
                ),
            },
            DetachedAttestationSignatureV1 {
                algorithm: AttestationSignatureAlgorithmV1::MlDsa65,
                key_id: "worker-b".into(),
                signature: mock_signature(
                    TRUST_GENERATION,
                    AttestationSignatureAlgorithmV1::MlDsa65,
                    "worker-b",
                    &pq_message,
                ),
            },
        ];
        let mut second = first.clone();
        second.signatures.reverse();
        assert_eq!(
            first.canonical_commitment_v1().unwrap(),
            second.canonical_commitment_v1().unwrap()
        );
    }
}
