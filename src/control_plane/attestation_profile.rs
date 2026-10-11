// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Exact cryptographic profile identities for control-plane attestations.
//!
//! The compact `AttestationSignatureAlgorithmV1` tags from `attestation` are not
//! free-form provider hints. This module freezes one exact verification profile
//! for each tag and provides the consequential-use wrapper that refuses providers
//! which do not explicitly implement that exact profile.
//!
//! ```text
//! algorithm tag
//!     = exact scheme
//!     + exact wire size
//!     + exact pure/prehash mode
//!     + exact context policy
//! ```
//!
//! In particular, `MlDsa65` means final FIPS 204 ML-DSA-65, pure mode, empty
//! context, with a 3309-byte signature. Legacy Dilithium3/draft-FIPS encodings
//! are not compatible with that tag.

use super::attestation::{
    AttestationErrorV1, AttestationPolicyV1, AttestationSignatureAlgorithmV1,
    AttestationTrustVerifierV1, DetachedAttestationEnvelopeV1, VerifiedAttestationV1,
};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// Exact message-processing semantics selected by one attestation profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttestationMessageModeV1 {
    /// RFC 8032 Ed25519, not Ed25519ctx and not Ed25519ph.
    Ed25519Rfc8032Pure,
    /// FIPS 204 ML-DSA, pure mode, with the external context fixed to empty.
    MlDsaFips204PureEmptyContext,
}

/// Frozen cryptographic contract corresponding to one compact attestation tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttestationCryptoProfileV1 {
    pub algorithm: AttestationSignatureAlgorithmV1,
    pub canonical_name: &'static str,
    pub public_key_bytes: usize,
    pub signature_bytes: usize,
    pub message_mode: AttestationMessageModeV1,
}

impl AttestationSignatureAlgorithmV1 {
    /// Resolve the compact protocol tag to its exact v1 cryptographic profile.
    ///
    /// This is normative for consequential control-plane use. A provider must not
    /// reinterpret the enum according to a library's historical naming.
    pub const fn exact_profile_v1(self) -> AttestationCryptoProfileV1 {
        match self {
            Self::Ed25519 => AttestationCryptoProfileV1 {
                algorithm: self,
                canonical_name: "ed25519-rfc8032-pure-v1",
                public_key_bytes: 32,
                signature_bytes: 64,
                message_mode: AttestationMessageModeV1::Ed25519Rfc8032Pure,
            },
            Self::MlDsa65 => AttestationCryptoProfileV1 {
                algorithm: self,
                canonical_name: "ml-dsa-65-fips204-pure-empty-context-v1",
                public_key_bytes: 1952,
                signature_bytes: 3309,
                message_mode: AttestationMessageModeV1::MlDsaFips204PureEmptyContext,
            },
            Self::MlDsa87 => AttestationCryptoProfileV1 {
                algorithm: self,
                canonical_name: "ml-dsa-87-fips204-pure-empty-context-v1",
                public_key_bytes: 2592,
                signature_bytes: 4627,
                message_mode: AttestationMessageModeV1::MlDsaFips204PureEmptyContext,
            },
        }
    }
}

/// Provider boundary for consequential use of an exact cryptographic profile.
///
/// Implementations must positively opt into the frozen profile rather than merely
/// accepting a short algorithm name. This does not replace implementation-specific
/// known-answer/vector qualification; it prevents accidental use of a provider
/// which only promises a looser or legacy interpretation.
pub trait ExactAttestationTrustVerifierV1: AttestationTrustVerifierV1 {
    fn supports_exact_profile_v1(
        &self,
        profile: AttestationCryptoProfileV1,
    ) -> Result<bool, String>;
}

/// Enforce exact wire/profile semantics on every signature in one envelope.
pub fn validate_exact_attestation_profiles_v1(
    envelope: &DetachedAttestationEnvelopeV1,
) -> Result<(), AttestationProfileErrorV1> {
    envelope
        .validate()
        .map_err(AttestationProfileErrorV1::Attestation)?;
    for signature in &envelope.signatures {
        let profile = signature.algorithm.exact_profile_v1();
        if signature.signature.len() != profile.signature_bytes {
            return Err(AttestationProfileErrorV1::SignatureLengthMismatch {
                algorithm: signature.algorithm,
                profile_name: profile.canonical_name,
                actual_bytes: signature.signature.len(),
                expected_bytes: profile.signature_bytes,
            });
        }
    }
    Ok(())
}

/// Verify one attestation only through exact v1 cryptographic profiles.
///
/// This wrapper is the required path for consequential consumers. It first freezes
/// wire shape, then requires the provider to explicitly support every exact profile
/// used by the envelope, then delegates lifecycle/currentness verification to the
/// base attestation kernel. The result is a distinct non-serializable capability
/// type so downstream code cannot confuse base verification with exact-profile
/// verification.
pub fn verify_current_attestation_exact_profile_v1(
    envelope: &DetachedAttestationEnvelopeV1,
    policy: &AttestationPolicyV1,
    provider: &dyn ExactAttestationTrustVerifierV1,
    evaluation_time_unix_s: u64,
) -> Result<ExactProfileVerifiedAttestationV1, AttestationProfileErrorV1> {
    validate_exact_attestation_profiles_v1(envelope)?;

    let mut used = BTreeSet::new();
    for signature in &envelope.signatures {
        if !used.insert(signature.algorithm) {
            continue;
        }
        let profile = signature.algorithm.exact_profile_v1();
        let supported = provider
            .supports_exact_profile_v1(profile)
            .map_err(AttestationProfileErrorV1::ProviderProfileCheck)?;
        if !supported {
            return Err(AttestationProfileErrorV1::ProviderProfileUnsupported {
                algorithm: signature.algorithm,
                profile_name: profile.canonical_name,
            });
        }
    }

    let verified = super::attestation::verify_current_attestation_v1(
        envelope,
        policy,
        provider,
        evaluation_time_unix_s,
    )
    .map_err(AttestationProfileErrorV1::Attestation)?;

    Ok(ExactProfileVerifiedAttestationV1 {
        verified,
        exact_profiles: used
            .into_iter()
            .map(AttestationSignatureAlgorithmV1::exact_profile_v1)
            .collect(),
    })
}

/// Positive witness that both lifecycle/currentness and exact profile checks pass.
///
/// This type intentionally has private fields, no public constructor, and no
/// serde derives. Returning a distinct type prevents downstream code from
/// accidentally inheriting exact-profile authority from a witness created
/// through the looser base verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactProfileVerifiedAttestationV1 {
    verified: VerifiedAttestationV1,
    exact_profiles: Vec<AttestationCryptoProfileV1>,
}

impl ExactProfileVerifiedAttestationV1 {
    /// Underlying lifecycle/currentness witness.
    pub fn attestation(&self) -> &VerifiedAttestationV1 {
        &self.verified
    }

    /// Exact cryptographic profiles accepted in this verification transaction.
    pub fn exact_profiles(&self) -> &[AttestationCryptoProfileV1] {
        &self.exact_profiles
    }

    pub fn subject(&self) -> &super::attestation::AttestationSubjectV1 {
        self.verified.subject()
    }

    pub fn envelope_commitment(&self) -> [u8; 32] {
        self.verified.envelope_commitment()
    }

    pub fn trust_generation_commitment(&self) -> [u8; 32] {
        self.verified.trust_generation_commitment()
    }

    pub fn evaluation_time_unix_s(&self) -> u64 {
        self.verified.evaluation_time_unix_s()
    }

    pub fn natural_valid_until_unix_s(&self) -> u64 {
        self.verified.natural_valid_until_unix_s()
    }

    /// Time-only check inherited from the underlying lifecycle witness.
    pub fn remains_within_time_bounds(&self, at_unix_s: u64) -> bool {
        self.verified.remains_within_time_bounds(at_unix_s)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttestationProfileErrorV1 {
    Attestation(AttestationErrorV1),
    SignatureLengthMismatch {
        algorithm: AttestationSignatureAlgorithmV1,
        profile_name: &'static str,
        actual_bytes: usize,
        expected_bytes: usize,
    },
    ProviderProfileCheck(String),
    ProviderProfileUnsupported {
        algorithm: AttestationSignatureAlgorithmV1,
        profile_name: &'static str,
    },
}

impl fmt::Display for AttestationProfileErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Attestation(error) => write!(f, "{error}"),
            Self::SignatureLengthMismatch {
                algorithm,
                profile_name,
                actual_bytes,
                expected_bytes,
            } => write!(
                f,
                "attestation signature {algorithm:?} uses profile {profile_name} and must be exactly {expected_bytes} bytes; got {actual_bytes}"
            ),
            Self::ProviderProfileCheck(error) => {
                write!(f, "attestation provider profile check failed: {error}")
            }
            Self::ProviderProfileUnsupported {
                algorithm,
                profile_name,
            } => write!(
                f,
                "attestation provider does not support exact profile {profile_name} for {algorithm:?}"
            ),
        }
    }
}

impl Error for AttestationProfileErrorV1 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Attestation(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control_plane::attestation::{
        AttestationKeyLifecycleV1, AttestationSubjectV1, AttestationTrustSnapshotV1,
        AttestationTrustedKeyV1, DetachedAttestationSignatureV1,
        CONTROL_PLANE_ATTESTATION_VERSION_V1,
    };
    use std::collections::{BTreeMap, BTreeSet};

    const GENERATION: [u8; 32] = [0x31; 32];
    const SUBJECT: [u8; 32] = [0x41; 32];
    const DOMAIN: &str = "test.attestation-profile.v1";

    #[derive(Clone)]
    struct ProfileProvider {
        snapshot: AttestationTrustSnapshotV1,
        keys: BTreeMap<(AttestationSignatureAlgorithmV1, String), AttestationTrustedKeyV1>,
        support_profiles: bool,
    }

    impl ProfileProvider {
        fn ed25519() -> Self {
            let mut keys = BTreeMap::new();
            keys.insert(
                (AttestationSignatureAlgorithmV1::Ed25519, "worker-a".into()),
                AttestationTrustedKeyV1 {
                    algorithm: AttestationSignatureAlgorithmV1::Ed25519,
                    key_id: "worker-a".into(),
                    valid_from_unix_s: 10,
                    valid_until_unix_s: 900,
                    lifecycle: AttestationKeyLifecycleV1::Active,
                    allowed_subject_domains: BTreeSet::from([DOMAIN.to_string()]),
                },
            );
            Self {
                snapshot: AttestationTrustSnapshotV1 {
                    generation_commitment: GENERATION,
                    valid_from_unix_s: 1,
                    valid_until_unix_s: 1_000,
                },
                keys,
                support_profiles: true,
            }
        }
    }

    impl AttestationTrustVerifierV1 for ProfileProvider {
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
            if trust_generation_commitment != self.snapshot.generation_commitment {
                return Ok(false);
            }
            Ok(signature == pseudo_signature(algorithm, key_id, message))
        }
    }

    impl ExactAttestationTrustVerifierV1 for ProfileProvider {
        fn supports_exact_profile_v1(
            &self,
            _profile: AttestationCryptoProfileV1,
        ) -> Result<bool, String> {
            Ok(self.support_profiles)
        }
    }

    fn pseudo_signature(
        algorithm: AttestationSignatureAlgorithmV1,
        key_id: &str,
        message: &[u8],
    ) -> Vec<u8> {
        let profile = algorithm.exact_profile_v1();
        let mut out = Vec::with_capacity(profile.signature_bytes);
        let mut counter = 0u32;
        while out.len() < profile.signature_bytes {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"test-only-exact-profile-signature\0");
            hasher.update(&[algorithm.canonical_tag()]);
            hasher.update(key_id.as_bytes());
            hasher.update(&counter.to_be_bytes());
            hasher.update(message);
            out.extend_from_slice(hasher.finalize().as_bytes());
            counter = counter.wrapping_add(1);
        }
        out.truncate(profile.signature_bytes);
        out
    }

    fn envelope(algorithm: AttestationSignatureAlgorithmV1) -> DetachedAttestationEnvelopeV1 {
        let mut envelope = DetachedAttestationEnvelopeV1 {
            protocol_version: CONTROL_PLANE_ATTESTATION_VERSION_V1,
            subject: AttestationSubjectV1::new(DOMAIN, SUBJECT).unwrap(),
            issued_at_unix_s: 100,
            valid_until_unix_s: 800,
            authority_generation_commitment: GENERATION,
            signatures: Vec::new(),
        };
        let message = envelope.signer_message_v1(algorithm, "worker-a").unwrap();
        envelope.signatures.push(DetachedAttestationSignatureV1 {
            algorithm,
            key_id: "worker-a".into(),
            signature: pseudo_signature(algorithm, "worker-a", &message),
        });
        envelope
    }

    fn policy() -> AttestationPolicyV1 {
        AttestationPolicyV1 {
            minimum_valid_signatures: 1,
            maximum_signatures: 4,
            maximum_signature_bytes: 8 * 1024,
            maximum_key_id_bytes: 128,
            required_algorithms: BTreeSet::from([AttestationSignatureAlgorithmV1::Ed25519]),
            allowed_key_ids: Some(BTreeSet::from(["worker-a".to_string()])),
        }
    }

    #[test]
    fn exact_profiles_freeze_scheme_wire_and_message_modes() {
        let ed = AttestationSignatureAlgorithmV1::Ed25519.exact_profile_v1();
        assert_eq!(ed.canonical_name, "ed25519-rfc8032-pure-v1");
        assert_eq!(ed.public_key_bytes, 32);
        assert_eq!(ed.signature_bytes, 64);
        assert_eq!(
            ed.message_mode,
            AttestationMessageModeV1::Ed25519Rfc8032Pure
        );

        let ml65 = AttestationSignatureAlgorithmV1::MlDsa65.exact_profile_v1();
        assert_eq!(
            ml65.canonical_name,
            "ml-dsa-65-fips204-pure-empty-context-v1"
        );
        assert_eq!(ml65.public_key_bytes, 1952);
        assert_eq!(ml65.signature_bytes, 3309);
        assert_eq!(
            ml65.message_mode,
            AttestationMessageModeV1::MlDsaFips204PureEmptyContext
        );

        let ml87 = AttestationSignatureAlgorithmV1::MlDsa87.exact_profile_v1();
        assert_eq!(
            ml87.canonical_name,
            "ml-dsa-87-fips204-pure-empty-context-v1"
        );
        assert_eq!(ml87.public_key_bytes, 2592);
        assert_eq!(ml87.signature_bytes, 4627);
        assert_eq!(
            ml87.message_mode,
            AttestationMessageModeV1::MlDsaFips204PureEmptyContext
        );
    }

    #[test]
    fn legacy_dilithium_draft_lengths_cannot_satisfy_fips_profiles() {
        let mut ml65 = envelope(AttestationSignatureAlgorithmV1::MlDsa65);
        ml65.signatures[0].signature.resize(3293, 0);
        assert!(matches!(
            validate_exact_attestation_profiles_v1(&ml65),
            Err(AttestationProfileErrorV1::SignatureLengthMismatch {
                expected_bytes: 3309,
                actual_bytes: 3293,
                ..
            })
        ));

        let mut ml87 = envelope(AttestationSignatureAlgorithmV1::MlDsa87);
        ml87.signatures[0].signature.resize(4595, 0);
        assert!(matches!(
            validate_exact_attestation_profiles_v1(&ml87),
            Err(AttestationProfileErrorV1::SignatureLengthMismatch {
                expected_bytes: 4627,
                actual_bytes: 4595,
                ..
            })
        ));
    }

    #[test]
    fn ed25519_wire_size_is_exact_not_merely_bounded() {
        let mut ed = envelope(AttestationSignatureAlgorithmV1::Ed25519);
        ed.signatures[0].signature.pop();
        assert!(matches!(
            validate_exact_attestation_profiles_v1(&ed),
            Err(AttestationProfileErrorV1::SignatureLengthMismatch {
                expected_bytes: 64,
                actual_bytes: 63,
                ..
            })
        ));
    }

    #[test]
    fn provider_must_explicitly_opt_into_exact_profile() {
        let envelope = envelope(AttestationSignatureAlgorithmV1::Ed25519);
        let mut provider = ProfileProvider::ed25519();
        provider.support_profiles = false;
        assert!(matches!(
            verify_current_attestation_exact_profile_v1(&envelope, &policy(), &provider, 200),
            Err(AttestationProfileErrorV1::ProviderProfileUnsupported { .. })
        ));
    }

    #[test]
    fn exact_profile_wrapper_preserves_base_lifecycle_verification() {
        let envelope = envelope(AttestationSignatureAlgorithmV1::Ed25519);
        let verified = verify_current_attestation_exact_profile_v1(
            &envelope,
            &policy(),
            &ProfileProvider::ed25519(),
            200,
        )
        .unwrap();
        assert_eq!(verified.subject(), &envelope.subject);
        assert_eq!(verified.attestation().valid_signers().len(), 1);
        assert_eq!(
            verified.exact_profiles(),
            &[AttestationSignatureAlgorithmV1::Ed25519.exact_profile_v1()]
        );
    }
}
