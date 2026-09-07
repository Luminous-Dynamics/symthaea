// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cryptographically bound exact-profile attestation verification.
//!
//! `attestation_profile` freezes exact scheme/wire/message semantics and requires
//! providers to declare support for them. This module closes the next interface
//! gap: the exact profile object itself is passed into the cryptographic
//! verification call, rather than being checked separately and then erased back
//! to a compact algorithm tag.
//!
//! ```text
//! provider says exact profile is supported
//!     !=
//! cryptographic verification call is bound to that profile
//! ```
//!
//! The positive witness here is therefore a stronger in-process capability than
//! either `VerifiedAttestationV1` or `ExactProfileVerifiedAttestationV1`. It is
//! still evidence about attestation verification, not domain/effect authority.

use super::attestation::{
    AttestationErrorV1, AttestationPolicyV1, AttestationSignatureAlgorithmV1,
    AttestationTrustSnapshotV1, AttestationTrustVerifierV1, AttestationTrustedKeyV1,
    DetachedAttestationEnvelopeV1, VerifiedAttestationV1, verify_current_attestation_v1,
};
use super::attestation_profile::{
    AttestationCryptoProfileV1, AttestationProfileErrorV1, ExactAttestationTrustVerifierV1,
    validate_exact_attestation_profiles_v1,
};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// Provider boundary whose cryptographic verification call is explicitly bound
/// to the exact frozen v1 profile.
///
/// Implementations must verify according to `profile` itself. They must not
/// reinterpret only `profile.algorithm` through legacy library naming.
pub trait ProfileBoundAttestationTrustVerifierV1: ExactAttestationTrustVerifierV1 {
    fn verify_signature_exact_profile_v1(
        &self,
        trust_generation_commitment: [u8; 32],
        profile: AttestationCryptoProfileV1,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

/// Private adapter that lets the lifecycle kernel reuse all of its currentness,
/// domain, key-lifecycle, validity-window, and policy checks while routing the
/// actual signature operation through the exact-profile method above.
struct ProfileBoundVerifierAdapter<'a, P: ?Sized> {
    provider: &'a P,
}

impl<P> AttestationTrustVerifierV1 for ProfileBoundVerifierAdapter<'_, P>
where
    P: ProfileBoundAttestationTrustVerifierV1 + ?Sized,
{
    fn current_snapshot(&self) -> Result<AttestationTrustSnapshotV1, String> {
        self.provider.current_snapshot()
    }

    fn key_record(
        &self,
        trust_generation_commitment: [u8; 32],
        algorithm: AttestationSignatureAlgorithmV1,
        key_id: &str,
    ) -> Result<Option<AttestationTrustedKeyV1>, String> {
        self.provider
            .key_record(trust_generation_commitment, algorithm, key_id)
    }

    fn verify_signature(
        &self,
        trust_generation_commitment: [u8; 32],
        algorithm: AttestationSignatureAlgorithmV1,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String> {
        let profile = algorithm.exact_profile_v1();
        let supported = self.provider.supports_exact_profile_v1(profile)?;
        if !supported {
            return Ok(false);
        }
        self.provider.verify_signature_exact_profile_v1(
            trust_generation_commitment,
            profile,
            key_id,
            message,
            signature,
        )
    }
}

/// Strong positive witness that exact wire/profile checks and the cryptographic
/// operation itself were performed under the same exact profile contract.
///
/// Private fields, no public constructor, and no serde derives prevent this from
/// becoming a portable self-asserted proof token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CryptographicallyBoundExactProfileAttestationV1 {
    verified: VerifiedAttestationV1,
    exact_profiles: Vec<AttestationCryptoProfileV1>,
}

impl CryptographicallyBoundExactProfileAttestationV1 {
    /// Underlying lifecycle/currentness witness.
    pub fn attestation(&self) -> &VerifiedAttestationV1 {
        &self.verified
    }

    /// Exact profiles whose wire shape and cryptographic verification call were
    /// both enforced in this verification transaction.
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

    /// Time-only check inherited from the lifecycle witness. Current trust
    /// generation/revocation still requires point-of-use revalidation.
    pub fn remains_within_time_bounds(&self, at_unix_s: u64) -> bool {
        self.verified.remains_within_time_bounds(at_unix_s)
    }
}

/// Verify one attestation with exact profile semantics bound directly into the
/// cryptographic verification operation.
pub fn verify_current_attestation_crypto_bound_profile_v1<P>(
    envelope: &DetachedAttestationEnvelopeV1,
    policy: &AttestationPolicyV1,
    provider: &P,
    evaluation_time_unix_s: u64,
) -> Result<CryptographicallyBoundExactProfileAttestationV1, AttestationCryptoProfileErrorV1>
where
    P: ProfileBoundAttestationTrustVerifierV1 + ?Sized,
{
    validate_exact_attestation_profiles_v1(envelope)
        .map_err(AttestationCryptoProfileErrorV1::Profile)?;

    let mut used_algorithms = BTreeSet::new();
    for signature in &envelope.signatures {
        if !used_algorithms.insert(signature.algorithm) {
            continue;
        }
        let profile = signature.algorithm.exact_profile_v1();
        let supported = provider
            .supports_exact_profile_v1(profile)
            .map_err(AttestationCryptoProfileErrorV1::Provider)?;
        if !supported {
            return Err(AttestationCryptoProfileErrorV1::UnsupportedProfile {
                algorithm: signature.algorithm,
                profile_name: profile.canonical_name,
            });
        }
    }

    let adapter = ProfileBoundVerifierAdapter { provider };
    let verified = verify_current_attestation_v1(
        envelope,
        policy,
        &adapter,
        evaluation_time_unix_s,
    )
    .map_err(AttestationCryptoProfileErrorV1::Attestation)?;

    Ok(CryptographicallyBoundExactProfileAttestationV1 {
        verified,
        exact_profiles: used_algorithms
            .into_iter()
            .map(AttestationSignatureAlgorithmV1::exact_profile_v1)
            .collect(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttestationCryptoProfileErrorV1 {
    Profile(AttestationProfileErrorV1),
    Attestation(AttestationErrorV1),
    Provider(String),
    UnsupportedProfile {
        algorithm: AttestationSignatureAlgorithmV1,
        profile_name: &'static str,
    },
}

impl fmt::Display for AttestationCryptoProfileErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Profile(error) => write!(f, "{error}"),
            Self::Attestation(error) => write!(f, "{error}"),
            Self::Provider(error) => write!(f, "attestation exact-profile provider failed: {error}"),
            Self::UnsupportedProfile {
                algorithm,
                profile_name,
            } => write!(
                f,
                "attestation provider does not support exact profile {profile_name} for {algorithm:?}"
            ),
        }
    }
}

impl Error for AttestationCryptoProfileErrorV1 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Profile(error) => Some(error),
            Self::Attestation(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control_plane::attestation::{
        AttestationKeyLifecycleV1, AttestationSubjectV1, DetachedAttestationSignatureV1,
        CONTROL_PLANE_ATTESTATION_VERSION_V1,
    };
    use crate::control_plane::attestation_profile::verify_current_attestation_exact_profile_v1;
    use std::collections::{BTreeMap, BTreeSet};

    const GENERATION: [u8; 32] = [0x81; 32];
    const SUBJECT: [u8; 32] = [0x91; 32];
    const DOMAIN: &str = "test.attestation-crypto-profile.v1";

    struct Provider {
        snapshot: AttestationTrustSnapshotV1,
        keys: BTreeMap<(AttestationSignatureAlgorithmV1, String), AttestationTrustedKeyV1>,
        exact_calls: std::cell::Cell<usize>,
    }

    impl Provider {
        fn new() -> Self {
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
                exact_calls: std::cell::Cell::new(0),
            }
        }
    }

    impl AttestationTrustVerifierV1 for Provider {
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
            _trust_generation_commitment: [u8; 32],
            _algorithm: AttestationSignatureAlgorithmV1,
            _key_id: &str,
            _message: &[u8],
            _signature: &[u8],
        ) -> Result<bool, String> {
            // Intentionally fail. A consequential exact-profile path must not
            // delegate the cryptographic operation through this erased method.
            Ok(false)
        }
    }

    impl ExactAttestationTrustVerifierV1 for Provider {
        fn supports_exact_profile_v1(
            &self,
            profile: AttestationCryptoProfileV1,
        ) -> Result<bool, String> {
            Ok(profile == AttestationSignatureAlgorithmV1::Ed25519.exact_profile_v1())
        }
    }

    impl ProfileBoundAttestationTrustVerifierV1 for Provider {
        fn verify_signature_exact_profile_v1(
            &self,
            trust_generation_commitment: [u8; 32],
            profile: AttestationCryptoProfileV1,
            key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            self.exact_calls.set(self.exact_calls.get() + 1);
            if trust_generation_commitment != self.snapshot.generation_commitment
                || profile != AttestationSignatureAlgorithmV1::Ed25519.exact_profile_v1()
            {
                return Ok(false);
            }
            Ok(signature == pseudo_signature(profile, key_id, message))
        }
    }

    fn pseudo_signature(
        profile: AttestationCryptoProfileV1,
        key_id: &str,
        message: &[u8],
    ) -> Vec<u8> {
        let mut out = Vec::with_capacity(profile.signature_bytes);
        let mut counter = 0u32;
        while out.len() < profile.signature_bytes {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"test-only-crypto-bound-profile-signature\0");
            hasher.update(&[profile.algorithm.canonical_tag()]);
            hasher.update(profile.canonical_name.as_bytes());
            hasher.update(key_id.as_bytes());
            hasher.update(&counter.to_be_bytes());
            hasher.update(message);
            out.extend_from_slice(hasher.finalize().as_bytes());
            counter = counter.wrapping_add(1);
        }
        out.truncate(profile.signature_bytes);
        out
    }

    fn envelope() -> DetachedAttestationEnvelopeV1 {
        let algorithm = AttestationSignatureAlgorithmV1::Ed25519;
        let profile = algorithm.exact_profile_v1();
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
            signature: pseudo_signature(profile, "worker-a", &message),
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
    fn crypto_bound_path_uses_exact_profile_operation_not_erased_base_method() {
        let provider = Provider::new();
        let envelope = envelope();

        // The older exact-profile wrapper still delegates the actual crypto call
        // through the base erased method, which this provider intentionally denies.
        assert!(
            verify_current_attestation_exact_profile_v1(&envelope, &policy(), &provider, 200)
                .is_err()
        );

        let verified = verify_current_attestation_crypto_bound_profile_v1(
            &envelope,
            &policy(),
            &provider,
            200,
        )
        .unwrap();

        assert_eq!(provider.exact_calls.get(), 1);
        assert_eq!(verified.subject(), &envelope.subject);
        assert_eq!(verified.attestation().valid_signers().len(), 1);
        assert_eq!(
            verified.exact_profiles(),
            &[AttestationSignatureAlgorithmV1::Ed25519.exact_profile_v1()]
        );
    }

    #[test]
    fn wire_shape_still_fails_before_crypto_provider_call() {
        let provider = Provider::new();
        let mut envelope = envelope();
        envelope.signatures[0].signature.push(0);

        let result = verify_current_attestation_crypto_bound_profile_v1(
            &envelope,
            &policy(),
            &provider,
            200,
        );
        assert!(result.is_err());
        assert_eq!(provider.exact_calls.get(), 0);
    }
}
