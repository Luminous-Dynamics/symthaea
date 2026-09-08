use super::*;
use crate::control_plane::attestation::{
    AttestationKeyLifecycleV1, AttestationPolicyV1, AttestationSignatureAlgorithmV1,
    AttestationSubjectV1, AttestationTrustSnapshotV1, AttestationTrustVerifierV1,
    AttestationTrustedKeyV1, DetachedAttestationEnvelopeV1, DetachedAttestationSignatureV1,
    CONTROL_PLANE_ATTESTATION_VERSION_V1,
};
use crate::control_plane::attestation_crypto_profile::{
    ProfileBoundAttestationTrustVerifierV1, verify_current_attestation_crypto_bound_profile_v1,
};
use crate::control_plane::attestation_identity_namespace::{
    AttestationIdentityNamespaceSetV1, AttestationIdentityNamespaceV1,
};
use crate::control_plane::attestation_profile::{
    AttestationCryptoProfileV1, ExactAttestationTrustVerifierV1,
};
use crate::control_plane::attestation_qualified_authority::{
    QualifiedAttestationAuthorityResolverV1, QualifiedAttestationAuthoritySnapshotV1,
    QualifiedAttestationDiversityPolicyV1, QualifiedAttestationSignerBindingV1,
    qualify_attestation_authority_diversity_v1,
};
use std::collections::BTreeSet;

const TRUST_GENERATION: [u8; 32] = [0x11; 32];
const BINDING_GENERATION: [u8; 32] = [0x22; 32];
const DOMAIN: &str = "test.attestation-currentness.v1";
const SUBJECT: [u8; 32] = [0x33; 32];
const ORIGINAL_TIME: u64 = 200;
const REVALIDATION_TIME: u64 = 250;

#[derive(Clone)]
struct Provider {
    snapshot: AttestationTrustSnapshotV1,
    key: AttestationTrustedKeyV1,
    exact_profile_supported: bool,
}

impl Provider {
    fn new() -> Self {
        Self {
            snapshot: AttestationTrustSnapshotV1 {
                generation_commitment: TRUST_GENERATION,
                valid_from_unix_s: 1,
                valid_until_unix_s: 1_000,
            },
            key: AttestationTrustedKeyV1 {
                algorithm: AttestationSignatureAlgorithmV1::Ed25519,
                key_id: "worker-a".into(),
                valid_from_unix_s: 10,
                valid_until_unix_s: 900,
                lifecycle: AttestationKeyLifecycleV1::Active,
                allowed_subject_domains: BTreeSet::from([DOMAIN.to_string()]),
            },
            exact_profile_supported: true,
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
        if trust_generation_commitment == self.snapshot.generation_commitment
            && algorithm == self.key.algorithm
            && key_id == self.key.key_id
        {
            Ok(Some(self.key.clone()))
        } else {
            Ok(None)
        }
    }

    fn verify_signature(
        &self,
        _trust_generation_commitment: [u8; 32],
        _algorithm: AttestationSignatureAlgorithmV1,
        _key_id: &str,
        _message: &[u8],
        _signature: &[u8],
    ) -> Result<bool, String> {
        Ok(false)
    }
}

impl ExactAttestationTrustVerifierV1 for Provider {
    fn supports_exact_profile_v1(
        &self,
        profile: AttestationCryptoProfileV1,
    ) -> Result<bool, String> {
        Ok(self.exact_profile_supported
            && profile == AttestationSignatureAlgorithmV1::Ed25519.exact_profile_v1())
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
        if trust_generation_commitment != self.snapshot.generation_commitment
            || !self.exact_profile_supported
            || profile != AttestationSignatureAlgorithmV1::Ed25519.exact_profile_v1()
            || key_id != self.key.key_id
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
        hasher.update(b"test-only-attestation-currentness-signature\0");
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

#[derive(Clone)]
struct Resolver {
    snapshot: QualifiedAttestationAuthoritySnapshotV1,
    binding: QualifiedAttestationSignerBindingV1,
}

impl Resolver {
    fn new() -> Self {
        let namespaces = AttestationIdentityNamespaceSetV1::new(
            AttestationIdentityNamespaceV1::new("principals-v1", [0x44; 32]).unwrap(),
            Some(AttestationIdentityNamespaceV1::new("authorities-v1", [0x55; 32]).unwrap()),
        )
        .unwrap();
        Self {
            snapshot: QualifiedAttestationAuthoritySnapshotV1 {
                generation_commitment: BINDING_GENERATION,
                valid_from_unix_s: 1,
                valid_until_unix_s: 850,
                identity_namespaces: namespaces,
            },
            binding: QualifiedAttestationSignerBindingV1 {
                generation_commitment: BINDING_GENERATION,
                algorithm: AttestationSignatureAlgorithmV1::Ed25519,
                key_id: "worker-a".into(),
                principal_id: "alice".into(),
                authority_id: Some("org-a".into()),
                valid_from_unix_s: 10,
                valid_until_unix_s: 700,
            },
        }
    }
}

impl QualifiedAttestationAuthorityResolverV1 for Resolver {
    fn current_snapshot(&self) -> Result<QualifiedAttestationAuthoritySnapshotV1, String> {
        Ok(self.snapshot.clone())
    }

    fn signer_binding(
        &self,
        generation_commitment: [u8; 32],
        algorithm: AttestationSignatureAlgorithmV1,
        key_id: &str,
    ) -> Result<Option<QualifiedAttestationSignerBindingV1>, String> {
        if generation_commitment == self.snapshot.generation_commitment
            && algorithm == self.binding.algorithm
            && key_id == self.binding.key_id
        {
            Ok(Some(self.binding.clone()))
        } else {
            Ok(None)
        }
    }
}

fn qualified(provider: &Provider, resolver: &Resolver) -> QualifiedAttestationAuthorityDiversityV1 {
    let algorithm = AttestationSignatureAlgorithmV1::Ed25519;
    let profile = algorithm.exact_profile_v1();
    let mut envelope = DetachedAttestationEnvelopeV1 {
        protocol_version: CONTROL_PLANE_ATTESTATION_VERSION_V1,
        subject: AttestationSubjectV1::new(DOMAIN, SUBJECT).unwrap(),
        issued_at_unix_s: 100,
        valid_until_unix_s: 800,
        authority_generation_commitment: TRUST_GENERATION,
        signatures: Vec::new(),
    };
    let message = envelope.signer_message_v1(algorithm, "worker-a").unwrap();
    envelope.signatures.push(DetachedAttestationSignatureV1 {
        algorithm,
        key_id: "worker-a".into(),
        signature: pseudo_signature(profile, "worker-a", &message),
    });

    let verified = verify_current_attestation_crypto_bound_profile_v1(
        &envelope,
        &AttestationPolicyV1 {
            minimum_valid_signatures: 1,
            maximum_signatures: 1,
            maximum_signature_bytes: 128,
            maximum_key_id_bytes: 64,
            required_algorithms: BTreeSet::from([algorithm]),
            allowed_key_ids: Some(BTreeSet::from(["worker-a".to_string()])),
        },
        provider,
        ORIGINAL_TIME,
    )
    .unwrap();

    qualify_attestation_authority_diversity_v1(
        &verified,
        &QualifiedAttestationDiversityPolicyV1 {
            minimum_distinct_principals: 1,
            minimum_distinct_authorities: Some(1),
            allowed_principals: None,
            allowed_authorities: None,
        },
        resolver,
        ORIGINAL_TIME,
    )
    .unwrap()
}

fn fixture() -> (QualifiedAttestationAuthorityDiversityV1, Provider, Resolver) {
    let provider = Provider::new();
    let resolver = Resolver::new();
    let witness = qualified(&provider, &resolver);
    (witness, provider, resolver)
}

#[test]
fn unchanged_current_views_revalidate_and_preserve_earliest_expiry() {
    let (qualified, provider, resolver) = fixture();
    let current = revalidate_qualified_attestation_currentness_v1(
        &qualified,
        &provider,
        &resolver,
        REVALIDATION_TIME,
    )
    .unwrap();

    assert_eq!(current.revalidated_at_unix_s(), REVALIDATION_TIME);
    assert_eq!(current.natural_valid_until_unix_s(), 700);
    assert!(current.remains_within_time_bounds(699));
    assert!(!current.remains_within_time_bounds(700));
}

#[test]
fn trust_generation_drift_fails_closed() {
    let (qualified, mut provider, resolver) = fixture();
    provider.snapshot.generation_commitment[0] ^= 0xFF;

    assert!(matches!(
        revalidate_qualified_attestation_currentness_v1(
            &qualified,
            &provider,
            &resolver,
            REVALIDATION_TIME,
        ),
        Err(AttestationCurrentnessErrorV1::TrustGenerationChanged)
    ));
}

#[test]
fn same_generation_key_revocation_fails_closed() {
    let (qualified, mut provider, resolver) = fixture();
    provider.key.lifecycle = AttestationKeyLifecycleV1::Revoked;

    assert!(matches!(
        revalidate_qualified_attestation_currentness_v1(
            &qualified,
            &provider,
            &resolver,
            REVALIDATION_TIME,
        ),
        Err(AttestationCurrentnessErrorV1::CurrentKeyRevoked { .. })
    ));
}

#[test]
fn same_generation_profile_support_withdrawal_fails_closed() {
    let (qualified, mut provider, resolver) = fixture();
    provider.exact_profile_supported = false;

    assert!(matches!(
        revalidate_qualified_attestation_currentness_v1(
            &qualified,
            &provider,
            &resolver,
            REVALIDATION_TIME,
        ),
        Err(AttestationCurrentnessErrorV1::ExactProfileNoLongerSupported { .. })
    ));
}

#[test]
fn same_generation_domain_removal_fails_closed() {
    let (qualified, mut provider, resolver) = fixture();
    provider.key.allowed_subject_domains.clear();
    provider.key.allowed_subject_domains.insert("different.domain".into());

    assert!(matches!(
        revalidate_qualified_attestation_currentness_v1(
            &qualified,
            &provider,
            &resolver,
            REVALIDATION_TIME,
        ),
        Err(AttestationCurrentnessErrorV1::SubjectDomainNoLongerAllowed { .. })
    ));
}

#[test]
fn same_generation_key_expiry_drift_fails_closed() {
    let (qualified, mut provider, resolver) = fixture();
    provider.key.valid_until_unix_s = REVALIDATION_TIME;

    assert!(matches!(
        revalidate_qualified_attestation_currentness_v1(
            &qualified,
            &provider,
            &resolver,
            REVALIDATION_TIME,
        ),
        Err(AttestationCurrentnessErrorV1::CurrentKeyExpired { .. })
    ));
}

#[test]
fn authority_generation_drift_fails_closed() {
    let (qualified, provider, mut resolver) = fixture();
    resolver.snapshot.generation_commitment[0] ^= 0xFF;

    assert!(matches!(
        revalidate_qualified_attestation_currentness_v1(
            &qualified,
            &provider,
            &resolver,
            REVALIDATION_TIME,
        ),
        Err(AttestationCurrentnessErrorV1::AuthorityBindingGenerationChanged)
    ));
}

#[test]
fn same_generation_namespace_semantics_drift_fails_closed() {
    let (qualified, provider, mut resolver) = fixture();
    resolver
        .snapshot
        .identity_namespaces
        .principal
        .semantics_commitment[0] ^= 0xFF;

    assert!(matches!(
        revalidate_qualified_attestation_currentness_v1(
            &qualified,
            &provider,
            &resolver,
            REVALIDATION_TIME,
        ),
        Err(AttestationCurrentnessErrorV1::IdentityNamespaceChanged)
    ));
}

#[test]
fn same_generation_principal_remap_fails_closed() {
    let (qualified, provider, mut resolver) = fixture();
    resolver.binding.principal_id = "mallory".into();

    assert!(matches!(
        revalidate_qualified_attestation_currentness_v1(
            &qualified,
            &provider,
            &resolver,
            REVALIDATION_TIME,
        ),
        Err(AttestationCurrentnessErrorV1::PrincipalBindingChanged { .. })
    ));
}

#[test]
fn same_generation_authority_remap_fails_closed() {
    let (qualified, provider, mut resolver) = fixture();
    resolver.binding.authority_id = Some("org-b".into());

    assert!(matches!(
        revalidate_qualified_attestation_currentness_v1(
            &qualified,
            &provider,
            &resolver,
            REVALIDATION_TIME,
        ),
        Err(AttestationCurrentnessErrorV1::AuthorityBindingChanged { .. })
    ));
}

#[test]
fn same_generation_binding_expiry_drift_fails_closed() {
    let (qualified, provider, mut resolver) = fixture();
    resolver.binding.valid_until_unix_s = REVALIDATION_TIME;

    assert!(matches!(
        revalidate_qualified_attestation_currentness_v1(
            &qualified,
            &provider,
            &resolver,
            REVALIDATION_TIME,
        ),
        Err(AttestationCurrentnessErrorV1::CurrentBindingExpired { .. })
    ));
}

#[test]
fn revalidation_cannot_move_back_before_original_decision() {
    let (qualified, provider, resolver) = fixture();

    assert!(matches!(
        revalidate_qualified_attestation_currentness_v1(
            &qualified,
            &provider,
            &resolver,
            ORIGINAL_TIME - 1,
        ),
        Err(AttestationCurrentnessErrorV1::RevalidationBeforeOriginalDecision { .. })
    ));
}
