use super::*;
use crate::control_plane::attestation::{
    AttestationKeyLifecycleV1, AttestationPolicyV1, AttestationSubjectV1,
    AttestationTrustSnapshotV1, AttestationTrustVerifierV1, AttestationTrustedKeyV1,
    DetachedAttestationEnvelopeV1, DetachedAttestationSignatureV1,
    CONTROL_PLANE_ATTESTATION_VERSION_V1,
};
use crate::control_plane::attestation_crypto_profile::{
    ProfileBoundAttestationTrustVerifierV1, verify_current_attestation_crypto_bound_profile_v1,
};
use crate::control_plane::attestation_identity_namespace::{
    AttestationIdentityNamespaceV1, AttestationIdentityNamespaceSetV1,
};
use crate::control_plane::attestation_profile::{
    AttestationCryptoProfileV1, ExactAttestationTrustVerifierV1,
};
use std::collections::{BTreeMap, BTreeSet};

const TRUST_GENERATION: [u8; 32] = [0x21; 32];
const BINDING_GENERATION: [u8; 32] = [0x31; 32];
const DOMAIN: &str = "test.qualified-attestation-authority.v1";
const SUBJECT: [u8; 32] = [0x41; 32];
const EVALUATION_TIME: u64 = 200;

struct CryptoProvider {
    snapshot: AttestationTrustSnapshotV1,
    keys: BTreeMap<(AttestationSignatureAlgorithmV1, String), AttestationTrustedKeyV1>,
}

impl CryptoProvider {
    fn new() -> Self {
        let mut keys = BTreeMap::new();
        for (algorithm, key_id) in [
            (AttestationSignatureAlgorithmV1::Ed25519, "worker-a"),
            (AttestationSignatureAlgorithmV1::MlDsa65, "worker-b"),
        ] {
            keys.insert(
                (algorithm, key_id.to_string()),
                AttestationTrustedKeyV1 {
                    algorithm,
                    key_id: key_id.to_string(),
                    valid_from_unix_s: 10,
                    valid_until_unix_s: 900,
                    lifecycle: AttestationKeyLifecycleV1::Active,
                    allowed_subject_domains: BTreeSet::from([DOMAIN.to_string()]),
                },
            );
        }
        Self {
            snapshot: AttestationTrustSnapshotV1 {
                generation_commitment: TRUST_GENERATION,
                valid_from_unix_s: 1,
                valid_until_unix_s: 1_000,
            },
            keys,
        }
    }
}

impl AttestationTrustVerifierV1 for CryptoProvider {
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
        // Consequential tests must route through the exact-profile operation.
        Ok(false)
    }
}

impl ExactAttestationTrustVerifierV1 for CryptoProvider {
    fn supports_exact_profile_v1(
        &self,
        profile: AttestationCryptoProfileV1,
    ) -> Result<bool, String> {
        Ok(matches!(
            profile.algorithm,
            AttestationSignatureAlgorithmV1::Ed25519 | AttestationSignatureAlgorithmV1::MlDsa65
        ))
    }
}

impl ProfileBoundAttestationTrustVerifierV1 for CryptoProvider {
    fn verify_signature_exact_profile_v1(
        &self,
        trust_generation_commitment: [u8; 32],
        profile: AttestationCryptoProfileV1,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String> {
        if trust_generation_commitment != self.snapshot.generation_commitment {
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
        hasher.update(b"test-only-qualified-attestation-signature\0");
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
    let mut envelope = DetachedAttestationEnvelopeV1 {
        protocol_version: CONTROL_PLANE_ATTESTATION_VERSION_V1,
        subject: AttestationSubjectV1::new(DOMAIN, SUBJECT).unwrap(),
        issued_at_unix_s: 100,
        valid_until_unix_s: 800,
        authority_generation_commitment: TRUST_GENERATION,
        signatures: Vec::new(),
    };

    for (algorithm, key_id) in [
        (AttestationSignatureAlgorithmV1::Ed25519, "worker-a"),
        (AttestationSignatureAlgorithmV1::MlDsa65, "worker-b"),
    ] {
        let profile = algorithm.exact_profile_v1();
        let message = envelope.signer_message_v1(algorithm, key_id).unwrap();
        envelope.signatures.push(DetachedAttestationSignatureV1 {
            algorithm,
            key_id: key_id.to_string(),
            signature: pseudo_signature(profile, key_id, &message),
        });
    }
    envelope
}

fn crypto_policy() -> AttestationPolicyV1 {
    AttestationPolicyV1 {
        minimum_valid_signatures: 2,
        maximum_signatures: 4,
        maximum_signature_bytes: 4 * 1024,
        maximum_key_id_bytes: 128,
        required_algorithms: BTreeSet::from([
            AttestationSignatureAlgorithmV1::Ed25519,
            AttestationSignatureAlgorithmV1::MlDsa65,
        ]),
        allowed_key_ids: Some(BTreeSet::from([
            "worker-a".to_string(),
            "worker-b".to_string(),
        ])),
    }
}

fn verified() -> CryptographicallyBoundExactProfileAttestationV1 {
    verify_current_attestation_crypto_bound_profile_v1(
        &envelope(),
        &crypto_policy(),
        &CryptoProvider::new(),
        EVALUATION_TIME,
    )
    .unwrap()
}

fn namespaces() -> AttestationIdentityNamespaceSetV1 {
    AttestationIdentityNamespaceSetV1::new(
        AttestationIdentityNamespaceV1::new("test-principals-v1", [0x51; 32]).unwrap(),
        Some(
            AttestationIdentityNamespaceV1::new("test-authorities-v1", [0x61; 32]).unwrap(),
        ),
    )
    .unwrap()
}

#[derive(Clone)]
struct Resolver {
    snapshot: QualifiedAttestationAuthoritySnapshotV1,
    bindings: BTreeMap<
        (AttestationSignatureAlgorithmV1, String),
        QualifiedAttestationSignerBindingV1,
    >,
}

impl Resolver {
    fn with_owners(
        a_principal: &str,
        a_authority: Option<&str>,
        b_principal: &str,
        b_authority: Option<&str>,
    ) -> Self {
        let snapshot = QualifiedAttestationAuthoritySnapshotV1 {
            generation_commitment: BINDING_GENERATION,
            valid_from_unix_s: 1,
            valid_until_unix_s: 850,
            identity_namespaces: namespaces(),
        };
        let mut bindings = BTreeMap::new();
        for (algorithm, key_id, principal_id, authority_id) in [
            (
                AttestationSignatureAlgorithmV1::Ed25519,
                "worker-a",
                a_principal,
                a_authority,
            ),
            (
                AttestationSignatureAlgorithmV1::MlDsa65,
                "worker-b",
                b_principal,
                b_authority,
            ),
        ] {
            bindings.insert(
                (algorithm, key_id.to_string()),
                QualifiedAttestationSignerBindingV1 {
                    generation_commitment: BINDING_GENERATION,
                    algorithm,
                    key_id: key_id.to_string(),
                    principal_id: principal_id.to_string(),
                    authority_id: authority_id.map(str::to_string),
                    valid_from_unix_s: 10,
                    valid_until_unix_s: if key_id == "worker-a" { 700 } else { 750 },
                },
            );
        }
        Self { snapshot, bindings }
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
        if generation_commitment != self.snapshot.generation_commitment {
            return Ok(None);
        }
        Ok(self.bindings.get(&(algorithm, key_id.to_string())).cloned())
    }
}

fn policy(
    principals: usize,
    authorities: Option<usize>,
) -> QualifiedAttestationDiversityPolicyV1 {
    QualifiedAttestationDiversityPolicyV1 {
        minimum_distinct_principals: principals,
        minimum_distinct_authorities: authorities,
        allowed_principals: None,
        allowed_authorities: None,
    }
}

#[test]
fn two_signature_entries_owned_by_one_principal_do_not_make_two_principals() {
    let resolver = Resolver::with_owners("alice", Some("org-a"), "alice", Some("org-a"));
    let result = qualify_attestation_authority_diversity_v1(
        &verified(),
        &policy(2, None),
        &resolver,
        EVALUATION_TIME,
    );
    assert!(matches!(
        result,
        Err(QualifiedAttestationAuthorityErrorV1::InsufficientDistinctPrincipals {
            actual: 1,
            required: 2,
        })
    ));
}

#[test]
fn two_exact_crypto_profiles_under_one_authority_do_not_make_two_authorities() {
    let resolver = Resolver::with_owners("alice", Some("org-a"), "bob", Some("org-a"));
    let result = qualify_attestation_authority_diversity_v1(
        &verified(),
        &policy(2, Some(2)),
        &resolver,
        EVALUATION_TIME,
    );
    assert!(matches!(
        result,
        Err(QualifiedAttestationAuthorityErrorV1::InsufficientDistinctAuthorities {
            actual: 1,
            required: 2,
        })
    ));
}

#[test]
fn distinct_namespace_qualified_principals_and_authorities_can_qualify() {
    let resolver = Resolver::with_owners("alice", Some("org-a"), "bob", Some("org-b"));
    let witness = qualify_attestation_authority_diversity_v1(
        &verified(),
        &policy(2, Some(2)),
        &resolver,
        EVALUATION_TIME,
    )
    .unwrap();

    assert_eq!(witness.distinct_principals().len(), 2);
    assert_eq!(witness.distinct_authorities().len(), 2);
    assert_eq!(witness.entries().len(), 2);
    assert!(
        witness
            .distinct_principals()
            .iter()
            .all(|identity| identity.kind == AttestationIdentityKindV1::Principal)
    );
    assert!(
        witness
            .distinct_authorities()
            .iter()
            .all(|identity| identity.kind == AttestationIdentityKindV1::Authority)
    );
    assert_eq!(witness.trust_generation_commitment(), TRUST_GENERATION);
    assert_eq!(
        witness.authority_binding_generation_commitment(),
        BINDING_GENERATION
    );
    assert_eq!(witness.natural_valid_until_unix_s(), 700);
}

#[test]
fn policy_allowlist_from_another_namespace_fails_before_identity_match() {
    let resolver = Resolver::with_owners("alice", Some("org-a"), "bob", Some("org-b"));
    let foreign = AttestationIdentityNamespaceSetV1::new(
        AttestationIdentityNamespaceV1::new("foreign-principals", [0x71; 32]).unwrap(),
        None,
    )
    .unwrap()
    .qualify_principal("alice")
    .unwrap();
    let policy = QualifiedAttestationDiversityPolicyV1 {
        minimum_distinct_principals: 1,
        minimum_distinct_authorities: None,
        allowed_principals: Some(BTreeSet::from([foreign])),
        allowed_authorities: None,
    };

    let result = qualify_attestation_authority_diversity_v1(
        &verified(),
        &policy,
        &resolver,
        EVALUATION_TIME,
    );
    assert!(matches!(
        result,
        Err(QualifiedAttestationAuthorityErrorV1::PolicyIdentityNamespaceMismatch {
            kind: AttestationIdentityKindV1::Principal,
        })
    ));
}

#[test]
fn authority_policy_requires_authority_namespace() {
    let mut resolver = Resolver::with_owners("alice", Some("org-a"), "bob", Some("org-b"));
    resolver.snapshot.identity_namespaces.authority = None;

    let result = qualify_attestation_authority_diversity_v1(
        &verified(),
        &policy(1, Some(1)),
        &resolver,
        EVALUATION_TIME,
    );
    assert!(matches!(
        result,
        Err(QualifiedAttestationAuthorityErrorV1::AuthorityNamespaceRequired)
    ));
}

#[test]
fn changed_namespace_semantics_invalidates_qualified_allowlist() {
    let resolver = Resolver::with_owners("alice", Some("org-a"), "bob", Some("org-b"));
    let alice = resolver
        .snapshot
        .identity_namespaces
        .qualify_principal("alice")
        .unwrap();
    let policy = QualifiedAttestationDiversityPolicyV1 {
        minimum_distinct_principals: 1,
        minimum_distinct_authorities: None,
        allowed_principals: Some(BTreeSet::from([alice])),
        allowed_authorities: None,
    };

    let mut changed = resolver.clone();
    changed
        .snapshot
        .identity_namespaces
        .principal
        .semantics_commitment[0] ^= 0xFF;

    let result = qualify_attestation_authority_diversity_v1(
        &verified(),
        &policy,
        &changed,
        EVALUATION_TIME,
    );
    assert!(matches!(
        result,
        Err(QualifiedAttestationAuthorityErrorV1::PolicyIdentityNamespaceMismatch {
            kind: AttestationIdentityKindV1::Principal,
        })
    ));
}

#[test]
fn binding_generation_substitution_fails() {
    let mut resolver = Resolver::with_owners("alice", Some("org-a"), "bob", Some("org-b"));
    resolver
        .bindings
        .get_mut(&(AttestationSignatureAlgorithmV1::Ed25519, "worker-a".to_string()))
        .unwrap()
        .generation_commitment = [0x99; 32];

    let result = qualify_attestation_authority_diversity_v1(
        &verified(),
        &policy(1, None),
        &resolver,
        EVALUATION_TIME,
    );
    assert!(matches!(
        result,
        Err(QualifiedAttestationAuthorityErrorV1::BindingGenerationMismatch { .. })
    ));
}

#[test]
fn identity_qualification_must_share_crypto_decision_time() {
    let resolver = Resolver::with_owners("alice", Some("org-a"), "bob", Some("org-b"));
    let result = qualify_attestation_authority_diversity_v1(
        &verified(),
        &policy(1, None),
        &resolver,
        EVALUATION_TIME + 1,
    );
    assert!(matches!(
        result,
        Err(QualifiedAttestationAuthorityErrorV1::EvaluationTimeMismatch { .. })
    ));
}

#[test]
fn missing_authority_identity_fails_only_when_authority_dimension_is_required() {
    let resolver = Resolver::with_owners("alice", None, "bob", None);
    assert!(qualify_attestation_authority_diversity_v1(
        &verified(),
        &policy(2, None),
        &resolver,
        EVALUATION_TIME,
    )
    .is_ok());

    let result = qualify_attestation_authority_diversity_v1(
        &verified(),
        &policy(2, Some(1)),
        &resolver,
        EVALUATION_TIME,
    );
    assert!(matches!(
        result,
        Err(QualifiedAttestationAuthorityErrorV1::AuthorityIdentityRequired { .. })
    ));
}
