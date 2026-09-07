use super::*;
use crate::control_plane::attestation::{
    AttestationKeyLifecycleV1, AttestationPolicyV1, AttestationSubjectV1,
    AttestationTrustSnapshotV1, AttestationTrustVerifierV1, AttestationTrustedKeyV1,
    DetachedAttestationEnvelopeV1, DetachedAttestationSignatureV1,
    CONTROL_PLANE_ATTESTATION_VERSION_V1,
};
use crate::control_plane::attestation_profile::{
    AttestationCryptoProfileV1, ExactAttestationTrustVerifierV1,
    ExactProfileVerifiedAttestationV1, verify_current_attestation_exact_profile_v1,
};
use std::collections::{BTreeMap, BTreeSet};

const TRUST_GENERATION: [u8; 32] = [0x51; 32];
const BINDING_GENERATION: [u8; 32] = [0x61; 32];
const SUBJECT: [u8; 32] = [0x71; 32];
const DOMAIN: &str = "test.attestation-authority.v1";

#[derive(Clone)]
struct CryptoProvider {
    snapshot: AttestationTrustSnapshotV1,
    keys: BTreeMap<(AttestationSignatureAlgorithmV1, String), AttestationTrustedKeyV1>,
}

impl CryptoProvider {
    fn with_keys(keys: &[(AttestationSignatureAlgorithmV1, &str)]) -> Self {
        let mut records = BTreeMap::new();
        for (algorithm, key_id) in keys {
            records.insert(
                (*algorithm, (*key_id).to_string()),
                AttestationTrustedKeyV1 {
                    algorithm: *algorithm,
                    key_id: (*key_id).to_string(),
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
            keys: records,
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
        if trust_generation_commitment != TRUST_GENERATION {
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
        if trust_generation_commitment != TRUST_GENERATION {
            return Ok(false);
        }
        Ok(signature == pseudo_signature(algorithm, key_id, message))
    }
}

impl ExactAttestationTrustVerifierV1 for CryptoProvider {
    fn supports_exact_profile_v1(
        &self,
        _profile: AttestationCryptoProfileV1,
    ) -> Result<bool, String> {
        Ok(true)
    }
}

#[derive(Clone)]
struct AuthorityResolver {
    snapshot: AttestationAuthorityBindingSnapshotV1,
    bindings:
        BTreeMap<(AttestationSignatureAlgorithmV1, String), AttestationSignerAuthorityBindingV1>,
}

impl AuthorityResolver {
    fn new(
        bindings: impl IntoIterator<Item = AttestationSignerAuthorityBindingV1>,
    ) -> Self {
        let bindings = bindings
            .into_iter()
            .map(|binding| ((binding.algorithm, binding.key_id.clone()), binding))
            .collect();
        Self {
            snapshot: AttestationAuthorityBindingSnapshotV1 {
                generation_commitment: BINDING_GENERATION,
                valid_from_unix_s: 1,
                valid_until_unix_s: 850,
            },
            bindings,
        }
    }
}

impl AttestationAuthorityResolverV1 for AuthorityResolver {
    fn current_snapshot(&self) -> Result<AttestationAuthorityBindingSnapshotV1, String> {
        Ok(self.snapshot.clone())
    }

    fn signer_binding(
        &self,
        generation_commitment: [u8; 32],
        algorithm: AttestationSignatureAlgorithmV1,
        key_id: &str,
    ) -> Result<Option<AttestationSignerAuthorityBindingV1>, String> {
        if generation_commitment != self.snapshot.generation_commitment {
            return Ok(None);
        }
        Ok(self
            .bindings
            .get(&(algorithm, key_id.to_string()))
            .cloned())
    }
}

fn binding(
    algorithm: AttestationSignatureAlgorithmV1,
    key_id: &str,
    principal_id: &str,
    authority_id: Option<&str>,
    valid_until_unix_s: u64,
) -> AttestationSignerAuthorityBindingV1 {
    AttestationSignerAuthorityBindingV1 {
        generation_commitment: BINDING_GENERATION,
        algorithm,
        key_id: key_id.to_string(),
        principal_id: principal_id.to_string(),
        authority_id: authority_id.map(str::to_string),
        valid_from_unix_s: 20,
        valid_until_unix_s,
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
        hasher.update(b"test-only-authority-diversity-signature\0");
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

fn verified(
    signers: &[(AttestationSignatureAlgorithmV1, &str)],
) -> ExactProfileVerifiedAttestationV1 {
    let mut envelope = DetachedAttestationEnvelopeV1 {
        protocol_version: CONTROL_PLANE_ATTESTATION_VERSION_V1,
        subject: AttestationSubjectV1::new(DOMAIN, SUBJECT).unwrap(),
        issued_at_unix_s: 100,
        valid_until_unix_s: 800,
        authority_generation_commitment: TRUST_GENERATION,
        signatures: Vec::new(),
    };

    for (algorithm, key_id) in signers {
        let message = envelope.signer_message_v1(*algorithm, key_id).unwrap();
        envelope.signatures.push(DetachedAttestationSignatureV1 {
            algorithm: *algorithm,
            key_id: (*key_id).to_string(),
            signature: pseudo_signature(*algorithm, key_id, &message),
        });
    }

    let policy = AttestationPolicyV1 {
        minimum_valid_signatures: signers.len(),
        maximum_signatures: 8,
        maximum_signature_bytes: 8 * 1024,
        maximum_key_id_bytes: 128,
        required_algorithms: signers.iter().map(|(algorithm, _)| *algorithm).collect(),
        allowed_key_ids: Some(
            signers
                .iter()
                .map(|(_, key_id)| (*key_id).to_string())
                .collect(),
        ),
    };
    verify_current_attestation_exact_profile_v1(
        &envelope,
        &policy,
        &CryptoProvider::with_keys(signers),
        200,
    )
    .unwrap()
}

fn policy(
    minimum_distinct_principals: usize,
    minimum_distinct_authorities: Option<usize>,
) -> AttestationAuthorityDiversityPolicyV1 {
    AttestationAuthorityDiversityPolicyV1 {
        minimum_distinct_principals,
        minimum_distinct_authorities,
        allowed_principal_ids: None,
        allowed_authority_ids: None,
    }
}

#[test]
fn two_signature_entries_do_not_imply_two_principals() {
    let verified = verified(&[
        (AttestationSignatureAlgorithmV1::Ed25519, "key-a"),
        (AttestationSignatureAlgorithmV1::Ed25519, "key-b"),
    ]);
    let resolver = AuthorityResolver::new([
        binding(
            AttestationSignatureAlgorithmV1::Ed25519,
            "key-a",
            "principal-1",
            Some("authority-a"),
            700,
        ),
        binding(
            AttestationSignatureAlgorithmV1::Ed25519,
            "key-b",
            "principal-1",
            Some("authority-a"),
            700,
        ),
    ]);

    assert!(matches!(
        qualify_attestation_authority_diversity_v1(&verified, &policy(2, None), &resolver, 200),
        Err(AttestationAuthorityErrorV1::InsufficientDistinctPrincipals {
            actual: 1,
            required: 2
        })
    ));
}

#[test]
fn algorithm_diversity_does_not_imply_authority_diversity() {
    let verified = verified(&[
        (AttestationSignatureAlgorithmV1::Ed25519, "classical-key"),
        (AttestationSignatureAlgorithmV1::MlDsa65, "pq-key"),
    ]);
    let resolver = AuthorityResolver::new([
        binding(
            AttestationSignatureAlgorithmV1::Ed25519,
            "classical-key",
            "principal-1",
            Some("authority-a"),
            700,
        ),
        binding(
            AttestationSignatureAlgorithmV1::MlDsa65,
            "pq-key",
            "principal-1",
            Some("authority-a"),
            700,
        ),
    ]);

    assert!(matches!(
        qualify_attestation_authority_diversity_v1(
            &verified,
            &policy(1, Some(2)),
            &resolver,
            200
        ),
        Err(AttestationAuthorityErrorV1::InsufficientDistinctAuthorities {
            actual: 1,
            required: 2
        })
    ));
}

#[test]
fn distinct_principals_and_authorities_are_separate_predicates() {
    let verified = verified(&[
        (AttestationSignatureAlgorithmV1::Ed25519, "key-a"),
        (AttestationSignatureAlgorithmV1::Ed25519, "key-b"),
    ]);
    let same_authority = AuthorityResolver::new([
        binding(
            AttestationSignatureAlgorithmV1::Ed25519,
            "key-a",
            "principal-1",
            Some("authority-a"),
            700,
        ),
        binding(
            AttestationSignatureAlgorithmV1::Ed25519,
            "key-b",
            "principal-2",
            Some("authority-a"),
            700,
        ),
    ]);

    assert!(matches!(
        qualify_attestation_authority_diversity_v1(
            &verified,
            &policy(2, Some(2)),
            &same_authority,
            200
        ),
        Err(AttestationAuthorityErrorV1::InsufficientDistinctAuthorities {
            actual: 1,
            required: 2
        })
    ));

    let different_authorities = AuthorityResolver::new([
        binding(
            AttestationSignatureAlgorithmV1::Ed25519,
            "key-a",
            "principal-1",
            Some("authority-a"),
            700,
        ),
        binding(
            AttestationSignatureAlgorithmV1::Ed25519,
            "key-b",
            "principal-2",
            Some("authority-b"),
            650,
        ),
    ]);
    let qualified = qualify_attestation_authority_diversity_v1(
        &verified,
        &policy(2, Some(2)),
        &different_authorities,
        200,
    )
    .unwrap();

    assert_eq!(qualified.distinct_principal_ids().len(), 2);
    assert_eq!(qualified.distinct_authority_ids().len(), 2);
    assert_eq!(qualified.natural_valid_until_unix_s(), 650);
}

#[test]
fn missing_authority_id_is_allowed_only_when_not_claimed() {
    let verified = verified(&[(AttestationSignatureAlgorithmV1::Ed25519, "key-a")]);
    let resolver = AuthorityResolver::new([binding(
        AttestationSignatureAlgorithmV1::Ed25519,
        "key-a",
        "principal-1",
        None,
        700,
    )]);

    let principal_only =
        qualify_attestation_authority_diversity_v1(&verified, &policy(1, None), &resolver, 200)
            .unwrap();
    assert!(principal_only.distinct_authority_ids().is_empty());

    assert!(matches!(
        qualify_attestation_authority_diversity_v1(
            &verified,
            &policy(1, Some(1)),
            &resolver,
            200
        ),
        Err(AttestationAuthorityErrorV1::AuthorityIdentityRequired { .. })
    ));
}

#[test]
fn binding_generation_and_identity_substitution_fail_closed() {
    let verified = verified(&[(AttestationSignatureAlgorithmV1::Ed25519, "key-a")]);

    let mut wrong_generation_binding = binding(
        AttestationSignatureAlgorithmV1::Ed25519,
        "key-a",
        "principal-1",
        Some("authority-a"),
        700,
    );
    wrong_generation_binding.generation_commitment = [0x99; 32];
    let wrong_generation = AuthorityResolver::new([wrong_generation_binding]);
    assert!(matches!(
        qualify_attestation_authority_diversity_v1(
            &verified,
            &policy(1, None),
            &wrong_generation,
            200
        ),
        Err(AttestationAuthorityErrorV1::BindingGenerationMismatch { .. })
    ));

    let mut wrong_identity = AuthorityResolver::new([binding(
        AttestationSignatureAlgorithmV1::Ed25519,
        "key-a",
        "principal-1",
        Some("authority-a"),
        700,
    )]);
    let binding = wrong_identity
        .bindings
        .get_mut(&(AttestationSignatureAlgorithmV1::Ed25519, "key-a".into()))
        .unwrap();
    binding.key_id = "key-b".into();
    assert!(matches!(
        qualify_attestation_authority_diversity_v1(
            &verified,
            &policy(1, None),
            &wrong_identity,
            200
        ),
        Err(AttestationAuthorityErrorV1::BindingIdentityMismatch { .. })
    ));
}

#[test]
fn authority_qualification_must_share_base_decision_time() {
    let verified = verified(&[(AttestationSignatureAlgorithmV1::Ed25519, "key-a")]);
    let resolver = AuthorityResolver::new([binding(
        AttestationSignatureAlgorithmV1::Ed25519,
        "key-a",
        "principal-1",
        Some("authority-a"),
        700,
    )]);

    assert!(matches!(
        qualify_attestation_authority_diversity_v1(&verified, &policy(1, None), &resolver, 201),
        Err(AttestationAuthorityErrorV1::EvaluationTimeMismatch {
            verified_at_unix_s: 200,
            requested_at_unix_s: 201
        })
    ));
}

#[test]
fn principal_and_authority_allowlists_are_independent() {
    let verified = verified(&[(AttestationSignatureAlgorithmV1::Ed25519, "key-a")]);
    let resolver = AuthorityResolver::new([binding(
        AttestationSignatureAlgorithmV1::Ed25519,
        "key-a",
        "principal-1",
        Some("authority-a"),
        700,
    )]);

    let principal_reject = AttestationAuthorityDiversityPolicyV1 {
        minimum_distinct_principals: 1,
        minimum_distinct_authorities: None,
        allowed_principal_ids: Some(BTreeSet::from(["principal-2".to_string()])),
        allowed_authority_ids: None,
    };
    assert!(matches!(
        qualify_attestation_authority_diversity_v1(
            &verified,
            &principal_reject,
            &resolver,
            200
        ),
        Err(AttestationAuthorityErrorV1::PrincipalNotAllowed(_))
    ));

    let authority_reject = AttestationAuthorityDiversityPolicyV1 {
        minimum_distinct_principals: 1,
        minimum_distinct_authorities: Some(1),
        allowed_principal_ids: None,
        allowed_authority_ids: Some(BTreeSet::from(["authority-b".to_string()])),
    };
    assert!(matches!(
        qualify_attestation_authority_diversity_v1(
            &verified,
            &authority_reject,
            &resolver,
            200
        ),
        Err(AttestationAuthorityErrorV1::AuthorityNotAllowed(_))
    ));
}
