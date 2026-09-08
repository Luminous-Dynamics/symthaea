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
use crate::control_plane::attestation_currentness::revalidate_qualified_attestation_currentness_v1;
use crate::control_plane::attestation_identity_namespace::{
    AttestationIdentityNamespaceSetV1, AttestationIdentityNamespaceV1,
};
use crate::control_plane::attestation_profile::{
    AttestationCryptoProfileV1, ExactAttestationTrustVerifierV1,
};
use crate::control_plane::attestation_qualified_authority::{
    QualifiedAttestationAuthorityDiversityV1, QualifiedAttestationAuthorityResolverV1,
    QualifiedAttestationAuthoritySnapshotV1, QualifiedAttestationDiversityPolicyV1,
    QualifiedAttestationSignerBindingV1, qualify_attestation_authority_diversity_v1,
};
use crate::control_plane::forge::{ForgeArtifactIdentityV1, FORGE_PROTOCOL_VERSION};
use crate::control_plane::forge_compilation::{
    ForgeCompilationProfileIdentityV1, ForgeCompilationRequestV1,
    ForgeCompilationReceiptV1, ForgeCompiledArtifactIdentityV1,
};
use crate::control_plane::forge_compilation_attestation::{
    FORGE_COMPILATION_RECEIPT_ATTESTATION_SUBJECT_DOMAIN_V1,
    ForgeCompilationAttestationBindingV1, bind_forge_compilation_attestation_v1,
};
use crate::control_plane::forge_compiled_artifact_preflight::ForgeCompiledArtifactPreflightV1;
use crate::control_plane::forge_profile::{
    ForgeDeterminismPolicyV1, ForgeExecutionProfileDefinitionV1, ForgeImportPolicyV1,
    ForgeInterruptionPolicyV1, ForgeRuntimeIdentityV1, ForgeVerifierAbiV1,
};
use std::collections::BTreeSet;

const TRUST_GENERATION: [u8; 32] = [0x11; 32];
const BINDING_GENERATION: [u8; 32] = [0x22; 32];
const RUNTIME_LINEAGE: [u8; 32] = [0x33; 32];
const FEATURE_POLICY: [u8; 32] = [0x44; 32];
const ORIGINAL_TIME: u64 = 200;
const POINT_OF_USE_TIME: u64 = 250;
const CANDIDATE_A: &[u8] = b"forge-current-attestation-a";
const CANDIDATE_B: &[u8] = b"forge-current-attestation-b";

#[derive(Clone)]
struct Provider {
    snapshot: AttestationTrustSnapshotV1,
    key: AttestationTrustedKeyV1,
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
                allowed_subject_domains: BTreeSet::from([
                    FORGE_COMPILATION_RECEIPT_ATTESTATION_SUBJECT_DOMAIN_V1.to_string(),
                ]),
            },
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
        if trust_generation_commitment != self.snapshot.generation_commitment
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
        hasher.update(b"test-only-current-forge-compilation-attestation\0");
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
        Self {
            snapshot: QualifiedAttestationAuthoritySnapshotV1 {
                generation_commitment: BINDING_GENERATION,
                valid_from_unix_s: 1,
                valid_until_unix_s: 850,
                identity_namespaces: AttestationIdentityNamespaceSetV1::new(
                    AttestationIdentityNamespaceV1::new("principals-v1", [0x55; 32]).unwrap(),
                    Some(
                        AttestationIdentityNamespaceV1::new("authorities-v1", [0x66; 32])
                            .unwrap(),
                    ),
                )
                .unwrap(),
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

fn execution_profile() -> ForgeExecutionProfileDefinitionV1 {
    ForgeExecutionProfileDefinitionV1 {
        protocol_version: FORGE_PROTOCOL_VERSION,
        profile_id: "run-v1".into(),
        runtime: ForgeRuntimeIdentityV1::new_wasmtime("44.0.1", RUNTIME_LINEAGE).unwrap(),
        wasm_feature_policy_commitment: FEATURE_POLICY,
        abi: ForgeVerifierAbiV1::NoArgsI32,
        imports: ForgeImportPolicyV1::NoImports,
        determinism: ForgeDeterminismPolicyV1::Strict,
        interruption: ForgeInterruptionPolicyV1::DeterministicFuel,
        fuel: 1_000,
        max_precompiled_artifact_bytes: 4_096,
        max_linear_memory_bytes: 0,
        max_memories: 0,
        max_table_elements: 0,
        max_tables: 0,
        max_instances: 1,
        max_wasm_stack_bytes: 64 * 1024,
        trap_on_grow_failure: true,
    }
}

fn receipt(candidate: &[u8], source_tag: u8) -> ForgeCompilationReceiptV1 {
    let request = ForgeCompilationRequestV1::new(
        ForgeArtifactIdentityV1::new_blake3_256([source_tag; 32], 128).unwrap(),
        ForgeCompilationProfileIdentityV1::new("compile-v1", [0x77; 32]).unwrap(),
        RUNTIME_LINEAGE,
        FEATURE_POLICY,
    )
    .unwrap();
    let compiled = ForgeCompiledArtifactIdentityV1::new_blake3_256(
        *blake3::hash(candidate).as_bytes(),
        u64::try_from(candidate.len()).unwrap(),
    )
    .unwrap();
    ForgeCompilationReceiptV1::succeeded(request, compiled).unwrap()
}

fn qualified_for_receipt(
    receipt: &ForgeCompilationReceiptV1,
    provider: &Provider,
    resolver: &Resolver,
) -> QualifiedAttestationAuthorityDiversityV1 {
    let subject = AttestationSubjectV1::new(
        FORGE_COMPILATION_RECEIPT_ATTESTATION_SUBJECT_DOMAIN_V1,
        receipt.canonical_commitment_v1().unwrap(),
    )
    .unwrap();
    let algorithm = AttestationSignatureAlgorithmV1::Ed25519;
    let profile = algorithm.exact_profile_v1();
    let mut envelope = DetachedAttestationEnvelopeV1 {
        protocol_version: CONTROL_PLANE_ATTESTATION_VERSION_V1,
        subject,
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

fn binding_and_currentness(
    candidate: &[u8],
    source_tag: u8,
) -> (
    ForgeCompilationAttestationBindingV1,
    RevalidatedQualifiedAttestationCurrentnessV1,
) {
    let provider = Provider::new();
    let resolver = Resolver::new();
    let receipt = receipt(candidate, source_tag);
    let profile = execution_profile();
    let preflight = ForgeCompiledArtifactPreflightV1::verify(&receipt, &profile, candidate).unwrap();
    let qualified = qualified_for_receipt(&receipt, &provider, &resolver);
    let binding = bind_forge_compilation_attestation_v1(&preflight, &qualified, ORIGINAL_TIME).unwrap();
    let currentness = revalidate_qualified_attestation_currentness_v1(
        &qualified,
        &provider,
        &resolver,
        POINT_OF_USE_TIME,
    )
    .unwrap();
    (binding, currentness)
}

#[test]
fn exact_same_receipt_binding_and_currentness_compose() {
    let (binding, currentness) = binding_and_currentness(CANDIDATE_A, 0x81);
    let current = bind_current_forge_compilation_attestation_v1(
        &binding,
        &currentness,
        POINT_OF_USE_TIME,
    )
    .unwrap();

    assert_eq!(current.point_of_use_unix_s(), POINT_OF_USE_TIME);
    assert_eq!(current.natural_valid_until_unix_s(), 700);
    assert_eq!(
        current.compilation_receipt_commitment(),
        binding.compilation_receipt_commitment()
    );
    assert_eq!(
        current.execution_profile_commitment(),
        binding.execution_profile_commitment()
    );
}

#[test]
fn fresh_currentness_for_another_qualified_attestation_cannot_substitute() {
    let (binding_a, _) = binding_and_currentness(CANDIDATE_A, 0x81);
    let (_, currentness_b) = binding_and_currentness(CANDIDATE_B, 0x82);

    assert!(matches!(
        bind_current_forge_compilation_attestation_v1(
            &binding_a,
            &currentness_b,
            POINT_OF_USE_TIME,
        ),
        Err(CurrentForgeCompilationAttestationErrorV1::QualifiedAttestationMismatch)
    ));
}

#[test]
fn point_of_use_time_must_be_the_exact_revalidation_instant() {
    let (binding, currentness) = binding_and_currentness(CANDIDATE_A, 0x81);

    assert!(matches!(
        bind_current_forge_compilation_attestation_v1(
            &binding,
            &currentness,
            POINT_OF_USE_TIME + 1,
        ),
        Err(CurrentForgeCompilationAttestationErrorV1::PointOfUseTimeMismatch { .. })
    ));
}
