use super::*;
use crate::control_plane::attestation::{
    AttestationKeyLifecycleV1, AttestationPolicyV1, AttestationSignatureAlgorithmV1,
    AttestationSubjectV1, AttestationTrustSnapshotV1, AttestationTrustVerifierV1,
    AttestationTrustedKeyV1, DetachedAttestationEnvelopeV1,
    DetachedAttestationSignatureV1, CONTROL_PLANE_ATTESTATION_VERSION_V1,
};
use crate::control_plane::attestation_crypto_profile::{
    ProfileBoundAttestationTrustVerifierV1,
    verify_current_attestation_crypto_bound_profile_v1,
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
use crate::control_plane::forge::{ForgeArtifactIdentityV1, FORGE_PROTOCOL_VERSION};
use crate::control_plane::forge_compilation::{
    ForgeCompilationProfileIdentityV1, ForgeCompilationReceiptV1,
    ForgeCompilationRequestV1, ForgeCompiledArtifactIdentityV1,
};
use crate::control_plane::forge_compilation_attestation::{
    FORGE_COMPILATION_RECEIPT_ATTESTATION_SUBJECT_DOMAIN_V1,
    bind_forge_compilation_attestation_v1,
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
const CANDIDATE: &[u8] = b"forge-currentness-precompiled-v1";
const ORIGINAL_TIME: u64 = 200;
const RECHECK_TIME: u64 = 250;

#[derive(Clone)]
struct CryptoProvider {
    snapshot: AttestationTrustSnapshotV1,
    key: AttestationTrustedKeyV1,
    supports_profile: bool,
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
        // Consequential qualification must use the exact-profile operation.
        Ok(false)
    }
}

impl ExactAttestationTrustVerifierV1 for CryptoProvider {
    fn supports_exact_profile_v1(
        &self,
        profile: AttestationCryptoProfileV1,
    ) -> Result<bool, String> {
        Ok(self.supports_profile
            && profile == AttestationSignatureAlgorithmV1::Ed25519.exact_profile_v1())
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
        if trust_generation_commitment != self.snapshot.generation_commitment
            || !self.supports_profile
            || profile != AttestationSignatureAlgorithmV1::Ed25519.exact_profile_v1()
            || key_id != self.key.key_id
        {
            return Ok(false);
        }
        Ok(signature == pseudo_signature(profile, key_id, message))
    }
}

#[derive(Clone)]
struct Resolver {
    snapshot: QualifiedAttestationAuthoritySnapshotV1,
    binding: QualifiedAttestationSignerBindingV1,
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

fn pseudo_signature(
    profile: AttestationCryptoProfileV1,
    key_id: &str,
    message: &[u8],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(profile.signature_bytes);
    let mut counter = 0u32;
    while out.len() < profile.signature_bytes {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"test-only-forge-currentness-signature\0");
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

fn request() -> ForgeCompilationRequestV1 {
    ForgeCompilationRequestV1::new(
        ForgeArtifactIdentityV1::new_blake3_256([0x55; 32], 128).unwrap(),
        ForgeCompilationProfileIdentityV1::new("compile-v1", [0x66; 32]).unwrap(),
        RUNTIME_LINEAGE,
        FEATURE_POLICY,
    )
    .unwrap()
}

fn receipt() -> ForgeCompilationReceiptV1 {
    let compiled = ForgeCompiledArtifactIdentityV1::new_blake3_256(
        *blake3::hash(CANDIDATE).as_bytes(),
        u64::try_from(CANDIDATE.len()).unwrap(),
    )
    .unwrap();
    ForgeCompilationReceiptV1::succeeded(request(), compiled).unwrap()
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

fn initial_crypto_provider() -> CryptoProvider {
    CryptoProvider {
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
        supports_profile: true,
    }
}

fn initial_resolver() -> Resolver {
    let namespaces = AttestationIdentityNamespaceSetV1::new(
        AttestationIdentityNamespaceV1::new("forge-builders-v1", [0x77; 32]).unwrap(),
        Some(
            AttestationIdentityNamespaceV1::new("forge-builder-authorities-v1", [0x88; 32])
                .unwrap(),
        ),
    )
    .unwrap();
    Resolver {
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
            principal_id: "builder-a".into(),
            authority_id: Some("build-org-a".into()),
            valid_from_unix_s: 10,
            valid_until_unix_s: 700,
        },
    }
}

fn original_binding(
    crypto: &CryptoProvider,
    resolver: &Resolver,
) -> ForgeCompilationAttestationBindingV1 {
    let receipt = receipt();
    let preflight = ForgeCompiledArtifactPreflightV1::verify(
        &receipt,
        &execution_profile(),
        CANDIDATE,
    )
    .unwrap();

    let algorithm = AttestationSignatureAlgorithmV1::Ed25519;
    let profile = algorithm.exact_profile_v1();
    let subject = AttestationSubjectV1::new(
        FORGE_COMPILATION_RECEIPT_ATTESTATION_SUBJECT_DOMAIN_V1,
        receipt.canonical_commitment_v1().unwrap(),
    )
    .unwrap();
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

    let attestation_policy = AttestationPolicyV1 {
        minimum_valid_signatures: 1,
        maximum_signatures: 2,
        maximum_signature_bytes: 4 * 1024,
        maximum_key_id_bytes: 128,
        required_algorithms: BTreeSet::from([algorithm]),
        allowed_key_ids: Some(BTreeSet::from(["worker-a".to_string()])),
    };
    let verified = verify_current_attestation_crypto_bound_profile_v1(
        &envelope,
        &attestation_policy,
        crypto,
        ORIGINAL_TIME,
    )
    .unwrap();
    let diversity_policy = QualifiedAttestationDiversityPolicyV1 {
        minimum_distinct_principals: 1,
        minimum_distinct_authorities: Some(1),
        allowed_principals: None,
        allowed_authorities: None,
    };
    let qualified = qualify_attestation_authority_diversity_v1(
        &verified,
        &diversity_policy,
        resolver,
        ORIGINAL_TIME,
    )
    .unwrap();

    bind_forge_compilation_attestation_v1(&preflight, &qualified, ORIGINAL_TIME).unwrap()
}

fn fixture() -> (ForgeCompilationAttestationBindingV1, CryptoProvider, Resolver) {
    let crypto = initial_crypto_provider();
    let resolver = initial_resolver();
    let binding = original_binding(&crypto, &resolver);
    (binding, crypto, resolver)
}

#[test]
fn unchanged_current_state_revalidates_and_retains_original_deadline() {
    let (binding, crypto, resolver) = fixture();
    let current = revalidate_forge_compilation_attestation_current_v1(
        &binding,
        &crypto,
        &resolver,
        RECHECK_TIME,
    )
    .unwrap();

    assert_eq!(current.checked_at_unix_s(), RECHECK_TIME);
    assert_eq!(current.current_valid_until_unix_s(), 700);
    assert_eq!(
        current.binding().compilation_receipt_commitment(),
        binding.compilation_receipt_commitment()
    );
}

#[test]
fn trust_generation_drift_fails() {
    let (binding, mut crypto, resolver) = fixture();
    crypto.snapshot.generation_commitment[0] ^= 0xFF;

    assert!(matches!(
        revalidate_forge_compilation_attestation_current_v1(
            &binding,
            &crypto,
            &resolver,
            RECHECK_TIME,
        ),
        Err(ForgeCompilationAttestationCurrentnessErrorV1::TrustGenerationChanged { .. })
    ));
}

#[test]
fn same_generation_key_revocation_still_fails() {
    let (binding, mut crypto, resolver) = fixture();
    crypto.key.lifecycle = AttestationKeyLifecycleV1::Revoked;

    assert!(matches!(
        revalidate_forge_compilation_attestation_current_v1(
            &binding,
            &crypto,
            &resolver,
            RECHECK_TIME,
        ),
        Err(ForgeCompilationAttestationCurrentnessErrorV1::SignerKeyRevoked { .. })
    ));
}

#[test]
fn same_generation_domain_removal_still_fails() {
    let (binding, mut crypto, resolver) = fixture();
    crypto.key.allowed_subject_domains = BTreeSet::from(["other.domain".to_string()]);

    assert!(matches!(
        revalidate_forge_compilation_attestation_current_v1(
            &binding,
            &crypto,
            &resolver,
            RECHECK_TIME,
        ),
        Err(
            ForgeCompilationAttestationCurrentnessErrorV1::SignerNoLongerAllowsForgeReceiptDomain { .. }
        )
    ));
}

#[test]
fn exact_crypto_profile_support_drift_fails() {
    let (binding, mut crypto, resolver) = fixture();
    crypto.supports_profile = false;

    assert!(matches!(
        revalidate_forge_compilation_attestation_current_v1(
            &binding,
            &crypto,
            &resolver,
            RECHECK_TIME,
        ),
        Err(
            ForgeCompilationAttestationCurrentnessErrorV1::ExactCryptoProfileNoLongerSupported { .. }
        )
    ));
}

#[test]
fn authority_binding_generation_drift_fails() {
    let (binding, crypto, mut resolver) = fixture();
    resolver.snapshot.generation_commitment[0] ^= 0xFF;

    assert!(matches!(
        revalidate_forge_compilation_attestation_current_v1(
            &binding,
            &crypto,
            &resolver,
            RECHECK_TIME,
        ),
        Err(
            ForgeCompilationAttestationCurrentnessErrorV1::AuthorityBindingGenerationChanged { .. }
        )
    ));
}

#[test]
fn same_generation_namespace_semantics_drift_fails() {
    let (binding, crypto, mut resolver) = fixture();
    resolver
        .snapshot
        .identity_namespaces
        .principal
        .semantics_commitment[0] ^= 0xFF;

    assert!(matches!(
        revalidate_forge_compilation_attestation_current_v1(
            &binding,
            &crypto,
            &resolver,
            RECHECK_TIME,
        ),
        Err(
            ForgeCompilationAttestationCurrentnessErrorV1::IdentityNamespaceSemanticsChanged { .. }
        )
    ));
}

#[test]
fn same_generation_principal_mapping_drift_fails() {
    let (binding, crypto, mut resolver) = fixture();
    resolver.binding.principal_id = "builder-b".into();

    assert!(matches!(
        revalidate_forge_compilation_attestation_current_v1(
            &binding,
            &crypto,
            &resolver,
            RECHECK_TIME,
        ),
        Err(ForgeCompilationAttestationCurrentnessErrorV1::PrincipalIdentityChanged { .. })
    ));
}

#[test]
fn same_generation_authority_mapping_drift_fails() {
    let (binding, crypto, mut resolver) = fixture();
    resolver.binding.authority_id = Some("build-org-b".into());

    assert!(matches!(
        revalidate_forge_compilation_attestation_current_v1(
            &binding,
            &crypto,
            &resolver,
            RECHECK_TIME,
        ),
        Err(ForgeCompilationAttestationCurrentnessErrorV1::AuthorityIdentityChanged { .. })
    ));
}

#[test]
fn current_key_expiry_fails_even_without_generation_change() {
    let (binding, mut crypto, resolver) = fixture();
    crypto.key.valid_until_unix_s = RECHECK_TIME;

    assert!(matches!(
        revalidate_forge_compilation_attestation_current_v1(
            &binding,
            &crypto,
            &resolver,
            RECHECK_TIME,
        ),
        Err(ForgeCompilationAttestationCurrentnessErrorV1::SignerKeyExpired { .. })
    ));
}

#[test]
fn current_binding_expiry_fails_even_without_generation_change() {
    let (binding, crypto, mut resolver) = fixture();
    resolver.binding.valid_until_unix_s = RECHECK_TIME;

    assert!(matches!(
        revalidate_forge_compilation_attestation_current_v1(
            &binding,
            &crypto,
            &resolver,
            RECHECK_TIME,
        ),
        Err(ForgeCompilationAttestationCurrentnessErrorV1::AuthorityBindingExpired { .. })
    ));
}

#[test]
fn current_state_can_tighten_the_original_validity_boundary() {
    let (binding, mut crypto, resolver) = fixture();
    crypto.key.valid_until_unix_s = 500;

    let current = revalidate_forge_compilation_attestation_current_v1(
        &binding,
        &crypto,
        &resolver,
        RECHECK_TIME,
    )
    .unwrap();
    assert_eq!(binding.natural_valid_until_unix_s(), 700);
    assert_eq!(current.current_valid_until_unix_s(), 500);
}

#[test]
fn check_at_original_half_open_expiry_fails_before_provider_reuse() {
    let (binding, crypto, resolver) = fixture();

    assert!(matches!(
        revalidate_forge_compilation_attestation_current_v1(
            &binding,
            &crypto,
            &resolver,
            binding.natural_valid_until_unix_s(),
        ),
        Err(
            ForgeCompilationAttestationCurrentnessErrorV1::OriginalBindingOutsideTimeBounds
        )
    ));
}
