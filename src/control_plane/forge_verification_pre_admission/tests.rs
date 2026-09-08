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
    QualifiedAttestationAuthorityResolverV1, QualifiedAttestationAuthoritySnapshotV1,
    QualifiedAttestationDiversityPolicyV1, QualifiedAttestationSignerBindingV1,
    qualify_attestation_authority_diversity_v1,
};
use crate::control_plane::forge::{
    ForgeArtifactIdentityV1, ForgeCapabilityScopeV1, ForgeExecutionModeV1,
    ForgeExecutionProfileIdentityV1, ForgeVerificationRequestV1, FORGE_PROTOCOL_VERSION,
};
use crate::control_plane::forge_compilation::{
    ForgeCompilationProfileIdentityV1, ForgeCompilationRequestV1,
    ForgeCompilationReceiptV1, ForgeCompiledArtifactIdentityV1,
};
use crate::control_plane::forge_compilation_attestation::{
    FORGE_COMPILATION_RECEIPT_ATTESTATION_SUBJECT_DOMAIN_V1,
    bind_forge_compilation_attestation_v1,
};
use crate::control_plane::forge_compiled_artifact_preflight::ForgeCompiledArtifactPreflightV1;
use crate::control_plane::forge_current_compilation_attestation::bind_current_forge_compilation_attestation_v1;
use crate::control_plane::forge_profile::{
    ForgeDeterminismPolicyV1, ForgeExecutionProfileDefinitionV1, ForgeImportPolicyV1,
    ForgeInterruptionPolicyV1, ForgeRuntimeIdentityV1, ForgeVerifierAbiV1,
};
use std::collections::BTreeSet;

const TRUST_GENERATION: [u8; 32] = [0x11; 32];
const BINDING_GENERATION: [u8; 32] = [0x22; 32];
const RUNTIME_LINEAGE: [u8; 32] = [0x33; 32];
const FEATURE_POLICY: [u8; 32] = [0x44; 32];
const SOURCE_DIGEST: [u8; 32] = [0x55; 32];
const COMPILE_PROFILE: [u8; 32] = [0x66; 32];
const ORIGINAL_TIME: u64 = 200;
const DECISION_TIME: u64 = 250;
const CANDIDATE: &[u8] = b"forge-pre-admission-compiled-artifact";

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
        generation: [u8; 32],
        algorithm: AttestationSignatureAlgorithmV1,
        key_id: &str,
    ) -> Result<Option<AttestationTrustedKeyV1>, String> {
        if generation == self.snapshot.generation_commitment
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
        _generation: [u8; 32],
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
        generation: [u8; 32],
        profile: AttestationCryptoProfileV1,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String> {
        if generation != self.snapshot.generation_commitment
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
        hasher.update(b"test-only-forge-pre-admission-signature\0");
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
                    AttestationIdentityNamespaceV1::new("principals-v1", [0x71; 32]).unwrap(),
                    Some(AttestationIdentityNamespaceV1::new("authorities-v1", [0x72; 32]).unwrap()),
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
        generation: [u8; 32],
        algorithm: AttestationSignatureAlgorithmV1,
        key_id: &str,
    ) -> Result<Option<QualifiedAttestationSignerBindingV1>, String> {
        if generation == self.snapshot.generation_commitment
            && algorithm == self.binding.algorithm
            && key_id == self.binding.key_id
        {
            Ok(Some(self.binding.clone()))
        } else {
            Ok(None)
        }
    }
}

fn source_artifact() -> ForgeArtifactIdentityV1 {
    ForgeArtifactIdentityV1::new_blake3_256(SOURCE_DIGEST, 128).unwrap()
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

fn receipt() -> ForgeCompilationReceiptV1 {
    let request = ForgeCompilationRequestV1::new(
        source_artifact(),
        ForgeCompilationProfileIdentityV1::new("compile-v1", COMPILE_PROFILE).unwrap(),
        RUNTIME_LINEAGE,
        FEATURE_POLICY,
    )
    .unwrap();
    let compiled = ForgeCompiledArtifactIdentityV1::new_blake3_256(
        *blake3::hash(CANDIDATE).as_bytes(),
        u64::try_from(CANDIDATE.len()).unwrap(),
    )
    .unwrap();
    ForgeCompilationReceiptV1::succeeded(request, compiled).unwrap()
}

fn current_compilation() -> (
    CurrentForgeCompilationAttestationV1,
    ForgeCompilationReceiptV1,
    ForgeExecutionProfileDefinitionV1,
) {
    let provider = Provider::new();
    let resolver = Resolver::new();
    let receipt = receipt();
    let profile = execution_profile();
    let preflight = ForgeCompiledArtifactPreflightV1::verify(&receipt, &profile, CANDIDATE).unwrap();

    let algorithm = AttestationSignatureAlgorithmV1::Ed25519;
    let crypto_profile = algorithm.exact_profile_v1();
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
        signature: pseudo_signature(crypto_profile, "worker-a", &message),
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
        &provider,
        ORIGINAL_TIME,
    )
    .unwrap();
    let qualified = qualify_attestation_authority_diversity_v1(
        &verified,
        &QualifiedAttestationDiversityPolicyV1 {
            minimum_distinct_principals: 1,
            minimum_distinct_authorities: Some(1),
            allowed_principals: None,
            allowed_authorities: None,
        },
        &resolver,
        ORIGINAL_TIME,
    )
    .unwrap();
    let receipt_binding = bind_forge_compilation_attestation_v1(&preflight, &qualified, ORIGINAL_TIME).unwrap();
    let currentness = revalidate_qualified_attestation_currentness_v1(
        &qualified,
        &provider,
        &resolver,
        DECISION_TIME,
    )
    .unwrap();
    let current = bind_current_forge_compilation_attestation_v1(
        &receipt_binding,
        &currentness,
        DECISION_TIME,
    )
    .unwrap();
    (current, receipt, profile)
}

fn request(profile: &ForgeExecutionProfileDefinitionV1) -> ForgeVerificationRequestV1 {
    let commitment = profile.canonical_commitment_v1().unwrap();
    ForgeVerificationRequestV1::new(
        source_artifact(),
        "verify",
        ForgeExecutionProfileIdentityV1::new(profile.profile_id.clone(), commitment).unwrap(),
    )
    .unwrap()
}

#[test]
fn exact_current_compilation_request_profile_and_scope_pre_admit() {
    let (current, receipt, profile) = current_compilation();
    let request = request(&profile);
    let scope = ForgeCapabilityScopeV1::exact_for_request(&request, true).unwrap();

    let candidate = bind_forge_verification_pre_admission_v1(
        &current,
        &receipt,
        &profile,
        &request,
        &scope,
        ForgeExecutionModeV1::Real,
        DECISION_TIME,
    )
    .unwrap();

    assert_eq!(candidate.mode(), ForgeExecutionModeV1::Real);
    assert_eq!(candidate.decision_time_unix_s(), DECISION_TIME);
    assert_eq!(candidate.request(), &request);
    assert_eq!(candidate.scope(), &scope);
    assert_eq!(candidate.request_commitment(), request.canonical_commitment_v1().unwrap());
    assert_eq!(candidate.scope_commitment(), scope.canonical_commitment_v1().unwrap());
}

#[test]
fn re_supplied_receipt_must_be_the_authenticated_receipt() {
    let (current, _receipt, profile) = current_compilation();
    let mut other_receipt = receipt();
    other_receipt.request.source_artifact.digest[0] ^= 0xFF;
    let request = request(&profile);
    let scope = ForgeCapabilityScopeV1::exact_for_request(&request, true).unwrap();

    assert!(matches!(
        bind_forge_verification_pre_admission_v1(
            &current,
            &other_receipt,
            &profile,
            &request,
            &scope,
            ForgeExecutionModeV1::Real,
            DECISION_TIME,
        ),
        Err(ForgeVerificationPreAdmissionErrorV1::CompilationReceiptMismatch)
    ));
}

#[test]
fn verification_request_must_name_the_compiled_source_artifact() {
    let (current, receipt, profile) = current_compilation();
    let mut request = request(&profile);
    request.artifact.digest[0] ^= 0xFF;
    let scope = ForgeCapabilityScopeV1::exact_for_request(&request, true).unwrap();

    assert!(matches!(
        bind_forge_verification_pre_admission_v1(
            &current,
            &receipt,
            &profile,
            &request,
            &scope,
            ForgeExecutionModeV1::Real,
            DECISION_TIME,
        ),
        Err(ForgeVerificationPreAdmissionErrorV1::SourceArtifactMismatch)
    ));
}

#[test]
fn request_profile_identity_must_match_exact_selected_definition() {
    let (current, receipt, profile) = current_compilation();
    let mut request = request(&profile);
    request.execution_profile.profile_commitment[0] ^= 0xFF;
    let scope = ForgeCapabilityScopeV1::exact_for_request(&request, true).unwrap();

    assert!(matches!(
        bind_forge_verification_pre_admission_v1(
            &current,
            &receipt,
            &profile,
            &request,
            &scope,
            ForgeExecutionModeV1::Real,
            DECISION_TIME,
        ),
        Err(ForgeVerificationPreAdmissionErrorV1::RequestedExecutionProfileMismatch)
    ));
}

#[test]
fn real_execution_requires_scope_that_explicitly_allows_real_mode() {
    let (current, receipt, profile) = current_compilation();
    let request = request(&profile);
    let simulation_only = ForgeCapabilityScopeV1::exact_for_request(&request, false).unwrap();

    assert!(matches!(
        bind_forge_verification_pre_admission_v1(
            &current,
            &receipt,
            &profile,
            &request,
            &simulation_only,
            ForgeExecutionModeV1::Real,
            DECISION_TIME,
        ),
        Err(ForgeVerificationPreAdmissionErrorV1::CapabilityScopeMismatch)
    ));
}

#[test]
fn pre_admission_time_must_equal_current_compilation_decision_time() {
    let (current, receipt, profile) = current_compilation();
    let request = request(&profile);
    let scope = ForgeCapabilityScopeV1::exact_for_request(&request, true).unwrap();

    assert!(matches!(
        bind_forge_verification_pre_admission_v1(
            &current,
            &receipt,
            &profile,
            &request,
            &scope,
            ForgeExecutionModeV1::Real,
            DECISION_TIME + 1,
        ),
        Err(ForgeVerificationPreAdmissionErrorV1::DecisionTimeMismatch { .. })
    ));
}
