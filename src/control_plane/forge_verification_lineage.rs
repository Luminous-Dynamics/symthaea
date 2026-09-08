// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bind a current attested compilation lineage to one exact Forge verification request.
//!
//! This module closes request/profile/source substitution before any effect authority:
//!
//! ```text
//! current attested compilation lineage
//!     + exact compilation receipt re-presented now
//!     + exact verification request
//!     + exact execution profile
//!     + exact non-authoritative capability scope
//!     + explicit execution mode
//!         -> ForgeVerificationLineageBindingV1
//! ```
//!
//! The result remains non-authoritative. It proves only that the proposed verification
//! intent names the same source artifact and execution profile represented by the
//! current authenticated compilation chain, and that the supplied scope exactly
//! covers the request/mode.

use super::forge::{
    ForgeCapabilityScopeV1, ForgeExecutionModeV1, ForgeVerificationRequestV1,
};
use super::forge_compilation::ForgeCompilationReceiptV1;
use super::forge_compilation_attestation_current::RevalidatedForgeCompilationAttestationV1;
use super::forge_profile::ForgeExecutionProfileDefinitionV1;
use std::error::Error;
use std::fmt;

/// Private in-process witness joining current compilation provenance to one exact
/// verification intent. No serde derives/public constructor are provided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForgeVerificationLineageBindingV1 {
    current_compilation: RevalidatedForgeCompilationAttestationV1,
    request: ForgeVerificationRequestV1,
    request_commitment: [u8; 32],
    scope_commitment: [u8; 32],
    execution_mode: ForgeExecutionModeV1,
    bound_at_unix_s: u64,
}

impl ForgeVerificationLineageBindingV1 {
    pub fn current_compilation(&self) -> &RevalidatedForgeCompilationAttestationV1 {
        &self.current_compilation
    }

    pub fn request(&self) -> &ForgeVerificationRequestV1 {
        &self.request
    }

    pub fn request_commitment(&self) -> [u8; 32] {
        self.request_commitment
    }

    pub fn scope_commitment(&self) -> [u8; 32] {
        self.scope_commitment
    }

    pub fn execution_mode(&self) -> ForgeExecutionModeV1 {
        self.execution_mode
    }

    pub fn bound_at_unix_s(&self) -> u64 {
        self.bound_at_unix_s
    }

    pub fn natural_valid_until_unix_s(&self) -> u64 {
        self.current_compilation.natural_valid_until_unix_s()
    }

    /// Time-only helper. A later consequential effect gate must re-establish
    /// current authority/policy at its own decision instant.
    pub fn remains_within_time_bounds(&self, at_unix_s: u64) -> bool {
        self.bound_at_unix_s <= at_unix_s
            && self.current_compilation.remains_within_time_bounds(at_unix_s)
    }
}

/// Bind one exact verification request to the current authenticated compilation
/// lineage without assigning execution authority.
pub fn bind_forge_verification_lineage_v1(
    current_compilation: &RevalidatedForgeCompilationAttestationV1,
    receipt: &ForgeCompilationReceiptV1,
    request: &ForgeVerificationRequestV1,
    scope: &ForgeCapabilityScopeV1,
    execution_profile: &ForgeExecutionProfileDefinitionV1,
    execution_mode: ForgeExecutionModeV1,
    at_unix_s: u64,
) -> Result<ForgeVerificationLineageBindingV1, ForgeVerificationLineageErrorV1> {
    if at_unix_s != current_compilation.revalidated_at_unix_s() {
        return Err(ForgeVerificationLineageErrorV1::DecisionTimeMismatch {
            currentness_at_unix_s: current_compilation.revalidated_at_unix_s(),
            requested_at_unix_s: at_unix_s,
        });
    }
    if !current_compilation.remains_within_time_bounds(at_unix_s) {
        return Err(ForgeVerificationLineageErrorV1::CurrentCompilationOutsideTimeBounds);
    }

    receipt
        .validate()
        .map_err(|error| ForgeVerificationLineageErrorV1::CompilationReceipt(error.to_string()))?;
    request
        .validate()
        .map_err(|error| ForgeVerificationLineageErrorV1::VerificationRequest(error.to_string()))?;
    execution_profile
        .validate()
        .map_err(|error| ForgeVerificationLineageErrorV1::ExecutionProfile(error.to_string()))?;

    let presented_receipt_commitment = receipt
        .canonical_commitment_v1()
        .map_err(|error| ForgeVerificationLineageErrorV1::CompilationReceipt(error.to_string()))?;
    let bound_receipt_commitment = current_compilation
        .binding()
        .compilation_receipt_commitment();
    if presented_receipt_commitment != bound_receipt_commitment {
        return Err(ForgeVerificationLineageErrorV1::CompilationReceiptCommitmentMismatch);
    }

    let presented_compiled_artifact = receipt
        .compiled_artifact()
        .ok_or(ForgeVerificationLineageErrorV1::CompilationReceiptNotSuccessful)?;
    if presented_compiled_artifact != current_compilation.binding().compiled_artifact() {
        return Err(ForgeVerificationLineageErrorV1::CompiledArtifactMismatch);
    }

    if receipt.request.source_artifact != request.artifact {
        return Err(ForgeVerificationLineageErrorV1::SourceArtifactMismatch);
    }

    let exact_profile_identity = execution_profile
        .identity_v1()
        .map_err(|error| ForgeVerificationLineageErrorV1::ExecutionProfile(error.to_string()))?;
    if exact_profile_identity != request.execution_profile {
        return Err(ForgeVerificationLineageErrorV1::RequestExecutionProfileMismatch);
    }

    let exact_profile_commitment = execution_profile
        .canonical_commitment_v1()
        .map_err(|error| ForgeVerificationLineageErrorV1::ExecutionProfile(error.to_string()))?;
    if exact_profile_commitment
        != current_compilation
            .binding()
            .execution_profile_commitment()
    {
        return Err(ForgeVerificationLineageErrorV1::CompiledPreflightExecutionProfileMismatch);
    }

    let compilation_compatible = receipt
        .request
        .matches_execution_profile(execution_profile)
        .map_err(|error| ForgeVerificationLineageErrorV1::CompilationReceipt(error.to_string()))?;
    if !compilation_compatible {
        return Err(ForgeVerificationLineageErrorV1::CompilationExecutionProfileMismatch);
    }

    let scope_matches = scope
        .matches_request(request, execution_mode)
        .map_err(|error| ForgeVerificationLineageErrorV1::CapabilityScope(error.to_string()))?;
    if !scope_matches {
        return Err(ForgeVerificationLineageErrorV1::CapabilityScopeMismatch);
    }

    let request_commitment = request
        .canonical_commitment_v1()
        .map_err(|error| ForgeVerificationLineageErrorV1::VerificationRequest(error.to_string()))?;
    let scope_commitment = scope
        .canonical_commitment_v1()
        .map_err(|error| ForgeVerificationLineageErrorV1::CapabilityScope(error.to_string()))?;

    Ok(ForgeVerificationLineageBindingV1 {
        current_compilation: current_compilation.clone(),
        request: request.clone(),
        request_commitment,
        scope_commitment,
        execution_mode,
        bound_at_unix_s: at_unix_s,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgeVerificationLineageErrorV1 {
    DecisionTimeMismatch {
        currentness_at_unix_s: u64,
        requested_at_unix_s: u64,
    },
    CurrentCompilationOutsideTimeBounds,
    CompilationReceipt(String),
    VerificationRequest(String),
    ExecutionProfile(String),
    CapabilityScope(String),
    CompilationReceiptCommitmentMismatch,
    CompilationReceiptNotSuccessful,
    CompiledArtifactMismatch,
    SourceArtifactMismatch,
    RequestExecutionProfileMismatch,
    CompiledPreflightExecutionProfileMismatch,
    CompilationExecutionProfileMismatch,
    CapabilityScopeMismatch,
}

impl fmt::Display for ForgeVerificationLineageErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DecisionTimeMismatch {
                currentness_at_unix_s,
                requested_at_unix_s,
            } => write!(
                f,
                "Forge verification lineage decision time {requested_at_unix_s} differs from currentness revalidation time {currentness_at_unix_s}"
            ),
            Self::CurrentCompilationOutsideTimeBounds => write!(
                f,
                "current Forge compilation attestation is outside its time bounds"
            ),
            Self::CompilationReceipt(error) => {
                write!(f, "invalid Forge compilation receipt: {error}")
            }
            Self::VerificationRequest(error) => {
                write!(f, "invalid Forge verification request: {error}")
            }
            Self::ExecutionProfile(error) => {
                write!(f, "invalid Forge execution profile: {error}")
            }
            Self::CapabilityScope(error) => {
                write!(f, "invalid Forge capability scope: {error}")
            }
            Self::CompilationReceiptCommitmentMismatch => write!(
                f,
                "presented Forge compilation receipt differs from the exact attested/preflighted receipt"
            ),
            Self::CompilationReceiptNotSuccessful => {
                write!(f, "presented Forge compilation receipt is not successful")
            }
            Self::CompiledArtifactMismatch => write!(
                f,
                "presented Forge compiled artifact differs from the exact preflighted artifact"
            ),
            Self::SourceArtifactMismatch => write!(
                f,
                "Forge verification request source artifact differs from the source artifact compiled by the attested receipt"
            ),
            Self::RequestExecutionProfileMismatch => write!(
                f,
                "Forge verification request does not name the exact selected execution profile"
            ),
            Self::CompiledPreflightExecutionProfileMismatch => write!(
                f,
                "selected execution profile differs from the execution profile used by compiled-artifact preflight"
            ),
            Self::CompilationExecutionProfileMismatch => write!(
                f,
                "Forge compilation receipt is incompatible with the selected execution profile"
            ),
            Self::CapabilityScopeMismatch => write!(
                f,
                "Forge capability scope does not exactly match the verification request and execution mode"
            ),
        }
    }
}

impl Error for ForgeVerificationLineageErrorV1 {}

#[cfg(test)]
mod tests {
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
    use crate::control_plane::forge::{
        ForgeArtifactIdentityV1, ForgeExecutionProfileIdentityV1, FORGE_PROTOCOL_VERSION,
    };
    use crate::control_plane::forge_compilation::{
        ForgeCompilationProfileIdentityV1, ForgeCompilationRequestV1,
        ForgeCompiledArtifactIdentityV1,
    };
    use crate::control_plane::forge_compilation_attestation::{
        FORGE_COMPILATION_RECEIPT_ATTESTATION_SUBJECT_DOMAIN_V1,
        bind_forge_compilation_attestation_v1,
    };
    use crate::control_plane::forge_compilation_attestation_current::{
        revalidate_forge_compilation_attestation_current_v1,
    };
    use crate::control_plane::forge_compiled_artifact_preflight::ForgeCompiledArtifactPreflightV1;
    use crate::control_plane::forge_profile::{
        ForgeDeterminismPolicyV1, ForgeImportPolicyV1, ForgeInterruptionPolicyV1,
        ForgeRuntimeIdentityV1, ForgeVerifierAbiV1,
    };
    use std::collections::{BTreeSet};

    const TRUST_GENERATION: [u8; 32] = [0x21; 32];
    const BINDING_GENERATION: [u8; 32] = [0x31; 32];
    const RUNTIME_LINEAGE: [u8; 32] = [0x41; 32];
    const FEATURE_POLICY: [u8; 32] = [0x51; 32];
    const SOURCE_DIGEST: [u8; 32] = [0x61; 32];
    const COMPILE_PROFILE: [u8; 32] = [0x71; 32];
    const CANDIDATE: &[u8] = b"forge-verification-lineage-precompiled-v1";
    const QUALIFIED_AT: u64 = 200;
    const CURRENT_AT: u64 = 300;

    #[derive(Clone)]
    struct Provider {
        trust_snapshot: AttestationTrustSnapshotV1,
        key: AttestationTrustedKeyV1,
        authority_snapshot: QualifiedAttestationAuthoritySnapshotV1,
        authority_binding: QualifiedAttestationSignerBindingV1,
    }

    impl Provider {
        fn new() -> Self {
            Self {
                trust_snapshot: AttestationTrustSnapshotV1 {
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
                authority_snapshot: QualifiedAttestationAuthoritySnapshotV1 {
                    generation_commitment: BINDING_GENERATION,
                    valid_from_unix_s: 1,
                    valid_until_unix_s: 850,
                    identity_namespaces: AttestationIdentityNamespaceSetV1::new(
                        AttestationIdentityNamespaceV1::new("principals-v1", [0x81; 32]).unwrap(),
                        Some(
                            AttestationIdentityNamespaceV1::new("authorities-v1", [0x91; 32])
                                .unwrap(),
                        ),
                    )
                    .unwrap(),
                },
                authority_binding: QualifiedAttestationSignerBindingV1 {
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

    impl AttestationTrustVerifierV1 for Provider {
        fn current_snapshot(&self) -> Result<AttestationTrustSnapshotV1, String> {
            Ok(self.trust_snapshot.clone())
        }

        fn key_record(
            &self,
            generation: [u8; 32],
            algorithm: AttestationSignatureAlgorithmV1,
            key_id: &str,
        ) -> Result<Option<AttestationTrustedKeyV1>, String> {
            if generation == self.trust_snapshot.generation_commitment
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
            if generation != self.trust_snapshot.generation_commitment
                || profile != AttestationSignatureAlgorithmV1::Ed25519.exact_profile_v1()
                || key_id != self.key.key_id
            {
                return Ok(false);
            }
            Ok(signature == pseudo_signature(profile, key_id, message))
        }
    }

    impl QualifiedAttestationAuthorityResolverV1 for Provider {
        fn current_snapshot(&self) -> Result<QualifiedAttestationAuthoritySnapshotV1, String> {
            Ok(self.authority_snapshot.clone())
        }

        fn signer_binding(
            &self,
            generation: [u8; 32],
            algorithm: AttestationSignatureAlgorithmV1,
            key_id: &str,
        ) -> Result<Option<QualifiedAttestationSignerBindingV1>, String> {
            if generation == self.authority_snapshot.generation_commitment
                && algorithm == self.authority_binding.algorithm
                && key_id == self.authority_binding.key_id
            {
                Ok(Some(self.authority_binding.clone()))
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
            hasher.update(b"test-only-forge-verification-lineage\0");
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
            ForgeArtifactIdentityV1::new_blake3_256(SOURCE_DIGEST, 128).unwrap(),
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

    fn current(provider: &Provider) -> RevalidatedForgeCompilationAttestationV1 {
        let receipt = receipt();
        let profile = execution_profile();
        let preflight = ForgeCompiledArtifactPreflightV1::verify(&receipt, &profile, CANDIDATE)
            .unwrap();
        let subject = AttestationSubjectV1::new(
            FORGE_COMPILATION_RECEIPT_ATTESTATION_SUBJECT_DOMAIN_V1,
            receipt.canonical_commitment_v1().unwrap(),
        )
        .unwrap();
        let algorithm = AttestationSignatureAlgorithmV1::Ed25519;
        let exact_profile = algorithm.exact_profile_v1();
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
            signature: pseudo_signature(exact_profile, "worker-a", &message),
        });
        let crypto = verify_current_attestation_crypto_bound_profile_v1(
            &envelope,
            &AttestationPolicyV1 {
                minimum_valid_signatures: 1,
                maximum_signatures: 2,
                maximum_signature_bytes: 4 * 1024,
                maximum_key_id_bytes: 128,
                required_algorithms: BTreeSet::from([algorithm]),
                allowed_key_ids: Some(BTreeSet::from(["worker-a".into()])),
            },
            provider,
            QUALIFIED_AT,
        )
        .unwrap();
        let qualified = qualify_attestation_authority_diversity_v1(
            &crypto,
            &QualifiedAttestationDiversityPolicyV1 {
                minimum_distinct_principals: 1,
                minimum_distinct_authorities: Some(1),
                allowed_principals: None,
                allowed_authorities: None,
            },
            provider,
            QUALIFIED_AT,
        )
        .unwrap();
        let bound = bind_forge_compilation_attestation_v1(&preflight, &qualified, QUALIFIED_AT)
            .unwrap();
        revalidate_forge_compilation_attestation_current_v1(
            &bound,
            provider,
            provider,
            CURRENT_AT,
        )
        .unwrap()
    }

    fn verification_request(profile: &ForgeExecutionProfileDefinitionV1) -> ForgeVerificationRequestV1 {
        ForgeVerificationRequestV1::new(
            ForgeArtifactIdentityV1::new_blake3_256(SOURCE_DIGEST, 128).unwrap(),
            "verify",
            profile.identity_v1().unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn exact_request_scope_and_compilation_lineage_bind() {
        let provider = Provider::new();
        let profile = execution_profile();
        let request = verification_request(&profile);
        let scope = ForgeCapabilityScopeV1::exact_for_request(&request, true).unwrap();
        let witness = bind_forge_verification_lineage_v1(
            &current(&provider),
            &receipt(),
            &request,
            &scope,
            &profile,
            ForgeExecutionModeV1::Real,
            CURRENT_AT,
        )
        .unwrap();

        assert_eq!(witness.request(), &request);
        assert_eq!(
            witness.request_commitment(),
            request.canonical_commitment_v1().unwrap()
        );
        assert_eq!(
            witness.scope_commitment(),
            scope.canonical_commitment_v1().unwrap()
        );
        assert_eq!(witness.execution_mode(), ForgeExecutionModeV1::Real);
    }

    #[test]
    fn different_receipt_or_source_artifact_fails_closed() {
        let provider = Provider::new();
        let current = current(&provider);
        let profile = execution_profile();
        let request = verification_request(&profile);
        let scope = ForgeCapabilityScopeV1::exact_for_request(&request, true).unwrap();

        let mut other_receipt = receipt();
        other_receipt.request.compilation_profile.profile_commitment[0] ^= 0xFF;
        assert!(matches!(
            bind_forge_verification_lineage_v1(
                &current,
                &other_receipt,
                &request,
                &scope,
                &profile,
                ForgeExecutionModeV1::Real,
                CURRENT_AT,
            ),
            Err(ForgeVerificationLineageErrorV1::CompilationReceiptCommitmentMismatch)
        ));

        let mut wrong_request = request.clone();
        wrong_request.artifact.digest[0] ^= 0xFF;
        let wrong_scope = ForgeCapabilityScopeV1::exact_for_request(&wrong_request, true).unwrap();
        assert!(matches!(
            bind_forge_verification_lineage_v1(
                &current,
                &receipt(),
                &wrong_request,
                &wrong_scope,
                &profile,
                ForgeExecutionModeV1::Real,
                CURRENT_AT,
            ),
            Err(ForgeVerificationLineageErrorV1::SourceArtifactMismatch)
        ));
    }

    #[test]
    fn profile_substitution_fails_closed() {
        let provider = Provider::new();
        let current = current(&provider);
        let profile = execution_profile();
        let request = verification_request(&profile);
        let scope = ForgeCapabilityScopeV1::exact_for_request(&request, true).unwrap();

        let mut changed = profile.clone();
        changed.fuel += 1;
        assert!(matches!(
            bind_forge_verification_lineage_v1(
                &current,
                &receipt(),
                &request,
                &scope,
                &changed,
                ForgeExecutionModeV1::Real,
                CURRENT_AT,
            ),
            Err(ForgeVerificationLineageErrorV1::RequestExecutionProfileMismatch)
        ));
    }

    #[test]
    fn scope_or_mode_substitution_fails_closed() {
        let provider = Provider::new();
        let current = current(&provider);
        let profile = execution_profile();
        let request = verification_request(&profile);
        let simulation_only = ForgeCapabilityScopeV1::exact_for_request(&request, false).unwrap();

        assert!(matches!(
            bind_forge_verification_lineage_v1(
                &current,
                &receipt(),
                &request,
                &simulation_only,
                &profile,
                ForgeExecutionModeV1::Real,
                CURRENT_AT,
            ),
            Err(ForgeVerificationLineageErrorV1::CapabilityScopeMismatch)
        ));

        bind_forge_verification_lineage_v1(
            &current,
            &receipt(),
            &request,
            &simulation_only,
            &profile,
            ForgeExecutionModeV1::Simulated,
            CURRENT_AT,
        )
        .unwrap();
    }

    #[test]
    fn lineage_binding_must_share_currentness_decision_time() {
        let provider = Provider::new();
        let current = current(&provider);
        let profile = execution_profile();
        let request = verification_request(&profile);
        let scope = ForgeCapabilityScopeV1::exact_for_request(&request, true).unwrap();

        assert!(matches!(
            bind_forge_verification_lineage_v1(
                &current,
                &receipt(),
                &request,
                &scope,
                &profile,
                ForgeExecutionModeV1::Real,
                CURRENT_AT + 1,
            ),
            Err(ForgeVerificationLineageErrorV1::DecisionTimeMismatch { .. })
        ));
    }
}
