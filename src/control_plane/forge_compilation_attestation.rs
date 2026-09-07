// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Exact binding between a structurally preflighted Forge compilation receipt
//! and a namespace-qualified, cryptographically bound worker attestation.
//!
//! This module intentionally stops before effect authority:
//!
//! ```text
//! ForgeCompiledArtifactPreflightV1
//!     + QualifiedAttestationAuthorityDiversityV1
//!     + exact receipt subject binding
//!         -> ForgeCompilationAttestationBindingV1
//!
//! ForgeCompilationAttestationBindingV1
//!     != Forge execution authority
//!     != Xenia/local effect authority
//!     != permission to call unsafe Wasmtime deserialization
//! ```

use super::attestation_qualified_authority::QualifiedAttestationAuthorityDiversityV1;
use super::forge_compilation::ForgeCompiledArtifactIdentityV1;
use super::forge_compiled_artifact_preflight::ForgeCompiledArtifactPreflightV1;
use std::error::Error;
use std::fmt;

/// Semantic attestation subject domain for the exact canonical Forge compilation
/// receipt commitment.
///
/// The compilation protocol's hash-domain constant carries a trailing NUL for
/// canonical byte framing; attestation subjects use the corresponding semantic
/// UTF-8 domain without that framing byte.
pub const FORGE_COMPILATION_RECEIPT_ATTESTATION_SUBJECT_DOMAIN_V1: &str =
    "symthaea.forge.compilation-receipt.v1";

/// Private in-process witness that the exact receipt used by a structural
/// preflight is the exact subject of a qualified current worker attestation.
///
/// Fields are private and there are no serde derives/public constructors. This
/// is evidence composition, not a portable authority token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForgeCompilationAttestationBindingV1 {
    preflight: ForgeCompiledArtifactPreflightV1,
    qualified_attestation: QualifiedAttestationAuthorityDiversityV1,
}

impl ForgeCompilationAttestationBindingV1 {
    pub fn preflight(&self) -> &ForgeCompiledArtifactPreflightV1 {
        &self.preflight
    }

    pub fn qualified_attestation(&self) -> &QualifiedAttestationAuthorityDiversityV1 {
        &self.qualified_attestation
    }

    pub fn compiled_artifact(&self) -> &ForgeCompiledArtifactIdentityV1 {
        self.preflight.compiled_artifact()
    }

    pub fn compilation_receipt_commitment(&self) -> [u8; 32] {
        self.preflight.compilation_receipt_commitment()
    }

    pub fn execution_profile_commitment(&self) -> [u8; 32] {
        self.preflight.execution_profile_commitment()
    }

    pub fn attestation_envelope_commitment(&self) -> [u8; 32] {
        self.qualified_attestation.envelope_commitment()
    }

    pub fn trust_generation_commitment(&self) -> [u8; 32] {
        self.qualified_attestation.trust_generation_commitment()
    }

    pub fn authority_binding_generation_commitment(&self) -> [u8; 32] {
        self.qualified_attestation
            .authority_binding_generation_commitment()
    }

    pub fn identity_namespace_set_commitment(&self) -> [u8; 32] {
        self.qualified_attestation
            .identity_namespace_set_commitment()
    }

    pub fn evaluation_time_unix_s(&self) -> u64 {
        self.qualified_attestation.evaluation_time_unix_s()
    }

    pub fn natural_valid_until_unix_s(&self) -> u64 {
        self.qualified_attestation.natural_valid_until_unix_s()
    }

    /// Time-only check. Point-of-use effect admission must still revalidate the
    /// current trust and authority-binding generations.
    pub fn remains_within_time_bounds(&self, at_unix_s: u64) -> bool {
        self.qualified_attestation
            .remains_within_time_bounds(at_unix_s)
    }
}

/// Bind qualified attestation evidence to the exact compilation receipt already
/// retained by one structural Forge preflight.
///
/// The binding must occur at the same logical decision instant as the qualified
/// attestation transaction. No claim is made that either trust generation remains
/// current after this function returns.
pub fn bind_forge_compilation_attestation_v1(
    preflight: &ForgeCompiledArtifactPreflightV1,
    qualified_attestation: &QualifiedAttestationAuthorityDiversityV1,
    evaluation_time_unix_s: u64,
) -> Result<ForgeCompilationAttestationBindingV1, ForgeCompilationAttestationBindingErrorV1> {
    if evaluation_time_unix_s != qualified_attestation.evaluation_time_unix_s() {
        return Err(
            ForgeCompilationAttestationBindingErrorV1::EvaluationTimeMismatch {
                qualified_at_unix_s: qualified_attestation.evaluation_time_unix_s(),
                requested_at_unix_s: evaluation_time_unix_s,
            },
        );
    }
    if !qualified_attestation.remains_within_time_bounds(evaluation_time_unix_s) {
        return Err(
            ForgeCompilationAttestationBindingErrorV1::QualifiedAttestationOutsideTimeBounds,
        );
    }

    let subject = qualified_attestation.subject();
    if subject.domain != FORGE_COMPILATION_RECEIPT_ATTESTATION_SUBJECT_DOMAIN_V1 {
        return Err(
            ForgeCompilationAttestationBindingErrorV1::SubjectDomainMismatch {
                actual: subject.domain.clone(),
            },
        );
    }
    if subject.commitment != preflight.compilation_receipt_commitment() {
        return Err(
            ForgeCompilationAttestationBindingErrorV1::CompilationReceiptCommitmentMismatch,
        );
    }

    Ok(ForgeCompilationAttestationBindingV1 {
        preflight: preflight.clone(),
        qualified_attestation: qualified_attestation.clone(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgeCompilationAttestationBindingErrorV1 {
    EvaluationTimeMismatch {
        qualified_at_unix_s: u64,
        requested_at_unix_s: u64,
    },
    QualifiedAttestationOutsideTimeBounds,
    SubjectDomainMismatch {
        actual: String,
    },
    CompilationReceiptCommitmentMismatch,
}

impl fmt::Display for ForgeCompilationAttestationBindingErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EvaluationTimeMismatch {
                qualified_at_unix_s,
                requested_at_unix_s,
            } => write!(
                f,
                "Forge compilation attestation binding time {requested_at_unix_s} differs from qualified attestation time {qualified_at_unix_s}"
            ),
            Self::QualifiedAttestationOutsideTimeBounds => write!(
                f,
                "qualified Forge compilation attestation is outside its time bounds"
            ),
            Self::SubjectDomainMismatch { actual } => write!(
                f,
                "Forge compilation attestation subject domain mismatch: expected {}, got {actual}",
                FORGE_COMPILATION_RECEIPT_ATTESTATION_SUBJECT_DOMAIN_V1
            ),
            Self::CompilationReceiptCommitmentMismatch => write!(
                f,
                "qualified attestation subject does not commit to the exact Forge compilation receipt used by preflight"
            ),
        }
    }
}

impl Error for ForgeCompilationAttestationBindingErrorV1 {}

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
    use crate::control_plane::forge::{ForgeArtifactIdentityV1, FORGE_PROTOCOL_VERSION};
    use crate::control_plane::forge_compilation::{
        FORGE_COMPILATION_RECEIPT_DOMAIN_V1, ForgeCompilationProfileIdentityV1,
        ForgeCompilationReceiptV1, ForgeCompilationRequestV1,
    };
    use crate::control_plane::forge_profile::{
        ForgeDeterminismPolicyV1, ForgeExecutionProfileDefinitionV1, ForgeImportPolicyV1,
        ForgeInterruptionPolicyV1, ForgeRuntimeIdentityV1, ForgeVerifierAbiV1,
    };
    use std::collections::{BTreeMap, BTreeSet};

    const TRUST_GENERATION: [u8; 32] = [0x21; 32];
    const BINDING_GENERATION: [u8; 32] = [0x31; 32];
    const RUNTIME_LINEAGE: [u8; 32] = [0x41; 32];
    const FEATURE_POLICY: [u8; 32] = [0x51; 32];
    const CANDIDATE: &[u8] = b"forge-attested-precompiled-artifact-v1";
    const EVALUATION_TIME: u64 = 200;
    const WRONG_SUBJECT_DOMAIN: &str = "symthaea.forge.not-a-compilation-receipt.v1";

    fn request() -> ForgeCompilationRequestV1 {
        ForgeCompilationRequestV1::new(
            ForgeArtifactIdentityV1::new_blake3_256([0x61; 32], 128).unwrap(),
            ForgeCompilationProfileIdentityV1::new("compile-v1", [0x71; 32]).unwrap(),
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

    fn preflight(profile: &ForgeExecutionProfileDefinitionV1) -> ForgeCompiledArtifactPreflightV1 {
        ForgeCompiledArtifactPreflightV1::verify(&receipt(), profile, CANDIDATE).unwrap()
    }

    struct CryptoProvider {
        snapshot: AttestationTrustSnapshotV1,
        key: AttestationTrustedKeyV1,
    }

    impl CryptoProvider {
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
                    // Trust both test domains so the wrong-domain regression reaches
                    // this Forge-specific binding layer rather than failing earlier.
                    allowed_subject_domains: BTreeSet::from([
                        FORGE_COMPILATION_RECEIPT_ATTESTATION_SUBJECT_DOMAIN_V1.to_string(),
                        WRONG_SUBJECT_DOMAIN.to_string(),
                    ]),
                },
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
            // This erased verification path must not qualify consequential tests.
            Ok(false)
        }
    }

    impl ExactAttestationTrustVerifierV1 for CryptoProvider {
        fn supports_exact_profile_v1(
            &self,
            profile: AttestationCryptoProfileV1,
        ) -> Result<bool, String> {
            Ok(profile == AttestationSignatureAlgorithmV1::Ed25519.exact_profile_v1())
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
            hasher.update(b"test-only-forge-compilation-attestation\0");
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

    fn qualified_for_subject(
        subject: AttestationSubjectV1,
    ) -> QualifiedAttestationAuthorityDiversityV1 {
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

        let crypto_policy = AttestationPolicyV1 {
            minimum_valid_signatures: 1,
            maximum_signatures: 2,
            maximum_signature_bytes: 4 * 1024,
            maximum_key_id_bytes: 128,
            required_algorithms: BTreeSet::from([algorithm]),
            allowed_key_ids: Some(BTreeSet::from(["worker-a".to_string()])),
        };
        let verified = verify_current_attestation_crypto_bound_profile_v1(
            &envelope,
            &crypto_policy,
            &CryptoProvider::new(),
            EVALUATION_TIME,
        )
        .unwrap();

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

        let namespaces = AttestationIdentityNamespaceSetV1::new(
            AttestationIdentityNamespaceV1::new("forge-builders-v1", [0x81; 32]).unwrap(),
            None,
        )
        .unwrap();
        let resolver = Resolver {
            snapshot: QualifiedAttestationAuthoritySnapshotV1 {
                generation_commitment: BINDING_GENERATION,
                valid_from_unix_s: 1,
                valid_until_unix_s: 850,
                identity_namespaces: namespaces,
            },
            binding: QualifiedAttestationSignerBindingV1 {
                generation_commitment: BINDING_GENERATION,
                algorithm,
                key_id: "worker-a".into(),
                principal_id: "builder-a".into(),
                authority_id: None,
                valid_from_unix_s: 10,
                valid_until_unix_s: 700,
            },
        };
        let diversity_policy = QualifiedAttestationDiversityPolicyV1 {
            minimum_distinct_principals: 1,
            minimum_distinct_authorities: None,
            allowed_principals: None,
            allowed_authorities: None,
        };
        qualify_attestation_authority_diversity_v1(
            &verified,
            &diversity_policy,
            &resolver,
            EVALUATION_TIME,
        )
        .unwrap()
    }

    fn qualified_for_receipt(
        receipt: &ForgeCompilationReceiptV1,
    ) -> QualifiedAttestationAuthorityDiversityV1 {
        qualified_for_subject(
            AttestationSubjectV1::new(
                FORGE_COMPILATION_RECEIPT_ATTESTATION_SUBJECT_DOMAIN_V1,
                receipt.canonical_commitment_v1().unwrap(),
            )
            .unwrap(),
        )
    }

    #[test]
    fn semantic_attestation_domain_matches_compilation_receipt_hash_domain() {
        let mut expected = FORGE_COMPILATION_RECEIPT_ATTESTATION_SUBJECT_DOMAIN_V1
            .as_bytes()
            .to_vec();
        expected.push(0);
        assert_eq!(expected.as_slice(), FORGE_COMPILATION_RECEIPT_DOMAIN_V1);
    }

    #[test]
    fn exact_receipt_subject_binds_to_preflight() {
        let receipt = receipt();
        let profile = execution_profile();
        let preflight =
            ForgeCompiledArtifactPreflightV1::verify(&receipt, &profile, CANDIDATE).unwrap();
        let qualified = qualified_for_receipt(&receipt);

        let bound = bind_forge_compilation_attestation_v1(
            &preflight,
            &qualified,
            EVALUATION_TIME,
        )
        .unwrap();

        assert_eq!(
            bound.compilation_receipt_commitment(),
            receipt.canonical_commitment_v1().unwrap()
        );
        assert_eq!(bound.compiled_artifact(), receipt.compiled_artifact().unwrap());
        assert_eq!(bound.trust_generation_commitment(), TRUST_GENERATION);
        assert_eq!(
            bound.authority_binding_generation_commitment(),
            BINDING_GENERATION
        );
        assert_eq!(bound.natural_valid_until_unix_s(), 700);
    }

    #[test]
    fn wrong_subject_domain_fails_at_forge_binding_layer() {
        let receipt = receipt();
        let preflight = preflight(&execution_profile());
        let qualified = qualified_for_subject(
            AttestationSubjectV1::new(
                WRONG_SUBJECT_DOMAIN,
                receipt.canonical_commitment_v1().unwrap(),
            )
            .unwrap(),
        );

        assert!(matches!(
            bind_forge_compilation_attestation_v1(
                &preflight,
                &qualified,
                EVALUATION_TIME,
            ),
            Err(ForgeCompilationAttestationBindingErrorV1::SubjectDomainMismatch { .. })
        ));
    }

    #[test]
    fn another_receipt_commitment_cannot_authenticate_this_preflight() {
        let receipt = receipt();
        let preflight = preflight(&execution_profile());
        let qualified = qualified_for_subject(
            AttestationSubjectV1::new(
                FORGE_COMPILATION_RECEIPT_ATTESTATION_SUBJECT_DOMAIN_V1,
                [0xA1; 32],
            )
            .unwrap(),
        );

        assert_ne!(
            qualified.subject().commitment,
            receipt.canonical_commitment_v1().unwrap()
        );
        assert!(matches!(
            bind_forge_compilation_attestation_v1(
                &preflight,
                &qualified,
                EVALUATION_TIME,
            ),
            Err(
                ForgeCompilationAttestationBindingErrorV1::CompilationReceiptCommitmentMismatch
            )
        ));
    }

    #[test]
    fn binding_is_same_decision_time_only() {
        let receipt = receipt();
        let qualified = qualified_for_receipt(&receipt);
        let preflight = preflight(&execution_profile());

        assert!(matches!(
            bind_forge_compilation_attestation_v1(
                &preflight,
                &qualified,
                EVALUATION_TIME + 1,
            ),
            Err(ForgeCompilationAttestationBindingErrorV1::EvaluationTimeMismatch { .. })
        ));
    }

    #[test]
    fn worker_attestation_authenticates_receipt_not_execution_only_limits() {
        let receipt = receipt();
        let qualified = qualified_for_receipt(&receipt);
        let profile_a = execution_profile();
        let mut profile_b = execution_profile();
        profile_b.fuel += 1;

        let a = bind_forge_compilation_attestation_v1(
            &ForgeCompiledArtifactPreflightV1::verify(&receipt, &profile_a, CANDIDATE).unwrap(),
            &qualified,
            EVALUATION_TIME,
        )
        .unwrap();
        let b = bind_forge_compilation_attestation_v1(
            &ForgeCompiledArtifactPreflightV1::verify(&receipt, &profile_b, CANDIDATE).unwrap(),
            &qualified,
            EVALUATION_TIME,
        )
        .unwrap();

        assert_eq!(
            a.compilation_receipt_commitment(),
            b.compilation_receipt_commitment()
        );
        assert_ne!(
            a.execution_profile_commitment(),
            b.execution_profile_commitment()
        );
    }
}
