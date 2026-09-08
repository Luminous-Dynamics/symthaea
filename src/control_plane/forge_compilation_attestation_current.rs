// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Point-of-use currentness revalidation for attested Forge compilation receipts.
//!
//! This module does not re-run cryptographic signature verification. Instead it
//! proves that a previously qualified exact-profile attestation still refers to
//! the exact current trust and authority-binding generations, and that every
//! retained signer/key/identity binding remains active and time-valid at the
//! point of use.
//!
//! ```text
//! ForgeCompilationAttestationBindingV1
//!     + unchanged current trust generation
//!     + unchanged current authority-binding generation
//!     + current key lifecycle/domain/time checks
//!     + current namespace-qualified identity binding checks
//!         -> RevalidatedForgeCompilationAttestationV1
//!
//! generation drift
//!     -> fail closed
//!     -> rebuild the full cryptographic + identity qualification chain
//! ```

use super::attestation::{
    AttestationKeyLifecycleV1, AttestationTrustVerifierV1,
};
use super::attestation_qualified_authority::QualifiedAttestationAuthorityResolverV1;
use super::forge_compilation_attestation::ForgeCompilationAttestationBindingV1;
use std::error::Error;
use std::fmt;

/// Positive in-process point-of-use witness that the previously qualified Forge
/// compilation attestation remains current under the exact same committed trust
/// and identity-binding generations.
///
/// Private fields and no serde derives prevent this from becoming a portable
/// authority token. It still does not grant Forge execution permission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RevalidatedForgeCompilationAttestationV1 {
    binding: ForgeCompilationAttestationBindingV1,
    revalidated_at_unix_s: u64,
    natural_valid_until_unix_s: u64,
}

impl RevalidatedForgeCompilationAttestationV1 {
    pub fn binding(&self) -> &ForgeCompilationAttestationBindingV1 {
        &self.binding
    }

    pub fn revalidated_at_unix_s(&self) -> u64 {
        self.revalidated_at_unix_s
    }

    /// Earliest currently observed natural expiry across the original binding,
    /// current trust snapshot, current authority snapshot, all current signer key
    /// records, and all current authority bindings.
    pub fn natural_valid_until_unix_s(&self) -> u64 {
        self.natural_valid_until_unix_s
    }

    /// Time-only check. Generation currentness must still be revalidated again at
    /// a later consequential point of use.
    pub fn remains_within_time_bounds(&self, at_unix_s: u64) -> bool {
        self.revalidated_at_unix_s <= at_unix_s
            && at_unix_s < self.natural_valid_until_unix_s
    }
}

/// Revalidate current trust/key and authority-binding state without pretending
/// that this is a new cryptographic signature decision.
///
/// V1 is deliberately conservative: any generation change invalidates the old
/// positive chain even if the individual key/identity records happen to look the
/// same. Callers must rebuild the full exact-profile attestation qualification
/// against the new generation instead of carrying old authority forward.
pub fn revalidate_forge_compilation_attestation_current_v1(
    binding: &ForgeCompilationAttestationBindingV1,
    trust_provider: &dyn AttestationTrustVerifierV1,
    authority_resolver: &dyn QualifiedAttestationAuthorityResolverV1,
    at_unix_s: u64,
) -> Result<RevalidatedForgeCompilationAttestationV1, ForgeCompilationAttestationCurrentErrorV1> {
    if !binding.remains_within_time_bounds(at_unix_s) {
        return Err(ForgeCompilationAttestationCurrentErrorV1::BindingOutsideTimeBounds);
    }

    let qualified = binding.qualified_attestation();
    let subject = qualified.subject();

    let trust_snapshot = trust_provider
        .current_snapshot()
        .map_err(ForgeCompilationAttestationCurrentErrorV1::TrustProvider)?;
    trust_snapshot
        .validate()
        .map_err(|error| ForgeCompilationAttestationCurrentErrorV1::TrustSnapshot(error.to_string()))?;

    if trust_snapshot.generation_commitment != binding.trust_generation_commitment() {
        return Err(ForgeCompilationAttestationCurrentErrorV1::TrustGenerationDrift);
    }
    if at_unix_s < trust_snapshot.valid_from_unix_s {
        return Err(ForgeCompilationAttestationCurrentErrorV1::TrustSnapshotNotYetValid);
    }
    if at_unix_s >= trust_snapshot.valid_until_unix_s {
        return Err(ForgeCompilationAttestationCurrentErrorV1::TrustSnapshotExpired);
    }

    let authority_snapshot = authority_resolver
        .current_snapshot()
        .map_err(ForgeCompilationAttestationCurrentErrorV1::AuthorityResolver)?;
    authority_snapshot
        .validate()
        .map_err(|error| ForgeCompilationAttestationCurrentErrorV1::AuthoritySnapshot(error.to_string()))?;

    if authority_snapshot.generation_commitment
        != binding.authority_binding_generation_commitment()
    {
        return Err(ForgeCompilationAttestationCurrentErrorV1::AuthorityBindingGenerationDrift);
    }
    if at_unix_s < authority_snapshot.valid_from_unix_s {
        return Err(ForgeCompilationAttestationCurrentErrorV1::AuthoritySnapshotNotYetValid);
    }
    if at_unix_s >= authority_snapshot.valid_until_unix_s {
        return Err(ForgeCompilationAttestationCurrentErrorV1::AuthoritySnapshotExpired);
    }

    let current_namespace_commitment = authority_snapshot
        .identity_namespaces
        .canonical_commitment_v1()
        .map_err(|error| ForgeCompilationAttestationCurrentErrorV1::IdentityNamespace(error.to_string()))?;
    if current_namespace_commitment != binding.identity_namespace_set_commitment() {
        return Err(ForgeCompilationAttestationCurrentErrorV1::IdentityNamespaceDrift);
    }

    let mut natural_valid_until_unix_s = binding
        .natural_valid_until_unix_s()
        .min(trust_snapshot.valid_until_unix_s)
        .min(authority_snapshot.valid_until_unix_s);

    for entry in qualified.entries() {
        let key = trust_provider
            .key_record(
                trust_snapshot.generation_commitment,
                entry.algorithm,
                &entry.key_id,
            )
            .map_err(ForgeCompilationAttestationCurrentErrorV1::TrustProvider)?
            .ok_or_else(|| ForgeCompilationAttestationCurrentErrorV1::MissingCurrentKey {
                key_id: entry.key_id.clone(),
            })?;
        key.validate()
            .map_err(|error| ForgeCompilationAttestationCurrentErrorV1::CurrentKey(error.to_string()))?;
        if key.algorithm != entry.algorithm || key.key_id != entry.key_id {
            return Err(ForgeCompilationAttestationCurrentErrorV1::CurrentKeyIdentityMismatch {
                expected_key_id: entry.key_id.clone(),
                actual_key_id: key.key_id,
            });
        }
        if !key.allowed_subject_domains.contains(&subject.domain) {
            return Err(ForgeCompilationAttestationCurrentErrorV1::CurrentKeyDomainDenied {
                key_id: entry.key_id.clone(),
                domain: subject.domain.clone(),
            });
        }
        match key.lifecycle {
            AttestationKeyLifecycleV1::Active => {}
            AttestationKeyLifecycleV1::Retired => {
                return Err(ForgeCompilationAttestationCurrentErrorV1::CurrentKeyRetired(
                    entry.key_id.clone(),
                ));
            }
            AttestationKeyLifecycleV1::Revoked => {
                return Err(ForgeCompilationAttestationCurrentErrorV1::CurrentKeyRevoked(
                    entry.key_id.clone(),
                ));
            }
        }
        if at_unix_s < key.valid_from_unix_s {
            return Err(ForgeCompilationAttestationCurrentErrorV1::CurrentKeyNotYetValid(
                entry.key_id.clone(),
            ));
        }
        if at_unix_s >= key.valid_until_unix_s {
            return Err(ForgeCompilationAttestationCurrentErrorV1::CurrentKeyExpired(
                entry.key_id.clone(),
            ));
        }
        natural_valid_until_unix_s = natural_valid_until_unix_s.min(key.valid_until_unix_s);

        let current_binding = authority_resolver
            .signer_binding(
                authority_snapshot.generation_commitment,
                entry.algorithm,
                &entry.key_id,
            )
            .map_err(ForgeCompilationAttestationCurrentErrorV1::AuthorityResolver)?
            .ok_or_else(|| ForgeCompilationAttestationCurrentErrorV1::MissingCurrentAuthorityBinding {
                key_id: entry.key_id.clone(),
            })?;
        current_binding
            .validate()
            .map_err(|error| ForgeCompilationAttestationCurrentErrorV1::CurrentAuthorityBinding(error.to_string()))?;
        if current_binding.generation_commitment != authority_snapshot.generation_commitment {
            return Err(
                ForgeCompilationAttestationCurrentErrorV1::CurrentAuthorityBindingGenerationMismatch {
                    key_id: entry.key_id.clone(),
                },
            );
        }
        if current_binding.algorithm != entry.algorithm || current_binding.key_id != entry.key_id {
            return Err(
                ForgeCompilationAttestationCurrentErrorV1::CurrentAuthorityBindingIdentityMismatch {
                    expected_key_id: entry.key_id.clone(),
                    actual_key_id: current_binding.key_id,
                },
            );
        }
        if at_unix_s < current_binding.valid_from_unix_s {
            return Err(
                ForgeCompilationAttestationCurrentErrorV1::CurrentAuthorityBindingNotYetValid {
                    key_id: entry.key_id.clone(),
                },
            );
        }
        if at_unix_s >= current_binding.valid_until_unix_s {
            return Err(
                ForgeCompilationAttestationCurrentErrorV1::CurrentAuthorityBindingExpired {
                    key_id: entry.key_id.clone(),
                },
            );
        }

        let principal = authority_snapshot
            .identity_namespaces
            .qualify_principal(current_binding.principal_id)
            .map_err(|error| ForgeCompilationAttestationCurrentErrorV1::IdentityNamespace(error.to_string()))?;
        if principal != entry.principal {
            return Err(
                ForgeCompilationAttestationCurrentErrorV1::CurrentPrincipalBindingMismatch {
                    key_id: entry.key_id.clone(),
                },
            );
        }

        let authority = current_binding
            .authority_id
            .map(|authority_id| {
                authority_snapshot
                    .identity_namespaces
                    .qualify_authority(authority_id)
            })
            .transpose()
            .map_err(|error| ForgeCompilationAttestationCurrentErrorV1::IdentityNamespace(error.to_string()))?;
        if authority != entry.authority {
            return Err(
                ForgeCompilationAttestationCurrentErrorV1::CurrentAuthorityIdentityMismatch {
                    key_id: entry.key_id.clone(),
                },
            );
        }

        natural_valid_until_unix_s =
            natural_valid_until_unix_s.min(current_binding.valid_until_unix_s);
    }

    Ok(RevalidatedForgeCompilationAttestationV1 {
        binding: binding.clone(),
        revalidated_at_unix_s: at_unix_s,
        natural_valid_until_unix_s,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgeCompilationAttestationCurrentErrorV1 {
    BindingOutsideTimeBounds,
    TrustProvider(String),
    TrustSnapshot(String),
    TrustGenerationDrift,
    TrustSnapshotNotYetValid,
    TrustSnapshotExpired,
    AuthorityResolver(String),
    AuthoritySnapshot(String),
    AuthorityBindingGenerationDrift,
    AuthoritySnapshotNotYetValid,
    AuthoritySnapshotExpired,
    IdentityNamespace(String),
    IdentityNamespaceDrift,
    MissingCurrentKey { key_id: String },
    CurrentKey(String),
    CurrentKeyIdentityMismatch { expected_key_id: String, actual_key_id: String },
    CurrentKeyDomainDenied { key_id: String, domain: String },
    CurrentKeyNotYetValid(String),
    CurrentKeyExpired(String),
    CurrentKeyRetired(String),
    CurrentKeyRevoked(String),
    MissingCurrentAuthorityBinding { key_id: String },
    CurrentAuthorityBinding(String),
    CurrentAuthorityBindingGenerationMismatch { key_id: String },
    CurrentAuthorityBindingIdentityMismatch { expected_key_id: String, actual_key_id: String },
    CurrentAuthorityBindingNotYetValid { key_id: String },
    CurrentAuthorityBindingExpired { key_id: String },
    CurrentPrincipalBindingMismatch { key_id: String },
    CurrentAuthorityIdentityMismatch { key_id: String },
}

impl fmt::Display for ForgeCompilationAttestationCurrentErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BindingOutsideTimeBounds => write!(f, "Forge compilation attestation binding is outside its time bounds"),
            Self::TrustProvider(error) => write!(f, "attestation trust provider failed during currentness revalidation: {error}"),
            Self::TrustSnapshot(error) => write!(f, "current trust snapshot is invalid: {error}"),
            Self::TrustGenerationDrift => write!(f, "cryptographic trust generation changed; full attestation requalification required"),
            Self::TrustSnapshotNotYetValid => write!(f, "current trust snapshot is not yet valid"),
            Self::TrustSnapshotExpired => write!(f, "current trust snapshot has expired"),
            Self::AuthorityResolver(error) => write!(f, "authority resolver failed during currentness revalidation: {error}"),
            Self::AuthoritySnapshot(error) => write!(f, "current authority-binding snapshot is invalid: {error}"),
            Self::AuthorityBindingGenerationDrift => write!(f, "authority-binding generation changed; full identity requalification required"),
            Self::AuthoritySnapshotNotYetValid => write!(f, "current authority-binding snapshot is not yet valid"),
            Self::AuthoritySnapshotExpired => write!(f, "current authority-binding snapshot has expired"),
            Self::IdentityNamespace(error) => write!(f, "current identity namespace is invalid: {error}"),
            Self::IdentityNamespaceDrift => write!(f, "identity namespace semantics changed; full identity requalification required"),
            Self::MissingCurrentKey { key_id } => write!(f, "current trust generation no longer contains signer key {key_id}"),
            Self::CurrentKey(error) => write!(f, "current signer key record is invalid: {error}"),
            Self::CurrentKeyIdentityMismatch { expected_key_id, actual_key_id } => write!(f, "current signer key identity mismatch: expected {expected_key_id}, got {actual_key_id}"),
            Self::CurrentKeyDomainDenied { key_id, domain } => write!(f, "current signer key {key_id} no longer permits attestation domain {domain}"),
            Self::CurrentKeyNotYetValid(key_id) => write!(f, "current signer key {key_id} is not yet valid"),
            Self::CurrentKeyExpired(key_id) => write!(f, "current signer key {key_id} has expired"),
            Self::CurrentKeyRetired(key_id) => write!(f, "current signer key {key_id} is retired"),
            Self::CurrentKeyRevoked(key_id) => write!(f, "current signer key {key_id} is revoked"),
            Self::MissingCurrentAuthorityBinding { key_id } => write!(f, "current authority generation no longer contains signer binding {key_id}"),
            Self::CurrentAuthorityBinding(error) => write!(f, "current authority binding is invalid: {error}"),
            Self::CurrentAuthorityBindingGenerationMismatch { key_id } => write!(f, "current authority binding generation mismatch for {key_id}"),
            Self::CurrentAuthorityBindingIdentityMismatch { expected_key_id, actual_key_id } => write!(f, "current authority binding identity mismatch: expected {expected_key_id}, got {actual_key_id}"),
            Self::CurrentAuthorityBindingNotYetValid { key_id } => write!(f, "current authority binding for {key_id} is not yet valid"),
            Self::CurrentAuthorityBindingExpired { key_id } => write!(f, "current authority binding for {key_id} has expired"),
            Self::CurrentPrincipalBindingMismatch { key_id } => write!(f, "current principal binding for {key_id} differs from the qualified witness"),
            Self::CurrentAuthorityIdentityMismatch { key_id } => write!(f, "current authority identity for {key_id} differs from the qualified witness"),
        }
    }
}

impl Error for ForgeCompilationAttestationCurrentErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control_plane::attestation::{
        AttestationKeyLifecycleV1, AttestationPolicyV1, AttestationSignatureAlgorithmV1,
        AttestationSubjectV1, AttestationTrustSnapshotV1, AttestationTrustedKeyV1,
        DetachedAttestationEnvelopeV1, DetachedAttestationSignatureV1,
        CONTROL_PLANE_ATTESTATION_VERSION_V1,
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
        bind_forge_compilation_attestation_v1,
    };
    use crate::control_plane::forge_compiled_artifact_preflight::ForgeCompiledArtifactPreflightV1;
    use crate::control_plane::forge_profile::{
        ForgeDeterminismPolicyV1, ForgeExecutionProfileDefinitionV1, ForgeImportPolicyV1,
        ForgeInterruptionPolicyV1, ForgeRuntimeIdentityV1, ForgeVerifierAbiV1,
    };
    use std::collections::{BTreeMap, BTreeSet};

    const TRUST_GENERATION: [u8; 32] = [0x21; 32];
    const BINDING_GENERATION: [u8; 32] = [0x31; 32];
    const RUNTIME_LINEAGE: [u8; 32] = [0x41; 32];
    const FEATURE_POLICY: [u8; 32] = [0x51; 32];
    const CANDIDATE: &[u8] = b"forge-currentness-precompiled-v1";
    const QUALIFIED_AT: u64 = 200;

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
                        AttestationIdentityNamespaceV1::new("principals-v1", [0x61; 32]).unwrap(),
                        Some(
                            AttestationIdentityNamespaceV1::new("authorities-v1", [0x71; 32])
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
            trust_generation_commitment: [u8; 32],
            algorithm: AttestationSignatureAlgorithmV1,
            key_id: &str,
        ) -> Result<Option<AttestationTrustedKeyV1>, String> {
            if trust_generation_commitment == self.trust_snapshot.generation_commitment
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
            if trust_generation_commitment != self.trust_snapshot.generation_commitment
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
            generation_commitment: [u8; 32],
            algorithm: AttestationSignatureAlgorithmV1,
            key_id: &str,
        ) -> Result<Option<QualifiedAttestationSignerBindingV1>, String> {
            if generation_commitment == self.authority_snapshot.generation_commitment
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
            hasher.update(b"test-only-forge-currentness\0");
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

    fn receipt() -> ForgeCompilationReceiptV1 {
        let request = ForgeCompilationRequestV1::new(
            ForgeArtifactIdentityV1::new_blake3_256([0x81; 32], 128).unwrap(),
            ForgeCompilationProfileIdentityV1::new("compile-v1", [0x91; 32]).unwrap(),
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

    fn bound(provider: &Provider) -> ForgeCompilationAttestationBindingV1 {
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
        bind_forge_compilation_attestation_v1(&preflight, &qualified, QUALIFIED_AT).unwrap()
    }

    #[test]
    fn unchanged_generations_and_bindings_revalidate() {
        let provider = Provider::new();
        let current = revalidate_forge_compilation_attestation_current_v1(
            &bound(&provider),
            &provider,
            &provider,
            300,
        )
        .unwrap();
        assert_eq!(current.revalidated_at_unix_s(), 300);
        assert_eq!(current.natural_valid_until_unix_s(), 700);
        assert!(current.remains_within_time_bounds(699));
        assert!(!current.remains_within_time_bounds(700));
    }

    #[test]
    fn trust_generation_drift_requires_full_requalification() {
        let original = Provider::new();
        let binding = bound(&original);
        let mut current = original.clone();
        current.trust_snapshot.generation_commitment[0] ^= 0xFF;

        assert!(matches!(
            revalidate_forge_compilation_attestation_current_v1(
                &binding, &current, &current, 300,
            ),
            Err(ForgeCompilationAttestationCurrentErrorV1::TrustGenerationDrift)
        ));
    }

    #[test]
    fn authority_generation_or_namespace_drift_requires_requalification() {
        let original = Provider::new();
        let binding = bound(&original);

        let mut generation_drift = original.clone();
        generation_drift.authority_snapshot.generation_commitment[0] ^= 0xFF;
        assert!(matches!(
            revalidate_forge_compilation_attestation_current_v1(
                &binding, &generation_drift, &generation_drift, 300,
            ),
            Err(
                ForgeCompilationAttestationCurrentErrorV1::AuthorityBindingGenerationDrift
            )
        ));

        let mut namespace_drift = original.clone();
        namespace_drift
            .authority_snapshot
            .identity_namespaces
            .principal
            .semantics_commitment[0] ^= 0xFF;
        assert!(matches!(
            revalidate_forge_compilation_attestation_current_v1(
                &binding, &namespace_drift, &namespace_drift, 300,
            ),
            Err(ForgeCompilationAttestationCurrentErrorV1::IdentityNamespaceDrift)
        ));
    }

    #[test]
    fn current_key_revocation_or_domain_removal_fails_closed() {
        let original = Provider::new();
        let binding = bound(&original);

        let mut revoked = original.clone();
        revoked.key.lifecycle = AttestationKeyLifecycleV1::Revoked;
        assert!(matches!(
            revalidate_forge_compilation_attestation_current_v1(
                &binding, &revoked, &revoked, 300,
            ),
            Err(ForgeCompilationAttestationCurrentErrorV1::CurrentKeyRevoked(_))
        ));

        let mut domain_removed = original.clone();
        domain_removed.key.allowed_subject_domains.clear();
        assert!(matches!(
            revalidate_forge_compilation_attestation_current_v1(
                &binding, &domain_removed, &domain_removed, 300,
            ),
            Err(ForgeCompilationAttestationCurrentErrorV1::CurrentKey(_))
        ));
    }

    #[test]
    fn principal_or_authority_remapping_fails_closed() {
        let original = Provider::new();
        let binding = bound(&original);

        let mut principal_changed = original.clone();
        principal_changed.authority_binding.principal_id = "mallory".into();
        assert!(matches!(
            revalidate_forge_compilation_attestation_current_v1(
                &binding, &principal_changed, &principal_changed, 300,
            ),
            Err(
                ForgeCompilationAttestationCurrentErrorV1::CurrentPrincipalBindingMismatch { .. }
            )
        ));

        let mut authority_changed = original.clone();
        authority_changed.authority_binding.authority_id = Some("org-b".into());
        assert!(matches!(
            revalidate_forge_compilation_attestation_current_v1(
                &binding, &authority_changed, &authority_changed, 300,
            ),
            Err(
                ForgeCompilationAttestationCurrentErrorV1::CurrentAuthorityIdentityMismatch { .. }
            )
        ));
    }

    #[test]
    fn key_or_binding_expiry_tightens_currentness() {
        let original = Provider::new();
        let binding = bound(&original);

        let mut key_expired = original.clone();
        key_expired.key.valid_until_unix_s = 250;
        assert!(matches!(
            revalidate_forge_compilation_attestation_current_v1(
                &binding, &key_expired, &key_expired, 300,
            ),
            Err(ForgeCompilationAttestationCurrentErrorV1::CurrentKeyExpired(_))
        ));

        let mut binding_expired = original.clone();
        binding_expired.authority_binding.valid_until_unix_s = 250;
        assert!(matches!(
            revalidate_forge_compilation_attestation_current_v1(
                &binding, &binding_expired, &binding_expired, 300,
            ),
            Err(
                ForgeCompilationAttestationCurrentErrorV1::CurrentAuthorityBindingExpired { .. }
            )
        ));
    }
}
