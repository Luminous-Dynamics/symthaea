// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Point-of-use currentness revalidation for a Forge compilation-attestation binding.
//!
//! #822 proves that one exact compilation receipt was structurally preflighted and
//! was the exact subject of a qualified worker attestation at one decision instant.
//! That positive witness is intentionally not durable authority. This module asks
//! the current trust and identity providers again immediately before consequential
//! use and fails closed on generation, lifecycle, namespace, identity, or expiry
//! drift.
//!
//! This is **currentness revalidation**, not signature re-verification: the compact
//! in-process witnesses do not retain the original detached signature bytes. A
//! provider that changes key material without changing its trust generation still
//! violates the provider contract and cannot be detected here cryptographically.
//!
//! ```text
//! ForgeCompilationAttestationBindingV1
//!     + current cryptographic trust state
//!     + current exact-profile support
//!     + current authority-binding state
//!         -> CurrentForgeCompilationAttestationV1
//!
//! CurrentForgeCompilationAttestationV1
//!     != Forge/Xenia effect authority
//!     != local PolicyBundle admission
//!     != permission to deserialize precompiled Wasm
//! ```

use super::attestation::AttestationKeyLifecycleV1;
use super::attestation_crypto_profile::ProfileBoundAttestationTrustVerifierV1;
use super::attestation_qualified_authority::{
    QualifiedAttestationAuthorityResolverV1, QualifiedAttestationAuthorityEntryV1,
};
use super::forge_compilation_attestation::{
    FORGE_COMPILATION_RECEIPT_ATTESTATION_SUBJECT_DOMAIN_V1,
    ForgeCompilationAttestationBindingV1,
};
use std::error::Error;
use std::fmt;

/// Positive in-process result of point-of-use currentness revalidation.
///
/// Private fields and no serde derives prevent this from becoming a portable
/// bearer token. The witness proves currentness only at [`Self::checked_at_unix_s`];
/// callers that reach a consequential boundary later must revalidate again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentForgeCompilationAttestationV1 {
    binding: ForgeCompilationAttestationBindingV1,
    checked_at_unix_s: u64,
    current_valid_until_unix_s: u64,
}

impl CurrentForgeCompilationAttestationV1 {
    pub fn binding(&self) -> &ForgeCompilationAttestationBindingV1 {
        &self.binding
    }

    /// Exact instant at which provider currentness was revalidated.
    pub fn checked_at_unix_s(&self) -> u64 {
        self.checked_at_unix_s
    }

    /// Earliest known natural expiry observed during this revalidation.
    ///
    /// This is evidence metadata, not a reusable currentness lease: provider
    /// generations, lifecycle state, or identity mappings may change before it.
    pub fn current_valid_until_unix_s(&self) -> u64 {
        self.current_valid_until_unix_s
    }
}

/// Revalidate every current-state component retained by one exact #822 binding.
///
/// This deliberately performs more than generation equality. Every signer key and
/// every key→identity binding is read again and compared to the exact positive
/// identities retained by the original binding. This catches mutable provider
/// records even when a buggy provider forgets to advance its generation.
pub fn revalidate_forge_compilation_attestation_current_v1<T, A>(
    binding: &ForgeCompilationAttestationBindingV1,
    trust_provider: &T,
    authority_resolver: &A,
    checked_at_unix_s: u64,
) -> Result<CurrentForgeCompilationAttestationV1, ForgeCompilationAttestationCurrentnessErrorV1>
where
    T: ProfileBoundAttestationTrustVerifierV1 + ?Sized,
    A: QualifiedAttestationAuthorityResolverV1 + ?Sized,
{
    if checked_at_unix_s < binding.evaluation_time_unix_s() {
        return Err(
            ForgeCompilationAttestationCurrentnessErrorV1::CheckBeforeOriginalDecision {
                original_unix_s: binding.evaluation_time_unix_s(),
                checked_at_unix_s,
            },
        );
    }
    if !binding.remains_within_time_bounds(checked_at_unix_s) {
        return Err(
            ForgeCompilationAttestationCurrentnessErrorV1::OriginalBindingOutsideTimeBounds,
        );
    }

    let trust_snapshot = trust_provider
        .current_snapshot()
        .map_err(ForgeCompilationAttestationCurrentnessErrorV1::TrustProvider)?;
    trust_snapshot
        .validate()
        .map_err(|error| {
            ForgeCompilationAttestationCurrentnessErrorV1::TrustState(error.to_string())
        })?;
    if trust_snapshot.generation_commitment != binding.trust_generation_commitment() {
        return Err(
            ForgeCompilationAttestationCurrentnessErrorV1::TrustGenerationChanged {
                expected: binding.trust_generation_commitment(),
                current: trust_snapshot.generation_commitment,
            },
        );
    }
    if checked_at_unix_s < trust_snapshot.valid_from_unix_s {
        return Err(ForgeCompilationAttestationCurrentnessErrorV1::TrustSnapshotNotYetValid);
    }
    if checked_at_unix_s >= trust_snapshot.valid_until_unix_s {
        return Err(ForgeCompilationAttestationCurrentnessErrorV1::TrustSnapshotExpired);
    }

    let authority_snapshot = authority_resolver
        .current_snapshot()
        .map_err(ForgeCompilationAttestationCurrentnessErrorV1::AuthorityResolver)?;
    authority_snapshot
        .validate()
        .map_err(|error| {
            ForgeCompilationAttestationCurrentnessErrorV1::AuthorityState(error.to_string())
        })?;
    if authority_snapshot.generation_commitment
        != binding.authority_binding_generation_commitment()
    {
        return Err(
            ForgeCompilationAttestationCurrentnessErrorV1::AuthorityBindingGenerationChanged {
                expected: binding.authority_binding_generation_commitment(),
                current: authority_snapshot.generation_commitment,
            },
        );
    }
    if checked_at_unix_s < authority_snapshot.valid_from_unix_s {
        return Err(
            ForgeCompilationAttestationCurrentnessErrorV1::AuthoritySnapshotNotYetValid,
        );
    }
    if checked_at_unix_s >= authority_snapshot.valid_until_unix_s {
        return Err(ForgeCompilationAttestationCurrentnessErrorV1::AuthoritySnapshotExpired);
    }

    let current_namespace_commitment = authority_snapshot
        .identity_namespaces
        .canonical_commitment_v1()
        .map_err(|error| {
            ForgeCompilationAttestationCurrentnessErrorV1::AuthorityState(error.to_string())
        })?;
    if current_namespace_commitment != binding.identity_namespace_set_commitment() {
        return Err(
            ForgeCompilationAttestationCurrentnessErrorV1::IdentityNamespaceSemanticsChanged {
                expected: binding.identity_namespace_set_commitment(),
                current: current_namespace_commitment,
            },
        );
    }

    let mut current_valid_until_unix_s = binding
        .natural_valid_until_unix_s()
        .min(trust_snapshot.valid_until_unix_s)
        .min(authority_snapshot.valid_until_unix_s);

    for entry in binding.qualified_attestation().entries() {
        current_valid_until_unix_s = revalidate_signer_key(
            entry,
            binding,
            trust_provider,
            checked_at_unix_s,
            current_valid_until_unix_s,
        )?;
        current_valid_until_unix_s = revalidate_authority_binding(
            entry,
            binding,
            authority_resolver,
            &authority_snapshot.identity_namespaces,
            checked_at_unix_s,
            current_valid_until_unix_s,
        )?;
    }

    if checked_at_unix_s >= current_valid_until_unix_s {
        return Err(
            ForgeCompilationAttestationCurrentnessErrorV1::CurrentValidityBoundaryReached,
        );
    }

    Ok(CurrentForgeCompilationAttestationV1 {
        binding: binding.clone(),
        checked_at_unix_s,
        current_valid_until_unix_s,
    })
}

fn revalidate_signer_key<T>(
    entry: &QualifiedAttestationAuthorityEntryV1,
    binding: &ForgeCompilationAttestationBindingV1,
    trust_provider: &T,
    checked_at_unix_s: u64,
    current_valid_until_unix_s: u64,
) -> Result<u64, ForgeCompilationAttestationCurrentnessErrorV1>
where
    T: ProfileBoundAttestationTrustVerifierV1 + ?Sized,
{
    let profile = entry.algorithm.exact_profile_v1();
    let supported = trust_provider
        .supports_exact_profile_v1(profile)
        .map_err(ForgeCompilationAttestationCurrentnessErrorV1::TrustProvider)?;
    if !supported {
        return Err(
            ForgeCompilationAttestationCurrentnessErrorV1::ExactCryptoProfileNoLongerSupported {
                key_id: entry.key_id.clone(),
                profile_name: profile.canonical_name,
            },
        );
    }

    let key = trust_provider
        .key_record(
            binding.trust_generation_commitment(),
            entry.algorithm,
            &entry.key_id,
        )
        .map_err(ForgeCompilationAttestationCurrentnessErrorV1::TrustProvider)?
        .ok_or_else(|| {
            ForgeCompilationAttestationCurrentnessErrorV1::SignerKeyMissing {
                key_id: entry.key_id.clone(),
            }
        })?;
    key.validate().map_err(|error| {
        ForgeCompilationAttestationCurrentnessErrorV1::TrustState(error.to_string())
    })?;
    if key.algorithm != entry.algorithm || key.key_id != entry.key_id {
        return Err(
            ForgeCompilationAttestationCurrentnessErrorV1::SignerKeyIdentityChanged {
                expected_key_id: entry.key_id.clone(),
                current_key_id: key.key_id,
            },
        );
    }
    match key.lifecycle {
        AttestationKeyLifecycleV1::Active => {}
        AttestationKeyLifecycleV1::Retired => {
            return Err(
                ForgeCompilationAttestationCurrentnessErrorV1::SignerKeyRetired {
                    key_id: entry.key_id.clone(),
                },
            );
        }
        AttestationKeyLifecycleV1::Revoked => {
            return Err(
                ForgeCompilationAttestationCurrentnessErrorV1::SignerKeyRevoked {
                    key_id: entry.key_id.clone(),
                },
            );
        }
    }
    if !key
        .allowed_subject_domains
        .contains(FORGE_COMPILATION_RECEIPT_ATTESTATION_SUBJECT_DOMAIN_V1)
    {
        return Err(
            ForgeCompilationAttestationCurrentnessErrorV1::SignerNoLongerAllowsForgeReceiptDomain {
                key_id: entry.key_id.clone(),
            },
        );
    }
    if checked_at_unix_s < key.valid_from_unix_s {
        return Err(
            ForgeCompilationAttestationCurrentnessErrorV1::SignerKeyNotYetValid {
                key_id: entry.key_id.clone(),
            },
        );
    }
    if checked_at_unix_s >= key.valid_until_unix_s {
        return Err(
            ForgeCompilationAttestationCurrentnessErrorV1::SignerKeyExpired {
                key_id: entry.key_id.clone(),
            },
        );
    }

    Ok(current_valid_until_unix_s.min(key.valid_until_unix_s))
}

fn revalidate_authority_binding<A>(
    entry: &QualifiedAttestationAuthorityEntryV1,
    binding: &ForgeCompilationAttestationBindingV1,
    authority_resolver: &A,
    namespaces: &super::attestation_identity_namespace::AttestationIdentityNamespaceSetV1,
    checked_at_unix_s: u64,
    current_valid_until_unix_s: u64,
) -> Result<u64, ForgeCompilationAttestationCurrentnessErrorV1>
where
    A: QualifiedAttestationAuthorityResolverV1 + ?Sized,
{
    let current = authority_resolver
        .signer_binding(
            binding.authority_binding_generation_commitment(),
            entry.algorithm,
            &entry.key_id,
        )
        .map_err(ForgeCompilationAttestationCurrentnessErrorV1::AuthorityResolver)?
        .ok_or_else(|| {
            ForgeCompilationAttestationCurrentnessErrorV1::AuthorityBindingMissing {
                key_id: entry.key_id.clone(),
            }
        })?;
    current.validate().map_err(|error| {
        ForgeCompilationAttestationCurrentnessErrorV1::AuthorityState(error.to_string())
    })?;
    if current.generation_commitment != binding.authority_binding_generation_commitment() {
        return Err(
            ForgeCompilationAttestationCurrentnessErrorV1::AuthorityBindingRecordGenerationChanged {
                key_id: entry.key_id.clone(),
            },
        );
    }
    if current.algorithm != entry.algorithm || current.key_id != entry.key_id {
        return Err(
            ForgeCompilationAttestationCurrentnessErrorV1::AuthorityBindingKeyIdentityChanged {
                expected_key_id: entry.key_id.clone(),
                current_key_id: current.key_id,
            },
        );
    }
    if checked_at_unix_s < current.valid_from_unix_s {
        return Err(
            ForgeCompilationAttestationCurrentnessErrorV1::AuthorityBindingNotYetValid {
                key_id: entry.key_id.clone(),
            },
        );
    }
    if checked_at_unix_s >= current.valid_until_unix_s {
        return Err(
            ForgeCompilationAttestationCurrentnessErrorV1::AuthorityBindingExpired {
                key_id: entry.key_id.clone(),
            },
        );
    }

    let current_principal = namespaces
        .qualify_principal(current.principal_id)
        .map_err(|error| {
            ForgeCompilationAttestationCurrentnessErrorV1::AuthorityState(error.to_string())
        })?;
    if current_principal != entry.principal {
        return Err(
            ForgeCompilationAttestationCurrentnessErrorV1::PrincipalIdentityChanged {
                key_id: entry.key_id.clone(),
            },
        );
    }

    let current_authority = current
        .authority_id
        .map(|authority_id| namespaces.qualify_authority(authority_id))
        .transpose()
        .map_err(|error| {
            ForgeCompilationAttestationCurrentnessErrorV1::AuthorityState(error.to_string())
        })?;
    if current_authority != entry.authority {
        return Err(
            ForgeCompilationAttestationCurrentnessErrorV1::AuthorityIdentityChanged {
                key_id: entry.key_id.clone(),
            },
        );
    }

    Ok(current_valid_until_unix_s.min(current.valid_until_unix_s))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgeCompilationAttestationCurrentnessErrorV1 {
    CheckBeforeOriginalDecision {
        original_unix_s: u64,
        checked_at_unix_s: u64,
    },
    OriginalBindingOutsideTimeBounds,
    TrustProvider(String),
    TrustState(String),
    TrustGenerationChanged {
        expected: [u8; 32],
        current: [u8; 32],
    },
    TrustSnapshotNotYetValid,
    TrustSnapshotExpired,
    ExactCryptoProfileNoLongerSupported {
        key_id: String,
        profile_name: &'static str,
    },
    SignerKeyMissing {
        key_id: String,
    },
    SignerKeyIdentityChanged {
        expected_key_id: String,
        current_key_id: String,
    },
    SignerKeyRetired {
        key_id: String,
    },
    SignerKeyRevoked {
        key_id: String,
    },
    SignerNoLongerAllowsForgeReceiptDomain {
        key_id: String,
    },
    SignerKeyNotYetValid {
        key_id: String,
    },
    SignerKeyExpired {
        key_id: String,
    },
    AuthorityResolver(String),
    AuthorityState(String),
    AuthorityBindingGenerationChanged {
        expected: [u8; 32],
        current: [u8; 32],
    },
    AuthoritySnapshotNotYetValid,
    AuthoritySnapshotExpired,
    IdentityNamespaceSemanticsChanged {
        expected: [u8; 32],
        current: [u8; 32],
    },
    AuthorityBindingMissing {
        key_id: String,
    },
    AuthorityBindingRecordGenerationChanged {
        key_id: String,
    },
    AuthorityBindingKeyIdentityChanged {
        expected_key_id: String,
        current_key_id: String,
    },
    AuthorityBindingNotYetValid {
        key_id: String,
    },
    AuthorityBindingExpired {
        key_id: String,
    },
    PrincipalIdentityChanged {
        key_id: String,
    },
    AuthorityIdentityChanged {
        key_id: String,
    },
    CurrentValidityBoundaryReached,
}

impl fmt::Display for ForgeCompilationAttestationCurrentnessErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ForgeCompilationAttestationCurrentnessErrorV1 as E;
        match self {
            E::CheckBeforeOriginalDecision { original_unix_s, checked_at_unix_s } => write!(
                f,
                "currentness check at {checked_at_unix_s} precedes original Forge attestation decision at {original_unix_s}"
            ),
            E::OriginalBindingOutsideTimeBounds => {
                write!(f, "Forge compilation-attestation binding is outside its original time bounds")
            }
            E::TrustProvider(error) => write!(f, "Forge attestation trust provider failed: {error}"),
            E::TrustState(error) => write!(f, "invalid current Forge attestation trust state: {error}"),
            E::TrustGenerationChanged { .. } => write!(f, "Forge attestation cryptographic trust generation changed"),
            E::TrustSnapshotNotYetValid => write!(f, "current Forge attestation trust snapshot is not yet valid"),
            E::TrustSnapshotExpired => write!(f, "current Forge attestation trust snapshot has expired"),
            E::ExactCryptoProfileNoLongerSupported { key_id, profile_name } => write!(
                f,
                "current provider no longer supports exact crypto profile {profile_name} for signer {key_id}"
            ),
            E::SignerKeyMissing { key_id } => write!(f, "current signer key is missing: {key_id}"),
            E::SignerKeyIdentityChanged { expected_key_id, current_key_id } => write!(
                f,
                "current signer key identity changed: expected {expected_key_id}, got {current_key_id}"
            ),
            E::SignerKeyRetired { key_id } => write!(f, "current signer key is retired: {key_id}"),
            E::SignerKeyRevoked { key_id } => write!(f, "current signer key is revoked: {key_id}"),
            E::SignerNoLongerAllowsForgeReceiptDomain { key_id } => write!(
                f,
                "current signer key {key_id} no longer allows the Forge compilation-receipt domain"
            ),
            E::SignerKeyNotYetValid { key_id } => write!(f, "current signer key is not yet valid: {key_id}"),
            E::SignerKeyExpired { key_id } => write!(f, "current signer key has expired: {key_id}"),
            E::AuthorityResolver(error) => write!(f, "Forge attestation authority resolver failed: {error}"),
            E::AuthorityState(error) => write!(f, "invalid current Forge attestation authority state: {error}"),
            E::AuthorityBindingGenerationChanged { .. } => {
                write!(f, "Forge attestation authority-binding generation changed")
            }
            E::AuthoritySnapshotNotYetValid => write!(f, "current Forge authority-binding snapshot is not yet valid"),
            E::AuthoritySnapshotExpired => write!(f, "current Forge authority-binding snapshot has expired"),
            E::IdentityNamespaceSemanticsChanged { .. } => {
                write!(f, "Forge attestation identity-namespace semantics changed")
            }
            E::AuthorityBindingMissing { key_id } => write!(f, "current authority binding is missing for {key_id}"),
            E::AuthorityBindingRecordGenerationChanged { key_id } => write!(
                f,
                "current authority-binding record generation changed for {key_id}"
            ),
            E::AuthorityBindingKeyIdentityChanged { expected_key_id, current_key_id } => write!(
                f,
                "current authority binding key identity changed: expected {expected_key_id}, got {current_key_id}"
            ),
            E::AuthorityBindingNotYetValid { key_id } => write!(f, "current authority binding is not yet valid for {key_id}"),
            E::AuthorityBindingExpired { key_id } => write!(f, "current authority binding expired for {key_id}"),
            E::PrincipalIdentityChanged { key_id } => write!(f, "current principal identity changed for {key_id}"),
            E::AuthorityIdentityChanged { key_id } => write!(f, "current authority identity changed for {key_id}"),
            E::CurrentValidityBoundaryReached => write!(f, "current Forge attestation validity boundary has been reached"),
        }
    }
}

impl Error for ForgeCompilationAttestationCurrentnessErrorV1 {}

#[cfg(test)]
mod tests;
