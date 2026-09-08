// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Point-of-use currentness revalidation for namespace-qualified attestations.
//!
//! A positive attestation/diversity witness proves what was true at its original
//! verification instant. Consequential use later must not assume that the same
//! trust generation, key lifecycle, exact-profile support, identity namespace,
//! or signer->principal/authority mapping is still current.
//!
//! This module re-reads those current provider views and returns a new private,
//! non-serializable witness only if the original qualified identity evidence is
//! still current at the requested point of use.
//!
//! It intentionally does **not** re-run detached signatures: the positive witness
//! does not retain raw signature bytes. This is currentness revalidation of an
//! already-verified in-process capability, not cryptographic re-verification.

use super::attestation::AttestationKeyLifecycleV1;
use super::attestation_crypto_profile::ProfileBoundAttestationTrustVerifierV1;
use super::attestation_qualified_authority::{
    QualifiedAttestationAuthorityDiversityV1, QualifiedAttestationAuthorityResolverV1,
};
use std::error::Error;
use std::fmt;

/// Positive in-process witness that one already-qualified attestation remained
/// current when revalidated at a later point of use.
///
/// Private fields and no serde derives prevent this from becoming a portable
/// authority token. It proves currentness of attestation/identity evidence only;
/// it grants no Forge, Xenia, deployment, promotion, or other effect authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RevalidatedQualifiedAttestationCurrentnessV1 {
    qualified: QualifiedAttestationAuthorityDiversityV1,
    revalidated_at_unix_s: u64,
    natural_valid_until_unix_s: u64,
}

impl RevalidatedQualifiedAttestationCurrentnessV1 {
    pub fn qualified(&self) -> &QualifiedAttestationAuthorityDiversityV1 {
        &self.qualified
    }

    pub fn revalidated_at_unix_s(&self) -> u64 {
        self.revalidated_at_unix_s
    }

    /// Earliest current natural expiry after re-reading trust, key, authority
    /// snapshot and signer-binding metadata.
    pub fn natural_valid_until_unix_s(&self) -> u64 {
        self.natural_valid_until_unix_s
    }

    /// Time-only check. A later consequential use still needs another currentness
    /// revalidation if trust/identity state may have changed in the meantime.
    pub fn remains_within_time_bounds(&self, at_unix_s: u64) -> bool {
        self.revalidated_at_unix_s <= at_unix_s
            && at_unix_s < self.natural_valid_until_unix_s
    }
}

/// Revalidate a namespace-qualified attestation witness against the providers'
/// exact current trust and authority-binding views.
///
/// The function is deliberately stricter than generation equality alone. Even if
/// a buggy provider mutates metadata without rotating its generation commitment,
/// same-generation key revocation/retirement, domain removal, profile-support
/// withdrawal, expiry, namespace drift, or identity remapping is detected.
pub fn revalidate_qualified_attestation_currentness_v1<P>(
    qualified: &QualifiedAttestationAuthorityDiversityV1,
    trust_provider: &P,
    authority_resolver: &dyn QualifiedAttestationAuthorityResolverV1,
    at_unix_s: u64,
) -> Result<RevalidatedQualifiedAttestationCurrentnessV1, AttestationCurrentnessErrorV1>
where
    P: ProfileBoundAttestationTrustVerifierV1 + ?Sized,
{
    if at_unix_s < qualified.evaluation_time_unix_s() {
        return Err(AttestationCurrentnessErrorV1::RevalidationBeforeOriginalDecision {
            original_at_unix_s: qualified.evaluation_time_unix_s(),
            requested_at_unix_s: at_unix_s,
        });
    }
    if !qualified.remains_within_time_bounds(at_unix_s) {
        return Err(AttestationCurrentnessErrorV1::PriorWitnessOutsideTimeBounds);
    }

    let trust_snapshot = trust_provider
        .current_snapshot()
        .map_err(AttestationCurrentnessErrorV1::TrustProvider)?;
    trust_snapshot
        .validate()
        .map_err(|error| AttestationCurrentnessErrorV1::TrustMetadata(error.to_string()))?;
    if trust_snapshot.generation_commitment != qualified.trust_generation_commitment() {
        return Err(AttestationCurrentnessErrorV1::TrustGenerationChanged);
    }
    if at_unix_s < trust_snapshot.valid_from_unix_s {
        return Err(AttestationCurrentnessErrorV1::TrustSnapshotNotYetValid);
    }
    if at_unix_s >= trust_snapshot.valid_until_unix_s {
        return Err(AttestationCurrentnessErrorV1::TrustSnapshotExpired);
    }

    let authority_snapshot = authority_resolver
        .current_snapshot()
        .map_err(AttestationCurrentnessErrorV1::AuthorityResolver)?;
    authority_snapshot
        .validate()
        .map_err(|error| AttestationCurrentnessErrorV1::AuthorityMetadata(error.to_string()))?;
    if authority_snapshot.generation_commitment
        != qualified.authority_binding_generation_commitment()
    {
        return Err(AttestationCurrentnessErrorV1::AuthorityBindingGenerationChanged);
    }
    if at_unix_s < authority_snapshot.valid_from_unix_s {
        return Err(AttestationCurrentnessErrorV1::AuthoritySnapshotNotYetValid);
    }
    if at_unix_s >= authority_snapshot.valid_until_unix_s {
        return Err(AttestationCurrentnessErrorV1::AuthoritySnapshotExpired);
    }

    let current_namespace_commitment = authority_snapshot
        .identity_namespaces
        .canonical_commitment_v1()
        .map_err(|error| AttestationCurrentnessErrorV1::AuthorityMetadata(error.to_string()))?;
    if current_namespace_commitment != qualified.identity_namespace_set_commitment() {
        return Err(AttestationCurrentnessErrorV1::IdentityNamespaceChanged);
    }

    let mut natural_valid_until_unix_s = qualified
        .natural_valid_until_unix_s()
        .min(trust_snapshot.valid_until_unix_s)
        .min(authority_snapshot.valid_until_unix_s);

    for entry in qualified.entries() {
        let profile = entry.algorithm.exact_profile_v1();
        let supported = trust_provider
            .supports_exact_profile_v1(profile)
            .map_err(AttestationCurrentnessErrorV1::TrustProvider)?;
        if !supported {
            return Err(AttestationCurrentnessErrorV1::ExactProfileNoLongerSupported {
                key_id: entry.key_id.clone(),
                profile_name: profile.canonical_name,
            });
        }

        let key = trust_provider
            .key_record(
                trust_snapshot.generation_commitment,
                entry.algorithm,
                &entry.key_id,
            )
            .map_err(AttestationCurrentnessErrorV1::TrustProvider)?
            .ok_or_else(|| AttestationCurrentnessErrorV1::CurrentKeyMissing {
                key_id: entry.key_id.clone(),
            })?;
        key.validate()
            .map_err(|error| AttestationCurrentnessErrorV1::TrustMetadata(error.to_string()))?;
        if key.algorithm != entry.algorithm || key.key_id != entry.key_id {
            return Err(AttestationCurrentnessErrorV1::CurrentKeyIdentityMismatch {
                expected_key_id: entry.key_id.clone(),
                returned_key_id: key.key_id,
            });
        }
        if !key
            .allowed_subject_domains
            .contains(&qualified.subject().domain)
        {
            return Err(AttestationCurrentnessErrorV1::SubjectDomainNoLongerAllowed {
                key_id: entry.key_id.clone(),
                domain: qualified.subject().domain.clone(),
            });
        }
        match key.lifecycle {
            AttestationKeyLifecycleV1::Active => {}
            AttestationKeyLifecycleV1::Retired => {
                return Err(AttestationCurrentnessErrorV1::CurrentKeyRetired {
                    key_id: entry.key_id.clone(),
                });
            }
            AttestationKeyLifecycleV1::Revoked => {
                return Err(AttestationCurrentnessErrorV1::CurrentKeyRevoked {
                    key_id: entry.key_id.clone(),
                });
            }
        }
        if at_unix_s < key.valid_from_unix_s {
            return Err(AttestationCurrentnessErrorV1::CurrentKeyNotYetValid {
                key_id: entry.key_id.clone(),
            });
        }
        if at_unix_s >= key.valid_until_unix_s {
            return Err(AttestationCurrentnessErrorV1::CurrentKeyExpired {
                key_id: entry.key_id.clone(),
            });
        }
        natural_valid_until_unix_s = natural_valid_until_unix_s.min(key.valid_until_unix_s);

        let binding = authority_resolver
            .signer_binding(
                authority_snapshot.generation_commitment,
                entry.algorithm,
                &entry.key_id,
            )
            .map_err(AttestationCurrentnessErrorV1::AuthorityResolver)?
            .ok_or_else(|| AttestationCurrentnessErrorV1::CurrentBindingMissing {
                key_id: entry.key_id.clone(),
            })?;
        binding
            .validate()
            .map_err(|error| AttestationCurrentnessErrorV1::AuthorityMetadata(error.to_string()))?;
        if binding.generation_commitment != authority_snapshot.generation_commitment {
            return Err(AttestationCurrentnessErrorV1::CurrentBindingGenerationMismatch {
                key_id: entry.key_id.clone(),
            });
        }
        if binding.algorithm != entry.algorithm || binding.key_id != entry.key_id {
            return Err(AttestationCurrentnessErrorV1::CurrentBindingIdentityMismatch {
                expected_key_id: entry.key_id.clone(),
                returned_key_id: binding.key_id,
            });
        }
        if at_unix_s < binding.valid_from_unix_s {
            return Err(AttestationCurrentnessErrorV1::CurrentBindingNotYetValid {
                key_id: entry.key_id.clone(),
            });
        }
        if at_unix_s >= binding.valid_until_unix_s {
            return Err(AttestationCurrentnessErrorV1::CurrentBindingExpired {
                key_id: entry.key_id.clone(),
            });
        }

        let current_principal = authority_snapshot
            .identity_namespaces
            .qualify_principal(binding.principal_id)
            .map_err(|error| AttestationCurrentnessErrorV1::AuthorityMetadata(error.to_string()))?;
        if current_principal != entry.principal {
            return Err(AttestationCurrentnessErrorV1::PrincipalBindingChanged {
                key_id: entry.key_id.clone(),
            });
        }

        let current_authority = binding
            .authority_id
            .map(|authority_id| {
                authority_snapshot
                    .identity_namespaces
                    .qualify_authority(authority_id)
            })
            .transpose()
            .map_err(|error| AttestationCurrentnessErrorV1::AuthorityMetadata(error.to_string()))?;
        if current_authority != entry.authority {
            return Err(AttestationCurrentnessErrorV1::AuthorityBindingChanged {
                key_id: entry.key_id.clone(),
            });
        }

        natural_valid_until_unix_s =
            natural_valid_until_unix_s.min(binding.valid_until_unix_s);
    }

    Ok(RevalidatedQualifiedAttestationCurrentnessV1 {
        qualified: qualified.clone(),
        revalidated_at_unix_s: at_unix_s,
        natural_valid_until_unix_s,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttestationCurrentnessErrorV1 {
    RevalidationBeforeOriginalDecision {
        original_at_unix_s: u64,
        requested_at_unix_s: u64,
    },
    PriorWitnessOutsideTimeBounds,
    TrustProvider(String),
    TrustMetadata(String),
    TrustGenerationChanged,
    TrustSnapshotNotYetValid,
    TrustSnapshotExpired,
    AuthorityResolver(String),
    AuthorityMetadata(String),
    AuthorityBindingGenerationChanged,
    AuthoritySnapshotNotYetValid,
    AuthoritySnapshotExpired,
    IdentityNamespaceChanged,
    ExactProfileNoLongerSupported {
        key_id: String,
        profile_name: &'static str,
    },
    CurrentKeyMissing {
        key_id: String,
    },
    CurrentKeyIdentityMismatch {
        expected_key_id: String,
        returned_key_id: String,
    },
    SubjectDomainNoLongerAllowed {
        key_id: String,
        domain: String,
    },
    CurrentKeyRetired {
        key_id: String,
    },
    CurrentKeyRevoked {
        key_id: String,
    },
    CurrentKeyNotYetValid {
        key_id: String,
    },
    CurrentKeyExpired {
        key_id: String,
    },
    CurrentBindingMissing {
        key_id: String,
    },
    CurrentBindingGenerationMismatch {
        key_id: String,
    },
    CurrentBindingIdentityMismatch {
        expected_key_id: String,
        returned_key_id: String,
    },
    CurrentBindingNotYetValid {
        key_id: String,
    },
    CurrentBindingExpired {
        key_id: String,
    },
    PrincipalBindingChanged {
        key_id: String,
    },
    AuthorityBindingChanged {
        key_id: String,
    },
}

impl fmt::Display for AttestationCurrentnessErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RevalidationBeforeOriginalDecision {
                original_at_unix_s,
                requested_at_unix_s,
            } => write!(
                f,
                "attestation currentness revalidation time {requested_at_unix_s} precedes original decision time {original_at_unix_s}"
            ),
            Self::PriorWitnessOutsideTimeBounds => {
                write!(f, "prior qualified attestation is outside its time bounds")
            }
            Self::TrustProvider(error) => write!(f, "attestation trust provider failed: {error}"),
            Self::TrustMetadata(error) => write!(f, "invalid current attestation trust metadata: {error}"),
            Self::TrustGenerationChanged => write!(f, "attestation trust generation changed"),
            Self::TrustSnapshotNotYetValid => write!(f, "current trust snapshot is not yet valid"),
            Self::TrustSnapshotExpired => write!(f, "current trust snapshot has expired"),
            Self::AuthorityResolver(error) => write!(f, "attestation authority resolver failed: {error}"),
            Self::AuthorityMetadata(error) => write!(f, "invalid current attestation authority metadata: {error}"),
            Self::AuthorityBindingGenerationChanged => write!(f, "attestation authority-binding generation changed"),
            Self::AuthoritySnapshotNotYetValid => write!(f, "current authority-binding snapshot is not yet valid"),
            Self::AuthoritySnapshotExpired => write!(f, "current authority-binding snapshot has expired"),
            Self::IdentityNamespaceChanged => write!(f, "attestation identity namespace semantics changed"),
            Self::ExactProfileNoLongerSupported { key_id, profile_name } => write!(
                f,
                "exact attestation profile {profile_name} is no longer supported for signer {key_id}"
            ),
            Self::CurrentKeyMissing { key_id } => write!(f, "current attestation key {key_id} is missing"),
            Self::CurrentKeyIdentityMismatch { expected_key_id, returned_key_id } => write!(
                f,
                "current attestation key identity mismatch: expected {expected_key_id}, got {returned_key_id}"
            ),
            Self::SubjectDomainNoLongerAllowed { key_id, domain } => write!(
                f,
                "current attestation key {key_id} no longer allows subject domain {domain}"
            ),
            Self::CurrentKeyRetired { key_id } => write!(f, "current attestation key {key_id} is retired"),
            Self::CurrentKeyRevoked { key_id } => write!(f, "current attestation key {key_id} is revoked"),
            Self::CurrentKeyNotYetValid { key_id } => write!(f, "current attestation key {key_id} is not yet valid"),
            Self::CurrentKeyExpired { key_id } => write!(f, "current attestation key {key_id} has expired"),
            Self::CurrentBindingMissing { key_id } => write!(f, "current authority binding for {key_id} is missing"),
            Self::CurrentBindingGenerationMismatch { key_id } => write!(f, "current authority binding generation mismatch for {key_id}"),
            Self::CurrentBindingIdentityMismatch { expected_key_id, returned_key_id } => write!(
                f,
                "current authority binding identity mismatch: expected {expected_key_id}, got {returned_key_id}"
            ),
            Self::CurrentBindingNotYetValid { key_id } => write!(f, "current authority binding for {key_id} is not yet valid"),
            Self::CurrentBindingExpired { key_id } => write!(f, "current authority binding for {key_id} has expired"),
            Self::PrincipalBindingChanged { key_id } => write!(f, "current principal binding changed for signer {key_id}"),
            Self::AuthorityBindingChanged { key_id } => write!(f, "current authority binding changed for signer {key_id}"),
        }
    }
}

impl Error for AttestationCurrentnessErrorV1 {}

#[cfg(test)]
mod tests;
