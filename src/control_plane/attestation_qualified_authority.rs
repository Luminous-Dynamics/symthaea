// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Namespace-qualified principal/authority diversity over a cryptographically
//! bound exact-profile attestation witness.
//!
//! This is the consequential composition layer for attestation identity claims:
//!
//! ```text
//! exact profile bound into crypto operation
//!     + exact current identity-binding generation
//!     + exact identity namespace semantics
//!     + explicit diversity policy
//!         -> namespace-qualified diversity witness
//! ```
//!
//! The resulting witness remains evidence about signer identity diversity. It is
//! not Forge, Xenia, deployment, promotion, or other effect authority.

use super::attestation::{
    AttestationSignatureAlgorithmV1, AttestationSubjectV1, MAX_ATTESTATION_KEY_ID_BYTES_V1,
    MAX_ATTESTATION_SIGNATURES_V1,
};
use super::attestation_crypto_profile::CryptographicallyBoundExactProfileAttestationV1;
use super::attestation_identity_namespace::{
    AttestationIdentityKindV1, AttestationIdentityNamespaceErrorV1,
    AttestationIdentityNamespaceSetV1, QualifiedAttestationIdentityV1,
};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const MAX_ATTESTATION_PRINCIPAL_ID_BYTES_V1: usize = 256;
pub const MAX_ATTESTATION_AUTHORITY_ID_BYTES_V1: usize = 256;

/// Exact current generation of signer→identity bindings and the exact namespace
/// semantics under which those identities are meaningful.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedAttestationAuthoritySnapshotV1 {
    pub generation_commitment: [u8; 32],
    pub valid_from_unix_s: u64,
    pub valid_until_unix_s: u64,
    pub identity_namespaces: AttestationIdentityNamespaceSetV1,
}

impl QualifiedAttestationAuthoritySnapshotV1 {
    pub fn validate(&self) -> Result<(), QualifiedAttestationAuthorityErrorV1> {
        if self.generation_commitment == [0; 32] {
            return Err(QualifiedAttestationAuthorityErrorV1::ZeroGenerationCommitment);
        }
        if self.valid_from_unix_s >= self.valid_until_unix_s {
            return Err(QualifiedAttestationAuthorityErrorV1::InvalidSnapshotWindow);
        }
        self.identity_namespaces
            .validate()
            .map_err(QualifiedAttestationAuthorityErrorV1::IdentityNamespace)
    }
}

/// Exact-generation identity binding for one already cryptographically verified
/// signature entry. Bare strings are accepted only at the resolver boundary and
/// are immediately qualified through the snapshot's namespace semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedAttestationSignerBindingV1 {
    pub generation_commitment: [u8; 32],
    pub algorithm: AttestationSignatureAlgorithmV1,
    pub key_id: String,
    pub principal_id: String,
    pub authority_id: Option<String>,
    pub valid_from_unix_s: u64,
    pub valid_until_unix_s: u64,
}

impl QualifiedAttestationSignerBindingV1 {
    pub fn validate(&self) -> Result<(), QualifiedAttestationAuthorityErrorV1> {
        if self.generation_commitment == [0; 32] {
            return Err(QualifiedAttestationAuthorityErrorV1::ZeroGenerationCommitment);
        }
        validate_id(
            "binding.key_id",
            &self.key_id,
            MAX_ATTESTATION_KEY_ID_BYTES_V1,
        )?;
        validate_id(
            "binding.principal_id",
            &self.principal_id,
            MAX_ATTESTATION_PRINCIPAL_ID_BYTES_V1,
        )?;
        if let Some(authority_id) = &self.authority_id {
            validate_id(
                "binding.authority_id",
                authority_id,
                MAX_ATTESTATION_AUTHORITY_ID_BYTES_V1,
            )?;
        }
        if self.valid_from_unix_s >= self.valid_until_unix_s {
            return Err(QualifiedAttestationAuthorityErrorV1::InvalidBindingWindow {
                key_id: self.key_id.clone(),
            });
        }
        Ok(())
    }
}

/// Resolver for one exact identity-binding generation.
///
/// It does not verify signatures. The input witness already proves lifecycle and
/// exact-profile cryptographic verification. Its responsibility is the current
/// key→principal/authority mapping plus namespace semantics.
pub trait QualifiedAttestationAuthorityResolverV1 {
    fn current_snapshot(&self) -> Result<QualifiedAttestationAuthoritySnapshotV1, String>;

    fn signer_binding(
        &self,
        generation_commitment: [u8; 32],
        algorithm: AttestationSignatureAlgorithmV1,
        key_id: &str,
    ) -> Result<Option<QualifiedAttestationSignerBindingV1>, String>;
}

/// Explicit diversity policy over namespace-qualified identities.
///
/// No default exists: deployments must state exactly which identity dimensions
/// they require. All allowlist members are fully qualified identities, never bare
/// strings whose namespace could be inferred later.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedAttestationDiversityPolicyV1 {
    pub minimum_distinct_principals: usize,
    pub minimum_distinct_authorities: Option<usize>,
    pub allowed_principals: Option<BTreeSet<QualifiedAttestationIdentityV1>>,
    pub allowed_authorities: Option<BTreeSet<QualifiedAttestationIdentityV1>>,
}

impl QualifiedAttestationDiversityPolicyV1 {
    pub fn validate(&self) -> Result<(), QualifiedAttestationAuthorityErrorV1> {
        if self.minimum_distinct_principals == 0
            || self.minimum_distinct_principals > MAX_ATTESTATION_SIGNATURES_V1
        {
            return Err(QualifiedAttestationAuthorityErrorV1::InvalidPolicy);
        }
        if matches!(self.minimum_distinct_authorities, Some(0))
            || self
                .minimum_distinct_authorities
                .is_some_and(|count| count > MAX_ATTESTATION_SIGNATURES_V1)
        {
            return Err(QualifiedAttestationAuthorityErrorV1::InvalidPolicy);
        }

        if let Some(allowed) = &self.allowed_principals {
            if allowed.is_empty() {
                return Err(QualifiedAttestationAuthorityErrorV1::InvalidPolicy);
            }
            for identity in allowed {
                identity
                    .validate()
                    .map_err(QualifiedAttestationAuthorityErrorV1::IdentityNamespace)?;
                if identity.kind != AttestationIdentityKindV1::Principal {
                    return Err(QualifiedAttestationAuthorityErrorV1::InvalidPolicyIdentityKind {
                        expected: AttestationIdentityKindV1::Principal,
                        actual: identity.kind,
                    });
                }
            }
        }

        if let Some(allowed) = &self.allowed_authorities {
            if allowed.is_empty() {
                return Err(QualifiedAttestationAuthorityErrorV1::InvalidPolicy);
            }
            for identity in allowed {
                identity
                    .validate()
                    .map_err(QualifiedAttestationAuthorityErrorV1::IdentityNamespace)?;
                if identity.kind != AttestationIdentityKindV1::Authority {
                    return Err(QualifiedAttestationAuthorityErrorV1::InvalidPolicyIdentityKind {
                        expected: AttestationIdentityKindV1::Authority,
                        actual: identity.kind,
                    });
                }
            }
        }
        Ok(())
    }

    fn requires_authority_identity(&self) -> bool {
        self.minimum_distinct_authorities.is_some() || self.allowed_authorities.is_some()
    }
}

/// Exact namespace-qualified mapping retained for one verified signature entry.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct QualifiedAttestationAuthorityEntryV1 {
    pub algorithm: AttestationSignatureAlgorithmV1,
    pub key_id: String,
    pub principal: QualifiedAttestationIdentityV1,
    pub authority: Option<QualifiedAttestationIdentityV1>,
}

/// Positive in-process witness that exact-profile crypto, namespace semantics,
/// current binding generation and explicit diversity predicates all composed.
///
/// Private fields and no serde derives keep it from becoming a portable authority
/// token. It remains identity evidence and grants no external effect permission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedAttestationAuthorityDiversityV1 {
    subject: AttestationSubjectV1,
    envelope_commitment: [u8; 32],
    trust_generation_commitment: [u8; 32],
    authority_binding_generation_commitment: [u8; 32],
    identity_namespace_set_commitment: [u8; 32],
    evaluation_time_unix_s: u64,
    natural_valid_until_unix_s: u64,
    entries: Vec<QualifiedAttestationAuthorityEntryV1>,
    distinct_principals: Vec<QualifiedAttestationIdentityV1>,
    distinct_authorities: Vec<QualifiedAttestationIdentityV1>,
}

impl QualifiedAttestationAuthorityDiversityV1 {
    pub fn subject(&self) -> &AttestationSubjectV1 {
        &self.subject
    }

    pub fn envelope_commitment(&self) -> [u8; 32] {
        self.envelope_commitment
    }

    pub fn trust_generation_commitment(&self) -> [u8; 32] {
        self.trust_generation_commitment
    }

    pub fn authority_binding_generation_commitment(&self) -> [u8; 32] {
        self.authority_binding_generation_commitment
    }

    pub fn identity_namespace_set_commitment(&self) -> [u8; 32] {
        self.identity_namespace_set_commitment
    }

    pub fn evaluation_time_unix_s(&self) -> u64 {
        self.evaluation_time_unix_s
    }

    pub fn natural_valid_until_unix_s(&self) -> u64 {
        self.natural_valid_until_unix_s
    }

    pub fn entries(&self) -> &[QualifiedAttestationAuthorityEntryV1] {
        &self.entries
    }

    pub fn distinct_principals(&self) -> &[QualifiedAttestationIdentityV1] {
        &self.distinct_principals
    }

    pub fn distinct_authorities(&self) -> &[QualifiedAttestationIdentityV1] {
        &self.distinct_authorities
    }

    /// Time-only check. It does not prove either trust generation or the identity
    /// namespace semantics remain current after the original verification.
    pub fn remains_within_time_bounds(&self, at_unix_s: u64) -> bool {
        self.evaluation_time_unix_s <= at_unix_s
            && at_unix_s < self.natural_valid_until_unix_s
    }
}

/// Qualify namespace-bound principal/authority diversity at the exact same logical
/// decision instant as the cryptographically bound profile verification.
pub fn qualify_attestation_authority_diversity_v1(
    verified: &CryptographicallyBoundExactProfileAttestationV1,
    policy: &QualifiedAttestationDiversityPolicyV1,
    resolver: &dyn QualifiedAttestationAuthorityResolverV1,
    evaluation_time_unix_s: u64,
) -> Result<QualifiedAttestationAuthorityDiversityV1, QualifiedAttestationAuthorityErrorV1> {
    policy.validate()?;

    if evaluation_time_unix_s != verified.evaluation_time_unix_s() {
        return Err(QualifiedAttestationAuthorityErrorV1::EvaluationTimeMismatch {
            verified_at_unix_s: verified.evaluation_time_unix_s(),
            requested_at_unix_s: evaluation_time_unix_s,
        });
    }
    if !verified.remains_within_time_bounds(evaluation_time_unix_s) {
        return Err(QualifiedAttestationAuthorityErrorV1::BaseWitnessOutsideTimeBounds);
    }

    let snapshot = resolver
        .current_snapshot()
        .map_err(QualifiedAttestationAuthorityErrorV1::Resolver)?;
    snapshot.validate()?;
    if evaluation_time_unix_s < snapshot.valid_from_unix_s {
        return Err(QualifiedAttestationAuthorityErrorV1::SnapshotNotYetValid);
    }
    if evaluation_time_unix_s >= snapshot.valid_until_unix_s {
        return Err(QualifiedAttestationAuthorityErrorV1::SnapshotExpired);
    }

    let principal_namespace_commitment = snapshot
        .identity_namespaces
        .principal
        .canonical_commitment_v1(AttestationIdentityKindV1::Principal)
        .map_err(QualifiedAttestationAuthorityErrorV1::IdentityNamespace)?;
    let authority_namespace_commitment = snapshot
        .identity_namespaces
        .authority
        .as_ref()
        .map(|namespace| {
            namespace.canonical_commitment_v1(AttestationIdentityKindV1::Authority)
        })
        .transpose()
        .map_err(QualifiedAttestationAuthorityErrorV1::IdentityNamespace)?;

    validate_policy_namespaces(
        policy,
        principal_namespace_commitment,
        authority_namespace_commitment,
    )?;

    let base = verified.attestation();
    let mut entries = Vec::with_capacity(base.valid_signers().len());
    let mut principals = BTreeSet::new();
    let mut authorities = BTreeSet::new();
    let mut natural_valid_until_unix_s = verified
        .natural_valid_until_unix_s()
        .min(snapshot.valid_until_unix_s);

    for signer in base.valid_signers() {
        let binding = resolver
            .signer_binding(
                snapshot.generation_commitment,
                signer.algorithm,
                &signer.key_id,
            )
            .map_err(QualifiedAttestationAuthorityErrorV1::Resolver)?
            .ok_or_else(|| QualifiedAttestationAuthorityErrorV1::MissingBinding {
                algorithm: signer.algorithm,
                key_id: signer.key_id.clone(),
            })?;

        binding.validate()?;
        if binding.generation_commitment != snapshot.generation_commitment {
            return Err(QualifiedAttestationAuthorityErrorV1::BindingGenerationMismatch {
                key_id: signer.key_id.clone(),
            });
        }
        if binding.algorithm != signer.algorithm || binding.key_id != signer.key_id {
            return Err(QualifiedAttestationAuthorityErrorV1::BindingIdentityMismatch {
                requested_key_id: signer.key_id.clone(),
                returned_key_id: binding.key_id,
            });
        }
        if evaluation_time_unix_s < binding.valid_from_unix_s {
            return Err(QualifiedAttestationAuthorityErrorV1::BindingNotYetValid {
                key_id: signer.key_id.clone(),
            });
        }
        if evaluation_time_unix_s >= binding.valid_until_unix_s {
            return Err(QualifiedAttestationAuthorityErrorV1::BindingExpired {
                key_id: signer.key_id.clone(),
            });
        }

        let principal = snapshot
            .identity_namespaces
            .qualify_principal(binding.principal_id)
            .map_err(QualifiedAttestationAuthorityErrorV1::IdentityNamespace)?;
        if let Some(allowed) = &policy.allowed_principals {
            if !allowed.contains(&principal) {
                return Err(QualifiedAttestationAuthorityErrorV1::PrincipalNotAllowed(
                    principal,
                ));
            }
        }

        if policy.requires_authority_identity() && binding.authority_id.is_none() {
            return Err(QualifiedAttestationAuthorityErrorV1::AuthorityIdentityRequired {
                key_id: signer.key_id.clone(),
            });
        }

        let authority = binding
            .authority_id
            .map(|authority_id| snapshot.identity_namespaces.qualify_authority(authority_id))
            .transpose()
            .map_err(QualifiedAttestationAuthorityErrorV1::IdentityNamespace)?;

        if let Some(authority) = &authority {
            if let Some(allowed) = &policy.allowed_authorities {
                if !allowed.contains(authority) {
                    return Err(QualifiedAttestationAuthorityErrorV1::AuthorityNotAllowed(
                        authority.clone(),
                    ));
                }
            }
            authorities.insert(authority.clone());
        }

        principals.insert(principal.clone());
        natural_valid_until_unix_s =
            natural_valid_until_unix_s.min(binding.valid_until_unix_s);
        entries.push(QualifiedAttestationAuthorityEntryV1 {
            algorithm: signer.algorithm,
            key_id: signer.key_id.clone(),
            principal,
            authority,
        });
    }

    if principals.len() < policy.minimum_distinct_principals {
        return Err(
            QualifiedAttestationAuthorityErrorV1::InsufficientDistinctPrincipals {
                actual: principals.len(),
                required: policy.minimum_distinct_principals,
            },
        );
    }
    if let Some(required) = policy.minimum_distinct_authorities {
        if authorities.len() < required {
            return Err(
                QualifiedAttestationAuthorityErrorV1::InsufficientDistinctAuthorities {
                    actual: authorities.len(),
                    required,
                },
            );
        }
    }

    entries.sort();
    let identity_namespace_set_commitment = snapshot
        .identity_namespaces
        .canonical_commitment_v1()
        .map_err(QualifiedAttestationAuthorityErrorV1::IdentityNamespace)?;

    Ok(QualifiedAttestationAuthorityDiversityV1 {
        subject: verified.subject().clone(),
        envelope_commitment: verified.envelope_commitment(),
        trust_generation_commitment: verified.trust_generation_commitment(),
        authority_binding_generation_commitment: snapshot.generation_commitment,
        identity_namespace_set_commitment,
        evaluation_time_unix_s,
        natural_valid_until_unix_s,
        entries,
        distinct_principals: principals.into_iter().collect(),
        distinct_authorities: authorities.into_iter().collect(),
    })
}

fn validate_policy_namespaces(
    policy: &QualifiedAttestationDiversityPolicyV1,
    principal_namespace_commitment: [u8; 32],
    authority_namespace_commitment: Option<[u8; 32]>,
) -> Result<(), QualifiedAttestationAuthorityErrorV1> {
    if let Some(allowed) = &policy.allowed_principals {
        for identity in allowed {
            if identity.namespace_commitment != principal_namespace_commitment {
                return Err(QualifiedAttestationAuthorityErrorV1::PolicyIdentityNamespaceMismatch {
                    kind: AttestationIdentityKindV1::Principal,
                });
            }
        }
    }

    if policy.requires_authority_identity() && authority_namespace_commitment.is_none() {
        return Err(QualifiedAttestationAuthorityErrorV1::AuthorityNamespaceRequired);
    }
    if let Some(allowed) = &policy.allowed_authorities {
        let expected = authority_namespace_commitment
            .ok_or(QualifiedAttestationAuthorityErrorV1::AuthorityNamespaceRequired)?;
        for identity in allowed {
            if identity.namespace_commitment != expected {
                return Err(QualifiedAttestationAuthorityErrorV1::PolicyIdentityNamespaceMismatch {
                    kind: AttestationIdentityKindV1::Authority,
                });
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualifiedAttestationAuthorityErrorV1 {
    InvalidPolicy,
    EmptyField(&'static str),
    FieldTooLong {
        field: &'static str,
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    ZeroGenerationCommitment,
    InvalidSnapshotWindow,
    IdentityNamespace(AttestationIdentityNamespaceErrorV1),
    Resolver(String),
    EvaluationTimeMismatch {
        verified_at_unix_s: u64,
        requested_at_unix_s: u64,
    },
    BaseWitnessOutsideTimeBounds,
    SnapshotNotYetValid,
    SnapshotExpired,
    MissingBinding {
        algorithm: AttestationSignatureAlgorithmV1,
        key_id: String,
    },
    InvalidBindingWindow {
        key_id: String,
    },
    BindingGenerationMismatch {
        key_id: String,
    },
    BindingIdentityMismatch {
        requested_key_id: String,
        returned_key_id: String,
    },
    BindingNotYetValid {
        key_id: String,
    },
    BindingExpired {
        key_id: String,
    },
    InvalidPolicyIdentityKind {
        expected: AttestationIdentityKindV1,
        actual: AttestationIdentityKindV1,
    },
    PolicyIdentityNamespaceMismatch {
        kind: AttestationIdentityKindV1,
    },
    PrincipalNotAllowed(QualifiedAttestationIdentityV1),
    AuthorityNamespaceRequired,
    AuthorityIdentityRequired {
        key_id: String,
    },
    AuthorityNotAllowed(QualifiedAttestationIdentityV1),
    InsufficientDistinctPrincipals {
        actual: usize,
        required: usize,
    },
    InsufficientDistinctAuthorities {
        actual: usize,
        required: usize,
    },
}

impl fmt::Display for QualifiedAttestationAuthorityErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPolicy => write!(f, "invalid qualified attestation diversity policy"),
            Self::EmptyField(field) => write!(f, "{field} must be non-empty"),
            Self::FieldTooLong {
                field,
                actual_bytes,
                maximum_bytes,
            } => write!(
                f,
                "{field} is {actual_bytes} bytes; maximum is {maximum_bytes}"
            ),
            Self::ZeroGenerationCommitment => write!(f, "generation commitment must not be zero"),
            Self::InvalidSnapshotWindow => write!(f, "invalid authority-binding snapshot window"),
            Self::IdentityNamespace(error) => write!(f, "identity namespace error: {error}"),
            Self::Resolver(error) => write!(f, "authority resolver failed: {error}"),
            Self::EvaluationTimeMismatch {
                verified_at_unix_s,
                requested_at_unix_s,
            } => write!(
                f,
                "identity qualification time {requested_at_unix_s} differs from cryptographic verification time {verified_at_unix_s}"
            ),
            Self::BaseWitnessOutsideTimeBounds => {
                write!(f, "cryptographically bound witness is outside its time bounds")
            }
            Self::SnapshotNotYetValid => write!(f, "authority-binding snapshot is not yet valid"),
            Self::SnapshotExpired => write!(f, "authority-binding snapshot has expired"),
            Self::MissingBinding { algorithm, key_id } => {
                write!(f, "missing authority binding for {algorithm:?}/{key_id}")
            }
            Self::InvalidBindingWindow { key_id } => {
                write!(f, "authority binding for {key_id} has invalid validity window")
            }
            Self::BindingGenerationMismatch { key_id } => {
                write!(f, "authority binding generation mismatch for {key_id}")
            }
            Self::BindingIdentityMismatch {
                requested_key_id,
                returned_key_id,
            } => write!(
                f,
                "authority binding identity mismatch: requested {requested_key_id}, got {returned_key_id}"
            ),
            Self::BindingNotYetValid { key_id } => {
                write!(f, "authority binding for {key_id} is not yet valid")
            }
            Self::BindingExpired { key_id } => {
                write!(f, "authority binding for {key_id} has expired")
            }
            Self::InvalidPolicyIdentityKind { expected, actual } => write!(
                f,
                "policy identity kind mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::PolicyIdentityNamespaceMismatch { kind } => {
                write!(f, "policy {kind:?} identity belongs to a different namespace")
            }
            Self::PrincipalNotAllowed(identity) => {
                write!(f, "principal is not allowed: {}", identity.id)
            }
            Self::AuthorityNamespaceRequired => {
                write!(f, "authority diversity requires an explicit authority namespace")
            }
            Self::AuthorityIdentityRequired { key_id } => {
                write!(f, "authority identity is required for signer {key_id}")
            }
            Self::AuthorityNotAllowed(identity) => {
                write!(f, "authority is not allowed: {}", identity.id)
            }
            Self::InsufficientDistinctPrincipals { actual, required } => write!(
                f,
                "insufficient distinct principals: have {actual}, require {required}"
            ),
            Self::InsufficientDistinctAuthorities { actual, required } => write!(
                f,
                "insufficient distinct authorities: have {actual}, require {required}"
            ),
        }
    }
}

impl Error for QualifiedAttestationAuthorityErrorV1 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::IdentityNamespace(error) => Some(error),
            _ => None,
        }
    }
}

fn validate_id(
    field: &'static str,
    value: &str,
    maximum_bytes: usize,
) -> Result<(), QualifiedAttestationAuthorityErrorV1> {
    if value.is_empty() {
        return Err(QualifiedAttestationAuthorityErrorV1::EmptyField(field));
    }
    if value.len() > maximum_bytes {
        return Err(QualifiedAttestationAuthorityErrorV1::FieldTooLong {
            field,
            actual_bytes: value.len(),
            maximum_bytes,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
