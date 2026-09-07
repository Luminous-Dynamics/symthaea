// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Principal and authority-identity diversity over a verified attestation.
//!
//! This layer deliberately starts *after* cryptographic/profile/lifecycle
//! verification. It prevents a numeric signature quorum from being misread as
//! evidence of distinct people, principals, organizations, or administrative
//! roots.
//!
//! ```text
//! signature-entry count
//!     != key diversity
//!     != principal-id diversity
//!     != authority-id diversity
//!     != real-world independence
//! ```
//!
//! The resolver supplies exact-generation key -> principal/authority bindings.
//! Distinct IDs are meaningful only within that resolver's qualified trust model;
//! this module does not claim legal, organizational, physical, or custody
//! independence merely because strings differ.

use super::attestation::{
    AttestationSignatureAlgorithmV1, AttestationSubjectV1, MAX_ATTESTATION_KEY_ID_BYTES_V1,
    MAX_ATTESTATION_SIGNATURES_V1,
};
use super::attestation_profile::ExactProfileVerifiedAttestationV1;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const MAX_ATTESTATION_PRINCIPAL_ID_BYTES_V1: usize = 256;
pub const MAX_ATTESTATION_AUTHORITY_ID_BYTES_V1: usize = 256;

/// Exact current generation of signer -> principal/authority identity bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestationAuthorityBindingSnapshotV1 {
    pub generation_commitment: [u8; 32],
    pub valid_from_unix_s: u64,
    pub valid_until_unix_s: u64,
}

impl AttestationAuthorityBindingSnapshotV1 {
    pub fn validate(&self) -> Result<(), AttestationAuthorityErrorV1> {
        if self.generation_commitment == [0; 32] {
            return Err(AttestationAuthorityErrorV1::ZeroGenerationCommitment);
        }
        if self.valid_from_unix_s >= self.valid_until_unix_s {
            return Err(AttestationAuthorityErrorV1::InvalidSnapshotWindow);
        }
        Ok(())
    }
}

/// Exact-generation identity binding for one already-verified signature entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestationSignerAuthorityBindingV1 {
    pub generation_commitment: [u8; 32],
    pub algorithm: AttestationSignatureAlgorithmV1,
    pub key_id: String,
    pub principal_id: String,
    /// Optional administrative/organizational authority identity.
    ///
    /// This is optional because some deployments only qualify principal
    /// distinctness. A policy that asks for authority IDs makes it mandatory.
    pub authority_id: Option<String>,
    pub valid_from_unix_s: u64,
    pub valid_until_unix_s: u64,
}

impl AttestationSignerAuthorityBindingV1 {
    pub fn validate(&self) -> Result<(), AttestationAuthorityErrorV1> {
        if self.generation_commitment == [0; 32] {
            return Err(AttestationAuthorityErrorV1::ZeroGenerationCommitment);
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
            return Err(AttestationAuthorityErrorV1::InvalidBindingWindow {
                key_id: self.key_id.clone(),
            });
        }
        Ok(())
    }
}

/// Provider for exact-generation identity ownership metadata.
///
/// This provider does not verify signatures; the input witness has already passed
/// the exact-profile attestation verifier. Its job is only to bind each exact
/// verified key entry to stable principal/authority identities under one current
/// generation.
pub trait AttestationAuthorityResolverV1 {
    fn current_snapshot(&self) -> Result<AttestationAuthorityBindingSnapshotV1, String>;

    fn signer_binding(
        &self,
        generation_commitment: [u8; 32],
        algorithm: AttestationSignatureAlgorithmV1,
        key_id: &str,
    ) -> Result<Option<AttestationSignerAuthorityBindingV1>, String>;
}

/// Explicit identity-diversity policy.
///
/// No `Default` is provided. Callers must say which identity dimensions are
/// actually required. `minimum_distinct_authorities = None` means this decision
/// makes no authority-ID diversity claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestationAuthorityDiversityPolicyV1 {
    pub minimum_distinct_principals: usize,
    pub minimum_distinct_authorities: Option<usize>,
    pub allowed_principal_ids: Option<BTreeSet<String>>,
    pub allowed_authority_ids: Option<BTreeSet<String>>,
}

impl AttestationAuthorityDiversityPolicyV1 {
    pub fn validate(&self) -> Result<(), AttestationAuthorityErrorV1> {
        if self.minimum_distinct_principals == 0
            || self.minimum_distinct_principals > MAX_ATTESTATION_SIGNATURES_V1
        {
            return Err(AttestationAuthorityErrorV1::InvalidPolicy);
        }
        if matches!(self.minimum_distinct_authorities, Some(0))
            || self
                .minimum_distinct_authorities
                .is_some_and(|count| count > MAX_ATTESTATION_SIGNATURES_V1)
        {
            return Err(AttestationAuthorityErrorV1::InvalidPolicy);
        }

        if let Some(allowed) = &self.allowed_principal_ids {
            if allowed.is_empty() {
                return Err(AttestationAuthorityErrorV1::InvalidPolicy);
            }
            for principal_id in allowed {
                validate_id(
                    "policy.allowed_principal_id",
                    principal_id,
                    MAX_ATTESTATION_PRINCIPAL_ID_BYTES_V1,
                )?;
            }
        }

        if let Some(allowed) = &self.allowed_authority_ids {
            if allowed.is_empty() {
                return Err(AttestationAuthorityErrorV1::InvalidPolicy);
            }
            for authority_id in allowed {
                validate_id(
                    "policy.allowed_authority_id",
                    authority_id,
                    MAX_ATTESTATION_AUTHORITY_ID_BYTES_V1,
                )?;
            }
        }

        Ok(())
    }

    fn requires_authority_identity(&self) -> bool {
        self.minimum_distinct_authorities.is_some() || self.allowed_authority_ids.is_some()
    }
}

/// Exact identity mapping retained for one verified signature entry.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct QualifiedAttestationIdentityEntryV1 {
    pub algorithm: AttestationSignatureAlgorithmV1,
    pub key_id: String,
    pub principal_id: String,
    pub authority_id: Option<String>,
}

/// Positive in-process witness that explicit identity-diversity predicates passed.
///
/// Private fields and no serde derives keep it from becoming a portable authority
/// token. It remains attestation identity evidence, not effect authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedAttestationDiversityV1 {
    subject: AttestationSubjectV1,
    envelope_commitment: [u8; 32],
    trust_generation_commitment: [u8; 32],
    authority_binding_generation_commitment: [u8; 32],
    evaluation_time_unix_s: u64,
    natural_valid_until_unix_s: u64,
    entries: Vec<QualifiedAttestationIdentityEntryV1>,
    distinct_principal_ids: Vec<String>,
    distinct_authority_ids: Vec<String>,
}

impl QualifiedAttestationDiversityV1 {
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

    pub fn evaluation_time_unix_s(&self) -> u64 {
        self.evaluation_time_unix_s
    }

    pub fn natural_valid_until_unix_s(&self) -> u64 {
        self.natural_valid_until_unix_s
    }

    pub fn entries(&self) -> &[QualifiedAttestationIdentityEntryV1] {
        &self.entries
    }

    pub fn distinct_principal_ids(&self) -> &[String] {
        &self.distinct_principal_ids
    }

    pub fn distinct_authority_ids(&self) -> &[String] {
        &self.distinct_authority_ids
    }

    /// Time-only check. It does not prove either trust generation is still current.
    pub fn remains_within_time_bounds(&self, at_unix_s: u64) -> bool {
        self.evaluation_time_unix_s <= at_unix_s
            && at_unix_s < self.natural_valid_until_unix_s
    }
}

/// Qualify explicit principal/authority ID diversity for one exact-profile
/// verified attestation.
///
/// The stronger input type is deliberate: a witness created through the looser
/// base verifier is not admissible here. This preserves evidence that the exact
/// cryptographic scheme/wire/message profile checks ran before any diversity
/// claim is composed.
///
/// Identity qualification must occur in the same logical decision instant as the
/// exact-profile cryptographic verification. This avoids composing a fresh
/// identity map with an old positive signature witness without re-verifying the
/// base trust.
pub fn qualify_attestation_authority_diversity_v1(
    verified: &ExactProfileVerifiedAttestationV1,
    policy: &AttestationAuthorityDiversityPolicyV1,
    resolver: &dyn AttestationAuthorityResolverV1,
    evaluation_time_unix_s: u64,
) -> Result<QualifiedAttestationDiversityV1, AttestationAuthorityErrorV1> {
    policy.validate()?;

    if evaluation_time_unix_s != verified.evaluation_time_unix_s() {
        return Err(AttestationAuthorityErrorV1::EvaluationTimeMismatch {
            verified_at_unix_s: verified.evaluation_time_unix_s(),
            requested_at_unix_s: evaluation_time_unix_s,
        });
    }
    if !verified.remains_within_time_bounds(evaluation_time_unix_s) {
        return Err(AttestationAuthorityErrorV1::BaseWitnessOutsideTimeBounds);
    }

    let snapshot = resolver
        .current_snapshot()
        .map_err(AttestationAuthorityErrorV1::Resolver)?;
    snapshot.validate()?;
    if evaluation_time_unix_s < snapshot.valid_from_unix_s {
        return Err(AttestationAuthorityErrorV1::SnapshotNotYetValid);
    }
    if evaluation_time_unix_s >= snapshot.valid_until_unix_s {
        return Err(AttestationAuthorityErrorV1::SnapshotExpired);
    }

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
            .map_err(AttestationAuthorityErrorV1::Resolver)?
            .ok_or_else(|| AttestationAuthorityErrorV1::MissingBinding {
                algorithm: signer.algorithm,
                key_id: signer.key_id.clone(),
            })?;

        binding.validate()?;
        if binding.generation_commitment != snapshot.generation_commitment {
            return Err(AttestationAuthorityErrorV1::BindingGenerationMismatch {
                key_id: signer.key_id.clone(),
            });
        }
        if binding.algorithm != signer.algorithm || binding.key_id != signer.key_id {
            return Err(AttestationAuthorityErrorV1::BindingIdentityMismatch {
                requested_key_id: signer.key_id.clone(),
                returned_key_id: binding.key_id,
            });
        }
        if evaluation_time_unix_s < binding.valid_from_unix_s {
            return Err(AttestationAuthorityErrorV1::BindingNotYetValid {
                key_id: signer.key_id.clone(),
            });
        }
        if evaluation_time_unix_s >= binding.valid_until_unix_s {
            return Err(AttestationAuthorityErrorV1::BindingExpired {
                key_id: signer.key_id.clone(),
            });
        }

        if let Some(allowed) = &policy.allowed_principal_ids {
            if !allowed.contains(&binding.principal_id) {
                return Err(AttestationAuthorityErrorV1::PrincipalNotAllowed(
                    binding.principal_id,
                ));
            }
        }

        if policy.requires_authority_identity() && binding.authority_id.is_none() {
            return Err(AttestationAuthorityErrorV1::AuthorityIdentityRequired {
                key_id: signer.key_id.clone(),
            });
        }

        if let Some(authority_id) = &binding.authority_id {
            if let Some(allowed) = &policy.allowed_authority_ids {
                if !allowed.contains(authority_id) {
                    return Err(AttestationAuthorityErrorV1::AuthorityNotAllowed(
                        authority_id.clone(),
                    ));
                }
            }
            authorities.insert(authority_id.clone());
        }

        principals.insert(binding.principal_id.clone());
        natural_valid_until_unix_s =
            natural_valid_until_unix_s.min(binding.valid_until_unix_s);
        entries.push(QualifiedAttestationIdentityEntryV1 {
            algorithm: signer.algorithm,
            key_id: signer.key_id.clone(),
            principal_id: binding.principal_id,
            authority_id: binding.authority_id,
        });
    }

    if principals.len() < policy.minimum_distinct_principals {
        return Err(AttestationAuthorityErrorV1::InsufficientDistinctPrincipals {
            actual: principals.len(),
            required: policy.minimum_distinct_principals,
        });
    }
    if let Some(required) = policy.minimum_distinct_authorities {
        if authorities.len() < required {
            return Err(AttestationAuthorityErrorV1::InsufficientDistinctAuthorities {
                actual: authorities.len(),
                required,
            });
        }
    }

    entries.sort();
    Ok(QualifiedAttestationDiversityV1 {
        subject: verified.subject().clone(),
        envelope_commitment: verified.envelope_commitment(),
        trust_generation_commitment: verified.trust_generation_commitment(),
        authority_binding_generation_commitment: snapshot.generation_commitment,
        evaluation_time_unix_s,
        natural_valid_until_unix_s,
        entries,
        distinct_principal_ids: principals.into_iter().collect(),
        distinct_authority_ids: authorities.into_iter().collect(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttestationAuthorityErrorV1 {
    InvalidPolicy,
    EmptyField(&'static str),
    FieldTooLong {
        field: &'static str,
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    ZeroGenerationCommitment,
    InvalidSnapshotWindow,
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
    PrincipalNotAllowed(String),
    AuthorityIdentityRequired {
        key_id: String,
    },
    AuthorityNotAllowed(String),
    InsufficientDistinctPrincipals {
        actual: usize,
        required: usize,
    },
    InsufficientDistinctAuthorities {
        actual: usize,
        required: usize,
    },
}

impl fmt::Display for AttestationAuthorityErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPolicy => write!(f, "invalid attestation authority-diversity policy"),
            Self::EmptyField(field) => write!(f, "attestation authority field {field} is empty"),
            Self::FieldTooLong {
                field,
                actual_bytes,
                maximum_bytes,
            } => write!(
                f,
                "attestation authority field {field} is {actual_bytes} bytes; maximum is {maximum_bytes}"
            ),
            Self::ZeroGenerationCommitment => {
                write!(f, "attestation authority generation may not be all-zero")
            }
            Self::InvalidSnapshotWindow => {
                write!(f, "invalid attestation authority snapshot validity window")
            }
            Self::Resolver(error) => write!(f, "attestation authority resolver failed: {error}"),
            Self::EvaluationTimeMismatch {
                verified_at_unix_s,
                requested_at_unix_s,
            } => write!(
                f,
                "authority qualification time {requested_at_unix_s} differs from base verification time {verified_at_unix_s}"
            ),
            Self::BaseWitnessOutsideTimeBounds => {
                write!(f, "base attestation witness is outside its natural time bounds")
            }
            Self::SnapshotNotYetValid => write!(f, "authority binding snapshot is not yet valid"),
            Self::SnapshotExpired => write!(f, "authority binding snapshot has expired"),
            Self::MissingBinding { algorithm, key_id } => {
                write!(f, "missing authority binding for {algorithm:?}:{key_id}")
            }
            Self::InvalidBindingWindow { key_id } => {
                write!(f, "invalid authority binding window for key {key_id}")
            }
            Self::BindingGenerationMismatch { key_id } => write!(
                f,
                "authority binding for key {key_id} does not belong to the current binding generation"
            ),
            Self::BindingIdentityMismatch {
                requested_key_id,
                returned_key_id,
            } => write!(
                f,
                "authority resolver returned key {returned_key_id} for requested key {requested_key_id}"
            ),
            Self::BindingNotYetValid { key_id } => {
                write!(f, "authority binding for key {key_id} is not yet valid")
            }
            Self::BindingExpired { key_id } => {
                write!(f, "authority binding for key {key_id} has expired")
            }
            Self::PrincipalNotAllowed(principal_id) => {
                write!(f, "attestation principal {principal_id} is not allowed by policy")
            }
            Self::AuthorityIdentityRequired { key_id } => write!(
                f,
                "policy requires an authority identity but key {key_id} has none"
            ),
            Self::AuthorityNotAllowed(authority_id) => {
                write!(f, "attestation authority {authority_id} is not allowed by policy")
            }
            Self::InsufficientDistinctPrincipals { actual, required } => write!(
                f,
                "attestation has {actual} distinct principal ids; policy requires {required}"
            ),
            Self::InsufficientDistinctAuthorities { actual, required } => write!(
                f,
                "attestation has {actual} distinct authority ids; policy requires {required}"
            ),
        }
    }
}

impl Error for AttestationAuthorityErrorV1 {}

fn validate_id(
    field: &'static str,
    value: &str,
    maximum_bytes: usize,
) -> Result<(), AttestationAuthorityErrorV1> {
    if value.trim().is_empty() {
        return Err(AttestationAuthorityErrorV1::EmptyField(field));
    }
    if value.len() > maximum_bytes {
        return Err(AttestationAuthorityErrorV1::FieldTooLong {
            field,
            actual_bytes: value.len(),
            maximum_bytes,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
