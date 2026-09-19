// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Content-addressed principal, organization, region, and role bindings.
//!
//! Signature identities alone are not enough to establish organizational or
//! regional independence. This module binds exact verification-key identities to
//! stable principals and their trusted metadata. The directory is still only a
//! content-addressed declaration in TRUST-006A; a later trust-root capability
//! must authorize an exact directory digest before these bindings become
//! authority-bearing.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::{FramedDigest, Sha256Digest, SignatureAlgorithm};

pub const TRUSTED_PRINCIPAL_DIRECTORY_SCHEMA: &str = "symthaea.trusted-principal-directory.v1";
const TRUSTED_PRINCIPAL_DIRECTORY_DOMAIN: &str =
    "symthaea.trusted-principal-directory.identity.v1";
pub const MAX_PRINCIPALS: usize = 4096;
pub const MAX_KEYS_PER_PRINCIPAL: usize = 32;
pub const MAX_PRINCIPAL_ID_BYTES: usize = 256;
pub const MAX_ORGANIZATION_ID_BYTES: usize = 256;
pub const MAX_REGION_ID_BYTES: usize = 256;
pub const MAX_PRINCIPAL_KEY_ID_BYTES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum TrustRole {
    Root,
    Freshness,
    KeyLifecycle,
    QualificationProfile,
    QualificationDecision,
    QualificationLifecycle,
    TransparencyLog,
    TransparencyWitness,
    EmergencyRecovery,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrustedKeyBinding {
    pub algorithm: SignatureAlgorithm,
    pub key_id: String,
    pub verification_key_sha256: Sha256Digest,
    pub roles: BTreeSet<TrustRole>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrustedPrincipal {
    pub principal_id: String,
    pub organization_id: String,
    pub region_id: String,
    pub keys: Vec<TrustedKeyBinding>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustedPrincipalDirectoryIssue {
    SequenceZero,
    EmptyPrincipals,
    TooManyPrincipals { actual: usize, maximum: usize },
    InvalidPrincipalId,
    PrincipalIdTooLong,
    DuplicatePrincipalId { principal_id: String },
    InvalidOrganizationId { principal_id: String },
    OrganizationIdTooLong { principal_id: String },
    InvalidRegionId { principal_id: String },
    RegionIdTooLong { principal_id: String },
    EmptyKeys { principal_id: String },
    TooManyKeys { principal_id: String, actual: usize, maximum: usize },
    InvalidAlgorithm { principal_id: String, key_id: String },
    InvalidKeyId { principal_id: String },
    KeyIdTooLong { principal_id: String, key_id: String },
    EmptyRoles { principal_id: String, key_id: String },
    DuplicateKeyWithinPrincipal { principal_id: String, key_id: String },
    KeyIdentityBoundToMultiplePrincipals {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrustedPrincipalDirectory {
    schema_version: String,
    sequence: u64,
    issued_at_unix_s: u64,
    principals: Vec<TrustedPrincipal>,
    directory_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrincipalResolutionError {
    UnknownKey,
    VerificationKeyDigestMismatch,
    RoleNotAllowed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedPrincipal<'a> {
    principal: &'a TrustedPrincipal,
    key: &'a TrustedKeyBinding,
}

impl<'a> ResolvedPrincipal<'a> {
    pub fn principal(&self) -> &'a TrustedPrincipal {
        self.principal
    }

    pub fn key(&self) -> &'a TrustedKeyBinding {
        self.key
    }

    pub fn principal_id(&self) -> &str {
        &self.principal.principal_id
    }

    pub fn organization_id(&self) -> &str {
        &self.principal.organization_id
    }

    pub fn region_id(&self) -> &str {
        &self.principal.region_id
    }
}

impl TrustedPrincipalDirectory {
    pub fn new(
        sequence: u64,
        issued_at_unix_s: u64,
        mut principals: Vec<TrustedPrincipal>,
    ) -> Result<Self, Vec<TrustedPrincipalDirectoryIssue>> {
        canonicalize_principals(&mut principals);
        let issues = validate_principals(sequence, &principals);
        if !issues.is_empty() {
            return Err(issues);
        }
        let directory_sha256 = directory_digest(sequence, issued_at_unix_s, &principals);
        Ok(Self {
            schema_version: TRUSTED_PRINCIPAL_DIRECTORY_SCHEMA.into(),
            sequence,
            issued_at_unix_s,
            principals,
            directory_sha256,
        })
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn issued_at_unix_s(&self) -> u64 {
        self.issued_at_unix_s
    }

    pub fn principals(&self) -> &[TrustedPrincipal] {
        &self.principals
    }

    pub fn directory_sha256(&self) -> &Sha256Digest {
        &self.directory_sha256
    }

    pub fn resolve_key(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        verification_key_sha256: &Sha256Digest,
    ) -> Result<ResolvedPrincipal<'_>, PrincipalResolutionError> {
        let Some((principal, key)) = self.principals.iter().find_map(|principal| {
            principal
                .keys
                .iter()
                .find(|key| &key.algorithm == algorithm && key.key_id == key_id)
                .map(|key| (principal, key))
        }) else {
            return Err(PrincipalResolutionError::UnknownKey);
        };
        if &key.verification_key_sha256 != verification_key_sha256 {
            return Err(PrincipalResolutionError::VerificationKeyDigestMismatch);
        }
        Ok(ResolvedPrincipal { principal, key })
    }

    pub fn resolve_key_for_role(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        verification_key_sha256: &Sha256Digest,
        role: TrustRole,
    ) -> Result<ResolvedPrincipal<'_>, PrincipalResolutionError> {
        let resolved = self.resolve_key(algorithm, key_id, verification_key_sha256)?;
        if !resolved.key.roles.contains(&role) {
            return Err(PrincipalResolutionError::RoleNotAllowed);
        }
        Ok(resolved)
    }

    /// Content addressing alone does not authorize this directory.
    pub const fn principal_authority_established(&self) -> bool {
        false
    }
}

fn canonicalize_principals(principals: &mut [TrustedPrincipal]) {
    for principal in principals.iter_mut() {
        principal.keys.sort_by(|left, right| {
            left.algorithm
                .cmp(&right.algorithm)
                .then(left.key_id.cmp(&right.key_id))
                .then(left.verification_key_sha256.cmp(&right.verification_key_sha256))
        });
    }
    principals.sort_by(|left, right| left.principal_id.cmp(&right.principal_id));
}

fn validate_principals(
    sequence: u64,
    principals: &[TrustedPrincipal],
) -> Vec<TrustedPrincipalDirectoryIssue> {
    let mut issues = Vec::new();
    if sequence == 0 {
        issues.push(TrustedPrincipalDirectoryIssue::SequenceZero);
    }
    if principals.is_empty() {
        issues.push(TrustedPrincipalDirectoryIssue::EmptyPrincipals);
    }
    if principals.len() > MAX_PRINCIPALS {
        issues.push(TrustedPrincipalDirectoryIssue::TooManyPrincipals {
            actual: principals.len(),
            maximum: MAX_PRINCIPALS,
        });
    }

    let mut principal_ids = BTreeSet::new();
    let mut global_keys = BTreeMap::new();
    for principal in principals {
        if !canonical_identifier(&principal.principal_id) {
            issues.push(TrustedPrincipalDirectoryIssue::InvalidPrincipalId);
        }
        if principal.principal_id.len() > MAX_PRINCIPAL_ID_BYTES {
            issues.push(TrustedPrincipalDirectoryIssue::PrincipalIdTooLong);
        }
        if !principal_ids.insert(principal.principal_id.clone()) {
            issues.push(TrustedPrincipalDirectoryIssue::DuplicatePrincipalId {
                principal_id: principal.principal_id.clone(),
            });
        }
        if !canonical_identifier(&principal.organization_id) {
            issues.push(TrustedPrincipalDirectoryIssue::InvalidOrganizationId {
                principal_id: principal.principal_id.clone(),
            });
        }
        if principal.organization_id.len() > MAX_ORGANIZATION_ID_BYTES {
            issues.push(TrustedPrincipalDirectoryIssue::OrganizationIdTooLong {
                principal_id: principal.principal_id.clone(),
            });
        }
        if !canonical_identifier(&principal.region_id) {
            issues.push(TrustedPrincipalDirectoryIssue::InvalidRegionId {
                principal_id: principal.principal_id.clone(),
            });
        }
        if principal.region_id.len() > MAX_REGION_ID_BYTES {
            issues.push(TrustedPrincipalDirectoryIssue::RegionIdTooLong {
                principal_id: principal.principal_id.clone(),
            });
        }
        if principal.keys.is_empty() {
            issues.push(TrustedPrincipalDirectoryIssue::EmptyKeys {
                principal_id: principal.principal_id.clone(),
            });
        }
        if principal.keys.len() > MAX_KEYS_PER_PRINCIPAL {
            issues.push(TrustedPrincipalDirectoryIssue::TooManyKeys {
                principal_id: principal.principal_id.clone(),
                actual: principal.keys.len(),
                maximum: MAX_KEYS_PER_PRINCIPAL,
            });
        }

        let mut local_keys = BTreeSet::new();
        for key in &principal.keys {
            if !key.algorithm.is_canonical() {
                issues.push(TrustedPrincipalDirectoryIssue::InvalidAlgorithm {
                    principal_id: principal.principal_id.clone(),
                    key_id: key.key_id.clone(),
                });
            }
            if !canonical_identifier(&key.key_id) {
                issues.push(TrustedPrincipalDirectoryIssue::InvalidKeyId {
                    principal_id: principal.principal_id.clone(),
                });
            }
            if key.key_id.len() > MAX_PRINCIPAL_KEY_ID_BYTES {
                issues.push(TrustedPrincipalDirectoryIssue::KeyIdTooLong {
                    principal_id: principal.principal_id.clone(),
                    key_id: key.key_id.clone(),
                });
            }
            if key.roles.is_empty() {
                issues.push(TrustedPrincipalDirectoryIssue::EmptyRoles {
                    principal_id: principal.principal_id.clone(),
                    key_id: key.key_id.clone(),
                });
            }
            let key_identity = (key.algorithm.clone(), key.key_id.clone());
            if !local_keys.insert(key_identity.clone()) {
                issues.push(TrustedPrincipalDirectoryIssue::DuplicateKeyWithinPrincipal {
                    principal_id: principal.principal_id.clone(),
                    key_id: key.key_id.clone(),
                });
            }
            if global_keys
                .insert(key_identity.clone(), principal.principal_id.clone())
                .is_some()
            {
                issues.push(
                    TrustedPrincipalDirectoryIssue::KeyIdentityBoundToMultiplePrincipals {
                        algorithm: key_identity.0,
                        key_id: key_identity.1,
                    },
                );
            }
        }
    }
    issues
}

fn directory_digest(
    sequence: u64,
    issued_at_unix_s: u64,
    principals: &[TrustedPrincipal],
) -> Sha256Digest {
    let mut digest = FramedDigest::new(TRUSTED_PRINCIPAL_DIRECTORY_DOMAIN);
    digest.text(TRUSTED_PRINCIPAL_DIRECTORY_SCHEMA);
    digest.text(&sequence.to_string());
    digest.text(&issued_at_unix_s.to_string());
    for principal in principals {
        digest.text("principal");
        digest.text(&principal.principal_id);
        digest.text(&principal.organization_id);
        digest.text(&principal.region_id);
        for key in &principal.keys {
            digest.text("key");
            digest_algorithm(&mut digest, &key.algorithm);
            digest.text(&key.key_id);
            digest.text(key.verification_key_sha256.as_str());
            for role in &key.roles {
                digest.text("role");
                digest.text(role_tag(*role));
            }
        }
    }
    digest.digest()
}

fn canonical_identifier(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'/' | b'-')
        })
}

fn digest_algorithm(digest: &mut FramedDigest, algorithm: &SignatureAlgorithm) {
    match algorithm {
        SignatureAlgorithm::Ed25519 => digest.text("builtin:ed25519"),
        SignatureAlgorithm::MlDsa65 => digest.text("builtin:ml-dsa-65"),
        SignatureAlgorithm::MlDsa87 => digest.text("builtin:ml-dsa-87"),
        SignatureAlgorithm::Other(name) => {
            digest.text("other");
            digest.text(name);
        }
    }
}

const fn role_tag(role: TrustRole) -> &'static str {
    match role {
        TrustRole::Root => "root",
        TrustRole::Freshness => "freshness",
        TrustRole::KeyLifecycle => "key-lifecycle",
        TrustRole::QualificationProfile => "qualification-profile",
        TrustRole::QualificationDecision => "qualification-decision",
        TrustRole::QualificationLifecycle => "qualification-lifecycle",
        TrustRole::TransparencyLog => "transparency-log",
        TrustRole::TransparencyWitness => "transparency-witness",
        TrustRole::EmergencyRecovery => "emergency-recovery",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(key_id: &str, roles: &[TrustRole]) -> TrustedKeyBinding {
        TrustedKeyBinding {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: key_id.into(),
            verification_key_sha256: Sha256Digest::of_bytes(key_id.as_bytes()),
            roles: roles.iter().copied().collect(),
        }
    }

    fn principal(
        principal_id: &str,
        organization_id: &str,
        region_id: &str,
        key_id: &str,
    ) -> TrustedPrincipal {
        TrustedPrincipal {
            principal_id: principal_id.into(),
            organization_id: organization_id.into(),
            region_id: region_id.into(),
            keys: vec![key(key_id, &[TrustRole::QualificationDecision])],
        }
    }

    #[test]
    fn directory_identity_is_order_independent() {
        let left = TrustedPrincipalDirectory::new(
            1,
            100,
            vec![
                principal("p-b", "org-b", "region-b", "key-b"),
                principal("p-a", "org-a", "region-a", "key-a"),
            ],
        )
        .unwrap();
        let right = TrustedPrincipalDirectory::new(
            1,
            100,
            vec![
                principal("p-a", "org-a", "region-a", "key-a"),
                principal("p-b", "org-b", "region-b", "key-b"),
            ],
        )
        .unwrap();
        assert_eq!(left.directory_sha256(), right.directory_sha256());
    }

    #[test]
    fn one_key_cannot_claim_multiple_principals() {
        let result = TrustedPrincipalDirectory::new(
            1,
            100,
            vec![
                principal("p-a", "org-a", "region-a", "shared"),
                principal("p-b", "org-b", "region-b", "shared"),
            ],
        );
        assert!(result.is_err());
    }

    #[test]
    fn role_resolution_uses_trusted_directory_metadata() {
        let directory = TrustedPrincipalDirectory::new(
            1,
            100,
            vec![principal("reviewer-1", "lab-1", "za-gp", "key-1")],
        )
        .unwrap();
        let digest = Sha256Digest::of_bytes(b"key-1");
        let resolved = directory
            .resolve_key_for_role(
                &SignatureAlgorithm::Ed25519,
                "key-1",
                &digest,
                TrustRole::QualificationDecision,
            )
            .unwrap();
        assert_eq!(resolved.principal_id(), "reviewer-1");
        assert_eq!(resolved.organization_id(), "lab-1");
        assert_eq!(resolved.region_id(), "za-gp");
        assert!(!directory.principal_authority_established());
    }
}
