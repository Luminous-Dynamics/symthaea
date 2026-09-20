// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Structural signature-feasibility proof for frozen trust roots.
//!
//! Historical `FrozenTrustRoot` identities remain unchanged. This module adds a
//! separate proof that every role policy can, in principle, satisfy its valid-
//! signature threshold using the exact root-bound principal directory. The proof
//! is structural only: it does not establish that any key actually signed, that
//! any trust snapshot is current, or that the root has institutional authority.

use serde::Serialize;

use crate::{FramedDigest, FrozenTrustRoot, Sha256Digest, TrustRole, TrustedPrincipalDirectory};

const TRUST_ROOT_FEASIBILITY_DOMAIN: &str =
    "symthaea.trust-root-signature-feasibility.identity.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RoleSignatureFeasibility {
    role: TrustRole,
    eligible_role_bound_keys: usize,
    minimum_valid_signatures: usize,
}

impl RoleSignatureFeasibility {
    pub fn role(&self) -> TrustRole {
        self.role
    }

    pub fn eligible_role_bound_keys(&self) -> usize {
        self.eligible_role_bound_keys
    }

    pub fn minimum_valid_signatures(&self) -> usize {
        self.minimum_valid_signatures
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustRootFeasibilityFinding {
    PrincipalDirectorySequenceMismatch {
        root: u64,
        directory: u64,
    },
    PrincipalDirectoryDigestMismatch,
    RoleHasTooFewEligibleKeys {
        role: TrustRole,
        actual: usize,
        required: usize,
    },
}

/// Non-deserializable proof that every frozen role has enough eligible role-
/// bound keys to meet its signature-count threshold in the exact bound directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrustRootSignatureFeasibilityProof {
    root_sha256: Sha256Digest,
    principal_directory_sha256: Sha256Digest,
    roles: Vec<RoleSignatureFeasibility>,
    proof_sha256: Sha256Digest,
}

impl TrustRootSignatureFeasibilityProof {
    pub fn root_sha256(&self) -> &Sha256Digest {
        &self.root_sha256
    }

    pub fn principal_directory_sha256(&self) -> &Sha256Digest {
        &self.principal_directory_sha256
    }

    pub fn roles(&self) -> &[RoleSignatureFeasibility] {
        &self.roles
    }

    pub fn proof_sha256(&self) -> &Sha256Digest {
        &self.proof_sha256
    }

    pub const fn signature_thresholds_structurally_feasible(&self) -> bool {
        true
    }

    pub const fn institutional_authority_established(&self) -> bool {
        false
    }
}

pub fn prove_trust_root_signature_feasibility(
    root: &FrozenTrustRoot,
    directory: &TrustedPrincipalDirectory,
) -> Result<TrustRootSignatureFeasibilityProof, Vec<TrustRootFeasibilityFinding>> {
    let mut findings = Vec::new();

    if root.principal_directory_sequence() != directory.sequence() {
        findings.push(TrustRootFeasibilityFinding::PrincipalDirectorySequenceMismatch {
            root: root.principal_directory_sequence(),
            directory: directory.sequence(),
        });
    }
    if root.principal_directory_sha256() != directory.directory_sha256() {
        findings.push(TrustRootFeasibilityFinding::PrincipalDirectoryDigestMismatch);
    }
    if !findings.is_empty() {
        return Err(findings);
    }

    let mut roles = Vec::with_capacity(root.role_policies().len());
    for policy in root.role_policies() {
        let eligible_role_bound_keys = directory
            .principals()
            .iter()
            .filter(|principal| match &policy.allowed_principal_ids {
                None => true,
                Some(allowed) => allowed.contains(&principal.principal_id),
            })
            .flat_map(|principal| principal.keys.iter())
            .filter(|key| key.roles.contains(&policy.role))
            .count();

        if eligible_role_bound_keys < policy.minimum_valid_signatures {
            findings.push(TrustRootFeasibilityFinding::RoleHasTooFewEligibleKeys {
                role: policy.role,
                actual: eligible_role_bound_keys,
                required: policy.minimum_valid_signatures,
            });
        }
        roles.push(RoleSignatureFeasibility {
            role: policy.role,
            eligible_role_bound_keys,
            minimum_valid_signatures: policy.minimum_valid_signatures,
        });
    }

    if !findings.is_empty() {
        return Err(findings);
    }

    let proof_sha256 = feasibility_digest(root, directory, &roles);
    Ok(TrustRootSignatureFeasibilityProof {
        root_sha256: root.root_sha256().clone(),
        principal_directory_sha256: directory.directory_sha256().clone(),
        roles,
        proof_sha256,
    })
}

fn feasibility_digest(
    root: &FrozenTrustRoot,
    directory: &TrustedPrincipalDirectory,
    roles: &[RoleSignatureFeasibility],
) -> Sha256Digest {
    let mut digest = FramedDigest::new(TRUST_ROOT_FEASIBILITY_DOMAIN);
    digest.text(root.root_sha256().as_str());
    digest.text(directory.directory_sha256().as_str());
    digest.text(&directory.sequence().to_string());
    for role in roles {
        digest.text("role");
        digest.text(role_tag(role.role));
        digest.text(&role.eligible_role_bound_keys.to_string());
        digest.text(&role.minimum_valid_signatures.to_string());
    }
    digest.text("signature-thresholds-structurally-feasible");
    digest.text("institutional-authority-not-established");
    digest.digest()
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
    use std::collections::BTreeSet;

    use super::*;
    use crate::{
        SignatureAlgorithm, TrustRolePolicy, TrustRootDraft, TrustedKeyBinding,
        TrustedPrincipal,
    };

    fn sha(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }

    fn directory() -> TrustedPrincipalDirectory {
        let principal = |id: &str, org: &str, region: &str, key_id: &str| TrustedPrincipal {
            principal_id: id.into(),
            organization_id: org.into(),
            region_id: region.into(),
            keys: vec![TrustedKeyBinding {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: key_id.into(),
                verification_key_sha256: sha(key_id),
                roles: BTreeSet::from([TrustRole::Root, TrustRole::Freshness]),
            }],
        };
        TrustedPrincipalDirectory::new(
            1,
            100,
            vec![
                principal("p-a", "org-a", "region-a", "key-a"),
                principal("p-b", "org-b", "region-b", "key-b"),
            ],
        )
        .unwrap()
    }

    fn policy(role: TrustRole, signatures: usize) -> TrustRolePolicy {
        TrustRolePolicy {
            role,
            minimum_valid_signatures: signatures,
            minimum_distinct_principals: 1,
            minimum_distinct_organizations: 1,
            minimum_distinct_regions: 1,
            required_algorithms: BTreeSet::new(),
            allowed_principal_ids: None,
        }
    }

    #[test]
    fn impossible_signature_threshold_is_detected_after_historical_freeze() {
        let directory = directory();
        let root = TrustRootDraft {
            version: 1,
            predecessor_root_sha256: None,
            issued_at_unix_s: 100,
            expires_at_unix_s: 1_000,
            role_policies: vec![
                policy(TrustRole::Root, 3),
                policy(TrustRole::Freshness, 1),
            ],
        }
        .freeze(&directory)
        .expect("historical root freeze does not count eligible keys");

        let findings = prove_trust_root_signature_feasibility(&root, &directory)
            .expect_err("three signatures cannot be produced by two role-bound keys");
        assert!(findings.iter().any(|finding| matches!(
            finding,
            TrustRootFeasibilityFinding::RoleHasTooFewEligibleKeys {
                role: TrustRole::Root,
                actual: 2,
                required: 3,
            }
        )));
    }

    #[test]
    fn feasible_signature_thresholds_produce_non_authorizing_proof() {
        let directory = directory();
        let root = TrustRootDraft {
            version: 1,
            predecessor_root_sha256: None,
            issued_at_unix_s: 100,
            expires_at_unix_s: 1_000,
            role_policies: vec![
                policy(TrustRole::Root, 2),
                policy(TrustRole::Freshness, 1),
            ],
        }
        .freeze(&directory)
        .unwrap();

        let proof = prove_trust_root_signature_feasibility(&root, &directory).unwrap();
        assert!(proof.signature_thresholds_structurally_feasible());
        assert!(!proof.institutional_authority_established());
    }
}
