// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Structural quorum-feasibility proof for frozen trust roots.
//!
//! Historical `FrozenTrustRoot` identities remain unchanged. This module proves
//! that every frozen role policy has a canonical structural witness capable, in
//! principle, of satisfying all monotone quorum minima at the same time:
//! signatures, distinct principals, distinct organizations, distinct regions,
//! and required signature algorithms.
//!
//! The witness is deliberately the complete, canonically sorted set of eligible
//! role-bound keys. All role-policy requirements are monotone under adding an
//! eligible signer, so this complete set is both the maximum structural quorum
//! and a deterministic feasibility witness. No exponential subset search is
//! necessary, and no weaker per-dimension existence argument is relied upon.
//!
//! This remains structural only. It does not establish that any key actually
//! signed, that any key is currently usable, that any trust snapshot is current,
//! or that the root has institutional authority.

use std::collections::BTreeSet;

use serde::Serialize;

use crate::attestation::digest_signature_algorithm;
use crate::{
    FramedDigest, FrozenTrustRoot, Sha256Digest, SignatureAlgorithm, TrustRole,
    TrustedPrincipalDirectory,
};

const TRUST_ROOT_FEASIBILITY_DOMAIN: &str =
    "symthaea.trust-root-quorum-feasibility.identity.v2";

/// One exact role-bound key admitted to the structural feasibility witness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RoleFeasibleSigner {
    algorithm: SignatureAlgorithm,
    key_id: String,
    verification_key_sha256: Sha256Digest,
    principal_id: String,
    organization_id: String,
    region_id: String,
}

impl RoleFeasibleSigner {
    pub fn algorithm(&self) -> &SignatureAlgorithm {
        &self.algorithm
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    pub fn verification_key_sha256(&self) -> &Sha256Digest {
        &self.verification_key_sha256
    }

    pub fn principal_id(&self) -> &str {
        &self.principal_id
    }

    pub fn organization_id(&self) -> &str {
        &self.organization_id
    }

    pub fn region_id(&self) -> &str {
        &self.region_id
    }
}

/// Complete structural quorum geometry for one frozen trust-root role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RoleSignatureFeasibility {
    role: TrustRole,
    eligible_role_bound_keys: usize,
    minimum_valid_signatures: usize,
    eligible_distinct_principals: usize,
    minimum_distinct_principals: usize,
    eligible_distinct_organizations: usize,
    minimum_distinct_organizations: usize,
    eligible_distinct_regions: usize,
    minimum_distinct_regions: usize,
    eligible_algorithms: Vec<SignatureAlgorithm>,
    required_algorithms: Vec<SignatureAlgorithm>,
    witness: Vec<RoleFeasibleSigner>,
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

    pub fn eligible_distinct_principals(&self) -> usize {
        self.eligible_distinct_principals
    }

    pub fn minimum_distinct_principals(&self) -> usize {
        self.minimum_distinct_principals
    }

    pub fn eligible_distinct_organizations(&self) -> usize {
        self.eligible_distinct_organizations
    }

    pub fn minimum_distinct_organizations(&self) -> usize {
        self.minimum_distinct_organizations
    }

    pub fn eligible_distinct_regions(&self) -> usize {
        self.eligible_distinct_regions
    }

    pub fn minimum_distinct_regions(&self) -> usize {
        self.minimum_distinct_regions
    }

    pub fn eligible_algorithms(&self) -> &[SignatureAlgorithm] {
        &self.eligible_algorithms
    }

    pub fn required_algorithms(&self) -> &[SignatureAlgorithm] {
        &self.required_algorithms
    }

    pub fn witness(&self) -> &[RoleFeasibleSigner] {
        &self.witness
    }

    /// Every policy dimension is a lower bound and the witness contains every
    /// eligible signer, so a successful value establishes simultaneous structural
    /// feasibility for the complete role quorum geometry.
    pub const fn quorum_geometry_structurally_feasible(&self) -> bool {
        true
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
    RoleHasTooFewEligiblePrincipals {
        role: TrustRole,
        actual: usize,
        required: usize,
    },
    RoleHasTooFewEligibleOrganizations {
        role: TrustRole,
        actual: usize,
        required: usize,
    },
    RoleHasTooFewEligibleRegions {
        role: TrustRole,
        actual: usize,
        required: usize,
    },
    RoleMissingRequiredEligibleAlgorithm {
        role: TrustRole,
        algorithm: SignatureAlgorithm,
    },
}

/// Non-deserializable proof that every frozen role has one exact canonical
/// witness satisfying its complete monotone quorum geometry.
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

    /// Compatibility name retained for existing callers. A v2 proof establishes
    /// more than signature-count feasibility: all quorum dimensions are proven
    /// simultaneously by the canonical complete witness.
    pub const fn signature_thresholds_structurally_feasible(&self) -> bool {
        true
    }

    pub const fn quorum_geometry_structurally_feasible(&self) -> bool {
        true
    }

    pub const fn signatures_observed(&self) -> bool {
        false
    }

    pub const fn current_key_usability_established(&self) -> bool {
        false
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
        let mut witness = Vec::new();
        for principal in directory.principals() {
            if policy
                .allowed_principal_ids
                .as_ref()
                .is_some_and(|allowed| !allowed.contains(&principal.principal_id))
            {
                continue;
            }
            for key in &principal.keys {
                if !key.roles.contains(&policy.role) {
                    continue;
                }
                witness.push(RoleFeasibleSigner {
                    algorithm: key.algorithm.clone(),
                    key_id: key.key_id.clone(),
                    verification_key_sha256: key.verification_key_sha256.clone(),
                    principal_id: principal.principal_id.clone(),
                    organization_id: principal.organization_id.clone(),
                    region_id: principal.region_id.clone(),
                });
            }
        }
        witness.sort_by(|left, right| {
            left.principal_id
                .cmp(&right.principal_id)
                .then(left.algorithm.cmp(&right.algorithm))
                .then(left.key_id.cmp(&right.key_id))
                .then(left.verification_key_sha256.cmp(&right.verification_key_sha256))
        });

        let principals: BTreeSet<_> = witness
            .iter()
            .map(|signer| signer.principal_id.as_str())
            .collect();
        let organizations: BTreeSet<_> = witness
            .iter()
            .map(|signer| signer.organization_id.as_str())
            .collect();
        let regions: BTreeSet<_> = witness
            .iter()
            .map(|signer| signer.region_id.as_str())
            .collect();
        let algorithms: BTreeSet<_> = witness
            .iter()
            .map(|signer| signer.algorithm.clone())
            .collect();

        if witness.len() < policy.minimum_valid_signatures {
            findings.push(TrustRootFeasibilityFinding::RoleHasTooFewEligibleKeys {
                role: policy.role,
                actual: witness.len(),
                required: policy.minimum_valid_signatures,
            });
        }
        if principals.len() < policy.minimum_distinct_principals {
            findings.push(TrustRootFeasibilityFinding::RoleHasTooFewEligiblePrincipals {
                role: policy.role,
                actual: principals.len(),
                required: policy.minimum_distinct_principals,
            });
        }
        if organizations.len() < policy.minimum_distinct_organizations {
            findings.push(TrustRootFeasibilityFinding::RoleHasTooFewEligibleOrganizations {
                role: policy.role,
                actual: organizations.len(),
                required: policy.minimum_distinct_organizations,
            });
        }
        if regions.len() < policy.minimum_distinct_regions {
            findings.push(TrustRootFeasibilityFinding::RoleHasTooFewEligibleRegions {
                role: policy.role,
                actual: regions.len(),
                required: policy.minimum_distinct_regions,
            });
        }
        for algorithm in &policy.required_algorithms {
            if !algorithms.contains(algorithm) {
                findings.push(
                    TrustRootFeasibilityFinding::RoleMissingRequiredEligibleAlgorithm {
                        role: policy.role,
                        algorithm: algorithm.clone(),
                    },
                );
            }
        }

        roles.push(RoleSignatureFeasibility {
            role: policy.role,
            eligible_role_bound_keys: witness.len(),
            minimum_valid_signatures: policy.minimum_valid_signatures,
            eligible_distinct_principals: principals.len(),
            minimum_distinct_principals: policy.minimum_distinct_principals,
            eligible_distinct_organizations: organizations.len(),
            minimum_distinct_organizations: policy.minimum_distinct_organizations,
            eligible_distinct_regions: regions.len(),
            minimum_distinct_regions: policy.minimum_distinct_regions,
            eligible_algorithms: algorithms.into_iter().collect(),
            required_algorithms: policy.required_algorithms.iter().cloned().collect(),
            witness,
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
        digest.text(&role.eligible_distinct_principals.to_string());
        digest.text(&role.minimum_distinct_principals.to_string());
        digest.text(&role.eligible_distinct_organizations.to_string());
        digest.text(&role.minimum_distinct_organizations.to_string());
        digest.text(&role.eligible_distinct_regions.to_string());
        digest.text(&role.minimum_distinct_regions.to_string());
        for algorithm in &role.eligible_algorithms {
            digest.text("eligible-algorithm");
            digest_signature_algorithm(&mut digest, algorithm);
        }
        for algorithm in &role.required_algorithms {
            digest.text("required-algorithm");
            digest_signature_algorithm(&mut digest, algorithm);
        }
        for signer in &role.witness {
            digest.text("eligible-signer");
            digest_signature_algorithm(&mut digest, &signer.algorithm);
            digest.text(&signer.key_id);
            digest.text(signer.verification_key_sha256.as_str());
            digest.text(&signer.principal_id);
            digest.text(&signer.organization_id);
            digest.text(&signer.region_id);
        }
        digest.text("complete-monotone-quorum-witness");
    }
    digest.text("quorum-geometry-structurally-feasible");
    digest.text("signatures-not-observed");
    digest.text("current-key-usability-not-established");
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
    use crate::{TrustRolePolicy, TrustRootDraft, TrustedKeyBinding, TrustedPrincipal};

    fn sha(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }

    fn key(
        algorithm: SignatureAlgorithm,
        key_id: &str,
        roles: BTreeSet<TrustRole>,
    ) -> TrustedKeyBinding {
        TrustedKeyBinding {
            algorithm,
            key_id: key_id.into(),
            verification_key_sha256: sha(key_id),
            roles,
        }
    }

    fn directory() -> TrustedPrincipalDirectory {
        let roles = BTreeSet::from([TrustRole::Root, TrustRole::Freshness]);
        TrustedPrincipalDirectory::new(
            1,
            100,
            vec![
                TrustedPrincipal {
                    principal_id: "p-a".into(),
                    organization_id: "org-a".into(),
                    region_id: "region-a".into(),
                    keys: vec![key(SignatureAlgorithm::Ed25519, "key-a", roles.clone())],
                },
                TrustedPrincipal {
                    principal_id: "p-b".into(),
                    organization_id: "org-b".into(),
                    region_id: "region-b".into(),
                    keys: vec![key(SignatureAlgorithm::MlDsa65, "key-b", roles)],
                },
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
    fn complete_witness_binds_all_monotone_quorum_dimensions() {
        let directory = directory();
        let policy = |role| TrustRolePolicy {
            role,
            minimum_valid_signatures: 2,
            minimum_distinct_principals: 2,
            minimum_distinct_organizations: 2,
            minimum_distinct_regions: 2,
            required_algorithms: BTreeSet::from([
                SignatureAlgorithm::Ed25519,
                SignatureAlgorithm::MlDsa65,
            ]),
            allowed_principal_ids: None,
        };
        let root = TrustRootDraft {
            version: 1,
            predecessor_root_sha256: None,
            issued_at_unix_s: 100,
            expires_at_unix_s: 1_000,
            role_policies: vec![policy(TrustRole::Root), policy(TrustRole::Freshness)],
        }
        .freeze(&directory)
        .unwrap();

        let proof = prove_trust_root_signature_feasibility(&root, &directory).unwrap();
        let root_role = proof
            .roles()
            .iter()
            .find(|role| role.role() == TrustRole::Root)
            .unwrap();

        assert_eq!(root_role.witness().len(), 2);
        assert_eq!(root_role.eligible_distinct_principals(), 2);
        assert_eq!(root_role.eligible_distinct_organizations(), 2);
        assert_eq!(root_role.eligible_distinct_regions(), 2);
        assert_eq!(root_role.eligible_algorithms().len(), 2);
        assert!(root_role.quorum_geometry_structurally_feasible());
        assert!(proof.signature_thresholds_structurally_feasible());
        assert!(proof.quorum_geometry_structurally_feasible());
        assert!(!proof.signatures_observed());
        assert!(!proof.current_key_usability_established());
        assert!(!proof.institutional_authority_established());
    }

    #[test]
    fn feasibility_identity_binds_exact_verification_key_material() {
        let left_directory = directory();
        let left_root = TrustRootDraft {
            version: 1,
            predecessor_root_sha256: None,
            issued_at_unix_s: 100,
            expires_at_unix_s: 1_000,
            role_policies: vec![
                policy(TrustRole::Root, 1),
                policy(TrustRole::Freshness, 1),
            ],
        }
        .freeze(&left_directory)
        .unwrap();
        let left = prove_trust_root_signature_feasibility(&left_root, &left_directory).unwrap();

        let roles = BTreeSet::from([TrustRole::Root, TrustRole::Freshness]);
        let right_directory = TrustedPrincipalDirectory::new(
            1,
            100,
            vec![
                TrustedPrincipal {
                    principal_id: "p-a".into(),
                    organization_id: "org-a".into(),
                    region_id: "region-a".into(),
                    keys: vec![TrustedKeyBinding {
                        algorithm: SignatureAlgorithm::Ed25519,
                        key_id: "key-a".into(),
                        verification_key_sha256: sha("different-key-material"),
                        roles: roles.clone(),
                    }],
                },
                TrustedPrincipal {
                    principal_id: "p-b".into(),
                    organization_id: "org-b".into(),
                    region_id: "region-b".into(),
                    keys: vec![key(SignatureAlgorithm::MlDsa65, "key-b", roles)],
                },
            ],
        )
        .unwrap();
        let right_root = TrustRootDraft {
            version: 1,
            predecessor_root_sha256: None,
            issued_at_unix_s: 100,
            expires_at_unix_s: 1_000,
            role_policies: vec![
                policy(TrustRole::Root, 1),
                policy(TrustRole::Freshness, 1),
            ],
        }
        .freeze(&right_directory)
        .unwrap();
        let right = prove_trust_root_signature_feasibility(&right_root, &right_directory).unwrap();

        assert_ne!(left.proof_sha256(), right.proof_sha256());
    }
}
