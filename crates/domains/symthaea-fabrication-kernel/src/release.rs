// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Role-aware release quorum for fabrication manifests.
//!
//! Cryptographic validity and key lifecycle eligibility answer whether a signer
//! is authentic and currently trusted. This module answers the separate policy
//! question: whether the set of trusted signers represents the independent
//! authorities required to release one fabrication manifest.

use crate::attestation::{SignatureAlgorithm, VerifiedAttestation};
use crate::crypto_digest::{Sha256, Sha256Digest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const RELEASE_POLICY_SCHEMA: &str = "symthaea.fabrication.release-policy.v1";
pub const MAX_RELEASE_BINDINGS: usize = 4096;
pub const MAX_RELEASE_REQUIREMENTS: usize = 64;
pub const MAX_RELEASE_ROLE_NAME_BYTES: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SignerRole {
    DesignAuthority,
    ManufacturingAuthority,
    SafetyAuthority,
    OperationsAuthority,
    Other(String),
}

impl SignerRole {
    fn validate(&self) -> Result<(), ReleasePolicyError> {
        if let Self::Other(name) = self {
            if name.trim().is_empty() || name != name.trim() {
                return Err(ReleasePolicyError::InvalidRoleName(name.clone()));
            }
            if name.len() > MAX_RELEASE_ROLE_NAME_BYTES {
                return Err(ReleasePolicyError::RoleNameTooLong {
                    actual: name.len(),
                    maximum: MAX_RELEASE_ROLE_NAME_BYTES,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseSignerBinding {
    pub algorithm: SignatureAlgorithm,
    pub key_id: String,
    pub roles: BTreeSet<SignerRole>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseQuorumRequirement {
    pub role: SignerRole,
    pub minimum_distinct_signers: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleasePolicy {
    pub schema_version: String,
    pub minimum_distinct_signers: usize,
    pub maximum_considered_signers: usize,
    pub require_algorithm_diversity: bool,
    pub requirements: Vec<ReleaseQuorumRequirement>,
    pub bindings: Vec<ReleaseSignerBinding>,
}

impl Default for ReleasePolicy {
    fn default() -> Self {
        Self {
            schema_version: RELEASE_POLICY_SCHEMA.into(),
            minimum_distinct_signers: 1,
            maximum_considered_signers: 16,
            require_algorithm_diversity: false,
            requirements: Vec::new(),
            bindings: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReleasePolicyError {
    UnsupportedSchema,
    InvalidSignerBounds,
    TooManyBindings {
        actual: usize,
        maximum: usize,
    },
    TooManyRequirements {
        actual: usize,
        maximum: usize,
    },
    InvalidAlgorithm(String),
    EmptyKeyId,
    NonCanonicalKeyId(String),
    EmptyRoles(String),
    DuplicateBinding {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
    InvalidRoleName(String),
    RoleNameTooLong {
        actual: usize,
        maximum: usize,
    },
    ZeroRoleQuorum(SignerRole),
    DuplicateRoleRequirement(SignerRole),
    Encoding(String),
}

impl ReleasePolicy {
    pub fn new(
        minimum_distinct_signers: usize,
        maximum_considered_signers: usize,
        require_algorithm_diversity: bool,
        requirements: Vec<ReleaseQuorumRequirement>,
        bindings: Vec<ReleaseSignerBinding>,
    ) -> Result<Self, ReleasePolicyError> {
        let mut policy = Self {
            schema_version: RELEASE_POLICY_SCHEMA.into(),
            minimum_distinct_signers,
            maximum_considered_signers,
            require_algorithm_diversity,
            requirements,
            bindings,
        };
        policy.canonicalize();
        policy.validate()?;
        Ok(policy)
    }

    pub fn canonicalize(&mut self) {
        self.bindings.sort_by(|left, right| {
            (&left.algorithm, left.key_id.as_str()).cmp(&(&right.algorithm, right.key_id.as_str()))
        });
        self.requirements
            .sort_by(|left, right| left.role.cmp(&right.role));
    }

    pub fn validate(&self) -> Result<(), ReleasePolicyError> {
        if self.schema_version != RELEASE_POLICY_SCHEMA {
            return Err(ReleasePolicyError::UnsupportedSchema);
        }
        if self.minimum_distinct_signers == 0
            || self.maximum_considered_signers == 0
            || self.minimum_distinct_signers > self.maximum_considered_signers
        {
            return Err(ReleasePolicyError::InvalidSignerBounds);
        }
        if self.bindings.len() > MAX_RELEASE_BINDINGS {
            return Err(ReleasePolicyError::TooManyBindings {
                actual: self.bindings.len(),
                maximum: MAX_RELEASE_BINDINGS,
            });
        }
        if self.requirements.len() > MAX_RELEASE_REQUIREMENTS {
            return Err(ReleasePolicyError::TooManyRequirements {
                actual: self.requirements.len(),
                maximum: MAX_RELEASE_REQUIREMENTS,
            });
        }

        let mut identities = BTreeSet::new();
        for binding in &self.bindings {
            if !binding.algorithm.is_canonical() {
                return Err(ReleasePolicyError::InvalidAlgorithm(format!(
                    "{:?}",
                    binding.algorithm
                )));
            }
            let key_id = binding.key_id.trim();
            if key_id.is_empty() {
                return Err(ReleasePolicyError::EmptyKeyId);
            }
            if key_id != binding.key_id {
                return Err(ReleasePolicyError::NonCanonicalKeyId(
                    binding.key_id.clone(),
                ));
            }
            if binding.roles.is_empty() {
                return Err(ReleasePolicyError::EmptyRoles(binding.key_id.clone()));
            }
            for role in &binding.roles {
                role.validate()?;
            }
            if !identities.insert((binding.algorithm.clone(), binding.key_id.clone())) {
                return Err(ReleasePolicyError::DuplicateBinding {
                    algorithm: binding.algorithm.clone(),
                    key_id: binding.key_id.clone(),
                });
            }
        }

        let mut required_roles = BTreeSet::new();
        for requirement in &self.requirements {
            requirement.role.validate()?;
            if requirement.minimum_distinct_signers == 0 {
                return Err(ReleasePolicyError::ZeroRoleQuorum(requirement.role.clone()));
            }
            if !required_roles.insert(requirement.role.clone()) {
                return Err(ReleasePolicyError::DuplicateRoleRequirement(
                    requirement.role.clone(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReleaseViolation {
    InvalidPolicy(ReleasePolicyError),
    TooManySigners {
        actual: usize,
        maximum: usize,
    },
    InsufficientDistinctSigners {
        actual: usize,
        required: usize,
    },
    MissingAlgorithmDiversity,
    MissingRoleQuorum {
        role: SignerRole,
        actual: usize,
        required: usize,
    },
    LifecycleEvidenceMissing,
    PolicyDigestUnavailable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseEvaluationReport {
    pub recognized_signers: Vec<(SignatureAlgorithm, String)>,
    pub role_counts: BTreeMap<SignerRole, usize>,
    pub violations: Vec<ReleaseViolation>,
}

impl ReleaseEvaluationReport {
    pub fn authorized(&self) -> bool {
        self.violations.is_empty()
    }
}

/// Reusable release approval for one exact manifest and one exact release policy.
#[derive(Debug, Clone)]
pub struct ReleaseAuthorization {
    manifest_digest: Sha256Digest,
    policy_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    evaluation_time_unix_s: u64,
    recognized_signers: Vec<(SignatureAlgorithm, String)>,
    role_counts: BTreeMap<SignerRole, usize>,
}

pub trait ReleaseAuthority {
    fn manifest_digest(&self) -> Sha256Digest;
    fn policy_digest(&self) -> Sha256Digest;
    fn trust_snapshot_digest(&self) -> Sha256Digest;
    fn evaluation_time_unix_s(&self) -> u64;
    fn delegation_digest(&self) -> Option<Sha256Digest> {
        None
    }
}

impl ReleaseAuthorization {
    pub fn manifest_digest(&self) -> Sha256Digest {
        self.manifest_digest
    }

    pub fn policy_digest(&self) -> Sha256Digest {
        self.policy_digest
    }

    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }

    pub fn evaluation_time_unix_s(&self) -> u64 {
        self.evaluation_time_unix_s
    }

    pub fn recognized_signers(&self) -> &[(SignatureAlgorithm, String)] {
        &self.recognized_signers
    }

    pub fn role_counts(&self) -> &BTreeMap<SignerRole, usize> {
        &self.role_counts
    }
}

impl ReleaseAuthority for ReleaseAuthorization {
    fn manifest_digest(&self) -> Sha256Digest {
        self.manifest_digest
    }
    fn policy_digest(&self) -> Sha256Digest {
        self.policy_digest
    }
    fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }
    fn evaluation_time_unix_s(&self) -> u64 {
        self.evaluation_time_unix_s
    }
}

pub fn canonical_release_policy_bytes(
    policy: &ReleasePolicy,
) -> Result<Vec<u8>, ReleasePolicyError> {
    let mut canonical = policy.clone();
    canonical.canonicalize();
    canonical.validate()?;
    serde_json::to_vec(&canonical).map_err(|error| ReleasePolicyError::Encoding(error.to_string()))
}

pub fn digest_release_policy(policy: &ReleasePolicy) -> Result<Sha256Digest, ReleasePolicyError> {
    let bytes = canonical_release_policy_bytes(policy)?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.release-policy-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn authorize_release(
    attestation: &VerifiedAttestation,
    policy: &ReleasePolicy,
) -> Result<ReleaseAuthorization, ReleaseEvaluationReport> {
    authorize_release_with_role_overrides(attestation, policy, &BTreeMap::new())
}

pub(crate) fn authorize_release_with_role_overrides(
    attestation: &VerifiedAttestation,
    policy: &ReleasePolicy,
    role_overrides: &BTreeMap<(SignatureAlgorithm, String), BTreeSet<SignerRole>>,
) -> Result<ReleaseAuthorization, ReleaseEvaluationReport> {
    let mut report = ReleaseEvaluationReport {
        recognized_signers: Vec::new(),
        role_counts: BTreeMap::new(),
        violations: Vec::new(),
    };

    if let Err(error) = policy.validate() {
        report
            .violations
            .push(ReleaseViolation::InvalidPolicy(error));
        return Err(report);
    }
    let Some(trust_snapshot_digest) = attestation.trust_snapshot_digest() else {
        report
            .violations
            .push(ReleaseViolation::LifecycleEvidenceMissing);
        return Err(report);
    };
    let Some(evaluation_time_unix_s) = attestation.evaluation_time_unix_s() else {
        report
            .violations
            .push(ReleaseViolation::LifecycleEvidenceMissing);
        return Err(report);
    };

    let valid_signers: BTreeSet<_> = attestation.valid_signers().iter().cloned().collect();
    let bindings: BTreeMap<_, _> = policy
        .bindings
        .iter()
        .map(|binding| {
            (
                (binding.algorithm.clone(), binding.key_id.clone()),
                &binding.roles,
            )
        })
        .collect();

    for signer in valid_signers {
        let mut roles = BTreeSet::new();
        if let Some(bound_roles) = bindings.get(&signer) {
            roles.extend((*bound_roles).iter().cloned());
        }
        if let Some(delegated_roles) = role_overrides.get(&signer) {
            roles.extend(delegated_roles.iter().cloned());
        }
        if !roles.is_empty() {
            report.recognized_signers.push(signer.clone());
            for role in roles {
                *report.role_counts.entry(role).or_insert(0) += 1;
            }
        }
    }
    report.recognized_signers.sort();

    if report.recognized_signers.len() > policy.maximum_considered_signers {
        report.violations.push(ReleaseViolation::TooManySigners {
            actual: report.recognized_signers.len(),
            maximum: policy.maximum_considered_signers,
        });
    }
    if report.recognized_signers.len() < policy.minimum_distinct_signers {
        report
            .violations
            .push(ReleaseViolation::InsufficientDistinctSigners {
                actual: report.recognized_signers.len(),
                required: policy.minimum_distinct_signers,
            });
    }
    if policy.require_algorithm_diversity {
        let algorithm_count = report
            .recognized_signers
            .iter()
            .map(|(algorithm, _)| algorithm)
            .collect::<BTreeSet<_>>()
            .len();
        if algorithm_count < 2 {
            report
                .violations
                .push(ReleaseViolation::MissingAlgorithmDiversity);
        }
    }
    for requirement in &policy.requirements {
        let actual = report
            .role_counts
            .get(&requirement.role)
            .copied()
            .unwrap_or(0);
        if actual < requirement.minimum_distinct_signers {
            report.violations.push(ReleaseViolation::MissingRoleQuorum {
                role: requirement.role.clone(),
                actual,
                required: requirement.minimum_distinct_signers,
            });
        }
    }

    if !report.authorized() {
        return Err(report);
    }
    let policy_digest = match digest_release_policy(policy) {
        Ok(digest) => digest,
        Err(_) => {
            report
                .violations
                .push(ReleaseViolation::PolicyDigestUnavailable);
            return Err(report);
        }
    };
    Ok(ReleaseAuthorization {
        manifest_digest: attestation.manifest_digest(),
        policy_digest,
        trust_snapshot_digest,
        evaluation_time_unix_s,
        recognized_signers: report.recognized_signers,
        role_counts: report.role_counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attestation::{
        AttestationPolicy, AttestationTrustContext, ManifestSignatureVerifier, ManifestSigner,
        attest_fabrication_manifest, verify_attestation_authority_with_trust,
    };
    use crate::crypto_digest::sha256;
    use crate::provenance::{FabricationManifest, StableFingerprint};
    use crate::trust::{KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot};

    struct Provider(&'static str, SignatureAlgorithm);

    impl ManifestSigner for Provider {
        fn algorithm(&self) -> SignatureAlgorithm {
            self.1.clone()
        }
        fn key_id(&self) -> &str {
            self.0
        }
        fn sign(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            let mut bytes = self.0.as_bytes().to_vec();
            bytes.extend_from_slice(message);
            Ok(sha256(&bytes).0.to_vec())
        }
    }

    struct Verifier;
    impl ManifestSignatureVerifier for Verifier {
        fn verify(
            &self,
            algorithm: &SignatureAlgorithm,
            key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            let mut bytes = key_id.as_bytes().to_vec();
            bytes.extend_from_slice(message);
            Ok(matches!(
                algorithm,
                SignatureAlgorithm::Ed25519 | SignatureAlgorithm::MlDsa65
            ) && signature == sha256(&bytes).0.as_slice())
        }
    }

    fn manifest() -> FabricationManifest {
        let fingerprint = StableFingerprint([1, 2, 3, 4]);
        FabricationManifest {
            schema_version: "symthaea.fabrication.manifest.v1".into(),
            geometry: fingerprint,
            process_policy: fingerprint,
            process_evidence: fingerprint,
            minimum_feature_policy: fingerprint,
            minimum_feature_evidence: fingerprint,
            slice_config: fingerprint,
            slice_layers: fingerprint,
            toolpath_config: fingerprint,
            machine_profile: fingerprint,
            gcode_program: fingerprint,
            pipeline: fingerprint,
            layer_count: 1,
            command_count: 1,
            total_extrusion_mm: 1.0,
        }
    }

    fn verified() -> VerifiedAttestation {
        let design = Provider("design", SignatureAlgorithm::Ed25519);
        let safety = Provider("safety", SignatureAlgorithm::MlDsa65);
        let attested = attest_fabrication_manifest(manifest(), &[&design, &safety]).unwrap();
        let keys = [
            (SignatureAlgorithm::Ed25519, "design"),
            (SignatureAlgorithm::MlDsa65, "safety"),
        ]
        .into_iter()
        .map(|(algorithm, key_id)| KeyTrustRecord {
            algorithm,
            key_id: key_id.into(),
            not_before_unix_s: 1,
            not_after_unix_s: None,
            status: KeyLifecycleStatus::Active,
            usages: BTreeSet::from([KeyUsage::FabricationManifest]),
        })
        .collect();
        let snapshot = TrustSnapshot::new(1, 1, 1_000, keys).unwrap();
        verify_attestation_authority_with_trust(
            attested,
            &AttestationPolicy {
                minimum_valid_signatures: 2,
                ..Default::default()
            },
            &Verifier,
            AttestationTrustContext {
                evaluation_time_unix_s: 100,
                snapshot: &snapshot,
            },
        )
        .unwrap()
    }

    fn policy() -> ReleasePolicy {
        ReleasePolicy::new(
            2,
            4,
            true,
            vec![
                ReleaseQuorumRequirement {
                    role: SignerRole::DesignAuthority,
                    minimum_distinct_signers: 1,
                },
                ReleaseQuorumRequirement {
                    role: SignerRole::SafetyAuthority,
                    minimum_distinct_signers: 1,
                },
            ],
            vec![
                ReleaseSignerBinding {
                    algorithm: SignatureAlgorithm::Ed25519,
                    key_id: "design".into(),
                    roles: BTreeSet::from([SignerRole::DesignAuthority]),
                },
                ReleaseSignerBinding {
                    algorithm: SignatureAlgorithm::MlDsa65,
                    key_id: "safety".into(),
                    roles: BTreeSet::from([SignerRole::SafetyAuthority]),
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn independent_roles_and_algorithms_grant_release_authority() {
        let authorization = authorize_release(&verified(), &policy()).unwrap();
        assert_eq!(authorization.recognized_signers().len(), 2);
        assert_eq!(authorization.role_counts()[&SignerRole::SafetyAuthority], 1);
    }

    #[test]
    fn missing_role_is_fail_closed() {
        let mut policy = policy();
        policy.bindings.retain(|binding| binding.key_id != "safety");
        let report = authorize_release(&verified(), &policy).unwrap_err();
        assert!(report.violations.iter().any(|violation| matches!(
            violation,
            ReleaseViolation::MissingRoleQuorum {
                role: SignerRole::SafetyAuthority,
                ..
            }
        )));
    }

    #[test]
    fn canonical_policy_digest_is_order_independent() {
        let left = policy();
        let mut right = policy();
        right.bindings.reverse();
        right.requirements.reverse();
        assert_eq!(
            digest_release_policy(&left).unwrap(),
            digest_release_policy(&right).unwrap()
        );
    }
}
