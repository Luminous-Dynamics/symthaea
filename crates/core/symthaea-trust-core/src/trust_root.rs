// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Frozen delegated trust-root policy and non-authorizing rotation contracts.
//!
//! A content-addressed root is policy, not authority. TRUST-007A deliberately
//! stops before bootstrap/rotation authorization: possession of a well-formed
//! root or transition contract cannot establish trust. A later capability must
//! authenticate bootstrap anchoring and the dual old-root/new-root thresholds.

use std::collections::BTreeSet;

use serde::Serialize;

use crate::{
    FramedDigest, Sha256Digest, SignatureAlgorithm, TrustRole,
    TrustedPrincipalDirectory,
};

pub const TRUST_ROOT_SCHEMA: &str = "symthaea.trust-root.v1";
pub const TRUST_ROOT_TRANSITION_SCHEMA: &str = "symthaea.trust-root-transition.v1";
const TRUST_ROOT_DOMAIN: &str = "symthaea.trust-root.identity.v1";
const TRUST_ROOT_TRANSITION_DOMAIN: &str = "symthaea.trust-root-transition.identity.v1";
pub const MAX_ROLE_POLICIES: usize = 64;
pub const MAX_ALLOWED_PRINCIPALS_PER_ROLE: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrustRolePolicy {
    pub role: TrustRole,
    pub minimum_valid_signatures: usize,
    pub minimum_distinct_principals: usize,
    pub minimum_distinct_organizations: usize,
    pub minimum_distinct_regions: usize,
    pub required_algorithms: BTreeSet<SignatureAlgorithm>,
    /// None means any principal in the authorized directory whose exact key
    /// binding includes this role may participate.
    pub allowed_principal_ids: Option<BTreeSet<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustRolePolicyIssue {
    ZeroSignatureThreshold,
    ZeroPrincipalThreshold,
    PrincipalThresholdExceedsSignatureThreshold,
    OrganizationThresholdExceedsPrincipalThreshold,
    RegionThresholdExceedsPrincipalThreshold,
    InvalidAlgorithm,
    TooManyAllowedPrincipals,
    InvalidAllowedPrincipalId,
}

impl TrustRolePolicy {
    pub fn validate(&self) -> Vec<TrustRolePolicyIssue> {
        let mut issues = Vec::new();
        if self.minimum_valid_signatures == 0 {
            issues.push(TrustRolePolicyIssue::ZeroSignatureThreshold);
        }
        if self.minimum_distinct_principals == 0 {
            issues.push(TrustRolePolicyIssue::ZeroPrincipalThreshold);
        }
        if self.minimum_distinct_principals > self.minimum_valid_signatures {
            issues.push(TrustRolePolicyIssue::PrincipalThresholdExceedsSignatureThreshold);
        }
        if self.minimum_distinct_organizations > self.minimum_distinct_principals {
            issues.push(TrustRolePolicyIssue::OrganizationThresholdExceedsPrincipalThreshold);
        }
        if self.minimum_distinct_regions > self.minimum_distinct_principals {
            issues.push(TrustRolePolicyIssue::RegionThresholdExceedsPrincipalThreshold);
        }
        if self.required_algorithms.iter().any(|algorithm| !algorithm.is_canonical()) {
            issues.push(TrustRolePolicyIssue::InvalidAlgorithm);
        }
        if let Some(ids) = &self.allowed_principal_ids {
            if ids.len() > MAX_ALLOWED_PRINCIPALS_PER_ROLE {
                issues.push(TrustRolePolicyIssue::TooManyAllowedPrincipals);
            }
            if ids.iter().any(|id| !canonical_identifier(id)) {
                issues.push(TrustRolePolicyIssue::InvalidAllowedPrincipalId);
            }
        }
        issues
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrustRootDraft {
    pub version: u64,
    pub predecessor_root_sha256: Option<Sha256Digest>,
    pub issued_at_unix_s: u64,
    pub expires_at_unix_s: u64,
    pub role_policies: Vec<TrustRolePolicy>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustRootIssue {
    VersionZero,
    InvalidWindow,
    GenesisHasPredecessor,
    NonGenesisMissingPredecessor,
    DirectoryIssuedAfterRoot,
    EmptyRolePolicies,
    TooManyRolePolicies { actual: usize, maximum: usize },
    DuplicateRole { role: TrustRole },
    MissingRootRole,
    MissingFreshnessRole,
    InvalidRolePolicy {
        role: TrustRole,
        issues: Vec<TrustRolePolicyIssue>,
    },
    AllowedPrincipalUnknown {
        role: TrustRole,
        principal_id: String,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrozenTrustRoot {
    schema_version: String,
    version: u64,
    predecessor_root_sha256: Option<Sha256Digest>,
    issued_at_unix_s: u64,
    expires_at_unix_s: u64,
    principal_directory_sequence: u64,
    principal_directory_sha256: Sha256Digest,
    role_policies: Vec<TrustRolePolicy>,
    root_sha256: Sha256Digest,
}

impl TrustRootDraft {
    pub fn freeze(
        mut self,
        directory: &TrustedPrincipalDirectory,
    ) -> Result<FrozenTrustRoot, Vec<TrustRootIssue>> {
        self.role_policies.sort_by_key(|policy| policy.role);
        let mut issues = Vec::new();
        if self.version == 0 {
            issues.push(TrustRootIssue::VersionZero);
        }
        if self.issued_at_unix_s >= self.expires_at_unix_s {
            issues.push(TrustRootIssue::InvalidWindow);
        }
        if self.version == 1 && self.predecessor_root_sha256.is_some() {
            issues.push(TrustRootIssue::GenesisHasPredecessor);
        }
        if self.version > 1 && self.predecessor_root_sha256.is_none() {
            issues.push(TrustRootIssue::NonGenesisMissingPredecessor);
        }
        if directory.issued_at_unix_s() > self.issued_at_unix_s {
            issues.push(TrustRootIssue::DirectoryIssuedAfterRoot);
        }
        if self.role_policies.is_empty() {
            issues.push(TrustRootIssue::EmptyRolePolicies);
        }
        if self.role_policies.len() > MAX_ROLE_POLICIES {
            issues.push(TrustRootIssue::TooManyRolePolicies {
                actual: self.role_policies.len(),
                maximum: MAX_ROLE_POLICIES,
            });
        }

        let mut roles = BTreeSet::new();
        for policy in &self.role_policies {
            if !roles.insert(policy.role) {
                issues.push(TrustRootIssue::DuplicateRole { role: policy.role });
            }
            let policy_issues = policy.validate();
            if !policy_issues.is_empty() {
                issues.push(TrustRootIssue::InvalidRolePolicy {
                    role: policy.role,
                    issues: policy_issues,
                });
            }

            let eligible = eligible_principals(directory, policy, &mut issues);
            if eligible.len() < policy.minimum_distinct_principals {
                issues.push(TrustRootIssue::RoleHasTooFewEligiblePrincipals {
                    role: policy.role,
                    actual: eligible.len(),
                    required: policy.minimum_distinct_principals,
                });
            }

            let mut organizations = BTreeSet::new();
            let mut regions = BTreeSet::new();
            let mut algorithms = BTreeSet::new();
            for principal in directory.principals() {
                if !eligible.contains(&principal.principal_id) {
                    continue;
                }
                organizations.insert(principal.organization_id.clone());
                regions.insert(principal.region_id.clone());
                for key in &principal.keys {
                    if key.roles.contains(&policy.role) {
                        algorithms.insert(key.algorithm.clone());
                    }
                }
            }
            if organizations.len() < policy.minimum_distinct_organizations {
                issues.push(TrustRootIssue::RoleHasTooFewEligibleOrganizations {
                    role: policy.role,
                    actual: organizations.len(),
                    required: policy.minimum_distinct_organizations,
                });
            }
            if regions.len() < policy.minimum_distinct_regions {
                issues.push(TrustRootIssue::RoleHasTooFewEligibleRegions {
                    role: policy.role,
                    actual: regions.len(),
                    required: policy.minimum_distinct_regions,
                });
            }
            for algorithm in &policy.required_algorithms {
                if !algorithms.contains(algorithm) {
                    issues.push(TrustRootIssue::RoleMissingRequiredEligibleAlgorithm {
                        role: policy.role,
                        algorithm: algorithm.clone(),
                    });
                }
            }
        }
        if !roles.contains(&TrustRole::Root) {
            issues.push(TrustRootIssue::MissingRootRole);
        }
        if !roles.contains(&TrustRole::Freshness) {
            issues.push(TrustRootIssue::MissingFreshnessRole);
        }
        if !issues.is_empty() {
            return Err(issues);
        }

        let root_sha256 = root_digest(
            self.version,
            self.predecessor_root_sha256.as_ref(),
            self.issued_at_unix_s,
            self.expires_at_unix_s,
            directory.sequence(),
            directory.directory_sha256(),
            &self.role_policies,
        );
        Ok(FrozenTrustRoot {
            schema_version: TRUST_ROOT_SCHEMA.into(),
            version: self.version,
            predecessor_root_sha256: self.predecessor_root_sha256,
            issued_at_unix_s: self.issued_at_unix_s,
            expires_at_unix_s: self.expires_at_unix_s,
            principal_directory_sequence: directory.sequence(),
            principal_directory_sha256: directory.directory_sha256().clone(),
            role_policies: self.role_policies,
            root_sha256,
        })
    }
}

impl FrozenTrustRoot {
    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn predecessor_root_sha256(&self) -> Option<&Sha256Digest> {
        self.predecessor_root_sha256.as_ref()
    }

    pub fn issued_at_unix_s(&self) -> u64 {
        self.issued_at_unix_s
    }

    pub fn expires_at_unix_s(&self) -> u64 {
        self.expires_at_unix_s
    }

    pub fn principal_directory_sequence(&self) -> u64 {
        self.principal_directory_sequence
    }

    pub fn principal_directory_sha256(&self) -> &Sha256Digest {
        &self.principal_directory_sha256
    }

    pub fn role_policies(&self) -> &[TrustRolePolicy] {
        &self.role_policies
    }

    pub fn role_policy(&self, role: TrustRole) -> Option<&TrustRolePolicy> {
        self.role_policies.iter().find(|policy| policy.role == role)
    }

    pub fn root_sha256(&self) -> &Sha256Digest {
        &self.root_sha256
    }

    /// Genesis still requires an out-of-band bootstrap anchor. Content-addressed
    /// root metadata is not self-authenticating.
    pub const fn bootstrap_authority_established(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustRootTransitionIssue {
    PreviousRootMismatch,
    VersionNotSequential { previous: u64, next: u64 },
    IssuedAtRegressed,
    TransitionOutsidePreviousRootWindow,
    TransitionOutsideNextRootWindow,
    SameAuthorizationReceiptUsedForBothSides,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrustRootTransitionContract {
    schema_version: String,
    previous_root_sha256: Sha256Digest,
    next_root_sha256: Sha256Digest,
    transition_at_unix_s: u64,
    old_root_authorization_receipt_sha256: Sha256Digest,
    new_root_authorization_receipt_sha256: Sha256Digest,
    transition_sha256: Sha256Digest,
}

impl TrustRootTransitionContract {
    pub fn new(
        previous: &FrozenTrustRoot,
        next: &FrozenTrustRoot,
        transition_at_unix_s: u64,
        old_root_authorization_receipt_sha256: Sha256Digest,
        new_root_authorization_receipt_sha256: Sha256Digest,
    ) -> Result<Self, Vec<TrustRootTransitionIssue>> {
        let mut issues = Vec::new();
        if next.predecessor_root_sha256() != Some(previous.root_sha256()) {
            issues.push(TrustRootTransitionIssue::PreviousRootMismatch);
        }
        if next.version() != previous.version().saturating_add(1) {
            issues.push(TrustRootTransitionIssue::VersionNotSequential {
                previous: previous.version(),
                next: next.version(),
            });
        }
        if next.issued_at_unix_s() < previous.issued_at_unix_s() {
            issues.push(TrustRootTransitionIssue::IssuedAtRegressed);
        }
        if transition_at_unix_s < previous.issued_at_unix_s()
            || transition_at_unix_s >= previous.expires_at_unix_s()
        {
            issues.push(TrustRootTransitionIssue::TransitionOutsidePreviousRootWindow);
        }
        if transition_at_unix_s < next.issued_at_unix_s()
            || transition_at_unix_s >= next.expires_at_unix_s()
        {
            issues.push(TrustRootTransitionIssue::TransitionOutsideNextRootWindow);
        }
        if old_root_authorization_receipt_sha256 == new_root_authorization_receipt_sha256 {
            issues.push(TrustRootTransitionIssue::SameAuthorizationReceiptUsedForBothSides);
        }
        if !issues.is_empty() {
            return Err(issues);
        }

        let transition_sha256 = transition_digest(
            previous.root_sha256(),
            next.root_sha256(),
            transition_at_unix_s,
            &old_root_authorization_receipt_sha256,
            &new_root_authorization_receipt_sha256,
        );
        Ok(Self {
            schema_version: TRUST_ROOT_TRANSITION_SCHEMA.into(),
            previous_root_sha256: previous.root_sha256().clone(),
            next_root_sha256: next.root_sha256().clone(),
            transition_at_unix_s,
            old_root_authorization_receipt_sha256,
            new_root_authorization_receipt_sha256,
            transition_sha256,
        })
    }

    pub fn transition_sha256(&self) -> &Sha256Digest {
        &self.transition_sha256
    }

    /// Structural presence of two receipt digests is not proof that either
    /// receipt satisfies the appropriate old/new Root role threshold.
    pub const fn rotation_authority_established(&self) -> bool {
        false
    }
}

fn eligible_principals(
    directory: &TrustedPrincipalDirectory,
    policy: &TrustRolePolicy,
    issues: &mut Vec<TrustRootIssue>,
) -> BTreeSet<String> {
    let mut eligible = BTreeSet::new();
    let known: BTreeSet<_> = directory
        .principals()
        .iter()
        .map(|principal| principal.principal_id.as_str())
        .collect();
    if let Some(allowed) = &policy.allowed_principal_ids {
        for principal_id in allowed {
            if !known.contains(principal_id.as_str()) {
                issues.push(TrustRootIssue::AllowedPrincipalUnknown {
                    role: policy.role,
                    principal_id: principal_id.clone(),
                });
            }
        }
    }
    for principal in directory.principals() {
        if policy
            .allowed_principal_ids
            .as_ref()
            .is_some_and(|allowed| !allowed.contains(&principal.principal_id))
        {
            continue;
        }
        if principal
            .keys
            .iter()
            .any(|key| key.roles.contains(&policy.role))
        {
            eligible.insert(principal.principal_id.clone());
        }
    }
    eligible
}

#[allow(clippy::too_many_arguments)]
fn root_digest(
    version: u64,
    predecessor_root_sha256: Option<&Sha256Digest>,
    issued_at_unix_s: u64,
    expires_at_unix_s: u64,
    principal_directory_sequence: u64,
    principal_directory_sha256: &Sha256Digest,
    role_policies: &[TrustRolePolicy],
) -> Sha256Digest {
    let mut digest = FramedDigest::new(TRUST_ROOT_DOMAIN);
    digest.text(TRUST_ROOT_SCHEMA);
    digest.text(&version.to_string());
    digest.optional_sha(predecessor_root_sha256);
    digest.text(&issued_at_unix_s.to_string());
    digest.text(&expires_at_unix_s.to_string());
    digest.text(&principal_directory_sequence.to_string());
    digest.text(principal_directory_sha256.as_str());
    for policy in role_policies {
        digest.text("role-policy");
        digest.text(role_tag(policy.role));
        digest.text(&policy.minimum_valid_signatures.to_string());
        digest.text(&policy.minimum_distinct_principals.to_string());
        digest.text(&policy.minimum_distinct_organizations.to_string());
        digest.text(&policy.minimum_distinct_regions.to_string());
        for algorithm in &policy.required_algorithms {
            digest.text("required-algorithm");
            digest_algorithm(&mut digest, algorithm);
        }
        match &policy.allowed_principal_ids {
            None => digest.text("all-role-bound-principals-allowed"),
            Some(ids) => {
                digest.text("restricted-principals");
                for principal_id in ids {
                    digest.text(principal_id);
                }
            }
        }
    }
    digest.digest()
}

fn transition_digest(
    previous_root_sha256: &Sha256Digest,
    next_root_sha256: &Sha256Digest,
    transition_at_unix_s: u64,
    old_root_authorization_receipt_sha256: &Sha256Digest,
    new_root_authorization_receipt_sha256: &Sha256Digest,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(TRUST_ROOT_TRANSITION_DOMAIN);
    digest.text(TRUST_ROOT_TRANSITION_SCHEMA);
    digest.text(previous_root_sha256.as_str());
    digest.text(next_root_sha256.as_str());
    digest.text(&transition_at_unix_s.to_string());
    digest.text(old_root_authorization_receipt_sha256.as_str());
    digest.text(new_root_authorization_receipt_sha256.as_str());
    digest.text("dual-old-root-new-root-authorization-required");
    digest.text("rotation-authority-not-established");
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
    use crate::{TrustedKeyBinding, TrustedPrincipal};

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

    fn role(role: TrustRole) -> TrustRolePolicy {
        TrustRolePolicy {
            role,
            minimum_valid_signatures: 2,
            minimum_distinct_principals: 2,
            minimum_distinct_organizations: 2,
            minimum_distinct_regions: 2,
            required_algorithms: BTreeSet::new(),
            allowed_principal_ids: None,
        }
    }

    fn root(version: u64, predecessor: Option<Sha256Digest>) -> FrozenTrustRoot {
        TrustRootDraft {
            version,
            predecessor_root_sha256: predecessor,
            issued_at_unix_s: if version == 1 { 100 } else { 200 },
            expires_at_unix_s: 1_000,
            role_policies: vec![role(TrustRole::Root), role(TrustRole::Freshness)],
        }
        .freeze(&directory())
        .unwrap()
    }

    #[test]
    fn frozen_root_requires_root_and_freshness_roles() {
        let result = TrustRootDraft {
            version: 1,
            predecessor_root_sha256: None,
            issued_at_unix_s: 100,
            expires_at_unix_s: 1_000,
            role_policies: vec![role(TrustRole::Root)],
        }
        .freeze(&directory());
        assert!(result.is_err());
    }

    #[test]
    fn transition_requires_exact_predecessor_and_sequential_version() {
        let first = root(1, None);
        let second = root(2, Some(first.root_sha256().clone()));
        let transition = TrustRootTransitionContract::new(
            &first,
            &second,
            250,
            sha("old-root-approval"),
            sha("new-root-approval"),
        )
        .unwrap();
        assert!(!transition.rotation_authority_established());
    }

    #[test]
    fn one_receipt_cannot_stand_in_for_both_root_thresholds() {
        let first = root(1, None);
        let second = root(2, Some(first.root_sha256().clone()));
        let same = sha("same-approval");
        assert!(TrustRootTransitionContract::new(
            &first,
            &second,
            250,
            same.clone(),
            same,
        )
        .is_err());
    }
}
