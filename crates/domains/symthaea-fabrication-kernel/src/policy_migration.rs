// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit policy migration proofs.
//!
//! A version bump is not evidence that authority semantics were preserved.
//! This module binds the exact predecessor and successor policy digests and
//! requires every predecessor invariant to be retained, strengthened, or
//! explicitly waived with bounded incident evidence.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::threshold::VerifiedThresholdCeremony;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const POLICY_BINDING_SCHEMA: &str = "symthaea.fabrication.policy-binding.v1";
pub const POLICY_MIGRATION_SCHEMA: &str = "symthaea.fabrication.policy-migration.v1";
pub const MAX_POLICY_INVARIANTS: usize = 512;
pub const MAX_POLICY_DOMAIN_BYTES: usize = 128;
pub const MAX_POLICY_VERSION_BYTES: usize = 64;
pub const MAX_INVARIANT_NAME_BYTES: usize = 192;
pub const MAX_MIGRATION_RATIONALE_BYTES: usize = 4 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyInvariantBinding {
    pub name: String,
    pub digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyBinding {
    pub schema_version: String,
    pub domain: String,
    pub policy_version: String,
    pub policy_digest: Sha256Digest,
    pub invariants: Vec<PolicyInvariantBinding>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyInvariantDisposition {
    Retained,
    Strengthened,
    Waived {
        incident_digest: Sha256Digest,
        expires_at_unix_s: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyInvariantMigration {
    pub name: String,
    pub predecessor_digest: Sha256Digest,
    pub successor_digest: Option<Sha256Digest>,
    pub disposition: PolicyInvariantDisposition,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyMigrationPlan {
    pub schema_version: String,
    pub predecessor: PolicyBinding,
    pub successor: PolicyBinding,
    pub activates_at_unix_s: u64,
    pub rollback_deadline_unix_s: u64,
    pub rationale: String,
    pub migrations: Vec<PolicyInvariantMigration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyMigrationPolicy {
    pub maximum_activation_delay_s: u64,
    pub maximum_rollback_window_s: u64,
    pub maximum_waiver_lifetime_s: u64,
    pub allow_waivers: bool,
}

impl Default for PolicyMigrationPolicy {
    fn default() -> Self {
        Self {
            maximum_activation_delay_s: 7 * 24 * 60 * 60,
            maximum_rollback_window_s: 30 * 24 * 60 * 60,
            maximum_waiver_lifetime_s: 24 * 60 * 60,
            allow_waivers: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizedPolicyMigration {
    pub plan: PolicyMigrationPlan,
    pub plan_digest: Sha256Digest,
    pub ceremony_digest: Sha256Digest,
    pub trust_snapshot_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyMigrationError {
    UnsupportedSchema,
    InvalidDomain,
    InvalidVersion,
    SamePolicy,
    EmptyInvariants,
    TooManyInvariants { actual: usize, maximum: usize },
    InvalidInvariantName(String),
    DuplicateInvariant(String),
    PolicyDigestZero,
    InvalidActivationWindow,
    ActivationTooLate,
    RollbackWindowTooLong,
    InvalidRationale,
    MigrationCountMismatch,
    UnknownPredecessorInvariant(String),
    DuplicateMigration(String),
    PredecessorDigestMismatch(String),
    MissingPredecessorInvariant(String),
    SuccessorInvariantMissing(String),
    SuccessorDigestMismatch(String),
    RetainedInvariantChanged(String),
    StrengthenedInvariantUnchanged(String),
    WaiverForbidden(String),
    WaiverHasSuccessorDigest(String),
    WaiverExpired(String),
    WaiverTooLong(String),
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    Encoding(String),
}

impl PolicyBinding {
    pub fn new(
        domain: impl Into<String>,
        policy_version: impl Into<String>,
        policy_digest: Sha256Digest,
        mut invariants: Vec<PolicyInvariantBinding>,
    ) -> Result<Self, PolicyMigrationError> {
        invariants.sort_by(|left, right| left.name.cmp(&right.name));
        let value = Self {
            schema_version: POLICY_BINDING_SCHEMA.into(),
            domain: domain.into(),
            policy_version: policy_version.into(),
            policy_digest,
            invariants,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), PolicyMigrationError> {
        if self.schema_version != POLICY_BINDING_SCHEMA {
            return Err(PolicyMigrationError::UnsupportedSchema);
        }
        validate_token(&self.domain, MAX_POLICY_DOMAIN_BYTES)
            .map_err(|_| PolicyMigrationError::InvalidDomain)?;
        validate_token(&self.policy_version, MAX_POLICY_VERSION_BYTES)
            .map_err(|_| PolicyMigrationError::InvalidVersion)?;
        if self.policy_digest.0 == [0; 32] {
            return Err(PolicyMigrationError::PolicyDigestZero);
        }
        if self.invariants.is_empty() {
            return Err(PolicyMigrationError::EmptyInvariants);
        }
        if self.invariants.len() > MAX_POLICY_INVARIANTS {
            return Err(PolicyMigrationError::TooManyInvariants {
                actual: self.invariants.len(),
                maximum: MAX_POLICY_INVARIANTS,
            });
        }
        let mut names = BTreeSet::new();
        let mut previous = None::<&str>;
        for invariant in &self.invariants {
            validate_token(&invariant.name, MAX_INVARIANT_NAME_BYTES)
                .map_err(|_| PolicyMigrationError::InvalidInvariantName(invariant.name.clone()))?;
            if invariant.digest.0 == [0; 32] {
                return Err(PolicyMigrationError::SuccessorDigestMismatch(
                    invariant.name.clone(),
                ));
            }
            if !names.insert(invariant.name.clone()) {
                return Err(PolicyMigrationError::DuplicateInvariant(
                    invariant.name.clone(),
                ));
            }
            if previous.is_some_and(|name| name >= invariant.name.as_str()) {
                return Err(PolicyMigrationError::DuplicateInvariant(
                    invariant.name.clone(),
                ));
            }
            previous = Some(invariant.name.as_str());
        }
        Ok(())
    }

    pub fn invariant_map(&self) -> BTreeMap<&str, Sha256Digest> {
        self.invariants
            .iter()
            .map(|item| (item.name.as_str(), item.digest))
            .collect()
    }
}

impl PolicyMigrationPlan {
    pub fn validate(
        &self,
        policy: &PolicyMigrationPolicy,
        proposed_at_unix_s: u64,
    ) -> Result<(), PolicyMigrationError> {
        if self.schema_version != POLICY_MIGRATION_SCHEMA {
            return Err(PolicyMigrationError::UnsupportedSchema);
        }
        self.predecessor.validate()?;
        self.successor.validate()?;
        if self.predecessor.domain != self.successor.domain {
            return Err(PolicyMigrationError::InvalidDomain);
        }
        if self.predecessor.policy_digest == self.successor.policy_digest
            || self.predecessor.policy_version == self.successor.policy_version
        {
            return Err(PolicyMigrationError::SamePolicy);
        }
        if self.activates_at_unix_s < proposed_at_unix_s
            || self.rollback_deadline_unix_s <= self.activates_at_unix_s
        {
            return Err(PolicyMigrationError::InvalidActivationWindow);
        }
        if self.activates_at_unix_s - proposed_at_unix_s > policy.maximum_activation_delay_s {
            return Err(PolicyMigrationError::ActivationTooLate);
        }
        if self.rollback_deadline_unix_s - self.activates_at_unix_s
            > policy.maximum_rollback_window_s
        {
            return Err(PolicyMigrationError::RollbackWindowTooLong);
        }
        if self.rationale.trim().is_empty()
            || self.rationale != self.rationale.trim()
            || self.rationale.len() > MAX_MIGRATION_RATIONALE_BYTES
            || self.rationale.chars().any(char::is_control)
        {
            return Err(PolicyMigrationError::InvalidRationale);
        }
        if self.migrations.len() != self.predecessor.invariants.len() {
            return Err(PolicyMigrationError::MigrationCountMismatch);
        }

        let predecessor = self.predecessor.invariant_map();
        let successor = self.successor.invariant_map();
        let mut migrated = BTreeSet::new();
        for migration in &self.migrations {
            let Some(expected_predecessor) = predecessor.get(migration.name.as_str()) else {
                return Err(PolicyMigrationError::UnknownPredecessorInvariant(
                    migration.name.clone(),
                ));
            };
            if !migrated.insert(migration.name.clone()) {
                return Err(PolicyMigrationError::DuplicateMigration(
                    migration.name.clone(),
                ));
            }
            if migration.predecessor_digest != *expected_predecessor {
                return Err(PolicyMigrationError::PredecessorDigestMismatch(
                    migration.name.clone(),
                ));
            }
            match &migration.disposition {
                PolicyInvariantDisposition::Retained => {
                    let Some(successor_digest) = successor.get(migration.name.as_str()) else {
                        return Err(PolicyMigrationError::SuccessorInvariantMissing(
                            migration.name.clone(),
                        ));
                    };
                    if migration.successor_digest != Some(*successor_digest) {
                        return Err(PolicyMigrationError::SuccessorDigestMismatch(
                            migration.name.clone(),
                        ));
                    }
                    if *successor_digest != migration.predecessor_digest {
                        return Err(PolicyMigrationError::RetainedInvariantChanged(
                            migration.name.clone(),
                        ));
                    }
                }
                PolicyInvariantDisposition::Strengthened => {
                    let Some(successor_digest) = successor.get(migration.name.as_str()) else {
                        return Err(PolicyMigrationError::SuccessorInvariantMissing(
                            migration.name.clone(),
                        ));
                    };
                    if migration.successor_digest != Some(*successor_digest) {
                        return Err(PolicyMigrationError::SuccessorDigestMismatch(
                            migration.name.clone(),
                        ));
                    }
                    if *successor_digest == migration.predecessor_digest {
                        return Err(PolicyMigrationError::StrengthenedInvariantUnchanged(
                            migration.name.clone(),
                        ));
                    }
                }
                PolicyInvariantDisposition::Waived {
                    incident_digest,
                    expires_at_unix_s,
                } => {
                    if !policy.allow_waivers {
                        return Err(PolicyMigrationError::WaiverForbidden(
                            migration.name.clone(),
                        ));
                    }
                    if migration.successor_digest.is_some()
                        || successor.contains_key(migration.name.as_str())
                    {
                        return Err(PolicyMigrationError::WaiverHasSuccessorDigest(
                            migration.name.clone(),
                        ));
                    }
                    if incident_digest.0 == [0; 32]
                        || *expires_at_unix_s <= self.activates_at_unix_s
                    {
                        return Err(PolicyMigrationError::WaiverExpired(migration.name.clone()));
                    }
                    if *expires_at_unix_s - self.activates_at_unix_s
                        > policy.maximum_waiver_lifetime_s
                    {
                        return Err(PolicyMigrationError::WaiverTooLong(migration.name.clone()));
                    }
                }
            }
        }
        for invariant in predecessor.keys() {
            if !migrated.contains(*invariant) {
                return Err(PolicyMigrationError::MissingPredecessorInvariant(
                    (*invariant).into(),
                ));
            }
        }
        Ok(())
    }
}

pub fn digest_policy_binding(
    binding: &PolicyBinding,
) -> Result<Sha256Digest, PolicyMigrationError> {
    binding.validate()?;
    let bytes = serde_json::to_vec(binding)
        .map_err(|error| PolicyMigrationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.policy-binding-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn digest_policy_migration_plan(
    plan: &PolicyMigrationPlan,
    policy: &PolicyMigrationPolicy,
    proposed_at_unix_s: u64,
) -> Result<Sha256Digest, PolicyMigrationError> {
    plan.validate(policy, proposed_at_unix_s)?;
    let bytes = serde_json::to_vec(plan)
        .map_err(|error| PolicyMigrationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.policy-migration-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn authorize_policy_migration(
    plan: PolicyMigrationPlan,
    policy: &PolicyMigrationPolicy,
    proposed_at_unix_s: u64,
    ceremony: &VerifiedThresholdCeremony,
) -> Result<AuthorizedPolicyMigration, PolicyMigrationError> {
    let plan_digest = digest_policy_migration_plan(&plan, policy, proposed_at_unix_s)?;
    if ceremony.purpose() != "policy-migration" {
        return Err(PolicyMigrationError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != plan_digest {
        return Err(PolicyMigrationError::CeremonyPayloadMismatch);
    }
    Ok(AuthorizedPolicyMigration {
        plan,
        plan_digest,
        ceremony_digest: ceremony.ceremony_digest(),
        trust_snapshot_digest: ceremony.trust_snapshot_digest(),
    })
}

fn validate_token(value: &str, maximum: usize) -> Result<(), ()> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > maximum
        || value.chars().any(char::is_control)
    {
        return Err(());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;

    fn binding(version: &str, safety: &[u8], audit: &[u8]) -> PolicyBinding {
        PolicyBinding::new(
            "machine-authority",
            version,
            sha256(version.as_bytes()),
            vec![
                PolicyInvariantBinding {
                    name: "audit-prefix".into(),
                    digest: sha256(audit),
                },
                PolicyInvariantBinding {
                    name: "safety-envelope".into(),
                    digest: sha256(safety),
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn every_predecessor_invariant_requires_an_explicit_disposition() {
        let predecessor = binding("1", b"s1", b"a1");
        let successor = binding("2", b"s2", b"a1");
        let plan = PolicyMigrationPlan {
            schema_version: POLICY_MIGRATION_SCHEMA.into(),
            predecessor: predecessor.clone(),
            successor: successor.clone(),
            activates_at_unix_s: 200,
            rollback_deadline_unix_s: 300,
            rationale: "strengthen the safety envelope".into(),
            migrations: vec![PolicyInvariantMigration {
                name: "safety-envelope".into(),
                predecessor_digest: sha256(b"s1"),
                successor_digest: Some(sha256(b"s2")),
                disposition: PolicyInvariantDisposition::Strengthened,
            }],
        };
        assert_eq!(
            plan.validate(&PolicyMigrationPolicy::default(), 100),
            Err(PolicyMigrationError::MigrationCountMismatch)
        );
    }

    #[test]
    fn retained_and_strengthened_invariants_validate() {
        let plan = PolicyMigrationPlan {
            schema_version: POLICY_MIGRATION_SCHEMA.into(),
            predecessor: binding("1", b"s1", b"a1"),
            successor: binding("2", b"s2", b"a1"),
            activates_at_unix_s: 200,
            rollback_deadline_unix_s: 300,
            rationale: "strengthen the safety envelope".into(),
            migrations: vec![
                PolicyInvariantMigration {
                    name: "audit-prefix".into(),
                    predecessor_digest: sha256(b"a1"),
                    successor_digest: Some(sha256(b"a1")),
                    disposition: PolicyInvariantDisposition::Retained,
                },
                PolicyInvariantMigration {
                    name: "safety-envelope".into(),
                    predecessor_digest: sha256(b"s1"),
                    successor_digest: Some(sha256(b"s2")),
                    disposition: PolicyInvariantDisposition::Strengthened,
                },
            ],
        };
        plan.validate(&PolicyMigrationPolicy::default(), 100)
            .unwrap();
    }
}
