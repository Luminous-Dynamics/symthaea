// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Secure authority handoff between fabrication-kernel releases.
//!
//! Upgrade authority binds both executable identity and durable operational
//! state. The successor must dominate the predecessor's authority epochs, carry
//! explicit policy-migration evidence, use quorum-derived time, preserve an
//! evidence-compaction checkpoint, and retain a bounded rollback target.

use crate::authority_epoch::{
    AuthorityEpochTrackingError, AuthorityEpochVector, digest_authority_epoch,
};
use crate::clock::VerifiedClockWindow;
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::policy_migration::AuthorizedPolicyMigration;
use crate::threshold::VerifiedThresholdCeremony;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const UPGRADE_ENDPOINT_SCHEMA: &str = "symthaea.fabrication.upgrade-endpoint.v1";
pub const UPGRADE_HANDOFF_SCHEMA: &str = "symthaea.fabrication.upgrade-handoff.v1";
pub const MAX_UPGRADE_VERSION_BYTES: usize = 96;
pub const MAX_UPGRADE_MIGRATIONS: usize = 128;
pub const MAX_UPGRADE_REASON_BYTES: usize = 4 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradeEndpoint {
    pub schema_version: String,
    pub software_version: String,
    pub source_tree_digest: Sha256Digest,
    pub executable_digest: Sha256Digest,
    pub durable_state_digest: Sha256Digest,
    pub replay_contract_digest: Sha256Digest,
    pub authority_epoch: AuthorityEpochVector,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradeHandoffPlan {
    pub schema_version: String,
    pub predecessor: UpgradeEndpoint,
    pub successor: UpgradeEndpoint,
    pub prepared_at_unix_ms: u64,
    pub activates_at_unix_ms: u64,
    pub finalization_deadline_unix_ms: u64,
    pub rollback_target_digest: Sha256Digest,
    pub policy_migration_digests: Vec<Sha256Digest>,
    pub clock_evidence_digest: Sha256Digest,
    pub evidence_checkpoint_digest: Sha256Digest,
    pub recovery_key_set_digest: Sha256Digest,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpgradeHandoffPolicy {
    pub maximum_activation_delay_ms: u64,
    pub maximum_finalization_window_ms: u64,
    pub require_policy_migration: bool,
}

impl Default for UpgradeHandoffPolicy {
    fn default() -> Self {
        Self {
            maximum_activation_delay_ms: 24 * 60 * 60 * 1_000,
            maximum_finalization_window_ms: 7 * 24 * 60 * 60 * 1_000,
            require_policy_migration: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizedUpgradeHandoff {
    pub plan: UpgradeHandoffPlan,
    pub plan_digest: Sha256Digest,
    pub ceremony_digest: Sha256Digest,
    pub trust_snapshot_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpgradeHandoffError {
    UnsupportedSchema,
    InvalidVersion,
    SameVersion,
    ZeroDigest(&'static str),
    SameSourceTree,
    SameExecutable,
    SameDurableState,
    InvalidAuthorityEpoch,
    AuthorityEpochRollback(AuthorityEpochTrackingError),
    InvalidWindow,
    ActivationTooLate,
    FinalizationWindowTooLong,
    ClockEvidenceMismatch,
    ClockDoesNotCoverPreparation,
    MissingPolicyMigration,
    TooManyPolicyMigrations { actual: usize, maximum: usize },
    DuplicatePolicyMigration,
    PolicyMigrationDigestMismatch,
    PolicyMigrationTargetsWrongWindow,
    InvalidReason,
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    Encoding(String),
}

impl UpgradeEndpoint {
    pub fn validate(&self) -> Result<(), UpgradeHandoffError> {
        if self.schema_version != UPGRADE_ENDPOINT_SCHEMA {
            return Err(UpgradeHandoffError::UnsupportedSchema);
        }
        validate_version(&self.software_version)?;
        for (name, digest) in [
            ("source_tree_digest", self.source_tree_digest),
            ("executable_digest", self.executable_digest),
            ("durable_state_digest", self.durable_state_digest),
            ("replay_contract_digest", self.replay_contract_digest),
        ] {
            if digest.0 == [0; 32] {
                return Err(UpgradeHandoffError::ZeroDigest(name));
            }
        }
        self.authority_epoch
            .validate()
            .map_err(|_| UpgradeHandoffError::InvalidAuthorityEpoch)?;
        Ok(())
    }
}

impl UpgradeHandoffPlan {
    pub fn validate(
        &self,
        policy: &UpgradeHandoffPolicy,
        clock: &VerifiedClockWindow,
        migrations: &[AuthorizedPolicyMigration],
    ) -> Result<(), UpgradeHandoffError> {
        if self.schema_version != UPGRADE_HANDOFF_SCHEMA {
            return Err(UpgradeHandoffError::UnsupportedSchema);
        }
        self.predecessor.validate()?;
        self.successor.validate()?;
        if self.predecessor.software_version == self.successor.software_version {
            return Err(UpgradeHandoffError::SameVersion);
        }
        if self.predecessor.source_tree_digest == self.successor.source_tree_digest {
            return Err(UpgradeHandoffError::SameSourceTree);
        }
        if self.predecessor.executable_digest == self.successor.executable_digest {
            return Err(UpgradeHandoffError::SameExecutable);
        }
        if self.predecessor.durable_state_digest == self.successor.durable_state_digest {
            return Err(UpgradeHandoffError::SameDurableState);
        }
        let advanced = self
            .successor
            .authority_epoch
            .dominates(&self.predecessor.authority_epoch)
            .map_err(UpgradeHandoffError::AuthorityEpochRollback)?;
        if !advanced {
            return Err(UpgradeHandoffError::InvalidAuthorityEpoch);
        }
        if self.prepared_at_unix_ms > self.activates_at_unix_ms
            || self.activates_at_unix_ms >= self.finalization_deadline_unix_ms
        {
            return Err(UpgradeHandoffError::InvalidWindow);
        }
        if self.activates_at_unix_ms - self.prepared_at_unix_ms > policy.maximum_activation_delay_ms
        {
            return Err(UpgradeHandoffError::ActivationTooLate);
        }
        if self.finalization_deadline_unix_ms - self.activates_at_unix_ms
            > policy.maximum_finalization_window_ms
        {
            return Err(UpgradeHandoffError::FinalizationWindowTooLong);
        }
        if self.clock_evidence_digest != clock.evidence_digest {
            return Err(UpgradeHandoffError::ClockEvidenceMismatch);
        }
        if self.prepared_at_unix_ms < clock.lower_unix_ms
            || self.prepared_at_unix_ms > clock.upper_unix_ms
        {
            return Err(UpgradeHandoffError::ClockDoesNotCoverPreparation);
        }
        for (name, digest) in [
            ("rollback_target_digest", self.rollback_target_digest),
            ("clock_evidence_digest", self.clock_evidence_digest),
            (
                "evidence_checkpoint_digest",
                self.evidence_checkpoint_digest,
            ),
            ("recovery_key_set_digest", self.recovery_key_set_digest),
        ] {
            if digest.0 == [0; 32] {
                return Err(UpgradeHandoffError::ZeroDigest(name));
            }
        }
        if policy.require_policy_migration && self.policy_migration_digests.is_empty() {
            return Err(UpgradeHandoffError::MissingPolicyMigration);
        }
        if self.policy_migration_digests.len() > MAX_UPGRADE_MIGRATIONS {
            return Err(UpgradeHandoffError::TooManyPolicyMigrations {
                actual: self.policy_migration_digests.len(),
                maximum: MAX_UPGRADE_MIGRATIONS,
            });
        }
        let unique = self
            .policy_migration_digests
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if unique.len() != self.policy_migration_digests.len() {
            return Err(UpgradeHandoffError::DuplicatePolicyMigration);
        }
        let supplied = migrations
            .iter()
            .map(|migration| migration.plan_digest)
            .collect::<BTreeSet<_>>();
        if unique != supplied {
            return Err(UpgradeHandoffError::PolicyMigrationDigestMismatch);
        }
        if migrations.iter().any(|migration| {
            migration.plan.activates_at_unix_s.saturating_mul(1_000) > self.activates_at_unix_ms
                || migration
                    .plan
                    .rollback_deadline_unix_s
                    .saturating_mul(1_000)
                    < self.finalization_deadline_unix_ms
        }) {
            return Err(UpgradeHandoffError::PolicyMigrationTargetsWrongWindow);
        }
        if self.reason.trim().is_empty()
            || self.reason != self.reason.trim()
            || self.reason.len() > MAX_UPGRADE_REASON_BYTES
            || self.reason.chars().any(char::is_control)
        {
            return Err(UpgradeHandoffError::InvalidReason);
        }
        Ok(())
    }
}

pub fn digest_upgrade_endpoint(
    endpoint: &UpgradeEndpoint,
) -> Result<Sha256Digest, UpgradeHandoffError> {
    endpoint.validate()?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.upgrade-endpoint-digest.v1\0");
    hasher.update(endpoint.software_version.as_bytes());
    hasher.update(&endpoint.source_tree_digest.0);
    hasher.update(&endpoint.executable_digest.0);
    hasher.update(&endpoint.durable_state_digest.0);
    hasher.update(&endpoint.replay_contract_digest.0);
    hasher.update(
        &digest_authority_epoch(&endpoint.authority_epoch)
            .map_err(|_| UpgradeHandoffError::InvalidAuthorityEpoch)?
            .0,
    );
    Ok(hasher.finalize())
}

pub fn digest_upgrade_handoff_plan(
    plan: &UpgradeHandoffPlan,
    policy: &UpgradeHandoffPolicy,
    clock: &VerifiedClockWindow,
    migrations: &[AuthorizedPolicyMigration],
) -> Result<Sha256Digest, UpgradeHandoffError> {
    plan.validate(policy, clock, migrations)?;
    let bytes = serde_json::to_vec(plan)
        .map_err(|error| UpgradeHandoffError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.upgrade-handoff-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn authorize_upgrade_handoff(
    plan: UpgradeHandoffPlan,
    policy: &UpgradeHandoffPolicy,
    clock: &VerifiedClockWindow,
    migrations: &[AuthorizedPolicyMigration],
    ceremony: &VerifiedThresholdCeremony,
) -> Result<AuthorizedUpgradeHandoff, UpgradeHandoffError> {
    let plan_digest = digest_upgrade_handoff_plan(&plan, policy, clock, migrations)?;
    if ceremony.purpose() != "upgrade-handoff" {
        return Err(UpgradeHandoffError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != plan_digest {
        return Err(UpgradeHandoffError::CeremonyPayloadMismatch);
    }
    Ok(AuthorizedUpgradeHandoff {
        plan,
        plan_digest,
        ceremony_digest: ceremony.ceremony_digest(),
        trust_snapshot_digest: ceremony.trust_snapshot_digest(),
    })
}

fn validate_version(version: &str) -> Result<(), UpgradeHandoffError> {
    if version.trim().is_empty()
        || version != version.trim()
        || version.len() > MAX_UPGRADE_VERSION_BYTES
        || version.chars().any(char::is_control)
        || !version
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || ".-+_".contains(character))
    {
        return Err(UpgradeHandoffError::InvalidVersion);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authority_epoch::AuthorityEpochVector;
    use crate::crypto_digest::sha256;

    fn endpoint(version: &str, generation: u64) -> UpgradeEndpoint {
        UpgradeEndpoint {
            schema_version: UPGRADE_ENDPOINT_SCHEMA.into(),
            software_version: version.into(),
            source_tree_digest: sha256(format!("source-{version}").as_bytes()),
            executable_digest: sha256(format!("exe-{version}").as_bytes()),
            durable_state_digest: sha256(format!("state-{version}").as_bytes()),
            replay_contract_digest: sha256(format!("replay-{version}").as_bytes()),
            authority_epoch: AuthorityEpochVector::new(2, 2, generation, 1, 1, 5, 3, 2).unwrap(),
        }
    }

    #[test]
    fn endpoint_identity_binds_state_and_authority_epoch() {
        let first = endpoint("0.17.0", 4);
        let mut changed = first.clone();
        changed.authority_epoch.gateway_generation = 5;
        assert_ne!(
            digest_upgrade_endpoint(&first).unwrap(),
            digest_upgrade_endpoint(&changed).unwrap()
        );
    }

    #[test]
    fn successor_epoch_must_dominate_predecessor() {
        let predecessor = endpoint("0.17.0", 5);
        let successor = endpoint("0.18.0", 4);
        assert!(matches!(
            successor
                .authority_epoch
                .dominates(&predecessor.authority_epoch),
            Err(AuthorityEpochTrackingError::ComponentRollback { .. })
        ));
    }
}
