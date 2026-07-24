// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Portable, bounded upgrade evidence bundles.

use crate::authority_epoch::{AuthorityEpochTracker, AuthorityEpochVector};
use crate::clock::{ClockEpochTracker, VerifiedClockWindow};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::evidence_compaction::{
    CompactedEvidence, EvidenceCompactionPolicy, EvidenceCompactionTracker,
};
use crate::policy_migration::AuthorizedPolicyMigration;
use crate::policy_migration_tracker::PolicyMigrationTracker;
use crate::recovery_key::{RecoveryActivationTracker, RecoveryKeySet};
use crate::upgrade_handoff::AuthorizedUpgradeHandoff;
use crate::upgrade_replay::{
    UpgradeReplayContract, UpgradeReplayError, build_upgrade_replay_contract,
    digest_upgrade_replay_contract, verify_upgrade_replay_contract,
};
use crate::upgrade_state::FabricationUpgradeState;
use crate::upgrade_tracker::UpgradeHandoffTracker;
use serde::{Deserialize, Serialize};

pub const UPGRADE_BUNDLE_SCHEMA: &str = "symthaea.fabrication.upgrade-bundle.v1";
pub const DEFAULT_MAX_UPGRADE_BUNDLE_BYTES: usize = 16 * 1024 * 1024;
pub const HARD_MAX_UPGRADE_BUNDLE_BYTES: usize = 64 * 1024 * 1024;
pub const DEFAULT_MAX_BUNDLE_MIGRATIONS: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradeEvidenceBundle {
    pub schema_version: String,
    pub source_tree_digest: Sha256Digest,
    pub handoff: AuthorizedUpgradeHandoff,
    pub migrations: Vec<AuthorizedPolicyMigration>,
    pub policy_tracker: PolicyMigrationTracker,
    pub evaluation_time_unix_s: u64,
    pub clock: VerifiedClockWindow,
    pub clock_tracker: ClockEpochTracker,
    pub authority_epoch: AuthorityEpochVector,
    pub authority_epoch_tracker: AuthorityEpochTracker,
    pub recovery_key_set: RecoveryKeySet,
    pub recovery_tracker: RecoveryActivationTracker,
    pub compacted_evidence: CompactedEvidence,
    pub compaction_policy: EvidenceCompactionPolicy,
    pub compaction_tracker: EvidenceCompactionTracker,
    pub upgrade_tracker: UpgradeHandoffTracker,
    pub upgrade_state: FabricationUpgradeState,
    pub replay_contract: UpgradeReplayContract,
    pub replay_contract_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpgradeBundleLimits {
    pub maximum_bytes: usize,
    pub maximum_migrations: usize,
}

impl Default for UpgradeBundleLimits {
    fn default() -> Self {
        Self {
            maximum_bytes: DEFAULT_MAX_UPGRADE_BUNDLE_BYTES,
            maximum_migrations: DEFAULT_MAX_BUNDLE_MIGRATIONS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpgradeBundleError {
    UnsupportedSchema,
    InvalidLimits,
    BundleTooLarge { actual: usize, maximum: usize },
    TooManyMigrations { actual: usize, maximum: usize },
    ReplayDigestMismatch,
    ReplayMismatch(String),
    Replay(UpgradeReplayError),
    Encoding(String),
}

#[allow(clippy::too_many_arguments)]
pub fn build_upgrade_evidence_bundle(
    source_tree_digest: Sha256Digest,
    handoff: AuthorizedUpgradeHandoff,
    migrations: Vec<AuthorizedPolicyMigration>,
    policy_tracker: PolicyMigrationTracker,
    evaluation_time_unix_s: u64,
    clock: VerifiedClockWindow,
    clock_tracker: ClockEpochTracker,
    authority_epoch: AuthorityEpochVector,
    authority_epoch_tracker: AuthorityEpochTracker,
    recovery_key_set: RecoveryKeySet,
    recovery_tracker: RecoveryActivationTracker,
    compacted_evidence: CompactedEvidence,
    compaction_policy: EvidenceCompactionPolicy,
    compaction_tracker: EvidenceCompactionTracker,
    upgrade_tracker: UpgradeHandoffTracker,
    upgrade_state: FabricationUpgradeState,
) -> Result<UpgradeEvidenceBundle, UpgradeBundleError> {
    let replay_contract = build_upgrade_replay_contract(
        source_tree_digest,
        &handoff,
        &migrations,
        &policy_tracker,
        evaluation_time_unix_s,
        &clock,
        &clock_tracker,
        &authority_epoch,
        &authority_epoch_tracker,
        &recovery_key_set,
        &recovery_tracker,
        &compacted_evidence,
        &compaction_policy,
        &compaction_tracker,
        &upgrade_tracker,
        &upgrade_state,
    )
    .map_err(UpgradeBundleError::Replay)?;
    let replay_contract_digest =
        digest_upgrade_replay_contract(&replay_contract).map_err(UpgradeBundleError::Replay)?;
    Ok(UpgradeEvidenceBundle {
        schema_version: UPGRADE_BUNDLE_SCHEMA.into(),
        source_tree_digest,
        handoff,
        migrations,
        policy_tracker,
        evaluation_time_unix_s,
        clock,
        clock_tracker,
        authority_epoch,
        authority_epoch_tracker,
        recovery_key_set,
        recovery_tracker,
        compacted_evidence,
        compaction_policy,
        compaction_tracker,
        upgrade_tracker,
        upgrade_state,
        replay_contract,
        replay_contract_digest,
    })
}

pub fn verify_upgrade_evidence_bundle(
    bundle: &UpgradeEvidenceBundle,
    limits: &UpgradeBundleLimits,
) -> Result<(), UpgradeBundleError> {
    validate_limits(limits)?;
    if bundle.schema_version != UPGRADE_BUNDLE_SCHEMA {
        return Err(UpgradeBundleError::UnsupportedSchema);
    }
    if bundle.migrations.len() > limits.maximum_migrations {
        return Err(UpgradeBundleError::TooManyMigrations {
            actual: bundle.migrations.len(),
            maximum: limits.maximum_migrations,
        });
    }
    let replay_digest = digest_upgrade_replay_contract(&bundle.replay_contract)
        .map_err(UpgradeBundleError::Replay)?;
    if replay_digest != bundle.replay_contract_digest {
        return Err(UpgradeBundleError::ReplayDigestMismatch);
    }
    let report = verify_upgrade_replay_contract(
        &bundle.replay_contract,
        bundle.source_tree_digest,
        &bundle.handoff,
        &bundle.migrations,
        &bundle.policy_tracker,
        bundle.evaluation_time_unix_s,
        &bundle.clock,
        &bundle.clock_tracker,
        &bundle.authority_epoch,
        &bundle.authority_epoch_tracker,
        &bundle.recovery_key_set,
        &bundle.recovery_tracker,
        &bundle.compacted_evidence,
        &bundle.compaction_policy,
        &bundle.compaction_tracker,
        &bundle.upgrade_tracker,
        &bundle.upgrade_state,
    )
    .map_err(UpgradeBundleError::Replay)?;
    if !report.exact() {
        return Err(UpgradeBundleError::ReplayMismatch(format!(
            "{:?}",
            report.mismatches
        )));
    }
    Ok(())
}

pub fn encode_upgrade_evidence_bundle(
    bundle: &UpgradeEvidenceBundle,
    limits: &UpgradeBundleLimits,
) -> Result<Vec<u8>, UpgradeBundleError> {
    verify_upgrade_evidence_bundle(bundle, limits)?;
    let bytes = serde_json::to_vec(bundle)
        .map_err(|error| UpgradeBundleError::Encoding(error.to_string()))?;
    if bytes.len() > limits.maximum_bytes {
        return Err(UpgradeBundleError::BundleTooLarge {
            actual: bytes.len(),
            maximum: limits.maximum_bytes,
        });
    }
    Ok(bytes)
}

pub fn decode_upgrade_evidence_bundle(
    bytes: &[u8],
    limits: &UpgradeBundleLimits,
) -> Result<UpgradeEvidenceBundle, UpgradeBundleError> {
    validate_limits(limits)?;
    if bytes.len() > limits.maximum_bytes {
        return Err(UpgradeBundleError::BundleTooLarge {
            actual: bytes.len(),
            maximum: limits.maximum_bytes,
        });
    }
    let bundle: UpgradeEvidenceBundle = serde_json::from_slice(bytes)
        .map_err(|error| UpgradeBundleError::Encoding(error.to_string()))?;
    verify_upgrade_evidence_bundle(&bundle, limits)?;
    Ok(bundle)
}

pub fn digest_upgrade_evidence_bundle(
    bundle: &UpgradeEvidenceBundle,
    limits: &UpgradeBundleLimits,
) -> Result<Sha256Digest, UpgradeBundleError> {
    let bytes = encode_upgrade_evidence_bundle(bundle, limits)?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.upgrade-bundle-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

fn validate_limits(limits: &UpgradeBundleLimits) -> Result<(), UpgradeBundleError> {
    if limits.maximum_bytes == 0
        || limits.maximum_bytes > HARD_MAX_UPGRADE_BUNDLE_BYTES
        || limits.maximum_migrations == 0
        || limits.maximum_migrations > DEFAULT_MAX_BUNDLE_MIGRATIONS
    {
        return Err(UpgradeBundleError::InvalidLimits);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_input_is_rejected_before_json_allocation() {
        let limits = UpgradeBundleLimits {
            maximum_bytes: 8,
            maximum_migrations: 1,
        };
        assert_eq!(
            decode_upgrade_evidence_bundle(&[0; 9], &limits),
            Err(UpgradeBundleError::BundleTooLarge {
                actual: 9,
                maximum: 8
            })
        );
    }
}
