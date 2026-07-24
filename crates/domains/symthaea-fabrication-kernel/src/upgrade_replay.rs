// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic replay contract for secure upgrade authority.

use crate::authority_epoch::{
    AuthorityEpochTracker, AuthorityEpochVector, digest_authority_epoch,
    digest_authority_epoch_tracker,
};
use crate::clock::{ClockEpochTracker, VerifiedClockWindow, digest_clock_epoch_tracker};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::evidence_compaction::{
    CompactedEvidence, EvidenceCompactionPolicy, EvidenceCompactionTracker,
    digest_compacted_evidence, digest_evidence_compaction_tracker,
};
use crate::policy_migration::AuthorizedPolicyMigration;
use crate::policy_migration_tracker::{PolicyMigrationTracker, digest_policy_migration_tracker};
use crate::recovery_key::{
    RecoveryActivationTracker, RecoveryKeySet, digest_recovery_activation_tracker,
    digest_recovery_key_set,
};
use crate::upgrade_handoff::AuthorizedUpgradeHandoff;
use crate::upgrade_state::{FabricationUpgradeState, digest_upgrade_state};
use crate::upgrade_tracker::{UpgradeHandoffTracker, digest_upgrade_tracker};
use serde::{Deserialize, Serialize};

pub const UPGRADE_REPLAY_SCHEMA: &str = "symthaea.fabrication.upgrade-replay.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradeReplayContract {
    pub schema_version: String,
    pub source_tree_digest: Sha256Digest,
    pub handoff_digest: Sha256Digest,
    pub policy_migration_set_digest: Sha256Digest,
    pub policy_migration_tracker_digest: Sha256Digest,
    pub clock_evidence_digest: Sha256Digest,
    pub clock_tracker_digest: Sha256Digest,
    pub authority_epoch_digest: Sha256Digest,
    pub authority_epoch_tracker_digest: Sha256Digest,
    pub recovery_key_set_digest: Sha256Digest,
    pub recovery_tracker_digest: Sha256Digest,
    pub evidence_compaction_digest: Sha256Digest,
    pub evidence_compaction_tracker_digest: Sha256Digest,
    pub upgrade_tracker_digest: Sha256Digest,
    pub upgrade_state_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpgradeReplayMismatch {
    SourceTree,
    Handoff,
    PolicyMigrationSet,
    PolicyMigrationTracker,
    ClockEvidence,
    ClockTracker,
    AuthorityEpoch,
    AuthorityEpochTracker,
    RecoveryKeySet,
    RecoveryTracker,
    EvidenceCompaction,
    EvidenceCompactionTracker,
    UpgradeTracker,
    UpgradeState,
    StateEvidence(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpgradeReplayVerificationReport {
    pub mismatches: Vec<UpgradeReplayMismatch>,
}

impl UpgradeReplayVerificationReport {
    pub fn exact(&self) -> bool {
        self.mismatches.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpgradeReplayError {
    UnsupportedSchema,
    ZeroSourceTree,
    Evidence(String),
    Encoding(String),
}

#[allow(clippy::too_many_arguments)]
pub fn build_upgrade_replay_contract(
    source_tree_digest: Sha256Digest,
    handoff: &AuthorizedUpgradeHandoff,
    migrations: &[AuthorizedPolicyMigration],
    policy_tracker: &PolicyMigrationTracker,
    now_unix_s: u64,
    clock: &VerifiedClockWindow,
    clock_tracker: &ClockEpochTracker,
    authority_epoch: &AuthorityEpochVector,
    authority_epoch_tracker: &AuthorityEpochTracker,
    recovery_key_set: &RecoveryKeySet,
    recovery_tracker: &RecoveryActivationTracker,
    compacted_evidence: &CompactedEvidence,
    compaction_policy: &EvidenceCompactionPolicy,
    compaction_tracker: &EvidenceCompactionTracker,
    upgrade_tracker: &UpgradeHandoffTracker,
    upgrade_state: &FabricationUpgradeState,
) -> Result<UpgradeReplayContract, UpgradeReplayError> {
    if source_tree_digest.0 == [0; 32] {
        return Err(UpgradeReplayError::ZeroSourceTree);
    }
    let policy_migration_set_digest = digest_policy_migration_set(migrations)?;
    let policy_migration_tracker_digest =
        digest_policy_migration_tracker(policy_tracker, now_unix_s)
            .map_err(|error| UpgradeReplayError::Evidence(format!("{error:?}")))?;
    let clock_tracker_digest = digest_clock_epoch_tracker(clock_tracker)
        .map_err(|error| UpgradeReplayError::Evidence(format!("{error:?}")))?;
    let authority_epoch_digest = digest_authority_epoch(authority_epoch)
        .map_err(|error| UpgradeReplayError::Evidence(format!("{error:?}")))?;
    let authority_epoch_tracker_digest = digest_authority_epoch_tracker(authority_epoch_tracker)
        .map_err(|error| UpgradeReplayError::Evidence(format!("{error:?}")))?;
    let recovery_key_set_digest = digest_recovery_key_set(recovery_key_set)
        .map_err(|error| UpgradeReplayError::Evidence(format!("{error:?}")))?;
    let recovery_tracker_digest = digest_recovery_activation_tracker(recovery_tracker)
        .map_err(|error| UpgradeReplayError::Evidence(format!("{error:?}")))?;
    let evidence_compaction_digest =
        digest_compacted_evidence(compacted_evidence, compaction_policy)
            .map_err(|error| UpgradeReplayError::Evidence(format!("{error:?}")))?;
    let evidence_compaction_tracker_digest = digest_evidence_compaction_tracker(compaction_tracker)
        .map_err(|error| UpgradeReplayError::Evidence(format!("{error:?}")))?;
    let upgrade_tracker_digest = digest_upgrade_tracker(upgrade_tracker, handoff)
        .map_err(|error| UpgradeReplayError::Evidence(format!("{error:?}")))?;
    let upgrade_state_digest = digest_upgrade_state(upgrade_state)
        .map_err(|error| UpgradeReplayError::Evidence(format!("{error:?}")))?;
    let contract = UpgradeReplayContract {
        schema_version: UPGRADE_REPLAY_SCHEMA.into(),
        source_tree_digest,
        handoff_digest: handoff.plan_digest,
        policy_migration_set_digest,
        policy_migration_tracker_digest,
        clock_evidence_digest: clock.evidence_digest,
        clock_tracker_digest,
        authority_epoch_digest,
        authority_epoch_tracker_digest,
        recovery_key_set_digest,
        recovery_tracker_digest,
        evidence_compaction_digest,
        evidence_compaction_tracker_digest,
        upgrade_tracker_digest,
        upgrade_state_digest,
    };
    verify_internal_bindings(&contract, handoff, clock, authority_epoch, upgrade_state)?;
    Ok(contract)
}

#[allow(clippy::too_many_arguments)]
pub fn verify_upgrade_replay_contract(
    contract: &UpgradeReplayContract,
    source_tree_digest: Sha256Digest,
    handoff: &AuthorizedUpgradeHandoff,
    migrations: &[AuthorizedPolicyMigration],
    policy_tracker: &PolicyMigrationTracker,
    now_unix_s: u64,
    clock: &VerifiedClockWindow,
    clock_tracker: &ClockEpochTracker,
    authority_epoch: &AuthorityEpochVector,
    authority_epoch_tracker: &AuthorityEpochTracker,
    recovery_key_set: &RecoveryKeySet,
    recovery_tracker: &RecoveryActivationTracker,
    compacted_evidence: &CompactedEvidence,
    compaction_policy: &EvidenceCompactionPolicy,
    compaction_tracker: &EvidenceCompactionTracker,
    upgrade_tracker: &UpgradeHandoffTracker,
    upgrade_state: &FabricationUpgradeState,
) -> Result<UpgradeReplayVerificationReport, UpgradeReplayError> {
    if contract.schema_version != UPGRADE_REPLAY_SCHEMA {
        return Err(UpgradeReplayError::UnsupportedSchema);
    }
    let rebuilt = build_upgrade_replay_contract(
        source_tree_digest,
        handoff,
        migrations,
        policy_tracker,
        now_unix_s,
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
    )?;
    let mut mismatches = Vec::new();
    compare(
        &mut mismatches,
        contract.source_tree_digest,
        rebuilt.source_tree_digest,
        UpgradeReplayMismatch::SourceTree,
    );
    compare(
        &mut mismatches,
        contract.handoff_digest,
        rebuilt.handoff_digest,
        UpgradeReplayMismatch::Handoff,
    );
    compare(
        &mut mismatches,
        contract.policy_migration_set_digest,
        rebuilt.policy_migration_set_digest,
        UpgradeReplayMismatch::PolicyMigrationSet,
    );
    compare(
        &mut mismatches,
        contract.policy_migration_tracker_digest,
        rebuilt.policy_migration_tracker_digest,
        UpgradeReplayMismatch::PolicyMigrationTracker,
    );
    compare(
        &mut mismatches,
        contract.clock_evidence_digest,
        rebuilt.clock_evidence_digest,
        UpgradeReplayMismatch::ClockEvidence,
    );
    compare(
        &mut mismatches,
        contract.clock_tracker_digest,
        rebuilt.clock_tracker_digest,
        UpgradeReplayMismatch::ClockTracker,
    );
    compare(
        &mut mismatches,
        contract.authority_epoch_digest,
        rebuilt.authority_epoch_digest,
        UpgradeReplayMismatch::AuthorityEpoch,
    );
    compare(
        &mut mismatches,
        contract.authority_epoch_tracker_digest,
        rebuilt.authority_epoch_tracker_digest,
        UpgradeReplayMismatch::AuthorityEpochTracker,
    );
    compare(
        &mut mismatches,
        contract.recovery_key_set_digest,
        rebuilt.recovery_key_set_digest,
        UpgradeReplayMismatch::RecoveryKeySet,
    );
    compare(
        &mut mismatches,
        contract.recovery_tracker_digest,
        rebuilt.recovery_tracker_digest,
        UpgradeReplayMismatch::RecoveryTracker,
    );
    compare(
        &mut mismatches,
        contract.evidence_compaction_digest,
        rebuilt.evidence_compaction_digest,
        UpgradeReplayMismatch::EvidenceCompaction,
    );
    compare(
        &mut mismatches,
        contract.evidence_compaction_tracker_digest,
        rebuilt.evidence_compaction_tracker_digest,
        UpgradeReplayMismatch::EvidenceCompactionTracker,
    );
    compare(
        &mut mismatches,
        contract.upgrade_tracker_digest,
        rebuilt.upgrade_tracker_digest,
        UpgradeReplayMismatch::UpgradeTracker,
    );
    compare(
        &mut mismatches,
        contract.upgrade_state_digest,
        rebuilt.upgrade_state_digest,
        UpgradeReplayMismatch::UpgradeState,
    );
    Ok(UpgradeReplayVerificationReport { mismatches })
}

pub fn digest_upgrade_replay_contract(
    contract: &UpgradeReplayContract,
) -> Result<Sha256Digest, UpgradeReplayError> {
    if contract.schema_version != UPGRADE_REPLAY_SCHEMA {
        return Err(UpgradeReplayError::UnsupportedSchema);
    }
    let bytes = serde_json::to_vec(contract)
        .map_err(|error| UpgradeReplayError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.upgrade-replay-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn digest_policy_migration_set(
    migrations: &[AuthorizedPolicyMigration],
) -> Result<Sha256Digest, UpgradeReplayError> {
    let mut digests = migrations
        .iter()
        .map(|migration| migration.plan_digest)
        .collect::<Vec<_>>();
    digests.sort();
    if digests.windows(2).any(|window| window[0] == window[1]) {
        return Err(UpgradeReplayError::Evidence(
            "duplicate policy migration".into(),
        ));
    }
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.policy-migration-set.v1\0");
    hasher.update(&(digests.len() as u64).to_le_bytes());
    for digest in digests {
        hasher.update(&digest.0);
    }
    Ok(hasher.finalize())
}

fn verify_internal_bindings(
    contract: &UpgradeReplayContract,
    handoff: &AuthorizedUpgradeHandoff,
    clock: &VerifiedClockWindow,
    authority_epoch: &AuthorityEpochVector,
    upgrade_state: &FabricationUpgradeState,
) -> Result<(), UpgradeReplayError> {
    let mut mismatches = Vec::new();
    if handoff.plan.clock_evidence_digest != clock.evidence_digest {
        mismatches.push(UpgradeReplayMismatch::StateEvidence("handoff-clock"));
    }
    if handoff.plan.recovery_key_set_digest != contract.recovery_key_set_digest {
        mismatches.push(UpgradeReplayMismatch::StateEvidence(
            "handoff-recovery-key-set",
        ));
    }
    if handoff.plan.evidence_checkpoint_digest != contract.evidence_compaction_digest {
        mismatches.push(UpgradeReplayMismatch::StateEvidence(
            "handoff-evidence-checkpoint",
        ));
    }
    if handoff.plan.successor.authority_epoch != *authority_epoch {
        mismatches.push(UpgradeReplayMismatch::StateEvidence(
            "successor-authority-epoch",
        ));
    }
    if upgrade_state.evidence.handoff_digest != handoff.plan_digest {
        mismatches.push(UpgradeReplayMismatch::StateEvidence(
            "upgrade-state-handoff",
        ));
    }
    if upgrade_state.evidence.upgrade_tracker_digest != contract.upgrade_tracker_digest {
        mismatches.push(UpgradeReplayMismatch::StateEvidence(
            "upgrade-state-tracker",
        ));
    }
    if upgrade_state.evidence.policy_migration_set_digest != contract.policy_migration_set_digest {
        mismatches.push(UpgradeReplayMismatch::StateEvidence(
            "upgrade-state-migrations",
        ));
    }
    if upgrade_state.evidence.clock_tracker_digest != contract.clock_tracker_digest {
        mismatches.push(UpgradeReplayMismatch::StateEvidence(
            "upgrade-state-clock-tracker",
        ));
    }
    if upgrade_state.evidence.authority_epoch_tracker_digest
        != contract.authority_epoch_tracker_digest
    {
        mismatches.push(UpgradeReplayMismatch::StateEvidence(
            "upgrade-state-epoch-tracker",
        ));
    }
    if upgrade_state.evidence.recovery_tracker_digest != contract.recovery_tracker_digest {
        mismatches.push(UpgradeReplayMismatch::StateEvidence(
            "upgrade-state-recovery-tracker",
        ));
    }
    if upgrade_state.evidence.evidence_compaction_tracker_digest
        != contract.evidence_compaction_tracker_digest
    {
        mismatches.push(UpgradeReplayMismatch::StateEvidence(
            "upgrade-state-compaction-tracker",
        ));
    }
    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(UpgradeReplayError::Evidence(format!(
            "internal binding mismatches: {mismatches:?}"
        )))
    }
}

fn compare(
    mismatches: &mut Vec<UpgradeReplayMismatch>,
    expected: Sha256Digest,
    actual: Sha256Digest,
    mismatch: UpgradeReplayMismatch,
) {
    if expected != actual {
        mismatches.push(mismatch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;

    #[test]
    fn migration_set_digest_is_order_independent_but_duplicate_sensitive() {
        let migration = |label: &[u8]| {
            AuthorizedPolicyMigration {
            plan: serde_json::from_str(r#"{
                "schema_version":"symthaea.fabrication.policy-migration.v1",
                "predecessor":{"schema_version":"symthaea.fabrication.policy-binding.v1","domain":"d","policy_version":"1","policy_digest":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],"invariants":[]},
                "successor":{"schema_version":"symthaea.fabrication.policy-binding.v1","domain":"d","policy_version":"2","policy_digest":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],"invariants":[]},
                "activates_at_unix_s":1,"rollback_deadline_unix_s":2,"rationale":"x","migrations":[]
            }"#).unwrap(),
            plan_digest: sha256(label),
            ceremony_digest: sha256(b"c"),
            trust_snapshot_digest: sha256(b"t"),
        }
        };
        let a = migration(b"a");
        let b = migration(b"b");
        assert_eq!(
            digest_policy_migration_set(&[a.clone(), b.clone()]).unwrap(),
            digest_policy_migration_set(&[b.clone(), a.clone()]).unwrap()
        );
        assert!(digest_policy_migration_set(&[a.clone(), a]).is_err());
    }
}
