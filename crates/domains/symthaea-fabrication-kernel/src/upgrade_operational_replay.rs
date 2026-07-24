// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic replay contract for post-upgrade operational authority.

use crate::automatic_rollback::{AutomaticRollbackTrigger, digest_automatic_rollback_trigger};
use crate::clock_continuity::{VerifiedClockContinuity, digest_clock_continuity};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::evidence_retention::{EvidenceRetentionPolicy, digest_evidence_retention_policy};
use crate::hardware_reauthorization_tracker::{
    HardwareReauthorizationTracker, digest_hardware_reauthorization_tracker,
};
use crate::key_continuity::{VerifiedKeyContinuity, digest_key_continuity};
use crate::upgrade_operational_state::{
    FabricationUpgradeOperationalState, digest_upgrade_operational_state,
};
use crate::upgrade_probation::{UpgradeProbationEvidence, digest_upgrade_probation_evidence};
use crate::upgrade_probation_tracker::{UpgradeProbationTracker, digest_upgrade_probation_tracker};
use crate::upgrade_state::{FabricationUpgradeState, digest_upgrade_state};
use serde::{Deserialize, Serialize};

pub const UPGRADE_OPERATIONAL_REPLAY_SCHEMA: &str =
    "symthaea.fabrication.upgrade-operational-replay.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradeOperationalReplayContract {
    pub schema_version: String,
    pub source_tree_digest: Sha256Digest,
    pub handoff_digest: Sha256Digest,
    pub upgrade_state_digest: Sha256Digest,
    pub probation_evidence_digest: Option<Sha256Digest>,
    pub probation_tracker_digest: Sha256Digest,
    pub hardware_reauthorization_tracker_digest: Sha256Digest,
    pub retention_policy_digest: Sha256Digest,
    pub key_continuity_digest: Sha256Digest,
    pub clock_continuity_digest: Sha256Digest,
    pub automatic_rollback_digest: Option<Sha256Digest>,
    pub operational_state_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpgradeOperationalReplayMismatch {
    SourceTree,
    Handoff,
    UpgradeState,
    ProbationEvidence,
    ProbationTracker,
    HardwareReauthorizationTracker,
    RetentionPolicy,
    KeyContinuity,
    ClockContinuity,
    AutomaticRollback,
    OperationalState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpgradeOperationalReplayVerificationReport {
    pub mismatches: Vec<UpgradeOperationalReplayMismatch>,
}

impl UpgradeOperationalReplayVerificationReport {
    pub fn exact(&self) -> bool {
        self.mismatches.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpgradeOperationalReplayError {
    UnsupportedSchema,
    ZeroDigest(&'static str),
    Evidence(String),
    StateBinding(&'static str),
    Encoding(String),
}

#[allow(clippy::too_many_arguments)]
pub fn build_upgrade_operational_replay_contract(
    source_tree_digest: Sha256Digest,
    handoff_digest: Sha256Digest,
    upgrade_state: &FabricationUpgradeState,
    probation_evidence: Option<&UpgradeProbationEvidence>,
    probation_tracker: &UpgradeProbationTracker,
    hardware_tracker: &HardwareReauthorizationTracker,
    retention_policy: &EvidenceRetentionPolicy,
    key_continuity: &VerifiedKeyContinuity,
    clock_continuity: &VerifiedClockContinuity,
    automatic_rollback: Option<&AutomaticRollbackTrigger>,
    operational_state: &FabricationUpgradeOperationalState,
) -> Result<UpgradeOperationalReplayContract, UpgradeOperationalReplayError> {
    if source_tree_digest.0 == [0; 32] {
        return Err(UpgradeOperationalReplayError::ZeroDigest(
            "source_tree_digest",
        ));
    }
    if handoff_digest.0 == [0; 32] {
        return Err(UpgradeOperationalReplayError::ZeroDigest("handoff_digest"));
    }
    let contract = UpgradeOperationalReplayContract {
        schema_version: UPGRADE_OPERATIONAL_REPLAY_SCHEMA.into(),
        source_tree_digest,
        handoff_digest,
        upgrade_state_digest: digest_upgrade_state(upgrade_state)
            .map_err(|error| UpgradeOperationalReplayError::Evidence(format!("{error:?}")))?,
        probation_evidence_digest: probation_evidence
            .map(digest_upgrade_probation_evidence)
            .transpose()
            .map_err(|error| UpgradeOperationalReplayError::Evidence(format!("{error:?}")))?,
        probation_tracker_digest: digest_upgrade_probation_tracker(probation_tracker)
            .map_err(|error| UpgradeOperationalReplayError::Evidence(format!("{error:?}")))?,
        hardware_reauthorization_tracker_digest: digest_hardware_reauthorization_tracker(
            hardware_tracker,
        )
        .map_err(|error| UpgradeOperationalReplayError::Evidence(format!("{error:?}")))?,
        retention_policy_digest: digest_evidence_retention_policy(retention_policy)
            .map_err(|error| UpgradeOperationalReplayError::Evidence(format!("{error:?}")))?,
        key_continuity_digest: digest_key_continuity(key_continuity)
            .map_err(|error| UpgradeOperationalReplayError::Evidence(format!("{error:?}")))?,
        clock_continuity_digest: digest_clock_continuity(clock_continuity)
            .map_err(|error| UpgradeOperationalReplayError::Evidence(format!("{error:?}")))?,
        automatic_rollback_digest: automatic_rollback
            .map(digest_automatic_rollback_trigger)
            .transpose()
            .map_err(|error| UpgradeOperationalReplayError::Evidence(format!("{error:?}")))?,
        operational_state_digest: digest_upgrade_operational_state(operational_state)
            .map_err(|error| UpgradeOperationalReplayError::Evidence(format!("{error:?}")))?,
    };
    verify_state_bindings(
        &contract,
        probation_evidence,
        probation_tracker,
        hardware_tracker,
        retention_policy,
        key_continuity,
        clock_continuity,
        operational_state,
    )?;
    Ok(contract)
}

#[allow(clippy::too_many_arguments)]
pub fn verify_upgrade_operational_replay_contract(
    contract: &UpgradeOperationalReplayContract,
    source_tree_digest: Sha256Digest,
    handoff_digest: Sha256Digest,
    upgrade_state: &FabricationUpgradeState,
    probation_evidence: Option<&UpgradeProbationEvidence>,
    probation_tracker: &UpgradeProbationTracker,
    hardware_tracker: &HardwareReauthorizationTracker,
    retention_policy: &EvidenceRetentionPolicy,
    key_continuity: &VerifiedKeyContinuity,
    clock_continuity: &VerifiedClockContinuity,
    automatic_rollback: Option<&AutomaticRollbackTrigger>,
    operational_state: &FabricationUpgradeOperationalState,
) -> Result<UpgradeOperationalReplayVerificationReport, UpgradeOperationalReplayError> {
    if contract.schema_version != UPGRADE_OPERATIONAL_REPLAY_SCHEMA {
        return Err(UpgradeOperationalReplayError::UnsupportedSchema);
    }
    let rebuilt = build_upgrade_operational_replay_contract(
        source_tree_digest,
        handoff_digest,
        upgrade_state,
        probation_evidence,
        probation_tracker,
        hardware_tracker,
        retention_policy,
        key_continuity,
        clock_continuity,
        automatic_rollback,
        operational_state,
    )?;
    let mut mismatches = Vec::new();
    compare(
        &mut mismatches,
        contract.source_tree_digest,
        rebuilt.source_tree_digest,
        UpgradeOperationalReplayMismatch::SourceTree,
    );
    compare(
        &mut mismatches,
        contract.handoff_digest,
        rebuilt.handoff_digest,
        UpgradeOperationalReplayMismatch::Handoff,
    );
    compare(
        &mut mismatches,
        contract.upgrade_state_digest,
        rebuilt.upgrade_state_digest,
        UpgradeOperationalReplayMismatch::UpgradeState,
    );
    compare_option(
        &mut mismatches,
        contract.probation_evidence_digest,
        rebuilt.probation_evidence_digest,
        UpgradeOperationalReplayMismatch::ProbationEvidence,
    );
    compare(
        &mut mismatches,
        contract.probation_tracker_digest,
        rebuilt.probation_tracker_digest,
        UpgradeOperationalReplayMismatch::ProbationTracker,
    );
    compare(
        &mut mismatches,
        contract.hardware_reauthorization_tracker_digest,
        rebuilt.hardware_reauthorization_tracker_digest,
        UpgradeOperationalReplayMismatch::HardwareReauthorizationTracker,
    );
    compare(
        &mut mismatches,
        contract.retention_policy_digest,
        rebuilt.retention_policy_digest,
        UpgradeOperationalReplayMismatch::RetentionPolicy,
    );
    compare(
        &mut mismatches,
        contract.key_continuity_digest,
        rebuilt.key_continuity_digest,
        UpgradeOperationalReplayMismatch::KeyContinuity,
    );
    compare(
        &mut mismatches,
        contract.clock_continuity_digest,
        rebuilt.clock_continuity_digest,
        UpgradeOperationalReplayMismatch::ClockContinuity,
    );
    compare_option(
        &mut mismatches,
        contract.automatic_rollback_digest,
        rebuilt.automatic_rollback_digest,
        UpgradeOperationalReplayMismatch::AutomaticRollback,
    );
    compare(
        &mut mismatches,
        contract.operational_state_digest,
        rebuilt.operational_state_digest,
        UpgradeOperationalReplayMismatch::OperationalState,
    );
    Ok(UpgradeOperationalReplayVerificationReport { mismatches })
}

pub fn digest_upgrade_operational_replay_contract(
    contract: &UpgradeOperationalReplayContract,
) -> Result<Sha256Digest, UpgradeOperationalReplayError> {
    if contract.schema_version != UPGRADE_OPERATIONAL_REPLAY_SCHEMA {
        return Err(UpgradeOperationalReplayError::UnsupportedSchema);
    }
    let bytes = serde_json::to_vec(contract)
        .map_err(|error| UpgradeOperationalReplayError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.upgrade-operational-replay-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[allow(clippy::too_many_arguments)]
fn verify_state_bindings(
    contract: &UpgradeOperationalReplayContract,
    probation_evidence: Option<&UpgradeProbationEvidence>,
    probation_tracker: &UpgradeProbationTracker,
    hardware_tracker: &HardwareReauthorizationTracker,
    retention_policy: &EvidenceRetentionPolicy,
    key_continuity: &VerifiedKeyContinuity,
    clock_continuity: &VerifiedClockContinuity,
    operational_state: &FabricationUpgradeOperationalState,
) -> Result<(), UpgradeOperationalReplayError> {
    if operational_state.handoff_digest != contract.handoff_digest {
        return Err(UpgradeOperationalReplayError::StateBinding("handoff"));
    }
    let evidence = &operational_state.evidence;
    if evidence.upgrade_state_digest != contract.upgrade_state_digest {
        return Err(UpgradeOperationalReplayError::StateBinding("upgrade-state"));
    }
    if evidence.probation_tracker_digest != contract.probation_tracker_digest
        || evidence.hardware_reauthorization_tracker_digest
            != contract.hardware_reauthorization_tracker_digest
        || evidence.retention_policy_digest != contract.retention_policy_digest
        || evidence.key_continuity_digest != contract.key_continuity_digest
        || evidence.clock_continuity_digest != contract.clock_continuity_digest
        || evidence.probation_clearance_digest != contract.probation_evidence_digest
        || evidence.automatic_rollback_digest != contract.automatic_rollback_digest
    {
        return Err(UpgradeOperationalReplayError::StateBinding(
            "evidence-digest",
        ));
    }
    if evidence.probation_sequence != probation_evidence.map(|value| value.probation_sequence)
        || evidence.probation_sequence != probation_tracker.latest_sequence
        || evidence.reauthorized_machine_count != hardware_tracker.records.len() as u64
        || evidence.retention_policy_sequence != retention_policy.sequence
        || evidence.key_snapshot_sequence != key_continuity.successor_snapshot_sequence
        || evidence.clock_epoch != clock_continuity.successor_epoch
    {
        return Err(UpgradeOperationalReplayError::StateBinding(
            "evidence-sequence",
        ));
    }
    Ok(())
}

fn compare(
    mismatches: &mut Vec<UpgradeOperationalReplayMismatch>,
    actual: Sha256Digest,
    expected: Sha256Digest,
    mismatch: UpgradeOperationalReplayMismatch,
) {
    if actual != expected {
        mismatches.push(mismatch);
    }
}

fn compare_option(
    mismatches: &mut Vec<UpgradeOperationalReplayMismatch>,
    actual: Option<Sha256Digest>,
    expected: Option<Sha256Digest>,
    mismatch: UpgradeOperationalReplayMismatch,
) {
    if actual != expected {
        mismatches.push(mismatch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;

    #[test]
    fn contract_digest_is_source_sensitive() {
        let first = UpgradeOperationalReplayContract {
            schema_version: UPGRADE_OPERATIONAL_REPLAY_SCHEMA.into(),
            source_tree_digest: sha256(b"one"),
            handoff_digest: sha256(b"handoff"),
            upgrade_state_digest: sha256(b"upgrade"),
            probation_evidence_digest: None,
            probation_tracker_digest: sha256(b"probation"),
            hardware_reauthorization_tracker_digest: sha256(b"hardware"),
            retention_policy_digest: sha256(b"retention"),
            key_continuity_digest: sha256(b"keys"),
            clock_continuity_digest: sha256(b"clock"),
            automatic_rollback_digest: None,
            operational_state_digest: sha256(b"state"),
        };
        let mut second = first.clone();
        second.source_tree_digest = sha256(b"two");
        assert_ne!(
            digest_upgrade_operational_replay_contract(&first).unwrap(),
            digest_upgrade_operational_replay_contract(&second).unwrap()
        );
    }
}
