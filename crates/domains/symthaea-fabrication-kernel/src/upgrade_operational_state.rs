// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hash-linked durable state for post-upgrade operational authority.

use crate::crypto_digest::{Sha256, Sha256Digest};
use serde::{Deserialize, Serialize};

pub const UPGRADE_OPERATIONAL_STATE_SCHEMA: &str =
    "symthaea.fabrication.upgrade-operational-state.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradeOperationalEvidenceDigests {
    pub upgrade_state_digest: Sha256Digest,
    pub probation_tracker_digest: Sha256Digest,
    pub hardware_reauthorization_tracker_digest: Sha256Digest,
    pub retention_policy_digest: Sha256Digest,
    pub key_continuity_digest: Sha256Digest,
    pub clock_continuity_digest: Sha256Digest,
    pub probation_clearance_digest: Option<Sha256Digest>,
    pub automatic_rollback_digest: Option<Sha256Digest>,
    pub probation_sequence: Option<u64>,
    pub reauthorized_machine_count: u64,
    pub retention_policy_sequence: u64,
    pub key_snapshot_sequence: u64,
    pub clock_epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FabricationUpgradeOperationalState {
    pub schema_version: String,
    pub generation: u64,
    pub committed_at_unix_ms: u64,
    pub previous_state_digest: Option<Sha256Digest>,
    pub handoff_digest: Sha256Digest,
    pub evidence: UpgradeOperationalEvidenceDigests,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpgradeOperationalStateError {
    UnsupportedSchema,
    GenerationZero,
    GenerationOverflow,
    ZeroDigest(&'static str),
    InvalidEvidenceState,
    InvalidGenesis,
    GenerationDiscontinuity { expected: u64, actual: u64 },
    PreviousDigestMismatch,
    CommitTimeRegression,
    HandoffSubstitution,
    ProbationSequenceRollback,
    ProbationSequenceCollision,
    MachineCountRollback,
    MachineCountCollision,
    RetentionPolicyRollback,
    RetentionPolicyCollision,
    KeySequenceRollback,
    KeySequenceCollision,
    ClockEpochRollback,
    ClockEpochCollision,
    ProbationClearanceRemoved,
    ProbationClearanceSubstitution,
    RollbackRemoved,
    RollbackSubstitution,
    ClearanceAfterRollback,
    Encoding(String),
}

impl FabricationUpgradeOperationalState {
    pub fn genesis(
        committed_at_unix_ms: u64,
        handoff_digest: Sha256Digest,
        evidence: UpgradeOperationalEvidenceDigests,
    ) -> Result<Self, UpgradeOperationalStateError> {
        let state = Self {
            schema_version: UPGRADE_OPERATIONAL_STATE_SCHEMA.into(),
            generation: 1,
            committed_at_unix_ms,
            previous_state_digest: None,
            handoff_digest,
            evidence,
        };
        state.validate_shape()?;
        Ok(state)
    }

    pub fn successor(
        previous: &Self,
        committed_at_unix_ms: u64,
        evidence: UpgradeOperationalEvidenceDigests,
    ) -> Result<Self, UpgradeOperationalStateError> {
        let generation = previous
            .generation
            .checked_add(1)
            .ok_or(UpgradeOperationalStateError::GenerationOverflow)?;
        let proposed = Self {
            schema_version: UPGRADE_OPERATIONAL_STATE_SCHEMA.into(),
            generation,
            committed_at_unix_ms,
            previous_state_digest: Some(digest_upgrade_operational_state(previous)?),
            handoff_digest: previous.handoff_digest,
            evidence,
        };
        verify_upgrade_operational_state_successor(previous, &proposed)?;
        Ok(proposed)
    }

    pub fn validate_shape(&self) -> Result<(), UpgradeOperationalStateError> {
        if self.schema_version != UPGRADE_OPERATIONAL_STATE_SCHEMA {
            return Err(UpgradeOperationalStateError::UnsupportedSchema);
        }
        if self.generation == 0 {
            return Err(UpgradeOperationalStateError::GenerationZero);
        }
        if self.handoff_digest.0 == [0; 32] {
            return Err(UpgradeOperationalStateError::ZeroDigest("handoff_digest"));
        }
        for (name, digest) in required_digests(&self.evidence) {
            if digest.0 == [0; 32] {
                return Err(UpgradeOperationalStateError::ZeroDigest(name));
            }
        }
        if self.evidence.retention_policy_sequence == 0
            || self.evidence.key_snapshot_sequence == 0
            || self.evidence.clock_epoch == 0
            || self
                .evidence
                .probation_sequence
                .is_some_and(|sequence| sequence == 0)
            || self
                .evidence
                .probation_clearance_digest
                .is_some_and(|digest| digest.0 == [0; 32])
            || self
                .evidence
                .automatic_rollback_digest
                .is_some_and(|digest| digest.0 == [0; 32])
            || (self.evidence.probation_clearance_digest.is_some()
                != self.evidence.probation_sequence.is_some())
        {
            return Err(UpgradeOperationalStateError::InvalidEvidenceState);
        }
        if self.generation == 1 {
            if self.previous_state_digest.is_some() {
                return Err(UpgradeOperationalStateError::InvalidGenesis);
            }
        } else if self.previous_state_digest.is_none() {
            return Err(UpgradeOperationalStateError::PreviousDigestMismatch);
        }
        Ok(())
    }
}

pub fn verify_upgrade_operational_state_successor(
    previous: &FabricationUpgradeOperationalState,
    proposed: &FabricationUpgradeOperationalState,
) -> Result<(), UpgradeOperationalStateError> {
    previous.validate_shape()?;
    proposed.validate_shape()?;
    let expected = previous
        .generation
        .checked_add(1)
        .ok_or(UpgradeOperationalStateError::GenerationOverflow)?;
    if proposed.generation != expected {
        return Err(UpgradeOperationalStateError::GenerationDiscontinuity {
            expected,
            actual: proposed.generation,
        });
    }
    if proposed.previous_state_digest != Some(digest_upgrade_operational_state(previous)?) {
        return Err(UpgradeOperationalStateError::PreviousDigestMismatch);
    }
    if proposed.committed_at_unix_ms < previous.committed_at_unix_ms {
        return Err(UpgradeOperationalStateError::CommitTimeRegression);
    }
    if proposed.handoff_digest != previous.handoff_digest {
        return Err(UpgradeOperationalStateError::HandoffSubstitution);
    }
    compare_optional_sequence(
        previous.evidence.probation_sequence,
        proposed.evidence.probation_sequence,
        previous.evidence.probation_tracker_digest,
        proposed.evidence.probation_tracker_digest,
    )?;
    compare_monotonic_counter(
        previous.evidence.reauthorized_machine_count,
        proposed.evidence.reauthorized_machine_count,
        previous.evidence.hardware_reauthorization_tracker_digest,
        proposed.evidence.hardware_reauthorization_tracker_digest,
        UpgradeOperationalStateError::MachineCountRollback,
        UpgradeOperationalStateError::MachineCountCollision,
    )?;
    compare_monotonic_counter(
        previous.evidence.retention_policy_sequence,
        proposed.evidence.retention_policy_sequence,
        previous.evidence.retention_policy_digest,
        proposed.evidence.retention_policy_digest,
        UpgradeOperationalStateError::RetentionPolicyRollback,
        UpgradeOperationalStateError::RetentionPolicyCollision,
    )?;
    compare_monotonic_counter(
        previous.evidence.key_snapshot_sequence,
        proposed.evidence.key_snapshot_sequence,
        previous.evidence.key_continuity_digest,
        proposed.evidence.key_continuity_digest,
        UpgradeOperationalStateError::KeySequenceRollback,
        UpgradeOperationalStateError::KeySequenceCollision,
    )?;
    compare_monotonic_counter(
        previous.evidence.clock_epoch,
        proposed.evidence.clock_epoch,
        previous.evidence.clock_continuity_digest,
        proposed.evidence.clock_continuity_digest,
        UpgradeOperationalStateError::ClockEpochRollback,
        UpgradeOperationalStateError::ClockEpochCollision,
    )?;
    preserve_optional_digest(
        previous.evidence.probation_clearance_digest,
        proposed.evidence.probation_clearance_digest,
        UpgradeOperationalStateError::ProbationClearanceRemoved,
        UpgradeOperationalStateError::ProbationClearanceSubstitution,
    )?;
    preserve_optional_digest(
        previous.evidence.automatic_rollback_digest,
        proposed.evidence.automatic_rollback_digest,
        UpgradeOperationalStateError::RollbackRemoved,
        UpgradeOperationalStateError::RollbackSubstitution,
    )?;
    if previous.evidence.automatic_rollback_digest.is_some()
        && previous.evidence.probation_clearance_digest.is_none()
        && proposed.evidence.probation_clearance_digest.is_some()
    {
        return Err(UpgradeOperationalStateError::ClearanceAfterRollback);
    }
    Ok(())
}

pub fn digest_upgrade_operational_state(
    state: &FabricationUpgradeOperationalState,
) -> Result<Sha256Digest, UpgradeOperationalStateError> {
    state.validate_shape()?;
    let bytes = serde_json::to_vec(state)
        .map_err(|error| UpgradeOperationalStateError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.upgrade-operational-state-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

fn required_digests(
    evidence: &UpgradeOperationalEvidenceDigests,
) -> [(&'static str, Sha256Digest); 6] {
    [
        ("upgrade_state_digest", evidence.upgrade_state_digest),
        (
            "probation_tracker_digest",
            evidence.probation_tracker_digest,
        ),
        (
            "hardware_reauthorization_tracker_digest",
            evidence.hardware_reauthorization_tracker_digest,
        ),
        ("retention_policy_digest", evidence.retention_policy_digest),
        ("key_continuity_digest", evidence.key_continuity_digest),
        ("clock_continuity_digest", evidence.clock_continuity_digest),
    ]
}

fn compare_optional_sequence(
    previous: Option<u64>,
    proposed: Option<u64>,
    previous_digest: Sha256Digest,
    proposed_digest: Sha256Digest,
) -> Result<(), UpgradeOperationalStateError> {
    match (previous, proposed) {
        (Some(previous), Some(proposed)) if proposed < previous => {
            Err(UpgradeOperationalStateError::ProbationSequenceRollback)
        }
        (Some(previous), Some(proposed))
            if proposed == previous && previous_digest != proposed_digest =>
        {
            Err(UpgradeOperationalStateError::ProbationSequenceCollision)
        }
        (Some(_), None) => Err(UpgradeOperationalStateError::ProbationSequenceRollback),
        (None, None) if previous_digest != proposed_digest => {
            Err(UpgradeOperationalStateError::ProbationSequenceCollision)
        }
        _ => Ok(()),
    }
}

#[allow(clippy::too_many_arguments)]
fn compare_monotonic_counter(
    previous: u64,
    proposed: u64,
    previous_digest: Sha256Digest,
    proposed_digest: Sha256Digest,
    rollback: UpgradeOperationalStateError,
    collision: UpgradeOperationalStateError,
) -> Result<(), UpgradeOperationalStateError> {
    if proposed < previous {
        return Err(rollback);
    }
    if proposed == previous && proposed_digest != previous_digest {
        return Err(collision);
    }
    Ok(())
}

fn preserve_optional_digest(
    previous: Option<Sha256Digest>,
    proposed: Option<Sha256Digest>,
    removed: UpgradeOperationalStateError,
    substituted: UpgradeOperationalStateError,
) -> Result<(), UpgradeOperationalStateError> {
    match (previous, proposed) {
        (Some(_), None) => Err(removed),
        (Some(previous), Some(proposed)) if previous != proposed => Err(substituted),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;

    fn evidence() -> UpgradeOperationalEvidenceDigests {
        UpgradeOperationalEvidenceDigests {
            upgrade_state_digest: sha256(b"upgrade"),
            probation_tracker_digest: sha256(b"probation-tracker"),
            hardware_reauthorization_tracker_digest: sha256(b"hardware-tracker"),
            retention_policy_digest: sha256(b"retention"),
            key_continuity_digest: sha256(b"keys"),
            clock_continuity_digest: sha256(b"clock"),
            probation_clearance_digest: None,
            automatic_rollback_digest: None,
            probation_sequence: None,
            reauthorized_machine_count: 0,
            retention_policy_sequence: 1,
            key_snapshot_sequence: 2,
            clock_epoch: 2,
        }
    }

    #[test]
    fn same_counter_cannot_substitute_tracker() {
        let first = FabricationUpgradeOperationalState::genesis(10, sha256(b"handoff"), evidence())
            .unwrap();
        let mut changed = evidence();
        changed.hardware_reauthorization_tracker_digest = sha256(b"other");
        assert_eq!(
            FabricationUpgradeOperationalState::successor(&first, 11, changed),
            Err(UpgradeOperationalStateError::MachineCountCollision)
        );
    }

    #[test]
    fn rollback_digest_cannot_be_removed() {
        let mut initial = evidence();
        initial.automatic_rollback_digest = Some(sha256(b"rollback"));
        let first =
            FabricationUpgradeOperationalState::genesis(10, sha256(b"handoff"), initial).unwrap();
        assert_eq!(
            FabricationUpgradeOperationalState::successor(&first, 11, evidence()),
            Err(UpgradeOperationalStateError::RollbackRemoved)
        );
    }
}
