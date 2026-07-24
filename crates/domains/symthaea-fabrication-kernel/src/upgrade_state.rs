// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hash-linked durable state for upgrade authority.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::upgrade_tracker::UpgradeStage;
use serde::{Deserialize, Serialize};

pub const UPGRADE_STATE_SCHEMA: &str = "symthaea.fabrication.upgrade-state.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradeEvidenceDigests {
    pub handoff_digest: Sha256Digest,
    pub upgrade_tracker_digest: Sha256Digest,
    pub policy_migration_set_digest: Sha256Digest,
    pub clock_tracker_digest: Sha256Digest,
    pub authority_epoch_tracker_digest: Sha256Digest,
    pub recovery_tracker_digest: Sha256Digest,
    pub evidence_compaction_tracker_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FabricationUpgradeState {
    pub schema_version: String,
    pub generation: u64,
    pub committed_at_unix_ms: u64,
    pub previous_state_digest: Option<Sha256Digest>,
    pub handoff_sequence: u64,
    pub active_stage: UpgradeStage,
    pub evidence: UpgradeEvidenceDigests,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpgradeStateError {
    UnsupportedSchema,
    GenerationZero,
    HandoffSequenceZero,
    ZeroEvidenceDigest(&'static str),
    InvalidGenesis,
    GenerationOverflow,
    GenerationDiscontinuity { expected: u64, actual: u64 },
    PreviousDigestMismatch,
    CommitTimeRegression,
    HandoffSequenceRollback,
    HandoffSequenceSkip,
    HandoffSubstitution,
    InvalidStageProgression,
    HistoricalEvidenceRegression(&'static str),
    Encoding(String),
}

impl FabricationUpgradeState {
    pub fn genesis(
        committed_at_unix_ms: u64,
        evidence: UpgradeEvidenceDigests,
    ) -> Result<Self, UpgradeStateError> {
        let state = Self {
            schema_version: UPGRADE_STATE_SCHEMA.into(),
            generation: 1,
            committed_at_unix_ms,
            previous_state_digest: None,
            handoff_sequence: 1,
            active_stage: UpgradeStage::Prepared,
            evidence,
        };
        state.validate_shape()?;
        Ok(state)
    }

    pub fn successor(
        previous: &Self,
        committed_at_unix_ms: u64,
        handoff_sequence: u64,
        active_stage: UpgradeStage,
        evidence: UpgradeEvidenceDigests,
    ) -> Result<Self, UpgradeStateError> {
        let generation = previous
            .generation
            .checked_add(1)
            .ok_or(UpgradeStateError::GenerationOverflow)?;
        let state = Self {
            schema_version: UPGRADE_STATE_SCHEMA.into(),
            generation,
            committed_at_unix_ms,
            previous_state_digest: Some(digest_upgrade_state(previous)?),
            handoff_sequence,
            active_stage,
            evidence,
        };
        verify_upgrade_state_successor(previous, &state)?;
        Ok(state)
    }

    pub fn validate_shape(&self) -> Result<(), UpgradeStateError> {
        if self.schema_version != UPGRADE_STATE_SCHEMA {
            return Err(UpgradeStateError::UnsupportedSchema);
        }
        if self.generation == 0 {
            return Err(UpgradeStateError::GenerationZero);
        }
        if self.handoff_sequence == 0 {
            return Err(UpgradeStateError::HandoffSequenceZero);
        }
        for (name, digest) in evidence_pairs(&self.evidence) {
            if digest.0 == [0; 32] {
                return Err(UpgradeStateError::ZeroEvidenceDigest(name));
            }
        }
        if self.generation == 1 {
            if self.previous_state_digest.is_some()
                || self.handoff_sequence != 1
                || self.active_stage != UpgradeStage::Prepared
            {
                return Err(UpgradeStateError::InvalidGenesis);
            }
        } else if self.previous_state_digest.is_none() {
            return Err(UpgradeStateError::PreviousDigestMismatch);
        }
        Ok(())
    }
}

pub fn verify_upgrade_state_successor(
    previous: &FabricationUpgradeState,
    proposed: &FabricationUpgradeState,
) -> Result<(), UpgradeStateError> {
    previous.validate_shape()?;
    proposed.validate_shape()?;
    let expected_generation = previous
        .generation
        .checked_add(1)
        .ok_or(UpgradeStateError::GenerationOverflow)?;
    if proposed.generation != expected_generation {
        return Err(UpgradeStateError::GenerationDiscontinuity {
            expected: expected_generation,
            actual: proposed.generation,
        });
    }
    if proposed.previous_state_digest != Some(digest_upgrade_state(previous)?) {
        return Err(UpgradeStateError::PreviousDigestMismatch);
    }
    if proposed.committed_at_unix_ms < previous.committed_at_unix_ms {
        return Err(UpgradeStateError::CommitTimeRegression);
    }
    if proposed.handoff_sequence < previous.handoff_sequence {
        return Err(UpgradeStateError::HandoffSequenceRollback);
    }
    if proposed.handoff_sequence > previous.handoff_sequence + 1 {
        return Err(UpgradeStateError::HandoffSequenceSkip);
    }
    if proposed.handoff_sequence == previous.handoff_sequence {
        if proposed.evidence.handoff_digest != previous.evidence.handoff_digest {
            return Err(UpgradeStateError::HandoffSubstitution);
        }
        if !valid_same_handoff_progression(previous.active_stage, proposed.active_stage) {
            return Err(UpgradeStateError::InvalidStageProgression);
        }
        for (name, previous_digest, proposed_digest) in monotonic_evidence_pairs(previous, proposed)
        {
            if previous_digest != proposed_digest && proposed.active_stage == previous.active_stage
            {
                return Err(UpgradeStateError::HistoricalEvidenceRegression(name));
            }
        }
    } else {
        if !matches!(
            previous.active_stage,
            UpgradeStage::Finalized | UpgradeStage::RolledBack | UpgradeStage::Failed
        ) || proposed.active_stage != UpgradeStage::Prepared
        {
            return Err(UpgradeStateError::InvalidStageProgression);
        }
    }
    Ok(())
}

pub fn digest_upgrade_state(
    state: &FabricationUpgradeState,
) -> Result<Sha256Digest, UpgradeStateError> {
    state.validate_shape()?;
    let bytes = serde_json::to_vec(state)
        .map_err(|error| UpgradeStateError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.upgrade-state-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn digest_upgrade_evidence_set(
    evidence: &UpgradeEvidenceDigests,
) -> Result<Sha256Digest, UpgradeStateError> {
    for (name, digest) in evidence_pairs(evidence) {
        if digest.0 == [0; 32] {
            return Err(UpgradeStateError::ZeroEvidenceDigest(name));
        }
    }
    let bytes = serde_json::to_vec(evidence)
        .map_err(|error| UpgradeStateError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.upgrade-evidence-set.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

fn valid_same_handoff_progression(previous: UpgradeStage, proposed: UpgradeStage) -> bool {
    previous == proposed
        || matches!(
            (previous, proposed),
            (UpgradeStage::Prepared, UpgradeStage::Activated)
                | (UpgradeStage::Prepared, UpgradeStage::Failed)
                | (UpgradeStage::Activated, UpgradeStage::Finalized)
                | (UpgradeStage::Activated, UpgradeStage::RolledBack)
                | (UpgradeStage::Activated, UpgradeStage::Failed)
        )
}

fn evidence_pairs(evidence: &UpgradeEvidenceDigests) -> [(&'static str, Sha256Digest); 7] {
    [
        ("handoff_digest", evidence.handoff_digest),
        ("upgrade_tracker_digest", evidence.upgrade_tracker_digest),
        (
            "policy_migration_set_digest",
            evidence.policy_migration_set_digest,
        ),
        ("clock_tracker_digest", evidence.clock_tracker_digest),
        (
            "authority_epoch_tracker_digest",
            evidence.authority_epoch_tracker_digest,
        ),
        ("recovery_tracker_digest", evidence.recovery_tracker_digest),
        (
            "evidence_compaction_tracker_digest",
            evidence.evidence_compaction_tracker_digest,
        ),
    ]
}

fn monotonic_evidence_pairs(
    previous: &FabricationUpgradeState,
    proposed: &FabricationUpgradeState,
) -> [(&'static str, Sha256Digest, Sha256Digest); 5] {
    [
        (
            "policy_migration_set_digest",
            previous.evidence.policy_migration_set_digest,
            proposed.evidence.policy_migration_set_digest,
        ),
        (
            "clock_tracker_digest",
            previous.evidence.clock_tracker_digest,
            proposed.evidence.clock_tracker_digest,
        ),
        (
            "authority_epoch_tracker_digest",
            previous.evidence.authority_epoch_tracker_digest,
            proposed.evidence.authority_epoch_tracker_digest,
        ),
        (
            "recovery_tracker_digest",
            previous.evidence.recovery_tracker_digest,
            proposed.evidence.recovery_tracker_digest,
        ),
        (
            "evidence_compaction_tracker_digest",
            previous.evidence.evidence_compaction_tracker_digest,
            proposed.evidence.evidence_compaction_tracker_digest,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;

    fn evidence(label: &[u8]) -> UpgradeEvidenceDigests {
        UpgradeEvidenceDigests {
            handoff_digest: sha256(&[label, b"-handoff"].concat()),
            upgrade_tracker_digest: sha256(&[label, b"-tracker"].concat()),
            policy_migration_set_digest: sha256(b"migration-set"),
            clock_tracker_digest: sha256(b"clock-tracker"),
            authority_epoch_tracker_digest: sha256(b"epoch-tracker"),
            recovery_tracker_digest: sha256(b"recovery-tracker"),
            evidence_compaction_tracker_digest: sha256(b"compaction-tracker"),
        }
    }

    #[test]
    fn same_handoff_cannot_be_substituted() {
        let first = FabricationUpgradeState::genesis(100, evidence(b"one")).unwrap();
        let result = FabricationUpgradeState::successor(
            &first,
            200,
            1,
            UpgradeStage::Activated,
            evidence(b"two"),
        );
        assert_eq!(result.unwrap_err(), UpgradeStateError::HandoffSubstitution);
    }

    #[test]
    fn new_handoff_requires_terminal_predecessor() {
        let first = FabricationUpgradeState::genesis(100, evidence(b"one")).unwrap();
        let result = FabricationUpgradeState::successor(
            &first,
            200,
            2,
            UpgradeStage::Prepared,
            evidence(b"two"),
        );
        assert_eq!(
            result.unwrap_err(),
            UpgradeStateError::InvalidStageProgression
        );
    }
}
