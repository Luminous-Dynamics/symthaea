// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Append-only secure-upgrade lifecycle tracking.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::upgrade_handoff::AuthorizedUpgradeHandoff;
use serde::{Deserialize, Serialize};

pub const UPGRADE_RECORD_SCHEMA: &str = "symthaea.fabrication.upgrade-record.v1";
pub const MAX_UPGRADE_RECORDS: usize = 100_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpgradeStage {
    Prepared,
    Activated,
    Finalized,
    RolledBack,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradeRecord {
    pub schema_version: String,
    pub sequence: u64,
    pub plan_digest: Sha256Digest,
    pub stage: UpgradeStage,
    pub observed_at_unix_ms: u64,
    pub observed_state_digest: Sha256Digest,
    pub evidence_digest: Sha256Digest,
    pub previous_record_digest: Sha256Digest,
    pub record_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradeHandoffTracker {
    pub plan_digest: Sha256Digest,
    pub records: Vec<UpgradeRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpgradeTrackingError {
    CapacityExceeded,
    UnsupportedSchema,
    PlanMismatch,
    InvalidSequence {
        expected: u64,
        actual: u64,
    },
    TimeRegression,
    PreviousDigestMismatch,
    RecordDigestMismatch,
    InvalidTransition {
        previous: Option<UpgradeStage>,
        proposed: UpgradeStage,
    },
    InvalidObservedState,
    ZeroEvidenceDigest,
    BeforeActivationWindow,
    AfterFinalizationDeadline,
    TerminalState,
    Encoding(String),
}

impl UpgradeHandoffTracker {
    pub fn new(handoff: &AuthorizedUpgradeHandoff) -> Self {
        Self {
            plan_digest: handoff.plan_digest,
            records: Vec::new(),
        }
    }

    pub fn validate(&self, handoff: &AuthorizedUpgradeHandoff) -> Result<(), UpgradeTrackingError> {
        if self.plan_digest != handoff.plan_digest {
            return Err(UpgradeTrackingError::PlanMismatch);
        }
        if self.records.len() > MAX_UPGRADE_RECORDS {
            return Err(UpgradeTrackingError::CapacityExceeded);
        }
        let mut previous_digest = empty_upgrade_record_digest();
        let mut previous_stage = None;
        let mut previous_time = None;
        for (index, record) in self.records.iter().enumerate() {
            if record.schema_version != UPGRADE_RECORD_SCHEMA {
                return Err(UpgradeTrackingError::UnsupportedSchema);
            }
            let expected = index as u64 + 1;
            if record.sequence != expected {
                return Err(UpgradeTrackingError::InvalidSequence {
                    expected,
                    actual: record.sequence,
                });
            }
            if record.plan_digest != handoff.plan_digest {
                return Err(UpgradeTrackingError::PlanMismatch);
            }
            if previous_time.is_some_and(|time| record.observed_at_unix_ms < time) {
                return Err(UpgradeTrackingError::TimeRegression);
            }
            if record.previous_record_digest != previous_digest {
                return Err(UpgradeTrackingError::PreviousDigestMismatch);
            }
            validate_transition(previous_stage, record.stage)?;
            validate_stage_evidence(record, handoff)?;
            let expected_digest = digest_upgrade_record_fields(record)?;
            if expected_digest != record.record_digest {
                return Err(UpgradeTrackingError::RecordDigestMismatch);
            }
            previous_digest = record.record_digest;
            previous_stage = Some(record.stage);
            previous_time = Some(record.observed_at_unix_ms);
        }
        Ok(())
    }

    pub fn append(
        &mut self,
        handoff: &AuthorizedUpgradeHandoff,
        stage: UpgradeStage,
        observed_at_unix_ms: u64,
        observed_state_digest: Sha256Digest,
        evidence_digest: Sha256Digest,
    ) -> Result<Sha256Digest, UpgradeTrackingError> {
        self.validate(handoff)?;
        if self.records.len() >= MAX_UPGRADE_RECORDS {
            return Err(UpgradeTrackingError::CapacityExceeded);
        }
        let previous_stage = self.records.last().map(|record| record.stage);
        if previous_stage.is_some_and(|value| {
            matches!(
                value,
                UpgradeStage::Finalized | UpgradeStage::RolledBack | UpgradeStage::Failed
            )
        }) {
            return Err(UpgradeTrackingError::TerminalState);
        }
        validate_transition(previous_stage, stage)?;
        let previous_record_digest = self
            .records
            .last()
            .map_or_else(empty_upgrade_record_digest, |record| record.record_digest);
        let mut record = UpgradeRecord {
            schema_version: UPGRADE_RECORD_SCHEMA.into(),
            sequence: self.records.len() as u64 + 1,
            plan_digest: handoff.plan_digest,
            stage,
            observed_at_unix_ms,
            observed_state_digest,
            evidence_digest,
            previous_record_digest,
            record_digest: Sha256Digest([0; 32]),
        };
        validate_stage_evidence(&record, handoff)?;
        record.record_digest = digest_upgrade_record_fields(&record)?;
        let digest = record.record_digest;
        self.records.push(record);
        Ok(digest)
    }

    pub fn latest_stage(&self) -> Option<UpgradeStage> {
        self.records.last().map(|record| record.stage)
    }

    pub fn head(&self) -> Sha256Digest {
        self.records
            .last()
            .map_or_else(empty_upgrade_record_digest, |record| record.record_digest)
    }
}

pub fn digest_upgrade_tracker(
    tracker: &UpgradeHandoffTracker,
    handoff: &AuthorizedUpgradeHandoff,
) -> Result<Sha256Digest, UpgradeTrackingError> {
    tracker.validate(handoff)?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.upgrade-tracker-digest.v1\0");
    hasher.update(&tracker.plan_digest.0);
    hasher.update(&(tracker.records.len() as u64).to_le_bytes());
    hasher.update(&tracker.head().0);
    Ok(hasher.finalize())
}

pub fn empty_upgrade_record_digest() -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.upgrade-record-empty.v1\0");
    hasher.finalize()
}

fn validate_transition(
    previous: Option<UpgradeStage>,
    proposed: UpgradeStage,
) -> Result<(), UpgradeTrackingError> {
    let valid = matches!(
        (previous, proposed),
        (None, UpgradeStage::Prepared)
            | (Some(UpgradeStage::Prepared), UpgradeStage::Activated)
            | (Some(UpgradeStage::Prepared), UpgradeStage::Failed)
            | (Some(UpgradeStage::Activated), UpgradeStage::Finalized)
            | (Some(UpgradeStage::Activated), UpgradeStage::RolledBack)
            | (Some(UpgradeStage::Activated), UpgradeStage::Failed)
    );
    if valid {
        Ok(())
    } else {
        Err(UpgradeTrackingError::InvalidTransition { previous, proposed })
    }
}

fn validate_stage_evidence(
    record: &UpgradeRecord,
    handoff: &AuthorizedUpgradeHandoff,
) -> Result<(), UpgradeTrackingError> {
    if record.evidence_digest.0 == [0; 32] {
        return Err(UpgradeTrackingError::ZeroEvidenceDigest);
    }
    match record.stage {
        UpgradeStage::Prepared => {
            if record.observed_at_unix_ms != handoff.plan.prepared_at_unix_ms
                || record.observed_state_digest != handoff.plan.predecessor.durable_state_digest
            {
                return Err(UpgradeTrackingError::InvalidObservedState);
            }
        }
        UpgradeStage::Activated => {
            if record.observed_at_unix_ms < handoff.plan.activates_at_unix_ms {
                return Err(UpgradeTrackingError::BeforeActivationWindow);
            }
            if record.observed_at_unix_ms >= handoff.plan.finalization_deadline_unix_ms {
                return Err(UpgradeTrackingError::AfterFinalizationDeadline);
            }
            if record.observed_state_digest != handoff.plan.successor.durable_state_digest {
                return Err(UpgradeTrackingError::InvalidObservedState);
            }
        }
        UpgradeStage::Finalized => {
            if record.observed_at_unix_ms >= handoff.plan.finalization_deadline_unix_ms {
                return Err(UpgradeTrackingError::AfterFinalizationDeadline);
            }
            if record.observed_state_digest != handoff.plan.successor.durable_state_digest {
                return Err(UpgradeTrackingError::InvalidObservedState);
            }
        }
        UpgradeStage::RolledBack => {
            if record.observed_at_unix_ms >= handoff.plan.finalization_deadline_unix_ms {
                return Err(UpgradeTrackingError::AfterFinalizationDeadline);
            }
            if record.observed_state_digest != handoff.plan.rollback_target_digest {
                return Err(UpgradeTrackingError::InvalidObservedState);
            }
        }
        UpgradeStage::Failed => {
            if record.observed_state_digest.0 == [0; 32] {
                return Err(UpgradeTrackingError::InvalidObservedState);
            }
        }
    }
    Ok(())
}

fn digest_upgrade_record_fields(
    record: &UpgradeRecord,
) -> Result<Sha256Digest, UpgradeTrackingError> {
    let bytes = serde_json::to_vec(&(
        &record.schema_version,
        record.sequence,
        record.plan_digest,
        record.stage,
        record.observed_at_unix_ms,
        record.observed_state_digest,
        record.evidence_digest,
        record.previous_record_digest,
    ))
    .map_err(|error| UpgradeTrackingError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.upgrade-record-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authority_epoch::AuthorityEpochVector;
    use crate::crypto_digest::sha256;
    use crate::upgrade_handoff::{
        UPGRADE_ENDPOINT_SCHEMA, UPGRADE_HANDOFF_SCHEMA, UpgradeEndpoint, UpgradeHandoffPlan,
    };

    fn handoff() -> AuthorizedUpgradeHandoff {
        let endpoint = |version: &str, generation: u64| UpgradeEndpoint {
            schema_version: UPGRADE_ENDPOINT_SCHEMA.into(),
            software_version: version.into(),
            source_tree_digest: sha256(format!("source-{version}").as_bytes()),
            executable_digest: sha256(format!("exe-{version}").as_bytes()),
            durable_state_digest: sha256(format!("state-{version}").as_bytes()),
            replay_contract_digest: sha256(format!("replay-{version}").as_bytes()),
            authority_epoch: AuthorityEpochVector::new(2, 2, generation, 1, 1, 5, 3, 2).unwrap(),
        };
        AuthorizedUpgradeHandoff {
            plan: UpgradeHandoffPlan {
                schema_version: UPGRADE_HANDOFF_SCHEMA.into(),
                predecessor: endpoint("0.17.0", 3),
                successor: endpoint("0.18.0", 4),
                prepared_at_unix_ms: 100,
                activates_at_unix_ms: 200,
                finalization_deadline_unix_ms: 400,
                rollback_target_digest: sha256(b"rollback-state"),
                policy_migration_digests: vec![sha256(b"migration")],
                clock_evidence_digest: sha256(b"clock"),
                evidence_checkpoint_digest: sha256(b"evidence"),
                recovery_key_set_digest: sha256(b"recovery"),
                reason: "test".into(),
            },
            plan_digest: sha256(b"plan"),
            ceremony_digest: sha256(b"ceremony"),
            trust_snapshot_digest: sha256(b"trust"),
        }
    }

    #[test]
    fn activation_cannot_skip_preparation_or_use_wrong_state() {
        let handoff = handoff();
        let mut tracker = UpgradeHandoffTracker::new(&handoff);
        assert!(matches!(
            tracker.append(
                &handoff,
                UpgradeStage::Activated,
                250,
                handoff.plan.successor.durable_state_digest,
                sha256(b"e")
            ),
            Err(UpgradeTrackingError::InvalidTransition { .. })
        ));
        tracker
            .append(
                &handoff,
                UpgradeStage::Prepared,
                100,
                handoff.plan.predecessor.durable_state_digest,
                sha256(b"p"),
            )
            .unwrap();
        assert_eq!(
            tracker.append(
                &handoff,
                UpgradeStage::Activated,
                250,
                sha256(b"wrong"),
                sha256(b"a")
            ),
            Err(UpgradeTrackingError::InvalidObservedState)
        );
    }

    #[test]
    fn rollback_is_terminal_and_exact_target_bound() {
        let handoff = handoff();
        let mut tracker = UpgradeHandoffTracker::new(&handoff);
        tracker
            .append(
                &handoff,
                UpgradeStage::Prepared,
                100,
                handoff.plan.predecessor.durable_state_digest,
                sha256(b"p"),
            )
            .unwrap();
        tracker
            .append(
                &handoff,
                UpgradeStage::Activated,
                250,
                handoff.plan.successor.durable_state_digest,
                sha256(b"a"),
            )
            .unwrap();
        tracker
            .append(
                &handoff,
                UpgradeStage::RolledBack,
                300,
                handoff.plan.rollback_target_digest,
                sha256(b"r"),
            )
            .unwrap();
        assert_eq!(
            tracker.append(
                &handoff,
                UpgradeStage::Finalized,
                350,
                handoff.plan.successor.durable_state_digest,
                sha256(b"f")
            ),
            Err(UpgradeTrackingError::TerminalState)
        );
    }
}
