// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Durable rollout-revocation history and authority checks.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::rollout::RolloutPhase;
use crate::rollout_revocation::{AuthorizedRolloutRevocation, RolloutRevocationScope};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const ROLLOUT_REVOCATION_TRACKER_SCHEMA: &str =
    "symthaea.fabrication.rollout-revocation-tracker.v1";
pub const MAX_ROLLOUT_REVOCATIONS: usize = 16_384;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RolloutRevocationRecord {
    pub revocation_sequence: u64,
    pub promotion_digest: Sha256Digest,
    pub rollout_plan_digest: Sha256Digest,
    pub scope: RolloutRevocationScope,
    pub effective_at_unix_s: u64,
    pub revocation_digest: Sha256Digest,
    pub ceremony_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RolloutRevocationTracker {
    pub schema_version: String,
    records: Vec<RolloutRevocationRecord>,
}

impl Default for RolloutRevocationTracker {
    fn default() -> Self {
        Self {
            schema_version: ROLLOUT_REVOCATION_TRACKER_SCHEMA.into(),
            records: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RolloutRevocationTrackingError {
    UnsupportedSchema,
    CapacityExceeded,
    InvalidRecord,
    SequenceRollback,
    SameSequenceSubstitution,
    MissingPriorRecord,
    PriorRecordChanged,
    Encoding(String),
}

impl RolloutRevocationTracker {
    pub fn records(&self) -> &[RolloutRevocationRecord] {
        &self.records
    }

    pub fn validate(&self) -> Result<(), RolloutRevocationTrackingError> {
        if self.schema_version != ROLLOUT_REVOCATION_TRACKER_SCHEMA {
            return Err(RolloutRevocationTrackingError::UnsupportedSchema);
        }
        if self.records.len() > MAX_ROLLOUT_REVOCATIONS {
            return Err(RolloutRevocationTrackingError::CapacityExceeded);
        }
        let mut previous_sequence = 0;
        let mut seen_sequences = BTreeSet::new();
        for record in &self.records {
            let scope_valid = match &record.scope {
                RolloutRevocationScope::EntirePromotion
                | RolloutRevocationScope::PhaseAndAbove(_) => true,
                RolloutRevocationScope::Machines(machines) => {
                    !machines.is_empty()
                        && machines.iter().all(|machine_id| {
                            !machine_id.trim().is_empty()
                                && machine_id == machine_id.trim()
                                && machine_id.len() <= 256
                                && !machine_id.chars().any(char::is_control)
                        })
                }
            };
            if record.revocation_sequence == 0
                || record.revocation_sequence <= previous_sequence
                || !seen_sequences.insert(record.revocation_sequence)
                || record.effective_at_unix_s == 0
                || !scope_valid
            {
                return Err(RolloutRevocationTrackingError::InvalidRecord);
            }
            previous_sequence = record.revocation_sequence;
        }
        Ok(())
    }

    pub fn apply(
        &mut self,
        revocation: &AuthorizedRolloutRevocation,
    ) -> Result<Sha256Digest, RolloutRevocationTrackingError> {
        self.validate()?;
        if self.records.len() >= MAX_ROLLOUT_REVOCATIONS {
            return Err(RolloutRevocationTrackingError::CapacityExceeded);
        }
        let evidence = revocation.evidence();
        if let Some(latest) = self.records.last() {
            if evidence.revocation_sequence < latest.revocation_sequence {
                return Err(RolloutRevocationTrackingError::SequenceRollback);
            }
            if evidence.revocation_sequence == latest.revocation_sequence {
                if latest.revocation_digest == revocation.revocation_digest() {
                    return Ok(latest.revocation_digest);
                }
                return Err(RolloutRevocationTrackingError::SameSequenceSubstitution);
            }
        }
        self.records.push(RolloutRevocationRecord {
            revocation_sequence: evidence.revocation_sequence,
            promotion_digest: evidence.promotion_digest,
            rollout_plan_digest: evidence.rollout_plan_digest,
            scope: evidence.scope.clone(),
            effective_at_unix_s: evidence.effective_at_unix_s,
            revocation_digest: revocation.revocation_digest(),
            ceremony_digest: revocation.ceremony_digest(),
        });
        Ok(revocation.revocation_digest())
    }

    pub fn permits(
        &self,
        promotion_digest: Sha256Digest,
        phase: RolloutPhase,
        machine_id: &str,
        unix_s: u64,
    ) -> bool {
        !self.records.iter().any(|record| {
            record.promotion_digest == promotion_digest
                && unix_s >= record.effective_at_unix_s
                && match &record.scope {
                    RolloutRevocationScope::EntirePromotion => true,
                    RolloutRevocationScope::PhaseAndAbove(minimum) => phase >= *minimum,
                    RolloutRevocationScope::Machines(machines) => machines.contains(machine_id),
                }
        })
    }

    pub fn verify_successor_of(
        &self,
        previous: &Self,
    ) -> Result<(), RolloutRevocationTrackingError> {
        self.validate()?;
        previous.validate()?;
        if self.records.len() < previous.records.len() {
            return Err(RolloutRevocationTrackingError::MissingPriorRecord);
        }
        if self.records[..previous.records.len()] != previous.records[..] {
            return Err(RolloutRevocationTrackingError::PriorRecordChanged);
        }
        Ok(())
    }
}

pub fn digest_rollout_revocation_tracker(
    tracker: &RolloutRevocationTracker,
) -> Result<Sha256Digest, RolloutRevocationTrackingError> {
    tracker.validate()?;
    let bytes = serde_json::to_vec(tracker)
        .map_err(|error| RolloutRevocationTrackingError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.rollout-revocation-tracker-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persisted_history_is_prefix_preserving() {
        let previous = RolloutRevocationTracker {
            schema_version: ROLLOUT_REVOCATION_TRACKER_SCHEMA.into(),
            records: vec![RolloutRevocationRecord {
                revocation_sequence: 1,
                promotion_digest: Sha256Digest([1; 32]),
                rollout_plan_digest: Sha256Digest([2; 32]),
                scope: RolloutRevocationScope::EntirePromotion,
                effective_at_unix_s: 10,
                revocation_digest: Sha256Digest([3; 32]),
                ceremony_digest: Sha256Digest([4; 32]),
            }],
        };
        let current = RolloutRevocationTracker::default();
        assert_eq!(
            current.verify_successor_of(&previous),
            Err(RolloutRevocationTrackingError::MissingPriorRecord)
        );
    }
}
