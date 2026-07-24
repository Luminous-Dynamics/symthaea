// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Persistent anti-rollback tracking for upgrade probation clearances.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::upgrade_probation::{
    AuthorizedUpgradeProbationClearance, UpgradeProbationError, digest_upgrade_probation_evidence,
};
use serde::{Deserialize, Serialize};

pub const UPGRADE_PROBATION_TRACKER_SCHEMA: &str =
    "symthaea.fabrication.upgrade-probation-tracker.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradeProbationTracker {
    pub schema_version: String,
    pub handoff_digest: Option<Sha256Digest>,
    pub latest_sequence: Option<u64>,
    pub latest_cleared_at_unix_ms: Option<u64>,
    pub latest_evidence_digest: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpgradeProbationTrackingError {
    UnsupportedSchema,
    InvalidTrackerState,
    InvalidEvidence(UpgradeProbationError),
    HandoffSubstitution,
    SequenceRollback { latest: u64, proposed: u64 },
    SequenceCollision { sequence: u64 },
    ClearanceTimeRegression { latest: u64, proposed: u64 },
    Encoding(String),
}

impl Default for UpgradeProbationTracker {
    fn default() -> Self {
        Self {
            schema_version: UPGRADE_PROBATION_TRACKER_SCHEMA.into(),
            handoff_digest: None,
            latest_sequence: None,
            latest_cleared_at_unix_ms: None,
            latest_evidence_digest: None,
        }
    }
}

impl UpgradeProbationTracker {
    pub fn validate(&self) -> Result<(), UpgradeProbationTrackingError> {
        if self.schema_version != UPGRADE_PROBATION_TRACKER_SCHEMA {
            return Err(UpgradeProbationTrackingError::UnsupportedSchema);
        }
        match (
            self.handoff_digest,
            self.latest_sequence,
            self.latest_cleared_at_unix_ms,
            self.latest_evidence_digest,
        ) {
            (None, None, None, None) => Ok(()),
            (Some(handoff), Some(sequence), Some(_), Some(digest))
                if handoff.0 != [0; 32] && sequence > 0 && digest.0 != [0; 32] =>
            {
                Ok(())
            }
            _ => Err(UpgradeProbationTrackingError::InvalidTrackerState),
        }
    }

    pub fn accept(
        &mut self,
        clearance: &AuthorizedUpgradeProbationClearance,
    ) -> Result<Sha256Digest, UpgradeProbationTrackingError> {
        self.validate()?;
        let evidence = clearance.evidence();
        let digest = digest_upgrade_probation_evidence(evidence)
            .map_err(UpgradeProbationTrackingError::InvalidEvidence)?;
        if digest != clearance.evidence_digest() {
            return Err(UpgradeProbationTrackingError::InvalidEvidence(
                UpgradeProbationError::CeremonyPayloadMismatch,
            ));
        }
        if self
            .handoff_digest
            .is_some_and(|handoff| handoff != evidence.handoff_digest)
        {
            return Err(UpgradeProbationTrackingError::HandoffSubstitution);
        }
        if let Some(latest) = self.latest_sequence {
            if evidence.probation_sequence < latest {
                return Err(UpgradeProbationTrackingError::SequenceRollback {
                    latest,
                    proposed: evidence.probation_sequence,
                });
            }
            if evidence.probation_sequence == latest {
                if self.latest_evidence_digest == Some(digest) {
                    return Ok(digest);
                }
                return Err(UpgradeProbationTrackingError::SequenceCollision {
                    sequence: evidence.probation_sequence,
                });
            }
        }
        if let Some(latest) = self.latest_cleared_at_unix_ms {
            if evidence.cleared_at_unix_ms < latest {
                return Err(UpgradeProbationTrackingError::ClearanceTimeRegression {
                    latest,
                    proposed: evidence.cleared_at_unix_ms,
                });
            }
        }
        self.handoff_digest = Some(evidence.handoff_digest);
        self.latest_sequence = Some(evidence.probation_sequence);
        self.latest_cleared_at_unix_ms = Some(evidence.cleared_at_unix_ms);
        self.latest_evidence_digest = Some(digest);
        Ok(digest)
    }
}

pub fn digest_upgrade_probation_tracker(
    tracker: &UpgradeProbationTracker,
) -> Result<Sha256Digest, UpgradeProbationTrackingError> {
    tracker.validate()?;
    let bytes = serde_json::to_vec(tracker)
        .map_err(|error| UpgradeProbationTrackingError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.upgrade-probation-tracker-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;

    #[test]
    fn malformed_partial_state_is_rejected() {
        let tracker = UpgradeProbationTracker {
            schema_version: UPGRADE_PROBATION_TRACKER_SCHEMA.into(),
            handoff_digest: Some(sha256(b"handoff")),
            latest_sequence: None,
            latest_cleared_at_unix_ms: None,
            latest_evidence_digest: None,
        };
        assert_eq!(
            tracker.validate(),
            Err(UpgradeProbationTrackingError::InvalidTrackerState)
        );
    }

    #[test]
    fn empty_tracker_has_stable_digest() {
        assert_eq!(
            digest_upgrade_probation_tracker(&UpgradeProbationTracker::default()).unwrap(),
            digest_upgrade_probation_tracker(&UpgradeProbationTracker::default()).unwrap()
        );
    }
}
