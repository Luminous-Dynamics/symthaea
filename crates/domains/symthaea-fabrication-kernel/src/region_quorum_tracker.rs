// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Persistent anti-rollback tracking for cross-region quorum evidence.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::region_quorum::{
    RegionalQuorumError, RegionalQuorumEvidence, digest_regional_quorum_evidence,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionalQuorumTracker {
    latest_membership_epoch: Option<u64>,
    latest_gateway_generation: Option<u64>,
    latest_evidence_digest: Option<Sha256Digest>,
    latest_consensus_digest: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegionalQuorumTrackingError {
    InvalidEvidence(RegionalQuorumError),
    MembershipEpochRollback { latest: u64, proposed: u64 },
    GatewayGenerationRollback { latest: u64, proposed: u64 },
    SameGenerationSubstitution,
    InvalidTrackerState,
    Encoding(String),
}

impl RegionalQuorumTracker {
    pub fn accept(
        &mut self,
        evidence: &RegionalQuorumEvidence,
    ) -> Result<Sha256Digest, RegionalQuorumTrackingError> {
        let digest = digest_regional_quorum_evidence(evidence)
            .map_err(RegionalQuorumTrackingError::InvalidEvidence)?;
        if let Some(latest) = self.latest_membership_epoch {
            if evidence.membership_epoch < latest {
                return Err(RegionalQuorumTrackingError::MembershipEpochRollback {
                    latest,
                    proposed: evidence.membership_epoch,
                });
            }
        }
        if let Some(latest) = self.latest_gateway_generation {
            if evidence.gateway_generation < latest {
                return Err(RegionalQuorumTrackingError::GatewayGenerationRollback {
                    latest,
                    proposed: evidence.gateway_generation,
                });
            }
            if evidence.gateway_generation == latest {
                if self.latest_evidence_digest == Some(digest)
                    && self.latest_consensus_digest == Some(evidence.gateway_consensus_digest)
                {
                    return Ok(digest);
                }
                return Err(RegionalQuorumTrackingError::SameGenerationSubstitution);
            }
        }
        self.latest_membership_epoch = Some(evidence.membership_epoch);
        self.latest_gateway_generation = Some(evidence.gateway_generation);
        self.latest_evidence_digest = Some(digest);
        self.latest_consensus_digest = Some(evidence.gateway_consensus_digest);
        Ok(digest)
    }

    pub fn latest_gateway_generation(&self) -> Option<u64> {
        self.latest_gateway_generation
    }
    pub fn latest_evidence_digest(&self) -> Option<Sha256Digest> {
        self.latest_evidence_digest
    }

    pub fn validate(&self) -> Result<(), RegionalQuorumTrackingError> {
        let populated = [
            self.latest_membership_epoch.is_some(),
            self.latest_gateway_generation.is_some(),
            self.latest_evidence_digest.is_some(),
            self.latest_consensus_digest.is_some(),
        ];
        if populated.iter().any(|value| *value) && populated.iter().any(|value| !*value) {
            return Err(RegionalQuorumTrackingError::InvalidTrackerState);
        }
        if self.latest_membership_epoch == Some(0)
            || self.latest_gateway_generation == Some(0)
            || self.latest_evidence_digest == Some(Sha256Digest([0; 32]))
            || self.latest_consensus_digest == Some(Sha256Digest([0; 32]))
        {
            return Err(RegionalQuorumTrackingError::InvalidTrackerState);
        }
        Ok(())
    }

    pub fn verify_successor_of(&self, previous: &Self) -> Result<(), RegionalQuorumTrackingError> {
        self.validate()?;
        previous.validate()?;
        if let (Some(old), Some(new)) = (
            previous.latest_membership_epoch,
            self.latest_membership_epoch,
        ) {
            if new < old {
                return Err(RegionalQuorumTrackingError::MembershipEpochRollback {
                    latest: old,
                    proposed: new,
                });
            }
        } else if previous.latest_membership_epoch.is_some() {
            return Err(RegionalQuorumTrackingError::MembershipEpochRollback {
                latest: previous.latest_membership_epoch.unwrap_or(0),
                proposed: 0,
            });
        }
        if let (Some(old), Some(new)) = (
            previous.latest_gateway_generation,
            self.latest_gateway_generation,
        ) {
            if new < old {
                return Err(RegionalQuorumTrackingError::GatewayGenerationRollback {
                    latest: old,
                    proposed: new,
                });
            }
            if new == old
                && (self.latest_evidence_digest != previous.latest_evidence_digest
                    || self.latest_consensus_digest != previous.latest_consensus_digest)
            {
                return Err(RegionalQuorumTrackingError::SameGenerationSubstitution);
            }
        } else if previous.latest_gateway_generation.is_some() {
            return Err(RegionalQuorumTrackingError::GatewayGenerationRollback {
                latest: previous.latest_gateway_generation.unwrap_or(0),
                proposed: 0,
            });
        }
        Ok(())
    }
}

pub fn digest_regional_quorum_tracker(
    tracker: &RegionalQuorumTracker,
) -> Result<Sha256Digest, RegionalQuorumTrackingError> {
    tracker.validate()?;
    let bytes = serde_json::to_vec(tracker)
        .map_err(|error| RegionalQuorumTrackingError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.regional-quorum-tracker-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
