// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Self-consistent evidence bundle for human-rescue ethics assurance.
//!
//! Reviewer authentication and cryptographic signing remain external. The
//! deterministic digest provider is for reproducible tests only.

use crate::certification_bundle::BuildIdentity;
use crate::rescue_ethics_validation::{RescueEthicsValidationReport, RescueEthicsValidator};
use crate::team_operations::DistributedRecoveryCheckpoint;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const RESCUE_ETHICS_BUNDLE_SCHEMA_VERSION: u16 = 1;
pub const MAX_RESCUE_ETHICS_REVIEWERS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RescueEthicsReviewerRole {
    SafetyReviewer,
    HumanFactorsReviewer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RescueEthicsReviewerAttestation {
    pub reviewer_id: String,
    pub role: RescueEthicsReviewerRole,
    pub hardware_backed: bool,
    pub externally_authenticated: bool,
    pub evidence_reference: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RescueEthicsEvidenceBundle {
    pub schema_version: u16,
    pub system: String,
    pub build: BuildIdentity,
    pub checkpoint: DistributedRecoveryCheckpoint,
    pub validation: RescueEthicsValidationReport,
    pub reviewers: Vec<RescueEthicsReviewerAttestation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RescueEthicsBundleError {
    UnsupportedSchema(u16),
    EmptyIdentity,
    InvalidCheckpoint,
    ValidationMismatch,
    TooManyReviewers,
    MissingReviewerRole(RescueEthicsReviewerRole),
    DuplicateReviewer,
    InvalidReviewer,
    Serialization,
}

pub trait RescueEthicsBundleDigestProvider {
    fn digest(&self, bytes: &[u8]) -> [u8; 32];
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DeterministicRescueEthicsBundleDigest;

impl RescueEthicsBundleDigestProvider for DeterministicRescueEthicsBundleDigest {
    fn digest(&self, bytes: &[u8]) -> [u8; 32] {
        let mut lanes = [
            0x243f_6a88_85a3_08d3u64,
            0x1319_8a2e_0370_7344u64,
            0xa409_3822_299f_31d0u64,
            0x082e_fa98_ec4e_6c89u64,
        ];
        for (index, byte) in bytes.iter().enumerate() {
            let lane = index % lanes.len();
            lanes[lane] ^= u64::from(*byte).wrapping_add((index as u64).rotate_left(9));
            lanes[lane] = lanes[lane]
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .rotate_left(((index + lane * 13) % 63 + 1) as u32);
        }
        let mut output = [0u8; 32];
        for (index, lane) in lanes.into_iter().enumerate() {
            output[index * 8..(index + 1) * 8].copy_from_slice(&lane.to_le_bytes());
        }
        output
    }
}

impl RescueEthicsEvidenceBundle {
    pub fn validate(&self) -> Result<(), RescueEthicsBundleError> {
        if self.schema_version != RESCUE_ETHICS_BUNDLE_SCHEMA_VERSION {
            return Err(RescueEthicsBundleError::UnsupportedSchema(
                self.schema_version,
            ));
        }
        if self.system.trim().is_empty()
            || self.build.source_tree.trim().is_empty()
            || self.build.toolchain.trim().is_empty()
            || self.build.dependency_profile.trim().is_empty()
            || self.build.campaign_id.trim().is_empty()
        {
            return Err(RescueEthicsBundleError::EmptyIdentity);
        }
        if !self.checkpoint.validate() {
            return Err(RescueEthicsBundleError::InvalidCheckpoint);
        }
        if self.validation != RescueEthicsValidator.validate() || !self.validation.passes() {
            return Err(RescueEthicsBundleError::ValidationMismatch);
        }
        if self.reviewers.len() > MAX_RESCUE_ETHICS_REVIEWERS {
            return Err(RescueEthicsBundleError::TooManyReviewers);
        }
        let mut identities = BTreeSet::new();
        let mut roles = BTreeSet::new();
        for reviewer in &self.reviewers {
            if reviewer.reviewer_id.trim().is_empty()
                || reviewer.evidence_reference.trim().is_empty()
                || !reviewer.hardware_backed
                || !reviewer.externally_authenticated
            {
                return Err(RescueEthicsBundleError::InvalidReviewer);
            }
            if !identities.insert(reviewer.reviewer_id.clone()) {
                return Err(RescueEthicsBundleError::DuplicateReviewer);
            }
            roles.insert(reviewer.role);
        }
        for role in [
            RescueEthicsReviewerRole::SafetyReviewer,
            RescueEthicsReviewerRole::HumanFactorsReviewer,
        ] {
            if !roles.contains(&role) {
                return Err(RescueEthicsBundleError::MissingReviewerRole(role));
            }
        }
        Ok(())
    }

    pub fn digest(
        &self,
        provider: &impl RescueEthicsBundleDigestProvider,
    ) -> Result<[u8; 32], RescueEthicsBundleError> {
        self.validate()?;
        let json = serde_json::to_string_pretty(self)
            .map_err(|_| RescueEthicsBundleError::Serialization)?;
        Ok(provider.digest(json.as_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::team::AgentId;
    use crate::team_operations::TeamCoordinator;

    fn bundle() -> RescueEthicsEvidenceBundle {
        RescueEthicsEvidenceBundle {
            schema_version: RESCUE_ETHICS_BUNDLE_SCHEMA_VERSION,
            system: "symthaea-subterranean".into(),
            build: BuildIdentity {
                source_tree: "tree-xxiii".into(),
                toolchain: "offline-structural".into(),
                dependency_profile: "standalone".into(),
                campaign_id: "campaign-xxiii".into(),
            },
            checkpoint: TeamCoordinator::new(AgentId::new(1)).recovery_checkpoint(),
            validation: RescueEthicsValidator.validate(),
            reviewers: vec![
                RescueEthicsReviewerAttestation {
                    reviewer_id: "safety-reviewer".into(),
                    role: RescueEthicsReviewerRole::SafetyReviewer,
                    hardware_backed: true,
                    externally_authenticated: true,
                    evidence_reference: "safety-review-xxiii".into(),
                },
                RescueEthicsReviewerAttestation {
                    reviewer_id: "human-factors-reviewer".into(),
                    role: RescueEthicsReviewerRole::HumanFactorsReviewer,
                    hardware_backed: true,
                    externally_authenticated: true,
                    evidence_reference: "human-factors-review-xxiii".into(),
                },
            ],
        }
    }

    #[test]
    fn canonical_bundle_is_self_consistent() {
        let bundle = bundle();
        assert_eq!(bundle.validate(), Ok(()));
        let left = bundle.digest(&DeterministicRescueEthicsBundleDigest);
        let right = bundle.digest(&DeterministicRescueEthicsBundleDigest);
        assert_eq!(left, right);
        assert!(left.is_ok());
    }

    #[test]
    fn one_identity_cannot_fill_both_review_roles() {
        let mut bundle = bundle();
        bundle.reviewers[1].reviewer_id = bundle.reviewers[0].reviewer_id.clone();
        assert_eq!(
            bundle.validate(),
            Err(RescueEthicsBundleError::DuplicateReviewer)
        );
    }
}
