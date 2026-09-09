// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! SHA-256 identity for the exact spatial preparation used by one Reach execution.
//!
//! Lower Reach evidence retains a compact `spatial_goal_fingerprint` for fast
//! identity checks. Promotion-grade evidence additionally needs the actual target
//! geometry so a 64-bit collision cannot substitute a different target. This
//! commitment binds the exact subject, semantic goal, selected hand, world/root
//! targets, workspace utilization and preparation instant recorded by the runtime.

use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::morphology::HandSide;
use crate::qualification::HumanoidQualificationSubject;
use crate::reach_execution::HumanoidPermittedReachPreparationReport;
use crate::types::{ActuationMode, HumanoidTask};

pub const HUMANOID_REACH_SPATIAL_GOAL_COMMITMENT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HumanoidReachSpatialGoalCommitment {
    subject_digest: HumanoidEvidenceDigest,
    goal_digest: HumanoidEvidenceDigest,
}

impl HumanoidReachSpatialGoalCommitment {
    pub fn from_preparation(
        subject: &HumanoidQualificationSubject,
        preparation: &HumanoidPermittedReachPreparationReport,
    ) -> Option<Self> {
        if !validate_preparation(subject, preparation) {
            return None;
        }
        let subject_digest = digest_subject(subject)?;
        let goal_digest = digest_goal(subject_digest, preparation)?;
        Some(Self {
            subject_digest,
            goal_digest,
        })
    }

    pub const fn subject_digest(&self) -> HumanoidEvidenceDigest {
        self.subject_digest
    }

    pub const fn goal_digest(&self) -> HumanoidEvidenceDigest {
        self.goal_digest
    }

    pub fn validate_for(
        &self,
        subject: &HumanoidQualificationSubject,
        preparation: &HumanoidPermittedReachPreparationReport,
    ) -> bool {
        if !validate_preparation(subject, preparation) {
            return false;
        }
        let Some(subject_digest) = digest_subject(subject) else {
            return false;
        };
        let Some(goal_digest) = digest_goal(subject_digest, preparation) else {
            return false;
        };
        self.subject_digest == subject_digest
            && self.goal_digest == goal_digest
            && !self.goal_digest.is_zero()
    }
}

fn validate_preparation(
    subject: &HumanoidQualificationSubject,
    preparation: &HumanoidPermittedReachPreparationReport,
) -> bool {
    subject.validate()
        && subject.task == HumanoidTask::Reach
        && preparation.validation_epoch != 0
        && valid_id(&preparation.goal_id)
        && preparation.spatial_goal_fingerprint != 0
        && preparation.target_world_m.iter().all(|value| value.is_finite())
        && preparation.target_root_m.iter().all(|value| value.is_finite())
        && preparation.workspace_utilization_sq.is_finite()
        && preparation.workspace_utilization_sq >= 0.0
        && preparation.workspace_utilization_sq <= 1.0
        && preparation.prepared_at_s.is_finite()
        && preparation.prepared_at_s >= 0.0
}

fn digest_subject(subject: &HumanoidQualificationSubject) -> Option<HumanoidEvidenceDigest> {
    if !subject.validate() || subject.task != HumanoidTask::Reach {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("reach.spatial-subject.v1");
    h.u32(HUMANOID_REACH_SPATIAL_GOAL_COMMITMENT_SCHEMA_VERSION)
        .u32(subject.schema_version)
        .string(subject.morphology.schema_id())
        .u64(task_id(subject.task))
        .u64(actuation_mode_id(subject.actuation_mode))
        .string(&subject.backend_profile_id);
    Some(h.finish())
}

fn digest_goal(
    subject_digest: HumanoidEvidenceDigest,
    preparation: &HumanoidPermittedReachPreparationReport,
) -> Option<HumanoidEvidenceDigest> {
    if subject_digest.is_zero()
        || preparation.validation_epoch == 0
        || !valid_id(&preparation.goal_id)
        || preparation.spatial_goal_fingerprint == 0
        || !preparation.target_world_m.iter().all(|value| value.is_finite())
        || !preparation.target_root_m.iter().all(|value| value.is_finite())
        || !preparation.workspace_utilization_sq.is_finite()
        || !preparation.prepared_at_s.is_finite()
    {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("reach.spatial-preparation.v1");
    h.u32(HUMANOID_REACH_SPATIAL_GOAL_COMMITMENT_SCHEMA_VERSION)
        .digest(subject_digest)
        .u64(preparation.validation_epoch)
        .string(&preparation.goal_id)
        .u64(preparation.spatial_goal_fingerprint)
        .u64(hand_id(preparation.hand));
    for value in preparation.target_world_m {
        h.f64(value);
    }
    for value in preparation.target_root_m {
        h.f64(value);
    }
    h.f64(preparation.workspace_utilization_sq)
        .f64(preparation.prepared_at_s);
    Some(h.finish())
}

fn hand_id(hand: HandSide) -> u64 {
    match hand {
        HandSide::Right => 1,
        HandSide::Left => 2,
    }
}

fn task_id(task: HumanoidTask) -> u64 {
    match task {
        HumanoidTask::Stand => 1,
        HumanoidTask::Walk => 2,
        HumanoidTask::Run => 3,
        HumanoidTask::Reach => 4,
        HumanoidTask::Grasp => 5,
    }
}

fn actuation_mode_id(mode: ActuationMode) -> u64 {
    match mode {
        ActuationMode::NormalizedTorque => 1,
        ActuationMode::TorqueNewtonMetres => 2,
        ActuationMode::NormalizedPosition => 3,
        ActuationMode::PositionTargetRadians => 4,
    }
}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !byte.is_ascii_whitespace())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphology::HumanoidMorphology;
    use crate::reach_execution::{HumanoidFrozenDynamicsLineage, HumanoidPermittedReachPreparationReport};
    use crate::cartesian_hand_reference::HumanoidCartesianHandReferenceReport;

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Reach,
            ActuationMode::NormalizedTorque,
            "spatial-commitment-test-v1",
        )
    }

    fn preparation() -> HumanoidPermittedReachPreparationReport {
        HumanoidPermittedReachPreparationReport {
            validation_epoch: 7,
            prepared_at_s: 1.5,
            goal_id: "cup-7".into(),
            spatial_goal_fingerprint: 11,
            hand: HandSide::Right,
            target_world_m: [0.4, -0.2, 1.0],
            target_root_m: [0.3, -0.2, 0.2],
            workspace_utilization_sq: 0.45,
            dynamics: HumanoidFrozenDynamicsLineage {
                rigid_model_id: None,
                rigid_sampled_at_s: None,
                full_model_id: "full-model".into(),
                full_sampled_at_s: 1.4,
                floating_model_id: None,
                floating_sampled_at_s: None,
            },
            cartesian_reference: HumanoidCartesianHandReferenceReport {
                position_error_norm_m: 0.25,
                desired_cartesian_speed_mps: 0.2,
                jacobian_confidence: 1.0,
                maximum_normalized_correction_used: 0.1,
                dynamics_age_s: 0.1,
            },
        }
    }

    #[test]
    fn world_target_change_changes_commitment() {
        let a = HumanoidReachSpatialGoalCommitment::from_preparation(&subject(), &preparation()).unwrap();
        let mut changed = preparation();
        changed.target_world_m[0] += 0.01;
        let b = HumanoidReachSpatialGoalCommitment::from_preparation(&subject(), &changed).unwrap();
        assert_ne!(a.goal_digest(), b.goal_digest());
    }
}
