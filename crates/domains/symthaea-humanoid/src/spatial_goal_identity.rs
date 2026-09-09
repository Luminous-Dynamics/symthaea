// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic identity for one exact spatial-goal observation.
//!
//! This checksum is intended for local evidence binding and replay detection. It
//! is not a cryptographic signature and must not be treated as operator authority.

use crate::morphology::HandSide;
use crate::spatial_goal::{HumanoidSpatialGoalEvidence, HumanoidSpatialTargetKind};

pub const HUMANOID_SPATIAL_GOAL_IDENTITY_SCHEMA_VERSION: u32 = 1;

/// Stable FNV-1a checksum over the exact admitted spatial-goal observation.
///
/// Refreshing the same logical target with a new observation timestamp/confidence
/// intentionally produces a new identity. Outcome evidence should bind to the
/// exact observation that was used for motion preparation, not merely to a reused
/// string goal id.
pub fn humanoid_spatial_goal_fingerprint(goal: &HumanoidSpatialGoalEvidence) -> u64 {
    if !goal.validate() {
        return 0;
    }
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    feed_u64(&mut hash, HUMANOID_SPATIAL_GOAL_IDENTITY_SCHEMA_VERSION as u64);
    feed_u64(&mut hash, target_kind_id(goal.kind));
    feed_u64(&mut hash, hand_id(goal.hand));
    feed_bytes(&mut hash, goal.goal_id.as_bytes());
    for value in goal.target_world_m {
        feed_u64(&mut hash, value.to_bits());
    }
    feed_u64(&mut hash, goal.observed_at_s.to_bits());
    feed_u64(&mut hash, goal.received_at_s.to_bits());
    feed_u64(&mut hash, goal.confidence.to_bits());
    if hash == 0 { 1 } else { hash }
}

fn target_kind_id(kind: HumanoidSpatialTargetKind) -> u64 {
    match kind {
        HumanoidSpatialTargetKind::Object => 1,
        HumanoidSpatialTargetKind::HumanContact => 2,
        HumanoidSpatialTargetKind::GenericPoint => 3,
    }
}

fn hand_id(hand: HandSide) -> u64 {
    match hand {
        HandSide::Right => 1,
        HandSide::Left => 2,
    }
}

fn feed_u64(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *hash ^= byte as u64;
        *hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
}

fn feed_bytes(hash: &mut u64, bytes: &[u8]) {
    feed_u64(hash, bytes.len() as u64);
    for byte in bytes {
        *hash ^= *byte as u64;
        *hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn goal() -> HumanoidSpatialGoalEvidence {
        HumanoidSpatialGoalEvidence {
            goal_id: "cup-7".into(),
            kind: HumanoidSpatialTargetKind::Object,
            hand: HandSide::Right,
            target_world_m: [0.4, -0.2, 1.0],
            observed_at_s: 1.0,
            received_at_s: 1.01,
            confidence: 0.95,
        }
    }

    #[test]
    fn exact_observation_has_stable_nonzero_identity() {
        let goal = goal();
        let a = humanoid_spatial_goal_fingerprint(&goal);
        let b = humanoid_spatial_goal_fingerprint(&goal);
        assert_ne!(a, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn target_or_observation_change_changes_identity() {
        let base = goal();
        let mut moved = base.clone();
        moved.target_world_m[0] += 0.01;
        let mut refreshed = base.clone();
        refreshed.observed_at_s += 0.01;
        refreshed.received_at_s += 0.01;
        assert_ne!(
            humanoid_spatial_goal_fingerprint(&base),
            humanoid_spatial_goal_fingerprint(&moved)
        );
        assert_ne!(
            humanoid_spatial_goal_fingerprint(&base),
            humanoid_spatial_goal_fingerprint(&refreshed)
        );
    }
}
