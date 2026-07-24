// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Role-bound sensorimotor controller input.
//!
//! Thought and body state are kept in separately bound subspaces before they
//! are bundled. This prevents a live controller from being trained on one
//! feature distribution and served on an unrelated one.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ControllerInputSchema {
    /// Historical controller input: one untyped hypervector.
    LegacyDirectV0,
    /// `role(THOUGHT) ⊗ thought + role(BODY) ⊗ proprioception`.
    SensorimotorBoundV1,
}

impl Default for ControllerInputSchema {
    fn default() -> Self {
        Self::LegacyDirectV0
    }
}

pub struct SensorimotorInputBinder {
    thought_role: ContinuousHV,
    body_role: ContinuousHV,
}

impl SensorimotorInputBinder {
    pub fn new(genesis: &GenesisSeed) -> Self {
        Self {
            thought_role: ContinuousHV::from_genesis(
                genesis,
                "manipulator::role::thought",
                HDC_DIMENSION,
            ),
            body_role: ContinuousHV::from_genesis(
                genesis,
                "manipulator::role::body",
                HDC_DIMENSION,
            ),
        }
    }

    pub fn fuse(&self, thought: &ContinuousHV, body: &ContinuousHV) -> ContinuousHV {
        let thought_bound = thought.bind(&self.thought_role);
        let body_bound = body.bind(&self.body_role);
        ContinuousHV::bundle(&[&thought_bound, &body_bound]).normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fusion_is_deterministic() {
        let genesis = GenesisSeed::from_phrase("sensorimotor-test");
        let binder_a = SensorimotorInputBinder::new(&genesis);
        let binder_b = SensorimotorInputBinder::new(&genesis);
        let thought = ContinuousHV::random(HDC_DIMENSION, 10);
        let body = ContinuousHV::random(HDC_DIMENSION, 20);
        assert!(
            binder_a
                .fuse(&thought, &body)
                .similarity(&binder_b.fuse(&thought, &body))
                > 0.999
        );
    }

    #[test]
    fn either_channel_changes_the_fused_input() {
        let genesis = GenesisSeed::from_phrase("sensorimotor-change-test");
        let binder = SensorimotorInputBinder::new(&genesis);
        let thought_a = ContinuousHV::random(HDC_DIMENSION, 1);
        let thought_b = ContinuousHV::random(HDC_DIMENSION, 2);
        let body_a = ContinuousHV::random(HDC_DIMENSION, 3);
        let body_b = ContinuousHV::random(HDC_DIMENSION, 4);

        let baseline = binder.fuse(&thought_a, &body_a);
        assert!(baseline.similarity(&binder.fuse(&thought_b, &body_a)) < 0.999);
        assert!(baseline.similarity(&binder.fuse(&thought_a, &body_b)) < 0.999);
    }
}
