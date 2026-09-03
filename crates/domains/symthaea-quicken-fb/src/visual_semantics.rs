// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Descriptive semantic signatures for Spore's presentation policy.
//!
//! This combines the pure composition and temporal policies into a small
//! signature so tests and evidence can detect accidental semantic collapse (for
//! example repair and mesh return becoming the same treatment). It performs no
//! rendering and has no boot authority.
//!
//! Numeric distance is deliberately descriptive. Constitutional regression
//! gates use categorical identity and domain relationships rather than arbitrary
//! aesthetic-distance thresholds.

use symthaea_boot_ecology::BootStageKind;

use crate::temporal_choreography::{MotionIntent, TemporalChoreography};
use crate::visual_composition::{VisualCompositionBudget, VisualHero, VisualProfile};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisualSemanticSignature {
    pub hero: VisualHero,
    pub motion: MotionIntent,
    pub topology: f32,
    pub accent: f32,
    pub mesh: f32,
    pub holography: f32,
    pub membrane: f32,
    pub caustics: f32,
    pub bloom: f32,
    pub identity: f32,
    pub ambient_motion: f32,
    pub damping: f32,
}

impl VisualSemanticSignature {
    pub fn derive(
        stage: BootStageKind,
        progress: f32,
        intensity: f32,
        profile: VisualProfile,
    ) -> Self {
        let budget = VisualCompositionBudget::derive(stage, progress, intensity, profile);
        let temporal = TemporalChoreography::derive(stage, progress, profile);
        Self {
            hero: budget.hero,
            motion: temporal.intent,
            topology: budget.topology,
            accent: budget.accent,
            mesh: budget.mesh,
            holography: budget.holography,
            membrane: budget.membrane,
            caustics: budget.caustics,
            bloom: budget.bloom,
            identity: budget.identity,
            ambient_motion: temporal.ambient_gain,
            damping: temporal.damping,
        }
    }

    /// Descriptive L1 distance across scalar presentation dimensions.
    ///
    /// This is useful in evidence reports and debugging, but has no universal
    /// perceptual meaning and must not become an aesthetic pass/fail threshold.
    pub fn scalar_distance(self, other: Self) -> f32 {
        [
            (self.topology - other.topology).abs(),
            (self.accent - other.accent).abs(),
            (self.mesh - other.mesh).abs(),
            (self.holography - other.holography).abs(),
            (self.membrane - other.membrane).abs(),
            (self.caustics - other.caustics).abs(),
            (self.bloom - other.bloom).abs(),
            (self.identity - other.identity).abs(),
            (self.ambient_motion - other.ambient_motion).abs(),
            (self.damping - other.damping).abs(),
        ]
        .into_iter()
        .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EVENT_STAGES: [BootStageKind; 5] = [
        BootStageKind::Repair,
        BootStageKind::GrowthRing,
        BootStageKind::HardwareBud,
        BootStageKind::RetractFailedGrowth,
        BootStageKind::MeshLink,
    ];

    fn signature(stage: BootStageKind) -> VisualSemanticSignature {
        VisualSemanticSignature::derive(stage, 0.5, 1.0, VisualProfile::Standard)
    }

    #[test]
    fn major_event_families_keep_distinct_categorical_identity() {
        for (index, stage) in EVENT_STAGES.iter().enumerate() {
            let a = signature(*stage);
            for other in EVENT_STAGES.iter().skip(index + 1) {
                let b = signature(*other);
                assert_ne!(
                    a.hero, b.hero,
                    "semantic heroes collapsed: {stage:?} vs {other:?}"
                );
                assert_ne!(
                    a.motion, b.motion,
                    "semantic motion intents collapsed: {stage:?} vs {other:?}"
                );
            }
        }
    }

    #[test]
    fn scalar_distance_is_descriptive_math_not_a_quality_gate() {
        for stage in EVENT_STAGES {
            let a = signature(stage);
            assert_eq!(a.scalar_distance(a), 0.0);
        }

        for (index, stage) in EVENT_STAGES.iter().enumerate() {
            let a = signature(*stage);
            for other in EVENT_STAGES.iter().skip(index + 1) {
                let b = signature(*other);
                let ab = a.scalar_distance(b);
                let ba = b.scalar_distance(a);
                assert!(ab.is_finite());
                assert!(ab > 0.0, "numeric signatures unexpectedly identical");
                assert!((ab - ba).abs() <= f32::EPSILON);
            }
        }
    }

    #[test]
    fn semantics_do_not_depend_on_richness_profile() {
        for stage in EVENT_STAGES {
            let calm = VisualSemanticSignature::derive(stage, 0.5, 1.0, VisualProfile::Calm);
            let rich = VisualSemanticSignature::derive(stage, 0.5, 1.0, VisualProfile::Rich);
            assert_eq!(calm.hero, rich.hero);
            assert_eq!(calm.motion, rich.motion);
        }
    }

    #[test]
    fn recovery_connection_and_persistent_change_have_domain_polarity() {
        let repair = signature(BootStageKind::Repair);
        let rollback = signature(BootStageKind::RetractFailedGrowth);
        let generation = signature(BootStageKind::GrowthRing);
        let hardware = signature(BootStageKind::HardwareBud);
        let mesh = signature(BootStageKind::MeshLink);

        // Recovery and persistent local change emphasize semantic accent rather
        // than connectivity spectacle.
        assert!(repair.accent > repair.mesh);
        assert!(rollback.accent > rollback.mesh);
        assert!(generation.accent > generation.mesh);
        assert!(hardware.accent > hardware.mesh);

        // Mesh return is the inverse: connection itself is the event.
        assert!(mesh.mesh > mesh.accent);

        // Rollback is deliberately quieter/more damped than active repair;
        // reconnection is allowed more ambient motion than either recovery path.
        assert!(rollback.damping > repair.damping);
        assert!(mesh.ambient_motion > rollback.ambient_motion);
        assert!(mesh.ambient_motion > repair.ambient_motion);
    }

    #[test]
    fn blackout_and_final_handoff_remain_effectively_empty() {
        let blackout = VisualSemanticSignature::derive(
            BootStageKind::Blackout,
            1.0,
            1.0,
            VisualProfile::Rich,
        );
        let handoff = VisualSemanticSignature::derive(
            BootStageKind::Handoff,
            1.0,
            1.0,
            VisualProfile::Rich,
        );

        assert_eq!(blackout.ambient_motion, 0.0);
        assert_eq!(handoff.ambient_motion, 0.0);
        assert!(handoff.topology < 0.01);
        assert!(handoff.accent < 0.01);
        assert!(handoff.mesh < 0.01);
        assert!(handoff.holography < 0.01);
        assert!(handoff.bloom < 0.01);
    }
}
