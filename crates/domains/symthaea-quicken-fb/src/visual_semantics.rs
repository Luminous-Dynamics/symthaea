// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Descriptive semantic signatures for Spore's presentation policy.
//!
//! This is a regression guard, not an aesthetic score. It combines the pure
//! composition and temporal policies into a small numeric signature so tests can
//! detect accidental semantic collapse (for example repair and mesh return
//! becoming effectively the same treatment). It performs no rendering and has
//! no boot authority.

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
    /// Categorical hero/motion identity is intentionally kept separate; callers
    /// should not interpret this scalar as perceptual quality or beauty.
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

    fn signature(stage: BootStageKind) -> VisualSemanticSignature {
        VisualSemanticSignature::derive(stage, 0.5, 1.0, VisualProfile::Standard)
    }

    fn assert_distinct(a: BootStageKind, b: BootStageKind, minimum_scalar_distance: f32) {
        let a = signature(a);
        let b = signature(b);
        assert_ne!(a.hero, b.hero, "semantic heroes collapsed: {a:?} vs {b:?}");
        assert_ne!(
            a.motion, b.motion,
            "semantic motion intents collapsed: {a:?} vs {b:?}"
        );
        assert!(
            a.scalar_distance(b) >= minimum_scalar_distance,
            "scalar treatment too similar: distance={}\na={a:?}\nb={b:?}",
            a.scalar_distance(b),
        );
    }

    #[test]
    fn major_event_families_remain_visually_distinct() {
        assert_distinct(BootStageKind::Repair, BootStageKind::MeshLink, 0.70);
        assert_distinct(BootStageKind::Repair, BootStageKind::GrowthRing, 0.20);
        assert_distinct(
            BootStageKind::Repair,
            BootStageKind::RetractFailedGrowth,
            0.20,
        );
        assert_distinct(BootStageKind::GrowthRing, BootStageKind::HardwareBud, 0.20);
        assert_distinct(BootStageKind::HardwareBud, BootStageKind::MeshLink, 0.60);
    }

    #[test]
    fn semantics_do_not_depend_on_richness_profile() {
        for stage in [
            BootStageKind::Repair,
            BootStageKind::GrowthRing,
            BootStageKind::HardwareBud,
            BootStageKind::RetractFailedGrowth,
            BootStageKind::MeshLink,
        ] {
            let calm = VisualSemanticSignature::derive(stage, 0.5, 1.0, VisualProfile::Calm);
            let rich = VisualSemanticSignature::derive(stage, 0.5, 1.0, VisualProfile::Rich);
            assert_eq!(calm.hero, rich.hero);
            assert_eq!(calm.motion, rich.motion);
        }
    }

    #[test]
    fn recovery_and_connection_cannot_collapse_to_same_signature() {
        let repair = signature(BootStageKind::Repair);
        let rollback = signature(BootStageKind::RetractFailedGrowth);
        let mesh = signature(BootStageKind::MeshLink);

        assert!(repair.accent > repair.mesh);
        assert!(rollback.accent > rollback.mesh);
        assert!(mesh.mesh > mesh.accent);
        assert!(mesh.ambient_motion > rollback.ambient_motion);
        assert!(rollback.damping > repair.damping);
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
